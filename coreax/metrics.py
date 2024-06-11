# © Crown Copyright GCHQ
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Classes and associated functionality to compute metrics assessing similarity of inputs.

Large parts of this codebase consider the generic problem of taking a
:math:`n \times d` dataset and creating an alternative representation of it in some way.
Having attained an alternative representation, we can then assess the quality of this
representation using some appropriate metric. Such metrics are implemented within this
module, all of which implement :class:`Metric`.
"""

from abc import abstractmethod
from itertools import product
from typing import Generic, Optional, TypeVar, Union

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jax import Array, jacfwd, vmap

import coreax.data
import coreax.kernel
import coreax.util
from coreax.score_matching import ScoreMatching
from coreax.solvers.coresubset import _convert_stein_kernel

_Data = TypeVar("_Data", bound=coreax.data.Data)


class Metric(eqx.Module, Generic[_Data]):
    """Base class for calculating metrics."""

    @abstractmethod
    def compute(self, reference_data: _Data, comparison_data: _Data, **kwargs) -> Array:
        r"""
        Compute the metric/distance between the reference and comparison data.

        :param reference_data: An instance of the class :class:`coreax.data.Data`,
            containing an :math:`n \times d` array of data
        :param comparison_data: An instance of the class :class:`coreax.data.Data` to
            compare against ``reference_data``, containing an :math:`m \times d` array
            of data
        :return: Computed metric as a zero-dimensional array
        """


class MMD(Metric[_Data]):
    r"""
    Definition and calculation of the (weighted) maximum mean discrepancy metric.

    For a dataset :math:`\mathcal{D}_1` of ``n`` points in ``d`` dimensions, and
    another dataset :math:`\mathcal{D}_2` of ``m`` points in ``d`` dimensions, the
    (weighted) maximum mean discrepancy is given by:

    .. math::
        \text{MMD}^2(\mathcal{D}_1,\mathcal{D}_2) = \mathbb{E}(k(\mathcal{D}_1,
        \mathcal{D}_1)) + \mathbb{E}(k(\mathcal{D}_2,\mathcal{D}_2))
        - 2\mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_2))

    where :math:`k` is the selected kernel, and the expectation is with respect to the
    normalized data weights.

    Common uses of MMD include comparing a reduced representation of a dataset to the
    original dataset, comparing different original datasets to one another, or
    comparing reduced representations of different original datasets to one another.

    :param kernel: Kernel object with compute method defined mapping
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param precision_threshold: Threshold above which negative values of the squared MMD
        are rounded to zero (accommodates precision loss)
    """

    kernel: coreax.kernel.Kernel
    precision_threshold: float = 1e-12

    def compute(
        self,
        reference_data: _Data,
        comparison_data: _Data,
        *,
        block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]] = None,
        unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
        **kwargs,
    ) -> Array:
        r"""
        Compute the (weighted) maximum mean discrepancy.

        .. math::
            \text{MMD}^2(\mathcal{D}_1,\mathcal{D}_2) = \mathbb{E}(k(\mathcal{D}_1,
            \mathcal{D}_1)) + \mathbb{E}(k(\mathcal{D}_2,\mathcal{D}_2))
            - 2\mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_2))

        :param reference_data: An instance of the class :class:`coreax.data.Data`,
            containing an :math:`n \times d` array of data
        :param comparison_data: An instance of the class :class:`coreax.data.Data` to
            compare against ``reference_data`` containing an :math:`m \times d` array of
            data
        :param block_size: Size of matrix blocks to process; a value of :data:`None`
            sets :math:`B_x = n` and :math:`B_y = m`, effectively disabling the block
            accumulation; an integer value ``B`` sets :math:`B_y = B_x = B`; a tuple
            allows different sizes to be specified for ``B_x`` and ``B_y``; to reduce
            overheads, it is often sensible to select the largest block size that does
            not exhaust the available memory resources
        :param unroll: Unrolling parameter for the outer and inner :func:`jax.lax.scan`
            calls, allows for trade-offs between compilation and runtime cost; consult
            the JAX docs for further information
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        del kwargs
        _block_size = coreax.util.tree_leaves_repeat(block_size, 2)
        bs_xx, bs_xy, _, bs_yy = tuple(product(_block_size, repeat=len(_block_size)))
        # Variable rename allows for nicer automatic formatting
        x, y = reference_data, comparison_data
        kernel_xx_mean = self.kernel.compute_mean(x, x, block_size=bs_xx, unroll=unroll)
        kernel_yy_mean = self.kernel.compute_mean(y, y, block_size=bs_yy, unroll=unroll)
        kernel_xy_mean = self.kernel.compute_mean(x, y, block_size=bs_xy, unroll=unroll)
        squared_mmd_threshold_applied = coreax.util.apply_negative_precision_threshold(
            kernel_xx_mean + kernel_yy_mean - 2 * kernel_xy_mean,
            self.precision_threshold,
        )
        return jnp.sqrt(squared_mmd_threshold_applied)


class KSD(Metric[_Data]):
    r"""
    Computation of the (regularised) (Laplace-corrected) kernel Stein discrepancy (KSD).

    For a set of ``n`` i.i.d. samples in ``d`` dimensions
    :math:`\mathcal{D}_1 \sim \mathbb{P}` and another set of ``m`` i.i.d. samples in
    ``d`` dimensions :math:`\mathcal{D}_2 \sim \mathbb{Q}`, the kernel Stein
    discrepancy is given by:

    .. math::

        KSD_{\lambda}^2(\mathbb{P}, \mathbb{Q})
        =  \frac{1}{m^2}\sum_{i \neq j}^m k_{\mathbb{P}}(x_i, x_j)
        + \frac{1}{m^2}\sum_{i = 1}^m [k_{\mathbb{P}}(x_i, x_i)
        + \Delta^+ \log(\mathbb{P}(x_i))]
        - \lambda frac{1}{m}\sum_{i = 1}^m \log(\mathbb{P}(x_i))

    where :math:`x \sim \mathbb{Q}`, :math:`k_{\mathbb{P}}` is the Stein kernel
    induced by the supplied base kernel and estimated with samples from
    :math:`\mathbb{P}`. The first term is vanilla KSD, the second term implements a
    Laplace correction, and the third term enforces entropic regularisation.

    Common uses of KSD include comparing a reduced representation of a dataset to the
    original dataset, comparing different original datasets to one another, or
    comparing reduced representations of different original datasets to one another.

    :param kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`;
        if 'kernel' is a :class:`~coreax.kernel.SteinKernel` and :code:`score_matching
        is not None`, a new instance of the kernel will be generated where the score
        function is given by :code:`score_matching.match(...)`
    :param score_matching: Specifies/overwrite the score function of the implied/passed
       :class:`~coreax.kernel.SteinKernel`; if :data:`None`, default to
       :class:`~coreax.score_matching.KernelDensityMatching` unless 'kernel' is a
       :class:`~coreax.kernel.SteinKernel`, in which case the kernel's existing score
       function is used.
    :param precision_threshold: Threshold above which negative values of the squared MMD
        are rounded to zero (accommodates precision loss)
    """

    kernel: coreax.kernel.Kernel
    score_matching: Optional[ScoreMatching] = None
    precision_threshold: float = 1e-12

    def compute(
        self,
        reference_data: _Data,
        comparison_data: _Data,
        *,
        laplace_correct: bool = True,
        regularisation: Optional[float] = None,
        block_size: Optional[int] = None,
        unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
        **kwargs,
    ) -> Array:
        r"""
        Compute the (regularised) (Laplace-corrected) kernel Stein discrepancy.

        .. math::

            KSD_{\lambda}^2(\mathbb{P}, \mathbb{Q})
            =  \frac{1}{m^2}\sum_{i \neq j}^m k_{\mathbb{P}}(x_i, x_j)
            + \frac{1}{m^2}\sum_{i = 1}^m [k_{\mathbb{P}}(x_i, x_i)
            + \Delta^+ \log(\mathbb{P}(x_i))]
            - \lambda frac{1}{m}\sum_{i = 1}^m \log(\mathbb{P}(x_i))

        :param reference_data: An instance of the class :class:`coreax.data.Data`,
            containing an :math:`n \times d` array of data sampled from
            :math:`\mathbb{P}`
        :param comparison_data: An instance of the class :class:`coreax.data.Data` to
            compare against ``reference_data`` containing an :math:`m \times d` array of
            data sampled from :math:`\mathbb{Q}`
        :param laplace_correct: Boolean that enforces Laplace correction, see Section
            3.1 of :cite:`benard2023kernel`.
        :param regularisation: Regularisation parameter, denoted :math:`lambda`
            above, which controls the strength of the entropic regularisation. If
            :data:`None`, defaults to value suggested in :cite:`benard2023kernel`.
            :math:`\lambda = frac{1}{m}. Negative values are converted to positive
            values.
        :param block_size: Size of matrix blocks to process; a value of :data:`None`
            sets `block_size`:math:`=n` effectively disabling the block accumulation; an
            integer value ``B`` sets `block_size`:math:`=B``, it is often sensible to
            select the largest block size that does not exhaust the available memory
            resources
        :param unroll: Unrolling parameter for the outer and inner :func:`jax.lax.scan`
            calls, allows for trade-offs between compilation and runtime cost; consult
            the JAX docs for further information
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        del kwargs
        # Train Stein kernel with data from P (no way to use weights here)
        x, w_x = jtu.tree_leaves(reference_data)
        kernel = _convert_stein_kernel(x, self.kernel, self.score_matching)

        # Variable rename allows for nicer automatic formatting, make sure we pass 'y'
        # as an instance of Data class to apply weights to mean computation.
        bs = block_size
        y = comparison_data
        # Compute kernel-mean term with data from Q.
        squared_ksd = kernel.compute_mean(y, y, block_size=bs, unroll=unroll)
        laplace_correction = 0.0
        entropic_regularisation = 0.0

        # Estimate entropic regularisation term (if regularisation strength is non-zero)
        if regularisation != 0:
            if regularisation is None:
                # Use regularisation parameter suggested in :cite:`benard2023kernel`
                regularisation = 1 / len(y)
            # Train logpdf with data from P, noticing we cannot guarantee that
            # kernel.base_kernel has a 'length_scale' attribute
            bandwidth_method = getattr(kernel.base_kernel, "length_scale", None)
            kde = jsp.stats.gaussian_kde(x.T, weights=w_x, bw_method=bandwidth_method)
            # Evaluate entropic regularisation term with data from Q
            entropic_regularisation = abs(regularisation) * kde.logpdf(y.data.T).sum()

        # Apply Laplace correction
        if laplace_correct:

            @vmap
            def _laplace_positive(x_: Array) -> Array:
                r"""Evaluate Laplace positive operator  :math:`\Delta^+ \log p(x)`."""
                hessian = jacfwd(kernel.score_function)(x_)
                return jnp.clip(jnp.diag(hessian), min=0.0).sum()

            laplace_correction = _laplace_positive(y.data)

        squared_ksd_threshold_applied = coreax.util.apply_negative_precision_threshold(
            squared_ksd + laplace_correction - entropic_regularisation,
            self.precision_threshold,
        )
        return jnp.sqrt(squared_ksd_threshold_applied)
