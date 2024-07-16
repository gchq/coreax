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
import jax.random as jr
import jax.scipy as jsp
import jax.tree_util as jtu
from jax import Array, jacfwd, vmap

import coreax.kernel
import coreax.util
from coreax.data import Data, SupervisedData
from coreax.inverses import LeastSquareApproximator, RegularisedInverseApproximator
from coreax.score_matching import ScoreMatching, convert_stein_kernel

_Data = TypeVar("_Data", bound=Data)


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


class MMD(Metric[Data]):
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
    normalised data weights.

    .. note::
        Assuming that the kernel is characteristic (:cite:`muandet2016rkhs`), it can be
        shown that :math:`\text{MMD}^2(\mathcal{D}_1,\mathcal{D}_2) = 0` if and only if
        :math:`\mathbb{P}^{(1)} = \mathbb{P}^{(2)}`, i.e. the distributions are the
        same. Therefore, the MMD gives us a way to measure if two datasets have the same
        (in the sense above) distribution.

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
        reference_data: Data,
        comparison_data: Data,
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
        _unroll = coreax.util.tree_leaves_repeat(unroll, 2)
        u_xx, u_xy, _, u_yy = tuple(product(_unroll, repeat=len(_unroll)))
        # Variable rename allows for nicer automatic formatting
        x, y = reference_data, comparison_data
        kernel_xx_mean = self.kernel.compute_mean(x, x, block_size=bs_xx, unroll=u_xx)
        kernel_yy_mean = self.kernel.compute_mean(y, y, block_size=bs_yy, unroll=u_yy)
        kernel_xy_mean = self.kernel.compute_mean(x, y, block_size=bs_xy, unroll=u_xy)
        squared_mmd_threshold_applied = coreax.util.apply_negative_precision_threshold(
            kernel_xx_mean + kernel_yy_mean - 2 * kernel_xy_mean,
            self.precision_threshold,
        )
        return jnp.sqrt(squared_mmd_threshold_applied)


class KSD(Metric[Data]):
    r"""
    Computation of the (regularised) (Laplace-corrected) kernel Stein discrepancy (KSD).

    For a set of ``n`` i.i.d. samples in ``d`` dimensions
    :math:`\mathcal{D}_1 \sim \mathbb{P}` and another set of ``m`` i.i.d. samples in
    ``d`` dimensions :math:`\mathcal{D}_2 \sim \mathbb{Q}`, the regularised
    Laplace-corrected kernel Stein discrepancy is given by:

    .. math::

        KSD_{\lambda}^2(\mathbb{P}, \mathbb{Q})
        =  \frac{1}{m^2}\sum_{i \neq j}^m k_{\mathbb{P}}(x_i, x_j)
        + \frac{1}{m^2}\sum_{i = 1}^m [k_{\mathbb{P}}(x_i, x_i)
        + \Delta^+ \log(\mathbb{P}(x_i))]
        - \lambda \frac{1}{m}\sum_{i = 1}^m \log(\mathbb{P}(x_i))

    where :math:`x \sim \mathbb{Q}`, :math:`k_{\mathbb{P}}` is the Stein kernel
    induced by a base kernel and estimated with samples from :math:`\mathbb{P}`.
    The first term is vanilla KSD, the second term implements a Laplace-correction, and
    the third term enforces entropic regularisation. See :cite:`benard2023kernel` for a
    discussion on the need for and effects of Laplace-correction and entropic
    regularisation.

    Common uses of KSD include comparing a reduced representation of a dataset to the
    original dataset, comparing different original datasets to one another, or
    comparing reduced representations of different original datasets to one another.

    .. note::
        The kernel stein discrepancy is not a metric like :class:`coreax.metrics.MMD`.
        It is instead a divergence, which is a kind of statistical distance that differs
        from a metric in a few ways. In particular, they are not symmetric. i.e.
        :math:`KSD_{\lambda}(\mathbb{P}, \mathbb{Q})
        \neq KSD_{\lambda}(\mathbb{Q}, \mathbb{P})`, and they generalise the concept
        of squared distance and so do not satisfy the triangle inequality.

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
    :param precision_threshold: Threshold above which negative values of the squared KSD
        are rounded to zero (accommodates precision loss)
    """

    kernel: coreax.kernel.Kernel
    score_matching: Optional[ScoreMatching] = None
    precision_threshold: float = 1e-12

    def compute(
        self,
        reference_data: Data,
        comparison_data: Data,
        *,
        laplace_correct: bool = False,
        regularise: bool = False,
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
            - \lambda \frac{1}{m}\sum_{i = 1}^m \log(\mathbb{P}(x_i))

        :param reference_data: An instance of the class :class:`coreax.data.Data`,
            containing an :math:`n \times d` array of data sampled from
            :math:`\mathbb{P}`
        :param comparison_data: An instance of the class :class:`coreax.data.Data` to
            compare against ``reference_data`` containing an :math:`m \times d` array of
            data sampled from :math:`\mathbb{Q}`
        :param laplace_correct: Boolean that enforces Laplace correction, see Section
            3.1 of :cite:`benard2023kernel`.
        :param regularise: Boolean that enforces entropic regularisation. :data:`True`,
            uses regularisation strength suggested in :cite:`benard2023kernel`.
            :math:`\lambda = \frac{1}{m}`.
        :param block_size: Size of matrix blocks to process; a value of :data:`None`
            sets ``block_size``:math:`=n` effectively disabling the block accumulation;
            an integer value ``B`` sets ``block_size``:math:`=B`, it is often sensible
            to select the largest block size that does not exhaust the available memory
            resources
        :param unroll: Unrolling parameter for the outer and inner :func:`jax.lax.scan`
            calls, allows for trade-offs between compilation and runtime cost; consult
            the JAX docs for further information
        :return: Kernel Stein Discrepancy as a 0-dimensional array
        """
        del kwargs
        # Train Stein kernel with data from P
        x, w_x = jtu.tree_leaves(reference_data)
        kernel = convert_stein_kernel(x, self.kernel, self.score_matching)

        # Variable rename allows for nicer automatic formatting.
        y = comparison_data
        squared_ksd = kernel.compute_mean(y, y, block_size=block_size, unroll=unroll)
        laplace_correction = 0.0
        entropic_regularisation = 0.0

        if regularise:
            # Train weighted kde with data from P, noticing we cannot guarantee that
            # kernel.base_kernel has a 'length_scale' attribute
            bandwidth_method = getattr(kernel.base_kernel, "length_scale", None)
            kde = jsp.stats.gaussian_kde(x.T, weights=w_x, bw_method=bandwidth_method)
            # Evaluate entropic regularisation term with data from Q using
            # regularisation parameter suggested in :cite:`benard2023kernel`
            entropic_regularisation = kde.logpdf(y.data.T).mean() / len(y)

        if laplace_correct:

            @vmap
            def _laplace_positive(x_: Array) -> Array:
                r"""Evaluate Laplace positive operator  :math:`\Delta^+ \log p(x)`."""
                hessian = jacfwd(kernel.score_function)(x_)
                return jnp.clip(jnp.diag(hessian), min=0.0).sum()

            laplace_correction = _laplace_positive(y.data).sum() / len(y) ** 2

        squared_ksd_threshold_applied = coreax.util.apply_negative_precision_threshold(
            squared_ksd + laplace_correction - entropic_regularisation,
            self.precision_threshold,
        )
        return jnp.sqrt(squared_ksd_threshold_applied)


class JMMD(Metric[SupervisedData]):
    r"""
    Definition and calculation of the (weighted) joint maximum mean discrepancy metric.

    For a dataset :math:`\mathcal{D}^{(1)} = \{(x_i, y_i)\}_{i=1}^n` of ``n`` pairs with
    :math:`x\in\mathbb{R}^d` and :math:`y\in\mathbb{R}^p`, and another dataset
    :math:`\mathcal{D}^{(2)} = \{(\tilde{x}_i, \tilde{y}_i)\}_{i=1}^m` of ``m`` pairs
    with :math:`\tilde{x}\in\mathbb{R}^d` and :math:`\tilde{y}\in\mathbb{R}^p`,
    the joint maximum mean discrepancy is given by:

    .. math::
        \text{JMMD}^2(\mathcal{D}_1,\mathcal{D}_2) = \mathbb{E}(k(\mathcal{D}_1,
        \mathcal{D}_1)) + \mathbb{E}(k(\mathcal{D}_2,\mathcal{D}_2))
        - 2\mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_2))

    where :math:`k` is a tensor-product kernel, and the expectation is with respect to
    the normalised data weights.

    .. note::
        Assuming that the feature and response kernels are characteristic
        (:cite:`muandet2016rkhs`), it can be shown
        that :math:`\text{JMMD}^2(\mathcal{D}_1,\mathcal{D}_2) = 0` if and only if
        :math:`\mathbb{P}^{(1)}_(X, Y) = \mathbb{P}^{(2)}_(X, Y)`, i.e. the joint
        distributions are the same. Therefore, the JMMD gives us a way to measure if two
        supervised datasets have the same (in the sense above) joint distribution.

    Common uses of JMMD include comparing a reduced representation of a dataset to the
    original dataset, comparing different original datasets to one another, or
    comparing reduced representations of different original datasets to one another.

    :param feature_kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` on
        the feature space
    :param response_kernel: :class:`~coreax.kernel.Kernel` instance implementing a
        kernel function :math:`k: \mathbb{R}^p \times \mathbb{R}^p \rightarrow
        \mathbb{R}` on the response space
    :param precision_threshold: Threshold above which negative values of the squared
        JMMD are rounded to zero (accommodates precision loss)
    """

    kernel: coreax.kernel.TensorProductKernel
    precision_threshold: float = 1e-12

    def __init__(
        self,
        feature_kernel: coreax.kernel.Kernel,
        response_kernel: coreax.kernel.Kernel,
        precision_threshold: float = 1e-12,
    ):
        """Initialise JMMD class and build tensor-product kernel."""
        self.kernel = coreax.kernel.TensorProductKernel(
            feature_kernel=feature_kernel,
            response_kernel=response_kernel,
        )
        self.precision_threshold = precision_threshold

    def compute(
        self,
        reference_data: SupervisedData,
        comparison_data: SupervisedData,
        *,
        block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]] = None,
        unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
        **kwargs,
    ) -> Array:
        r"""
        Compute the (weighted) joint maximum mean discrepancy.

        .. math::
            \text{JMMD}^2(\mathcal{D}_1,\mathcal{D}_2) = \mathbb{E}(k(\mathcal{D}_1,
            \mathcal{D}_1)) + \mathbb{E}(k(\mathcal{D}_2,\mathcal{D}_2))
            - 2\mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_2))

        :param reference_data: The original supervised dataset :math:`\mathcal{D}^{(1)}=
            \{(x_i, y_i)\}_{i=1}^n` of ``n`` pairs with :math:`x\in\mathbb{R}^d` and
            :math:`y\in\mathbb{R}^p`
        :param comparison_data: Supervised dataset
            :math:`\mathcal{D}^{(2)} = \{(\tilde{x}_i, \tilde{y}_i)\}_{i=1}^n` of ``m``
            pairs with :math:`\tilde{x}\in\mathbb{R}^d` and
            :math:`\tilde{y}\in\mathbb{R}^p`
        :param block_size: Size of matrix blocks to process; a value of :data:`None`
            sets :math:`B_x = n` and :math:`B_y = m`, effectively disabling the block
            accumulation; an integer value ``B`` sets :math:`B_y = B_x = B`; a tuple
            allows different sizes to be specified for ``B_x`` and ``B_y``; to reduce
            overheads, it is often sensible to select the largest block size that does
            not exhaust the available memory resources
        :param unroll: Unrolling parameter for the outer and inner :func:`jax.lax.scan`
            calls, allows for trade-offs between compilation and runtime cost; consult
            the JAX docs for further information
        :return: Joint maximum mean discrepancy as a 0-dimensional array
        """
        return MMD(self.kernel).compute(
            reference_data=reference_data,
            comparison_data=comparison_data,
            block_size=block_size,
            unroll=unroll,
        )


class CMMD(Metric[SupervisedData]):
    r"""
    Definition and calculation of the conditional maximum mean discrepancy metric.

    For a dataset :math:`\mathcal{D}^{(1)} = \{(x_i, y_i)\}_{i=1}^n` of ``n`` pairs with
    :math:`x\in\mathbb{R}^d` and :math:`y\in\mathbb{R}^p`, and another dataset
    :math:`\mathcal{D}^{(2)} = \{(\tilde{x}_i, \tilde{y}_i)\}_{i=1}^n` of ``m`` pairs
    with :math:`\tilde{x}\in\mathbb{R}^d` and :math:`\tilde{y}\in\mathbb{R}^p`,
    the conditional maximum mean discrepancy is given by:

    .. math::
        \text{CMMD}^2(\mathcal{D}^{(1)}, \mathcal{D}^{(2)}) =
        ||\hat{\mu}^{(1)}_{Y|X} - \hat{\mu}^{(2)_{Y|X}}||^2_{\mathcal{H}_k \otimes
        \mathcal{H}_l}

    where :math:`\hat{\mu}^{(1)}_{Y|X}, \hat{\mu}^{(2)}_{Y|X}` are the conditional mean
    embeddings (:cite:`muandet2016rkhs`) estimated with :math:`\mathcal{D}^{(1)}` and
    :math:`\mathcal{D}^{(2)}` respectively, and :math:`\mathcal{H}_k, \mathcal{H}_l` are
    the RKHSs corresponding to the kernel functions :math:`k: \mathbb{R}^d \times
    \mathbb{R}^d \rightarrow \mathbb{R}` and :math:`l: \mathbb{R}^p \times \mathbb{R}^p
    \rightarrow \mathbb{R}` respectively.

    .. note::
        Given certain assumptions (:cite:`ren2016conditional`), including that
        the feature kernel is characteristic (:cite:`muandet2016rkhs`), it can be shown
        that if :math:`\mu^{(1)}_{Y|X} = \mu^{(2)}_{Y|X}`, then
        :math:`\mathbb{P}^{(1)}_{Y|X} = \mathbb{P}^{(2)}_{Y|X}` in the sense that for
        every fixed :math:`x`, we have
        :math:`\mathbb{P}^{(1)}_{Y|x} = \mathbb{P}^{(2)}_{Y|x}`.
        Therefore, the CMMD gives us a way to measure if two supervised datasets have
        the same (in the sense above) conditional distributions.

    :param feature_kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` on
        the feature space
    :param response_kernel: :class:`~coreax.kernel.Kernel` instance implementing a
        kernel function :math:`k: \mathbb{R}^p \times \mathbb{R}^p \rightarrow
        \mathbb{R}` on the response space
    :param regularisation_parameter: Regularisation parameter for stable inversion
            of arrays, negative values will be converted to positive
    :param precision_threshold: Positive threshold we compare against for precision
    :param inverse_approximator: Instance of
        :class:`coreax.inverses.RegularisedInverseApproximator`, defaults to
        :class:`coreax.inverses.LeastSquareApproximator` which solves a linear
        system at cost :math:`\mathcal{O}(n^3)`
    """

    feature_kernel: coreax.kernel.Kernel
    response_kernel: coreax.kernel.Kernel
    regularisation_parameter: float
    precision_threshold: float = 1e-2
    inverse_approximator: Optional[RegularisedInverseApproximator] = None

    def compute(
        self,
        reference_data: SupervisedData,
        comparison_data: SupervisedData,
        **kwargs,
    ) -> Array:
        r"""
        Compute the conditional maximum mean discrepancy between the two datasets.

        For a dataset :math:`\mathcal{D}^{(1)} = \{(x_i, y_i)\}_{i=1}^n=`
        `reference_data` of ``n`` pairs with :math:`x\in\mathbb{R}^d` and
        :math:`y\in\mathbb{R}^p`, and another dataset
        :math:`\mathcal{D}^{(2)} = \{(\tilde{x}_i, \tilde{y}_i)\}_{i=1}^n=`
        `comparison_data` of ``n`` pairs with
        :math:`\tilde{x}\in\mathbb{R}^d` and :math:`\tilde{y}\in \mathbb{R}^p`, the
        conditional maximum mean discrepancy is given by:

        .. math::
            \text{CMMD}^2(\mathcal{D}^{(1)}, \mathcal{D}^{(2)}) = ||\hat{\mu}^{(1)} -
            \hat{\mu}^{(2)}||^2_{\mathcal{H}_k \otimes \mathcal{H}_l}

        where :math:`\hat{\mu}^{(1)},\hat{\mu}^{(2)}` are the conditional mean
        embeddings estimated with :math:`\mathcal{D}^{(1)}` and
        :math:`\mathcal{D}^{(2)}` respectively, and :math:`\mathcal{H}_k,\mathcal{H}_l`
        are the RKHSs corresponding to the kernel functions
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` and
        :math:`l: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}` respectively.

        :param reference_data: The original supervised dataset :math:`\mathcal{D}^{(1)}=
            \{(x_i, y_i)\}_{i=1}^n` of ``n`` pairs with :math:`x\in\mathbb{R}^d` and
            :math:`y\in\mathbb{R}^p`
        :param comparison_data: Supervised dataset
            :math:`\mathcal{D}^{(2)} = \{(\tilde{x}_i, \tilde{y}_i)\}_{i=1}^n` of ``m``
            pairs with :math:`\tilde{x}\in\mathbb{R}^d` and
            :math:`\tilde{y}\in\mathbb{R}^p`
        :return: Conditional maximum mean discrepancy as a 0-dimensional array
        """
        del kwargs
        # Extract features and responses from reference and comparison datasets
        x1, y1 = reference_data.data, reference_data.supervision
        x2, y2 = comparison_data.data, comparison_data.supervision
        # Compute feature kernel gramians
        feature_gramian_1 = self.feature_kernel.compute(x1, x1)
        feature_gramian_2 = self.feature_kernel.compute(x2, x2)

        # Invert feature kernel gramians
        if self.inverse_approximator is None:
            inverse_approximator = LeastSquareApproximator(jr.key(2_024))
        else:
            inverse_approximator = self.inverse_approximator
        inverse_feature_gramian_1 = inverse_approximator.approximate(
            array=feature_gramian_1,
            regularisation_parameter=self.regularisation_parameter,
            identity=jnp.eye(feature_gramian_1.shape[0]),
        )

        inverse_feature_gramian_2 = inverse_approximator.approximate(
            array=feature_gramian_2,
            regularisation_parameter=self.regularisation_parameter,
            identity=jnp.eye(feature_gramian_2.shape[0]),
        )

        # # Compute each term in the CMMD
        term_1 = (
            inverse_feature_gramian_1
            @ self.response_kernel.compute(y1, y1)
            @ inverse_feature_gramian_1
            @ feature_gramian_1
        )
        term_2 = (
            inverse_feature_gramian_2
            @ self.response_kernel.compute(y2, y2)
            @ inverse_feature_gramian_2
            @ feature_gramian_2
        )
        term_3 = (
            inverse_feature_gramian_1
            @ self.response_kernel.compute(y1, y2)
            @ inverse_feature_gramian_2
            @ self.feature_kernel.compute(x2, x1)
        )

        # Compute CMMD
        squared_cmmd = jnp.trace(term_1) + jnp.trace(term_2) - 2 * jnp.trace(term_3)
        squared_cmmd_threshold_applied = coreax.util.apply_negative_precision_threshold(
            squared_cmmd,
            self.precision_threshold,
        )
        return jnp.sqrt(squared_cmmd_threshold_applied)
