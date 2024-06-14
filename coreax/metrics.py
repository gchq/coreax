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
from typing import Generic, TypeVar, Union

import equinox as eqx
import jax.numpy as jnp
from jax import Array

import coreax.util
from coreax.data import Data, SupervisedData
from coreax.kernel import Kernel, TensorProductKernel

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
    normalized data weights.

    Common uses of MMD include comparing a reduced representation of a dataset to the
    original dataset, comparing different original datasets to one another, or
    comparing reduced representations of different original datasets to one another.

    :param kernel: Kernel object with compute method defined mapping
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param precision_threshold: Threshold above which negative values of the squared MMD
        are rounded to zero (accommodates precision loss)
    """

    kernel: Kernel
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


class JMMD(Metric[SupervisedData]):
    r"""
    Definition and calculation of the (weighted) joint maximum mean discrepancy metric.

    For a dataset :math:`\mathcal{D}^{(1)} = \{(x_i, y_i)\}_{i=1}^n` of ``n`` pairs with
    :math:`x\in\mathbb{R}^d` and :math:`y\in\mathbb{R}^p`, and another dataset
    :math:`\mathcal{D}^{(2)} = \{(\tilde{x}_i, \tilde{y}_i)\}_{i=1}^n` of ``m`` pairs
    with :math:`\tilde{x}\in\mathbb{R}^d` and :math:`\tilde{y}\in\mathbb{R}^p`,
    the joint maximum mean discrepancy is given by:

    .. math::
        \text{JMMD}^2(\mathcal{D}_1,\mathcal{D}_2) = \mathbb{E}(k(\mathcal{D}_1,
        \mathcal{D}_1)) + \mathbb{E}(k(\mathcal{D}_2,\mathcal{D}_2))
        - 2\mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_2))

    where :math:`k` is the selected tensor-product kernel, and the expectation is with
    respect to the normalized data weights.

    Common uses of JMMD include comparing a reduced representation of a dataset to the
    original dataset, comparing different original datasets to one another, or
    comparing reduced representations of different original datasets to one another.

    :param kernel: :class:`~coreax.kernel.TensorProductKernel` instance implementing a
        tensor-product kernel function
        :math:`k: (\mathbb{R}^d \times \mathbb{R}^p) \times
        (\mathbb{R}^d \times \mathbb{R}^p) \rightarrow \mathbb{R}`
    :param precision_threshold: Threshold above which negative values of the squared MMD
        are rounded to zero (accommodates precision loss)
    """

    kernel: TensorProductKernel
    precision_threshold: float = 1e-12

    def __init__(
        self,
        feature_kernel: Kernel,
        response_kernel: Kernel,
        precision_threshold: float = 1e-12,
    ):
        """Initialise JointKernelHerding class and build tensor-product kernel."""
        self.kernel = TensorProductKernel(
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
        del kwargs
        _block_size = coreax.util.tree_leaves_repeat(block_size, 2)
        bs_aa, bs_ab, _, bs_bb = tuple(product(_block_size, repeat=len(_block_size)))
        # Variable rename allows for nicer automatic formatting
        x, y = reference_data, comparison_data
        kernel_aa_mean = self.kernel.compute_mean(x, x, block_size=bs_aa, unroll=unroll)
        kernel_bb_mean = self.kernel.compute_mean(y, y, block_size=bs_bb, unroll=unroll)
        kernel_ab_mean = self.kernel.compute_mean(x, y, block_size=bs_ab, unroll=unroll)
        squared_mmd_threshold_applied = coreax.util.apply_negative_precision_threshold(
            kernel_aa_mean + kernel_bb_mean - 2 * kernel_ab_mean,
            self.precision_threshold,
        )
        return jnp.sqrt(squared_mmd_threshold_applied)
