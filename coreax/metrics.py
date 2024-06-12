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

import coreax.data
import coreax.kernel
import coreax.util

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
