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

# Support annotations with | in Python < 3.10
from __future__ import annotations

from abc import abstractmethod
from itertools import product
from typing import Generic, Optional, TypeVar
from warnings import warn

import equinox as eqx
import jax.numpy as jnp
from jax import Array

import coreax.data
import coreax.kernel
import coreax.util
from coreax.approximation import RegularisedInverseApproximator

_Data = TypeVar("_Data", bound=coreax.data.Data)
_SupervisedData = TypeVar("_SupervisedData", bound=coreax.data.SupervisedData)


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
        block_size: int | None | tuple[int | None, int | None] = None,
        unroll: tuple[int | bool, int | bool] = (1, 1),
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


class CMMD(Metric):
    r"""
    Definition and calculation of the conditional maximum mean discrepancy metric.

    For a dataset :math:`\mathcal{D}^{(1)} = \{(x_i, y_i)\}_{i=1}^n` of ``n`` pairs with
    :math:`x\in\mathbb{R}^d` and :math:`y\in\mathbb{R}^p`, and another dataset
    :math:`\mathcal{D}^{(2)} = \{(\tilde{x}_i, \tilde{y}_i)\}_{i=1}^n` of ``m`` pairs
    with :math:`\tilde{x}\in\mathbb{R}^d` and :math:`\tilde{y}\in\mathbb{R}^p`,
    the conditional maximum mean discrepancy is given by:

    .. math::

        \text{CMMD}^2(\mathcal{D}^{(1)}, \mathcal{D}^{(2)}) =
        ||\hat{\mu}^{(1)} - \hat{\mu}^{(2)}||^2_{\mathcal{H}_k \otimes \mathcal{H}_l}

    where :math:`\hat{\mu}^{(1)},\hat{\mu}^{(2)}` are the conditional mean
    embeddings estimated with :math:`\mathcal{D}^{(1)}` and :math:`\mathcal{D}^{(2)}`
    respectively, and :math:`\mathcal{H}_k,\mathcal{H}_l` are the RKHSs corresponding
    to the kernel functions :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow
    \mathbb{R}` and :math:`l: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}`
    respectively.

    :param feature_kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` on
        the feature space
    :param response_kernel: :class:`~coreax.kernel.Kernel` instance implementing a
        kernel function :math:`k: \mathbb{R}^p \times \mathbb{R}^p \rightarrow
        \mathbb{R}` on the response space
    :param num_feature_dimensions: An integer representing the dimensionality of the
        features :math:`x`
    :param regularisation_parameters: A  :math:`1 \times 2` array of regularisation
        parameters corresponding to the first dataset :math:`\mathcal{D}^{(1)}` and
        second datasets :math:`\mathcal{D}^{(2)}` respectively
    :param precision_threshold: Positive threshold we compare against for precision
    """

    def __init__(
        self,
        feature_kernel: coreax.kernel.Kernel,
        response_kernel: coreax.kernel.Kernel,
        num_feature_dimensions: int,
        regularisation_parameters: float,
        precision_threshold: float = 1e-6,
        inverse_approximator: Optional[RegularisedInverseApproximator] = None,
    ):
        """Initialise CMMD class."""
        self.feature_kernel = feature_kernel
        self.response_kernel = response_kernel
        self.num_feature_dimensions = num_feature_dimensions
        self.regularisation_parameters = regularisation_parameters
        self.precision_threshold = precision_threshold
        self.inverse_approximator = inverse_approximator

        # Initialise parent
        super().__init__()

    def compute(
        self,
        reference_data: _SupervisedData,
        comparison_data: _SupervisedData,
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
        x1 = reference_data.data
        y1 = reference_data.supervision
        x2 = comparison_data.data
        y2 = comparison_data.supervision

        # Compute feature kernel gramians
        feature_gramian_1 = self.feature_kernel.compute(x1, x1)
        feature_gramian_2 = self.feature_kernel.compute(x2, x2)

        # Invert feature kernel gramians, approximating if required
        if self.inverse_approximator is not None:
            inverse_feature_gramian_1 = self.inverse_approximator.approximate(
                kernel_gramian=feature_gramian_1,
                regularisation_parameter=self.regularisation_parameters,
                identity=jnp.eye(feature_gramian_1.shape[0]),
            )

            inverse_feature_gramian_2 = self.inverse_approximator.approximate(
                kernel_gramian=feature_gramian_2,
                regularisation_parameter=self.regularisation_parameters,
                identity=jnp.eye(feature_gramian_2.shape[0]),
            )
        else:
            inverse_feature_gramian_1 = coreax.util.invert_regularised_array(
                array=feature_gramian_1,
                regularisation_parameter=self.regularisation_parameters,
                identity=jnp.eye(feature_gramian_1.shape[0]),
            )
            inverse_feature_gramian_2 = coreax.util.invert_regularised_array(
                array=feature_gramian_2,
                regularisation_parameter=self.regularisation_parameters,
                identity=jnp.eye(feature_gramian_2.shape[0]),
            )

        # Compute each term in the CMMD
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

        # Compute CMMD, warn if numerical instability has resulted in a spurious value
        squared_result = jnp.trace(term_1) + jnp.trace(term_2) - 2 * jnp.trace(term_3)
        if squared_result < 0:
            warn(
                f"Squared CMMD ({round(squared_result.item(), 4)}) is negative,"
                + " increase precision threshold or regularisation strength, "
                + "returning zero..",
                Warning,
                stacklevel=2,
            )
            return jnp.array([0])
        result = jnp.sqrt(
            coreax.util.apply_negative_precision_threshold(
                squared_result,
                self.precision_threshold,
            )
        )
        return result
