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

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from warnings import warn

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import coreax.kernel
import coreax.proposed_data
import coreax.util

_Data = TypeVar("_Data", bound=coreax.proposed_data.WeightedData)
_SupervisedData = TypeVar(
    "_SupervisedData", bound=coreax.proposed_data.SupervisedWeightedData
)


class Metric(ABC, Generic[_Data]):
    """Base class for calculating metrics."""

    @abstractmethod
    def compute(
        self,
        reference_data,
        comparison_data,
    ) -> Array:
        r"""
        Compute the metric.

        Return a zero-dimensional array.

        :param reference_data: An :math:`n \times d` array defining the full dataset
        :param comparison_data: An :math:`m \times d` array defining a representation of
            ``reference_data``, for example a coreset
        :return: Metric computed as a zero-dimensional array
        """


class MMD(Metric):
    r"""
    Definition and calculation of the maximum mean discrepancy metric.

    For a dataset of ``n`` points in ``d`` dimensions, :math:`\mathcal_{D}_1`, and
    another dataset :math:`\mathcal_{D}_2` of ``m`` points in ``d`` dimensions, the
    maximum mean discrepancy is given by:

    .. math::

        \text{MMD}^2(\mathcal_{D}_1,\mathcal_{D}_2) = \mathbb{E}(k(\mathcal_{D}_1,
        \mathcal_{D}_1)) + \mathbb{E}(k(\mathcal_{D}_2,\mathcal_{D}_2))
        - 2\mathbb{E}(k(\mathcal_{D}_1,\mathcal_{D}_2))

    where :math:`k` is the selected kernel.

    :param kernel: Kernel object with compute method defined mapping
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param precision_threshold: Positive threshold we compare against for precision
    """

    def __init__(self, kernel: coreax.kernel.Kernel, precision_threshold: float = 1e-8):
        """Calculate maximum mean discrepancy between two datasets."""
        self.kernel = kernel
        self.precision_threshold = precision_threshold

        # Initialise parent
        super().__init__()

    def compute(
        self,
        reference_data,
        comparison_data,
        block_size: int | None = None,
        reference_weights: ArrayLike | None = None,
        comparison_weights: ArrayLike | None = None,
    ) -> Array:
        r"""
        Calculate maximum mean discrepancy.

        If no weights are given for dataset ``reference_data``, standard MMD is
        calculated. If weights are given, weighted MMD is calculated. For both cases, if
        the size of matrix blocks to process is less than the size of both datasets and
        if ``reference_weights`` is given in the weighted case, the calculation
        is done block-wise to limit memory requirements.

        :param reference_data: The original :math:`n \times d` data
        :param comparison_data: :math:`m \times d` dataset
        :param block_size: Size of matrix block to process, or :data:`None` to not split
            into blocks
        :param reference_weights: An :math:`1 \times n` array of weights for
            associated points in ``reference_data``, or :data:`None` if not required
        :param comparison_weights: An :math:`1 \times m` array of weights for
            associated points in ``comparison_data``, or :data:`None` if not required
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Format inputs
        reference_data = jnp.atleast_2d(reference_data)
        comparison_data = jnp.atleast_2d(comparison_data)
        if reference_weights is not None:
            reference_weights = jnp.atleast_1d(reference_weights)
        if comparison_weights is not None:
            comparison_weights = jnp.atleast_1d(comparison_weights)

        num_reference_points = len(reference_data)
        num_comparison_points = len(comparison_data)

        if comparison_weights is None:
            if block_size is None or block_size > max(
                num_reference_points, num_comparison_points
            ):
                return self.maximum_mean_discrepancy(reference_data, comparison_data)
            return self.maximum_mean_discrepancy_block(
                reference_data, comparison_data, block_size
            )

        if (
            block_size is None
            or reference_weights is None
            or block_size > max(num_reference_points, num_comparison_points)
        ):
            return self.weighted_maximum_mean_discrepancy(
                reference_data, comparison_data, comparison_weights
            )

        return self.weighted_maximum_mean_discrepancy_block(
            reference_data,
            comparison_data,
            reference_weights,
            comparison_weights,
            block_size,
        )

    def maximum_mean_discrepancy(
        self, reference_data: ArrayLike, comparison_data: ArrayLike
    ) -> Array:
        r"""
        Calculate standard, unweighted MMD.

        For a dataset of ``n`` points in ``d`` dimensions, :math:`\mathcal_{D}_1`, and
        another dataset :math:`\mathcal_{D}_2` of ``m`` points in ``d`` dimensions, the
        maximum mean discrepancy is given by:

        .. math::

            \text{MMD}^2(\mathcal_{D}_1,\mathcal_{D}_2) = \mathbb{E}(k(\mathcal_{D}_1,
            \mathcal_{D}_1)) + \mathbb{E}(k(\mathcal_{D}_2,\mathcal_{D}_2))
            - 2\mathbb{E}(k(\mathcal_{D}_1,\mathcal_{D}_2))

        where :math:`k` is the selected kernel.

        :param reference_data: The original :math:`n \times d` data
        :param comparison_data: An :math:`m \times d` array defining a representation of
          ``reference_data``, for example a coreset
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Format inputs
        reference_data = jnp.atleast_2d(reference_data)
        comparison_data = jnp.atleast_2d(comparison_data)

        # Compute each term in the MMD formula
        kernel_nn = self.kernel.compute(reference_data, reference_data)
        kernel_mm = self.kernel.compute(comparison_data, comparison_data)
        kernel_nm = self.kernel.compute(reference_data, comparison_data)

        # Compute MMD
        result = jnp.sqrt(
            coreax.util.apply_negative_precision_threshold(
                kernel_nn.mean() + kernel_mm.mean() - 2 * kernel_nm.mean(),
                self.precision_threshold,
            )
        )
        return result

    def weighted_maximum_mean_discrepancy(
        self,
        reference_data: ArrayLike,
        comparison_data: ArrayLike,
        comparison_weights: ArrayLike,
    ) -> Array:
        r"""
        Calculate one-sided, weighted maximum mean discrepancy (WMMD).

        Only data points in ``comparison_data`` are weighted.

        :param reference_data: The original :math:`n \times d` data
        :param comparison_data: An :math:`m \times d` array defining a representation of
             ``reference_data``, for example a coreset
        :param comparison_weights: :math:`m \times 1` weights vector for data
            ``comparison_data``
        :return: Weighted maximum mean discrepancy as a 0-dimensional array
        """
        # Format inputs
        reference_data = jnp.atleast_2d(reference_data)
        comparison_data = jnp.atleast_2d(comparison_data)
        comparison_weights = jnp.atleast_1d(comparison_weights)

        num_reference_points = float(len(reference_data))

        # Compute each term in the weighted MMD formula
        kernel_nn = self.kernel.compute(reference_data, reference_data)
        kernel_mm = self.kernel.compute(comparison_data, comparison_data)
        kernel_nm = self.kernel.compute(reference_data, comparison_data)

        # Compute weighted MMD, correcting for any numerical precision issues, where we
        # would otherwise square-root a negative number very close to 0.0.
        result = jnp.sqrt(
            coreax.util.apply_negative_precision_threshold(
                jnp.dot(comparison_weights.T, jnp.dot(kernel_mm, comparison_weights))
                + kernel_nn.sum() / num_reference_points**2
                - 2 * jnp.dot(comparison_weights.T, kernel_nm.mean(axis=0)),
                self.precision_threshold,
            )
        )
        return result

    def maximum_mean_discrepancy_block(
        self,
        reference_data: ArrayLike,
        comparison_data: ArrayLike,
        block_size: int = 10_000,
    ) -> Array:
        r"""
        Calculate maximum mean discrepancy (MMD) whilst limiting memory requirements.

        :param reference_data: The original :math:`n \times d` data
        :param comparison_data: An :math:`m \times d` array defining a representation of
            ``reference_data``, for example a coreset
        :param block_size: Size of matrix blocks to process
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Format inputs
        reference_data = jnp.atleast_2d(reference_data)
        comparison_data = jnp.atleast_2d(comparison_data)

        num_reference_points = float(len(reference_data))
        num_comparison_points = float(len(comparison_data))

        # Compute each term in the weighted MMD formula
        kernel_nn = self.sum_pairwise_distances(
            reference_data, reference_data, block_size
        )
        kernel_mm = self.sum_pairwise_distances(
            comparison_data, comparison_data, block_size
        )
        kernel_nm = self.sum_pairwise_distances(
            reference_data, comparison_data, block_size
        )

        # Compute MMD, correcting for any numerical precision issues, where we would
        # otherwise square-root a negative number very close to 0.0.
        result = jnp.sqrt(
            coreax.util.apply_negative_precision_threshold(
                kernel_nn / num_reference_points**2
                + kernel_mm / num_comparison_points**2
                - 2 * kernel_nm / (num_reference_points * num_comparison_points),
                self.precision_threshold,
            )
        )
        return result

    def weighted_maximum_mean_discrepancy_block(
        self,
        reference_data: ArrayLike,
        comparison_data: ArrayLike,
        reference_weights: ArrayLike,
        comparison_weights: ArrayLike,
        block_size: int = 10_000,
    ) -> Array:
        r"""
        Calculate weighted maximum mean discrepancy (MMD).

        This calculation is executed whilst limiting memory requirements.

        :param reference_data: The original :math:`n \times d` data
        :param comparison_data: An :math:`m \times d` array defining a representation of
            ``reference_data``, for example a coreset
        :param reference_weights: :math:`n` weights of reference data
        :param comparison_weights: :math:`m` weights of points in ``comparison_data``
        :param block_size: Size of matrix blocks to process
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Format inputs
        reference_data = jnp.atleast_2d(reference_data)
        comparison_data = jnp.atleast_2d(comparison_data)
        reference_weights = jnp.atleast_1d(reference_weights)
        comparison_weights = jnp.atleast_1d(comparison_weights)

        num_reference_points = reference_weights.sum()
        num_comparison_points = comparison_weights.sum()

        kernel_nn = self.sum_weighted_pairwise_distances(
            reference_data,
            reference_data,
            reference_weights,
            reference_weights,
            block_size,
        )
        kernel_mm = self.sum_weighted_pairwise_distances(
            comparison_data,
            comparison_data,
            comparison_weights,
            comparison_weights,
            block_size,
        )
        kernel_nm = self.sum_weighted_pairwise_distances(
            reference_data,
            comparison_data,
            reference_weights,
            comparison_weights,
            block_size,
        )

        # Compute MMD, correcting for any numerical precision issues, where we would
        # otherwise square-root a negative number very close to 0.0.
        result = jnp.sqrt(
            coreax.util.apply_negative_precision_threshold(
                kernel_nn / num_reference_points**2
                + kernel_mm / num_comparison_points**2
                - 2 * kernel_nm / (num_reference_points * num_comparison_points),
                self.precision_threshold,
            )
        )
        return result

    def sum_pairwise_distances(
        self,
        reference_data: ArrayLike,
        comparison_data: ArrayLike,
        block_size: int = 10_000,
    ) -> float:
        r"""
        Sum the kernel distance between all pairs of points in the two input datasets.

        The summation is done in blocks to avoid excessive memory usage.

        :param reference_data: :math:`n \times 1` array
        :param comparison_data: :math:`m \times 1` array
        :param block_size: Size of matrix blocks to process
        :return: The sum of pairwise distances between points in ``reference_data`` and
            ``comparison_data``
        """
        # Format inputs
        reference_data = jnp.atleast_2d(reference_data)
        comparison_data = jnp.atleast_2d(comparison_data)
        block_size = max(0, block_size)

        num_reference_points = len(reference_data)
        num_comparison_points = len(comparison_data)

        # If block_size is larger than both inputs, we don't need to consider block-wise
        # computation
        if block_size > max(num_reference_points, num_comparison_points):
            pairwise_distance_sum = self.kernel.compute(
                reference_data, comparison_data
            ).sum()

        else:
            try:
                row_index_range = range(0, num_reference_points, block_size)
            except ValueError as exception:
                if block_size == 0:
                    raise ValueError(
                        "block_size must be a positive integer"
                    ) from exception
                raise
            except TypeError as exception:
                raise TypeError("block_size must be a positive integer") from exception

            pairwise_distance_sum = 0
            for i in row_index_range:
                for j in range(0, num_comparison_points, block_size):
                    pairwise_distances_part = self.kernel.compute(
                        reference_data[i : i + block_size],
                        comparison_data[j : j + block_size],
                    )
                    pairwise_distance_sum += pairwise_distances_part.sum()

        return pairwise_distance_sum

    def sum_weighted_pairwise_distances(
        self,
        reference_data: ArrayLike,
        comparison_data: ArrayLike,
        reference_weights: ArrayLike,
        comparison_weights: ArrayLike,
        block_size: int = 10_000,
    ) -> float:
        r"""
        Sum weighted kernel distance between all pairs of points in the two datasets.

        The summation is done in blocks to avoid excessive memory usage.

        :param reference_data: :math:`n \times 1` array
        :param comparison_data: :math:`m \times 1` array
        :param reference_weights: :math:`n \times 1` array of weights for
            ``reference_data``
        :param comparison_weights: :math:`m \times 1` array of weights for
            ``comparison_data``
        :param block_size: Size of matrix blocks to process
        :return: The sum of pairwise distances between points in ``reference_data`` and
            ``comparison_data``, with contributions weighted as defined by
            ``reference_weights`` and ``comparison_weights``
        """
        # Format inputs
        reference_data = jnp.atleast_2d(reference_data)
        comparison_data = jnp.atleast_2d(comparison_data)
        reference_weights = jnp.atleast_1d(reference_weights)
        comparison_weights = jnp.atleast_1d(comparison_weights)
        block_size = max(0, block_size)

        num_reference_points = len(reference_data)
        num_comparison_points = len(comparison_data)

        # If block_size is larger than both inputs, we don't need to consider block-wise
        # computation
        if block_size > max(num_reference_points, num_comparison_points):
            kernel_weights = (
                self.kernel.compute(reference_data, comparison_data)
                * comparison_weights
            )
            weighted_pairwise_distance_sum = (
                reference_weights * kernel_weights.T
            ).sum()

        else:
            weighted_pairwise_distance_sum = 0

            try:
                row_index_range = range(0, num_reference_points, block_size)
            except ValueError as exception:
                if block_size == 0:
                    raise ValueError(
                        "block_size must be a positive integer"
                    ) from exception
                raise
            except TypeError as exception:
                raise TypeError("block_size must be a positive integer") from exception

            for i in row_index_range:
                for j in range(0, num_comparison_points, block_size):
                    pairwise_distances_part = (
                        reference_weights[i : i + block_size, None]
                        * self.kernel.compute(
                            reference_data[i : i + block_size],
                            comparison_data[j : j + block_size],
                        )
                        * comparison_weights[None, j : j + block_size]
                    )
                    weighted_pairwise_distance_sum += pairwise_distances_part.sum()

        return weighted_pairwise_distance_sum


class CMMD(Metric, Generic[_SupervisedData]):
    r"""
    Definition and calculation of the conditional maximum mean discrepancy metric.

    For a dataset :math:`\mathcal{D}_1 = \{(x_i, y_i)\}_{i=1}^n` of ``n`` pairs with
    :math:`x\in\mathbb{R}^d` and :math:`y\in\mathbb{R}^p`, and another dataset
    :math:`\mathcal{D}_2 = \{(\tilde{x}_i, \tilde{y}_i)\}_{i=1}^n` of ``m`` pairs
    with :math:`\tilde{x}\in\mathbb{R}^d` and :math:`\tilde{y}\in\mathbb{R}^p`,
    the conditional maximum mean discrepancy is given by:

    .. math::

        \text{CMMD}^2(\mathcal{D}_1, \mathcal{D}_2) =
        ||\hat{\mu}_1 - \hat{\mu}_2||^2_{\mathcal{H}_k \otimes \mathcal{H}_l}

    where :math:`\hat{\mu}_1,\hat{\mu}_2` are the conditional mean
    embeddings estimated with :math:`\mathcal{D}_1` and :math:`\mathcal{D}_2`
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
        parameters corresponding to the first dataset :math:`\mathcal{D}_1` and
        second datasets :math:`\mathcal{D}_2` respectively
    :param precision_threshold: Positive threshold we compare against for precision
    """

    def __init__(
        self,
        feature_kernel: coreax.kernel.Kernel,
        response_kernel: coreax.kernel.Kernel,
        num_feature_dimensions: int,
        regularisation_parameters: ArrayLike,
        precision_threshold: float = 1e-6,
    ):
        """Calculate conditional maximum mean discrepancy between two datasets."""
        self.feature_kernel = feature_kernel
        self.response_kernel = response_kernel
        self.num_feature_dimensions = num_feature_dimensions
        self.regularisation_parameters = regularisation_parameters
        self.precision_threshold = precision_threshold

        # Initialise parent
        super().__init__()

    def compute(self, reference_data: ArrayLike, comparison_data: ArrayLike) -> Array:
        r"""
        Calculate conditional maximum mean discrepancy.

        :param reference_data: The original dataset :math:`\mathcal{D}_1 =
            \{(x_i, y_i)\}_{i=1}^n` of ``n`` pairs with :math:`x\in\mathbb{R}^d` and
            :math:`y\in\mathbb{R}^p`, responses should be concatenated after the
            features
        :param comparison_data: Dataset :math:`\mathcal{D}_2 = \{(\tilde{x}_i,
            \tilde{y}_i)\}_{i=1}^n` of ``m`` pairs with :math:`\tilde{x}\in\mathbb{R}^d`
            and :math:`\tilde{y}\in\mathbb{R}^p`, responses should be
            concatenated after the features
        :return: Conditional maximum mean discrepancy as a 0-dimensional array
        """
        return self.conditional_maximum_mean_discrepancy(
            reference_data, comparison_data
        )

    def conditional_maximum_mean_discrepancy(
        self, reference_data: ArrayLike, comparison_data: ArrayLike
    ) -> Array:
        r"""
        Calculate standard conditional maximum mean discrepancy metric.

        For a dataset :math:`\mathcal{D}_1 = \{(x_i, y_i)\}_{i=1}^n` of ``n`` pairs
        with :math:`x\in\mathbb{R}^d` and :math:`y\in\mathbb{R}^p`, and another dataset
        :math:`\mathcal{D}_2 = \{(\tilde{x}_i, \tilde{y}_i)\}_{i=1}^n` of ``n``
        pairs with :math:`\tilde{x}\in\mathbb{R}^d` and :math:`\tilde{y}\in
        \mathbb{R}^p`, the conditional maximum mean discrepancy is given by:

        .. math::

            \text{CMMD}^2(\mathcal{D}_1, \mathcal{D}_2) = ||\hat{\mu}_1 -
            \hat{\mu}_2||^2_{\mathcal{H}_k \otimes \mathcal{H}_l}

        where :math:`\hat{\mu}_1,\hat{\mu}_2` are the conditional mean
        embeddings estimated with :math:`\mathcal{D}_1` and
        :math:`\mathcal{D}_2` respectively, and :math:`\mathcal{H}_k,\mathcal{H}_l`
        are the RKHSs corresponding to the kernel functions
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` and
        :math:`l: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}` respectively.

        :param reference_data: The original dataset :math:`\mathcal{D}_1 =
            \{(x_i, y_i)\}_{i=1}^n` of ``n`` pairs with :math:`x\in\mathbb{R}^d` and
            :math:`y\in\mathbb{R}^p`, responses should be concatenated after the
            features
        :param comparison_data: Dataset :math:`\mathcal{D}_2 = \{(\tilde{x}_i,
            \tilde{y}_i)\}_{i=1}^n` of ``m`` pairs with :math:`\tilde{x}\in\mathbb{R}^d`
            and :math:`\tilde{y}\in\mathbb{R}^p`, responses should be
            concatenated after the features
        """
        # Extract and format features and responses from D1 and D2
        x1 = jnp.atleast_2d(reference_data[:, : self.num_feature_dimensions])
        y1 = jnp.atleast_2d(reference_data[:, self.num_feature_dimensions :])
        x2 = jnp.atleast_2d(comparison_data[:, : self.num_feature_dimensions])
        y2 = jnp.atleast_2d(comparison_data[:, self.num_feature_dimensions :])

        # Compute feature kernel gramians
        feature_gramian_1 = self.feature_kernel.compute(x1, x1)
        feature_gramian_2 = self.feature_kernel.compute(x2, x2)

        # Invert feature kernel gramians
        inverse_feature_gramian_1 = coreax.util.invert_regularised_array(
            array=feature_gramian_1,
            regularisation_parameter=self.regularisation_parameters[0],
            identity=jnp.eye(feature_gramian_1.shape[0]),
        )
        inverse_feature_gramian_2 = coreax.util.invert_regularised_array(
            array=feature_gramian_2,
            regularisation_parameter=self.regularisation_parameters[1],
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

        # Compute CMMD
        squared_result = jnp.trace(term_1) + jnp.trace(term_2) - 2 * jnp.trace(term_3)
        if squared_result < 0:
            warn(
                f"Squared CMMD ({round(squared_result.item(), 4)}) is negative,"
                + " increase precision threshold or regularisation strength.",
                Warning,
                stacklevel=2,
            )
        result = jnp.sqrt(
            coreax.util.apply_negative_precision_threshold(
                squared_result,
                self.precision_threshold,
            )
        )
        return result
