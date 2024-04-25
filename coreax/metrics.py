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
from warnings import warn

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import coreax.kernel
import coreax.util


class Metric(ABC):
    """Base class for calculating metrics."""

    @abstractmethod
    def compute(
        self,
        x: ArrayLike,
        y: ArrayLike,
        block_size: int | None = None,
        weights_x: ArrayLike | None = None,
        weights_y: ArrayLike | None = None,
    ) -> Array:
        r"""
        Compute the metric.

        Return a zero-dimensional array.

        :param x: An :math:`n \times d` array defining the full dataset
        :param y: An :math:`m \times d` array defining a representation of ``x``, for
            example a coreset
        :param block_size: Size of matrix block to process, or :data:`None` to not split
            into blocks
        :param weights_x: An :math:`1 \times n` array of weights for associated points
            in ``x``, or :data:`None` if not required
        :param weights_y: An :math:`1 \times m` array of weights for associated points
            in ``y``, or :data:`None` if not required
        :return: Metric computed as a zero-dimensional array
        """


class MMD(Metric):
    r"""
    Definition and calculation of the maximum mean discrepancy metric.

    For a dataset of ``n`` points in ``d`` dimensions, :math:`x`, and another dataset
    :math:`y` of ``m`` points in ``d`` dimensions, the maximum mean discrepancy is given
    by:

    .. math::

        \text{MMD}^2(x,y) = \mathbb{E}(k(x,x)) + \mathbb{E}(k(y,y))
        - 2\mathbb{E}(k(x,y))

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
        x: ArrayLike,
        y: ArrayLike,
        block_size: int | None = None,
        weights_x: ArrayLike | None = None,
        weights_y: ArrayLike | None = None,
    ) -> Array:
        r"""
        Calculate maximum mean discrepancy.

        If no weights are given for dataset ``y``, standard MMD is calculated. If
        weights are given, weighted MMD is calculated. For both cases, if the size of
        matrix blocks to process is less than the size of both datasets and if
        ``weights_x`` is given in the weighted case, the calculation
        is done block-wise to limit memory requirements.

        :param x: The original :math:`n \times d` data
        :param y: :math:`m \times d` dataset
        :param block_size: Size of matrix block to process, or :data:`None` to not split
            into blocks
        :param weights_x: An :math:`1 \times n` array of weights for associated points
            in ``x``, or :data:`None` if not required
        :param weights_y: An :math:`1 \times m` array of weights for associated points
            in ``y``, or :data:`None` if not required
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Format inputs
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        if weights_x is not None:
            weights_x = jnp.atleast_1d(weights_x)
        if weights_y is not None:
            weights_y = jnp.atleast_1d(weights_y)

        num_points_x = len(x)
        num_points_y = len(y)

        if weights_y is None:
            if block_size is None or block_size > max(num_points_x, num_points_y):
                return self.maximum_mean_discrepancy(x, y)
            return self.maximum_mean_discrepancy_block(x, y, block_size)

        if (
            block_size is None
            or weights_x is None
            or block_size > max(num_points_x, num_points_y)
        ):
            return self.weighted_maximum_mean_discrepancy(x, y, weights_y)

        return self.weighted_maximum_mean_discrepancy_block(
            x, y, weights_x, weights_y, block_size
        )

    def maximum_mean_discrepancy(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Calculate standard, unweighted MMD.

        For a dataset of ``n`` points in ``d`` dimensions, :math:`x`, and another
        dataset :math:`y` of ``m`` points in ``d`` dimensions, the maximum mean
        discrepancy is given by:

        .. math::

            \text{MMD}^2(x,y) = \mathbb{E}(k(x,x)) + \mathbb{E}(k(y,y))
            - 2\mathbb{E}(k(x,y))

        where :math:`k` is the selected kernel.

        :param x: The original :math:`n \times d` data
        :param y: An :math:`m \times d` array defining a representation of ``x``, for
            example a coreset
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Format inputs
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)

        # Compute each term in the MMD formula
        kernel_nn = self.kernel.compute(x, x)
        kernel_mm = self.kernel.compute(y, y)
        kernel_nm = self.kernel.compute(x, y)

        # Compute MMD
        result = jnp.sqrt(
            coreax.util.apply_negative_precision_threshold(
                kernel_nn.mean() + kernel_mm.mean() - 2 * kernel_nm.mean(),
                self.precision_threshold,
            )
        )
        return result

    def weighted_maximum_mean_discrepancy(
        self, x: ArrayLike, y: ArrayLike, weights_y: ArrayLike
    ) -> Array:
        r"""
        Calculate one-sided, weighted maximum mean discrepancy (WMMD).

        Only data points in ``y`` are weighted.

        :param x: The original :math:`n \times d` data
        :param y: An :math:`m \times d` array defining a representation of ``x``, for
            example a coreset
        :param weights_y: :math:`m \times 1` weights vector for data ``y``
        :return: Weighted maximum mean discrepancy as a 0-dimensional array
        """
        # Format inputs
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        weights_y = jnp.atleast_1d(weights_y)

        num_points_x = float(len(x))

        # Compute each term in the weighted MMD formula
        kernel_nn = self.kernel.compute(x, x)
        kernel_mm = self.kernel.compute(y, y)
        kernel_nm = self.kernel.compute(x, y)

        # Compute weighted MMD, correcting for any numerical precision issues, where we
        # would otherwise square-root a negative number very close to 0.0.
        result = jnp.sqrt(
            coreax.util.apply_negative_precision_threshold(
                jnp.dot(weights_y.T, jnp.dot(kernel_mm, weights_y))
                + kernel_nn.sum() / num_points_x**2
                - 2 * jnp.dot(weights_y.T, kernel_nm.mean(axis=0)),
                self.precision_threshold,
            )
        )
        return result

    def maximum_mean_discrepancy_block(
        self,
        x: ArrayLike,
        y: ArrayLike,
        block_size: int = 10_000,
    ) -> Array:
        r"""
        Calculate maximum mean discrepancy (MMD) whilst limiting memory requirements.

        :param x: The original :math:`n \times d` data
        :param y: An :math:`m \times d` array defining a representation of ``x``, for
            example a coreset
        :param block_size: Size of matrix blocks to process
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Format inputs
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)

        num_points_x = float(len(x))
        num_points_y = float(len(y))

        # Compute each term in the weighted MMD formula
        kernel_nn = self.sum_pairwise_distances(x, x, block_size)
        kernel_mm = self.sum_pairwise_distances(y, y, block_size)
        kernel_nm = self.sum_pairwise_distances(x, y, block_size)

        # Compute MMD, correcting for any numerical precision issues, where we would
        # otherwise square-root a negative number very close to 0.0.
        result = jnp.sqrt(
            coreax.util.apply_negative_precision_threshold(
                kernel_nn / num_points_x**2
                + kernel_mm / num_points_y**2
                - 2 * kernel_nm / (num_points_x * num_points_y),
                self.precision_threshold,
            )
        )
        return result

    def weighted_maximum_mean_discrepancy_block(
        self,
        x: ArrayLike,
        y: ArrayLike,
        weights_x: ArrayLike,
        weights_y: ArrayLike,
        block_size: int = 10_000,
    ) -> Array:
        r"""
        Calculate weighted maximum mean discrepancy (MMD).

        This calculation is executed whilst limiting memory requirements.

        :param x: The original :math:`n \times d` data
        :param y: An :math:`m \times d` array defining a representation of ``x``, for
            example a coreset
        :param weights_x: :math:`n` weights of original data
        :param weights_y: :math:`m` weights of points in ``y``
        :param block_size: Size of matrix blocks to process
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Format inputs
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        weights_x = jnp.atleast_1d(weights_x)
        weights_y = jnp.atleast_1d(weights_y)

        num_points_x = weights_x.sum()
        num_points_y = weights_y.sum()

        kernel_nn = self.sum_weighted_pairwise_distances(
            x, x, weights_x, weights_x, block_size
        )
        kernel_mm = self.sum_weighted_pairwise_distances(
            y, y, weights_y, weights_y, block_size
        )
        kernel_nm = self.sum_weighted_pairwise_distances(
            x, y, weights_x, weights_y, block_size
        )

        # Compute MMD, correcting for any numerical precision issues, where we would
        # otherwise square-root a negative number very close to 0.0.
        result = jnp.sqrt(
            coreax.util.apply_negative_precision_threshold(
                kernel_nn / num_points_x**2
                + kernel_mm / num_points_y**2
                - 2 * kernel_nm / (num_points_x * num_points_y),
                self.precision_threshold,
            )
        )
        return result

    def sum_pairwise_distances(
        self,
        x: ArrayLike,
        y: ArrayLike,
        block_size: int = 10_000,
    ) -> float:
        r"""
        Sum the kernel distance between all pairs of points in ``x`` and ``y``.

        The summation is done in blocks to avoid excessive memory usage.

        :param x: :math:`n \times 1` array
        :param y: :math:`m \times 1` array
        :param block_size: Size of matrix blocks to process
        :return: The sum of pairwise distances between points in ``x`` and ``y``
        """
        # Format inputs
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        block_size = max(0, block_size)

        num_points_x = len(x)
        num_points_y = len(y)

        # If block_size is larger than both inputs, we don't need to consider block-wise
        # computation
        if block_size > max(num_points_x, num_points_y):
            pairwise_distance_sum = self.kernel.compute(x, y).sum()

        else:
            try:
                row_index_range = range(0, num_points_x, block_size)
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
                for j in range(0, num_points_y, block_size):
                    pairwise_distances_part = self.kernel.compute(
                        x[i : i + block_size], y[j : j + block_size]
                    )
                    pairwise_distance_sum += pairwise_distances_part.sum()

        return pairwise_distance_sum

    def sum_weighted_pairwise_distances(
        self,
        x: ArrayLike,
        y: ArrayLike,
        weights_x: ArrayLike,
        weights_y: ArrayLike,
        block_size: int = 10_000,
    ) -> float:
        r"""
        Sum the weighted kernel distance between all pairs of points in ``x`` and ``y``.

        The summation is done in blocks to avoid excessive memory usage.

        :param x: :math:`n \times 1` array
        :param y: :math:`m \times 1` array
        :param weights_x: :math:`n \times 1` array of weights for ``x``
        :param weights_y: :math:`m \times 1` array of weights for ``y``
        :param block_size: Size of matrix blocks to process
        :return: The sum of pairwise distances between points in ``x`` and ``y``,
            with contributions weighted as defined by ``weights_x`` and ``weights_y``
        """
        # Format inputs
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        weights_x = jnp.atleast_1d(weights_x)
        weights_y = jnp.atleast_1d(weights_y)
        block_size = max(0, block_size)

        num_points_x = len(x)
        num_points_y = len(y)

        # If block_size is larger than both inputs, we don't need to consider block-wise
        # computation
        if block_size > max(num_points_x, num_points_y):
            kernel_weights = self.kernel.compute(x, y) * weights_y
            weighted_pairwise_distance_sum = (weights_x * kernel_weights.T).sum()

        else:
            weighted_pairwise_distance_sum = 0

            try:
                row_index_range = range(0, num_points_x, block_size)
            except ValueError as exception:
                if block_size == 0:
                    raise ValueError(
                        "block_size must be a positive integer"
                    ) from exception
                raise
            except TypeError as exception:
                raise TypeError("block_size must be a positive integer") from exception

            for i in row_index_range:
                for j in range(0, num_points_y, block_size):
                    pairwise_distances_part = (
                        weights_x[i : i + block_size, None]
                        * self.kernel.compute(
                            x[i : i + block_size], y[j : j + block_size]
                        )
                        * weights_y[None, j : j + block_size]
                    )
                    weighted_pairwise_distance_sum += pairwise_distances_part.sum()

        return weighted_pairwise_distance_sum


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
        parameters corresponding to the original dataset :math:`\mathcal{D}^{(1)}` and
        the coreset :math:`\mathcal{D}^{(2)}` respectively
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

    # Disable pylint warning for arguments-renamed as the use of x and y as arguments
    # for different datasets is bound to create misunderstandings when we are
    # considering supervised data.
    # pylint: disable=arguments-renamed
    def compute(
        self,
        dataset_1: ArrayLike,
        dataset_2: ArrayLike,
        block_size: None = None,
        weights_x: None = None,
        weights_y: None = None,
    ) -> Array:
        r"""
        Calculate conditional maximum mean discrepancy.

        :param dataset_1: The original dataset :math:`\mathcal{D}^{(1)} =
            \{(x_i, y_i)\}_{i=1}^n` of ``n`` pairs with :math:`x\in\mathbb{R}^d` and
            :math:`y\in\mathbb{R}^p`, responses should be concatenated after the
            features
        :param dataset_2: Dataset :math:`\mathcal{D}^{(2)} = \{(\tilde{x}_i,
            \tilde{y}_i)\}_{i=1}^n` of ``m`` pairs with :math:`\tilde{x}\in\mathbb{R}^d`
            and :math:`\tilde{y}\in\mathbb{R}^p`, responses should be
            concatenated after the features
        :param block_size: :data:`None`, included for compatibility reasons
        :param weights_x: :data:`None`, included for compatibility reasons
        :param weights_y: :data:`None`, included for compatibility reasons
        :return: Conditional maximum mean discrepancy as a 0-dimensional array
        """
        # Make sure that compatibility params are None
        assert block_size is None, "CMMD computation does not support blocking"
        assert weights_x is None, "CMMD computation does not support weights"
        assert weights_y is None, "CMMD computation does not support weights"

        return self.conditional_maximum_mean_discrepancy(dataset_1, dataset_2)

    # pylint: enable=arguments-renamed
    # Clarity means I need to assign more local variables
    # pylint: disable=too-many-locals
    def conditional_maximum_mean_discrepancy(
        self, dataset_1: ArrayLike, dataset_2: ArrayLike
    ) -> Array:
        r"""
        Calculate standard conditional maximum mean discrepancy metric.

        For a dataset :math:`\mathcal{D}^{(1)} = \{(x_i, y_i)\}_{i=1}^n` of ``n`` pairs
        with :math:`x\in\mathbb{R}^d` and :math:`y\in\mathbb{R}^p`, and another dataset
        :math:`\mathcal{D}^{(2)} = \{(\tilde{x}_i, \tilde{y}_i)\}_{i=1}^n` of ``n``
        pairs with :math:`\tilde{x}\in\mathbb{R}^d` and :math:`\tilde{y}\in
        \mathbb{R}^p`, the conditional maximum mean discrepancy is given by:

        .. math::

            \text{CMMD}^2(\mathcal{D}^{(1)}, \mathcal{D}^{(2)}) = ||\hat{\mu}^{(1)} -
            \hat{\mu}^{(2)}||^2_{\mathcal{H}_k \otimes \mathcal{H}_l}

        where :math:`\hat{\mu}^{(1)},\hat{\mu}^{(2)}` are the conditional mean
        embeddings estimated with :math:`\mathcal{D}^{(1)}` and
        :math:`\mathcal{D}^{(2)}` respectively, and :math:`\mathcal{H}_k,\mathcal{H}_l`
        are the RKHSs corresponding to the kernel functions
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` and
        :math:`l: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}` respectively.

        :param dataset_1: The original dataset :math:`\mathcal{D}^{(1)} =
            \{(x_i, y_i)\}_{i=1}^n` of ``n`` pairs with :math:`x\in\mathbb{R}^d` and
            :math:`y\in\mathbb{R}^p`, responses should be concatenated after the
            features
        :param dataset_2: Dataset :math:`\mathcal{D}^{(2)} = \{(\tilde{x}_i,
            \tilde{y}_i)\}_{i=1}^n` of ``m`` pairs with :math:`\tilde{x}\in\mathbb{R}^d`
            and :math:`\tilde{y}\in\mathbb{R}^p`, responses should be
            concatenated after the features
        """
        # Extract and format features and responses from D1 and D2
        x1 = jnp.atleast_2d(dataset_1[:, : self.num_feature_dimensions])
        y1 = jnp.atleast_2d(dataset_1[:, self.num_feature_dimensions :])
        x2 = jnp.atleast_2d(dataset_2[:, : self.num_feature_dimensions])
        y2 = jnp.atleast_2d(dataset_2[:, self.num_feature_dimensions :])

        # Compute feature kernel gramians
        feature_gramian_1 = self.feature_kernel.compute(x1, x1)
        feature_gramian_2 = self.feature_kernel.compute(x2, x2)
        cross_feature_gramian = self.feature_kernel.compute(x2, x1)

        # Compute response kernel gramians
        response_gramian_1 = self.response_kernel.compute(y1, y1)
        response_gramian_2 = self.response_kernel.compute(y2, y2)
        cross_response_gramian = self.response_kernel.compute(y1, y2)

        # Invert feature kernel gramians
        identity_1 = jnp.eye(feature_gramian_1.shape[0])
        inverse_feature_gramian_1 = coreax.util.invert_regularised_array(
            array=feature_gramian_1,
            regularisation_parameter=self.regularisation_parameters[0],
            identity=identity_1,
        )

        identity_2 = jnp.eye(feature_gramian_2.shape[0])
        inverse_feature_gramian_2 = coreax.util.invert_regularised_array(
            array=feature_gramian_2,
            regularisation_parameter=self.regularisation_parameters[1],
            identity=identity_2,
        )

        # Compute each term in the CMMD
        term_1 = (
            inverse_feature_gramian_1
            @ response_gramian_1
            @ inverse_feature_gramian_1
            @ feature_gramian_1
        )
        term_2 = (
            inverse_feature_gramian_2
            @ response_gramian_2
            @ inverse_feature_gramian_2
            @ feature_gramian_2
        )
        term_3 = (
            inverse_feature_gramian_1
            @ cross_response_gramian
            @ inverse_feature_gramian_2
            @ cross_feature_gramian
        )

        # Compute CMMD
        squared_result = jnp.trace(term_1) + jnp.trace(term_2) - 2 * jnp.trace(term_3)
        if squared_result < 0:
            warn(
                f"Squared CMMD ({round(squared_result.item(), 4)}) is negative,"
                + " increase precision threshold or regularisation strength.",
                Warning,
                stacklevel=1,
            )
        result = jnp.sqrt(
            coreax.util.apply_negative_precision_threshold(
                squared_result,
                self.precision_threshold,
            )
        )
        return result


# pylint: enable=too-many-locals
