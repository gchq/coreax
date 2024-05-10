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

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import coreax.data
import coreax.kernel
import coreax.util

_Data = TypeVar("_Data", bound=coreax.data.Data)


class Metric(ABC, Generic[_Data]):
    """Base class for calculating metrics."""

    @abstractmethod
    def compute(
        self,
        reference_data: _Data,
        comparison_data: _Data,
    ) -> Array:
        r"""
        Compute the metric.

        Return a zero-dimensional array.

        :param reference_data: An instance of the class :class:`coreax.data.Data`,
            containing an :math:`n \times d` array of data. Or an instance of the class
            :class:`coreax.data.SupervisedData` containing an :math:`n \times d` array
            of data, and a corresponding :math:`n \times p` array of supervision.
        :param comparison_data: An instance of the class :class:`coreax.data.Data` to
            compare against ``reference_data`` containing an :math:`m \times d` array of
            data. Or an instance of the class :class:`coreax.data.SupervisedData` to
            compare against ``reference_data`` containing an :math:`m \times d` array of
            data, and a corresponding :math:`m \times p` array of supervision.
        :return: Metric computed as a zero-dimensional array
        """


class MMD(Metric, Generic[_Data]):
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
        reference_data: _Data,
        comparison_data: _Data,
        block_size: int | None = None,
    ) -> Array:
        r"""
        Calculate maximum mean discrepancy.

        If uniform weights are given for dataset ``reference_data``, standard MMD is
        calculated. If non-uniform weights are given, weighted MMD is calculated. For
        both cases, if the size of matrix blocks to process is less than the size of
        both datasets and reference weights are not uniform in the weighted case, the
        calculation is done block-wise to limit memory requirements.

        :param reference_data: An instance of the class :class:`coreax.data.Data`,
            containing an :math:`n \times d` array of data
        :param comparison_data: An instance of the class :class:`coreax.data.Data` to
            compare against ``reference_data`` containing an :math:`m \times d` array of
            data
        :param block_size: Size of matrix block to process, or :data:`None` to not split
            into blocks
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Format inputs
        reference_points = reference_data.data
        num_reference_points = len(reference_points)
        reference_weights = jnp.atleast_1d(reference_data.weights)
        if jnp.allclose(
            reference_weights,
            jnp.broadcast_to(1 / num_reference_points, (num_reference_points,)),
        ):
            reference_weights = None

        comparison_points = comparison_data.data
        num_comparison_points = len(comparison_points)
        comparison_weights = jnp.atleast_1d(comparison_data.weights)
        if jnp.allclose(
            comparison_weights,
            jnp.broadcast_to(1 / num_comparison_points, (num_comparison_points,)),
        ):
            comparison_weights = None

        if comparison_weights is None and reference_weights is None:
            if block_size is None or block_size > max(
                num_reference_points, num_comparison_points
            ):
                return self.maximum_mean_discrepancy(
                    reference_points, comparison_points
                )
            return self.maximum_mean_discrepancy_block(
                reference_points, comparison_points, block_size
            )

        if block_size is None or block_size > max(
            num_reference_points, num_comparison_points
        ):
            return self.weighted_maximum_mean_discrepancy(
                reference_points,
                comparison_points,
                reference_weights,
                comparison_weights,
            )

        return self.weighted_maximum_mean_discrepancy_block(
            reference_points,
            comparison_points,
            reference_weights,
            comparison_weights,
            block_size,
        )

    def maximum_mean_discrepancy(
        self, reference_points: ArrayLike, comparison_points: ArrayLike
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

        :param reference_points: The original :math:`n \times d` data
        :param comparison_points: An :math:`m \times d` array to compare to
            ``reference_points``
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Format inputs
        reference_points = jnp.atleast_2d(reference_points)
        comparison_points = jnp.atleast_2d(comparison_points)

        # Compute each term in the MMD formula
        kernel_nn = self.kernel.compute(reference_points, reference_points)
        kernel_mm = self.kernel.compute(comparison_points, comparison_points)
        kernel_nm = self.kernel.compute(reference_points, comparison_points)

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
        reference_points: ArrayLike,
        comparison_points: ArrayLike,
        reference_weights: ArrayLike | None = None,
        comparison_weights: ArrayLike | None = None,
    ) -> Array:
        r"""
        Calculate weighted maximum mean discrepancy (WMMD).

        :param reference_points: :math:`n \times d` array of reference data
        :param comparison_points: An :math:`m \times d` array to compare to
            ``reference_points``
        :param reference_weights: :math:`m \times 1` weights vector for data
            ``reference_points``, or :data:`None` (default) if unweighted
        :param comparison_weights: :math:`m \times 1` weights vector for data
            ``comparison_points``, or :data:`None` (default) if unweighted
        :return: Weighted maximum mean discrepancy as a 0-dimensional array
        """
        # Format inputs
        reference_points = jnp.atleast_2d(reference_points)
        num_reference_points = len(reference_points)
        if reference_weights is None:
            reference_weights = jnp.broadcast_to(
                1 / num_reference_points, (num_reference_points,)
            )
        else:
            reference_weights = jnp.atleast_1d(reference_weights)

        comparison_points = jnp.atleast_2d(comparison_points)
        num_comparison_points = len(comparison_points)
        if comparison_weights is None:
            comparison_weights = jnp.broadcast_to(
                1 / num_comparison_points, (num_comparison_points,)
            )
        else:
            comparison_weights = jnp.atleast_1d(comparison_weights)

        # Compute each term in the weighted MMD formula
        kernel_nn = self.kernel.compute(reference_points, reference_points)
        kernel_mm = self.kernel.compute(comparison_points, comparison_points)
        kernel_nm = self.kernel.compute(reference_points, comparison_points)

        # Compute weighted MMD, correcting for any numerical precision issues, where we
        # would otherwise square-root a negative number very close to 0.0.
        result = jnp.sqrt(
            coreax.util.apply_negative_precision_threshold(
                jnp.dot(reference_weights.T, jnp.dot(kernel_nn, reference_weights))
                + jnp.dot(comparison_weights.T, jnp.dot(kernel_mm, comparison_weights))
                - 2
                * jnp.dot(jnp.dot(reference_weights.T, kernel_nm), comparison_weights),
                self.precision_threshold,
            )
        )
        return result

    def maximum_mean_discrepancy_block(
        self,
        reference_points: ArrayLike,
        comparison_points: ArrayLike,
        block_size: int = 10_000,
    ) -> Array:
        r"""
        Calculate maximum mean discrepancy (MMD) whilst limiting memory requirements.

        :param reference_points: :math:`n \times d` array of reference data
        :param comparison_points: An :math:`m \times d` array to compare to
            ``reference_points``
        :param block_size: Size of matrix blocks to process
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Format inputs
        reference_points = jnp.atleast_2d(reference_points)
        comparison_points = jnp.atleast_2d(comparison_points)

        num_reference_points = float(len(reference_points))
        num_comparison_points = float(len(comparison_points))

        # Compute each term in the weighted MMD formula
        kernel_nn = self.sum_pairwise_distances(
            reference_points, reference_points, block_size
        )
        kernel_mm = self.sum_pairwise_distances(
            comparison_points, comparison_points, block_size
        )
        kernel_nm = self.sum_pairwise_distances(
            reference_points, comparison_points, block_size
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
        reference_points: ArrayLike,
        comparison_points: ArrayLike,
        reference_weights: ArrayLike,
        comparison_weights: ArrayLike,
        block_size: int = 10_000,
    ) -> Array:
        r"""
        Calculate weighted maximum mean discrepancy (MMD).

        This calculation is executed whilst limiting memory requirements.

        :param reference_points: :math:`n \times d` array of reference data
        :param comparison_points: An :math:`m \times d` array to compare to
            ``reference_points```
        :param reference_weights: :math:`n` weights of reference data
        :param comparison_weights: :math:`m` weights of points in ``comparison_points``
        :param block_size: Size of matrix blocks to process
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Format inputs
        reference_points = jnp.atleast_2d(reference_points)
        comparison_points = jnp.atleast_2d(comparison_points)
        reference_weights = jnp.atleast_1d(reference_weights)
        comparison_weights = jnp.atleast_1d(comparison_weights)

        num_reference_points = reference_weights.sum()
        num_comparison_points = comparison_weights.sum()

        kernel_nn = self.sum_weighted_pairwise_distances(
            reference_points,
            reference_points,
            reference_weights,
            reference_weights,
            block_size,
        )
        kernel_mm = self.sum_weighted_pairwise_distances(
            comparison_points,
            comparison_points,
            comparison_weights,
            comparison_weights,
            block_size,
        )
        kernel_nm = self.sum_weighted_pairwise_distances(
            reference_points,
            comparison_points,
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
        reference_points: ArrayLike,
        comparison_points: ArrayLike,
        block_size: int = 10_000,
    ) -> float:
        r"""
        Sum the kernel distance between all pairs of points in the two input datasets.

        The summation is done in blocks to avoid excessive memory usage.

        :param reference_points: :math:`n \times 1` array
        :param comparison_points: :math:`m \times 1` array
        :param block_size: Size of matrix blocks to process
        :return: The sum of pairwise distances between points in ``reference_points``
            and ``comparison_points``
        """
        # Format inputs
        reference_points = jnp.atleast_2d(reference_points)
        comparison_points = jnp.atleast_2d(comparison_points)
        block_size = max(0, block_size)

        num_reference_points = len(reference_points)
        num_comparison_points = len(comparison_points)

        # If block_size is larger than both inputs, we don't need to consider block-wise
        # computation
        if block_size > max(num_reference_points, num_comparison_points):
            pairwise_distance_sum = self.kernel.compute(
                reference_points, comparison_points
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
                        reference_points[i : i + block_size],
                        comparison_points[j : j + block_size],
                    )
                    pairwise_distance_sum += pairwise_distances_part.sum()

        return pairwise_distance_sum

    def sum_weighted_pairwise_distances(
        self,
        reference_points: ArrayLike,
        comparison_points: ArrayLike,
        reference_weights: ArrayLike,
        comparison_weights: ArrayLike,
        block_size: int = 10_000,
    ) -> float:
        r"""
        Sum weighted kernel distance between all pairs of points in the two datasets.

        The summation is done in blocks to avoid excessive memory usage.

        :param reference_points: :math:`n \times 1` array
        :param comparison_points: :math:`m \times 1` array
        :param reference_weights: :math:`n \times 1` array of weights for
            ``reference_points``
        :param comparison_weights: :math:`m \times 1` array of weights for
            ``comparison_points``
        :param block_size: Size of matrix blocks to process
        :return: The sum of pairwise distances between points in ``reference_points``
            and ``comparison_points``, with contributions weighted as defined by
            ``reference_weights`` and ``comparison_weights``
        """
        # Format inputs
        reference_points = jnp.atleast_2d(reference_points)
        comparison_points = jnp.atleast_2d(comparison_points)
        reference_weights = jnp.atleast_1d(reference_weights)
        comparison_weights = jnp.atleast_1d(comparison_weights)
        block_size = max(0, block_size)

        num_reference_points = len(reference_points)
        num_comparison_points = len(comparison_points)

        # If block_size is larger than both inputs, we don't need to consider block-wise
        # computation
        if block_size > max(num_reference_points, num_comparison_points):
            kernel_weights = (
                self.kernel.compute(reference_points, comparison_points)
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
                            reference_points[i : i + block_size],
                            comparison_points[j : j + block_size],
                        )
                        * comparison_weights[None, j : j + block_size]
                    )
                    weighted_pairwise_distance_sum += pairwise_distances_part.sum()

        return weighted_pairwise_distance_sum
