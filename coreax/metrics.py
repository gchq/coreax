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

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import coreax.kernel as ck
import coreax.util as cu
from coreax.validation import (
    cast_as_type,
    validate_array_size,
    validate_in_range,
    validate_is_instance,
)


class Metric(ABC):
    """
    Base class for calculating metrics.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def compute(
        self,
        x: ArrayLike,
        y: ArrayLike,
        max_size: int = None,
        weights_x: ArrayLike = None,
        weights_y: ArrayLike = None,
    ) -> Array:
        r"""
        Compute the metric.

        Return a zero-dimensional array.

        :param x: An :math:`n \times d` array defining the full dataset
        :param y: An :math:`m \times d` array defining a representation of ``x``, for
            example a coreset
        :param max_size: Size of matrix block to process
        :param weights_x: An :math:`1 \times n` array of weights for associated points
            in ``x``
        :param weights_y: An :math:`1 \times m` array of weights for associated points
            in ``y``
        :return: Metric computed as a zero-dimensional array.
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

    def __init__(self, kernel: ck.Kernel, precision_threshold: float = 1e-8):
        r"""
        Calculate maximum mean discrepancy between two datasets in d dimensions.
        """
        # Validate inputs
        precision_threshold = cast_as_type(
            x=precision_threshold, object_name="precision_threshold", type_caster=float
        )
        validate_is_instance(x=kernel, object_name="kernel", expected_type=ck.Kernel)

        self.kernel = kernel
        self.precision_threshold = precision_threshold

        # initialise parent
        super().__init__()

    def compute(
        self,
        x: ArrayLike,
        y: ArrayLike,
        weights_x: ArrayLike = None,
        weights_y: ArrayLike = None,
        max_size: int = None,
    ) -> Array:
        r"""
        Calculate maximum mean discrepancy.

        If no weights (for the dataset ``y``) are given, standard MMD is calculated. If
        weights are given, weighted MMD is calculated. For both cases, if the size of
        matrix blocks to process is less than the size of both datasets, the calculation
        is done block-wise to limit memory requirements.

        :param x: The original :math:`n \times d` data
        :param y: :math:`m \times d` dataset
        :param max_size: (Optional) Size of matrix block to process, for memory limits
        :param weights_x: (Optional)  :math:`n \times 1` weights for original data ``x``
        :param weights_y: (Optional)  :math:`m \times 1` weights for dataset ``y``
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Validate inputs
        x = cast_as_type(x=x, object_name="x", type_caster=jnp.atleast_2d)
        y = cast_as_type(x=y, object_name="y", type_caster=jnp.atleast_2d)
        weights_x = cast_as_type(
            x=weights_x, object_name="weights_x", type_caster=jnp.atleast_2d
        )
        weights_y = cast_as_type(
            x=weights_y, object_name="weights_y", type_caster=jnp.atleast_2d
        )
        max_size = cast_as_type(x=max_size, object_name="max_size", type_caster=int)

        num_points_x = len(x)
        num_points_y = len(y)

        if weights_y is None:
            if max_size is None or max_size > max(num_points_x, num_points_y):
                return self.maximum_mean_discrepancy(x, y)
            else:
                return self.maximum_mean_discrepancy_block(x, y, max_size)

        else:
            if max_size is None or max_size > max(num_points_x, num_points_y):
                return self.weighted_maximum_mean_discrepancy(x, y, weights_y)
            else:
                return self.weighted_maximum_mean_discrepancy_block(
                    x, y, weights_x, weights_y, max_size
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
        # Validate inputs
        x = cast_as_type(x=x, object_name="x", type_caster=jnp.atleast_2d)
        y = cast_as_type(x=y, object_name="y", type_caster=jnp.atleast_2d)

        # Compute each term in the MMD formula
        kernel_nn = self.kernel.compute(x, x)
        kernel_mm = self.kernel.compute(y, y)
        kernel_nm = self.kernel.compute(x, y)

        # Compute MMD
        result = jnp.sqrt(
            cu.apply_negative_precision_threshold(
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
        # Ensure data is in desired format
        x = cast_as_type(x=x, object_name="x", type_caster=jnp.atleast_2d)
        y = cast_as_type(x=y, object_name="y", type_caster=jnp.atleast_2d)
        weights_y = cast_as_type(
            x=weights_y, object_name="weights_y", type_caster=jnp.atleast_2d
        )
        num_points_x = float(len(x))

        # Compute each term in the weighted MMD formula
        kernel_nn = self.kernel.compute(x, x)
        kernel_mm = self.kernel.compute(y, y)
        kernel_nm = self.kernel.compute(x, y)

        # Compute weighted MMD, correcting for any numerical precision issues, where we
        # would otherwise square-root a negative number very close to 0.0.
        result = jnp.sqrt(
            cu.apply_negative_precision_threshold(
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
        max_size: int = 10_000,
    ) -> Array:
        r"""
        Calculate maximum mean discrepancy (MMD) whilst limiting memory requirements.

        :param x: The original :math:`n \times d` data
        :param y: An :math:`m \times d` array defining a representation of ``x``, for
            example a coreset
        :param max_size: Size of matrix blocks to process
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Ensure data is in desired format
        x = cast_as_type(x=x, object_name="x", type_caster=jnp.atleast_2d)
        y = cast_as_type(x=y, object_name="y", type_caster=jnp.atleast_2d)
        max_size = cast_as_type(x=max_size, object_name="max_size", type_caster=int)
        num_points_x = float(len(x))
        num_points_y = float(len(y))

        # Compute each term in the weighted MMD formula
        kernel_nn = self.sum_pairwise_distances(x, x, max_size)
        kernel_mm = self.sum_pairwise_distances(y, y, max_size)
        kernel_nm = self.sum_pairwise_distances(x, y, max_size)

        # Compute MMD, correcting for any numerical precision issues, where we would
        # otherwise square-root a negative number very close to 0.0.
        result = jnp.sqrt(
            cu.apply_negative_precision_threshold(
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
        max_size: int = 10_000,
    ) -> Array:
        r"""
        Calculate weighted maximum mean discrepancy (MMD).

        This calculation is executed whilst limiting memory requirements.

        :param x: The original :math:`n \times d` data
        :param y: An :math:`m \times d` array defining a representation of ``x``, for
            example a coreset
        :param weights_x: :math:`n` weights of original data
        :param weights_y: :math:`m` weights of points in ``y``
        :param max_size: Size of matrix blocks to process
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Ensure data is in desired format
        x = cast_as_type(x=x, object_name="x", type_caster=jnp.atleast_2d)
        y = cast_as_type(x=y, object_name="y", type_caster=jnp.atleast_2d)
        weights_x = cast_as_type(
            x=weights_x, object_name="weights_x", type_caster=jnp.atleast_2d
        )
        weights_y = cast_as_type(
            x=weights_y, object_name="weights_y", type_caster=jnp.atleast_2d
        )
        max_size = cast_as_type(x=max_size, object_name="max_size", type_caster=int)
        num_points_x = weights_x.sum()
        num_points_y = weights_y.sum()

        # TODO: Needs changing to self.kernel.calculate_K_sum, when kernels support
        #  weighted inputs
        kernel_nn = self.sum_weighted_pairwise_distances(
            x, x, weights_x, weights_x, max_size
        )
        kernel_mm = self.sum_weighted_pairwise_distances(
            y, y, weights_y, weights_y, max_size
        )
        kernel_nm = self.sum_weighted_pairwise_distances(
            x, y, weights_x, weights_y, max_size
        )

        # Compute MMD, correcting for any numerical precision issues, where we would
        # otherwise square-root a negative number very close to 0.0.
        result = jnp.sqrt(
            cu.apply_negative_precision_threshold(
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
        max_size: int = 10_000,
    ) -> float:
        r"""
        Sum the kernel distance between all pairs of points in ``x`` and ``y``.

        The summation is done in blocks to avoid excessive memory usage.

        :param x: :math:`n \times 1` array
        :param y: :math:`m \times 1` array
        :param max_size: Size of matrix blocks to process
        :return: The sum of pairwise distances between points in ``x`` and ``y``
        """
        # Ensure data is in desired format
        x = cast_as_type(x=x, object_name="x", type_caster=jnp.atleast_2d)
        y = cast_as_type(x=y, object_name="y", type_caster=jnp.atleast_2d)
        max_size = cast_as_type(x=max_size, object_name="max_size", type_caster=int)
        num_points_x = len(x)
        num_points_y = len(y)

        # If max_size is larger than both inputs, we don't need to consider block-wise
        # computation
        if max_size > max(num_points_x, num_points_y):
            pairwise_distance_sum = self.kernel.compute(x, y).sum()

        else:
            pairwise_distance_sum = 0
            for i in range(0, num_points_x, max_size):
                for j in range(0, num_points_y, max_size):
                    pairwise_distances_part = self.kernel.compute(
                        x[i : i + max_size], y[j : j + max_size]
                    )
                    pairwise_distance_sum += pairwise_distances_part.sum()

        return pairwise_distance_sum

    def sum_weighted_pairwise_distances(
        self,
        x: ArrayLike,
        y: ArrayLike,
        weights_x: ArrayLike,
        weights_y: ArrayLike,
        max_size: int = 10_000,
    ) -> float:
        r"""
        Sum the weighted kernel distance between all pairs of points in ``x`` and ``y``.

        The summation is done in blocks to avoid excessive memory usage.

        :param x: :math:`n \times 1` array
        :param y: :math:`m \times 1` array
        :param weights_x: :math:`n \times 1` array of weights for ``x``
        :param weights_y: :math:`m \times 1` array of weights for ``y``
        :param max_size: Size of matrix blocks to process
        :return: The sum of pairwise distances between points in ``x`` and ``y``,
            with contributions weighted as defined by ``weights_x`` and ``weights_y``
        """
        # Ensure data is in desired format
        x = cast_as_type(x=x, object_name="x", type_caster=jnp.atleast_2d)
        y = cast_as_type(x=y, object_name="y", type_caster=jnp.atleast_2d)
        weights_x = cast_as_type(
            x=weights_x, object_name="weights_x", type_caster=jnp.atleast_2d
        )
        weights_y = cast_as_type(
            x=weights_y, object_name="weights_y", type_caster=jnp.atleast_2d
        )
        max_size = cast_as_type(x=max_size, object_name="max_size", type_caster=int)
        num_points_x = len(x)
        num_points_y = len(y)

        # If max_size is larger than both inputs, we don't need to consider block-wise
        # computation
        if max_size > max(num_points_x, num_points_y):
            kernel_weights = self.kernel.compute(x, y) * weights_y
            weighted_pairwise_distance_sum = (weights_x * kernel_weights.T).sum()

        else:
            weighted_pairwise_distance_sum = 0
            for i in range(0, num_points_x, max_size):
                for j in range(0, num_points_y, max_size):
                    pairwise_distances_part = (
                        weights_x[i : i + max_size, None]
                        * self.kernel.compute(x[i : i + max_size], y[j : j + max_size])
                        * weights_y[None, j : j + max_size]
                    )
                    weighted_pairwise_distance_sum += pairwise_distances_part.sum()

        return weighted_pairwise_distance_sum
