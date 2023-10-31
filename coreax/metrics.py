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

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import coreax.kernel as ck
import coreax.util as cu


class Metric(ABC):
    r"""
    Base class for calculating metrics.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def compute(
        self,
        x: ArrayLike,
        x_c: ArrayLike,
        max_size: int = None,
        weights_x: ArrayLike = None,
        weights_x_c: ArrayLike = None,
    ) -> Array:
        r"""
        Compute the metric.

        Return a zero-dimensional array.

        :param x: An :math:`n \times d` array defining the full dataset
        :param x_c: An :math:`n \times d` array defining a representation of x
        :param max_size: Size of matrix block to process
        :param weights_x: An :math:`1 \times n` array of weights for associated points
            in x
        :param weights_x_c: An :math:`1 \times n` array of weights for associated points
            in x_c
        :return: Metric computed as a zero-dimensional array.
        """


class MMD(Metric):
    r"""
    Calculation for maximum mean discrepancy.
    """

    def __init__(self, kernel: ck.Kernel, precision_threshold: float = 1e-8):
        r"""
        Calculate maximum mean discrepancy between two datasets in d dimensions.

        :param kernel: Kernel object with compute method defined mapping
            :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
        :param precision_threshold: Positive threshold we compare against for precision
        """
        self.kernel = kernel
        self.precision_threshold = precision_threshold

        # initialise parent
        super().__init__()

    def compute(
        self,
        x: ArrayLike,
        x_c: ArrayLike,
        weights_x: ArrayLike = None,
        weights_x_c: ArrayLike = None,
        max_size: int = None,
    ) -> Array:
        r"""
        Calculate maximum mean discrepancy.

        If no weights (for the coreset) are given, standard MMD is calculated.
        If weights are given, weighted MMD is calculated.
        For both cases, if the size of matrix blocks to process is less than the size of
        the coreset (which is, by construction, less than or equal to the size of the
        original data), the calculation is done block-wise to limit memory requirements.

        :param x: The original :math:`n \times d` data
        :param x_c: :math:`m \times d` coreset
        :param max_size: (Optional) Size of matrix block to process, for memory limits
        :param weights_x: (Optional)  :math:`n \times 1` weights for original data x
        :param weights_x_c: (Optional)  :math:`m \times 1` weights for coreset x_c
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Compute number of points in each input dataset
        num_points_x = len(jnp.asarray(x))
        num_points_x_c = len(jnp.asarray(x_c))

        # Choose which MMD computation method to call depending on max_size and weights
        if weights_x_c is None:
            if max_size is None or max_size > max(num_points_x, num_points_x_c):
                return self.maximum_mean_discrepancy(x, x_c)
            else:
                return self.maximum_mean_discrepancy_block(x, x_c, max_size)

        else:
            if max_size is None or max_size > max(num_points_x, num_points_x_c):
                return self.weighted_maximum_mean_discrepancy(x, x_c, weights_x_c)
            else:
                return self.weighted_maximum_mean_discrepancy_block(
                    x, x_c, weights_x, weights_x_c, max_size
                )

    def maximum_mean_discrepancy(self, x: ArrayLike, x_c: ArrayLike) -> Array:
        r"""
        Calculate standard MMD.

        For dataset of n points in d dimensions, :math:`X`, and a coreset :math:`Y` of m
        points in d dimensions, the maximum mean discrepancy is given by:

        .. math::

            \text{MMD}^2(X,X_c) = \mathbb{E}(k(X,X)) + \mathbb{E}(k(X_c,X_c))
            - 2\mathbb{E}(k(X,X_c))

        :param x: The original :math:`n \times d` data
        :param x_c: :math:`m \times d` coreset
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Compute each term in the MMD formula
        kernel_nn = self.kernel.compute(x, x)
        kernel_mm = self.kernel.compute(x_c, x_c)
        kernel_nm = self.kernel.compute(x, x_c)

        # Compute MMD
        result = jnp.sqrt(
            cu.apply_negative_precision_threshold(
                kernel_nn.mean() + kernel_mm.mean() - 2 * kernel_nm.mean(),
                self.precision_threshold,
            )
        )
        return result

    def weighted_maximum_mean_discrepancy(
        self, x: ArrayLike, x_c: ArrayLike, weights_x_c: ArrayLike
    ) -> Array:
        r"""
        Calculate one-sided, weighted maximum mean discrepancy (WMMD).

        Only coreset points are weighted.

        :param x: The original :math:`n \times d` data
        :param x_c: :math:`m \times d` coreset
        :param weights_x_c: :math:`m \times 1` weights vector for coreset
        :return: Weighted maximum mean discrepancy as a 0-dimensional array
        """
        # Ensure data is in desired format
        x = jnp.asarray(x)
        n = float(len(x))

        # Compute each term in the weighted MMD formula
        kernel_nn = self.kernel.compute(x, x)
        kernel_mm = self.kernel.compute(x_c, x_c)
        kernel_nm = self.kernel.compute(x, x_c)

        # Compute weighted MMD, correcting for any numerical precision issues, where we
        # would otherwise square-root a negative number very close to 0.0.
        result = jnp.sqrt(
            cu.apply_negative_precision_threshold(
                jnp.dot(weights_x_c.T, jnp.dot(kernel_mm, weights_x_c))
                + kernel_nn.sum() / n**2
                - 2 * jnp.dot(weights_x_c.T, kernel_nm.mean(axis=0)),
                self.precision_threshold,
            )
        )
        return result

    def maximum_mean_discrepancy_block(
        self,
        x: ArrayLike,
        x_c: ArrayLike,
        max_size: int = 10_000,
    ) -> Array:
        r"""
        Calculate maximum mean discrepancy (MMD) whilst limiting memory requirements.

        :param x: The original :math:`n \times d` data
        :param x_c: :math:`m \times d` coreset
        :param max_size: Size of matrix blocks to process
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Ensure data is in desired format
        x = jnp.asarray(x)
        x_c = jnp.asarray(x_c)
        num_points_x = float(len(x))
        num_points_x_c = float(len(x_c))

        # Compute each term in the weighted MMD formula
        kernel_nn = self.sum_pairwise_distances(x, x, max_size)
        kernel_mm = self.sum_pairwise_distances(x_c, x_c, max_size)
        kernel_nm = self.sum_pairwise_distances(x, x_c, max_size)

        # Compute MMD, correcting for any numerical precision issues, where we would
        # otherwise square-root a negative number very close to 0.0.
        result = jnp.sqrt(
            cu.apply_negative_precision_threshold(
                kernel_nn / num_points_x**2
                + kernel_mm / num_points_x_c**2
                - 2 * kernel_nm / (num_points_x * num_points_x_c),
                self.precision_threshold,
            )
        )
        return result

    def weighted_maximum_mean_discrepancy_block(
        self,
        x: ArrayLike,
        x_c: ArrayLike,
        weights_x: ArrayLike,
        weights_x_c: ArrayLike,
        max_size: int = 10_000,
    ) -> Array:
        r"""
        Calculate weighted maximum mean discrepancy (MMD).

        This calculation is executed whilst limiting memory requirements.

        :param x: The original :math:`n \times d` data
        :param x_c: :math:`m \times d` coreset
        :param weights_x: :math:`n` weights of original data
        :param weights_x_c: :math:`m` weights of coreset points
        :param max_size: Size of matrix blocks to process
        :return: Maximum mean discrepancy as a 0-dimensional array
        """
        # Ensure data is in desired format
        weights_x = jnp.asarray(weights_x)
        weights_x_c = jnp.asarray(weights_x_c)
        num_points_x = weights_x.sum()
        num_points_x_c = weights_x_c.sum()

        # needs changing to self.kernel.calculate_K_sum:
        kernel_nn = self.sum_weighted_pairwise_distances(
            x, x, weights_x, weights_x, max_size
        )
        kernel_mm = self.sum_weighted_pairwise_distances(
            x_c, x_c, weights_x_c, weights_x_c, max_size
        )
        kernel_nm = self.sum_weighted_pairwise_distances(
            x, x_c, weights_x, weights_x_c, max_size
        )

        # Compute MMD, correcting for any numerical precision issues, where we would
        # otherwise square-root a negative number very close to 0.0.
        result = jnp.sqrt(
            cu.apply_negative_precision_threshold(
                kernel_nn / num_points_x**2
                + kernel_mm / num_points_x_c**2
                - 2 * kernel_nm / (num_points_x * num_points_x_c),
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
        Sum the kernel distance between all pairs of points in x and y.

        The summation is done in blocks to avoid excessive memory usage.

        :param x: :math:`n \times 1` array
        :param y: :math:`m \times 1` array
        :param max_size: Size of matrix blocks to process
        """
        # Ensure data is in desired format
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        num_points_x = len(x)
        num_points_y = len(y)

        # If max_size is larger than both inputs, we don't need to consider block-wise
        # computation
        if max_size > max(num_points_x, num_points_y):
            output = self.kernel.compute(x, y).sum()

        else:
            output = 0
            for i in range(0, num_points_x, max_size):
                for j in range(0, num_points_y, max_size):
                    pairwise_distances_part = self.kernel.compute(
                        x[i : i + max_size], y[j : j + max_size]
                    )
                    output += pairwise_distances_part.sum()

        return output

    def sum_weighted_pairwise_distances(
        self,
        x: ArrayLike,
        y: ArrayLike,
        weights_x: ArrayLike,
        weights_y: ArrayLike,
        max_size: int = 10_000,
    ) -> float:
        r"""
        Sum the kernel distance (weighted) between all pairs of points in x and y.

        The summation is done in blocks to avoid excessive memory usage.

        :param x: :math:`n \times 1` array
        :param y: :math:`m \times 1` array
        :param weights_x: :math: weights for x, `n \times 1` array
        :param weights_y: :math: weights for y, `m \times 1` array
        :param max_size: Size of matrix blocks to process
        """
        # Ensure data is in desired format
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        num_points_x = len(x)
        num_points_y = len(y)

        # If max_size is larger than both inputs, we don't need to consider block-wise
        # computation
        if max_size > max(num_points_x, num_points_y):
            kernel_weights = self.kernel.compute(x, y) * weights_y
            output = (weights_x * kernel_weights.T).sum()

        else:
            output = 0
            for i in range(0, num_points_x, max_size):
                for j in range(0, num_points_y, max_size):
                    pairwise_distances_part = (
                        weights_x[i : i + max_size, None]
                        * self.kernel.compute(x[i : i + max_size], y[j : j + max_size])
                        * weights_y[None, j : j + max_size]
                    )
                    output += pairwise_distances_part.sum()

        return output
