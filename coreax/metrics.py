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

import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

from coreax.utils import KernelFunction, apply_negative_precision_threshold


class Metric:
    r"""
    Base class for calculating metrics.
    """

    def __init__(self) -> None:
        pass

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
        """
        raise NotImplementedError


class MMD(Metric):
    r"""
    Calculation for maximum mean discrepancy.
    """

    def __init__(self, kernel: KernelFunction, precision_threshold: float = 1e-8):
        r"""
        Calculate maximum mean discrepancy between two datasets in d dimensions.

        :param kernel: Kernel function
                    :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
        :param precision_threshold: Positive threshold we compare against for precision
        """
        self.kernel = kernel
        self.precision_threshold = precision_threshold

        self.k_pairwise = jit(
            vmap(
                vmap(self.kernel, in_axes=(None, 0), out_axes=0),  # self.kernel.compute
                in_axes=(0, None),
                out_axes=0,
            )
        )

        # initialise parent
        super().__init__()

    def compute(
        self,
        x: ArrayLike,
        x_c: ArrayLike,
        max_size: int = None,
        weights_x: ArrayLike = None,
        weights_x_c: ArrayLike = None,
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
        :param weights_x: (Optional)  :math:`n \times 1` weights for original data X
        :param weights_x_c: (Optional)  :math:`m \times 1` weights for coreset Y
        :return: Maximum mean discrepancy as a 0-dimensional array
        """

        n = len(jnp.asarray(x))
        m = len(jnp.asarray(x_c))

        if weights_x_c is None:
            if max_size is None or max_size > max(n, m):
                return mmd(self, x, x_c)
            else:
                return mmd_block(self, x, x_c, max_size)

        else:
            if max_size is None or max_size > max(n, m):
                return wmmd(self, x, x_c, weights_x_c)
            else:
                return mmd_weight_block(self, x, x_c, max_size, weights_x, weights_x_c)


def mmd(self, x: ArrayLike, x_c: ArrayLike) -> Array:
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

    kernel_nn = self.k_pairwise(x, x)
    kernel_mm = self.k_pairwise(x_c, x_c)
    kernel_nm = self.k_pairwise(x, x_c)

    result = jnp.sqrt(
        apply_negative_precision_threshold(
            kernel_nn.mean() + kernel_mm.mean() - 2 * kernel_nm.mean(),
            self.precision_threshold,
        )
    )
    return result


def wmmd(self, x: ArrayLike, x_c: ArrayLike, weights_x_c: ArrayLike) -> Array:
    r"""
    Calculate one-sided, weighted maximum mean discrepancy (WMMD).

    Only coreset points are weighted.

    :param x: The original :math:`n \times d` data
    :param x_c: :math:`m \times d` coreset
    :param weights_x_c: :math:`m \times 1` weights vector for coreset
    :return: Weighted maximum mean discrepancy as a 0-dimensional array
    """

    x = jnp.asarray(x)
    n = float(len(x))
    kernel_nn = self.k_pairwise(x, x)
    kernel_mm = self.k_pairwise(x_c, x_c)
    kernel_nm = self.k_pairwise(x, x_c)

    # Compute MMD, correcting for any numerical precision issues, where we would
    # otherwise square-root a negative number very close to 0.0.
    result = jnp.sqrt(
        apply_negative_precision_threshold(
            jnp.dot(weights_x_c.T, jnp.dot(kernel_mm, weights_x_c))
            + kernel_nn.sum() / n**2
            - 2 * jnp.dot(weights_x_c.T, kernel_nm.mean(axis=0)),
            self.precision_threshold,
        )
    )
    return result


def mmd_block(self, x: ArrayLike, x_c: ArrayLike, max_size: int) -> Array:
    r"""
    Calculate maximum mean discrepancy (MMD) whilst limiting memory requirements.

    :param x: The original :math:`n \times d` data
    :param x_c: :math:`m \times d` coreset
    :param max_size: Size of matrix blocks to process
    :return: Maximum mean discrepancy as a 0-dimensional array
    """

    x = jnp.asarray(x)
    x_c = jnp.asarray(x_c)
    n = float(len(x))
    m = float(len(x_c))
    kernel_nn = sum_K(x, x, self.k_pairwise, max_size)
    kernel_mm = sum_K(x_c, x_c, self.k_pairwise, max_size)
    kernel_nm = sum_K(x, x_c, self.k_pairwise, max_size)  # self.kernel.calculate_K_sum

    # Compute MMD, correcting for any numerical precision issues, where we would
    # otherwise square-root a negative number very close to 0.0.
    result = jnp.sqrt(
        apply_negative_precision_threshold(
            kernel_nn / n**2 + kernel_mm / m**2 - 2 * kernel_nm / (n * m),
            self.precision_threshold,
        )
    )
    return result


def mmd_weight_block(
    self,
    x: ArrayLike,
    x_c: ArrayLike,
    max_size: int,
    weights_x: ArrayLike,
    weights_x_c: ArrayLike,
) -> Array:
    r"""
    Calculate weighted maximum mean discrepancy (MMD).

    This calculation is executed whilst limiting memory requirements.

    :param x: The original :math:`n \times d` data
    :param x_c: :math:`m \times d` coreset
    :param max_size: Size of matrix blocks to process
    :param weights_x: :math:`n` weights of original data
    :param weights_x_c: :math:`m` weights of coreset points
    :return: Maximum mean discrepancy as a 0-dimensional array
    """

    weights_x = jnp.asarray(weights_x)
    weights_x_c = jnp.asarray(weights_x_c)
    n = weights_x.sum()
    m = weights_x_c.sum()
    # needs changing to self.kernel.calculate_K_sum:
    kernel_nn = sum_weight_K(x, x, weights_x, weights_x, self.k_pairwise, max_size)
    kernel_mm = sum_weight_K(
        x_c, x_c, weights_x_c, weights_x_c, self.k_pairwise, max_size
    )
    kernel_nm = sum_weight_K(x, x_c, weights_x, weights_x_c, self.k_pairwise, max_size)

    # Compute MMD, correcting for any numerical precision issues, where we would
    # otherwise square-root a negative number very close to 0.0.
    result = jnp.sqrt(
        apply_negative_precision_threshold(
            kernel_nn / n**2 + kernel_mm / m**2 - 2 * kernel_nm / (n * m),
            self.precision_threshold,
        )
    )
    return result


# Below to be moved to Kernel.py


def sum_K(
    x: ArrayLike,
    y: ArrayLike,
    k_pairwise: KernelFunction,
    max_size: int = 10_000,
) -> float:
    r"""
    Sum the kernel distance between all pairs of points in x and y.

    The summation is done in blocks to avoid excessive memory usage.

    :param x: :math:`n \times 1` array
    :param y: :math:`m \times 1` array
    :param k_pairwise: Kernel function
    :param max_size: Size of matrix blocks to process
    """

    x = jnp.asarray(x)
    y = jnp.asarray(y)
    n = len(x)
    m = len(y)

    if max_size > max(m, n):
        output = k_pairwise(x, y).sum()

    else:
        output = 0
        for i in range(0, n, max_size):
            for j in range(0, m, max_size):
                K_part = k_pairwise(x[i : i + max_size], y[j : j + max_size])
                output += K_part.sum()

    return output


def sum_weight_K(
    x: ArrayLike,
    y: ArrayLike,
    w_x: ArrayLike,
    w_y: ArrayLike,
    k_pairwise: KernelFunction,
    max_size: int = 10_000,
) -> float:
    r"""
    Sum the kernel distance (weighted) between all pairs of points in x and y.

    The summation is done in blocks to avoid excessive memory usage.

    :param x: :math:`n \times 1` array
    :param y: :math:`m \times 1` array
    :param w_x: :math: weights for x, `n \times 1` array
    :param w_y: :math: weights for y, `m \times 1` array
    :param k_pairwise: Kernel function
    :param max_size: Size of matrix blocks to process
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    n = len(x)
    m = len(y)

    if max_size > max(m, n):
        Kw = k_pairwise(x, y) * w_y
        output = (w_x * Kw.T).sum()

    else:
        output = 0
        for i in range(0, n, max_size):
            for j in range(0, m, max_size):
                K_part = (
                    w_x[i : i + max_size, None]
                    * k_pairwise(x[i : i + max_size], y[j : j + max_size])
                    * w_y[None, j : j + max_size]
                )
                output += K_part.sum()

    return output
