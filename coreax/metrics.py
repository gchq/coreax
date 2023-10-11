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

from coreax.util import KernelFunction, apply_negative_precision_threshold


def mmd(
    x: ArrayLike,
    x_c: ArrayLike,
    kernel: KernelFunction,
    precision_threshold: float = 1e-8,
) -> Array:
    r"""
     Calculate maximum mean discrepancy (MMD).

     :param x: The original :math:`n \times d` data
     :param x_c: :math:`m \times d` coreset
     :param kernel: Kernel function
                    :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param precision_threshold: Positive threshold we compare against for precision
     :return: Maximum mean discrepancy as a 0-dimensional array
    """
    k_pairwise = jit(
        vmap(vmap(kernel, in_axes=(None, 0), out_axes=0), in_axes=(0, None), out_axes=0)
    )

    # Compute MMD, correcting for any numerical precision issues, where we would
    # otherwise square-root a negative number very close to 0.0.
    result = jnp.sqrt(
        apply_negative_precision_threshold(
            k_pairwise(x, x).mean()
            + k_pairwise(x_c, x_c).mean()
            - 2 * k_pairwise(x, x_c).mean(),
            precision_threshold,
        )
    )
    return result


def wmmd(
    x: ArrayLike,
    x_c: ArrayLike,
    kernel: KernelFunction,
    weights: ArrayLike,
    precision_threshold: float = 1e-8,
) -> float:
    r"""
    Calculate one-sided, weighted maximum mean discrepancy (MMD).

    Only coreset points are weighted.

    :param x: The original :math:`n \times d` data
    :param x_c: :math:`m \times d` coreset
    :param kernel: Kernel function
                   :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param weights: :math:`n \times 1` weights vector
    :param precision_threshold: Positive threshold we compare against for precision
    :return: Maximum mean discrepancy as a 0-dimensional array
    """
    k_pairwise = jit(
        vmap(vmap(kernel, in_axes=(None, 0), out_axes=0), in_axes=(0, None), out_axes=0)
    )
    x = jnp.asarray(x)
    n = float(len(x))
    Kmm = k_pairwise(x_c, x_c)
    Knn = k_pairwise(x, x)
    Kmn = k_pairwise(x_c, x).mean(axis=1)

    # Compute MMD, correcting for any numerical precision issues, where we would
    # otherwise square-root a negative number very close to 0.0.
    result = jnp.sqrt(
        apply_negative_precision_threshold(
            jnp.dot(weights.T, jnp.dot(Kmm, weights))
            + Knn.sum() / n**2
            - 2 * jnp.dot(weights.T, Kmn),
            precision_threshold,
        )
    )
    return result


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


def mmd_block(
    x: ArrayLike,
    x_c: ArrayLike,
    kernel: KernelFunction,
    max_size: int = 10_000,
    precision_threshold: float = 1e-8,
) -> Array:
    r"""
    Calculate maximum mean discrepancy (MMD) whilst limiting memory requirements.

    :param x: The original :math:`n \times d` data
    :param x_c: :math:`m \times d` coreset
    :param kernel: Kernel function
                   :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param max_size: Size of matrix blocks to process
    :param precision_threshold: Positive threshold we compare against for precision
    :return: Maximum mean discrepancy as a 0-dimensional array
    """
    k_pairwise = jit(
        vmap(vmap(kernel, in_axes=(None, 0), out_axes=0), in_axes=(0, None), out_axes=0)
    )

    x = jnp.asarray(x)
    x_c = jnp.asarray(x_c)
    n = float(len(x))
    m = float(len(x_c))
    K_n = sum_K(x, x, k_pairwise, max_size)
    K_m = sum_K(x_c, x_c, k_pairwise, max_size)
    K_nm = sum_K(x, x_c, k_pairwise, max_size)

    # Compute MMD, correcting for any numerical precision issues, where we would
    # otherwise square-root a negative number very close to 0.0.
    result = jnp.sqrt(
        apply_negative_precision_threshold(
            K_n / n**2 + K_m / m**2 - 2 * K_nm / (n * m), precision_threshold
        )
    )
    return result


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


def mmd_weight_block(
    x: ArrayLike,
    x_c: ArrayLike,
    w: ArrayLike,
    w_c: ArrayLike,
    kernel: KernelFunction,
    max_size: int = 10_000,
    precision_threshold: float = 1e-8,
) -> Array:
    r"""
    Calculate weighted maximum mean discrepancy (MMD).

    This calculation is executed whilst limiting memory requirements.

    :param x: The original :math:`n \times d` data
    :param x_c: :math:`m \times d` coreset
    :param w: :math:`n` weights of original data
    :param w_c: :math:`m` weights of coreset points
    :param kernel: Kernel function
                   :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param max_size: Size of matrix blocks to process
    :param precision_threshold: Positive threshold we compare against for precision
    :return: Maximum mean discrepancy as a 0-dimensional array
    """
    k_pairwise = jit(
        vmap(vmap(kernel, in_axes=(None, 0), out_axes=0), in_axes=(0, None), out_axes=0)
    )

    w = jnp.asarray(w)
    w_c = jnp.asarray(w_c)
    n = w.sum()
    m = w_c.sum()
    K_n = sum_weight_K(x, x, w, w, k_pairwise, max_size)
    K_m = sum_weight_K(x_c, x_c, w_c, w_c, k_pairwise, max_size)
    K_nm = sum_weight_K(x, x_c, w, w_c, k_pairwise, max_size)

    # Compute MMD, correcting for any numerical precision issues, where we would
    # otherwise square-root a negative number very close to 0.0.
    result = jnp.sqrt(
        apply_negative_precision_threshold(
            K_n / n**2 + K_m / m**2 - 2 * K_nm / (n * m), precision_threshold
        )
    )
    return result
