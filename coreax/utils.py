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

# Support annotations with | in Python < 3.10
# TODO: Remove once no longer supporting old code
from __future__ import annotations

from collections.abc import Callable
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike


KernelFunction = Callable[[ArrayLike, ArrayLike], Array]

# Pairwise kernel evaluation if grads and nu are defined
KernelFunctionWithGrads = Callable[
    [ArrayLike, ArrayLike, ArrayLike, ArrayLike, int, float],
    Array
]


def update_K_sum(
        X: ArrayLike,
        K_sum: ArrayLike,
        i: int,
        j: int,
        max_size: int,
        k_pairwise: KernelFunction | KernelFunctionWithGrads,
        grads: ArrayLike | None = None,
        nu: float | None = None,
) -> Array:
    """Update row sum with the kernel matrix block i:i+max_size x j:j+max_size
    exploiting symmetry of kernel matix to reduce repeated calculation. 

    Args:
        X: Data matrix, n x d.
        K_sum: Full data structure for Gram matrix row sum, 1 x n.
        i: Block start.
        j: Block end.
        max_size: Size of matrix block to process.
        k_pairwise: Pairwise kernel evaluation function. This should be k(x, y) if grads and nu are None, else k(x, y, grads, grads, n, nu)
        grads: Array of gradients, if applicable, n x d. Optional, defaults to None.
        nu: Base kernel bandwidth. Optional, defaults to None.

    Returns:
        K_sum, with elements i: i + max_size, and j: j + max_size populated.
    """
    X = jnp.asarray(X)
    K_sum = jnp.asarray(K_sum)
    n = X.shape[0]
    if grads is None:
        K_part = k_pairwise(X[i:i + max_size], X[j:j + max_size])
    else:
        grads = jnp.asarray(grads)
        K_part = k_pairwise(X[i:i + max_size], X[j:j + max_size],
                            grads[i:i + max_size], grads[j:j + max_size], n, nu)
    K_sum = K_sum.at[i:i +
                     max_size].set(K_sum[i:i + max_size] + K_part.sum(axis=1))

    if i != j:
        K_sum = K_sum.at[j:j +
                         max_size].set(K_sum[j:j+max_size] + K_part.sum(axis=0))

    return K_sum


def calculate_K_sum(
        X: ArrayLike,
        k_pairwise: KernelFunction | KernelFunctionWithGrads,
        max_size: int,
        grads: ArrayLike | None = None,
        nu: ArrayLike | None = None,
) -> Array:
    """Calculate row sum of the kernel matrix. This is done blockwise to limit memory overhead.

    Args:
        X: Data matrix, n x d.
        k_pairwise: Pairwise kernel evaluation function. This should be k(x, y) if grads and nu are None, else k(x, y, grads, grads, n, nu).
        max_size: Size of matrix block to process.
        grads: Matrix of gradients, if applicable, n x d. Optional, defaults to None.
        nu: Kernel bandwidth parameter, if applicable, n x d. Optional, defaults to None.

    Returns:
        Kernel matrix row sum.
    """
    X = jnp.asarray(X)
    n = len(X)
    K_sum = jnp.zeros(n)
    # Iterate over upper trangular blocks
    for i in range(0, n, max_size):
        for j in range(i, n, max_size):
            K_sum = update_K_sum(X, K_sum, i, j, max_size,
                                 k_pairwise, grads, nu)

    return K_sum
