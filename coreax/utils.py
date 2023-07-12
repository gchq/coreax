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


def update_K_sum(X, K_sum, i, j, max_size, k_pairwise, grads=None, nu=None):
    """Update row sum with the kernel matrix block i:i+max_size x j:j+max_size
    exploiting symmetry of kernel matix to reduce repeated calculation. 

    Args:
        X (array_like): Data matrix, n x d.
        K_sum (array_like): Full data structure for Gram matrix row sum, 1 x n.
        i (int): Block start.
        j (int): Block end.
        max_size (int): Size of matrix block to process.
        k_pairwise (callable): Pairwise kernel evaluation function. This should be k(x, y) if grads and nu are None, else k(x, y, grads, grads, nu)
        grads (array_like, optional): Array of gradients, if applicable, n x d. Defaults to None.
        nu (float, optional): Base kernel bandwidth. Defaults to None.

    Returns:
        ndarray: K_sum, with elements i: i + max_size, and j: j + max_size populated.
    """
    n = X.shape[0]
    if grads is None:
        K_part = k_pairwise(X[i:i + max_size], X[j:j + max_size])
    else:
        K_part = k_pairwise(X[i:i + max_size], X[j:j + max_size],
                            grads[i:i + max_size], grads[j:j + max_size], n, nu)
    K_sum = K_sum.at[i:i +
                     max_size].set(K_sum[i:i + max_size] + K_part.sum(axis=1))

    if i != j:
        K_sum = K_sum.at[j:j +
                         max_size].set(K_sum[j:j+max_size] + K_part.sum(axis=0))

    return K_sum


def calculate_K_sum(X, k_pairwise, max_size, grads=None, nu=None):
    """Calculate row sum of the kernel matrix. This is done blockwise to limit memory overhead.

    Args:
        X (array_like): Data matrix, n x d.
        k_pairwise (callable): Pairwise kernel evaluation function. This should be k(x, y) if grads and nu are None, else k(x, y, grads, grads, nu).
        max_size (int): Size of matrix block to process.
        grads (array_like, optional): Matrix of gradients, if applicable, n x d. Defaults to None.
        nu (float, optional): Kernel bandwidth parameter, if applicable, n x d. Defaults to None.

    Returns:
        ndarray: Kernel matrix row sum.
    """
    n = len(X)
    K_sum = jnp.zeros(n)
    # Iterate over upper trangular blocks
    for i in range(0, n, max_size):
        for j in range(i, n, max_size):
            K_sum = update_K_sum(X, K_sum, i, j, max_size,
                                 k_pairwise, grads, nu)

    return K_sum
