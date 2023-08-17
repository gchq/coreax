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

from jax import jit, vmap, Array
import jax.numpy as jnp
from jax.typing import ArrayLike
from coreax.utils import calculate_K_sum, KernelFunction
from .kernel_herding import kernel_herding_block
from .refine import refine, refine_rand, refine_rev

#
# Kernel Herding Refine Functions 
# 
# Combine kernel herding with a refine function to produce a coreset. 

def kernel_herding_refine_block(
        x: ArrayLike,
        n_core: int,
        kernel: KernelFunction,
        max_size: int = 10000,
        K_mean: ArrayLike | None = None,
) -> Array:
    """
    Implementation of kernel herding refine algorithm using jax. 
    Args:
        x: n x d original data
        n_core: number of coreset points to calculate
        kernel: kernel function k: R^d x R^d \to R
        max_size: size of matrix block to process
        K_mean: n row sum of kernel matrix divided by n
    
    Returns:
        coreset point indices
    """
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    x = jnp.asarray(x)
    n = len(x)
    if K_mean is None:
        K_mean = calculate_K_sum(x, k_pairwise, max_size)/n
    S = kernel_herding_block(x, n_core, kernel, max_size, K_mean)[0]
    S = refine(x, S, kernel, K_mean)

    return S

def kernel_herding_refine_rand_block(
        x: ArrayLike,
        n_core: int,
        kernel: KernelFunction,
        p: float = 0.1,
        max_size: int = 10000,
        K_mean: ArrayLike | None = None,
) -> Array:
    """
    Implementation of kernel herding random refine algorithm using jax. 
    Args:
        x: n x d original data
        n_core: number of coreset points to calculate
        kernel: kernel function k: R^d x R^d \to R
        p: proportion of original data to use as candidates
        max_size: size of matrix block to process
        K_mean: n row sum of kernel matrix divided by n
    
    Returns:
        coreset point indices
    """
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    x = jnp.asarray(x)
    n = len(x)
    if K_mean is None:
        K_mean = calculate_K_sum(x, k_pairwise, max_size)/n
    S, _, _ = kernel_herding_block(x, n_core, kernel, max_size, K_mean)
    S = refine_rand(x, S, kernel, K_mean, p)

    return S

def kernel_herding_refine_rev_block(
        x: ArrayLike,
        n_core: int,
        kernel: KernelFunction,
        max_size: int = 10000,
        K_mean: ArrayLike | None = None,
) -> Array:
    """
    Implementation of kernel herding random refine algorithm using jax. 
    Args:
        x: n x d original data
        n_core: number of coreset points to calculate
        kernel: kernel function k: R^d x R^d \to R
        max_size: size of matrix block to process
        K_mean: n row sum of kernel matrix divided by n
    
    Returns:
        coreset point indices
    """
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    x = jnp.asarray(x)
    n = len(x)
    if K_mean is None:
        K_mean = calculate_K_sum(x, k_pairwise, max_size)/n
    S, _, _ = kernel_herding_block(x, n_core, kernel, max_size, K_mean)
    S = refine_rev(x, S, kernel, K_mean)

    return S
