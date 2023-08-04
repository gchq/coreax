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
from jax import jit, vmap
from coreax.utils import calculate_K_sum
from coreax.kernel_herding import kernel_herding_block
from coreax.refine import refine, refine_rand, refine_rev

#
# Kernel Herding Refine Functions 
# 
# Combine kernel herding with a refine function to produce a coreset. 

def kernel_herding_refine_block(x, n_core, kernel, max_size=10000, K_mean=None):
    """
    Implementation of kernel herding refine algorithm using jax. 
    Args:
        x (array_like): n x d original data 
        n_core (int): number of coreset points to calculate
        kernel (callable): kernel function k: R^d x R^d \to R
        max_size: size of matrix block to process
        K_mean (array_like): n row sum of kernel matrix divided by n 
    
    Returns:
        array: coreset point indices 
    """
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    n = len(x)
    if K_mean == None:
        K_mean = calculate_K_sum(x, k_pairwise, max_size)/n
    S = kernel_herding_block(x, n_core, kernel, max_size, K_mean)
    S = refine(x, S, kernel, K_mean)

    return S

def kernel_herding_refine_rand_block(x, n_core, kernel, p=0.1, max_size=10000, K_mean=None):
    """
    Implementation of kernel herding random refine algorithm using jax. 
    Args:
        x (array_like): n x d original data 
        n_core (int): number of coreset points to calculate
        kernel (callable): kernel function k: R^d x R^d \to R
        max_size: size of matrix block to process
        K_mean (array_like): n row sum of kernel matrix divided by n
    
    Returns:
        array: coreset point indices 
    """
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    n = len(x)
    if K_mean == None:
        K_mean = calculate_K_sum(x, k_pairwise, max_size)/n
    S = kernel_herding_block(x, n_core, kernel, max_size, K_mean)
    S = refine_rand(x, S, p, kernel, K_mean)

    return S

def kernel_herding_refine_rev_block(x, n_core, kernel, max_size=10000, K_mean=None):
    """
    Implementation of kernel herding random refine algorithm using jax. 
    Args:
        x (array_like): n x d original data 
        n_core (int): number of coreset points to calculate
        kernel (callable): kernel function k: R^d x R^d \to R
        max_size: size of matrix block to process
        K_mean (array_like): n row sum of kernel matrix divided by n
    
    Returns:
        array: coreset point indices 
    """
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    n = len(x)
    if K_mean == None:
        K_mean = calculate_K_sum(x, k_pairwise, max_size)/n
    S = kernel_herding_block(x, n_core, kernel, max_size, K_mean)
    S = refine_rev(x, S, kernel, K_mean)

    return S