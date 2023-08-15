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

from collections.abc import Callable
import jax.numpy as jnp 
from jax import vmap, jit, Array
from jax.typing import ArrayLike

from coreax.utils import Kernel


def mmd(x: ArrayLike, x_c: ArrayLike, kernel: Kernel) -> Array:
    """
    Calculates maximum mean discrepancy

    Args:
        x: n x d original data
        x_c: m x d  coreset
        kernel: kernel function k: R^d x R^d \to R

    Returns:
        maximum mean discrepancy, as a zero-dimensional array
    """

    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))

    return  jnp.sqrt(k_pairwise(x,x).mean() + k_pairwise(x_c,x_c).mean() - 2*k_pairwise(x,x_c).mean())

def wmmd(
        x: ArrayLike,
        x_c: ArrayLike,
        kernel: Kernel,
        weights: ArrayLike,
) -> float:
    """One sided, weighted MMD, where weights are on the coreset points only.

    Args:
        x: original data points
        x_c: coreset points
        kernel: kernel function
        weights: weights' vector

    Returns:
        MMD value
    """
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    x = jnp.asarray(x)
    n = float(len(x))
    Kmm = k_pairwise(x_c, x_c)
    Knn = k_pairwise(x, x)
    Kmn = k_pairwise(x_c, x).mean(axis=1)
    return  jnp.sqrt(jnp.dot(weights.T, jnp.dot(Kmm, weights)) + Knn.sum()/n**2 - 2*jnp.dot(weights.T, Kmn)).item()

def sum_K(
        x: ArrayLike,
        y: ArrayLike,
        k_pairwise: Kernel,
        max_size: int = 10000,
) -> float:
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    n = len(x)
    m = len(y)
    output = 0
    for i in range(0,n,max_size):
        for j in range(0,m,max_size):
            
            K_part = k_pairwise(x[i:i+max_size],y[j:j+max_size])
            output += K_part.sum()
            
    return output
    

def mmd_block(
        x: ArrayLike,
        x_c: ArrayLike,
        kernel: Kernel,
        max_size: int = 10000,
) -> Array:
    """
    Calculates maximum mean discrepancy limiting memory requirements
    Args:
        x: n X d original data
        x_c: m x d  coreset
        kernel: kernel function k: R^d x R^d \to R
        max_size: size of matrix block to process

    
    Returns:
        maximum mean discrepancy, as a zero-dimensional array
    """

    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))

    x = jnp.asarray(x)
    x_c = jnp.asarray(x_c)
    n = float(len(x))
    m = float(len(x_c))
    K_n = sum_K(x, x, k_pairwise, max_size)
    K_m = sum_K(x_c, x_c, k_pairwise, max_size)
    K_nm = sum_K(x, x_c, k_pairwise, max_size)
            
    return jnp.sqrt(K_n/n**2 + K_m/m**2 - 2*K_nm/(n*m))

def sum_weight_K(
        x: ArrayLike,
        y: ArrayLike,
        w_x: ArrayLike,
        w_y: ArrayLike,
        k_pairwise: Callable[[ArrayLike, ArrayLike], Array],
        max_size: int = 10000,
) -> float:
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    n = len(x)
    m = len(y)
    output = 0
    for i in range(0,n,max_size):
        for j in range(0,m,max_size):
            
            K_part = w_x[i:i+max_size, None]*k_pairwise(x[i:i+max_size],y[j:j+max_size])*w_y[None,j:j+max_size]
            output += K_part.sum()
            
    return output

def mmd_weight_block(
        x: ArrayLike,
        x_c: ArrayLike,
        w: ArrayLike,
        w_c: ArrayLike,
        kernel: Kernel,
        max_size: int = 10000,
) -> Array:
    """
    Calculates weighted maximum mean discrepancy limiting memory requirements
    Args:
        x: n X d original data
        x_c: m x d  coreset
        w: n weights of original data
        w_c: m weights of coreset points
        kernel: kernel function k: R^d x R^d \to R
        max_size: size of matrix block to process

    Returns:
        weighted maximum mean discrepancy, as a zero-dimensional array
    """

    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))

    w = jnp.asarray(w)
    w_c = jnp.asarray(w_c)
    n = w.sum()
    m = w_c.sum()
    K_n = sum_weight_K(x, x, w, w, k_pairwise, max_size)
    K_m = sum_weight_K(x_c, x_c, w_c, w_c, k_pairwise, max_size)
    K_nm = sum_weight_K(x, x_c, w, w_c, k_pairwise, max_size)
            
    return jnp.sqrt(K_n/n**2 + K_m/m**2 - 2*K_nm/(n*m))
