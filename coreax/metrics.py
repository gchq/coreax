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
from jax import vmap, jit

def mmd(x,x_c,kernel):
    """
    Calculates maximum mean discrepancy

    Args:
        x (array_like): n x d original data 
        x_c (array_like): m x d  coreset
        kernel (callable): kernel function k: R^d x R^d \to R

    Returns:
        float: maximum mean discrepancy 
    """

    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))

    return  jnp.sqrt(k_pairwise(x,x).mean() + k_pairwise(x_c,x_c).mean() - 2*k_pairwise(x,x_c).mean())

def wmmd(x, x_c, kernel, weights):
    """One sided, weighted MMD, where weights are on the coreset points only.

    Args:
        x (array_like): original data points
        x_c (array_like): coreset points
        kernel (callable): kernel function
        weights (array_like): weights' vector

    Returns:
        float: MMD value
    """
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    n = float(len(x))
    Kmm = k_pairwise(x_c, x_c)
    Knn = k_pairwise(x, x)
    Kmn = k_pairwise(x_c, x).mean(axis=1)
    return  jnp.sqrt(jnp.dot(weights.T, jnp.dot(Kmm, weights)) + Knn.sum()/n**2 - 2*jnp.dot(weights.T, Kmn)).item()

def sum_K(x, y, k_pairwise, max_size=10000):
    
    n = len(x)
    m = len(y)
    output = 0
    for i in range(0,n,max_size):
        for j in range(0,m,max_size):
            
            K_part = k_pairwise(x[i:i+max_size],y[j:j+max_size])
            output += K_part.sum()
            
    return output
    

def mmd_block(x, x_c, kernel, max_size=10000):
    """
    Calculates maximum mean discrepancy limiting memory requirements
    Args:
        x (array_like): n X d original data 
        x_c (array_like): m x d  coreset
        kernel (callable): kernel function k: R^d x R^d \to R
        max_size: size of matrix block to process

    
    Returns:
        float: maximum mean discrepancy 
    """

    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    
    n = float(len(x))
    m = float(len(x_c))
    K_n = sum_K(x, x, k_pairwise, max_size)
    K_m = sum_K(x_c, x_c, k_pairwise, max_size)
    K_nm = sum_K(x, x_c, k_pairwise, max_size)
            
    return jnp.sqrt(K_n/n**2 + K_m/m**2 - 2*K_nm/(n*m))

def sum_weight_K(x, y, w_x, w_y, k_pairwise, max_size=10000):
    
    n = len(x)
    m = len(y)
    output = 0
    for i in range(0,n,max_size):
        for j in range(0,m,max_size):
            
            K_part = w_x[i:i+max_size, None]*k_pairwise(x[i:i+max_size],y[j:j+max_size])*w_y[None,j:j+max_size]
            output += K_part.sum()
            
    return output

def mmd_weight_block(x, x_c, w, w_c, kernel, max_size=10000):
    """
    Calculates weighted maximum mean discrepancy limiting memory requirements
    Args:
        x (array_like): n X d original data 
        x_c (array_like): m x d  coreset
        w (array_like): n weights of original data 
        w_c (array_like): m weights of coreset points   
        kernel (callable): kernel function k: R^d x R^d \to R
        max_size: size of matrix block to process

    Returns:
        float: weighted maximum mean discrepancy 
    """

    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    
    n = w.sum()
    m = w_c.sum()
    K_n = sum_weight_K(x, x, w, w, k_pairwise, max_size)
    K_m = sum_weight_K(x_c, x_c, w_c, w_c, k_pairwise, max_size)
    K_nm = sum_weight_K(x, x_c, w, w_c, k_pairwise, max_size)
            
    return jnp.sqrt(K_n/n**2 + K_m/m**2 - 2*K_nm/(n*m))