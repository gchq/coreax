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
from jax import vmap, jit, random, lax
from functools import partial

def K_mean_rand_approx(key, x, kernel, n_points=1000, n_train=2000):
    """
    Method for approximating the kernel matrix row sum divided by n using kernel regression on points selected
    randomly

    Args:
        key: key for random number generation
        x (array_like): n X d original data 
        kernel (callable): kernel function k: R^d x R^d \to R
        n_points (int): number of kernel evaluations points
        n_train (int): number of training points to use to fit kernel regression 
 
    Returns:
        array: kernel matrix row sum divide by n approximation 
    """
    n = len(x)
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))

    # Randomly select points for kernel regression
    key, subkey = random.split(key)
    features_idx = random.choice(subkey, n, (n_points,), replace=False)
    features = k_pairwise(x, x[features_idx])

    # Select training points 
    train_idx = random.choice(key, n, (n_train,), replace=False)
    target = k_pairwise(x[train_idx],x).sum(axis=1)/n

    # Solve regression problem.
    params, _, _, _ = jnp.linalg.lstsq(features[train_idx], target)
    
    return features@params


def K_mean_ANNchor_approx(key, x, kernel, n_points=1000, n_train=2000):
    """
    Method for approximating the kernel matrix row sum divided by n using kernel regression on points selected
    by ANNchor construction

    Args:
        key: key for random number generation
        x (array_like): n X d original data 
        kernel (callable): kernel function k: R^d x R^d \to R
        n_points (int): number of kernel evaluations points
        n_train (int): number of training points to use to fit kernel regression 
 
    Returns:
        array: kernel matrix row sum divide by n approximation 
    """
    n = len(x)
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    k_vec = jit(vmap(kernel, in_axes=(0,None)))

    # Select point for kernel regression using ANNchor construction
    features = jnp.zeros((n,n_points))
    features = features.at[:,0].set(k_vec(x,x[0]))
    body = partial(anchor_body, x=x, k_vec=k_vec)
    features = lax.fori_loop(1, n_points, body, features)

    train_idx = random.choice(key, n, (n_train,), replace=False)
    target = k_pairwise(x[train_idx],x).sum(axis=1)/n

    # solve regression problem
    params, _, _, _ = jnp.linalg.lstsq(features[train_idx], target)
    
    return features@params

@partial(jit, static_argnames=["k_vec"])
def anchor_body(i, features, x, k_vec):
    
    j  = features.max(axis=1).argmin()
    features = features.at[:,i].set(k_vec(x,x[j]))
    
    return features

def K_mean_nystrom_approx(key, x, kernel, n_points=1000):
    """
    Method for approximating the kernel matrix row sum divided by n using nystrom approximation 
    https://arxiv.org/abs/2201.13055

    Args:
        key: key for random number generation
        x (array_like): n X d original data 
        kernel (callable): kernel function k: R^d x R^d \to R
        n_points (int): number of kernel evaluations points
 
    Returns:
        array: kernel matrix row sum divide by n approximation 
    """
    n = len(x)
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    S = random.choice(key, n, (n_points,))
    K_mn = k_pairwise(x[S],x)
    K_mm = k_pairwise(x[S],x[S])
    alpha = (jnp.linalg.pinv(K_mm)@K_mn).sum(axis=1)/n
    return K_mn.T @ alpha

