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
from jax.typing import ArrayLike
from jax import vmap, jit, Array
from jaxopt import OSQP

from coreax.utils import KernelFunction


def calculate_BQ_weights(
        x: ArrayLike,
        x_c: ArrayLike,
        kernel: KernelFunction,
) -> Array:
    """Weights from sequential Bayesian quadrature (SBQ). See https://arxiv.org/pdf/1204.1664.pdf

    These are equivalent to the unconstrained weighted MMD optimum.

    Args:
        x: n x d original data
        x_c: m x d coreset
        kernel: kernel function k: R^d x R^d \to R

    Returns:
        optimal solution
    """
    x = jnp.asarray(x)
    x_c = jnp.asarray(x_c)
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    z = k_pairwise(x_c, x).sum(axis=1)/len(x)
    K = k_pairwise(x_c,x_c) + 1e-10*jnp.identity(len(x_c))
    return jnp.linalg.solve(K, z)

def simplex_weights(
        x: ArrayLike,
        x_c: ArrayLike,
        kernel: KernelFunction,
) -> Array:
    """Compute optimal weights given the simplex constraint.

    Args:
        x: n x d original data
        x_c: m x d coreset
        kernel: kernel function k: R^d x R^d \to R

    Returns:
        optimal solution
    """
    x = jnp.asarray(x)
    x_c = jnp.asarray(x_c)
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0))
    kbar = k_pairwise(x_c, x).sum(axis=1)/len(x)
    Kmm = k_pairwise(x_c, x_c) + 1e-10*jnp.identity(len(x_c))
    sol = qp(Kmm, kbar)
    return sol

def qp(Kmm: ArrayLike, Kbar: ArrayLike) -> Array:
    """Quadratic programming solver from jaxopt. Solves simplex weight problems of the form

    .. math::
        \mathbf{w}^{\mathrm{T}} \mathbf{K} \mathbf{w} + \bar{\mathbf{k}}^{\mathrm{T}} \mathbf{w} = 0
    
    subject to

    .. math::
        \mathbf{Aw} = \mathbf{1}, \qquad \mathbf{Gx} \le 0.

    Args:
        Kmm: m x m coreset Gram matrix
        Kbar: m x d array of Gram matrix means

    Returns:
        optimal solution
    """
    Q = jnp.array(Kmm)
    c = -jnp.array(Kbar)
    m = Q.shape[0]
    A = jnp.ones((1, m))
    b = jnp.array([1.0])
    G = jnp.eye(m) * -1.
    h = jnp.zeros(m)

    qp = OSQP()
    sol = qp.run(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h)).params
    return sol.primal
