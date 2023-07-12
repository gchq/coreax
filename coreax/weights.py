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
from jaxopt import OSQP

def calculate_BQ_weights(x,x_c,kernel):
    """Weights from sequential Bayesian quadrature (SBQ). See https://arxiv.org/pdf/1204.1664.pdf

    These are equivalent to the unconstrained weighted MMD optimum.

    Args:
        x (array_like): n x d original data
        x_c (array_like): m x d coreset
        kernel (callable): kernel function k: R^d x R^d \to R

    Returns:
        ndarray: optimal solution
    """
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    z = k_pairwise(x_c, x).sum(axis=1)/len(x)
    K = k_pairwise(x_c,x_c) + 1e-10*jnp.identity(len(x_c))
    return jnp.linalg.solve(K, z)

def simplex_weights(x, x_c, kernel):
    """Compute optimal weights given the simplex constraint.

    Args:
        x (array_like): n x d original data
        x_c (array_like): m x d coreset
        kernel (callable): kernel function k: R^d x R^d \to R

    Returns:
        ndarray: optimal solution
    """
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0))
    kbar = k_pairwise(x_c, x).sum(axis=1)/len(x)
    Kmm = k_pairwise(x_c, x_c) + 1e-10*jnp.identity(len(x_c))
    sol = qp(Kmm, kbar)
    return sol

def qp(Kmm, Kbar):
    """Quadratic programming solver from jaxopt. Solves simplex weight problems of the form

    .. math::
        \mathbf{w}^{\mathrm{T}} \mathbf{K} \mathbf{w} + \bar{\mathbf{k}}^{\mathrm{T}} \mathbf{w} = 0
    
    subject to

    .. math::
        \mathbf{Aw} = \mathbf{1}, \qquad \mathbf{Gx} \le 0.

    Args:
        Kmm (array_like): m x m coreset Gram matrix
        Kbar (array_like): m x d array of Gram matrix means

    Returns:
        ndarray: optimal solution
    """
    m = Kmm.shape[0]
    Q = jnp.array(Kmm)
    c = -jnp.array(Kbar)
    A = jnp.ones((1, m))
    b = jnp.array([1.0])
    G = jnp.eye(m) * -1.
    h = jnp.zeros(m)

    qp = OSQP()
    sol = qp.run(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h)).params
    return sol.primal