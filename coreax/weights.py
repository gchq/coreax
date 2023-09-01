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
from jaxopt import OSQP

from coreax.utils import KernelFunction


def calculate_BQ_weights(
        x: ArrayLike,
        x_c: ArrayLike,
        kernel: KernelFunction,
) -> Array:
    r"""
    Calculate weights from Sequential Bayesian Quadrature (SBQ).

    References for this technique can be found in
    [huszar2016optimallyweighted]_. These are equivalent to the unconstrained weighted
    maximum mean discrepancy (MMD) optimum.

    :param x: The original :math:`n \times d` data
    :param x_c: :math:`m times d` coreset
    :param kernel: Kernel function
                   :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :return: Optimal weights
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
    r"""
    Compute optimal weights given the simplex constraint.

    :param x: The original :math:`n \times d` data
    :param x_c: :math:`m times d` coreset
    :param kernel: Kernel function
                   :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :return: Optimal weights
    """
    x = jnp.asarray(x)
    x_c = jnp.asarray(x_c)
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0))
    kbar = k_pairwise(x_c, x).sum(axis=1)/len(x)
    Kmm = k_pairwise(x_c, x_c) + 1e-10*jnp.identity(len(x_c))
    sol = qp(Kmm, kbar)
    return sol

def qp(Kmm: ArrayLike, Kbar: ArrayLike) -> Array:
    r"""
    Solve quadratic programs with :mod:`jaxopt`.

    Solves simplex weight problems of the form:

    .. math::

        \mathbf{w}^{\mathrm{T}} \mathbf{K} \mathbf{w} + \bar{\mathbf{k}}^{\mathrm{T}} \mathbf{w} = 0

    subject to

    .. math::

        \mathbf{Aw} = \mathbf{1}, \qquad \mathbf{Gx} \le 0.

    :param Kmm: :math:`m \times m` coreset Gram matrix
    :param Kbar: :math`m \times d` array of Gram matrix means
    :return: Optimised solution for the quadratic program
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
