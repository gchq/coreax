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

"""TODO: Create top-level docstring."""

import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

from coreax.util import KernelFunction, solve_qp


def calculate_BQ_weights(
    x: ArrayLike,
    x_c: ArrayLike,
    kernel: KernelFunction,
) -> Array:
    r"""
    Calculate weights from Sequential Bayesian Quadrature (SBQ).

    References for this technique can be found in
    :cite:p:`huszar2016optimally`. These are equivalent to the unconstrained
    weighted maximum mean discrepancy (MMD) optimum.

    The BQ estimate of the integral :math:`\int f(x) p(x) dx` can be viewed as a
    weighted version of kernel herding. The Bayesian quadrature weights, :math:`w_{BQ}`,
    are given by

    .. math::

        w_{BQ}^{(n)} = \sum_m z_m^T K_{mn}^{-1}

    for a dataset :math:`X` of :math:`n` points, with coreset :math:`X_c` of :math:`m`
    points. Here, for given kernel :math:`k`, we have :math:`z = \int k(X,X_c)p(x) dx`
    and :math:`K = k(X_c, X_c)` in the above expression.

    The weights do not need to sum to 1, and are not even necessarily positive.

    :param x: The original :math:`n \times d` data
    :param x_c: :math:`m \times d` coreset
    :param kernel: Kernel function
                   :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :return: Optimal weights
    """
    x = jnp.asarray(x)
    x_c = jnp.asarray(x_c)
    k_pairwise = jit(
        vmap(vmap(kernel, in_axes=(None, 0), out_axes=0), in_axes=(0, None), out_axes=0)
    )
    z = k_pairwise(x_c, x).sum(axis=1) / len(x)
    K = k_pairwise(x_c, x_c) + 1e-10 * jnp.identity(len(x_c))
    return jnp.linalg.solve(K, z)


def simplex_weights(
    x: ArrayLike,
    x_c: ArrayLike,
    kernel: KernelFunction,
) -> Array:
    r"""
    Compute optimal weights given the simplex constraint.

    :param x: The original :math:`n \times d` data
    :param x_c: :math:`m \times d` coreset
    :param kernel: Kernel function
                   :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :return: Optimal weights
    """
    x = jnp.asarray(x)
    x_c = jnp.asarray(x_c)
    k_pairwise = jit(
        vmap(vmap(kernel, in_axes=(None, 0), out_axes=0), in_axes=(0, None), out_axes=0)
    )
    z = k_pairwise(x_c, x).sum(axis=1) / len(x)
    K = k_pairwise(x_c, x_c) + 1e-10 * jnp.identity(len(x_c))
    sol = solve_qp(kmm=K, kbar=z)
    return sol
