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

# Support annotations with | in Python < 3.10
# TODO: Remove once no longer supporting old code
from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike
from jaxopt import OSQP

KernelFunction = Callable[[ArrayLike, ArrayLike], Array]

# Pairwise kernel evaluation if grads and nu are defined
KernelFunctionWithGrads = Callable[
    [ArrayLike, ArrayLike, ArrayLike, ArrayLike, int, float], Array
]


def apply_negative_precision_threshold(
    x: ArrayLike, precision_threshold: float = 1e-8
) -> float:
    """
    Round a number to 0.0 if it is negative but within precision_threshold of 0.0.

    :param x: Value we wish to compare to 0.0
    :param precision_threshold: Positive threshold we compare against for precision
    :returns: ``x``, rounded to 0.0 if it is between -precision_threshold and 0.0
    """
    # Cast to float. Will raise TypeError if array is not zero-dimensional.
    x = float(x)

    if precision_threshold < 0.0:
        raise ValueError(
            f"precision_threshold must be positive; value {precision_threshold} given."
        )

    if -precision_threshold < x < 0.0:
        return 0.0

    return x


@jit
def sq_dist(x: ArrayLike, y: ArrayLike) -> Array:
    """
    Calculate the squared distance between two vectors.

    :param x: First vector argument
    :param y: Second vector argument
    :return: Dot product of ``x - y`` and ``x - y``, the square distance between ``x``
             and ``y``
    """
    return jnp.dot(x - y, x - y)


@jit
def sq_dist_pairwise(x: ArrayLike, y: ArrayLike) -> Array:
    r"""
    Calculate efficient pairwise square distance between two arrays.

    :param x: First set of vectors as a :math:`n \times d` array
    :param y: Second set of vectors as a :math:`m \times d` array
    :return: Pairwise squared distances between ``x_array`` and ``y_array`` as an
        :math:`n \times m` array
    """
    # Use vmap to turn distance between individual vectors into a pairwise distance.
    d1 = vmap(sq_dist, in_axes=(None, 0), out_axes=0)
    d2 = vmap(d1, in_axes=(0, None), out_axes=0)

    return d2(x, y)


@jit
def diff(x: ArrayLike, y: ArrayLike) -> Array:
    """
    Calculate vector difference for a pair of vectors.

    :param x: First vector
    :param y: Second vector
    :return: Vector difference ``x - y``
    """
    return x - y


@jit
def pdiff(x_array: ArrayLike, y_array: ArrayLike) -> Array:
    r"""
    Calculate efficient pairwise difference between two arrays of vectors.

    :param x_array: First set of vectors as a :math:`n \times d` array
    :param y_array: Second set of vectors as a :math:`m \times d` array
    :return: Pairwise differences between ``x_array`` and ``y_array`` as an
        :math:`n \times m \times d` array
    """
    d1 = vmap(diff, in_axes=(0, None), out_axes=0)
    d2 = vmap(d1, in_axes=(None, 0), out_axes=1)

    return d2(x_array, y_array)


def solve_qp(kmm: ArrayLike, kbar: ArrayLike) -> Array:
    r"""
    Solve quadratic programs with :mod:`jax`.

    Solves simplex weight problems of the form:

    .. math::

        \mathbf{w}^{\mathrm{T}} \mathbf{k} \mathbf{w} + \bar{\mathbf{k}}^{\mathrm{T}}
        \mathbf{w} = 0

    subject to

    .. math::

        \mathbf{Aw} = \mathbf{1}, \qquad \mathbf{Gx} \le 0.

    :param kmm: :math:`m \times m` coreset Gram matrix
    :param kbar: :math`m \times d` array of Gram matrix means
    :return: Optimised solution for the quadratic program
    """
    q_array = jnp.array(kmm)
    c = -jnp.array(kbar)
    m = q_array.shape[0]
    a_array = jnp.ones((1, m))
    b = jnp.array([1.0])
    g_array = jnp.eye(m) * -1.0
    h = jnp.zeros(m)

    qp = OSQP()
    sol = qp.run(
        params_obj=(q_array, c), params_eq=(a_array, b), params_ineq=(g_array, h)
    ).params
    return sol.primal
