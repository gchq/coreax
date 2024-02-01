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

"""
Functionality to perform simple, generic tasks and operations.

The functions within this module are simple solutions to various problems or
requirements that are sufficiently generic to be useful across multiple areas of the
codebase. Examples of this include computation of squared distances, definition of
class factories and checks for numerical precision.
"""

# Support annotations with | in Python < 3.10
from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike
from jaxopt import OSQP

import coreax.validation

#: Kernel evaluation function.
KernelComputeType = Callable[[ArrayLike, ArrayLike], Array]


class NotCalculatedError(Exception):
    """Raise when trying to use a variable that has not been calculated yet."""


def apply_negative_precision_threshold(
    x: ArrayLike, precision_threshold: float = 1e-8
) -> float:
    """
    Round a number to 0.0 if it is negative but within precision_threshold of 0.0.

    :param x: Scalar value we wish to compare to 0.0
    :param precision_threshold: Positive threshold we compare against for precision
    :return: ``x``, rounded to 0.0 if it is between ``-precision_threshold`` and 0.0
    """
    # Validate inputs
    x = coreax.validation.cast_as_type(x=x, object_name="x", type_caster=float)
    precision_threshold = coreax.validation.cast_as_type(
        x=precision_threshold, object_name="precision_threshold", type_caster=float
    )
    coreax.validation.validate_in_range(
        x=precision_threshold,
        object_name="precision_threshold",
        strict_inequalities=False,
        lower_bound=0,
    )

    if precision_threshold < 0.0:
        raise ValueError(
            f"precision_threshold must be positive; value {precision_threshold} given."
        )

    if -precision_threshold < x < 0.0:
        return 0.0

    return x


@jit
def squared_distance(x: ArrayLike, y: ArrayLike) -> Array:
    """
    Calculate the squared distance between two vectors.

    :param x: First vector argument
    :param y: Second vector argument
    :return: Dot product of ``x - y`` and ``x - y``, the square distance between ``x``
        and ``y``
    """
    # Validate inputs
    x = coreax.validation.cast_as_type(x=x, object_name="x", type_caster=jnp.atleast_1d)
    y = coreax.validation.cast_as_type(x=y, object_name="y", type_caster=jnp.atleast_1d)
    return jnp.dot(x - y, x - y)


@jit
def squared_distance_pairwise(x: ArrayLike, y: ArrayLike) -> Array:
    r"""
    Calculate efficient pairwise square distance between two arrays.

    :param x: First set of vectors as a :math:`n \times d` array
    :param y: Second set of vectors as a :math:`m \times d` array
    :return: Pairwise squared distances between ``x_array`` and ``y_array`` as an
        :math:`n \times m` array
    """
    # Validate inputs
    x = coreax.validation.cast_as_type(x=x, object_name="x", type_caster=jnp.atleast_2d)
    y = coreax.validation.cast_as_type(x=y, object_name="y", type_caster=jnp.atleast_2d)
    # Use vmap to turn distance between individual vectors into a pairwise distance.
    fn = vmap(
        vmap(squared_distance, in_axes=(None, 0), out_axes=0),
        in_axes=(0, None),
        out_axes=0,
    )
    return fn(x, y)


@jit
def difference(x: ArrayLike, y: ArrayLike) -> Array:
    """
    Calculate vector difference for a pair of vectors.

    :param x: First vector
    :param y: Second vector
    :return: Vector difference ``x - y``
    """
    # Validate inputs
    x = coreax.validation.cast_as_type(x=x, object_name="x", type_caster=jnp.atleast_1d)
    y = coreax.validation.cast_as_type(x=y, object_name="y", type_caster=jnp.atleast_1d)
    return x - y


@jit
def pairwise_difference(x: ArrayLike, y: ArrayLike) -> Array:
    r"""
    Calculate efficient pairwise difference between two arrays of vectors.

    :param x: First set of vectors as a :math:`n \times d` array
    :param y: Second set of vectors as a :math:`m \times d` array
    :return: Pairwise differences between ``x_array`` and ``y_array`` as an
        :math:`n \times m \times d` array
    """
    # Validate inputs
    x = coreax.validation.cast_as_type(x=x, object_name="x", type_caster=jnp.atleast_2d)
    y = coreax.validation.cast_as_type(x=y, object_name="y", type_caster=jnp.atleast_2d)
    fn = vmap(
        vmap(difference, in_axes=(0, None), out_axes=0), in_axes=(None, 0), out_axes=1
    )
    return fn(x, y)


def solve_qp(kernel_mm: ArrayLike, kernel_matrix_row_sum_mean: ArrayLike) -> Array:
    r"""
    Solve quadratic programs with :mod:`jaxopt`.

    Solves simplex weight problems of the form:

    .. math::

        \mathbf{w}^{\mathrm{T}} \mathbf{k} \mathbf{w} +
        \bar{\mathbf{k}}^{\mathrm{T}} \mathbf{w} = 0

    subject to

    .. math::

        \mathbf{Aw} = \mathbf{1}, \qquad \mathbf{Gx} \le 0.

    :param kernel_mm: :math:`m \times m` coreset Gram matrix
    :param kernel_matrix_row_sum_mean: :math:`m \times 1` array of Gram matrix means
    :return: Optimised solution for the quadratic program
    """
    # Validate inputs
    coreax.validation.validate_is_instance(
        x=kernel_mm, object_name="kernel_mm", expected_type=Array
    )
    coreax.validation.validate_is_instance(
        x=kernel_matrix_row_sum_mean,
        object_name="kernel_matrix_row_sum_mean",
        expected_type=Array,
    )

    # Setup optimisation problem - all variable names are consistent with the OSQP
    # terminology. Begin with the objective parameters
    q_array = jnp.array(kernel_mm)
    c = -jnp.array(kernel_matrix_row_sum_mean)

    # Define the equality constraint parameters
    num_points = q_array.shape[0]
    a_array = jnp.ones((1, num_points))
    b = jnp.array([1.0])

    # Define the inequality constraint parameters
    g_array = jnp.eye(num_points) * -1.0
    h = jnp.zeros(num_points)

    # Define solver object and run solver
    qp = OSQP()
    sol = qp.run(
        params_obj=(q_array, c), params_eq=(a_array, b), params_ineq=(g_array, h)
    ).params
    return sol.primal


def jit_test(fn: Callable, *args, **kwargs) -> tuple[float, float]:
    """
    Verify JIT performance by comparing timings of a before and after run of a function.

    The function is called with supplied arguments twice, and timed for each run. These
    timings are returned in a 2-tuple.

    Note that `fn` often uses a lambda wrapper around a function or method call (see
    performance tests) to ensure that the function or method is recompiled when called
    multiple times, to truly test the JIT performance. In some cases, not doing this
    will result in the re-use of previously cached information.

    :param fn: Function callable to test
    :return: (First run time, Second run time)
    """
    start_time = time.time()
    fn(*args, **kwargs)
    end_time = time.time()
    pre_delta = end_time - start_time
    start_time = time.time()
    fn(*args, **kwargs)
    end_time = time.time()
    post_delta = end_time - start_time
    return pre_delta, post_delta


T = TypeVar("T")


class SilentTQDM:
    """
    Class implementing interface of :class:`~tqdm.tqdm` that does nothing.

    It can substitute :class:`~tqdm.tqdm` to silence all output.

    Based on `code by Pro Q <https://stackoverflow.com/a/77450937>`_.

    Additional parameters are accepted and ignored to match interface of
    :class:`~tqdm.tqdm`.

    :param iterable: Iterable of tasks to (not) indicate progress for
    """

    def __init__(self, iterable: Iterable[T], *_args, **_kwargs):
        """Store iterable."""
        self.iterable = iterable

    def __iter__(self) -> Iterator[T]:
        """
        Iterate.

        :return: Next item
        """
        return iter(self.iterable)

    def write(self, *_args, **_kwargs) -> None:
        """
        Do nothing instead of writing to output.

        :return: Nothing
        """
