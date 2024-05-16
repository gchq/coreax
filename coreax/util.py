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
from functools import partial, wraps
from typing import Any, TypeVar

import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array, block_until_ready, jit, vmap
from jax.typing import ArrayLike
from jaxopt import OSQP
from typing_extensions import TypeAlias, deprecated

PyTreeDef: TypeAlias = Any
Leaf: TypeAlias = Any

#: JAX random key type annotations.
KeyArray: TypeAlias = Array
KeyArrayLike: TypeAlias = ArrayLike


class NotCalculatedError(Exception):
    """Raise when trying to use a variable that has not been calculated yet."""


class InvalidKernel:
    """
    Simple class that does not have a compute method on to test kernel.

    This is used across several testing instances to ensure the consequence of invalid
    inputs is correctly caught.
    """

    def __init__(self, x: float):
        """Initialise the invalid kernel object."""
        self.x = x


def tree_leaves_repeat(tree: PyTreeDef, length: int = 2) -> list[Leaf]:
    """
    Flatten a PyTree to its leaves and (potentially) repeat the trailing leaf.

    The PyTree 'tree' is flattened, but unlike the standard flattening, :data:`None` is
    treated as a valid leaf and the trailing leaf (potentially) repeated such that
    the length of the collection of leaves is given by the 'length' parameter.

    :param tree: The PyTree to flatten and whose trailing leaf to (potentially) repeat
    :param length: The length of the flattened PyTree after any repetition; values are
        implicitly clipped by :code:`max(len(tree_leaves), length)`
    :return: The PyTree leaves, with the trailing leaf repeated as many times as
        required for the collection of leaves to have length 'repeated_length'
    """
    tree_leaves = jtu.tree_leaves(tree, is_leaf=lambda x: x is None)
    num_repeats = length - len(tree_leaves)
    return tree_leaves + tree_leaves[-1:] * num_repeats


def apply_negative_precision_threshold(
    x: ArrayLike, precision_threshold: float = 1e-8
) -> Array:
    """
    Round a number to 0.0 if it is negative but within precision_threshold of 0.0.

    :param x: Scalar value we wish to compare to 0.0
    :param precision_threshold: Positive threshold we compare against for precision
    :return: ``x``, rounded to 0.0 if it is between ``-precision_threshold`` and 0.0
    """
    _x = jnp.asarray(x)
    return jnp.where((-jnp.abs(precision_threshold) < _x) & (_x < 0.0), 0.0, _x)


def pairwise(
    fn: Callable[[ArrayLike, ArrayLike], Array],
) -> Callable[[ArrayLike, ArrayLike], Array]:
    """
    Transform a function so it returns all pairwise evaluations of its inputs.

    :param fn: the function to apply the pairwise transform to.
    :returns: function that returns an array whose entries are the evaluations of `fn`
        for every pairwise combination of its input arguments.
    """

    @wraps(fn)
    def pairwise_fn(x: ArrayLike, y: ArrayLike) -> Array:
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        return vmap(
            vmap(fn, in_axes=(0, None), out_axes=0),
            in_axes=(None, 0),
            out_axes=1,
        )(x, y)

    return pairwise_fn


@jit
def squared_distance(x: ArrayLike, y: ArrayLike) -> Array:
    """
    Calculate the squared distance between two vectors.

    :param x: First vector argument
    :param y: Second vector argument
    :return: Dot product of ``x - y`` and ``x - y``, the square distance between ``x``
        and ``y``
    """
    x = jnp.atleast_1d(x)
    y = jnp.atleast_1d(y)
    return jnp.dot(x - y, x - y)


@deprecated(
    "Use coreax.util.pairwise(coreax.util.squared_distance)(x, y);"
    "will be removed in version 0.3.0"
)
def squared_distance_pairwise(x: ArrayLike, y: ArrayLike) -> Array:
    r"""
    Calculate efficient pairwise square distance between two arrays.

    :param x: First set of vectors as a :math:`n \times d` array
    :param y: Second set of vectors as a :math:`m \times d` array
    :return: Pairwise squared distances between ``x_array`` and ``y_array`` as an
        :math:`n \times m` array
    """
    return pairwise(squared_distance)(x, y)


@jit
def difference(x: ArrayLike, y: ArrayLike) -> Array:
    """
    Calculate vector difference for a pair of vectors.

    :param x: First vector
    :param y: Second vector
    :return: Vector difference ``x - y``
    """
    x = jnp.atleast_1d(x)
    y = jnp.atleast_1d(y)
    return x - y


@deprecated(
    "Use coreax.util.pairwise(coreax.util.difference)(x, y);"
    "will be removed in version 0.3.0"
)
def pairwise_difference(x: ArrayLike, y: ArrayLike) -> Array:
    r"""
    Calculate efficient pairwise difference between two arrays of vectors.

    :param x: First set of vectors as a :math:`n \times d` array
    :param y: Second set of vectors as a :math:`m \times d` array
    :return: Pairwise differences between ``x_array`` and ``y_array`` as an
        :math:`n \times m \times d` array
    """
    return pairwise(difference)(x, y)


def solve_qp(kernel_mm: ArrayLike, gramian_row_mean: ArrayLike) -> Array:
    r"""
    Solve quadratic programs with the :class:`jaxopt.OSQP` solver.

    Solves simplex weight problems of the form:

    .. math::

        \mathbf{w}^{\mathrm{T}} \mathbf{k} \mathbf{w} +
        \bar{\mathbf{k}}^{\mathrm{T}} \mathbf{w} = 0

    subject to

    .. math::

        \mathbf{Aw} = \mathbf{1}, \qquad \mathbf{Gx} \le 0.

    :param kernel_mm: :math:`m \times m` coreset Gram matrix
    :param gramian_row_mean: :math:`m \times 1` array of Gram matrix means
    :return: Optimised solution for the quadratic program
    """
    # Setup optimisation problem - all variable names are consistent with the OSQP
    # terminology. Begin with the objective parameters.
    q_array = jnp.asarray(kernel_mm)
    c = -jnp.asarray(gramian_row_mean)

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


def jit_test(
    fn: Callable,
    fn_args: tuple = (),
    fn_kwargs: dict | None = None,
    jit_kwargs: dict | None = None,
) -> tuple[float, float]:
    """
    Verify JIT performance by comparing timings of a before and after run of a function.

    The function is called with supplied arguments twice, and timed for each run. These
    timings are returned in a 2-tuple.

    :param fn: Function callable to test
    :param fn_args: Arguments passed during the calls to the passed function
    :param fn_kwargs: Keyword arguments passed during the calls to the passed function
    :param jit_kwargs: Keyword arguments that are partially applied to :func:`jax.jit`
        before being called to compile the passed function.
    :return: (First run time, Second run time)
    """
    # Avoid dangerous default values - Pylint W0102
    if fn_kwargs is None:
        fn_kwargs = {}
    if jit_kwargs is None:
        jit_kwargs = {}

    @partial(jit, **jit_kwargs)
    def _fn(*args, **kwargs):
        return fn(*args, **kwargs)

    assert hash(_fn) != hash(fn), "Cannot guarantee recompilation of `fn`."

    start_time = time.time()
    block_until_ready(_fn(*fn_args, **fn_kwargs))
    end_time = time.time()
    pre_delta = end_time - start_time
    start_time = time.time()
    block_until_ready(_fn(*fn_args, **fn_kwargs))
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
        """Do nothing instead of writing to output."""
