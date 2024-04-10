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
from functools import partial
from typing import TypeVar

import jax.numpy as jnp
from jax import Array, block_until_ready, jit, vmap
from jax.random import split, permutation
from jax.typing import ArrayLike
from jaxopt import OSQP
from typing_extensions import TypeAlias

#: Kernel evaluation function.
KernelComputeType = Callable[[ArrayLike, ArrayLike], Array]

#: JAX random key type annotations.
KeyArray: TypeAlias = Array
KeyArrayLike: TypeAlias = ArrayLike


class NotCalculatedError(Exception):
    """Raise when trying to use a variable that has not been calculated yet."""


# pylint: disable=too-few-public-methods
class InvalidKernel:
    """
    Simple class that does not have a compute method on to test kernel.

    This is used across several testing instances to ensure the consequence of invalid
    inputs is correctly caught.
    """

    def __init__(self, x: float):
        """Initialise the invalid kernel object."""
        self.x = x


# pylint: enable=too-few-public-methods


def apply_negative_precision_threshold(
    x: float, precision_threshold: float = 1e-8
) -> float:
    """
    Round a number to 0.0 if it is negative but within precision_threshold of 0.0.

    :param x: Scalar value we wish to compare to 0.0
    :param precision_threshold: Positive threshold we compare against for precision
    :return: ``x``, rounded to 0.0 if it is between ``-precision_threshold`` and 0.0
    """
    if precision_threshold < 0.0:
        raise ValueError("precision_threshold must not be negative.")
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
    x = jnp.atleast_1d(x)
    y = jnp.atleast_1d(y)
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
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
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
    x = jnp.atleast_1d(x)
    y = jnp.atleast_1d(y)
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
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    fn = vmap(
        vmap(difference, in_axes=(0, None), out_axes=0), in_axes=(None, 0), out_axes=1
    )
    return fn(x, y)


def solve_qp(kernel_mm: ArrayLike, kernel_matrix_row_sum_mean: ArrayLike) -> Array:
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
    :param kernel_matrix_row_sum_mean: :math:`m \times 1` array of Gram matrix means
    :return: Optimised solution for the quadratic program
    """
    # Setup optimisation problem - all variable names are consistent with the OSQP
    # terminology. Begin with the objective parameters.
    q_array = jnp.asarray(kernel_mm)
    c = -jnp.asarray(kernel_matrix_row_sum_mean)

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


@jit
def invert_regularised_array(
    array: ArrayLike,
    regularisation_parameter: float,
    identity: ArrayLike,
    rcond: float | None = None
) -> ArrayLike:
    """
    Using a least-squares solver, regularise the array and then invert it.
    
    The function is designed to invert square block arrays where only the top-left block is non-zero.
    That is, we return a block array, the same size as the input array, where each block consists
    of zeros except for the top-left block, which is the inverse of the original non-zero block. To achieve
    this the 'identity' array must be a zero matrix except for ones on the diagonal up to the size
    of the non-zero block.

    :param array: Array to be inverted
    :param regularisation_parameter: Regularisation parameter for stable inversion of array
    :param identity: Block identity matrix
    :param rcond: Cut-off ratio for small singular values of a. For the purposes of rank determination,
        singular values are treated as zero if they are smaller than rcond times the largest singular value of a
    :return: Inverse of regularised array
    """
    return jnp.linalg.lstsq(
        array + regularisation_parameter * identity,
        identity,
        rcond=rcond
    )[0]


@jit
def invert_stacked_regularised_arrays(
    stacked_arrays: ArrayLike,
    regularisation_parameter: float,
    identity: ArrayLike,
    rcond: float | None = None
) -> ArrayLike:
    """
    Efficiently invert a stack of regularised square arrays.

    The function is designed to invert a stack of square block arrays where only the top-left block is non-zero.
    That is, we return a stack of block arrays, the same size as the stack of input arrays, where each block consists
    of zeros except for the top-left block, which is the inverse of the original non-zero block. To achieve
    this the 'identity' array must be a zero matrix except for ones on the diagonal up to the size
    of the non-zero block.

    :param array: Stack of arrays to be inverted
    :param regularisation_parameter: Regularisation parameter for stable inversion of arrays
    :param identity: Block identity matrix
    :param rcond: Cut-off ratio for small singular values of a. For the purposes of rank determination,
        singular values are treated as zero if they are smaller than rcond times the largest singular value of a
    :return: Stack of inverted regularised arrays
    """
    return vmap(
        partial(
            invert_regularised_array,
            regularisation_parameter=regularisation_parameter,
            identity=identity,
            rcond=rcond
        )
    )(stacked_arrays)

def sample_batch_indices(
    random_key: coreax.util.KeyArrayLike,
    data_size: int,
    batch_size: int,
    num_batches: int
) -> tuple[coreax.util.KeyArrayLike, ArrayLike]:
    """
    Sample an array of column-unique indices where the largest possible index is dictated by data_size.

    :param random_key: Key for random number generation
    :param data_size: Size of the data we wish to sample from
    :param batch_size: Size of the batch we wish to sample
    :param num_batches: Number of batches to sample
    :return: Array of batch indices of size batch_size x num_batches
    """
    return permutation(
        key=random_key,
        x=jnp.tile(jnp.arange(data_size, dtype=jnp.int32), (num_batches, 1) ).T,
        axis=0,
        independent=True
    )[:batch_size, :]

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
