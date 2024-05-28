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
from typing import TypeVar

import jax.numpy as jnp
from jax import Array, block_until_ready, jit, vmap
from jax.random import normal, permutation
from jax.typing import ArrayLike
from jaxopt import OSQP
from typing_extensions import TypeAlias, deprecated

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


def sample_batch_indices(
    random_key: KeyArrayLike,
    data_size: int,
    batch_size: int,
    num_batches: int,
) -> ArrayLike:
    """
    Sample an array of indices of size batch_size x num_batches.

    Each column of the sampled array will contain unique elements. The largest possible
    index is dictated by data_size.

    :param random_key: Key for random number generation
    :param data_size: Size of the data we wish to sample from
    :param batch_size: Size of the batch we wish to sample
    :param num_batches: Number of batches to sample
    :return: Array of batch indices of size batch_size x num_batches
    """
    if data_size < batch_size:
        raise ValueError("data_size must be greater than or equal to batch_size")
    if (data_size <= 0.0) or not isinstance(data_size, int):
        raise ValueError("data_size must be a positive integer")
    if (batch_size <= 0.0) or not isinstance(batch_size, int):
        raise ValueError("batch_size must be a positive integer")
    if (num_batches <= 0.0) or not isinstance(num_batches, int):
        raise ValueError("num_batches must be a positive integer")

    return permutation(
        key=random_key,
        x=jnp.tile(jnp.arange(data_size, dtype=jnp.int32), (num_batches, 1)).T,
        axis=0,
        independent=True,
    )[:batch_size, :]


@partial(jit, static_argnames="rcond")
def invert_regularised_array(
    array: ArrayLike,
    regularisation_parameter: float,
    identity: ArrayLike,
    rcond: float | None = None,
) -> ArrayLike:
    """
    Regularise an array and then invert it using a least-squares solver.

    .. note::
        The function is designed to invert square block arrays where only the top-left
        block contains non-zero elements. We return a block array, the same size as the
        input array, where each block has only zero elements except for the top-left
        block, which is the inverse of the non-zero input block. The most efficient way
        to compute this in JAX requires the 'identity' array to be a matrix of zeros
        except for ones on the diagonal up to the size of the non-zero block.

    :param array: Array to be inverted
    :param regularisation_parameter: Regularisation parameter for stable inversion of
        array, negative values will be converted to positive
    :param identity: Block "identity" matrix
    :param rcond: Cut-off ratio for small singular values of a. For the purposes of rank
        determination, singular values are treated as zero if they are smaller than
        rcond times the largest singular value of a. The default value of None will use
        the machine precision multiplied by the largest dimension of the array.
        An alternate value of -1 will use machine precision.
    :return: Inverse of regularised array
    """
    if rcond is not None:
        if rcond < 0 and rcond != -1:
            raise ValueError("rcond must be non-negative, except for value of -1")
    if array.shape != identity.shape:
        raise ValueError("Leading dimensions of array and identity must match")

    return jnp.linalg.lstsq(
        array + abs(regularisation_parameter) * identity, identity, rcond=rcond
    )[0]


@partial(jit, static_argnames=("oversampling_parameter", "power_iterations"))
def randomised_eigendecomposition(
    random_key: KeyArrayLike,
    array: ArrayLike,
    oversampling_parameter: int = 10,
    power_iterations: int = 1,
):
    r"""
    Approximate the eigendecomposition of kernel gram matrices.

    Using (:cite:`halko2011randomness` Algorithm 4.4. and 5.3) we approximate the
    eigendecomposition of a kernel gram matrix. The parameters 'oversampling_parameter'
    and 'power_iterations' present a trade-off between speed and approximation quality.

    Given the gram matrix :math:`K \in \mathbb{R}^{n\times n} and
    :math:`r=`oversampling_parameter we return a diagonal array of eigenvalues
    :math:`\Lambda \in \mathbb{R}^{r \times r}` and a rectangular array of eigenvectors
    :math:`U\in\mathbb{R}^{n\times p}` such that we have :math:`K \approx U\Lambda U^T`.

    :param random_key: Key for random number generation
    :param array: Array to be decomposed
    :param oversampling_parameter: Number of random columns to sample; the larger the
        oversampling_parameter, the more accurate, but slower the method will be
    :param power_iterations: Number of power iterations to do; the larger the
        power_iterations, the more accurate, but slower the method will be
    :return: eigenvalues and eigenvectors that approximately decompose the target array
    """
    # Input handling
    supported_array_shape = 2
    if len(array.shape) != supported_array_shape:
        raise ValueError("array must be two-dimensional")
    if array.shape[0] != array.shape[1]:
        raise ValueError("array must be square")
    if (oversampling_parameter <= 0.0) or not isinstance(oversampling_parameter, int):
        raise ValueError("oversampling_parameter must be a positive integer")
    if (power_iterations <= 0.0) or not isinstance(power_iterations, int):
        raise ValueError("power_iterations must be a positive integer")

    standard_gaussian_draws = normal(
        random_key, shape=(array.shape[0], oversampling_parameter)
    )

    # QR decomposition to find orthonormal array with range approximating range of array
    approximate_range = array @ standard_gaussian_draws
    q, _ = jnp.linalg.qr(approximate_range)

    # Power iterations for improved accuracy
    for _ in range(power_iterations):
        approximate_range_ = array.T @ q
        q_, _ = jnp.linalg.qr(approximate_range_)
        approximate_range = array @ q_
        q, _ = jnp.linalg.qr(approximate_range)

    # Form the low rank array, compute its exact eigendecomposition and ortho-normalise
    # the eigenvectors.
    array_approximation = q.T @ array @ q
    approximate_eigenvalues, eigenvectors = jnp.linalg.eigh(array_approximation)
    approximate_eigenvectors = q @ eigenvectors

    return approximate_eigenvalues, approximate_eigenvectors


@partial(jit, static_argnames=("rcond", "oversampling_parameter", "power_iterations"))
def randomised_invert_regularised_array(
    random_key: KeyArrayLike,
    array: ArrayLike,
    regularisation_parameter: float,
    identity: ArrayLike,
    rcond: float | None = None,
    oversampling_parameter: int = 25,
    power_iterations: int = 1,
) -> tuple[Array]:
    """
    Invert a regularised kernel matrix using its randomised eigendecomposition.

    Using (:cite:`halko2011randomness` Algorithm 4.4. and 5.3) we regularise and then
    approximate the eigendecomposition of the input kernel matrix. The parameters
    'oversampling_parameter' and 'power_iterations' present a trade-off between speed
    and approximation quality.

    .. note::
        The function is designed to invert square block arrays where only the top-left
        block contains non-zero elements. We return a block array, the same size as the
        input array, where each block has only zero elements except for the top-left
        block, which is the inverse of the non-zero input block. The most efficient way
        to compute this in JAX requires the 'identity' array to be a matrix of zeros
        except for ones on the diagonal up to the size of the non-zero block.

    :param array: Array to be inverted
    :param regularisation_parameter: Regularisation parameter for stable inversion of
        array, negative values will be converted to positive
    :param identity: Block identity matrix
    :param rcond: Cut-off ratio for small singular values of a. For the purposes of rank
        determination, singular values are treated as zero if they are smaller than
        rcond times the largest singular value of a. The default value of None will use
        the machine precision multiplied by the largest dimension of the array.
        An alternate value of -1 will use machine precision.
    :param oversampling_parameter: Number of random columns to sample; the larger the
        oversampling_parameter, the more accurate, but slower the method will be
    :param power_iterations: Number of power iterations to do; the larger the
        power_iterations, the more accurate, but slower the method will be
    :return: eigenvalues and eigenvectors that approximately decompose the target array
    """
    # Input validation
    if rcond is not None:
        if rcond < 0 and rcond != -1:
            raise ValueError("rcond must be non-negative, except for value of -1")
    if array.shape != identity.shape:
        raise ValueError("Leading dimensions of array and identity must match")

    # Set rcond parameter if not given
    n, m = array.shape
    machine_precision = jnp.finfo(array.dtype).eps
    if rcond is None:
        rcond = machine_precision * max(n, m)
    elif rcond == -1:
        rcond = machine_precision

    # Get randomised eigendecomposition
    approximate_eigenvalues, approximate_eigenvectors = randomised_eigendecomposition(
        random_key=random_key,
        array=array + abs(regularisation_parameter) * identity,
        oversampling_parameter=oversampling_parameter,
        power_iterations=power_iterations,
    )

    # Mask the eigenvalues that are zero or almost zero according to value of rcond
    # for safe inversion.
    mask = approximate_eigenvalues >= jnp.array(rcond) * approximate_eigenvalues[-1]
    safe_approximate_eigenvalues = jnp.where(mask, approximate_eigenvalues, 1)

    # Invert the eigenvalues and extend array ready for broadcasting
    approximate_inverse_eigenvalues = jnp.where(
        mask, 1 / safe_approximate_eigenvalues, 0
    )[:, jnp.newaxis]

    # Solve Ax = I, x = A^-1 = UL^-1U^T
    return approximate_eigenvectors.dot(
        (approximate_inverse_eigenvalues * approximate_eigenvectors.T).dot(identity)
    )


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
