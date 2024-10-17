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

import logging
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from functools import partial, wraps
from math import log10
from typing import (
    Any,
    Dict,
    Generic,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import Array, block_until_ready, jit, vmap
from jax.typing import ArrayLike
from jaxtyping import Shaped
from typing_extensions import TypeAlias, deprecated

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

PyTreeDef: TypeAlias = Any
Leaf: TypeAlias = Any

#: JAX random key type annotations.
KeyArray: TypeAlias = Array
KeyArrayLike: TypeAlias = ArrayLike


class NotCalculatedError(Exception):
    """Raise when trying to use a variable that has not been calculated yet."""


class JITCompilableFunction(NamedTuple):
    """
    Parameters for :func:`jit_test`.

    :param fn: JIT-compilable function callable to test
    :param fn_args: Arguments passed during the calls to the passed function
    :param fn_kwargs: Keyword arguments passed during the calls to the passed function
    :param jit_kwargs: Keyword arguments that are partially applied to :func:`jax.jit`
        before being called to compile the passed function
    """

    fn: Callable
    fn_args: tuple = ()
    fn_kwargs: Optional[Dict[str, Any]] = None
    jit_kwargs: Optional[Dict[str, Any]] = None
    name: Optional[str] = None

    def without_name(
        self,
    ) -> Tuple[Callable, Tuple, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Return the tuple (fn, fn_args, fn_kwargs, jit_kwargs)."""
        return self.fn, self.fn_args, self.fn_kwargs, self.jit_kwargs


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


def tree_zero_pad_leading_axis(tree: PyTreeDef, pad_width: int) -> PyTreeDef:
    """
    Pad each array leaf of 'tree' with 'pad_width' trailing zeros.

    :param tree: The PyTree whose array leaves to pad with trailing zeros
    :param pad_width: The number of trailing zeros to pad with
    :return: A copy of the original PyTree with the array leaves padded
    """
    if int(pad_width) < 0:
        raise ValueError("'pad_width' must be a positive integer")
    leaves_to_pad, leaves_to_keep = eqx.partition(tree, eqx.is_array)

    def _pad(x: Shaped[Array, " n"]) -> Shaped[Array, " n + pad_width"]:
        padding = (0, int(pad_width))
        skip_padding = ((0, 0),) * (jnp.ndim(x) - 1)
        return jnp.pad(x, (padding, *skip_padding))

    padded_leaves = jtu.tree_map(_pad, leaves_to_pad)
    return eqx.combine(padded_leaves, leaves_to_keep)


def apply_negative_precision_threshold(
    x: Union[Shaped[Array, ""], float, int], precision_threshold: float = 1e-8
) -> Shaped[Array, ""]:
    """
    Round a number to 0.0 if it is negative but within precision_threshold of 0.0.

    :param x: Scalar value we wish to compare to 0.0
    :param precision_threshold: Positive threshold we compare against for precision
    :return: ``x``, rounded to 0.0 if it is between ``-precision_threshold`` and 0.0
    """
    _x = jnp.asarray(x)
    return jnp.where((-jnp.abs(precision_threshold) < _x) & (_x < 0.0), 0.0, _x)


def pairwise(
    fn: Callable[
        [
            Union[Shaped[Array, " d"], Shaped[Array, ""], float, int],
            Union[Shaped[Array, " d"], Shaped[Array, ""], float, int],
        ],
        Shaped[Array, " *d"],
    ],
) -> Callable[
    [
        Union[
            Shaped[Array, " n d"], Shaped[Array, " d"], Shaped[Array, ""], float, int
        ],
        Union[
            Shaped[Array, " m d"], Shaped[Array, " d"], Shaped[Array, ""], float, int
        ],
    ],
    Shaped[Array, " n m *d"],
]:
    """
    Transform a function so it returns all pairwise evaluations of its inputs.

    :param fn: the function to apply the pairwise transform to.
    :returns: function that returns an array whose entries are the evaluations of `fn`
        for every pairwise combination of its input arguments.
    """

    @wraps(fn)
    def pairwise_fn(
        x: Union[
            Shaped[Array, " n d"], Shaped[Array, " d"], Shaped[Array, ""], float, int
        ],
        y: Union[
            Shaped[Array, " m d"], Shaped[Array, " d"], Shaped[Array, ""], float, int
        ],
    ) -> Shaped[Array, " n m *d"]:
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        return vmap(
            vmap(fn, in_axes=(0, None), out_axes=0),
            in_axes=(None, 0),
            out_axes=1,
        )(x, y)

    return pairwise_fn


@jit
def squared_distance(
    x: Union[Shaped[Array, " d"], Shaped[Array, ""], float, int],
    y: Union[Shaped[Array, " d"], Shaped[Array, ""], float, int],
) -> Shaped[Array, ""]:
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
def difference(
    x: Union[Shaped[Array, " d"], Shaped[Array, ""], float, int],
    y: Union[Shaped[Array, " d"], Shaped[Array, ""], float, int],
) -> Shaped[Array, ""]:
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
    "Use coreax.kernels.util.median_heuristic instead."
    + " Deprecated since version 0.3.0."
    + " Will be removed in version 0.4.0."
)
@jit
def median_heuristic(
    x: Union[Shaped[Array, " n d"], Shaped[Array, " n"], Shaped[Array, ""], float, int],
) -> Shaped[Array, ""]:
    """
    Compute the median heuristic for setting kernel bandwidth.

    Analysis of the performance of the median heuristic can be found in
    :cite:`garreau2018median`.

    :param x: Input array of vectors
    :return: Bandwidth parameter, computed from the median heuristic, as a
        zero-dimensional array
    """
    # Format inputs
    x = jnp.atleast_2d(x)
    # Calculate square distances as an upper triangular matrix
    square_distances = jnp.triu(pairwise(squared_distance)(x, x), k=1)
    # Calculate the median of the square distances
    median_square_distance = jnp.median(
        square_distances[jnp.triu_indices_from(square_distances, k=1)]
    )

    return jnp.sqrt(median_square_distance / 2.0)


def sample_batch_indices(
    random_key: KeyArrayLike,
    max_index: int,
    batch_size: int,
    num_batches: int,
) -> Shaped[Array, " num_batches batch_size"]:
    """
    Sample an array of indices of size `num_batches` x `batch_size`.

    Each row (batch) of the sampled array will contain unique elements.

    :param random_key: Key for random number generation
    :param max_index: Largest index we wish to sample
    :param batch_size: Size of the batch we wish to sample
    :param num_batches: Number of batches to sample

    :return: Array of batch indices of size `num_batches` x `batch_size`
    """
    if max_index < batch_size:
        raise ValueError("'max_index' must be greater than or equal to 'batch_size'")
    if batch_size < 0.0:
        raise ValueError("'batch_size' must be non-negative")

    batch_keys = jr.split(random_key, num_batches)
    batch_permutation = vmap(jr.permutation, in_axes=(0, None))
    return batch_permutation(batch_keys, max_index)[:, :batch_size]


def jit_test(
    fn: Callable,
    fn_args: tuple = (),
    fn_kwargs: Optional[dict] = None,
    jit_kwargs: Optional[dict] = None,
) -> tuple[float, float]:
    """
    Measure execution times of two runs of a JIT-compilable function.

    The function is called with supplied arguments twice, and timed for each run. These
    timings are returned in a 2-tuple. These timings can help verify the JIT performance
    by comparing timings of a before and after run of a function.

    :param fn: JIT-compilable function callable to test
    :param fn_args: Arguments passed during the calls to the passed function
    :param fn_kwargs: Keyword arguments passed during the calls to the passed function
    :param jit_kwargs: Keyword arguments that are partially applied to :func:`jax.jit`
        before being called to compile the passed function
    :return: (First run time, Second run time), in seconds
    """
    # Avoid dangerous default values - Pylint W0102
    if fn_kwargs is None:
        fn_kwargs = {}
    if jit_kwargs is None:
        jit_kwargs = {}

    @partial(jit, **jit_kwargs)
    def _fn(*args, **kwargs):
        return fn(*args, **kwargs)

    start_time = time.perf_counter()
    block_until_ready(_fn(*fn_args, **fn_kwargs))
    end_time = time.perf_counter()
    pre_delta = end_time - start_time

    start_time = time.perf_counter()
    block_until_ready(_fn(*fn_args, **fn_kwargs))
    end_time = time.perf_counter()
    post_delta = end_time - start_time

    return pre_delta, post_delta


def format_time(num: float) -> str:
    """
    Standardise the format of the input time.

    Floats will be converted to a standard format, e.g. 0.4531 -> "453.1 ms".

    :param num: Float to be converted
    :return: Formatted time as a string
    """
    try:
        order = log10(abs(num))
    except ValueError:
        return "0 s"

    if order >= 2:  # noqa: PLR2004
        scaled_time = num / 60
        unit_string = "mins"
    elif order < -9:  # noqa: PLR2004
        scaled_time = 1e12 * num
        unit_string = "ps"
    elif order < -6:  # noqa: PLR2004
        scaled_time = 1e9 * num
        unit_string = "ns"
    elif order < -3:  # noqa: PLR2004
        scaled_time = 1e6 * num
        unit_string = "\u03bcs"
    elif order < 0:  # noqa: PLR2004
        scaled_time = 1e3 * num
        unit_string = "ms"
    else:
        scaled_time = num
        unit_string = "s"

    return f"{round(scaled_time, 2)} {unit_string}"


def speed_comparison_test(
    function_setups: Sequence[JITCompilableFunction],
    num_runs: int = 10,
    log_results: bool = False,
    normalisation: Optional[Tuple[float, float]] = None,
) -> tuple[list[tuple[Array, Array]], dict[str, Array]]:
    """
    Compare compilation time and runtime of a list of JIT-able functions.

    :param function_setups: Sequence of instances of :class:`JITCompilableFunction`
    :param num_runs: Number of times to average function timings over
    :param log_results: If :data:`True`, the results are formatted and logged
    :param normalisation: Tuple (compilation normalisation, execution normalisation).
        If provided, returned compilation/execution times are normalised so that this
        time is 1 time unit.
    :return: List of tuples (means, standard deviations) for each function containing
        JIT compilation and execution times as array components; Dictionary with
        key function name and value array of estimated compilation times in first
        column and execution time in second column
    """
    timings_dict = {}
    results = []
    for i, function in enumerate(function_setups):
        name = function.name
        name = name if name is not None else f"function_{i + 1}"
        if log_results:
            _logger.info("------------------- %s -------------------", name)
        timings = jnp.zeros((num_runs, 2))
        for j in range(num_runs):
            timings = timings.at[j, :].set(jit_test(*function.without_name()))
        # Compute the time just spent on compilation
        timings = timings.at[:, 0].set(timings[:, 0] - timings[:, 1])
        # Normalise, if necessary
        if normalisation is not None:
            timings = timings.at[:, 0].set(timings[:, 0] / normalisation[0])
            timings = timings.at[:, 1].set(timings[:, 1] / normalisation[1])
        timings_dict[name] = timings
        # Compute summary statistics
        mean = timings.mean(axis=0)
        std = timings.std(axis=0)
        results.append((mean, std))

        if log_results:
            if normalisation:
                _logger.info(
                    "Compilation time: %.4g units ± %.4g units per run "
                    "(mean ± std. dev. of %s runs)",
                    mean[0].item(),
                    std[0].item(),
                    num_runs,
                )
                _logger.info(
                    "Execution time: %.4g units ± %.4g units per run "
                    "(mean ± std. dev. of %s runs)",
                    mean[1].item(),
                    std[1].item(),
                    num_runs,
                )
            else:
                _logger.info(
                    "Compilation time: %s ± %s per run "
                    "(mean ± std. dev. of %s runs)",
                    format_time(mean[0].item()),
                    format_time(std[0].item()),
                    num_runs,
                )
                _logger.info(
                    "Execution time: %s ± %s per run (mean ± std. dev. of %s runs)",
                    format_time(mean[1].item()),
                    format_time(std[1].item()),
                    num_runs,
                )

    return results, timings_dict


T = TypeVar("T")


class SilentTQDM(Generic[T]):
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

    @staticmethod
    def write(*_args, **_kwargs) -> None:
        """Do nothing instead of writing to output."""
