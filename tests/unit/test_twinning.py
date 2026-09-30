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

"""Check Data Twinning against explicit selections and its invariants."""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from coreax.data import Data, SupervisedData
from coreax.solvers import DataTwinning


@pytest.mark.parametrize(
    ("factor", "expected"), [(2, [0, 2, 4, 6]), (3, [0, 3, 6]), (8, [0]), (20, [0])]
)
def test_line_groups(
    jit_variant: Callable[[Callable], Callable], factor: int, expected: list[int]
) -> None:
    """Keep one point per consecutive neighbourhood, including a short last group."""
    data = Data(jnp.arange(8))
    result, state = jit_variant(DataTwinning(factor, start_index=0).reduce)(data)
    np.testing.assert_array_equal(result.unweighted_indices, expected)
    np.testing.assert_array_equal(result.points.data, np.arange(8)[expected, None])
    assert eqx.tree_equal(result.pre_coreset_data, data)
    assert state is None


def test_continue_from_farthest_neighbour(
    jit_variant: Callable[[Callable], Callable],
) -> None:
    """Continue from the farthest removed neighbour."""
    data = Data(
        jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, 3.0], [0.0, 4.0]]
        )
    )
    result, _ = jit_variant(DataTwinning(3, start_index=0).reduce)(data)
    np.testing.assert_array_equal(result.unweighted_indices, [0, 4])


def test_default_start(jit_variant: Callable[[Callable], Callable]) -> None:
    """Choose the point farthest from the standardised centroid by default."""
    data = Data(jnp.array([[0.0], [1.0], [10.0]]))
    result, _ = jit_variant(DataTwinning(3).reduce)(data)
    np.testing.assert_array_equal(result.unweighted_indices, [2])


def test_duplicate_points(jit_variant: Callable[[Callable], Callable]) -> None:
    """Resolve ties by original order without reusing any row."""
    result, _ = jit_variant(DataTwinning(3, start_index=0).reduce)(
        Data(jnp.ones((8, 2)))
    )
    np.testing.assert_array_equal(result.unweighted_indices, [0, 3, 6])


def test_supervised_selection(jit_variant: Callable[[Callable], Callable]) -> None:
    """Include the responses in distances while preserving all original row data."""
    data = SupervisedData(
        jnp.zeros((6, 1)), jnp.array([[0.0], [100.0], [1.0], [99.0], [2.0], [98.0]])
    )
    result, _ = jit_variant(DataTwinning(2, start_index=0).reduce)(data)
    np.testing.assert_array_equal(result.unweighted_indices, [0, 4, 3])
    assert isinstance(result.points, SupervisedData)
    np.testing.assert_array_equal(result.points.supervision, [[0.0], [2.0], [99.0]])
    assert eqx.tree_equal(result.pre_coreset_data, data)


def test_standardisation(jit_variant: Callable[[Callable], Callable]) -> None:
    """Independent changes of units and constant columns preserve the selected rows."""
    points = jnp.array(
        [[0.0, 2.0], [2.0, 7.0], [3.0, 1.0], [8.0, 4.0], [11.0, 10.0], [16.0, 20.0]]
    )
    transform = jnp.column_stack(
        (points * jnp.array([10.0, 0.5]) + jnp.array([20.0, 9.0]), jnp.ones(6))
    )
    solver = jit_variant(DataTwinning(2, start_index=0).reduce)
    original, _ = solver(Data(points))
    changed, _ = solver(Data(transform))
    np.testing.assert_array_equal(
        changed.unweighted_indices, original.unweighted_indices
    )


def test_identity_reduction(jit_variant: Callable[[Callable], Callable]) -> None:
    """A factor of one returns every row exactly once."""
    data = Data(jnp.array([[2.0], [5.0], [1.0], [10.0]]), weights=7)
    result, _ = jit_variant(DataTwinning(1).reduce)(data)
    np.testing.assert_array_equal(jnp.sort(result.unweighted_indices), jnp.arange(4))
    assert eqx.tree_equal(data, result.pre_coreset_data)


def test_reproducibility(jit_variant: Callable[[Callable], Callable]) -> None:
    """Repeated calls select identical indices without mutating their input."""
    data = Data(jnp.array([[2.0], [5.0], [1.0], [10.0], [12.0]]))
    solver = jit_variant(DataTwinning(2).reduce)
    first, _ = solver(data)
    second, _ = solver(data)
    assert eqx.tree_equal(first, second)
    np.testing.assert_array_equal(data.data, [[2.0], [5.0], [1.0], [10.0], [12.0]])


def test_single_row(jit_variant: Callable[[Callable], Callable]) -> None:
    """Handle zero variance with a single observation."""
    result, _ = jit_variant(DataTwinning().reduce)(Data(jnp.array([[5.0, 10.0]])))
    np.testing.assert_array_equal(result.unweighted_indices, [0])


@pytest.mark.parametrize(
    "weights", [[0.0, 0.0], [1.0, 2.0], [-1.0, -1.0], [np.nan, 1.0], [np.inf, np.inf]]
)
def test_invalid_weights(
    jit_variant: Callable[[Callable], Callable], weights: list[float]
) -> None:
    """Reject weighted measures for which the unweighted algorithm is not defined."""
    with pytest.raises((RuntimeError, ValueError), match="uniform positive weights"):
        result, _ = jit_variant(DataTwinning().reduce)(
            Data(jnp.array([[0.0], [1.0]]), jnp.array(weights))
        )
        result.unweighted_indices.block_until_ready()


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_non_finite_data(
    jit_variant: Callable[[Callable], Callable], value: float
) -> None:
    """Report invalid distances before selecting arbitrary indices."""
    with pytest.raises((RuntimeError, ValueError), match="finite data"):
        result, _ = jit_variant(DataTwinning().reduce)(
            Data(jnp.array([[0.0], [value]]))
        )
        result.unweighted_indices.block_until_ready()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thinning_factor": 0},
        {"thinning_factor": -1},
        {"thinning_factor": 1.5},
        {"thinning_factor": True},
        {"start_index": -1},
        {"start_index": 0.5},
    ],
)
def test_invalid_configuration(kwargs: dict[str, Any]) -> None:
    """Reject non-integer group sizes and invalid starting positions."""
    with pytest.raises(ValueError):
        DataTwinning(**kwargs)


def test_invalid_shape() -> None:
    """Reject empty arrays, complex data and out-of-bounds starting rows."""
    for points in (jnp.empty((0, 1)), jnp.ones((3, 0)), jnp.ones((3, 2, 2))):
        with pytest.raises(ValueError, match="non-empty two-dimensional"):
            DataTwinning().reduce(Data(points))
    with pytest.raises(ValueError, match="real data"):
        DataTwinning().reduce(Data(jnp.array([[1j], [2j]])))
    with pytest.raises(ValueError, match="smaller than"):
        DataTwinning(start_index=2).reduce(Data(jnp.zeros((2, 1))))
