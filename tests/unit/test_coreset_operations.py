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

"""Tests for coreset-oriented metric and weight optimisation interfaces."""

from collections.abc import Callable
from unittest.mock import Mock

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from coreax.coreset import AbstractCoreset, Coresubset, PseudoCoreset
from coreax.data import Data, SupervisedData
from coreax.kernels import SquaredExponentialKernel
from coreax.metrics import MMD, Metric
from coreax.weights import SBQWeightsOptimiser, WeightsOptimiser


class LabelledCoresubset(Coresubset):
    """A custom coresubset retaining extra metadata when re-weighted."""

    label: str = "test"


@pytest.fixture(name="coreset", params=["subset", "pseudo", "supervised", "labelled"])
def sample_coreset(request: pytest.FixtureRequest) -> AbstractCoreset:
    """Build weighted coresets, including repeated subset indices."""
    data = Data(jnp.array([[0.0], [1.0], [2.0]]), jnp.array([0.2, 0.3, 0.5]))
    indices = Data(jnp.array([2, 0, 2]), jnp.array([0.1, 0.2, 0.7]))
    if request.param == "pseudo":
        return PseudoCoreset(Data(jnp.array([[0.25], [1.5]])), data)
    if request.param == "supervised":
        data = SupervisedData(data.data, data.data * 2, data.weights)
    if request.param == "labelled":
        return LabelledCoresubset(indices, data)
    return Coresubset(indices, data)


def test_metric_forwards_data_and_options(coreset: AbstractCoreset) -> None:
    """Forward both datasets, in order, without dropping metric options."""
    metric = Mock(spec=Metric)
    metric.compute.return_value = jnp.asarray(0.125)
    result = Metric.compute_on_coreset(metric, coreset, block_size=2, custom=True)
    metric.compute.assert_called_once_with(
        coreset.pre_coreset_data, coreset.points, block_size=2, custom=True
    )
    np.testing.assert_array_equal(result, 0.125)


def test_optimiser_forwards_data_and_options(coreset: AbstractCoreset) -> None:
    """Return a re-weighted copy while retaining all original data and metadata."""
    optimiser = Mock(spec=WeightsOptimiser)
    weights = jnp.arange(len(coreset), dtype=jnp.float32) + 1
    optimiser.solve.return_value = weights
    original_points = coreset.points
    result = WeightsOptimiser.solve_on_coreset(
        optimiser, coreset, epsilon=0.1, block_size=2
    )
    optimiser.solve.assert_called_once_with(
        coreset.pre_coreset_data, original_points, epsilon=0.1, block_size=2
    )
    assert type(result) is type(coreset)
    assert result is not coreset
    assert eqx.tree_equal(result.pre_coreset_data, coreset.pre_coreset_data)
    assert eqx.tree_equal(coreset.points, original_points)
    np.testing.assert_array_equal(result.points.weights, weights)
    np.testing.assert_array_equal(result.points.data, original_points.data)
    if isinstance(coreset, Coresubset):
        assert isinstance(result, Coresubset)
        np.testing.assert_array_equal(
            result.unweighted_indices, coreset.unweighted_indices
        )
    if isinstance(coreset.points, SupervisedData):
        np.testing.assert_array_equal(
            result.points.supervision, original_points.supervision
        )
    if isinstance(coreset, LabelledCoresubset):
        assert isinstance(result, LabelledCoresubset)
        assert result.label == coreset.label


class TestCompiledOperations:
    """Exercise real metrics, solvers and immutable weight updates under JIT."""

    def test_with_weights(
        self, coreset: AbstractCoreset, jit_variant: Callable[[Callable], Callable]
    ) -> None:
        """Update only the coreset weights with either execution mode."""
        weights = jnp.linspace(0.1, 0.9, len(coreset))
        result = jit_variant(lambda c, w: c.with_weights(w))(coreset, weights)
        np.testing.assert_allclose(result.points.weights, weights)
        np.testing.assert_array_equal(result.points.data, coreset.points.data)
        assert eqx.tree_equal(result.pre_coreset_data, coreset.pre_coreset_data)

    @pytest.mark.parametrize("kind", [Coresubset, PseudoCoreset])
    def test_metric_and_weight_solver(
        self,
        kind: type[Coresubset] | type[PseudoCoreset],
        jit_variant: Callable[[Callable], Callable],
    ) -> None:
        """Match the underlying numeric APIs for both representations."""
        data = Data(jnp.array([[0.0], [1.0], [2.0]]))
        nodes = (
            Data(jnp.array([0, 2])) if kind is Coresubset else data[jnp.array([0, 2])]
        )
        coreset = kind(nodes, data)
        kernel = SquaredExponentialKernel()
        metric = MMD(kernel)
        optimiser = SBQWeightsOptimiser(kernel)
        expected_weights = optimiser.solve(data, coreset.points, epsilon=1e-4)
        result = jit_variant(optimiser.solve_on_coreset)(coreset, epsilon=1e-4)
        np.testing.assert_allclose(result.points.weights, expected_weights, atol=1e-6)
        actual = jit_variant(metric.compute_on_coreset)(result, block_size=2)
        expected = metric.compute(data, result.points, block_size=2)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
