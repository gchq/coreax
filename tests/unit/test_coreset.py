"""Tests for coreset data-structures."""

from unittest.mock import MagicMock

import equinox as eqx
import jax.numpy as jnp
import pytest

from coreax.coreset import Coreset, Coresubset
from coreax.data import Data, SupervisedData
from coreax.metrics import Metric
from coreax.weights import WeightsOptimiser

DATA = Data(jnp.arange(5, dtype=jnp.int32)[..., None])
SUPERVISED_DATA = SupervisedData(
    jnp.arange(5, dtype=jnp.int32)[..., None], jnp.arange(5, dtype=jnp.int32)[..., None]
)
PRE_CORESET_DATA = Data(jnp.arange(10)[..., None])


@pytest.mark.parametrize("coreset_type", [Coreset, Coresubset])
@pytest.mark.parametrize("data", [DATA, SUPERVISED_DATA])
class TestCoresetCommon:
    """Common tests for `coreax.coreset.Coreset` and `coreax.coreset.Coresubset`."""

    def test_init_array_conversion(self, coreset_type, data):
        """
        Test the initialisation behaviour.

        The nodes can be passed as an 'Array' or as a 'Data' instance. In the former
        case, we expect this array to be automatically converted to a 'Data' instance.
        """
        array_nodes = data.data
        data_obj = Data(data.data, data.weights)
        coreset_array_nodes = coreset_type(array_nodes, PRE_CORESET_DATA)
        coreset_data_nodes = coreset_type(data_obj, PRE_CORESET_DATA)
        assert coreset_array_nodes == coreset_data_nodes

    def test_materialization(self, coreset_type, data):
        """Test the coreset materialisation behaviour."""
        coreset = coreset_type(data, PRE_CORESET_DATA)
        expected_materialization = coreset.nodes
        if isinstance(coreset, Coresubset):
            materialized_nodes = PRE_CORESET_DATA.data[data.data.squeeze()]
            expected_materialization = Data(materialized_nodes)
        assert expected_materialization == coreset.coreset

    def test_len(self, coreset_type, data):
        """Test the coreset length."""
        coreset = coreset_type(data, PRE_CORESET_DATA)
        assert len(coreset) == len(data.data)

    def test_solve_weights(self, coreset_type, data):
        """Test the weights solving convenience interface."""
        solver = MagicMock(WeightsOptimiser)
        solved_weights = jnp.full_like(jnp.asarray(data), 123)
        solver.solve.return_value = solved_weights
        re_weighted_nodes = eqx.tree_at(lambda x: x.weights, data, solved_weights)
        coreset = coreset_type(data, PRE_CORESET_DATA)
        coreset_expected = coreset_type(re_weighted_nodes, PRE_CORESET_DATA)
        kwargs = {"test": None}
        coreset_solved_weights = coreset.solve_weights(solver, **kwargs)
        assert eqx.tree_equal(coreset_solved_weights, coreset_expected)
        solver.solve.assert_called_with(
            coreset.pre_coreset_data, coreset.coreset, **kwargs
        )

    def test_compute_metric(self, coreset_type, data):
        """Test the metric computation convenience interface."""
        metric = MagicMock(Metric)
        expected_metric = jnp.asarray(123)
        metric.compute.return_value = expected_metric
        coreset = coreset_type(data, PRE_CORESET_DATA)
        kwargs = {"test": None}
        coreset_metric = coreset.compute_metric(metric, **kwargs)
        assert eqx.tree_equal(coreset_metric, expected_metric)
        metric.compute.assert_called_with(
            coreset.pre_coreset_data, coreset.coreset, **kwargs
        )


class TestCoresubset:
    """Tests specific to `coreax.coreset.Coresubset`."""

    def test_unweighted_indices(self):
        """Test the coresubset 'unweighted_indices' property."""
        coresubset = Coresubset(DATA, PRE_CORESET_DATA)
        expected_indices = DATA.data.squeeze()
        assert eqx.tree_equal(expected_indices, coresubset.unweighted_indices)
