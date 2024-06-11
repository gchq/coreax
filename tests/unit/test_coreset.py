"""Tests for coreset data-structures."""

from unittest.mock import MagicMock

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

from coreax.coreset import Coreset, Coresubset
from coreax.data import Data
from coreax.metrics import Metric
from coreax.weights import WeightsOptimiser

NODES = Data(jnp.arange(5, dtype=jnp.int32)[..., None])
PRE_CORESET_DATA = Data(jnp.arange(10)[..., None])


@pytest.mark.parametrize("coreset_type", [Coreset, Coresubset])
class TestCoresetCommon:
    """Common tests for `coreax.coreset.Coreset` and `coreax.coreset.Coresubset`."""

    def test_init_array_conversion(self, coreset_type):
        """
        Test the initialisation behaviour.

        The nodes can be passed as an 'Array' or as a 'Data' instance. In the former
        case, we expect this array to be automatically converted to a 'Data' instance.
        """
        array_nodes = NODES.data
        coreset_array_nodes = coreset_type(array_nodes, PRE_CORESET_DATA)
        coreset_data_nodes = coreset_type(NODES, PRE_CORESET_DATA)
        assert coreset_array_nodes == coreset_data_nodes

    def test_materialization(self, coreset_type):
        """Test the coreset materialisation behaviour."""
        coreset = coreset_type(NODES, PRE_CORESET_DATA)
        expected_materialization = coreset.nodes
        if isinstance(coreset, Coresubset):
            materialized_nodes = PRE_CORESET_DATA.data[NODES.data.squeeze()]
            expected_materialization = Data(materialized_nodes)
        assert expected_materialization == coreset.coreset

    def test_len(self, coreset_type):
        """Test the coreset length."""
        coreset = coreset_type(NODES, PRE_CORESET_DATA)
        assert len(coreset) == len(NODES.data)

    def test_solve_weights(self, coreset_type):
        """Test the weights solving convenience interface."""
        solver = MagicMock(WeightsOptimiser)
        solved_weights = jnp.full_like(jnp.asarray(NODES), 123)
        solver.solve.return_value = solved_weights
        re_weighted_nodes = eqx.tree_at(lambda x: x.weights, NODES, solved_weights)
        coreset = coreset_type(NODES, PRE_CORESET_DATA)
        coreset_expected = coreset_type(re_weighted_nodes, PRE_CORESET_DATA)
        kwargs = {"test": None}
        coreset_solved_weights = coreset.solve_weights(solver, **kwargs)
        assert eqx.tree_equal(coreset_solved_weights, coreset_expected)
        solver.solve.assert_called_with(
            coreset.pre_coreset_data, coreset.coreset, **kwargs
        )

    def test_compute_metric(self, coreset_type):
        """Test the metric computation convenience interface."""
        metric = MagicMock(Metric)
        expected_metric = jnp.asarray(123)
        metric.compute.return_value = expected_metric
        coreset = coreset_type(NODES, PRE_CORESET_DATA)
        kwargs = {"test": None}
        coreset_metric = coreset.compute_metric(metric, **kwargs)
        assert eqx.tree_equal(coreset_metric, expected_metric)
        # pylint: disable=no-member
        metric.compute.assert_called_with(
            coreset.pre_coreset_data, coreset.coreset, **kwargs
        )
        # pylint: enable=no-member


class TestCoresubset:
    """Tests specific to `coreax.coreset.Coresubset`."""

    def test_unweighted_indices(self):
        """Test the coresubset 'unweighted_indices' property."""
        coresubset = Coresubset(NODES, PRE_CORESET_DATA)
        expected_indices = NODES.data.squeeze()
        assert eqx.tree_equal(expected_indices, coresubset.unweighted_indices)

    def test_reverse(self):
        """Test the coresubset 'unweighted_indices' is reversed by 'reverse' method."""
        coresubset = Coresubset(NODES, PRE_CORESET_DATA)
        expected_indices = jnp.flip(NODES.data.squeeze())
        assert eqx.tree_equal(expected_indices, coresubset.reverse().unweighted_indices)

    def test_permute(self):
        """Test the coresubset 'unweighted_indices' is permuted by 'permuted' method."""
        random_key = jr.key(2_024)
        coresubset = Coresubset(NODES, PRE_CORESET_DATA)
        expected_indices = jr.permutation(random_key, NODES.data.squeeze())
        assert eqx.tree_equal(
            expected_indices, coresubset.permute(random_key).unweighted_indices
        )
