"""Tests for coreset data-structures."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from coreax.coreset import Coreset, Coresubset
from coreax.data import Data

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


class TestCoresubset:
    """Tests specific to `coreax.coreset.Coresubset`."""

    def test_unweighted_indices(self):
        """Test the coresubset 'unweighted_indices' property."""
        coresubset = Coresubset(NODES, PRE_CORESET_DATA)
        expected_indices = NODES.data.squeeze()
        assert eqx.tree_equal(expected_indices, coresubset.unweighted_indices)
