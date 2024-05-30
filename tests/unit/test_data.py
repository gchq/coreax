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
Tests for data classes and functionality.

The tests within this file verify that approaches to handling and processing data
produce the expected results on simple examples.
"""

import unittest
from functools import partial
from unittest.mock import MagicMock

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike

import coreax.data
import coreax.reduction

DATA_ARRAY = jnp.array([[1], [2], [3]])
SUPERVISION = jnp.array([[4], [5], [6]])


def test_as_data():
    """Test functionality of `as_data` converter method."""
    _array = jnp.array([1, 2, 3])
    _data = coreax.data.Data(_array)
    assert eqx.tree_equal(coreax.data.as_data(_array), _data)


def test_is_data():
    """Test functionality of `is_data` filter method."""
    assert not coreax.data.is_data(123)
    assert coreax.data.is_data(coreax.data.Data(1))


@pytest.mark.parametrize(
    "data_type",
    [
        partial(coreax.data.Data, DATA_ARRAY),
        partial(coreax.data.SupervisedData, DATA_ARRAY, SUPERVISION),
    ],
)
class TestData:
    """Test operation of Data class."""

    @pytest.mark.parametrize(
        "weights, expected_weights",
        (
            (None, jnp.broadcast_to(1, DATA_ARRAY.shape[0])),
            (3, jnp.broadcast_to(3, DATA_ARRAY.shape[0])),
            (DATA_ARRAY.reshape(-1), DATA_ARRAY.reshape(-1)),
        ),
    )
    def test_weights(self, data_type, weights, expected_weights):
        """Test that if no weights are given a uniform weight vector is made."""
        _data = data_type(weights)
        assert eqx.tree_equal(_data.weights, expected_weights)

    def test_invalid_weight_dimensions(self, data_type):
        """Test that __init__ raises expected errors."""
        with pytest.raises(ValueError, match="Incompatible shapes for broadcasting"):
            invalid_weights = jnp.ones(DATA_ARRAY.shape[0] + 1)
            data_type(weights=invalid_weights)

    @pytest.mark.parametrize("index", (0, -1, slice(0, DATA_ARRAY.shape[0])))
    def test_getitem(self, data_type, index):
        """Test indexing data as a JAX array."""
        _data = data_type()
        _expected_indexed_data = jtu.tree_map(lambda x: x[index], _data)
        assert eqx.tree_equal(_data[index], _expected_indexed_data)

    def test_arraylike(self, data_type):
        """Test interpreting data as a JAX array."""
        _data = data_type()
        assert eqx.tree_equal(jnp.asarray(_data), _data.data)

    def test_len(self, data_type):
        """Test length of data."""
        _data = data_type()
        assert len(_data) == len(_data.data)

    @pytest.mark.parametrize("weights", (None, 0, 3, DATA_ARRAY.reshape(-1)))
    def test_normalize(self, data_type, weights):
        """Test weight normalization."""
        data = data_type(weights)
        expected_weights = data.weights / jnp.sum(data.weights)
        if jnp.all(weights != 0):
            normalized_data = data.normalize()
            assert eqx.tree_equal(normalized_data.weights, expected_weights)
        normalized_with_zeros = data.normalize(preserve_zeros=True)
        assert eqx.tree_equal(
            normalized_with_zeros.weights, jnp.nan_to_num(expected_weights)
        )


class TestSupervisedData:
    """Test operation of SupervisedData class."""

    def test_invalid_supervision_dimensions(self):
        """
        Test that __check_init__ raises ValueError when supervision has bad dimension.
        """
        with pytest.raises(
            ValueError,
            match="Leading dimensions of 'supervision' and 'data' must be equal",
        ):
            invalid_supervision = jnp.ones(DATA_ARRAY.shape[0])
            coreax.data.SupervisedData(DATA_ARRAY, invalid_supervision)


class DataReaderConcrete(coreax.data.DataReader):
    """Concrete implementation of DataReader class to allow testing."""

    @classmethod
    def load(cls, original_data: ArrayLike) -> coreax.data.DataReader:
        raise NotImplementedError

    def format(self, coreset: coreax.reduction.Coreset) -> Array:
        raise NotImplementedError


class TestDataReader(unittest.TestCase):
    """Test operation of DataReader class."""

    def test_init_scalars(self):
        """Test that scalars are cast properly."""
        actual = DataReaderConcrete(original_data=1, pre_coreset_array=2)
        self.assertEqual(actual.original_data, jnp.array(1))
        self.assertEqual(actual.pre_coreset_array, jnp.array([[2]]))

    def test_non_abstract_methods_raise(self):
        """
        Test calls to the non-abstract methods of DataReader raise the expected errors.
        """
        reader_object = DataReaderConcrete(original_data=1, pre_coreset_array=2)
        self.assertRaises(NotImplementedError, reader_object.render, MagicMock())
        self.assertRaises(NotImplementedError, reader_object.reduce_dimension, 2)
        self.assertRaises(
            NotImplementedError, reader_object.restore_dimension, MagicMock()
        )


class TestArrayData(unittest.TestCase):
    """Test ArrayData class."""

    def test_load(self):
        """Check that no preprocessing is done during load."""
        original_data = jnp.array([[1, 2]])
        actual = coreax.data.ArrayData.load(original_data)
        np.testing.assert_array_equal(actual.original_data, original_data)
        np.testing.assert_array_equal(actual.pre_coreset_array, original_data)

    def test_format(self):
        """Check that coreset is returned without further formatting."""
        original_data = jnp.array([[2, 3], [4, 5], [6, 7]])
        coreset_array = jnp.array([[2, 3], [4, 5]])
        coreset_obj = MagicMock()
        coreset_obj.coreset = coreset_array
        data_reader = coreax.data.ArrayData(original_data, original_data)
        np.testing.assert_array_equal(data_reader.format(coreset_obj), coreset_array)


if __name__ == "__main__":
    unittest.main()
