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
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import coreax.data
import coreax.reduction


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
