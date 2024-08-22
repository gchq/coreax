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

from functools import partial
from typing import Union

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from jax import Array
from jaxtyping import Shaped

import coreax.data

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
    "array",
    [
        (1,),
        (1.0,),
        (jnp.array(1),),
        (jnp.array([1, 1]),),
        (jnp.array([[1], [1]]),),
        (jnp.array([[[1]], [[1]]]),),
    ],
    ids=[
        "int",
        "float",
        "zero_dimensional_array",
        "one_dimensional_array",
        "two_dimensional_array",
        "three_dimensional_array",
    ],
)
def test_atleast_2d_consistent(
    array: Union[
        Shaped[Array, " n p"],
        Shaped[Array, " n"],
        Shaped[Array, ""],
        Union[float, int],
    ],
) -> None:
    """Check ``atleast_2d_consistent`` returns arrays with expected dimension."""
    min_dimension = 2

    # pylint: disable=protected-access
    arrays_atleast_2d = coreax.data._atleast_2d_consistent(array)
    # pylint: enable=protected-access

    array = jnp.asarray(array)
    array_shape = array.shape
    if len(array_shape) <= min_dimension:
        assert len(arrays_atleast_2d.shape) == min_dimension
    else:
        assert arrays_atleast_2d.shape == array_shape


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
            invalid_supervision = jnp.ones(DATA_ARRAY.shape[0] + 1)
            coreax.data.SupervisedData(DATA_ARRAY, invalid_supervision)
