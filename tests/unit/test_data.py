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

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest
from beartype.door import is_bearable
from jax import Array
from jaxtyping import Float, Int, Shaped

import coreax.data

DATA_ARRAY = jnp.array([[1], [2], [3]])
SUPERVISION = jnp.array([[4], [5], [6]])


def test_as_data():
    """Test functionality of `as_data` converter method."""
    _array = jnp.array([1, 2, 3])
    _data = coreax.data.Data(_array)
    assert eqx.tree_equal(coreax.data.as_data(_array), _data)


def test_as_supervised_data():
    """Test functionality of `as_supervised_data` converter method."""
    _array = jnp.array([1, 2, 3])
    _data = coreax.data.SupervisedData(_array, _array)
    assert eqx.tree_equal(coreax.data.as_supervised_data((_array, _array)), _data)


@pytest.mark.parametrize(
    "arrays",
    [
        (jnp.array(1),),
        (jnp.array(1), jnp.array(1)),
        (jnp.array([1, 1]),),
        (jnp.array([1, 1]), jnp.array([1, 1])),
        (jnp.array([[1], [1]]),),
        (jnp.array([[1], [1]]), jnp.array([[1], [1]])),
        (jnp.array([[[1]], [[1]]]),),
        (jnp.array([[[1]], [[1]]]), jnp.array([[[1]], [[1]]])),
        (1.0,),
        (1.0, 1.0),
        (1,),
        (1, 1),
    ],
    ids=[
        "single_zero_dimensional_array",
        "multiple_zero_dimensional_arrays",
        "single_one_dimensional_array",
        "multiple_one_dimensional_arrays",
        "single_two_dimensional_array",
        "multiple_two_dimensional_arrays",
        "single_three_dimensional_array",
        "multiple_three_dimensional_arrays",
        "single_float",
        "multiple_floats",
        "single_int",
        "multiple_ints",
    ],
)
def test_atleast_2d_consistent(arrays: tuple[Array]) -> None:
    """Check ``atleast_2d_consistent`` returns arrays with expected dimension."""
    min_dimension = 2
    num_arrays = len(arrays)

    # pylint: disable=protected-access
    arrays_atleast_2d = coreax.data._atleast_2d_consistent(*arrays)
    # pylint: enable=protected-access

    if num_arrays == 1:
        array = jnp.asarray(arrays[0])
        if len(array.shape) <= min_dimension:
            # Check we have expanded to two dimensions
            assert len(arrays_atleast_2d.shape) == min_dimension
        else:
            # Do nothing
            assert arrays_atleast_2d.shape == array.shape
    else:
        for i in range(num_arrays):
            array = jnp.asarray(arrays[i])
            if len(array.shape) <= min_dimension:
                assert len(arrays_atleast_2d[i].shape) == min_dimension
            else:
                assert arrays_atleast_2d[i].shape == array.shape


@pytest.mark.parametrize(
    "data_type",
    [
        pytest.param(jtu.Partial(coreax.data.Data, DATA_ARRAY), id="Data"),
        pytest.param(
            jtu.Partial(coreax.data.SupervisedData, DATA_ARRAY, SUPERVISION),
            id="SupervisedData",
        ),
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

    def test_asarray(self, data_type):
        """Test interpreting data as a JAX array."""
        _data = data_type()
        if isinstance(_data, coreax.data.SupervisedData):
            assert eqx.tree_equal(
                jnp.asarray(_data), jnp.hstack((_data.data, _data.supervision))
            )
        else:
            assert eqx.tree_equal(jnp.asarray(_data), _data.data)

    def test_len(self, data_type):
        """Test length of data."""
        _data = data_type()
        assert len(_data) == len(_data.data)

    def test_dtype(self, data_type):
        """Test dtype property; required for jaxtyping annotations."""
        _data = data_type()
        assert _data.data.dtype == _data.dtype

    def test_shape(self, data_type):
        """Test shape property; required for jaxtyping annotations."""
        _data = data_type()
        assert _data.data.shape == _data.shape

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

    @pytest.mark.parametrize(
        "dtype, valid_jax_type, invalid_jax_type",
        [(jnp.int32, Int, Float), (jnp.float32, Float, Int)],
    )
    def test_jaxtyping_compatibility(
        self, data_type, dtype, valid_jax_type, invalid_jax_type
    ):
        """
        Test `Data` compatibility with jaxtyping annotations.

        Checks the following cases:
            - Correct narrowed shape,
            - Correct narrowed shape and narrowed data type,
            - Correct narrowed shape and incorrect narrowed data type,
            - Incorrect narrowed shape
            - Incorrectly narrowed instance type
        """
        data_factory = eqx.tree_at(
            lambda x: x.args,
            data_type,
            replace=jtu.tree_map(lambda y: jnp.astype(y, dtype), data_type.args),
        )
        data = data_factory()
        valid_shape = " ".join(str(dim) for dim in data.shape)
        invalid_shape = " ".join(str(dim + 1) for dim in data.shape)

        assert is_bearable(data, Shaped[coreax.data.Data, valid_shape])  # pyright: ignore reportArgumentType see #1007
        assert is_bearable(data, valid_jax_type[coreax.data.Data, valid_shape])
        assert not is_bearable(data, invalid_jax_type[coreax.data.Data, invalid_shape])
        assert not is_bearable(data, Shaped[coreax.data.Data, invalid_shape])  # pyright: ignore reportArgumentType see #1007
        if not isinstance(data, coreax.data.SupervisedData):
            incorrect_instance_type = Shaped[coreax.data.SupervisedData, "..."]
            assert not is_bearable(data, incorrect_instance_type)  # pyright: ignore reportArgumentType see #1007


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
