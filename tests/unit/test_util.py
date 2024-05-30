# © Crown Copyright GCHQ
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""
Tests for utility functions.

The tests within this file verify that various utility functions written produce the
expected results on simple examples.
"""

import time
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Union
from unittest.mock import Mock

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax.random import key
from jax.typing import ArrayLike
from scipy.stats import ortho_group

from coreax.util import (
    SilentTQDM,
    apply_negative_precision_threshold,
    difference,
    invert_regularised_array,
    jit_test,
    pairwise,
    sample_batch_indices,
    solve_qp,
    squared_distance,
    tree_leaves_repeat,
    tree_zero_pad_leading_axis,
)


class TestUtil:
    """Tests for general utility functions."""

    def test_squared_distance_dist(self) -> None:
        """Test square distance under float32."""
        x, y = ortho_group.rvs(dim=2)
        expected_distance = jnp.linalg.norm(x - y) ** 2
        output_distance = squared_distance(x, y)
        assert output_distance == pytest.approx(expected_distance, abs=1e-3)
        output_distance = squared_distance(x, x)
        assert output_distance == pytest.approx(0.0, abs=1e-3)
        output_distance = squared_distance(y, y)
        assert output_distance == pytest.approx(0.0, abs=1e-3)

    def test_pairwise_squared_distance(self) -> None:
        """Test the pairwise transform on the squared distance function."""
        # create an orthonormal matrix
        dimension = 3
        orthonormal_matrix = ortho_group.rvs(dim=dimension)
        inner_distance = pairwise(squared_distance)(
            orthonormal_matrix, orthonormal_matrix
        )
        # Use original numpy because Jax arrays are immutable
        expected_output = np.ones((dimension, dimension)) * 2.0
        np.fill_diagonal(expected_output, 0.0)
        # Frobenius norm
        difference_in_distances = jnp.linalg.norm(inner_distance - expected_output)
        assert difference_in_distances.ndim == 0
        assert difference_in_distances == pytest.approx(0.0, abs=1e-3)

    def test_pairwise_difference(self) -> None:
        """Test the pairwise transform on the difference function."""
        num_points_x = 10
        num_points_y = 10
        dimension = 3
        generator = np.random.default_rng(1_989)
        x_array = generator.random((num_points_x, dimension))
        y_array = generator.random((num_points_y, dimension))
        expected_output = np.array([[x - y for y in y_array] for x in x_array])
        output = pairwise(difference)(x_array, y_array)
        assert jnp.linalg.norm(output - expected_output) == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.parametrize(
        "length",
        (0, 1, -1, 2, 10),
        ids=[
            "zero",
            "below_tree_length",
            "negative",
            "tree_length",
            "above_tree_length",
        ],
    )
    def test_tree_leaves_repeat(self, length: int) -> None:
        """Test tree_leaves_repeat for various length parameters."""
        tree = [None, 1]
        repeated_tree_leaves = tree_leaves_repeat(tree, length)
        expected_leaves = tree + [1] * (length - len(tree))
        assert repeated_tree_leaves == expected_leaves

    @pytest.mark.parametrize(
        "padding, context",
        [
            (0, does_not_raise()),
            (1, does_not_raise()),
            (-1, pytest.raises(ValueError, match="positive integer")),
            ("not_cast-able_to_int", pytest.raises(ValueError)),
        ],
    )
    def test_tree_zero_pad_leading_axis(
        self, padding: int, context: AbstractContextManager
    ) -> None:
        """Test tree_zero_pad_leading_axis for various padding widths."""
        tree = ("Test", jnp.array([[1, 2], [3, 4]]), jnp.array([5, 6]))
        if padding == 1:
            expected_tree = (
                "Test",
                jnp.array([[1, 2], [3, 4], [0, 0]]),
                jnp.array([5, 6, 0]),
            )
        else:
            expected_tree = tree
        with context:
            padded_tree = tree_zero_pad_leading_axis(tree, padding)
            assert eqx.tree_equal(padded_tree, expected_tree)

    @pytest.mark.parametrize(
        "value, threshold, expected",
        [
            (-0.01, 0.001, -0.01),
            (-0.01, -0.001, -0.01),
            (-0.0001, 0.001, 0.0),
            (-0.0001, -0.001, 0.0),
            (0.01, 0.001, 0.01),
            (0.000001, 0.001, 0.000001),
        ],
        ids=[
            "no_change",
            "negative_threshold_no_change",
            "with_change",
            "negative_threshold_with_change",
            "positive_input_1",
            "positive_input_2",
        ],
    )
    def test_apply_negative_precision_threshold(
        self, value: float, threshold: float, expected: float
    ) -> None:
        """Test apply_negative_precision_threshold for valid thresholds."""
        func_out = apply_negative_precision_threshold(
            x=value, precision_threshold=threshold
        )
        assert func_out == expected

    def test_solve_qp_invalid_kernel_mm(self) -> None:
        """
        Test how solve_qp handles invalid inputs of kernel_mm.

        The output of solve_qp is indirectly tested when testing the various weight
        optimisers that are used in this codebase. This test just ensures sensible
        behaviour occurs when unexpected inputs are passed to the function.
        """
        # Attempt to solve a QP with an input that cannot be converted to a JAX array -
        # this should error as no sensible result can be found in such a case.
        with pytest.raises(TypeError, match="not a valid JAX array type"):
            solve_qp(
                kernel_mm="invalid_kernel_mm",
                gramian_row_mean=np.array([1, 2, 3]),
            )

    def test_solve_qp_invalid_gramian_row_mean(self) -> None:
        """
        Test how solve_qp handles invalid inputs of gramian_row_mean.

        The output of solve_qp is indirectly tested when testing the various weight
        optimisers that are used in this codebase. This test just ensures sensible
        behaviour occurs when unexpected inputs are passed to the function.
        """
        # Attempt to solve a QP with an input that cannot be converted to a JAX array -
        # this should error as no sensible result can be found in such a case.
        with pytest.raises(TypeError, match="not a valid JAX array type"):
            solve_qp(
                kernel_mm=np.array([1, 2, 3]),
                gramian_row_mean="invalid_gramian_row_mean",
            )

    @pytest.mark.parametrize(
        "array, identity, rcond",
        [(jnp.eye((2)), jnp.eye((2)), -10), (jnp.eye((2)), jnp.eye((3)), None)],
        ids=["rcond_negative_not_negative_one", "unequal_array_dimensions"],
    )
    def test_approximator_invalid_inputs(
        self,
        array: ArrayLike,
        identity: ArrayLike,
        rcond: Union[int, float, None],
    ) -> None:
        """Test that `invert_regularised_array` handles invalid inputs."""
        with pytest.raises(ValueError):
            invert_regularised_array(
                array=array,
                regularisation_parameter=1e-6,
                identity=identity,
                rcond=rcond,
            )

    def test_invert_regularised_array(self) -> None:
        """
        Test invert_regularised_array.
        """
        regularisation_parameter = 1
        identity = jnp.eye(2)
        rcond = -1
        array = jnp.ones((2, 2))

        expected_output = jnp.array([[2 / 3, -1 / 3], [-1 / 3, 2 / 3]])

        output = invert_regularised_array(
            array,
            regularisation_parameter,
            identity=identity,
            rcond=rcond,
        )
        assert jnp.linalg.norm(output - expected_output) == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.parametrize(
        "data_size, batch_size, num_batches",
        [
            (1.0, 1, 1),
            (1, 1.0, 1),
            (1, 1, 1.0),
            (-1, 1, 1),
            (1, -1, 1),
            (1, 1, -1),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 0),
            (1, 2, 1),
        ],
        ids=[
            "float_data_size",
            "float_batch_size",
            "float_num_batches",
            "negative_data_size",
            "negative_batch_size",
            "negative_num_batches",
            "zero_data_size",
            "zero_batch_size",
            "zero_num_batches",
            "data_size_smaller_than_batch_size",
        ],
    )
    def test_sample_batch_indices(
        self,
        data_size: int,
        batch_size: int,
        num_batches: int,
    ) -> None:
        """
        Test sample_batch_indices for valid input parameters.
        """
        with pytest.raises(ValueError):
            sample_batch_indices(
                random_key=key(0),
                data_size=data_size,
                batch_size=batch_size,
                num_batches=num_batches,
            )

    @pytest.mark.flaky(reruns=3)
    @pytest.mark.parametrize(
        "args, kwargs, jit_kwargs",
        [
            ((2,), {}, {}),
            ((2,), {"a": 3}, {"static_argnames": "a"}),
            ((), {"a": 3}, {"static_argnames": "a"}),
        ],
    )
    def test_jit_test(self, args, kwargs, jit_kwargs) -> None:
        """
        Check that ``jit_test`` returns the expected timings.

        ``jit_test`` returns a pre_time and a post_time. The former is the time for
        JIT compiling and executing the passed function, the latter is the time for
        dispatching the JIT compiled function.
        """
        wait_time = 2
        trace_counter = Mock()

        def _mock(x=1.0, *, a=2.0):
            trace_counter()
            time.sleep(wait_time)
            return x + a

        pre_time, post_time = jit_test(
            _mock, fn_args=args, fn_kwargs=kwargs, jit_kwargs=jit_kwargs
        )
        # Tracing should only occur once, thus, `trace_counter` should only
        # be called once. Also implicitly checked in the below timing checks.
        trace_counter.assert_called_once()

        # At trace time `time.sleep` will be called. Thus, we can be sure that,
        # `pre_time` is lower bounded by `wait_time`.
        assert pre_time > wait_time
        # Post compilation `time.sleep` will be ignored, with JAX compiling the
        # function to the identity function. Thus, we can be almost sure that
        # `post_time` is upper bounded by `pre_time - wait_time`.
        assert post_time < (pre_time - wait_time)


class TestSilentTQDM:
    """Test silent substitute for TQDM."""

    def test_iterator(self):
        """Test that iterator works."""
        iterator_length = 10
        expect = list(range(iterator_length))
        actual = list(SilentTQDM(range(iterator_length)))
        assert actual == expect

    def test_write(self):
        """Test that silenced version of TQDM write command does not crash."""
        assert SilentTQDM(range(1)).write("something") is None
