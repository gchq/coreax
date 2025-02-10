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
Tests for utility functions.

The tests within this file verify that various utility functions written produce the
expected results on simple examples.
"""

import time
from contextlib import (
    AbstractContextManager,
    nullcontext as does_not_raise,
)
from unittest.mock import Mock

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy.stats import ortho_group

from coreax.util import (
    JITCompilableFunction,
    SilentTQDM,
    apply_negative_precision_threshold,
    difference,
    format_time,
    jit_test,
    pairwise,
    sample_batch_indices,
    speed_comparison_test,
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
        orthonormal_matrix = jnp.asarray(ortho_group.rvs(dim=dimension))
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
        x_array = jnp.asarray(generator.random((num_points_x, dimension)))
        y_array = jnp.asarray(generator.random((num_points_y, dimension)))
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

    @pytest.mark.parametrize(
        "max_index, batch_size, num_batches",
        [
            (1, -1, 1),
            (1, 2, 1),
        ],
        ids=[
            "negative_batch_size",
            "max_index_smaller_than_batch_size",
        ],
    )
    def test_sample_batch_indices_invalid_inputs(
        self,
        max_index: int,
        batch_size: int,
        num_batches: int,
    ) -> None:
        """Test sample_batch_indices for valid input parameters."""
        with pytest.raises(ValueError):
            sample_batch_indices(
                random_key=jr.key(0),
                max_index=max_index,
                batch_size=batch_size,
                num_batches=num_batches,
            )

    def test_sample_batch_shape(self) -> None:
        """Test that sample_batch_indices produces an array with the expected shape."""
        batch_size = 50
        num_batches = 10
        batch_indices = sample_batch_indices(
            random_key=jr.key(0),
            max_index=100,
            batch_size=batch_size,
            num_batches=num_batches,
        )
        assert batch_indices.shape == (num_batches, batch_size)

    def test_sample_batch_indices_intra_row_uniqueness(self) -> None:
        """Test that sample_batch_indices obeys intra-row uniqueness."""
        max_index = 100
        batch_size = 50
        num_batches = 10
        batch_indices = sample_batch_indices(
            random_key=jr.key(0),
            max_index=max_index,
            batch_size=batch_size,
            num_batches=num_batches,
        )
        for row in batch_indices:
            assert jnp.unique(row).shape[0] == batch_size

    def test_sample_batch_indices_inter_row_uniqueness(self) -> None:
        """
        Test that sample_batch_indices obeys inter-row uniqueness.

        For large enough `max_index` and small enough `batch_size` we expect inter-row
        uniqueness as the probability of the converse is vanishingly small.
        """
        max_index = 1000
        batch_size = 5
        num_batches = 1000
        batch_indices = sample_batch_indices(
            random_key=jr.key(0),
            max_index=max_index,
            batch_size=batch_size,
            num_batches=num_batches,
        )
        assert jnp.unique(batch_indices, axis=0).shape[0] == num_batches

    def test_format_time(self) -> None:
        """Test that `format_time` outputs expected strings."""
        assert format_time(0.0004531) == "453.1 \u03bcs"
        assert format_time(-0.032) == "-32.0 ms"
        assert format_time(125) == "2.08 mins"
        assert format_time(1e-15) == "0.0 ps"
        assert format_time(0) == "0 s"
        assert format_time(0.00000000113) == "1.13 ns"
        assert format_time(10.15) == "10.15 s"

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

        x_in = jnp.ones(1000)
        a_in = 2 * jnp.ones(1000)

        def _mock(x=x_in, *, a=a_in):
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
        assert 0 < post_time < (pre_time - wait_time)

    def test_speed_comparison_test(self) -> None:
        """
        Check that ``speed_comparison_test`` returns expected timings.

        ``speed_comparison_test`` returns raw timings and a list of summary statistics
        for each passed function. The summary statistics include an  estimate of the
        mean time for JIT compiling the passed function, and an estimate of the mean
        time for executing the JIT compiled function.

        In this test we check that the passed functions are compiled for each run and
        not cached, and that a for-looped variant of a mean function takes longer to
        compile than the in-built vectorised version.
        """
        trace_counter = Mock()
        # Define a for-looped version of a mean computation, this will be very slow to
        # compile.

        def _slow_mean(a):
            trace_counter()
            num_points = a.shape[0]
            total = 0
            for i in range(num_points):
                total += a[i]
            return total / num_points

        random_vector = jr.normal(jr.key(2_024), shape=(100,))

        num_runs = 10
        summary_stats, result_dict = speed_comparison_test(
            [
                JITCompilableFunction(
                    _slow_mean, fn_kwargs={"a": random_vector}, name="slow_mean"
                ),
                JITCompilableFunction(
                    jnp.mean, fn_kwargs={"a": random_vector}, name="jnp_mean"
                ),
            ],
            num_runs=num_runs,
            log_results=False,
        )

        # Tracing should occur for each run of the function, not just once, thus
        # `trace_counter` should be called num_runs times.
        assert trace_counter.call_count == num_runs

        # Assert that indeed the mean compilation time of slow_mean is slower than
        # jnp.mean.
        slow_mean_compilation_time = summary_stats[0][0][0]
        fast_mean_compilation_time = summary_stats[1][0][0]
        assert slow_mean_compilation_time > fast_mean_compilation_time > 0

        # Check result dictionary has the correct size
        assert len(result_dict["slow_mean"]) == num_runs
        assert len(result_dict["jnp_mean"]) == num_runs

        # Check summary stats have been computed correctly
        assert jnp.all(result_dict["slow_mean"].mean(axis=0) == summary_stats[0][0])
        assert jnp.all(result_dict["slow_mean"].std(axis=0) == summary_stats[0][1])
        assert jnp.all(result_dict["jnp_mean"].mean(axis=0) == summary_stats[1][0])
        assert jnp.all(result_dict["jnp_mean"].std(axis=0) == summary_stats[1][1])


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
