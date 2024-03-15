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

import unittest
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
from scipy.stats import ortho_group

import coreax.util


class TestUtil(unittest.TestCase):
    """
    Tests for general utility functions.
    """

    def test_squared_distance_dist(self) -> None:
        """
        Test square distance under float32.
        """
        x, y = ortho_group.rvs(dim=2)
        expected_distance = jnp.linalg.norm(x - y) ** 2
        output_distance = coreax.util.squared_distance(x, y)
        self.assertAlmostEqual(output_distance, expected_distance, places=3)
        output_distance = coreax.util.squared_distance(x, x)
        self.assertAlmostEqual(output_distance, 0.0, places=3)
        output_distance = coreax.util.squared_distance(y, y)
        self.assertAlmostEqual(output_distance, 0.0, places=3)

    def test_squared_distance_dist_pairwise(self) -> None:
        """
        Test vmap version of sq distance.
        """
        # create an orthonormal matrix
        dimension = 3
        orthonormal_matrix = ortho_group.rvs(dim=dimension)
        inner_distance = coreax.util.squared_distance_pairwise(
            orthonormal_matrix, orthonormal_matrix
        )
        # Use original numpy because Jax arrays are immutable
        expected_output = np.ones((dimension, dimension)) * 2.0
        np.fill_diagonal(expected_output, 0.0)
        # Frobenius norm
        difference_in_distances = jnp.linalg.norm(inner_distance - expected_output)
        self.assertEqual(difference_in_distances.ndim, 0)
        self.assertAlmostEqual(float(difference_in_distances), 0.0, places=3)

    def test_pairwise_difference(self) -> None:
        """
        Test the function pairwise_difference.

        This test ensures efficient computation of pairwise differences.
        """
        num_points_x = 10
        num_points_y = 10
        dimension = 3
        generator = np.random.default_rng(1_989)
        x_array = generator.random((num_points_x, dimension))
        y_array = generator.random((num_points_y, dimension))
        expected_output = np.array([[x - y for y in y_array] for x in x_array])
        output = coreax.util.pairwise_difference(x_array, y_array)
        self.assertAlmostEqual(
            float(jnp.linalg.norm(output - expected_output)), 0.0, places=3
        )

    def test_apply_negative_precision_threshold_invalid(self) -> None:
        """
        Test the function apply_negative_precision_threshold with an invalid threshold.

        A negative precision threshold is given, which should be rejected by the
        function.
        """
        self.assertRaises(
            ValueError,
            coreax.util.apply_negative_precision_threshold,
            x=0.1,
            precision_threshold=-1e-8,
        )

    def test_apply_negative_precision_threshold_valid_no_change(self) -> None:
        """
        Test the function apply_negative_precision_threshold with no change needed.

        This test questions if the value -0.01 is sufficiently close to 0 to set it to
        0, however the precision threshold is sufficiently small to consider this a
        distinct value and not apply a cap.
        """
        func_out = coreax.util.apply_negative_precision_threshold(
            x=-0.01, precision_threshold=0.001
        )
        self.assertEqual(func_out, -0.01)

    def test_apply_negative_precision_threshold_valid_with_change(self) -> None:
        """
        Test the function apply_negative_precision_threshold with a change needed.

        This test questions if the value -0.0001 is sufficiently close to 0 to set it to
        0. In this instance, the precision threshold is sufficiently large to consider
        -0.0001 close enough to 0 to apply a cap.
        """
        func_out = coreax.util.apply_negative_precision_threshold(
            x=-0.0001, precision_threshold=0.001
        )
        self.assertEqual(func_out, 0.0)

    def test_apply_negative_precision_threshold_valid_positive_input(self) -> None:
        """
        Test the function apply_negative_precision_threshold with no change needed.

        This test questions if the value 0.01 is sufficiently close to 0 to set it to
        0. Since the function should only cap negative numbers to 0 if they are
        sufficiently close, it should have no impact on this positive input, regardless
        of the threshold supplied.
        """
        func_out_1 = coreax.util.apply_negative_precision_threshold(
            x=0.01, precision_threshold=0.001
        )
        self.assertEqual(func_out_1, 0.01)

        func_out_2 = coreax.util.apply_negative_precision_threshold(
            x=0.000001, precision_threshold=0.001
        )
        self.assertEqual(func_out_2, 0.000001)

    def test_solve_qp_invalid_kernel_mm(self) -> None:
        """
        Test how solve_qp handles invalid inputs of kernel_mm.

        The output of solve_qp is indirectly tested when testing the various weight
        optimisers that are used in this codebase. This test just ensures sensible
        behaviour occurs when unexpected inputs are passed to the function.
        """
        # Attempt to solve a QP with an input that cannot be converted to a JAX array -
        # this should error as no sensible result can be found in such a case.
        with self.assertRaisesRegex(TypeError, "not a valid JAX array type"):
            coreax.util.solve_qp(
                kernel_mm="invalid_kernel_mm",
                kernel_matrix_row_sum_mean=np.array([1, 2, 3]),
            )

    def test_solve_qp_invalid_kernel_matrix_row_sum_mean(self) -> None:
        """
        Test how solve_qp handles invalid inputs of kernel_matrix_row_sum_mean.

        The output of solve_qp is indirectly tested when testing the various weight
        optimisers that are used in this codebase. This test just ensures sensible
        behaviour occurs when unexpected inputs are passed to the function.
        """
        # Attempt to solve a QP with an input that cannot be converted to a JAX array -
        # this should error as no sensible result can be found in such a case.
        with self.assertRaisesRegex(TypeError, "not a valid JAX array type"):
            coreax.util.solve_qp(
                kernel_mm=np.array([1, 2, 3]),
                kernel_matrix_row_sum_mean="invalid_kernel_matrix_row_sum_mean",
            )

    def test_jit_test(self) -> None:
        """
        Test jit_test calls the function in question twice when checking performance.

        The function jit_test is used to assess the performance of other functions and
        methods in the codebase. It's inputs are a function (denoted fn) and inputs to
        provide to fn. This unit test checks that fn is called twice. In a practical
        usage of jit_test, the first call to fn performs the JIT compilation, and the
        second call assesses if a performance improvement has occurred given the
        JIT compilation.
        """
        mock_function = MagicMock()
        coreax.util.jit_test(mock_function)
        num_calls = mock_function.call_count
        self.assertEqual(num_calls, 2)


class TestSilentTQDM(unittest.TestCase):
    """Test silent substitute for TQDM."""

    def test_iterator(self):
        """Test that iterator works."""
        iterator_length = 10
        expect = list(range(iterator_length))
        actual = list(coreax.util.SilentTQDM(range(iterator_length)))
        self.assertListEqual(actual, expect)

    def test_write(self):
        """Test that silenced version of TQDM write command does not crash."""
        self.assertIsNone(coreax.util.SilentTQDM(range(1)).write("something"))


if __name__ == "__main__":
    unittest.main()
