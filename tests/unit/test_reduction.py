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

import unittest
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
from jax import Array, random

import coreax.coresubset
import coreax.data
import coreax.kernel
import coreax.reduction
import coreax.refine
import coreax.util
import coreax.weights


class MockCreatedInstance:
    @staticmethod
    def solve(*args):
        return tuple(*args)


class CoresetMock(coreax.reduction.Coreset):
    """Test version of :class:`Coreset` with all methods implemented."""

    def fit_to_size(self, coreset_size: int) -> None:
        raise NotImplementedError


class TestCoreset(unittest.TestCase):
    """Tests related to :class:`Coreset`."""

    def test_clone_empty(self):
        """
        Check that a copy is returned and data deleted by clone_empty.

        Also test attributes are populated by init.
        """
        with patch("coreax.validation.validate_is_instance") as _:
            # Define original instance
            weights_optimiser = MagicMock(bound=coreax.weights.WeightsOptimiser)
            kernel = MagicMock(bound=coreax.kernel.Kernel)
            refine_method = MagicMock(bound=coreax.refine.Refine)
            original_data = MagicMock()
            coreset = MagicMock()
            coreset_indices = MagicMock()
            kernel_matrix_row_sum_mean = MagicMock()
            original = CoresetMock(
                weights_optimiser=weights_optimiser,
                kernel=kernel,
                refine_method=refine_method,
            )
        original.original_data = original_data
        original.coreset = coreset
        original.coreset_indices = coreset_indices
        original.kernel_matrix_row_sum_mean = kernel_matrix_row_sum_mean

        # Create copy
        duplicate = original.clone_empty()

        # Check identities of attributes on original and duplicate
        self.assertIs(original.weights_optimiser, weights_optimiser)
        self.assertIs(duplicate.weights_optimiser, weights_optimiser)
        self.assertIs(original.kernel, kernel)
        self.assertIs(duplicate.kernel, kernel)
        self.assertIs(original.refine_method, refine_method)
        self.assertIs(duplicate.refine_method, refine_method)
        self.assertIs(original.original_data, original_data)
        self.assertIsNone(duplicate.original_data)
        self.assertIs(original.coreset, coreset)
        self.assertIsNone(duplicate.coreset)
        self.assertIs(original.coreset_indices, coreset_indices)
        self.assertIsNone(duplicate.coreset_indices)
        self.assertIs(original.kernel_matrix_row_sum_mean, kernel_matrix_row_sum_mean)
        self.assertIsNone(duplicate.kernel_matrix_row_sum_mean)

    def test_fit(self):
        """Test that original data is saved and reduction strategy called."""
        coreset = CoresetMock()
        original_data = MagicMock(spec=coreax.data.DataReader)
        strategy = MagicMock(spec=coreax.reduction.ReductionStrategy)
        coreset.fit(original_data, strategy)
        self.assertIs(coreset.original_data, original_data)
        strategy.reduce.assert_called_once_with(coreset)

    def test_solve_weights(self):
        """Check that solve_weights is called correctly."""
        with (patch("coreax.validation.validate_is_instance") as _,):
            weights_optimiser = MagicMock()
            coreset = CoresetMock(weights_optimiser=weights_optimiser)
            coreset.original_data = MagicMock(spec=coreax.data.DataReader)

            # First try prior to fitting a coreset
            self.assertRaises(coreax.util.NotCalculatedError, coreset.solve_weights)

            # Now test with a calculated coreset
            coreset.coreset = MagicMock(spec=coreax.reduction.Coreset)
            coreset.solve_weights()
            weights_optimiser.solve.assert_called_once_with(
                coreset.original_data.pre_coreset_array, coreset.coreset
            )

    def test_compute_metric(self):
        """Check that compute_metric is called correctly."""
        with patch("coreax.validation.validate_is_instance") as _:
            coreset = CoresetMock()
            coreset.original_data = MagicMock(spec=coreax.data.DataReader)
            metric = MagicMock()
            block_size = 10

            # First try prior to fitting a coreset
            self.assertRaises(
                coreax.util.NotCalculatedError,
                coreset.compute_metric,
                metric,
                block_size,
            )

            # Now test with a calculated coreset
            coreset.coreset = MagicMock(spec=Array)
            coreset.compute_metric(metric, block_size)
            metric.compute.assert_called_once_with(
                coreset.original_data.pre_coreset_array, coreset.coreset, block_size
            )

    def test_refine(self):
        """Check that refine is called correctly."""
        coreset = CoresetMock()
        coreset.original_data = MagicMock(spec=coreax.data.DataReader)
        refine_method = MagicMock(spec=coreax.refine.Refine)

        # Test with refine_method unset
        self.assertRaisesRegex(TypeError, "without a refine_method", coreset.refine)

        # Test with a coresubset
        coreset.refine_method = refine_method
        coreset.coreset_indices = MagicMock(spec=Array)
        coreset.refine()
        refine_method.refine.assert_called_once_with(coreset)

    def test_copy_fit_shallow(self):
        """Check that default behaviour of copy_fit points to other coreset array."""
        with patch("coreax.validation.validate_is_instance") as _:
            array = jnp.array([[1, 2], [3, 4]])
            indices = jnp.array([5, 6])
            this_obj = CoresetMock()
            other = CoresetMock()
            other.coreset = array
            other.coreset_indices = indices
            this_obj.copy_fit(other)
            self.assertIs(this_obj.coreset, array)
            self.assertIs(this_obj.coreset_indices, indices)

    def test_copy_fit_deep(self):
        """Check that copy_fit with deep=True creates copies of coreset arrays."""
        with patch("coreax.validation.validate_is_instance") as _:
            array = jnp.array([[1, 2], [3, 4]])
            indices = jnp.array([5, 6])
            this_obj = CoresetMock()
            other = CoresetMock()
            other.coreset = array
            other.coreset_indices = indices
            this_obj.copy_fit(other, True)
            self.assertIsNot(this_obj.coreset, array)
            np.testing.assert_equal(this_obj.coreset, array)
            self.assertIsNot(this_obj.coreset_indices, indices)
            np.testing.assert_equal(this_obj.coreset_indices, indices)

    def test_validate_fitted_ok(self):
        """Check no error raised when fit has been called."""
        obj = CoresetMock()
        obj.original_data = coreax.data.ArrayData(1, 1)
        obj.coreset = jnp.array(1)
        obj.validate_fitted("func")

    def test_validate_fitted_no_original(self):
        """Check error is raised when original data is missing."""
        obj = CoresetMock()
        obj.coreset = jnp.array(1)
        self.assertRaises(coreax.util.NotCalculatedError, obj.validate_fitted, "func")

    def test_validate_fitted_no_coreset(self):
        """Check error is raised when coreset is missing."""
        obj = CoresetMock()
        obj.original_data = coreax.data.ArrayData(1, 1)
        self.assertRaises(coreax.util.NotCalculatedError, obj.validate_fitted, "func")


class TestSizeReduce(unittest.TestCase):
    """Test :class:`SizeReduce`."""

    def test_random_sample(self):
        """Test reduction with :class:`RandomSample`."""
        orig_data = coreax.data.ArrayData.load(jnp.array([i, 2 * i] for i in range(20)))
        strategy = coreax.reduction.SizeReduce(10)
        coreset = coreax.coresubset.RandomSample()
        coreset.original_data = orig_data
        strategy.reduce(coreset)
        # Check shape of output
        self.assertEqual(coreset.coreset.format().shape, [10, 2])
        # Check values are permitted in output
        for idx, row in zip(coreset.coreset_indices, coreset.coreset):
            np.testing.assert_equal(row, np.array([idx, 2 * idx]))


class TestMapReduce(unittest.TestCase):
    """Test :class:`MapReduce`."""

    def test_random_sample(self):
        """Test map reduction with :class:`RandomSample`."""
        n = 100
        orig_data = coreax.data.ArrayData.load(
            jnp.column_stack(
                (jnp.arange(start=0, stop=n), jnp.arange(start=0, stop=2 * n, step=2))
            )
        )
        strategy = coreax.reduction.MapReduce(coreset_size=10, leaf_size=50)
        coreset = coreax.coresubset.RandomSample()
        coreset.original_data = orig_data
        strategy.reduce(coreset)
        # Check shape of output
        self.assertEqual(coreset.format().shape, (10, 2))
        # Check values are permitted in output
        for idx, row in zip(coreset.coreset_indices, coreset.coreset):
            np.testing.assert_equal(row, np.array([idx, 2 * idx]))


if __name__ == "__main__":
    unittest.main()
