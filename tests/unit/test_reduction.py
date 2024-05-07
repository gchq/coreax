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
Tests for reduction implementations.

The tests within this file verify that the reduction classes and functionality used to
construct coresets throughout the codebase produce the expected results on simple
examples.
"""

import unittest
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
from jax import Array, random

import coreax.coresubset
import coreax.data
import coreax.kernel
import coreax.metrics
import coreax.reduction
import coreax.refine
import coreax.util
import coreax.weights


class MockCreatedInstance:
    """
    Mock implementation of a reduction instance for testing purposes.
    """

    @staticmethod
    def solve(*args):
        """
        Solve a reduction problem defined by input args.
        """
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
        # Define original instance
        weights_optimiser = MagicMock(spec=coreax.weights.WeightsOptimiser)
        kernel = MagicMock(spec=coreax.kernel.Kernel)
        refine_method = MagicMock(spec=coreax.refine.Refine)
        original_data = MagicMock()
        coreset = MagicMock()
        coreset_indices = MagicMock()
        gramian_row_mean = MagicMock()
        original = CoresetMock(
            weights_optimiser=weights_optimiser,
            kernel=kernel,
            refine_method=refine_method,
        )
        original.original_data = original_data
        original.coreset = coreset
        original.coreset_indices = coreset_indices
        original.gramian_row_mean = gramian_row_mean

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
        self.assertIs(original.gramian_row_mean, gramian_row_mean)
        self.assertIsNone(duplicate.gramian_row_mean)

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
        weights_optimiser = MagicMock(spec=coreax.weights.WeightsOptimiser)
        coreset = CoresetMock(weights_optimiser=weights_optimiser)
        coreset.original_data = MagicMock(spec=coreax.data.DataReader)
        coreset.original_data.pre_coreset_array = MagicMock(spec=Array)

        # First try prior to fitting a coreset
        with self.assertRaises(coreax.util.NotCalculatedError):
            coreset.solve_weights()

        # Now test with a calculated coreset
        coreset.coreset = MagicMock(spec=Array)
        coreset.solve_weights()
        weights_optimiser.solve.assert_called_once_with(
            coreset.original_data.pre_coreset_array, coreset.coreset
        )

    def test_compute_metric(self):
        """Check that compute_metric is called correctly."""
        coreset = CoresetMock()
        coreset.original_data = MagicMock(spec=coreax.data.DataReader)
        coreset.original_data.pre_coreset_array = MagicMock(spec=Array)
        metric = MagicMock(spec=coreax.metrics.Metric)
        block_size = 10

        # First try prior to fitting a coreset
        with self.assertRaises(coreax.util.NotCalculatedError):
            coreset.compute_metric(metric, block_size)

        # Now test with a calculated coreset
        coreset.coreset = MagicMock(spec=Array)
        coreset.compute_metric(metric, block_size)
        metric.compute.assert_called_once_with(
            coreset.original_data.pre_coreset_array,
            coreset.coreset,
            block_size=block_size,
            weights_x=None,
            weights_y=None,
        )

    def test_refine(self):
        """Check that refine is called correctly."""
        coreset = CoresetMock()
        coreset.original_data = MagicMock(spec=coreax.data.DataReader)
        refine_method = MagicMock(spec=coreax.refine.Refine)

        # Test with refine_method unset
        with self.assertRaisesRegex(TypeError, "without a refine_method"):
            coreset.refine()

        # Test with a coresubset
        coreset.refine_method = refine_method
        coreset.coreset_indices = MagicMock(spec=Array)
        coreset.refine()
        refine_method.refine.assert_called_once_with(coreset)

    def test_render(self):
        """Check that render is called correctly."""
        coreset = CoresetMock()
        original_data = MagicMock(spec=coreax.data.DataReader)
        coreset.original_data = original_data

        # Test when coreset.coreset is unpopulated
        with self.assertRaises(coreax.util.NotCalculatedError):
            coreset.render()

        # Test with coreset.coreset existing
        coreset.coreset = MagicMock(spec=Array)
        coreset.render()
        # Check the render method in the DataReader class is called exactly once
        original_data.render.assert_called_once()

    def test_copy_fit_shallow(self):
        """Check that default behaviour of copy_fit points to other coreset array."""
        array = jnp.array([[1, 2], [3, 4]])
        indices = jnp.array([5, 6])
        this_obj = CoresetMock()
        other = CoresetMock()
        other.original_data = MagicMock(spec=coreax.data.DataReader)
        other.coreset = array
        other.coreset_indices = indices
        this_obj.copy_fit(other)
        # Check original_data not copied
        self.assertIsNone(this_obj.original_data)
        # Check copy
        self.assertIs(this_obj.coreset, array)
        self.assertIs(this_obj.coreset_indices, indices)

    def test_copy_fit_deep(self):
        """Check that copy_fit with deep=True creates copies of coreset arrays."""
        array = jnp.array([[1, 2], [3, 4]])
        indices = jnp.array([5, 6])
        this_obj = CoresetMock()
        other = CoresetMock()
        other.original_data = MagicMock(spec=coreax.data.DataReader)
        other.coreset = array
        other.coreset_indices = indices
        this_obj.copy_fit(other, True)
        # Check original_data not copied
        self.assertIsNone(this_obj.original_data)
        # Check copy
        self.assertIsNot(this_obj.coreset, array)
        np.testing.assert_array_equal(this_obj.coreset, array)
        self.assertIsNot(this_obj.coreset_indices, indices)
        np.testing.assert_array_equal(this_obj.coreset_indices, indices)

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
        with self.assertRaises(coreax.util.NotCalculatedError):
            obj.validate_fitted("func")

    def test_validate_fitted_no_coreset(self):
        """Check error is raised when coreset is missing."""
        obj = CoresetMock()
        obj.original_data = coreax.data.ArrayData(1, 1)
        with self.assertRaises(coreax.util.NotCalculatedError):
            obj.validate_fitted("func")

    def test_coreset_inputs(self):
        """
        Check how Coreset handles invalid inputs.

        Note that the  fit, fit_to_size and solve_weights methods on the Coreset class
        are tested when specific coreset construction methods are tested, for example in
        test_coresubset.py.
        """
        # Define a coreset object, with all inputs as invalid
        original = CoresetMock(
            weights_optimiser=coreax.util.InvalidKernel,
            kernel=coreax.util.InvalidKernel,
            refine_method=coreax.util.InvalidKernel,
        )

        # When we call clone_empty, this should not depend on the inputs being invalid,
        # so a call should execute without issue
        original.clone_empty()

        # If we try to call the method copy_fit with an object that does not have the
        # coreset or coreset_indices attributes or the validate_fitted method, we expect
        # an attribute error
        with self.assertRaises(AttributeError) as error_raised:
            original.copy_fit(other=coreax.util.InvalidKernel)

        self.assertEqual(
            error_raised.exception.args[0],
            "type object 'InvalidKernel' has no attribute 'validate_fitted'",
        )

        # Define a validate_fitted method and check that we still get an attribute error
        # when we try to copy a coreset that does not exist
        invalid_with_validate = coreax.util.InvalidKernel(x=1.0)
        invalid_with_validate.validate_fitted = MagicMock()
        with self.assertRaises(AttributeError) as error_raised:
            original.copy_fit(other=invalid_with_validate)
        self.assertEqual(
            error_raised.exception.args[0],
            "'InvalidKernel' object has no attribute 'coreset'",
        )

        # Assess how an actual Coreset object handles a non-string value of caller_name
        # when calling the method validate_fitted. We expect a NotCalculatedError to be
        # raised, with this object inserted into the error string
        with self.assertRaises(coreax.util.NotCalculatedError) as error_raised:
            original.validate_fitted(coreax.util.InvalidKernel)
        self.assertEqual(
            error_raised.exception.args[0],
            "Need to call fit before calling <class 'coreax.util.InvalidKernel'>",
        )


class TestSizeReduce(unittest.TestCase):
    """Test :class:`~coreax.reduction.SizeReduce`."""

    def setUp(self):
        self.random_key = random.key(0)

    def test_random_sample(self):
        """Test reduction with :class:`~coreax.coresubset.RandomSample`."""
        orig_data = coreax.data.ArrayData.load(
            jnp.array([[i, 2 * i] for i in range(20)])
        )
        strategy = coreax.reduction.SizeReduce(10)
        coreset = coreax.coresubset.RandomSample(self.random_key)
        coreset.original_data = orig_data
        strategy.reduce(coreset)
        # Check shape of output
        self.assertEqual(coreset.coreset.shape, (10, 2))
        # Check values are permitted in output
        for idx, row in zip(coreset.coreset_indices, coreset.coreset):
            np.testing.assert_array_equal(row, np.array([idx, 2 * idx]))

    def test_size_reduce_zero_size(self):
        """
        Test how SizeReduce handles invalid inputs.
        """
        orig_data = coreax.data.ArrayData.load(
            jnp.array([[i, 2 * i] for i in range(20)])
        )
        coreset = coreax.coresubset.RandomSample(self.random_key)
        coreset.original_data = orig_data

        # Consider a zero sized coreset
        coreax.reduction.SizeReduce(coreset_size=0).reduce(coreset)
        np.testing.assert_array_equal(coreset.coreset_indices, np.empty([]))
        np.testing.assert_array_equal(coreset.coreset, np.empty([]))

        # When trying to apply size reduce with a non-integer coreset size, we should
        # have a value error raised
        with self.assertRaises(ValueError) as error_raised:
            coreax.reduction.SizeReduce(coreset_size=0.1).reduce(coreset)
        self.assertEqual(
            error_raised.exception.args[0],
            "coreset_size must be a positive integer",
        )

        # When trying to apply size reduce with a negative coreset size, we should have
        # a value error raised
        with self.assertRaises(ValueError) as error_raised:
            coreax.reduction.SizeReduce(coreset_size=-2).reduce(coreset)
        self.assertEqual(
            error_raised.exception.args[0],
            "coreset_size must be a positive integer",
        )


class TestMapReduce(unittest.TestCase):
    """Test :class:`MapReduce`."""

    def setUp(self):
        self.random_key = random.key(0)

    def test_random_sample(self):
        """Test map reduction with :class:`~coreax.coresubset.RandomSample`."""
        num_data_points = 100
        orig_data = coreax.data.ArrayData.load(
            jnp.array([[i, 2 * i] for i in range(num_data_points)])
        )
        strategy = coreax.reduction.MapReduce(coreset_size=10, leaf_size=20)
        coreset = coreax.coresubset.RandomSample(self.random_key)
        coreset.original_data = orig_data

        # Disable pylint warning for protected-access as we are testing a single part of
        # the over-arching algorithm
        # pylint: disable=protected-access
        with patch.object(
            coreax.reduction.MapReduce,
            "_reduce_recursive",
            wraps=strategy._reduce_recursive,
        ) as mock:
            # Perform the reduction
            strategy.reduce(coreset)
            num_calls_reduce_recursive = mock.call_count
        # pylint: enable=protected-access

        # Check the shape of the output
        self.assertEqual(coreset.format().shape, (10, 2))
        # Check _reduce_recursive is called exactly three times
        self.assertEqual(num_calls_reduce_recursive, 3)
        # Check values are permitted in output
        for idx, row in zip(coreset.coreset_indices, coreset.coreset):
            np.testing.assert_equal(row, np.array([idx, 2 * idx]))

    def test_random_sample_not_parallel(self):
        """
        Test map reduction with :class:`~coreax.coresubset.RandomSample` in series.
        """
        num_data_points = 100
        orig_data = coreax.data.ArrayData.load(
            jnp.array([[i, 2 * i] for i in range(num_data_points)])
        )
        strategy = coreax.reduction.MapReduce(
            coreset_size=10, leaf_size=20, parallel=False
        )
        coreset = coreax.coresubset.RandomSample(self.random_key)
        coreset.original_data = orig_data
        # Disable pylint warning for protected-access as we are testing a single part of
        # the over-arching algorithm
        # pylint: disable=protected-access
        with patch.object(
            coreax.reduction.MapReduce,
            "_reduce_recursive",
            wraps=strategy._reduce_recursive,
        ) as mock:
            # Perform the reduction
            strategy.reduce(coreset)
            num_calls_reduce_recursive = mock.call_count
        # pylint: enable=protected-access

        # Check the shape of the output
        self.assertEqual(coreset.format().shape, (10, 2))
        # Check _reduce_recursive is called exactly three times
        self.assertEqual(num_calls_reduce_recursive, 3)
        # Check values are permitted in output
        for idx, row in zip(coreset.coreset_indices, coreset.coreset):
            np.testing.assert_equal(row, np.array([idx, 2 * idx]))

    def test_random_sample_big_leaves(self):
        """
        Test map reduction with :class:`~coreax.coresubset.RandomSample` and big leaves.

        This test sets leaf_size = num_data_points and checks the recursive function
        is called only once.
        """
        num_data_points = 100
        orig_data = coreax.data.ArrayData.load(
            jnp.array([[i, 2 * i] for i in range(num_data_points)])
        )
        strategy = coreax.reduction.MapReduce(
            coreset_size=10, leaf_size=num_data_points
        )
        coreset = coreax.coresubset.RandomSample(self.random_key)
        coreset.original_data = orig_data

        # Disable pylint warning for protected-access as we are testing a single part of
        # the over-arching algorithm
        # pylint: disable=protected-access
        with (
            patch.object(
                coreax.reduction.MapReduce,
                "_reduce_recursive",
                wraps=strategy._reduce_recursive,
            ) as mock_reduce_recursive,
            patch.object(
                coreax.reduction.MapReduce,
                "_coreset_copy_fit",
                wraps=strategy._coreset_copy_fit,
            ) as mock_coreset_copy_fit,
        ):
            # Perform the reduction
            strategy.reduce(coreset)
            # Check _reduce_recursive is called only once
            mock_reduce_recursive.assert_called_once()
            # Check _coreset_copy_fit is called only once
            mock_coreset_copy_fit.assert_called_once()
        # pylint: enable=protected-access

    def test_reduce_recursive_unset_coreset_indices(self):
        """Test MapReduce with a Coreset that does not have ``coreset_indices``."""
        num_data_points = 100
        orig_data = coreax.data.ArrayData.load(
            jnp.array([[i, 2 * i] for i in range(num_data_points)])
        )
        strategy = coreax.reduction.MapReduce(coreset_size=10, leaf_size=100)
        coreset = coreax.coresubset.RandomSample(self.random_key)
        coreset.original_data = orig_data
        # Check AssertionError raises when method is called with no input_indices
        with self.assertRaises(AssertionError):
            input_data = coreset.original_data.pre_coreset_array
            # Disable pylint warning for protected-access as we are testing a single
            # part of the over-arching algorithm
            # pylint: disable=protected-access
            strategy._reduce_recursive(template=coreset, input_data=input_data)
            # pylint: enable=protected-access

    def test_map_reduce_zero_coreset_size(self):
        """
        Test how MapReduce handles a coreset_size of zero.
        """
        num_data_points = 100
        orig_data = coreax.data.ArrayData.load(
            jnp.array([[i, 2 * i] for i in range(num_data_points)])
        )
        strategy = coreax.reduction.MapReduce(coreset_size=0, leaf_size=100)
        coreset = coreax.coresubset.RandomSample(self.random_key)
        coreset.original_data = orig_data

        # Call reduce with a coreset size of 0 - which should just assign empty coreset
        # and coreset indices to the coreset object
        strategy.reduce(coreset)
        self.assertEqual(len(coreset.coreset), 0)
        self.assertEqual(len(coreset.coreset_indices), 0)

    def test_map_reduce_negative_coreset_size(self):
        """
        Test how MapReduce handles a negative coreset_size.
        """
        num_data_points = 100
        orig_data = coreax.data.ArrayData.load(
            jnp.array([[i, 2 * i] for i in range(num_data_points)])
        )
        strategy = coreax.reduction.MapReduce(coreset_size=-5, leaf_size=100)
        coreset = coreax.coresubset.RandomSample(self.random_key)
        coreset.original_data = orig_data

        # Call reduce with a negative coreset size - which should cause a value error
        # within the RandomSample usage
        with self.assertRaises(ValueError) as error_raised:
            strategy.reduce(coreset)

        self.assertEqual(
            error_raised.exception.args[0],
            "coreset_size must be a positive integer",
        )

    def test_map_reduce_float_coreset_size(self):
        """
        Test how MapReduce handles a float coreset_size.
        """
        num_data_points = 100
        orig_data = coreax.data.ArrayData.load(
            jnp.array([[i, 2 * i] for i in range(num_data_points)])
        )
        strategy = coreax.reduction.MapReduce(coreset_size=5.0, leaf_size=100)
        coreset = coreax.coresubset.RandomSample(self.random_key)
        coreset.original_data = orig_data

        # Call reduce with a float coreset size - which should cause a value error
        # within the RandomSample usage
        with self.assertRaises(ValueError) as error_raised:
            strategy.reduce(coreset)

        self.assertEqual(
            error_raised.exception.args[0],
            "coreset_size must be a positive integer",
        )

    def test_map_reduce_zero_leaf_size(self):
        """
        Test how MapReduce handles a leaf_size of zero.
        """
        num_data_points = 100
        orig_data = coreax.data.ArrayData.load(
            jnp.array([[i, 2 * i] for i in range(num_data_points)])
        )
        strategy = coreax.reduction.MapReduce(coreset_size=5, leaf_size=0)
        coreset = coreax.coresubset.RandomSample(self.random_key)
        coreset.original_data = orig_data

        # Call reduce with a zero leaf size - which should cause a value error
        # within the KDTree functionality used
        with self.assertRaises(ValueError) as error_raised:
            strategy.reduce(coreset)

        self.assertEqual(
            error_raised.exception.args[0],
            "leaf_size must be greater than or equal to 1",
        )

    def test_map_reduce_negative_leaf_size(self):
        """
        Test how MapReduce handles a negative leaf_size.
        """
        num_data_points = 100
        orig_data = coreax.data.ArrayData.load(
            jnp.array([[i, 2 * i] for i in range(num_data_points)])
        )
        strategy = coreax.reduction.MapReduce(coreset_size=5, leaf_size=-50)
        coreset = coreax.coresubset.RandomSample(self.random_key)
        coreset.original_data = orig_data

        # Call reduce with a zero leaf size - which should cause a value error
        # within the KDTree functionality used
        with self.assertRaises(ValueError) as error_raised:
            strategy.reduce(coreset)

        self.assertEqual(
            error_raised.exception.args[0],
            "leaf_size must be greater than or equal to 1",
        )

    def test_map_reduce_float_leaf_size(self):
        """
        Test how MapReduce handles a float leaf_size.
        """
        num_data_points = 100
        orig_data = coreax.data.ArrayData.load(
            jnp.array([[i, 2 * i] for i in range(num_data_points)])
        )
        strategy = coreax.reduction.MapReduce(coreset_size=5, leaf_size=50.0)
        coreset = coreax.coresubset.RandomSample(self.random_key)
        coreset.original_data = orig_data

        # Call reduce with a float leaf size - which should cause a value error
        # set for ease of use
        with self.assertRaises(ValueError) as error_raised:
            strategy.reduce(coreset)

        self.assertEqual(
            error_raised.exception.args[0],
            "leaf_size must be a positive integer",
        )


if __name__ == "__main__":
    unittest.main()
