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
Tests for approximation approaches.

Approximations are used to reduce computational demand when computing coresets. The
tests within this file verify that these approximations produce the expected results on
simple examples.
"""

import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
from jax import random

import coreax.approximation
import coreax.kernel
import coreax.util

# pylint: disable=too-many-public-methods


class TestApproximations(unittest.TestCase):
    """
    Tests related to approximation.py classes & functions.
    """

    def setUp(self) -> None:
        r"""
        Define data shared across tests.

        We consider the data:

        .. math::

            x = [ [0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [-1.0, 0.0] ]

        and a SquaredExponentialKernel which is defined as
        :math:`k(x,y) = \text{output_scale}\exp(-||x-y||^2/2 * \text{length_scale}^2)`.
        For simplicity, we set ``length_scale`` to :math:`1.0/np.sqrt(2)`
        and ``output_scale`` to 1.0.

        The tests here ensure that approximations to the kernel matrix row sum mean are
        valid. For a single row (data record), kernel matrix row sum mean is computed by
        applying the kernel to this data record and all other data records. We then sum
        the results and divide by the number of data records. The first
        data-record ``[0, 0]`` in the data considered here therefore gives a result of:

        .. math::

              (1/4) * (
              exp(-((0.0 - 0.0)^2 + (0.0 - 0.0)^2)) +
              exp(-((0.0 - 0.5)^2 + (0.0 - 0.5)^2)) +
              exp(-((0.0 - 1.0)^2 + (0.0 - 0.0)^2)) +
              exp(-((0.0 - -1.0)^2 + (0.0 - 0.0)^2))
              )

        which evaluates to 0.5855723855138795.

        We can repeat the above but considering each data-point in ``x`` in turn and
        attain a set of true distances to use as the ground truth in the tests.
        """
        self.random_key = random.key(10)

        # Setup a 'small' toy example that can be computed by hand
        self.data = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [-1.0, 0.0]])
        self.num_kernel_points = 3
        self.num_train_points = 3

        # Define a kernel object
        self.kernel = coreax.kernel.SquaredExponentialKernel(
            length_scale=1.0 / np.sqrt(2), output_scale=1.0
        )

        # We can repeat the above, but changing the point with which we are comparing
        # to get:
        self.true_distances = np.array(
            [
                0.5855723855138795,
                0.5737865795122914,
                0.4981814349432025,
                0.3670700196710188,
            ]
        )

    def test_kernel_mean_approximator_creation(self) -> None:
        """
        Test the class KernelMeanApproximator initialises correctly.
        """
        # Disable pylint warning for abstract-class-instantiated as we are intentionally
        # patching these whilst testing creation of the parent class
        # pylint: disable=abstract-class-instantiated
        # Patch the abstract method (approximate) of the KernelMeanApproximator, so it
        # can be created
        p = patch.multiple(
            coreax.approximation.KernelMeanApproximator, __abstractmethods__=set()
        )
        p.start()

        # Define the approximator
        approximator = coreax.approximation.KernelMeanApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
        )
        # pylint: enable=abstract-class-instantiated

        # Check parameters have been set
        self.assertEqual(approximator.kernel, self.kernel)
        np.testing.assert_array_equal(approximator.random_key, self.random_key)
        self.assertEqual(approximator.num_kernel_points, self.num_kernel_points)

    def test_random_approximator(self) -> None:
        """
        Verify random approximator performance on toy problem.

        This test verifies that, when the entire dataset is used to train the
        approximation, the result is very close to the true value. We further check if a
        subset of the data is used for training, that the approximation is still close
        to the true value, but less so than when using a larger training set.
        """
        # Define the approximator - full dataset used to fit the approximation
        approximator_full = coreax.approximation.RandomApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.data.shape[0],
            num_train_points=self.data.shape[0],
        )

        # Define the approximator - full dataset used to fit the approximation
        approximator_partial = coreax.approximation.RandomApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
            num_train_points=self.num_train_points,
        )

        # Approximate the kernel row mean using the full training set (so the
        # approximation should be very close to the true) and only part of the data for
        # training (so the error should grow)
        approximate_kernel_mean_full = approximator_full.approximate(self.data)
        approximate_kernel_mean_partial = approximator_partial.approximate(self.data)

        # Check the approximation is close to the true value
        np.testing.assert_array_almost_equal(
            approximate_kernel_mean_full, self.true_distances, decimal=5
        )
        np.testing.assert_array_almost_equal(
            approximate_kernel_mean_partial, self.true_distances, decimal=1
        )

        # Compute the approximation error and check if is better if we use more data
        approx_error_full = np.sum(
            np.square(self.true_distances - approximate_kernel_mean_full)
        )
        approx_error_partial = np.sum(
            np.square(self.true_distances - approximate_kernel_mean_partial)
        )
        self.assertTrue(approx_error_full <= approx_error_partial)

    def test_random_approximator_negative_num_kernel_points(self) -> None:
        """
        Test the class RandomApproximator rejects negative values of num_kernel_points.
        """
        # Define the approximator with a negative value of num_kernel_points
        approximator = coreax.approximation.RandomApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=-self.num_kernel_points,
            num_train_points=self.data.shape[0],
        )
        with self.assertRaises(ValueError) as error_raised:
            approximator.approximate(self.data)

        self.assertEqual(
            error_raised.exception.args[0], "num_kernel_points must be positive"
        )

    def test_random_approximator_zero_num_kernel_points(self) -> None:
        """
        Test the class RandomApproximator does not fail with zero num_kernel_points.
        """
        # Define the approximator with a zero value of num_kernel_points
        approximator = coreax.approximation.RandomApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=0,
            num_train_points=self.data.shape[0],
        )

        # In the case of zero num_kernel_points, we expect zero features to be used
        # during the approximation, and this results in an approximation vector of zeros
        np.testing.assert_array_equal(
            approximator.approximate(self.data), jnp.zeros_like(self.true_distances)
        )

    def test_random_approximator_large_num_kernel_points(self) -> None:
        """
        Test the class RandomApproximator rejects too large values of num_kernel_points.
        """
        # Define the approximator with a value of num_kernel_points larger than the
        # number of data-points in the dataset provided
        approximator = coreax.approximation.RandomApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=100 * self.data.shape[0],
            num_train_points=self.data.shape[0],
        )
        with self.assertRaises(ValueError) as error_raised:
            approximator.approximate(self.data)

        self.assertEqual(
            error_raised.exception.args[0],
            "num_kernel_points must be no larger than the number of points in "
            "the provided data",
        )

    def test_random_approximator_negative_num_train_points(self) -> None:
        """
        Test the class RandomApproximator rejects negative values of num_train_points.
        """
        # Define the approximator with a negative value of num_train_points
        approximator = coreax.approximation.RandomApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
            num_train_points=-self.data.shape[0],
        )
        with self.assertRaises(ValueError) as error_raised:
            approximator.approximate(self.data)

        self.assertEqual(
            error_raised.exception.args[0], "num_train_points must be positive"
        )

    def test_random_approximator_zero_num_train_points(self) -> None:
        """
        Test the class RandomApproximator does not fail with zero num_train_points.
        """
        # Define the approximator with a zero value of num_train_points
        approximator = coreax.approximation.RandomApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
            num_train_points=0,
        )

        # In the case of zero num_train_points, we expect zero features to be used
        # during the approximation, and this results in an approximation vector of zeros
        np.testing.assert_array_equal(
            approximator.approximate(self.data), jnp.zeros_like(self.true_distances)
        )

    def test_random_approximator_large_num_train_points(self) -> None:
        """
        Test the class RandomApproximator rejects too large values of num_train_points.
        """
        # Define the approximator with a value of num_train_points larger than the
        # dataset provided
        approximator = coreax.approximation.RandomApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
            num_train_points=100 * self.data.shape[0],
        )
        with self.assertRaises(ValueError) as error_raised:
            approximator.approximate(self.data)

        self.assertEqual(
            error_raised.exception.args[0],
            "num_train_points must be no larger than the number of points in "
            "the provided data",
        )

    def test_random_approximator_invalid_kernel(self) -> None:
        """
        Test the class RandomApproximator rejects an invalid kernel.
        """
        approximator = coreax.approximation.RandomApproximator(
            kernel=coreax.util.InvalidKernel,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
            num_train_points=self.data.shape[0],
        )
        with self.assertRaises(AttributeError) as error_raised:
            approximator.approximate(self.data)

        self.assertEqual(
            error_raised.exception.args[0],
            "type object 'InvalidKernel' has no attribute 'compute'",
        )

    def test_annchor_approximator(self) -> None:
        """
        Verify Annchor approximator performance on toy problem.

        This test verifies that, when the entire dataset is used to train the
        approximation, the result is very close to the true value. We further check if a
        subset of the data is used for training, that the approximation is still close
        to the true value, but less so than when using a larger training set.
        """
        # Define the approximator - full dataset used to fit the approximation
        approximator_full = coreax.approximation.ANNchorApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.data.shape[0],
            num_train_points=self.data.shape[0],
        )

        # Define the approximator - full dataset used to fit the approximation
        approximator_partial = coreax.approximation.ANNchorApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
            num_train_points=self.num_train_points,
        )

        # Approximate the kernel row mean using the full training set (so the
        # approximation should be very close to the true) and only part of the data for
        # training (so the error should grow)
        approximate_kernel_mean_full = approximator_full.approximate(self.data)
        approximate_kernel_mean_partial = approximator_partial.approximate(self.data)

        # Check the approximation is close to the true value
        np.testing.assert_array_almost_equal(
            approximate_kernel_mean_full, self.true_distances, decimal=0
        )
        np.testing.assert_array_almost_equal(
            approximate_kernel_mean_partial, self.true_distances, decimal=0
        )

        # Compute the approximation error and check if is better if we use more data
        approx_error_full = np.sum(
            np.square(self.true_distances - approximate_kernel_mean_full)
        )
        approx_error_partial = np.sum(
            np.square(self.true_distances - approximate_kernel_mean_partial)
        )
        self.assertTrue(approx_error_full <= approx_error_partial)

    def test_annchor_approximator_negative_num_kernel_points(self) -> None:
        """
        Test the class ANNchorApproximator rejects negative values of num_kernel_points.
        """
        # Define the approximator with a negative value of num_kernel_points
        approximator = coreax.approximation.ANNchorApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=-self.num_kernel_points,
            num_train_points=self.data.shape[0],
        )
        with self.assertRaises(ValueError) as error_raised:
            approximator.approximate(self.data)

        self.assertEqual(
            error_raised.exception.args[0], "num_kernel_points must be positive"
        )

    def test_annchor_approximator_zero_num_kernel_points(self) -> None:
        """
        Test the class ANNchorApproximator rejects zero values of num_kernel_points.
        """
        # Define the approximator with a zero value of num_kernel_points
        approximator = coreax.approximation.ANNchorApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=0,
            num_train_points=self.data.shape[0],
        )
        with self.assertRaises(ValueError) as error_raised:
            approximator.approximate(self.data)

        self.assertEqual(
            error_raised.exception.args[0],
            "num_kernel_points must be positive and non-zero",
        )

    def test_annchor_approximator_large_num_kernel_points(self) -> None:
        """
        Test the class ANNchorApproximator usage of large values of num_kernel_points.
        """
        # Define an approximator with a value of num_kernel_points larger than the
        # dataset provided
        approximator_more_than_num_data = coreax.approximation.ANNchorApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=100 * self.data.shape[0],
            num_train_points=self.data.shape[0],
        )
        result_more_than_num_data = approximator_more_than_num_data.approximate(
            self.data
        )

        # Define an approximator with a value of num_kernel_points equal to the size of
        # the dataset provided
        approximator_exact_num_data = coreax.approximation.ANNchorApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.data.shape[0],
            num_train_points=self.data.shape[0],
        )
        result_exactly_num_data = approximator_exact_num_data.approximate(self.data)

        # Check the output is very close if we use all the data provided, or ask for
        # more than the number of points we have
        approx_error_exact_num_data = np.sum(
            np.square(self.true_distances - result_exactly_num_data)
        )
        approx_error_more_than_num_data = np.sum(
            np.square(self.true_distances - result_more_than_num_data)
        )
        self.assertAlmostEqual(approx_error_exact_num_data, 0)
        self.assertAlmostEqual(approx_error_more_than_num_data, 0)

    def test_annchor_approximator_negative_num_train_points(self) -> None:
        """
        Test the class ANNchorApproximator rejects negative values of num_train_points.
        """
        # Define the approximator with a negative value of num_train_points
        approximator = coreax.approximation.ANNchorApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
            num_train_points=-self.data.shape[0],
        )
        with self.assertRaises(ValueError) as error_raised:
            approximator.approximate(self.data)

        self.assertEqual(
            error_raised.exception.args[0], "num_train_points must be positive"
        )

    def test_annchor_approximator_zero_num_train_points(self) -> None:
        """
        Test the class ANNchorApproximator does not fail with zero num_train_points.
        """
        # Define the approximator with a zero value of num_train_points
        approximator = coreax.approximation.ANNchorApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
            num_train_points=0,
        )

        # In the case of zero num_train_points, we expect zero features to be used
        # during the approximation, and this results in an approximation vector of zeros
        np.testing.assert_array_equal(
            approximator.approximate(self.data), jnp.zeros_like(self.true_distances)
        )

    def test_annchor_approximator_large_num_train_points(self) -> None:
        """
        Test the class ANNchorApproximator rejects too large values of num_train_points.
        """
        # Define the approximator with a value of num_train_points larger than the
        # dataset provided
        approximator = coreax.approximation.ANNchorApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
            num_train_points=100 * self.data.shape[0],
        )
        with self.assertRaises(ValueError) as error_raised:
            approximator.approximate(self.data)

        self.assertEqual(
            error_raised.exception.args[0],
            "num_train_points must be no larger than the number of points in "
            "the provided data",
        )

    def test_annchor_approximator_invalid_kernel(self) -> None:
        """
        Test the class ANNchorApproximator rejects an invalid kernel.
        """
        approximator = coreax.approximation.ANNchorApproximator(
            kernel=coreax.util.InvalidKernel,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
            num_train_points=self.data.shape[0],
        )
        with self.assertRaises(AttributeError) as error_raised:
            approximator.approximate(self.data)

        self.assertEqual(
            error_raised.exception.args[0],
            "type object 'InvalidKernel' has no attribute 'compute'",
        )

    def test_nystrom_approximator(self) -> None:
        """
        Verify Nystrom approximator performance on toy problem.

        This test verifies that, when the entire dataset is used to train the
        approximation, the result is very close to the true value. We further check if a
        subset of the data is used for training, that the approximation performance
        degrades compared to using a larger training set.
        """
        # Define the approximator - full dataset used to fit the approximation
        approximator_full = coreax.approximation.NystromApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.data.shape[0],
        )

        # Define the approximator - full dataset used to fit the approximation
        approximator_partial = coreax.approximation.NystromApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
        )

        # Approximate the kernel row mean using the full training set (so the
        # approximation should be very close to the true) and only part of the data for
        # training (so the error should grow)
        approximate_kernel_mean_full = approximator_full.approximate(self.data)
        approximate_kernel_mean_partial = approximator_partial.approximate(self.data)

        # Check the approximation is close to the true value. Note that this particular
        # approximation on this dataset is very poor when using a subset of points, so
        # we do not check if approximate_kernel_mean_partial is within some precision of
        # the true distances. Instead, we only check later in this test if there has
        # been an improvement in the approximation as we use more data.
        np.testing.assert_array_almost_equal(
            approximate_kernel_mean_full, self.true_distances, decimal=0
        )

        # Compute the approximation error and check if is better if we use more data
        approx_error_full = np.sum(
            np.square(self.true_distances - approximate_kernel_mean_full)
        )
        approx_error_partial = np.sum(
            np.square(self.true_distances - approximate_kernel_mean_partial)
        )
        self.assertTrue(approx_error_full <= approx_error_partial)

    def test_nystrom_approximator_negative_num_kernel_points(self) -> None:
        """
        Test the class NystromApproximator rejects negative values of num_kernel_points.
        """
        # Define the approximator with a negative value of num_kernel_points
        approximator = coreax.approximation.NystromApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=-self.num_kernel_points,
        )
        with self.assertRaises(ValueError) as error_raised:
            approximator.approximate(self.data)

        self.assertEqual(
            error_raised.exception.args[0], "num_kernel_points must be positive"
        )

    def test_nystrom_approximator_zero_num_kernel_points(self) -> None:
        """
        Test the class NystromApproximator rejects zero values of num_kernel_points.
        """
        # Define the approximator with a zero value of num_kernel_points
        approximator = coreax.approximation.NystromApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=0,
        )

        # In the case of zero num_train_points, we expect zero features to be used
        # during the approximation, and this results in an approximation vector of zeros
        np.testing.assert_array_equal(
            approximator.approximate(self.data), jnp.zeros_like(self.true_distances)
        )

    def test_nystrom_approximator_large_num_kernel_points(self) -> None:
        """
        Test the class NystromApproximator rejects large values of num_kernel_points.
        """
        # Define an approximator with a value of num_kernel_points larger than the
        # dataset provided
        approximator = coreax.approximation.NystromApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=100 * self.data.shape[0],
        )

        with self.assertRaises(ValueError) as error_raised:
            approximator.approximate(self.data)

        self.assertEqual(
            error_raised.exception.args[0],
            "num_kernel_points must be no larger than the number of points in "
            "the provided data",
        )

    def test_nystrom_approximator_invalid_kernel(self) -> None:
        """
        Test the class NystromApproximator rejects an invalid kernel.
        """
        approximator = coreax.approximation.NystromApproximator(
            kernel=coreax.util.InvalidKernel,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
        )
        with self.assertRaises(AttributeError) as error_raised:
            approximator.approximate(self.data)

        self.assertEqual(
            error_raised.exception.args[0],
            "type object 'InvalidKernel' has no attribute 'compute'",
        )


# pylint: enable=too-many-public-methods


if __name__ == "__main__":
    unittest.main()
