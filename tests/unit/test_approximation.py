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
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
from jax import random

import coreax.approximation as ca
import coreax.kernel as ck


class TestApproximations(unittest.TestCase):
    """
    Tests related to approximation.py classes & functions.
    """

    def setUp(self) -> None:
        """
        Define data shared across tests.
        """
        self.random_key = random.PRNGKey(10)

        # Setup a 'small' toy example that can be computed by hand
        self.data = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [-1.0, 0.0]])
        self.num_kernel_points = 3
        self.num_train_points = 3

        # Compute the true kernel mean distance by hand. Since we use a square distance
        # kernel, the first of these is:
        # [
        #   (0.0 - 0.0)^2 + (0.0 - 0.0)^2 +
        #   (0.0 - 0.5)^2 + (0.0 - 0.5)^2 +
        #   (0.0 - 1.0)^2 + (0.0 - 0.0)^2 +
        #   (0.0 - -1.0)^2 + (0.0 - 0.0)^2
        # ] / 4 = 0.625

        # We can repeat the above, but changing the point with which we are comparing
        # to get:
        self.true_distances = np.array([0.625, 0.875, 1.375, 1.875])

    def test_kernel_mean_approximator_creation(self) -> None:
        """
        Test the class KernelMeanApproximator initialises correctly.
        """
        # Patch the abstract method (approximate) of the KernelMeanApproximator so it
        # can be created
        p = patch.multiple(ca.KernelMeanApproximator, __abstractmethods__=set())
        p.start()

        # Define the approximator
        approximator = ca.KernelMeanApproximator(
            kernel_evaluation=ck.sq_dist,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
        )

        # Check parameters have been set
        self.assertEqual(approximator.kernel_evaluation, ck.sq_dist)
        self.assertEqual(approximator.random_key[0], self.random_key[0])
        self.assertEqual(approximator.random_key[1], self.random_key[1])
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
        approximator_full = ca.RandomApproximator(
            kernel_evaluation=ck.sq_dist,
            random_key=self.random_key,
            num_kernel_points=self.data.shape[0],
            num_train_points=self.data.shape[0],
        )

        # Define the approximator - full dataset used to fit the approximation
        approximator_partial = ca.RandomApproximator(
            kernel_evaluation=ck.sq_dist,
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

    def test_annchor_approximator(self) -> None:
        """
        Verify Annchor approximator performance on toy problem.

        This test verifies that, when the entire dataset is used to train the
        approximation, the result is very close to the true value. We further check if a
        subset of the data is used for training, that the approximation is still close
        to the true value, but less so than when using a larger training set.
        """
        # Define the approximator - full dataset used to fit the approximation
        approximator_full = ca.ANNchorApproximator(
            kernel_evaluation=ck.sq_dist,
            random_key=self.random_key,
            num_kernel_points=self.data.shape[0],
            num_train_points=self.data.shape[0],
        )

        # Define the approximator - full dataset used to fit the approximation
        approximator_partial = ca.ANNchorApproximator(
            kernel_evaluation=ck.sq_dist,
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

    def test_nystrom_approximator(self) -> None:
        """
        Verify Nystrom approximator performance on toy problem.

        This test verifies that, when the entire dataset is used to train the
        approximation, the result is very close to the true value. We further check if a
        subset of the data is used for training, that the approximation performance
        degrades compared to using a larger training set.
        """
        # Define the approximator - full dataset used to fit the approximation
        approximator_full = ca.NystromApproximator(
            kernel_evaluation=ck.sq_dist,
            random_key=self.random_key,
            num_kernel_points=self.data.shape[0],
        )

        # Define the approximator - full dataset used to fit the approximation
        approximator_partial = ca.NystromApproximator(
            kernel_evaluation=ck.sq_dist,
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


if __name__ == "__main__":
    unittest.main()
