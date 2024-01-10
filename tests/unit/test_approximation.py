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

import coreax.approximation
import coreax.kernel


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
        self.random_key = random.PRNGKey(10)

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

        # Check parameters have been set
        self.assertEqual(approximator.kernel, self.kernel)
        self.assertEqual(approximator.random_key[0], self.random_key[0])
        self.assertEqual(approximator.random_key[1], self.random_key[1])
        self.assertEqual(approximator.num_kernel_points, self.num_kernel_points)

    def test_kernel_mean_approximator_creation_invalid_types(self) -> None:
        """
        Test the class KernelMeanApproximator rejects invalid input types.
        """
        # Patch the abstract method (approximate) of the KernelMeanApproximator, so it
        # can be created
        p = patch.multiple(
            coreax.approximation.KernelMeanApproximator, __abstractmethods__=set()
        )
        p.start()

        # Define the approximator with an incorrect kernel type
        self.assertRaises(
            TypeError,
            coreax.approximation.KernelMeanApproximator,
            kernel="not_a_kernel",
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
        )

        # Define the approximator with an incorrect random_key type, but that can be
        # converted into an array
        approximator = coreax.approximation.KernelMeanApproximator(
            kernel=self.kernel,
            random_key=123,
            num_kernel_points=self.num_kernel_points,
        )
        np.testing.assert_array_equal(approximator.random_key, np.array([123]))

        # Define the approximator with an incorrect random_key type, that cannot be
        # cast as an array
        self.assertRaises(
            TypeError,
            coreax.approximation.KernelMeanApproximator,
            kernel=self.kernel,
            random_key=int,
            num_kernel_points=self.num_kernel_points,
        )

        # Define the approximator with an incorrect num_kernel_points type (float) but
        # that can be cast into an int
        approximator = coreax.approximation.KernelMeanApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=1.0 * self.num_kernel_points,
        )
        self.assertEqual(approximator.num_kernel_points, self.num_kernel_points)

        # Define the approximator with an incorrect num_kernel_points type (float) that
        # cannot be cast into an int
        self.assertRaises(
            TypeError,
            coreax.approximation.KernelMeanApproximator,
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=[1],
        )

        # Define the approximator with a negative value of num_kernel_points
        self.assertRaises(
            ValueError,
            coreax.approximation.KernelMeanApproximator,
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=-self.num_kernel_points,
        )

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

    def test_random_approximator_creation_invalid_types(self) -> None:
        """
        Test the class RandomApproximator rejects invalid input types.
        """
        # Define the approximator with an incorrect kernel type
        self.assertRaises(
            TypeError,
            coreax.approximation.RandomApproximator,
            kernel="not_a_kernel",
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
            num_train_points=self.data.shape[0],
        )

        # Define the approximator with an incorrect random_key type, but that can be
        # converted into an array.
        approximator = coreax.approximation.RandomApproximator(
            kernel=self.kernel,
            random_key=123,
            num_kernel_points=self.num_kernel_points,
            num_train_points=self.data.shape[0],
        )
        np.testing.assert_array_equal(approximator.random_key, np.array([123]))

        # Define the approximator with an incorrect random_key type, that cannot be
        # cast as an array
        self.assertRaises(
            TypeError,
            coreax.approximation.RandomApproximator,
            kernel=self.kernel,
            random_key=int,
            num_kernel_points=self.num_kernel_points,
            num_train_points=self.data.shape[0],
        )

        # Define the approximator with an incorrect num_kernel_points type (float) but
        # that can be cast into an int
        approximator = coreax.approximation.RandomApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=1.0 * self.num_kernel_points,
            num_train_points=self.data.shape[0],
        )
        self.assertEqual(approximator.num_kernel_points, self.num_kernel_points)

        # Define the approximator with an incorrect num_kernel_points type (float) that
        # cannot be cast into an int
        self.assertRaises(
            TypeError,
            coreax.approximation.RandomApproximator,
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=[1.0],
            num_train_points=self.data.shape[0],
        )

        # Define the approximator with a negative value of num_kernel_points
        self.assertRaises(
            ValueError,
            coreax.approximation.RandomApproximator,
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=-self.num_kernel_points,
            num_train_points=self.data.shape[0],
        )

        # Define the approximator with an incorrect num_train_points type (float) but
        # that can be cast to an int
        approximator = coreax.approximation.RandomApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
            num_train_points=10.0,
        )
        self.assertEqual(approximator.num_train_points, 10)

        # Define the approximator with an incorrect num_train_points type (float) and
        # that cannot be cast to an int
        self.assertRaises(
            TypeError,
            coreax.approximation.RandomApproximator,
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
            num_train_points=[10.0],
        )

        # Define the approximator with a negative value of num_train_points
        self.assertRaises(
            ValueError,
            coreax.approximation.RandomApproximator,
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=-self.num_kernel_points,
            num_train_points=-10,
        )

        # Define a valid approximator, but call approximate with an invalid input
        approximator = coreax.approximation.RandomApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.data.shape[0],
            num_train_points=self.data.shape[0],
        )
        self.assertRaises(TypeError, approximator.approximate, "not_data")

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

    def test_annchor_approximator_creation_invalid_types(self) -> None:
        """
        Test the class ANNchorApproximator rejects invalid input types.
        """
        # Define the approximator with an incorrect kernel type
        self.assertRaises(
            TypeError,
            coreax.approximation.ANNchorApproximator,
            kernel="not_a_kernel",
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
            num_train_points=self.data.shape[0],
        )

        # Define the approximator with an incorrect random_key type, but that can be
        # converted into an array.
        approximator = coreax.approximation.ANNchorApproximator(
            kernel=self.kernel,
            random_key=123,
            num_kernel_points=self.num_kernel_points,
            num_train_points=self.data.shape[0],
        )
        np.testing.assert_array_equal(approximator.random_key, np.array([123]))

        # Define the approximator with an incorrect random_key type, that cannot be
        # cast as an array
        self.assertRaises(
            TypeError,
            coreax.approximation.ANNchorApproximator,
            kernel=self.kernel,
            random_key=int,
            num_kernel_points=self.num_kernel_points,
            num_train_points=self.data.shape[0],
        )

        # Define the approximator with an incorrect num_kernel_points type (float) but
        # that can be cast into an int
        approximator = coreax.approximation.ANNchorApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=1.0 * self.num_kernel_points,
            num_train_points=self.data.shape[0],
        )
        self.assertEqual(approximator.num_kernel_points, self.num_kernel_points)

        # Define the approximator with an incorrect num_kernel_points type (float) that
        # cannot be cast into an int
        self.assertRaises(
            TypeError,
            coreax.approximation.ANNchorApproximator,
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=[1.0],
            num_train_points=self.data.shape[0],
        )

        # Define the approximator with a negative value of num_kernel_points
        self.assertRaises(
            ValueError,
            coreax.approximation.ANNchorApproximator,
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=-self.num_kernel_points,
            num_train_points=self.data.shape[0],
        )

        # Define the approximator with an incorrect num_train_points type (float) but
        # that can be cast to an int
        approximator = coreax.approximation.ANNchorApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
            num_train_points=10.0,
        )
        self.assertEqual(approximator.num_train_points, 10)

        # Define the approximator with an incorrect num_train_points type (float) and
        # that cannot be cast to an int
        self.assertRaises(
            TypeError,
            coreax.approximation.ANNchorApproximator,
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
            num_train_points=[10.0],
        )

        # Define the approximator with a negative value of num_train_points
        self.assertRaises(
            ValueError,
            coreax.approximation.ANNchorApproximator,
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=-self.num_kernel_points,
            num_train_points=-10,
        )

        # Define a valid approximator, but call approximate with an invalid input
        approximator = coreax.approximation.ANNchorApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.data.shape[0],
            num_train_points=self.data.shape[0],
        )
        self.assertRaises(TypeError, approximator.approximate, "not_data")

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

    def test_nystrom_approximator_creation_invalid_types(self) -> None:
        """
        Test the class NystromApproximator rejects invalid input types.
        """
        # Define the approximator with an incorrect kernel type
        self.assertRaises(
            TypeError,
            coreax.approximation.NystromApproximator,
            kernel="not_a_kernel",
            random_key=self.random_key,
            num_kernel_points=self.num_kernel_points,
        )

        # Define the approximator with an incorrect random_key type, but that can be
        # converted into an array
        approximator = coreax.approximation.NystromApproximator(
            kernel=self.kernel,
            random_key=123,
            num_kernel_points=self.num_kernel_points,
        )
        np.testing.assert_array_equal(approximator.random_key, np.array([123]))

        # Define the approximator with an incorrect random_key type, that cannot be
        # cast as an array
        self.assertRaises(
            TypeError,
            coreax.approximation.NystromApproximator,
            kernel=self.kernel,
            random_key=int,
            num_kernel_points=self.num_kernel_points,
        )

        # Define the approximator with an incorrect num_kernel_points type (float) but
        # that can be cast into an int
        approximator = coreax.approximation.NystromApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=1.0 * self.num_kernel_points,
        )
        self.assertEqual(approximator.num_kernel_points, self.num_kernel_points)

        # Define the approximator with an incorrect num_kernel_points type (float) that
        # cannot be cast into an int
        self.assertRaises(
            TypeError,
            coreax.approximation.NystromApproximator,
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=[1],
        )

        # Define the approximator with a negative value of num_kernel_points
        self.assertRaises(
            ValueError,
            coreax.approximation.NystromApproximator,
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=-self.num_kernel_points,
        )

        # Define a valid approximator, but call approximate with an invalid input
        approximator = coreax.approximation.NystromApproximator(
            kernel=self.kernel,
            random_key=self.random_key,
            num_kernel_points=self.data.shape[0],
        )
        self.assertRaises(TypeError, approximator.approximate, "not_data")


if __name__ == "__main__":
    unittest.main()
