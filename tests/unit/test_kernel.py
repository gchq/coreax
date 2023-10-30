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

import numpy as np
import scipy.stats
from jax import numpy as jnp
from jax.typing import ArrayLike

import coreax.approximation as ca
import coreax.kernel as ck


class TestKernelABC(unittest.TestCase):
    """
    Tests related to the Kernel abstract base class in kernel.py
    """

    def test_create_approximator(self) -> None:
        """
        Test creation of approximation object within the Kernel class.
        """
        # Patch the abstract methods of the Kernel ABC so it can be created
        p = patch.multiple(ck.Kernel, __abstractmethods__=set())
        p.start()

        # Define known approximator names
        known_approximators = {
            "random": ca.RandomApproximator,
            "nystrom": ca.NystromApproximator,
            "annchor": ca.ANNchorApproximator,
        }

        # Create the kernel
        kernel = ck.Kernel()

        # Call the approximation method with an invalid approximator string
        self.assertRaises(KeyError, kernel.create_approximator, approximator="example")

        # Call the approximation method with each known approximator name
        for name, approx_type in known_approximators.items():
            self.assertTrue(
                isinstance(kernel.create_approximator(approximator=name), approx_type)
            )

        # Pre-create a KernelMeanApproximator and check that it is returned when passed
        self.assertTrue(
            isinstance(
                kernel.create_approximator(approximator=ca.RandomApproximator),
                ca.RandomApproximator,
            )
        )


class TestSquaredExponentialKernel(unittest.TestCase):
    """
    Tests related to the SquaredExponentialKernel defined in kernel.py
    """

    def test_se_kernel_init(self) -> None:
        r"""
        Test the class SquaredExponentialKernel initilisation with a negative lengthscale.
        """
        # Create the kernel with a negative lengthscale - we expect a value error to be
        # raised
        self.assertRaises(ValueError, ck.SquaredExponentialKernel, lengthscale=-1.0)

    def test_se_kernel_compute_two_floats(self) -> None:
        r"""
        Test the class SquaredExponentialKernel distance computations.

        The SquaredExponential kernel is defined as :math:`k(x,y) = \exp (-||x-y||^2/2 * lengthscale)`.

        For the two input floats
        .. math::

            x = 0.5

            y = 2.0

        For our choices of x and y, we have:

        .. math::

            ||x - y||^2 &= (0.5 - 2.0)^2
                        &= 2.25

        If we take the lengthscale to be :math:`\pi / 2.0` we get:
            k(x, y) &= \exp(- 2.25 / \pi)
                    &= 0.48860678

        If the lengthscale is instead taken to be :math:`\pi`, we get:
            k(x, y) &= \exp(- 2.25 / (2.0\pi))
                    &= 0.6990041
        """
        # Define data and lengthscale
        x = 0.5
        y = 2.0
        lengthscale = np.sqrt(np.float32(np.pi) / 2.0)

        # Define the expected distance - it should just be a number in this case since
        # we have floats as inputs, so treat these single data-points in space
        expected_distance = 0.48860678

        # Create the kernel
        kernel = ck.SquaredExponentialKernel(lengthscale=lengthscale)

        # Evaluate the kernel - which computes the distance between the two vectors x
        # and y
        output = kernel.compute(x, y)

        # Check the output matches the expected distance
        self.assertAlmostEqual(output[0, 0], expected_distance, places=5)

        # Alter the lengthscale, and check the jit decorator catches the update
        kernel.lengthscale = np.sqrt(np.float32(np.pi))

        # Set expected output with this new lengthscale
        expected_distance = 0.6990041

        # Evaluate the kernel - which computes the distance between the two vectors x
        # and y, with the new, altered lengthscale
        output = kernel.compute(x, y)

        # Check the output matches the expected distance
        self.assertAlmostEqual(output[0, 0], expected_distance, places=5)

    def test_se_kernel_compute_two_vectors(self) -> None:
        r"""
        Test the class SquaredExponentialKernel distance computations.

        The SquaredExponential kernel is defined as :math:`k(x,y) = \exp (-||x-y||^2/2 * lengthscale)`.

        For the two input vectors
        .. math::

            x = [0, 1, 2, 3]

            y = [1, 2, 3, 4]

        For our choices of x and y, we have:

        .. math::

            ||x - y||^2 &= (0 - 1)^2 + (1 - 2)^2 + (2 - 3)^2 + (3 - 4)^2
                        &= 4

        If we take the lengthscale to be :math:`\pi / 2.0` we get:
            k(x, y) &= \exp(- 4 / \pi)
                    &= 0.279923327
        """
        # Define data and lengthscale
        x = 1.0 * np.arange(4)
        y = x + 1.0
        lengthscale = np.sqrt(np.float32(np.pi) / 2.0)

        # Define the expected distance - it should just be a number in this case since
        # we have 1-dimensional arrays, so treat these as single data-points in space
        expected_distance = 0.279923327

        # Create the kernel
        kernel = ck.SquaredExponentialKernel(lengthscale=lengthscale)

        # Evaluate the kernel - which computes the distance between the two vectors x
        # and y
        output = kernel.compute(x, y)

        # Check the output matches the expected distance
        self.assertAlmostEqual(output[0, 0], expected_distance, places=5)

    def test_se_kernel_compute_two_arrays(self) -> None:
        r"""
        Test the class SquaredExponentialKernel distance computations on arrays.

        The SquaredExponential kernel is defined as :math:`k(x,y) = \exp (-||x-y||^2/2 * lengthscale)`.

        For the two input vectors
        .. math::

            x = [ [0, 1, 2, 3], [5, 6, 7, 8] ]

            y = [ [1, 2, 3, 4], [5, 6, 7, 8] ]

        For our choices of x and y, we have distances of:

        .. math::

            ||x - y||^2 = [4, 0]

        If we take the lengthscale to be :math:`\sqrt(\pi / 2.0` we get:
            k(x[0], y[0]) &= \exp(- 4 / \pi)
                          &= 0.279923327
            k(x[0], y[1]) &= \exp(- 100 / \pi)
                          &= 1.4996075 \times 10^{-14}
            k(x[1], y[0]) &= \exp(- 64 / \pi)
                          &= 1.4211038 \times 10^{-9}
            k(x[1], y[1]) &= \exp(- 0 / \pi)
                          &= 1.0

        """
        # Define data and lengthscale
        x = np.array(([0, 1, 2, 3], [5, 6, 7, 8]))
        y = np.array(([1, 2, 3, 4], [5, 6, 7, 8]))
        lengthscale = np.sqrt(np.float32(np.pi) / 2.0)

        # Define the expected Gram matrix
        expected_distances = np.array(
            [[0.279923327, 1.4996075e-14], [1.4211038e-09, 1.0]]
        )

        # Create the kernel
        kernel = ck.SquaredExponentialKernel(lengthscale=lengthscale)

        # Evaluate the kernel - which computes the Gram matrix between x and y
        output = kernel.compute(x, y)

        # Check the output matches the expected distance
        np.testing.assert_array_almost_equal(output, expected_distances, decimal=5)

    def test_se_kernel_gradients_wrt_x(self) -> None:
        """
        Test the class SquaredExponentialKernel gradient computations.

        The SquaredExponential kernel is defined as :math:`k(x,y) = \exp (-||x-y||^2/2 * lengthscale)`.
        The gradient of this with respect to x is:

        ..math:
            - \frac{(x - y)}{lengthscale^{3}\sqrt(2\pi)}e^{-\frac{|x-y|^2}{2 lengthscale^2}}
        """
        # Define some data
        lengthscale = 1 / np.sqrt(2)
        num_points = 10
        dimension = 2
        x = np.random.random((num_points, dimension))
        y = np.random.random((num_points, dimension))

        # Compute the actual gradients of the kernel with respect to x
        true_gradients = np.zeros((num_points, num_points, dimension))
        for i, x_ in enumerate(x):
            for j, y_ in enumerate(y):
                true_gradients[i, j] = (
                    -(x_ - y_)
                    / lengthscale**2
                    * np.exp(-np.linalg.norm(x_ - y_) ** 2 / (2 * lengthscale**2))
                )

        # Create the kernel
        kernel = ck.SquaredExponentialKernel(lengthscale=lengthscale)

        # Evaluate the gradient
        output = kernel.grad_x(x, y)

        # Check output matches expected
        self.assertAlmostEqual(
            float(jnp.linalg.norm(true_gradients - output)), 0.0, places=3
        )

    def test_se_kernel_gradients_wrt_y(self) -> None:
        """
        Test the class SquaredExponentialKernel gradient computations.

        The SquaredExponential kernel is defined as :math:`k(x,y) = \exp (-||x-y||^2/2 * lengthscale)`.
        The gradient of this with respect to y is:

        ..math:
            \frac{(x - y)}{lengthscale^{3}\sqrt(2\pi)}e^{-\frac{|x-y|^2}{2 lengthscale^2}}
        """
        # Define some data
        lengthscale = 1 / np.sqrt(2)
        num_points = 10
        dimension = 2
        x = np.random.random((num_points, dimension))
        y = np.random.random((num_points, dimension))

        # Compute the actual gradients of the kernel with respect to y
        true_gradients = np.zeros((num_points, num_points, dimension))
        for i, x_ in enumerate(x):
            for j, y_ in enumerate(y):
                true_gradients[i, j] = (
                    (x_ - y_)
                    / lengthscale**2
                    * np.exp(-np.linalg.norm(x_ - y_) ** 2 / (2 * lengthscale**2))
                )

        # Create the kernel
        kernel = ck.SquaredExponentialKernel(lengthscale=lengthscale)

        # Evaluate the gradient
        output = kernel.grad_y(x, y)

        # Check output matches expected
        self.assertAlmostEqual(
            float(jnp.linalg.norm(true_gradients - output)), 0.0, places=3
        )

    # def test_se_kernel_log_pdf_gradients_wrt_x(self) -> None:
    #     """
    #     Test the class SquaredExponentialKernel score-function gradient computations.
    #     """
    #     # Define parameters for data
    #     lengthscale = 1 / np.sqrt(2)
    #     num_points = 10
    #     dimension = 2
    #     x = np.random.random((num_points, dimension))

    #     # Sample points to build the distribution from
    #     kde_points = np.random.random((num_points, dimension))

    #     # Define a kernel density estimate function for the distribution
    #     kde = lambda input_var: (
    #         np.exp(
    #             -np.linalg.norm(input_var - kde_points, axis=1)[:, None] ** 2
    #             / (2 * lengthscale**2)
    #         )
    #         / (np.sqrt(2 * np.pi) * lengthscale)
    #     ).mean(axis=0)

    #     # Define a matrix to store gradients in - this is a jacobian because it's the
    #     # matrix of partial derivatives
    #     jacobian = np.zeros((num_points, dimension))

    #     # For each data-point in x, compute the gradients
    #     for i, x_ in enumerate(x):
    #         jacobian[i] = (
    #             -(x_ - kde_points)
    #             / lengthscale**3
    #             * np.exp(
    #                 -np.linalg.norm(x_ - kde_points, axis=1)[:, None] ** 2
    #                 / (2 * lengthscale**2)
    #             )
    #             / (np.sqrt(2 * np.pi))
    #         ).mean(axis=0) / (kde(x_)[:, None])

    #     # Compute the gradient of the score function using the class
    #     kernel = ck.SquaredExponentialKernel(lengthscale=lengthscale)
    #     output = kernel.grad_log_x(x, kde_points, lengthscale)

    #     # Check output matches expected
    #     np.testing.assert_array_almost_equal(output, jacobian, decimal=3)

    def test_define_pairwise_kernel_evaluation_no_grads(self) -> None:
        """
        Test the definition of pairwise kernel evaluation functions, without gradients.

        Pairwise distances mean, given two input arrays, we should return a matrix
        where the values correspond to the distance, as judged by the kernel, between
        each point in the first array and each point in the second array.

        The SquaredExponential kernel is defined as :math:`k(x,y) = \exp (-||x-y||^2/2 * lengthscale)`.
        If we have two input arrays:

        ..math:
            x = [0.0, 1.0, 2.0, 3.0, 4.0]
            y = [10.0, 3.0, 0.0]

        then we expect an output matrix with 5 rows and 3 columns. Entry [0, 0] in that
        matrix is the kernel distance between points 0.0 and 10.0, entry [0, 1] is the
        kernel distance between 0.0 and 3.0 and so on.
        """
        # Define parameters for data
        lengthscale = 1 / np.sqrt(2)
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
        y = np.array([10.0, 3.0, 0.0]).reshape(-1, 1)

        # Define the kernel object
        kernel = ck.SquaredExponentialKernel(lengthscale=lengthscale)

        # Define expected output for pairwise distances
        expected_output = np.zeros([5, 3])
        for x_index, x_ in enumerate(x):
            for y_index, y_ in enumerate(y):
                expected_output[x_index, y_index] = kernel.compute(x_, y_)[0, 0]

        # Compute the pairwise distances between the data using the kernel
        output = kernel.compute(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_update_kernel_matrix_row_sum(self) -> None:
        """
        Test updates to the kernel matrix row sum.

        In this test, we consider a subset of points (determined by the parameter
        max_size) and then compute the sum of pairwise distances between them. Any
        points outside this subset should not alter the existing kernel row sum values.
        """
        # Define parameters for data
        lengthscale = 1 / np.sqrt(2)
        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]).reshape(-1, 1)
        max_size = 3

        # Pre-specify an empty kernel matrix row sum to update as we go
        kernel_row_sum = jnp.zeros(len(x))

        # Define the kernel object
        kernel = ck.SquaredExponentialKernel(lengthscale=lengthscale)

        # Define expected output for pairwise distances - in this case we are looking
        # from the start index (0) to start index + max_size (0 + 3) in both axis (rows
        # and columns) and then adding the pairwise distances up to this point. Any
        # pairs of points beyond this subset of indices should not be computed
        expected_output = np.zeros([5, 5])
        for x_1_index, x_1 in enumerate(x[0:max_size]):
            for x_2_index, x_2 in enumerate(x[0:max_size]):
                expected_output[x_1_index, x_2_index] = kernel.compute(x_1, x_2)[0, 0]
        expected_output = expected_output.sum(axis=1)

        # Compute the kernel matrix row sum with the class
        output = kernel.update_kernel_matrix_row_sum(
            x, kernel_row_sum, 0, 0, kernel.compute, max_size=max_size
        )

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_calculate_kernel_matrix_row_sum(self) -> None:
        """
        Test computation of the kernel matrix row sum.

        We compute the distance between all pairs of points, and then sum these
        distances, giving a single value for each data-point.
        """
        # Define parameters for data
        lengthscale = 1 / np.sqrt(2)
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Define the kernel object
        kernel = ck.SquaredExponentialKernel(lengthscale=lengthscale)

        # Define expected output for pairwise distances
        expected_output = np.zeros([5, 5])
        for x_1_index, x_1 in enumerate(x):
            for x_2_index, x_2 in enumerate(x):
                expected_output[x_1_index, x_2_index] = kernel.compute(x_1, x_2)[0, 0]
        expected_output = expected_output.sum(axis=1)

        # Compute the kernel matrix row sum with the class
        output = kernel.calculate_kernel_matrix_row_sum(x)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_calculate_kernel_matrix_row_sum_mean(self) -> None:
        """
        Test computation of the mean of the kernel matrix row sum.

        We compute the distance between all pairs of points, and then take the mean of
        these distances, giving a single value for each data-point.
        """
        # Define parameters for data
        lengthscale = 1 / np.sqrt(2)
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Define the kernel object
        kernel = ck.SquaredExponentialKernel(lengthscale=lengthscale)

        # Define expected output for pairwise distances, then take the mean of them
        expected_output = np.zeros([5, 5])
        for x_1_index, x_1 in enumerate(x):
            for x_2_index, x_2 in enumerate(x):
                expected_output[x_1_index, x_2_index] = kernel.compute(x_1, x_2)[0, 0]
        expected_output = expected_output.mean(axis=1)

        # Compute the kernel matrix row sum mean with the class
        output = kernel.calculate_kernel_matrix_row_sum_mean(x)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output)

    def test_compute_normalised(self) -> None:
        """
        Test computation of normalised SquaredExponential kernel.

        A normalised SquaredExponential kernel is also known as a Gaussian kernel. We generate data and
        compare to a standard implementation of the Gaussian PDF.
        """
        # Setup some data
        std_dev = np.e
        num_points = 10
        x = np.arange(num_points)
        y = x + 1.0

        # Compute expected output using standard implementation of the Gaussian PDF
        expected_output = np.zeros((num_points, num_points))
        for i, x_ in enumerate(x):
            for j, y_ in enumerate(y):
                expected_output[i, j] = scipy.stats.norm(y_, std_dev).pdf(x_)

        # Compute the normalised PDF output using the kernel class
        kernel = ck.SquaredExponentialKernel(
            lengthscale=std_dev, scale=1 / (np.sqrt(2 * np.pi) * std_dev)
        )
        output = kernel.compute(x.reshape(-1, 1), y.reshape(-1, 1))

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    # def test_construct_pdf(self) -> None:
    #     """
    #     Test construction of PDF using SquaredExponential kernel.
    #     """
    #     # Setup some data
    #     std_dev = np.e
    #     num_points = 10
    #     x = np.arange(num_points)
    #     y = x + 1.0

    #     # Compute expected output using standard implementation of the Gaussian PDF
    #     expected_output = np.zeros((num_points, num_points))
    #     for i, x_ in enumerate(x):
    #         for j, y_ in enumerate(y):
    #             expected_output[i, j] = scipy.stats.norm(y_, std_dev).pdf(x_)

    #     # Compute the normalised PDF output using the kernel class
    #     kernel = ck.SquaredExponentialKernel(lengthscale=std_dev)
    #     output_mean, output_kernel = kernel.construct_pdf(x, y)

    #     # Check output matches expected
    #     np.testing.assert_array_almost_equal(
    #         output_mean, expected_output.mean(axis=1), decimal=3
    #     )
    #     np.testing.assert_array_almost_equal(output_kernel, expected_output, decimal=3)

    def test_define_pairwise_kernel_evaluation_with_grads(self) -> None:
        """
        Test the definition of pairwise kernel evaluation functions, with gradients.

        # TODO: What is a sensible test-case for this? Functionality may not work.
        """
        pass

    def test_se_div_x_grad_y(self) -> None:
        """
        Test the divergence wrt x of kernel Jacobian wrt y
        """
        # Setup data
        lengthscale = 1 / np.sqrt(2)
        num_points = 10
        dimension = 2
        x = np.random.random((num_points, dimension))
        y = np.random.random((num_points, dimension))

        # Define expected output
        expected_output = np.zeros((num_points, num_points))
        for i, x_ in enumerate(x):
            for j, y_ in enumerate(y):
                dp = np.dot(x_ - y_, x_ - y_)
                expected_output[i, j] = 2 * np.exp(-dp) * (dimension - 2 * dp)
        # Compute output using Kernel class
        kernel = ck.SquaredExponentialKernel(lengthscale=lengthscale)
        output = kernel.divergence_x_grad_y(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)


class TestPCIMQKernel(unittest.TestCase):
    """
    Tests related to the PCIMQKernel defined in kernel.py
    """

    def test_pcimq_kernel_init(self) -> None:
        r"""
        Test the class PCIMQKernel initilisation with a negative lengthscale.
        """
        # Create the kernel with a negative lengthscale - we expect a value error to be
        # raised
        self.assertRaises(ValueError, ck.PCIMQKernel, lengthscale=-1.0)

    def test_pcimq_kernel_compute(self) -> None:
        r"""
        Test the class PCIMQKernel distance computations.

        The PCIMQ kernel is defined as
        :math:`k(x,y) = \frac{1.0}{1.0 / \sqrt(1.0 + ((x - y) / std_dev) ** 2 / 2.0)}`.
        """
        # Define input data
        std_dev = np.e
        num_points = 10
        x = np.arange(num_points).reshape(-1, 1)
        y = x + 1.0

        # Compute expected output
        expected_output = np.zeros((num_points, num_points))
        for i, x_ in enumerate(x):
            for j, y_ in enumerate(y):
                expected_output[i, j] = 1.0 / np.sqrt(
                    1.0 + ((x_[0] - y_[0]) / std_dev) ** 2 / 2.0
                )

        # Compute distance using the kernel class
        kernel = ck.PCIMQKernel(lengthscale=std_dev)
        output = kernel.compute(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    def test_pcimq_kernel_gradients_wrt_x(self) -> None:
        r"""
        Test the class PCIMQ gradient computations with respect to x.
        """
        # Setup data
        lengthscale = 1 / np.sqrt(2)
        num_points = 10
        dimension = 2
        x = np.random.random((num_points, dimension))
        y = np.random.random((num_points, dimension))

        # Define expected output
        expected_output = np.zeros((num_points, num_points, dimension))
        for i, x_ in enumerate(x):
            for j, y_ in enumerate(y):
                expected_output[i, j] = -(x_ - y_) / (
                    1 + np.linalg.norm(x_ - y_) ** 2
                ) ** (3 / 2)

        # Compute output using Kernel class
        kernel = ck.PCIMQKernel(lengthscale=lengthscale)
        output = kernel.grad_x(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    def test_pcimq_kernel_gradients_wrt_y(self) -> None:
        """
        Test the class PCIMQ gradient computations with respect to y.
        """
        # Setup data
        lengthscale = 1 / np.sqrt(2)
        num_points = 10
        dimension = 2
        x = np.random.random((num_points, dimension))
        y = np.random.random((num_points, dimension))

        # Define expected output
        expected_output = np.zeros((num_points, num_points, dimension))
        for i, x_ in enumerate(x):
            for j, y_ in enumerate(y):
                expected_output[i, j] = (x_ - y_) / (
                    1 + np.linalg.norm(x_ - y_) ** 2
                ) ** (3 / 2)

        # Compute output using Kernel class
        kernel = ck.PCIMQKernel(lengthscale=lengthscale)
        output = kernel.grad_y(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    def test_pcimq_div_x_grad_y(self) -> None:
        """
        Test the divergence wrt x of kernel Jacobian wrt y
        """
        # Setup data
        lengthscale = 1 / np.sqrt(2)
        num_points = 10
        dimension = 2
        x = np.random.random((num_points, dimension))
        y = np.random.random((num_points, dimension))

        # Define expected output
        expected_output = np.zeros((num_points, num_points))
        for i, x_ in enumerate(x):
            for j, y_ in enumerate(y):
                dp = np.dot(x_ - y_, x_ - y_)
                den = (1 + dp) ** (3 / 2)
                expected_output[i, j] = dimension / den - 3 * dp / den ** (5 / 3)
        # Compute output using Kernel class
        kernel = ck.PCIMQKernel(lengthscale=lengthscale)
        output = kernel.divergence_x_grad_y(x, y)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)


class TestSteinKernel(unittest.TestCase):
    """
    Tests related to the SteinKernel defined in kernel.py
    """

    def test_stein_kernel_computation(self) -> None:
        r"""
        Test computation of the SteinKernel.

        Due to the complexity of the Stein kernel, we check the size of the output
        matches the expected size, not the numerical values in the output array itself.
        """
        # Setup some data
        num_points_x = 10
        num_points_y = 5
        dimension = 2
        lengthscale = 1 / np.sqrt(2)

        def score_function(x):
            return -x

        # Setup data
        x = np.random.random((num_points_x, dimension))
        y = np.random.random((num_points_y, dimension))

        # Set expected output sizes
        expected_size = (10, 5)

        # Compute output using Kernel class
        kernel = ck.SteinKernel(
            base_kernel=ck.PCIMQKernel(lengthscale=lengthscale),
            score_function=score_function,
        )
        output = kernel.compute(x, y)

        # Check output sizes match the expected
        self.assertEqual(output.shape, expected_size)

    def test_stein_kernel_element_computation(self) -> None:
        r"""
        Test computation of a single element of the SteinKernel.
        """
        # Setup some data
        num_points_x = 10
        num_points_y = 5
        dimension = 2
        lengthscale = 1 / np.sqrt(2)
        beta = 0.5

        def score_function(x):
            return -x

        def k_x_y(x, y):
            r"""
            The Stein kernel.

            The base kernel is :math:`(1 + \lvert \mathbf{x} - \mathbf{y}
            \rvert^2)^{-1/2}`. :math:`\mathbb{P}` is :math:`\mathcal{N}(0, \mathbf{I})`
            with :math:`\nabla \log f_X(\mathbf{x}) = -\mathbf{x}`.

            In the code: l, m and r refer to shared denominators in the Stein kernel
            equation (rather than divergence, x_, y_ and z in the main code function).

            :math:`k_\mathbb{P}(\mathbf{x}, \mathbf{y}) = l + m + r`.

            :math:`l := -\frac{3 \lvert \mathbf{x} - \mathbf{y} \rvert^2}{(1 + \lvert
            \mathbf{x} - \mathbf{y} \rvert^2)^{5/2}}`.

            :math:`m := 2\beta\left[ \frac{d + [\mathbf{y} -
            \mathbf{x}]^\intercal[\mathbf{x} - \mathbf{y}]}{(1 + \lvert \mathbf{x} -
            \mathbf{y} \rvert^2)^{3/2}} \right]`.

            :math:`r := \frac{\mathbf{x}^\intercal \mathbf{y}}{(1 + \lvert \mathbf{x} -
            \mathbf{y} \rvert^2)^{1/2}}`.

            :param x: a d-dimensional vector
            :param y: a d-dimensional vector
            :return: kernel evaluated at x, y
            """
            norm_sq = np.linalg.norm(x - y) ** 2
            l = -3 * norm_sq / (1 + norm_sq) ** 2.5
            m = (
                2
                * beta
                * (dimension + np.dot(score_function(x) - score_function(y), x - y))
                / (1 + norm_sq) ** 1.5
            )
            r = np.dot(score_function(x), score_function(y)) / (1 + norm_sq) ** 0.5
            return l + m + r

        # Setup data
        x = np.random.random((num_points_x, dimension))
        y = np.random.random((num_points_y, dimension))

        # Compute output using Kernel class
        kernel = ck.SteinKernel(
            base_kernel=ck.PCIMQKernel(lengthscale=lengthscale),
            score_function=score_function,
        )

        # Compute the output step-by-step with the element method
        expected_output = np.zeros([x.shape[0], y.shape[0]])
        output = kernel.compute(x, y)
        for i, x_ in enumerate(x):
            for j, y_ in enumerate(y):
                # Compute via our hand-coded kernel evaluation
                expected_output[i, j] = k_x_y(x_, y_)

        # Check output matches the expected
        np.testing.assert_array_almost_equal(output, expected_output)


if __name__ == "__main__":
    unittest.main()
