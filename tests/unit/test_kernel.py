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
from functools import partial
from typing import Callable

import numpy as np
from jax import grad
from jax import numpy as jnp
from jax import vjp
from scipy.stats import norm, ortho_group

import coreax.kernel as ck


class TestKernels(unittest.TestCase):
    """
    Tests related to kernel.py functions & classes.
    """

    def test_rbf_kernel_init(self) -> None:
        r"""
        Test the class RBFKernel initilisation with a negative bandwidth.
        """
        # Create the kernel with a negative bandwidth - we expect a value error to be
        # raised
        self.assertRaises(ValueError, ck.RBFKernel, bandwidth=-1.0)

    def test_rbf_kernel_distance_two_floats(self) -> None:
        r"""
        Test the class RBFKernel distance computations.

        The RBF kernel is defined as :math:`k(x,y) = \exp (-||x-y||^2/2 * bandwidth)`.

        For the two input floats
        .. math::

            x = 0.5

            y = 2.0

        For our choices of x and y, we have:

        .. math::

            ||x - y||^2 &= (0.5 - 2.0)^2
                        &= 2.25

        If we take the bandwidth to be :math:`\pi / 2.0` we get:
            k(x, y) &= \exp(- 2.25 / \pi)
                    &= 0.48860678
        """
        # Define data and bandwidth
        x = 0.5
        y = 2.0
        bandwidth = np.float32(np.pi) / 2.0

        # Define the expected distance - it should just be a number in this case since
        # we have floats as inputs, so treat these single data-points in space
        expected_distance = 0.48860678

        # Create the kernel
        kernel = ck.RBFKernel(bandwidth=bandwidth)

        # Evaluate the kernel - which computes the distance between the two vectors x
        # and y
        output = kernel.compute(x, y)

        # Check the output matches the expected distance
        self.assertAlmostEqual(output, expected_distance, places=5)

    def test_rbf_kernel_distance_two_vectors(self) -> None:
        r"""
        Test the class RBFKernel distance computations.

        The RBF kernel is defined as :math:`k(x,y) = \exp (-||x-y||^2/2 * bandwidth)`.

        For the two input vectors
        .. math::

            x = [0, 1, 2, 3]

            y = [1, 2, 3, 4]

        For our choices of x and y, we have:

        .. math::

            ||x - y||^2 &= (0 - 1)^2 + (1 - 2)^2 + (2 - 3)^2 + (3 - 4)^2
                        &= 4

        If we take the bandwidth to be :math:`\pi / 2.0` we get:
            k(x, y) &= \exp(- 4 / \pi)
                    &= 0.279923327
        """
        # Define data and bandwidth
        x = 1.0 * np.arange(4)
        y = x + 1.0
        bandwidth = np.float32(np.pi) / 2.0

        # Define the expected distance - it should just be a number in this case since
        # we have 1-dimensional arrays, so treat these as single data-points in space
        expected_distance = 0.279923327

        # Create the kernel
        kernel = ck.RBFKernel(bandwidth=bandwidth)

        # Evaluate the kernel - which computes the distance between the two vectors x
        # and y
        output = kernel.compute(x, y)

        # Check the output matches the expected distance
        self.assertAlmostEqual(output, expected_distance, places=5)

    def test_rbf_kernel_distance_two_matrices(self) -> None:
        r"""
        Test the class RBFKernel distance computations.

        The RBF kernel is defined as :math:`k(x,y) = \exp (-||x-y||^2/2 * bandwidth)`.

        For the two input vectors
        .. math::

            x = [ [0, 1, 2, 3], [5, 6, 7, 8] ]

            y = [ [1, 2, 3, 4], [5, 6, 7, 8] ]

        For our choices of x and y, we have distances of:

        .. math::

            ||x - y||^2 = [4, 0]

        If we take the bandwidth to be :math:`\pi / 2.0` we get:
            k(x[0], y[0]) &= \exp(- 4 / \pi)
                          &= 0.279923327
            k(x[1 y[1]) &= \exp(- 0 / \pi)
                          &= 1.0

        """
        # Define data and bandwidth
        x = np.array(([0, 1, 2, 3], [5, 6, 7, 8]))
        y = np.array(([1, 2, 3, 4], [5, 6, 7, 8]))
        bandwidth = np.float32(np.pi) / 2.0

        # Define the expected distance - it should just be an array in this case since
        # we have an input that has multiple 'rows' and a defined second dimension
        # (column)
        expected_distances = np.array([0.279923327, 1.0])

        # Create the kernel
        kernel = ck.RBFKernel(bandwidth=bandwidth)

        # Evaluate the kernel - which computes the distance between the two vectors x
        # and y
        output = kernel.compute(x, y)

        # Check the output matches the expected distance
        np.testing.assert_array_almost_equal(output, expected_distances, decimal=5)

    def test_rbf_kernel_gradients_wrt_x(self) -> None:
        """
        Test the class RBFKernel gradient computations.

        The RBF kernel is defined as :math:`k(x,y) = \exp (-||x-y||^2/2 * bandwidth)`.
        The gradient of this with respect to x is:

        ..math:
            - \frac{(x - y)}{bandwidth^{3}\sqrt(2\pi)}e^{-\frac{|x-y|^2}{2 bandwidth^2}}
        """
        # Define some data
        bandwidth = 1 / np.sqrt(2)
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
                    / bandwidth**3
                    * np.exp(-np.linalg.norm(x_ - y_) ** 2 / (2 * bandwidth**2))
                    / (np.sqrt(2 * np.pi))
                )

        # Create the kernel
        kernel = ck.RBFKernel(bandwidth=bandwidth)

        # Evaluate the gradient
        output = kernel.grad_x(x, y)

        # Check output matches expected
        self.assertAlmostEqual(jnp.linalg.norm(true_gradients - output), 0.0, places=3)

    def test_rbf_kernel_gradients_wrt_y(self) -> None:
        """
        Test the class RBFKernel gradient computations.

        The RBF kernel is defined as :math:`k(x,y) = \exp (-||x-y||^2/2 * bandwidth)`.
        The gradient of this with respect to y is:

        ..math:
            \frac{(x - y)}{bandwidth^{3}\sqrt(2\pi)}e^{-\frac{|x-y|^2}{2 bandwidth^2}}
        """
        # Define some data
        bandwidth = 1 / np.sqrt(2)
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
                    / bandwidth**3
                    * np.exp(-np.linalg.norm(x_ - y_) ** 2 / (2 * bandwidth**2))
                    / (np.sqrt(2 * np.pi))
                )

        # Create the kernel
        kernel = ck.RBFKernel(bandwidth=bandwidth)

        # Evaluate the gradient
        output = kernel.grad_y(x, y)

        # Check output matches expected
        self.assertAlmostEqual(jnp.linalg.norm(true_gradients - output), 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
