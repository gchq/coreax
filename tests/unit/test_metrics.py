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
Tests for computations of metrics for coresets.

Metrics evaluate the quality of a coreset by some measure. The tests within this file
verify that metric computations produce the expected results on simple examples.
"""

import unittest
from unittest.mock import MagicMock, NonCallableMagicMock, patch

import jax.numpy as jnp
from jax import random

import coreax.kernel
import coreax.metrics
import coreax.util

# pylint: disable=too-many-public-methods


class TestMetrics(unittest.TestCase):
    r"""
    Tests related to Metric abstract base class in metrics.py.
    """

    def test_metric_creation(self) -> None:
        r"""
        Test the class Metric initialises correctly.
        """
        # Disable pylint warning for abstract-class-instantiated as we are patching
        # these whilst testing creation of the parent class
        # pylint: disable=abstract-class-instantiated
        # Patch the abstract methods of the Metric ABC, so it can be created
        p = patch.multiple(coreax.metrics.Metric, __abstractmethods__=set())
        p.start()

        # Create a metric object
        metric = coreax.metrics.Metric()
        # pylint: enable=abstract-class-instantiated

        # Check the compute method exists
        self.assertTrue(hasattr(metric, "compute"))


class TestMMD(unittest.TestCase):
    """
    Tests related to the maximum mean discrepancy (MMD) class in metrics.py.
    """

    # Disable pylint warning for too-many-instance-attributes as we use each of these in
    # subsequent tests, variable names ensure human readability and understanding
    # pylint: disable=too-many-instance-attributes
    def setUp(self):
        r"""
        Generate data for shared use across unit tests.

        Generate ``num_points_x`` random points in ``d`` dimensions from a uniform
        distribution [0, 1).
        Randomly select ``num_points_y`` points for second dataset ``y``.
        Generate weights: ``weights_x`` for original data, ``weights_y`` for second
        dataset ``y``.

        :num_points_x: Number of test data points
        :d: Dimension of data
        :num_points_y: Number of points to randomly select for second dataset ``y``
        :block_size: Maximum number of points for block calculations
        """
        # Define data parameters
        self.num_points_x = 30
        self.dimension = 10
        self.num_points_y = 5
        self.block_size = 3

        # Define example datasets
        self.x = random.uniform(
            random.key(0), shape=(self.num_points_x, self.dimension)
        )
        self.y = random.choice(random.key(0), self.x, shape=(self.num_points_y,))
        self.weights_x = (
            random.uniform(random.key(0), shape=(self.num_points_x,))
            / self.num_points_x
        )
        self.weights_y = (
            random.uniform(random.key(0), shape=(self.num_points_y,))
            / self.num_points_y
        )

    def test_mmd_compare_same_data(self) -> None:
        r"""
        Test the MMD of a dataset with itself is zero, for several different kernels.
        """
        # Define a metric object using the SquaredExponentialKernel
        metric = coreax.metrics.MMD(
            kernel=coreax.kernel.SquaredExponentialKernel(length_scale=1.0)
        )
        self.assertAlmostEqual(float(metric.compute(self.x, self.x)), 0.0)

        # Define a metric object using the LaplacianKernel
        metric = coreax.metrics.MMD(
            kernel=coreax.kernel.LaplacianKernel(length_scale=1.0)
        )
        self.assertAlmostEqual(float(metric.compute(self.x, self.x)), 0.0)

        # Define a metric object using the PCIMQKernel
        metric = coreax.metrics.MMD(kernel=coreax.kernel.PCIMQKernel(length_scale=1.0))
        self.assertAlmostEqual(float(metric.compute(self.x, self.x)), 0.0)

    def test_mmd_ones(self):
        r"""
        Test MMD computation with a small example dataset of ones and zeros.

        For the dataset of 4 points in 2 dimensions, :math:`x`, and another dataset
        :math:`y`, given by:

        .. math::

            x = [[0,0], [1,1], [0,0], [1,1]]

            y = [[0,0], [1,1]]

        the Gaussian (aka radial basis function) kernel,
        :math:`k(x,y) = \exp (-||x-y||^2/2\sigma^2)`, gives:

        .. math::

            k(x,x) = \exp(-\begin{bmatrix}0 & 2 & 0 & 2 \\ 2 & 0 & 2 & 0\\ 0 & 2 & 0 &
            2 \\ 2 & 0 & 2 & 0\end{bmatrix}/2\sigma^2).

            k(y,y) = \exp(-\begin{bmatrix}0 & 2  \\ 2 & 0 \end{bmatrix}/2\sigma^2).

            k(x,y) = \exp(-\begin{bmatrix}0 & 2  \\ 2 & 0 \\0 & 2  \\ 2 & 0
            \end{bmatrix}/2\sigma^2).

        Then

        .. math::

            \text{MMD}^2(x,y) = \mathbb{E}(k(x,x)) + \mathbb{E}(k(y,y)) -
            2\mathbb{E}(k(x,y))

            = \frac{1}{2} + e^{-1/2} + \frac{1}{2} + e^{-1/2} - 2\left(\frac{1}{2}
            + e^{-1/2}\right)

            = 0.
        """
        # Setup data
        x = jnp.array([[0, 0], [1, 1], [0, 0], [1, 1]])
        y = jnp.array([[0, 0], [1, 1]])
        length_scale = 1.0

        # Set expected MMD
        expected_output = 0.0

        # Define a metric object using an RBF kernel
        metric = coreax.metrics.MMD(
            kernel=coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)
        )

        # Compute MMD using the metric object
        output = metric.compute(x=x, y=y)

        # Check output matches expected
        self.assertAlmostEqual(float(output), expected_output, places=5)

    def test_mmd_ints(self):
        r"""
        Test MMD computation with a small example dataset of integers.

        For the dataset of 3 points in 2 dimensions, :math:`x`, and second dataset
        :math:`y`, given by:

        .. math::

            x = [[0,0], [1,1], [2,2]]

            y = [[0,0], [1,1]]

        the RBF kernel, :math:`k(x,y) = \exp (-||x-y||^2/2\sigma^2)`, gives:

        .. math::

            k(x,x) = \exp(-\begin{bmatrix}0 & 2 & 8 \\ 2 & 0 & 2 \\ 8 & 2 & 0
            \end{bmatrix}/2\sigma^2) = \begin{bmatrix}1 & e^{-1} & e^{-4} \\ e^{-1} &
            1 & e^{-1} \\ e^{-4} & e^{-1} & 1\end{bmatrix}.

            k(y,y) = \exp(-\begin{bmatrix}0 & 2 \\ 2 & 0 \end{bmatrix}/2\sigma^2) =
             \begin{bmatrix}1 & e^{-1}\\ e^{-1} & 1\end{bmatrix}.

            k(x,y) =  \exp(-\begin{bmatrix}0 & 2 & 8 \\ 2 & 0 & 2 \end{bmatrix}
            /2\sigma^2) = \begin{bmatrix}1 & e^{-1} \\  e^{-1} & 1 \\ e^{-4} & e^{-1}
            \end{bmatrix}.

        Then

        .. math::

            \text{MMD}^2(x,y) = \mathbb{E}(k(x,x)) + \mathbb{E}(k(y,y)) -
            2\mathbb{E}(k(x,y))

            = \frac{3+4e^{-1}+2e^{-4}}{9} + \frac{2 + 2e^{-1}}{2} -2 \times
            \frac{2 + 3e^{-1}+e^{-4}}{6}

            = \frac{3 - e^{-1} -2e^{-4}}{18}.
        """
        # Setup data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])
        length_scale = 1.0

        # Set expected MMD
        expected_output = jnp.sqrt((3 - jnp.exp(-1) - 2 * jnp.exp(-4)) / 18)

        # Define a metric object using an RBF kernel
        metric = coreax.metrics.MMD(
            kernel=coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)
        )

        # Compute MMD using the metric object
        output = metric.compute(x=x, y=y)

        # Check output matches expected
        self.assertAlmostEqual(float(output), float(expected_output), places=5)

    def test_mmd_rand(self):
        r"""
        Test MMD computed from randomly generated test data agrees with method result.
        """
        # Define a kernel object
        length_scale = 1.0
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Compute each term in the MMD formula
        kernel_nn = kernel.compute(self.x, self.x)
        kernel_mm = kernel.compute(self.y, self.y)
        kernel_nm = kernel.compute(self.x, self.y)

        # Compute overall MMD by
        expected_mmd = (
            kernel_nn.mean() + kernel_mm.mean() - 2 * kernel_nm.mean()
        ) ** 0.5

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute MMD using the metric object
        output = metric.compute(x=self.x, y=self.y)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_mmd, places=5)

    def test_weighted_mmd_ints(self) -> None:
        r"""
        Test weighted MMD computation with a small example dataset of integers.

        Weighted mmd is calculated if and only if weights_y are provided. When
        `weights_y` = :data:`None`, the MMD class computes the standard, non-weighted
        MMD.

        For the dataset of 3 points in 2 dimensions :math:`x`, second dataset :math:`y`,
        and weights for this second dataset :math:`w_y`, given by:

        .. math::

            x = [[0,0], [1,1], [2,2]]

            y = [[0,0], [1,1]]

            w_y = [1,0]

        the weighted maximum mean discrepancy is calculated via:

        .. math::

            \text{WMMD}^2(x,y) = \mathbb{E}(k(x,x)) + w_y^T k(y,y) w_y
             - 2\mathbb{E}_x(k(x,y)) w_y

            = \frac{3+4e^{-1}+2e^{-4}}{9} + 1 - 2 \times \frac{1 + e^{-1} + e^{-4}}{3}

            = \frac{2}{3} - \frac{2}{9}e^{-1} - \frac{4}{9}e^{-4}.
        """
        # Setup data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])
        weights_y = jnp.array([1, 0])
        length_scale = 1.0

        # Define expected output
        expected_output = jnp.sqrt(
            2 / 3 - (2 / 9) * jnp.exp(-1) - (4 / 9) * jnp.exp(-4)
        )

        # Define a metric object
        metric = coreax.metrics.MMD(
            kernel=coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)
        )

        # Compute weighted mmd using the metric object
        output = metric.compute(x=x, y=y, weights_y=weights_y)

        # Check output matches expected
        self.assertAlmostEqual(float(output), float(expected_output), places=5)

    def test_weighted_mmd_rand(self) -> None:
        r"""
        Test weighted MMD computations on randomly generated test data.
        """
        # Define a kernel object
        length_scale = 1.0
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Compute each term in the MMD formula
        kernel_nn = kernel.compute(self.x, self.x)
        kernel_mm = kernel.compute(self.y, self.y)
        kernel_nm = kernel.compute(self.x, self.y)

        # Define expected output
        expected_output = (
            jnp.mean(kernel_nn)
            + jnp.dot(self.weights_y.T, jnp.dot(kernel_mm, self.weights_y))
            - 2 * jnp.dot(self.weights_y.T, kernel_nm.mean(axis=0))
        ).item() ** 0.5

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute weighted MMD using the metric object
        output = metric.compute(x=self.x, y=self.y, weights_y=self.weights_y)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=5)

    def test_weighted_mmd_uniform_weights(self) -> None:
        r"""
        Test that weighted MMD equals MMD if weights are uniform, :math:`w_y = 1/m`.
        """
        # Define a kernel object
        length_scale = 1.0
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute weighted MMD with all weights being uniform
        uniform_wmmd = metric.compute(
            self.x,
            self.y,
            weights_y=jnp.ones(self.num_points_y) / self.num_points_y,
        )

        # Compute MMD without the weights
        mmd = metric.compute(self.x, self.y)

        # Check uniform weighted MMD and MMD without weights give the same result
        self.assertAlmostEqual(float(uniform_wmmd), float(mmd), places=5)

    def test_sum_pairwise_distances(self) -> None:
        r"""
        Test sum_pairwise_distances() with a small integer example.

        For the dataset of 3 points in 2 dimensions :math:`x`, and second dataset
        :math:`y`:

        .. math::

            x = [[0,0], [1,1], [2,2]]

            y = [[0,0], [1,1]]

        the pairwise square distances are given by the matrix:

        .. math::

            \begin{bmatrix}0 & 2 \\ 2 & 0 \\ 8 & 2 \end{bmatrix}

        which, summing across both axes, gives the result :math:`14`.
        """
        # Setup data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])

        # Set expected output
        expected_output = 14

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute the sum of pairwise distances using the metric object
        output = metric.sum_pairwise_distances(x=x, y=y, block_size=2)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=5)

    def test_sum_pairwise_distances_large_block_size(self) -> None:
        r"""
        Test sum_pairwise_distances() with a block size larger than the input data.

        This test uses the same data as `test_sum_pairwise_distances` above, and so
        expects to get exactly the same output. The difference between the tests is,
        here ``block_size`` is large enough to not need block-wise computation.
        """
        # Setup data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])

        # Set expected output
        expected_output = 14

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute the sum of pairwise distances using the metric object
        output = metric.sum_pairwise_distances(x=x, y=y, block_size=2000)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=5)

    def test_sum_pairwise_distances_zero_block_size(self):
        """
        Test sum_pairwise_distances when a zero block size is given.
        """
        # Define a metric object
        metric = coreax.metrics.MMD(kernel=MagicMock())

        # Compute sum of pairwise distances with a zero block_size - this should
        # raise an error highlighting that block_size should be a positive integer
        with self.assertRaises(ValueError) as error_raised:
            metric.sum_pairwise_distances(x=self.x, y=self.y, block_size=0)
        self.assertEqual(
            error_raised.exception.args[0],
            "block_size must be a positive integer",
        )

    def test_sum_pairwise_distances_negative_block_size(self):
        """
        Test sum_pairwise_distances when a negative block size is given.
        """
        # Define a metric object
        metric = coreax.metrics.MMD(kernel=MagicMock())

        # Compute sum of pairwise distances with a negative block_size - this should
        # raise an error highlighting that block_size should be a positive integer
        with self.assertRaises(ValueError) as error_raised:
            metric.sum_pairwise_distances(x=self.x, y=self.y, block_size=-5)
        self.assertEqual(
            error_raised.exception.args[0],
            "block_size must be a positive integer",
        )

    def test_sum_pairwise_distances_float_block_size(self):
        """
        Test sum_pairwise_distances when a float block size is given.
        """
        # Define a metric object
        metric = coreax.metrics.MMD(kernel=MagicMock())

        # Compute sum of pairwise distances with a float block_size - this should
        # raise an error highlighting that block_size should be a positive integer
        with self.assertRaises(TypeError) as error_raised:
            metric.sum_pairwise_distances(x=self.x, y=self.y, block_size=2.0)
        self.assertEqual(
            error_raised.exception.args[0],
            "block_size must be a positive integer",
        )

    def test_maximum_mean_discrepancy_block_ints(self) -> None:
        r"""
        Test maximum_mean_discrepancy_block while limiting memory requirements.

        This test uses the same 2D, three-point dataset and second dataset as
        test_mmd_ints().
        """
        # Setup data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])

        # Define expected output
        expected_output = jnp.sqrt((3 - jnp.exp(-1) - 2 * jnp.exp(-4)) / 18)

        # Define a kernel object
        length_scale = 1.0
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute MMD block-wise
        mmd_block_test = metric.compute(x=x, y=y, block_size=2)

        # Check output matches expected
        self.assertAlmostEqual(float(mmd_block_test), float(expected_output), places=5)

    def test_maximum_mean_discrepancy_block_rand(self) -> None:
        r"""
        Test that mmd block-computed for random test data equals method output.
        """
        # Define a kernel object
        length_scale = 1.0
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Compute MMD term with x and itself
        kernel_nn = 0.0
        for i1 in range(0, self.num_points_x, self.block_size):
            for i2 in range(0, self.num_points_x, self.block_size):
                kernel_nn += kernel.compute(
                    self.x[i1 : i1 + self.block_size, :],
                    self.x[i2 : i2 + self.block_size, :],
                ).sum()

        # Compute MMD term with y and itself
        kernel_mm = 0.0
        for j1 in range(0, self.num_points_y, self.block_size):
            for j2 in range(0, self.num_points_y, self.block_size):
                kernel_mm += kernel.compute(
                    self.y[j1 : j1 + self.block_size, :],
                    self.y[j2 : j2 + self.block_size, :],
                ).sum()

        # Compute MMD term with x and y
        kernel_nm = 0.0
        for i in range(0, self.num_points_x, self.block_size):
            for j in range(0, self.num_points_y, self.block_size):
                kernel_nm += kernel.compute(
                    self.x[i : i + self.block_size, :],
                    self.y[j : j + self.block_size, :],
                ).sum()

        # Compute expected output from MMD
        expected_output = (
            kernel_nn / self.num_points_x**2
            + kernel_mm / self.num_points_y**2
            - 2 * kernel_nm / (self.num_points_x * self.num_points_y)
        ) ** 0.5

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute MMD
        output = metric.compute(self.x, self.y, block_size=self.block_size)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=5)

    def test_maximum_mean_discrepancy_equals_maximum_mean_discrepancy_block(
        self,
    ) -> None:
        r"""
        Test that MMD computations agree when done block-wise and all at once.
        """
        # Define a kernel object
        length_scale = 1.0
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Check outputs are the same
        self.assertAlmostEqual(
            float(metric.compute(self.x, self.y)),
            float(metric.compute(self.x, self.y, block_size=self.block_size)),
            places=5,
        )

    def test_maximum_mean_discrepancy_block_zero_block_size(self) -> None:
        """
        Test maximum_mean_discrepancy_block when given a zero block size.
        """
        # Define a metric object
        metric = coreax.metrics.MMD(kernel=MagicMock())

        # Compute MMD with a zero block_size - the full text of the inbuilt error raised
        # from the range function should provide all information needed for users to
        # identify the issue
        with self.assertRaises(ValueError) as error_raised:
            metric.maximum_mean_discrepancy_block(x=self.x, y=self.y, block_size=0)
        self.assertEqual(
            error_raised.exception.args[0],
            "block_size must be a positive integer",
        )

    def test_maximum_mean_discrepancy_block_negative_block_size(self) -> None:
        """
        Test maximum_mean_discrepancy_block when given a negative block size.
        """
        # Define a metric object
        metric = coreax.metrics.MMD(kernel=MagicMock())

        # Compute MMD with a negative block_size - this should raise an error
        # highlighting that block_size should be a positive integer
        with self.assertRaises(ValueError) as error_raised:
            metric.maximum_mean_discrepancy_block(x=self.x, y=self.y, block_size=-2)
        self.assertEqual(
            error_raised.exception.args[0],
            "block_size must be a positive integer",
        )

    def test_maximum_mean_discrepancy_block_float_block_size(self) -> None:
        """
        Test maximum_mean_discrepancy_block when given a float block size.
        """
        # Define a metric object
        metric = coreax.metrics.MMD(kernel=MagicMock())

        # Compute MMD with a float block_size - an error should be raised highlighting
        # this should be a positive integer
        with self.assertRaises(TypeError) as error_raised:
            metric.maximum_mean_discrepancy_block(x=self.x, y=self.y, block_size=1.0)
        self.assertEqual(
            error_raised.exception.args[0],
            "block_size must be a positive integer",
        )

    def test_sum_weighted_pairwise_distances(self) -> None:
        r"""
        Test sum_weighted_pairwise_distances, which calculates w^T*K*w matrices.

        Computations are done in blocks of size block_size.

        For the dataset of 3 points in 2 dimensions :math:`x`, and second dataset
        :math:`y`:

        .. math::

            x = [[0,0], [1,1], [2,2]]

            y = [[0,0], [1,1]]

        the pairwise square distances are given by the matrix:

        .. math::

            k(x, y) = \begin{bmatrix}0 & 2 \\ 2 & 0 \\ 8 & 2 \end{bmatrix}.

        Then, for weights vectors:

        .. math::

            w = [0.5, 0.5, 0],

            w_y = [1, 0]

        the product :math:`w^T*k(x, y)*w_y = 1`.
        """
        # Setup some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])
        weights_x = jnp.array([0.5, 0.5, 0])
        weights_y = jnp.array([1, 0])

        # Define expected output
        expected_output = 1.0

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute sum of weighted pairwise distances
        output = metric.sum_weighted_pairwise_distances(
            x=x, y=y, weights_x=weights_x, weights_y=weights_y, block_size=2
        )

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=5)

    def test_sum_weighted_pairwise_distances_big_block_size(self) -> None:
        r"""
        Test sum_weighted_pairwise_distances with block size larger than the input data.

        This test uses the same data as `test_sum_weighted_pairwise_distances` above,
        and so expects to get exactly the same output. The difference between the tests
        is, here ``block_size`` is large enough to not need block-wise computation.
        """
        # Setup some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])
        weights_x = jnp.array([0.5, 0.5, 0])
        weights_y = jnp.array([1, 0])

        # Define expected output
        expected_output = 1.0

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute sum of weighted pairwise distances
        output = metric.sum_weighted_pairwise_distances(
            x=x, y=y, weights_x=weights_x, weights_y=weights_y, block_size=2000
        )

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=5)

    def test_sum_weighted_pairwise_distances_zero_block_size(self):
        """
        Test sum_weighted_pairwise_distances when a zero block size is given.
        """
        # Define a metric object
        metric = coreax.metrics.MMD(kernel=MagicMock())

        # Compute sum of weighted pairwise distances with a zero block_size - this
        # should error and state a positive integer is required
        with self.assertRaises(ValueError) as error_raised:
            metric.sum_weighted_pairwise_distances(
                x=self.x,
                y=self.y,
                weights_x=self.weights_x,
                weights_y=self.weights_y,
                block_size=0,
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "block_size must be a positive integer",
        )

    def test_sum_weighted_pairwise_distances_negative_block_size(self):
        """
        Test sum_weighted_pairwise_distances when a negative block size is given.
        """
        # Define a metric object
        metric = coreax.metrics.MMD(kernel=MagicMock())

        # Compute sum of weighted pairwise distances with a negative block_size - this
        # should raise an error highlighting that block_size should be a positive
        # integer
        with self.assertRaises(ValueError) as error_raised:
            metric.sum_weighted_pairwise_distances(
                x=self.x,
                y=self.y,
                weights_x=self.weights_x,
                weights_y=self.weights_y,
                block_size=-5,
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "block_size must be a positive integer",
        )

    def test_sum_weighted_pairwise_distances_float_block_size(self):
        """
        Test sum_weighted_pairwise_distances when a float block size is given.
        """
        # Define a metric object
        metric = coreax.metrics.MMD(kernel=MagicMock())

        # Compute sum of weighted pairwise distances with a float block_size - this
        # should raise an error highlighting that block_size should be a positive
        # integer
        with self.assertRaises(TypeError) as error_raised:
            metric.sum_weighted_pairwise_distances(
                x=self.x,
                y=self.y,
                weights_x=self.weights_x,
                weights_y=self.weights_y,
                block_size=2.0,
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "block_size must be a positive integer",
        )

    def test_weighted_maximum_mean_discrepancy_block_int(self) -> None:
        r"""
        Test the weighted_maximum_mean_discrepancy_block computations.

        For
        .. math::

            x = [[0,0], [1,1], [2,2]],

            y = [[0,0], [1,1]],

            w^T = [0.5, 0.5, 0],

            w_y^T = [1, 0],

        the weighted maximum mean discrepancy is given by:

        .. math::

            \text{WMMD}^2(x, x) =
            w^T k(x, x) w + w_y^T k(y, y) w_y - 2 w^T k(x, y) w_y

        which, when :math:`k(x,y)` is the RBF kernel, reduces to:

        .. math::

            \frac{1}{2} + \frac{e^{-1}}{2} + 1 - 2\times(\frac{1}{2} + \frac{e^{-1}}{2})

            = \frac{1}{2} - \frac{e^{-1}}{2}.
        """
        # Define some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])
        weights_x = jnp.array([0.5, 0.5, 0])
        weights_y = jnp.array([1, 0])

        # Define expected output
        expected_output = jnp.sqrt(1 / 2 - jnp.exp(-1) / 2)

        # Define a kernel object
        length_scale = 1.0
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute weighted MMD block-wise
        output = metric.compute(
            x=x, y=y, weights_x=weights_x, weights_y=weights_y, block_size=2
        )

        # Check output matches expected
        self.assertAlmostEqual(float(output), float(expected_output), places=5)

    def test_weighted_maximum_mean_discrepancy_block_equals_mmd(self) -> None:
        r"""
        Test weighted MMD computations with uniform weights.

        One expects that, when weights are uniform: :math:`w = 1/n` and
        :math:`w_y = 1/m`, the output of weighted_maximum_mean_discrepancy_block should
        be the same as an unweighted computation.
        """
        # Define a kernel object
        length_scale = 1.0
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute weighted MMD with uniform weights
        output = metric.compute(
            self.x,
            self.y,
            weights_x=jnp.ones(self.num_points_x) / self.num_points_x,
            weights_y=jnp.ones(self.num_points_y) / self.num_points_y,
            block_size=self.block_size,
        )

        # Check output matches expected
        self.assertAlmostEqual(
            float(output), float(metric.compute(self.x, self.y)), places=5
        )

    def test_weighted_maximum_mean_discrepancy_block_zero_block_size(self) -> None:
        """
        Test weighted_maximum_mean_discrepancy_block when given a zero block size.
        """
        # Define a metric object
        metric = coreax.metrics.MMD(kernel=MagicMock())

        # Compute weighted MMD with a zero block_size - this should error as we require
        # a positive integer
        with self.assertRaises(ValueError) as error_raised:
            metric.weighted_maximum_mean_discrepancy_block(
                x=self.x,
                y=self.y,
                weights_x=self.weights_x,
                weights_y=self.weights_y,
                block_size=0,
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "block_size must be a positive integer",
        )

    def test_weighted_maximum_mean_discrepancy_block_negative_block_size(self) -> None:
        """
        Test weighted_maximum_mean_discrepancy_block when given a negative block size.
        """
        # Define a metric object
        metric = coreax.metrics.MMD(kernel=MagicMock())

        # Compute weighted MMD with a negative block_size - this should raise an error
        # highlighting that block_size should be a positive integer
        with self.assertRaises(ValueError) as error_raised:
            metric.weighted_maximum_mean_discrepancy_block(
                x=self.x,
                y=self.y,
                weights_x=self.weights_x,
                weights_y=self.weights_y,
                block_size=-2,
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "block_size must be a positive integer",
        )

    def test_weighted_maximum_mean_discrepancy_block_float_block_size(self) -> None:
        """
        Test weighted_maximum_mean_discrepancy_block when given a float block size.
        """
        # Define a metric object
        metric = coreax.metrics.MMD(kernel=MagicMock())

        # Compute weighted MMD with a float block_size - this should raise an error
        # explaining this parameter must be a positive integer
        with self.assertRaises(TypeError) as error_raised:
            metric.weighted_maximum_mean_discrepancy_block(
                x=self.x,
                y=self.y,
                weights_x=self.weights_x,
                weights_y=self.weights_y,
                block_size=2.0,
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "block_size must be a positive integer",
        )

    def test_compute_zero_block_size(self) -> None:
        """
        Test compute when given a zero block size.
        """
        # Define a metric object
        metric = coreax.metrics.MMD(kernel=MagicMock())

        # Compute MMD with a zero block_size - this should raise an error explaining
        # this parameter must be a positive integer
        with self.assertRaises(ValueError) as error_raised:
            metric.compute(x=self.x, y=self.y, block_size=0)
        self.assertEqual(
            error_raised.exception.args[0],
            "block_size must be a positive integer",
        )

    def test_compute_negative_block_size(self) -> None:
        """
        Test compute when given a negative block size.
        """
        # Define a metric object
        metric = coreax.metrics.MMD(kernel=MagicMock())

        # Compute MMD with a negative block_size - this should raise an error to
        # highlight this must be a positive integer
        with self.assertRaises(ValueError) as error_raised:
            metric.compute(x=self.x, y=self.y, block_size=-2)
        self.assertEqual(
            error_raised.exception.args[0],
            "block_size must be a positive integer",
        )

    def test_compute_float_block_size(self) -> None:
        """
        Test compute when given a float block size.
        """
        # Define a metric object
        metric = coreax.metrics.MMD(kernel=MagicMock())

        # Compute MMD with a float block_size - the full text of the inbuilt error
        # raised from the range function should provide all information needed for users
        # to identify the issue
        with self.assertRaises(TypeError) as error_raised:
            metric.compute(x=self.x, y=self.y, block_size=2.0)
        self.assertEqual(
            error_raised.exception.args[0],
            "block_size must be a positive integer",
        )


class TestCMMD(unittest.TestCase):
    """
    Tests for the conditional maximum mean discrepancy (CMMD) class in metrics.py.
    """

    # Disable pylint warning for too-many-instance-attributes as we use each of these in
    # subsequent tests, variable names ensure human readability and understanding
    # pylint: disable=too-many-instance-attributes
    def setUp(self) -> None:
        r"""
        Generate data for shared use across unit tests.

        Generate two supervised datasets of size ``dataset_size_1`` and
        ``dataset_size_2`` with feature dimension of size ``feature_dimension``,
        and response dimension of size ``response_dimension``.

        :dataset_size_1: Number of data points in first dataset
        :dataset_size_2: Number of data points in second dataset
        :feature_dimension: Dimension of feature space
        :response_dimension: Dimension of response space
        :regularisation_parameters: A  :math:`1 \times 2` array of regularisation
            parameters corresponding to the original dataset :math:`\mathcal{D}^{(1)}`
            and the coreset :math:`\mathcal{D}^{(2)}` respectively
        :precision_threshold: Positive threshold we compare against for precision
        """
        # Define data parameters
        self.dataset_size_1 = 10
        self.dataset_size_2 = 15
        self.feature_dimension = 5
        self.response_dimension = 5
        self.regularisation_parameters = jnp.array([1e-6, 1e-6])
        self.precision_threshold = 1e-6

        # Generate first multi-output supervised dataset
        x1 = jnp.sin(
            10
            * random.uniform(
                random.key(0),
                shape=(
                    self.dataset_size_1,
                    self.feature_dimension,
                ),
            )
        )
        coefficients1 = random.uniform(
            random.key(0),
            shape=(
                self.feature_dimension,
                self.response_dimension,
            ),
        )
        errors1 = random.normal(
            random.key(0),
            shape=(
                self.dataset_size_1,
                self.response_dimension,
            ),
        )
        y1 = x1 @ coefficients1 + 0.1 * errors1
        self.dataset_1 = jnp.hstack((x1, y1))

        # Generate second multi-output supervised dataset
        x2 = jnp.sin(
            10
            * random.uniform(
                random.key(1),
                shape=(
                    self.dataset_size_2,
                    self.feature_dimension,
                ),
            )
        )
        coefficients2 = random.uniform(
            random.key(1),
            shape=(
                self.feature_dimension,
                self.response_dimension,
            ),
        )
        errors2 = random.normal(
            random.key(1),
            shape=(
                self.dataset_size_2,
                self.response_dimension,
            ),
        )
        y2 = x2 @ coefficients2 + 0.1 * errors2
        self.dataset_2 = jnp.hstack((x2, y2))

    def test_cmmd_compare_same_data(self) -> None:
        r"""
        Test the CMMD of a dataset with itself is zero, for several different kernels.
        """
        # Define a metric object using the SquaredExponentialKernel
        metric = coreax.metrics.CMMD(
            feature_kernel=coreax.kernel.SquaredExponentialKernel(length_scale=1.0),
            response_kernel=coreax.kernel.SquaredExponentialKernel(length_scale=1.0),
            num_feature_dimensions=self.feature_dimension,
            regularisation_parameters=self.regularisation_parameters,
            precision_threshold=self.precision_threshold,
        )
        self.assertAlmostEqual(
            float(metric.compute(self.dataset_1, self.dataset_1)), 0.0
        )

        # Define a metric object using the LaplacianKernel
        metric = coreax.metrics.CMMD(
            feature_kernel=coreax.kernel.LaplacianKernel(length_scale=1.0),
            response_kernel=coreax.kernel.LaplacianKernel(length_scale=1.0),
            num_feature_dimensions=self.feature_dimension,
            regularisation_parameters=self.regularisation_parameters,
            precision_threshold=self.precision_threshold,
        )
        self.assertAlmostEqual(
            float(metric.compute(self.dataset_1, self.dataset_1)), 0.0
        )

        # Define a metric object using the PCIMQKernel
        metric = coreax.metrics.CMMD(
            feature_kernel=coreax.kernel.PCIMQKernel(length_scale=1.0),
            response_kernel=coreax.kernel.PCIMQKernel(length_scale=1.0),
            num_feature_dimensions=self.feature_dimension,
            regularisation_parameters=self.regularisation_parameters,
            precision_threshold=self.precision_threshold,
        )
        self.assertAlmostEqual(
            float(metric.compute(self.dataset_1, self.dataset_1)), 0.0
        )

    def test_cmmd_ints_no_regularisation(self) -> None:
        r"""
        Test CMMD computation with a small dataset of integers with no regularisation.

        For the first dataset of 3 pairs in 2 feature dimensions and 1 response
        dimension, :math:`\mathcal{D}_1` given by:
        .. math::

            x_1 = [[0,0], [1,1]]
            y_1 = [[0], [1]]

        and a second dataset of 2 pairs in 2 feature dimensions and 1 response
        dimension, :math:`\mathcal{D}_2` given by:

        .. math::

            x_2 = [[1,1], [3,3]]
            y_2 = [[1], [3]]

        the Gaussian (aka radial basis function) kernels with length-scale equal to one,
        :math:`k(a,b) = l(a,b) = \exp (-||a-b||^2/2)` gives:

        .. math::

            k(x_1_x_1) := K_{11} = \begin{bmatrix}1 & e^{-1}\\ e^{-1} & 1\end{bmatrix},

            K_{11}^{-1} = \frac{1}{1 - e^{-2}}\begin{bmatrix}1 & -e^{-1}\\ -e^{-1} & 1
            \end{bmatrix},

            K_{22} = \begin{bmatrix}1 & e^{-4}\\ e^{-4} & 1\end{bmatrix},

            K_{22}^{-1} = \frac{1}{1 - e^{-8}}\begin{bmatrix}1 & -e^{-4}\\ -e^{-4} & 1
            \end{bmatrix},

            K_{21} = \begin{bmatrix}e^{-1} & 1\\ e^{-9} & e^{-4}\end{bmatrix},

            L_{11} = \begin{bmatrix}1 & e^{-1/2}\\ e^{-1/2} &1\end{bmatrix},

            L_{22} = \begin{bmatrix}1 & e^{-2}\\ e^{-2} &1\end{bmatrix},

            L_{12} = \begin{bmatrix}e^{-1/2} & e^{-9/2}\\ 1 & e^{-2}\end{bmatrix}.

        Then

        .. math::

            \text{CMMD}(\mathcal{D}_1, \mathcal{D}_2) = \sqrt{\text{Tr}(T_1) +
            \text{Tr}(T_2) - 2\text{Tr}(T_3)}

        where

        .. math::

            T_1 := K_{11}^{-1}L_{11}K_{11}^{-1}K_{11} = K_{11}^{-1}L_{11}

            = \frac{1}{1 - e^{-2}}\begin{bmatrix}1 & -e^{-1}\\ -e^{-1} & 1\end{bmatrix}
            \begin{bmatrix}1 & e^{-1/2}\\ e^{-1/2} &1\end{bmatrix}

            = \frac{1}{1 - e^{-2}}\begin{bmatrix}1 - e^{-3/2} & e^{-1/2} - e^{-1}\\
            e^{-1/2} - e^{-1} & 1 - e^{-3/2}\end{bmatrix}

            \implies \text{Tr}(T_1) = \frac{2(1 - e^{-3/2})}{1 - e^{-2}}


            T_2 := K_{22}^{-1}L_{22}K_{22}^{-1}K_{22} = K_{22}^{-1}L_{22}

            =\frac{1}{1 - e^{-8}}\begin{bmatrix}1 & -e^{-4}\\ -e^{-4} & 1\end{bmatrix}
            \begin{bmatrix}1 & e^{-2}\\ e^{-2} &1\end{bmatrix}

            = \frac{1}{1 - e^{-8}}\begin{bmatrix}1 - e^{-6} & e^{-2} - e^{-4}\\ e^{-2}
            - e^{-4} & 1 - e^{-6}\end{bmatrix}

            \implies \text{Tr}(T_2) = \frac{2(1 - e^{-6})}{1 - e^{-8}}


            T_3 := K_{21}^{-1}K_{11}^{-1}L_{12}K_{22}^{-1}

            = \frac{1}{(1 - e^{-2})(1 - e^{-8})}
            \begin{bmatrix}e^{-1} & 1\\ e^{-9} & e^{-4}\end{bmatrix}
            \begin{bmatrix}1 & -e^{-1}\\ -e^{-1} & 1\end{bmatrix}
            \begin{bmatrix}e^{-1/2} & e^{-9/2}\\ 1 & e^{-2}\end{bmatrix}
            \begin{bmatrix}1 & -e^{-4}\\ -e^{-4} & 1\end{bmatrix}

            = \frac{1}{(1 - e^{-2})(1 - e^{-8})}
            \begin{bmatrix}0 & 1 - e^{-2}\\ e^{-9} - e^{-5} &
            e^{-4} - e^{-10}\end{bmatrix}
            \begin{bmatrix}e^{-1/2} - e^{-17/2} & 0\\ 1 -e^{-6} &
            e^{-2} - e^{-4}\end{bmatrix}

            = \frac{1}{(1 - e^{-2})(1 - e^{-8})}\begin{bmatrix}(1 - e^{-2})(1 - e^{-6})
            & (1 - e^{-2})(e^{-2} - e^{-4})\\  (e^{-9} - e^{-5}) (e^{-1/2} - e^{-17/2})
            +  (e^{-4} - e^{-10}) (1 - e^{-6})
            & (e^{-2} - e^{-4})(e^{-4} - e^{-10})\end{bmatrix}

            \implies \text{Tr}(T_3) = \frac{(1 - e^{-2})(1 - e^{-6})  + (e^{-2} -
            e^{-4})(e^{-4} - e^{-10})}{(1 - e^{-2})(1 - e^{-8})}

        therefore

        .. math::

            \text{CMMD}(\mathcal{D}_1, \mathcal{D}_2) &= \sqrt{\text{Tr}(T_1) +
            \text{Tr}(T_2) - 2\text{Tr}(T_3)}

            =  \sqrt{ \frac{2(1 - e^{-3/2})}{1 - e^{-2}} + \frac{2(1 - e^{-6})}
            {1 - e^{-8}} -2 \frac{(1 - e^{-2})(1 - e^{-6})  + (e^{-2} - e^{-4})
            (e^{-4} - e^{-10})}{(1 - e^{-2})(1 - e^{-8})} }
        """
        # Setup data
        x1 = jnp.array([[0, 0], [1, 1]])
        y1 = jnp.array([[0], [1]])
        dataset_1 = jnp.hstack((x1, y1))

        x2 = jnp.array([[1, 1], [3, 3]])
        y2 = jnp.array([[1], [3]])
        dataset_2 = jnp.hstack((x2, y2))

        # Set expected CMMD
        t1 = (2 * (1 - jnp.exp(-3 / 2))) / (1 - jnp.exp(-2))
        t2 = (2 * (1 - jnp.exp(-6))) / (1 - jnp.exp(-8))
        t3 = (
            ((1 - jnp.exp(-2)) * (1 - jnp.exp(-6)))
            + ((jnp.exp(-2) - jnp.exp(-4)) * (jnp.exp(-4) - jnp.exp(-10)))
        ) / ((1 - jnp.exp(-2)) * (1 - jnp.exp(-8)))
        expected_cmmd = jnp.sqrt(t1 + t2 - 2 * t3)

        # Define a metric object using an RBF kernel
        metric = coreax.metrics.CMMD(
            feature_kernel=coreax.kernel.SquaredExponentialKernel(length_scale=1.0),
            response_kernel=coreax.kernel.SquaredExponentialKernel(length_scale=1.0),
            num_feature_dimensions=2,
            regularisation_parameters=jnp.array([0, 0]),
            precision_threshold=self.precision_threshold,
        )

        # Compute MMD using the metric object
        output = metric.compute(dataset_1=dataset_1, dataset_2=dataset_2)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_cmmd, places=5)

    # We require more variables for the supervised case. Computing the metric outside
    # the class for a random dataset requires a bit of repetition, still useful as a
    # check that the CMMD class implementation hasn't been changed incorrectly.
    # pylint: disable=too-many-locals
    # pylint: disable=duplicate-code
    def test_cmmd_rand(self) -> None:
        r"""
        Test CMMD computed from randomly generated test data agrees with method result.
        """
        # Define kernel objects
        feature_kernel = coreax.kernel.SquaredExponentialKernel(length_scale=1.0)
        response_kernel = coreax.kernel.SquaredExponentialKernel(length_scale=1.0)

        # Extract data
        x1 = jnp.atleast_2d(self.dataset_1[:, : self.feature_dimension])
        y1 = jnp.atleast_2d(self.dataset_1[:, self.feature_dimension :])
        x2 = jnp.atleast_2d(self.dataset_2[:, : self.feature_dimension])
        y2 = jnp.atleast_2d(self.dataset_2[:, self.feature_dimension :])

        # Compute feature kernel gramians
        feature_gramian_1 = feature_kernel.compute(x1, x1)
        feature_gramian_2 = feature_kernel.compute(x2, x2)
        cross_feature_gramian = feature_kernel.compute(x2, x1)

        # Compute response kernel gramians
        response_gramian_1 = response_kernel.compute(y1, y1)
        response_gramian_2 = response_kernel.compute(y2, y2)
        cross_response_gramian = response_kernel.compute(y1, y2)

        # Invert feature kernel gramians
        inverse_feature_gramian_1 = coreax.util.invert_regularised_array(
            array=feature_gramian_1,
            regularisation_parameter=self.regularisation_parameters[0].item(),
            identity=jnp.eye(feature_gramian_1.shape[0]),
        )

        inverse_feature_gramian_2 = coreax.util.invert_regularised_array(
            array=feature_gramian_2,
            regularisation_parameter=self.regularisation_parameters[1].item(),
            identity=jnp.eye(feature_gramian_2.shape[0]),
        )

        # Compute each term in the CMMD
        term_1 = (
            inverse_feature_gramian_1
            @ response_gramian_1
            @ inverse_feature_gramian_1
            @ feature_gramian_1
        )
        term_2 = (
            inverse_feature_gramian_2
            @ response_gramian_2
            @ inverse_feature_gramian_2
            @ feature_gramian_2
        )
        term_3 = (
            inverse_feature_gramian_1
            @ cross_response_gramian
            @ inverse_feature_gramian_2
            @ cross_feature_gramian
        )

        # Compute CMMD
        expected_cmmd = (
            jnp.trace(term_1) + jnp.trace(term_2) - 2 * jnp.trace(term_3)
        ) ** 0.5

        # Define a metric object
        metric = coreax.metrics.CMMD(
            feature_kernel=feature_kernel,
            response_kernel=response_kernel,
            num_feature_dimensions=self.feature_dimension,
            regularisation_parameters=self.regularisation_parameters,
            precision_threshold=self.precision_threshold,
        )

        # Compute CMMD using the metric object
        output = metric.compute(dataset_1=self.dataset_1, dataset_2=self.dataset_2)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_cmmd, places=5)

    # pylint: enable=duplicate-code

    def test_compute_given_block_size_not_none(self) -> None:
        """
        Test compute when given a block size which is not None.
        """
        # Define a metric object
        metric = coreax.metrics.CMMD(
            feature_kernel=MagicMock(),
            response_kernel=MagicMock(),
            num_feature_dimensions=NonCallableMagicMock(),
            regularisation_parameters=NonCallableMagicMock(),
        )

        # Compute CMMD with a block size which is not None - this should raise an error
        # explaining this parameter is not supported.
        with self.assertRaises(AssertionError) as error_raised:
            metric.compute(
                dataset_1=self.dataset_1, dataset_2=self.dataset_2, block_size=0
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "CMMD computation does not support blocking",
        )

    # pylint: enable=too-many-locals

    def test_compute_given_weights_x_not_none(self) -> None:
        """
        Test compute when given weights_x which is not None.
        """
        # Define a metric object
        metric = coreax.metrics.CMMD(
            feature_kernel=MagicMock(),
            response_kernel=MagicMock(),
            num_feature_dimensions=NonCallableMagicMock(),
            regularisation_parameters=NonCallableMagicMock(),
        )

        # Compute CMMD with weights_x vector is not None - this should raise an error
        # explaining this parameter is not supported.
        with self.assertRaises(AssertionError) as error_raised:
            metric.compute(
                dataset_1=self.dataset_1,
                dataset_2=self.dataset_2,
                weights_x=jnp.ones(self.dataset_1.shape[0]),
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "CMMD computation does not support weights",
        )

    def test_compute_given_weights_y_not_none(self) -> None:
        """
        Test compute when given weights_x which is not None.
        """
        # Define a metric object
        metric = coreax.metrics.CMMD(
            feature_kernel=MagicMock(),
            response_kernel=MagicMock(),
            num_feature_dimensions=NonCallableMagicMock(),
            regularisation_parameters=NonCallableMagicMock(),
        )

        # Compute CMMD with weights_y vector is not None - this should raise an error
        # explaining this parameter is not supported.
        with self.assertRaises(AssertionError) as error_raised:
            metric.compute(
                dataset_1=self.dataset_1,
                dataset_2=self.dataset_2,
                weights_y=jnp.ones(self.dataset_1.shape[0]),
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "CMMD computation does not support weights",
        )


# pylint: enable=too-many-public-methods


if __name__ == "__main__":
    unittest.main()
