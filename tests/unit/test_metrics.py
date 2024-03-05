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
from unittest.mock import MagicMock, patch

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
        :m: Number of points to randomly select for second dataset ``y``
        :max_size: Maximum number of points for block calculations
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
        # Setup some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute sum of pairwise distances with a zero block_size - the full text of
        # the inbuilt error raised from the range function should provide all
        # information needed for users to identify the issue
        with self.assertRaises(ValueError) as error_raised:
            metric.sum_pairwise_distances(x=x, y=y, block_size=0)
        self.assertEqual(
            error_raised.exception.args[0],
            "range() arg 3 must not be zero",
        )

    def test_sum_pairwise_distances_negative_block_size(self):
        """
        Test sum_pairwise_distances when a negative block size is given.
        """
        # Setup some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute sum of pairwise distances with a negative block_size - this should be
        # capped at 0, and then the full text of the inbuilt error raised from the range
        # function should provide all information needed for users to identify the issue
        with self.assertRaises(ValueError) as error_raised:
            metric.sum_pairwise_distances(x=x, y=y, block_size=-5)
        self.assertEqual(
            error_raised.exception.args[0],
            "range() arg 3 must not be zero",
        )

    def test_sum_pairwise_distances_float_block_size(self):
        """
        Test sum_pairwise_distances when a float block size is given.
        """
        # Setup some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute sum of pairwise distances with a float block_size - the full text of
        # the inbuilt error raised from the range function should provide all
        # information needed for users to identify the issue
        with self.assertRaises(TypeError) as error_raised:
            metric.sum_pairwise_distances(x=x, y=y, block_size=2.0)
        self.assertEqual(
            error_raised.exception.args[0],
            "'float' object cannot be interpreted as an integer",
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
        # Setup some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute MMD with a zero block_size - the full text of the inbuilt error raised
        # from the range function should provide all information needed for users to
        # identify the issue
        with self.assertRaises(ValueError) as error_raised:
            metric.maximum_mean_discrepancy_block(x=x, y=y, block_size=0)
        self.assertEqual(
            error_raised.exception.args[0],
            "range() arg 3 must not be zero",
        )

    def test_maximum_mean_discrepancy_block_negative_block_size(self) -> None:
        """
        Test maximum_mean_discrepancy_block when given a negative block size.
        """
        # Setup some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute MMD with a negative block_size - this should be capped at 0,
        # and then the full text of the inbuilt error raised from the range function
        # should provide all information needed for users to identify the issue
        with self.assertRaises(ValueError) as error_raised:
            metric.maximum_mean_discrepancy_block(x=x, y=y, block_size=-2)
        self.assertEqual(
            error_raised.exception.args[0],
            "range() arg 3 must not be zero",
        )

    def test_maximum_mean_discrepancy_block_float_block_size(self) -> None:
        """
        Test maximum_mean_discrepancy_block when given a float block size.
        """
        # Setup some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute MMD with a float block_size - the full text of the inbuilt error
        # raised from the range function should provide all information needed for
        # users to identify the issue
        with self.assertRaises(TypeError) as error_raised:
            metric.maximum_mean_discrepancy_block(x=x, y=y, block_size=2.0)
        self.assertEqual(
            error_raised.exception.args[0],
            "'float' object cannot be interpreted as an integer",
        )

    def test_sum_weighted_pairwise_distances(self) -> None:
        r"""
        Test sum_weighted_pairwise_distances, which calculates w^T*K*w matrices.

        Computations are done in blocks of size max_size.

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
        # Setup some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])
        weights_x = jnp.array([0.5, 0.5, 0])
        weights_y = jnp.array([1, 0])

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute sum of weighted pairwise distances with a zero block_size - the
        # full text of the inbuilt error raised from the range function should provide
        # all information needed for users to identify the issue
        with self.assertRaises(ValueError) as error_raised:
            metric.sum_weighted_pairwise_distances(
                x=x, y=y, weights_x=weights_x, weights_y=weights_y, block_size=0
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "range() arg 3 must not be zero",
        )

    def test_sum_weighted_pairwise_distances_negative_block_size(self):
        """
        Test sum_weighted_pairwise_distances when a negative block size is given.
        """
        # Setup some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])
        weights_x = jnp.array([0.5, 0.5, 0])
        weights_y = jnp.array([1, 0])

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute sum of weighted pairwise distances with a negative block_size - this
        # should be capped at 0, and then the full text of the inbuilt error raised from
        # the range function should provide all information needed for users to identify
        # the issue
        with self.assertRaises(ValueError) as error_raised:
            metric.sum_weighted_pairwise_distances(
                x=x, y=y, weights_x=weights_x, weights_y=weights_y, block_size=-5
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "range() arg 3 must not be zero",
        )

    def test_sum_weighted_pairwise_distances_float_block_size(self):
        """
        Test sum_weighted_pairwise_distances when a float block size is given.
        """
        # Setup some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])
        weights_x = jnp.array([0.5, 0.5, 0])
        weights_y = jnp.array([1, 0])

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute sum of weighted pairwise distances with a float block_size - the
        # full text of the inbuilt error raised from the range function should provide
        # all information needed for users to identify the issue
        with self.assertRaises(TypeError) as error_raised:
            metric.sum_weighted_pairwise_distances(
                x=x, y=y, weights_x=weights_x, weights_y=weights_y, block_size=2.0
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "'float' object cannot be interpreted as an integer",
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
        # Setup some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])
        weights_x = jnp.array([0.5, 0.5, 0])
        weights_y = jnp.array([1, 0])

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute weighted MMD with a zero block_size - the full text of the inbuilt
        # error raised from the range function should provide all information needed for
        # users to identify the issue
        with self.assertRaises(ValueError) as error_raised:
            metric.weighted_maximum_mean_discrepancy_block(
                x=x, y=y, weights_x=weights_x, weights_y=weights_y, block_size=0
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "range() arg 3 must not be zero",
        )

    def test_weighted_maximum_mean_discrepancy_block_negative_block_size(self) -> None:
        """
        Test weighted_maximum_mean_discrepancy_block when given a negative block size.
        """
        # Setup some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])
        weights_x = jnp.array([0.5, 0.5, 0])
        weights_y = jnp.array([1, 0])

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute weighted MMD with a negative block_size - this should be capped at 0,
        # and then the full text of the inbuilt error raised from the range function
        # should provide all information needed for users to identify the issue
        with self.assertRaises(ValueError) as error_raised:
            metric.weighted_maximum_mean_discrepancy_block(
                x=x, y=y, weights_x=weights_x, weights_y=weights_y, block_size=-2
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "range() arg 3 must not be zero",
        )

    def test_weighted_maximum_mean_discrepancy_block_float_block_size(self) -> None:
        """
        Test weighted_maximum_mean_discrepancy_block when given a float block size.
        """
        # Setup some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])
        weights_x = jnp.array([0.5, 0.5, 0])
        weights_y = jnp.array([1, 0])

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute weighted MMD with a float block_size - the full text of the inbuilt
        # error raised from the range function should provide all information needed
        # for users to identify the issue
        with self.assertRaises(TypeError) as error_raised:
            metric.weighted_maximum_mean_discrepancy_block(
                x=x, y=y, weights_x=weights_x, weights_y=weights_y, block_size=2.0
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "'float' object cannot be interpreted as an integer",
        )

    def test_compute_zero_block_size(self) -> None:
        """
        Test compute when given a zero block size.
        """
        # Setup some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute MMD with a zero block_size - the full text of the inbuilt error raised
        # from the range function should provide all information needed for users to
        # identify the issue
        with self.assertRaises(ValueError) as error_raised:
            metric.compute(x=x, y=y, block_size=0)
        self.assertEqual(
            error_raised.exception.args[0],
            "range() arg 3 must not be zero",
        )

    def test_compute_negative_block_size(self) -> None:
        """
        Test compute when given a negative block size.
        """
        # Setup some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute MMD with a negative block_size - this should be capped at 0, and then
        # the full text of the inbuilt error raised from the range function should
        # provide all information needed for users to identify the issue
        with self.assertRaises(ValueError) as error_raised:
            metric.compute(x=x, y=y, block_size=-2)
        self.assertEqual(
            error_raised.exception.args[0],
            "range() arg 3 must not be zero",
        )

    def test_compute_float_block_size(self) -> None:
        """
        Test compute when given a float block size.
        """
        # Setup some data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = coreax.util.squared_distance_pairwise

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute MMD with a float block_size - the full text of the inbuilt error
        # raised from the range function should provide all information needed for users
        # to identify the issue
        with self.assertRaises(TypeError) as error_raised:
            metric.compute(x=x, y=y, block_size=2.0)
        self.assertEqual(
            error_raised.exception.args[0],
            "'float' object cannot be interpreted as an integer",
        )


# pylint: enable=too-many-public-methods


if __name__ == "__main__":
    unittest.main()
