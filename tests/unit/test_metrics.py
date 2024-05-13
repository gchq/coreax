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

import coreax.data
import coreax.kernel
import coreax.metrics
from coreax.util import pairwise, squared_distance

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

        Generate ``num_reference_points`` random points in ``d`` dimensions from a
        uniform distribution [0, 1).
        Randomly select ``num_comparison_points`` points for second dataset
        ``comparison_data``.
        Generate weights: ``reference_weights`` for reference data,
        ``comparison_weights`` for comparison data.

        :num_reference_points: Number of test data points
        :d: Dimension of data
        :num_comparison_points: Number of points to randomly select for comparison data
        :block_size: Maximum number of points for block calculations
        """
        # Define data parameters
        self.num_reference_points = 30
        self.dimension = 10
        self.num_comparison_points = 5
        self.block_size = 3

        # Define example datasets
        self.reference_points = random.uniform(
            random.key(0), shape=(self.num_reference_points, self.dimension)
        )
        self.reference_weights = (
            random.uniform(random.key(0), shape=(self.num_reference_points,))
            / self.num_reference_points
        )
        self.reference_data = coreax.data.Data(
            data=self.reference_points, weights=self.reference_weights
        )

        self.comparison_points = random.choice(
            random.key(0), self.reference_points, shape=(self.num_comparison_points,)
        )
        self.comparison_weights = (
            random.uniform(random.key(0), shape=(self.num_comparison_points,))
            / self.num_comparison_points
        )
        self.comparison_data = coreax.data.Data(
            data=self.comparison_points, weights=self.comparison_weights
        )

    def test_mmd_compare_same_data(self) -> None:
        r"""
        Test the MMD of a dataset with itself is zero, for several different kernels.
        """
        # Define a metric object using the SquaredExponentialKernel
        metric = coreax.metrics.MMD(
            kernel=coreax.kernel.SquaredExponentialKernel(length_scale=1.0)
        )
        self.assertAlmostEqual(
            float(
                metric.maximum_mean_discrepancy(
                    self.reference_points, self.reference_points
                )
            ),
            0.0,
        )

        # Define a metric object using the LaplacianKernel
        metric = coreax.metrics.MMD(
            kernel=coreax.kernel.LaplacianKernel(length_scale=1.0)
        )
        self.assertAlmostEqual(
            float(
                metric.maximum_mean_discrepancy(
                    self.reference_points, self.reference_points
                )
            ),
            0.0,
        )

        # Define a metric object using the PCIMQKernel
        metric = coreax.metrics.MMD(kernel=coreax.kernel.PCIMQKernel(length_scale=1.0))
        self.assertAlmostEqual(
            float(
                metric.maximum_mean_discrepancy(
                    self.reference_points, self.reference_points
                )
            ),
            0.0,
        )

    def test_mmd_ones(self):
        r"""
        Test MMD computation with a small example dataset of ones and zeros.

        For the dataset of 4 points in 2 dimensions, :math:`\mathcal{D}_1`, and another
        dataset :math:`\mathcal{D}_2`, given by:

        .. math::

            reference_data = [[0,0], [1,1], [0,0], [1,1]]

            comparison_data = [[0,0], [1,1]]

        the Gaussian (aka radial basis function) kernel,
        :math:`k(x,y) = \exp (-||x-y||^2/2\sigma^2)`, gives:

        .. math::

            k(\mathcal{D}_1,x) = \exp(-\begin{bmatrix}0 & 2 & 0 & 2 \\ 2 & 0 & 2 & 0\\
            0 & 2 & 0 & 2 \\ 2 & 0 & 2 & 0\end{bmatrix}/2\sigma^2).

            k(\mathcal{D}_2,\mathcal{D}_2) = \exp(-\begin{bmatrix}0 & 2  \\ 2 & 0
            \end{bmatrix}/2\sigma^2).

            k(\mathcal{D}_1,\mathcal{D}_2) = \exp(-\begin{bmatrix}0 & 2  \\ 2 & 0
            \\0 & 2  \\ 2 & 0
            \end{bmatrix}/2\sigma^2).

        Then

        .. math::

            \text{MMD}^2(\mathcal{D}_1,\mathcal{D}_2) = \mathbb{E}(k(\mathcal{D}_1,
            \mathcal{D}_1))
            + \mathbb{E}(k(\mathcal{D}_2,\mathcal{D}_2)) - 2\mathbb{E}(k(\mathcal{D}_1,
            \mathcal{D}_2))

            = \frac{1}{2} + e^{-1/2} + \frac{1}{2} + e^{-1/2} - 2\left(\frac{1}{2}
            + e^{-1/2}\right)

            = 0.
        """
        # Setup data
        reference_points = jnp.array([[0, 0], [1, 1], [0, 0], [1, 1]])
        reference_data = coreax.data.Data(data=reference_points)
        comparison_points = jnp.array([[0, 0], [1, 1]])
        comparison_data = coreax.data.Data(data=comparison_points)
        length_scale = 1.0

        # Set expected MMD
        expected_output = 0.0

        # Define a metric object using an RBF kernel
        metric = coreax.metrics.MMD(
            kernel=coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)
        )

        # Compute MMD using the metric object
        output = metric.compute(
            reference_data=reference_data, comparison_data=comparison_data
        )

        # Check output matches expected
        self.assertAlmostEqual(float(output), expected_output, places=5)

    def test_mmd_ints(self):
        r"""
        Test MMD computation with a small example dataset of integers.

        For the dataset of 3 points in 2 dimensions, :math:`\mathcal{D}_1`, and second
        dataset :math:`\mathcal{D}_2`, given by:

        .. math::

            reference_data = [[0,0], [1,1], [2,2]]

            comparison_data = [[0,0], [1,1]]

        the RBF kernel, :math:`k(x,y) = \exp (-||x-y||^2/2\sigma^2)`, gives:

        .. math::

            k(\mathcal{D}_1,\mathcal{D}_1) = \exp(-\begin{bmatrix}0 & 2 & 8 \\ 2 & 0 & 2
            \\ 8 & 2 & 0
            \end{bmatrix}/2\sigma^2) = \begin{bmatrix}1 & e^{-1} & e^{-4} \\ e^{-1} &
            1 & e^{-1} \\ e^{-4} & e^{-1} & 1\end{bmatrix}.

            k(\mathcal{D}_2,\mathcal{D}_2) = \exp(-\begin{bmatrix}0 & 2 \\ 2 & 0
            \end{bmatrix}/2\sigma^2) = \begin{bmatrix}1 & e^{-1}\\ e^{-1} & 1
            \end{bmatrix}.

            k(\mathcal{D}_1,\mathcal{D}_2) =  \exp(-\begin{bmatrix}0 & 2 & 8
            \\ 2 & 0 & 2 \end{bmatrix}
            /2\sigma^2) = \begin{bmatrix}1 & e^{-1} \\  e^{-1} & 1 \\ e^{-4} & e^{-1}
            \end{bmatrix}.

        Then

        .. math::

            \text{MMD}^2(\mathcal{D}_1,\mathcal{D}_2) =
            \mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_1)) +
            \mathbb{E}(k(\mathcal{D}_2,\mathcal{D}_2)) -
            2\mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_2))

            = \frac{3+4e^{-1}+2e^{-4}}{9} + \frac{2 + 2e^{-1}}{2} -2 \times
            \frac{2 + 3e^{-1}+e^{-4}}{6}

            = \frac{3 - e^{-1} -2e^{-4}}{18}.
        """
        # Setup data
        reference_points = jnp.array([[0, 0], [1, 1], [2, 2]])
        reference_data = coreax.data.Data(data=reference_points)
        comparison_points = jnp.array([[0, 0], [1, 1]])
        comparison_data = coreax.data.Data(data=comparison_points)
        length_scale = 1.0

        # Set expected MMD
        expected_output = jnp.sqrt((3 - jnp.exp(-1) - 2 * jnp.exp(-4)) / 18)

        # Define a metric object using an RBF kernel
        metric = coreax.metrics.MMD(
            kernel=coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)
        )

        # Compute MMD using the metric object
        output = metric.compute(
            reference_data=reference_data, comparison_data=comparison_data
        )

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
        kernel_nn = kernel.compute(self.reference_points, self.reference_points)
        kernel_mm = kernel.compute(self.comparison_points, self.comparison_points)
        kernel_nm = kernel.compute(self.reference_points, self.comparison_points)

        # Compute overall MMD by
        expected_mmd = (
            kernel_nn.mean() + kernel_mm.mean() - 2 * kernel_nm.mean()
        ) ** 0.5

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute MMD using the metric object
        output = metric.maximum_mean_discrepancy(
            reference_points=self.reference_points,
            comparison_points=self.comparison_points,
        )

        # Check output matches expected
        self.assertAlmostEqual(output, expected_mmd, places=5)

    def test_weighted_mmd_ints(self) -> None:
        r"""
        Test weighted MMD computation with a small example dataset of integers.

        Weighted mmd is calculated if and only if comparison_weights are provided. When
        `comparison_weights` = :data:`None`, the MMD class computes the standard,
        non-weighted MMD.

        For the dataset of 3 points in 2 dimensions :math:`\mathcal{D}_1`, second
        dataset :math:`\mathcal{D}_2`, and weights for this second dataset :math:`w_2`,
        given by:

        .. math::

            reference_data = [[0,0], [1,1], [2,2]]

            w_1 = [1,1,1]

            comparison_data = [[0,0], [1,1]]

            w_2 = [1,0]

        the weighted maximum mean discrepancy is calculated via:

        .. math::

            \text{WMMD}^2(\mathcal{D}_1,\mathcal{D}_2) =
            \frac{1}{||w_1||**2}w_1^T \mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_1)) w_1
             + \frac{1}{||w_2||**2}w_2^T k(\mathcal{D}_2,\mathcal{D}_2) w_2
             - \frac{2}{||w_1||||w_2||} w_1
             \mathbb{E}_x(k(\mathcal{D}_1,\mathcal{D}_2)) w_2

            = \frac{3+4e^{-1}+2e^{-4}}{9} + 1 - 2 \times \frac{1 + e^{-1} + e^{-4}}{3}

            = \frac{2}{3} - \frac{2}{9}e^{-1} - \frac{4}{9}e^{-4}.
        """
        # Setup data
        reference_points = jnp.array([[0, 0], [1, 1], [2, 2]])
        reference_weights = jnp.array([1, 1, 1])
        reference_data = coreax.data.Data(
            data=reference_points, weights=reference_weights
        )
        comparison_points = jnp.array([[0, 0], [1, 1]])
        comparison_weights = jnp.array([1, 0])
        comparison_data = coreax.data.Data(
            data=comparison_points, weights=comparison_weights
        )
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
        output = metric.compute(
            reference_data=reference_data,
            comparison_data=comparison_data,
        )

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
        kernel_nn = kernel.compute(self.reference_points, self.reference_points)
        kernel_mm = kernel.compute(self.comparison_points, self.comparison_points)
        kernel_nm = kernel.compute(self.reference_points, self.comparison_points)

        # Define expected output
        comparison_weights = self.comparison_data.weights
        reference_weights = self.reference_data.weights
        expected_output = jnp.sqrt(
            jnp.dot(reference_weights.T, jnp.dot(kernel_nn, reference_weights))
            / reference_weights.sum() ** 2
            + jnp.dot(comparison_weights.T, jnp.dot(kernel_mm, comparison_weights))
            / comparison_weights.sum() ** 2
            - 2
            * jnp.dot(jnp.dot(reference_weights.T, kernel_nm), comparison_weights)
            / (reference_weights.sum() * comparison_weights.sum())
        )

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute weighted MMD using the metric object
        output = metric.weighted_maximum_mean_discrepancy(
            reference_points=self.reference_points,
            comparison_points=self.comparison_points,
            reference_weights=self.reference_weights,
            comparison_weights=self.comparison_weights,
        )

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=5)

    def test_weighted_mmd_uniform_weights(self) -> None:
        r"""
        Test that weighted MMD equals MMD if weights are uniform.

        Uniform weights are such that :math:`w_1 = 1/n` and :math:`w_2 = 1/m`.
        """
        # Define a kernel object
        length_scale = 1.0
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute weighted MMD with all weights being uniform
        uniform_wmmd = metric.weighted_maximum_mean_discrepancy(
            reference_points=self.reference_points,
            comparison_points=self.comparison_points,
            reference_weights=jnp.ones(self.num_reference_points)
            / self.num_reference_points,
            comparison_weights=jnp.ones(self.num_comparison_points)
            / self.num_comparison_points,
        )

        # Compute MMD without the weights
        mmd = metric.maximum_mean_discrepancy(
            self.reference_points, self.comparison_points
        )

        # Check uniform weighted MMD and MMD without weights give the same result
        self.assertAlmostEqual(float(uniform_wmmd), float(mmd), places=5)

    def test_sum_pairwise_distances(self) -> None:
        r"""
        Test sum_pairwise_distances() with a small integer example.

        For the dataset of 3 points in 2 dimensions :math:`\mathcal{D}_1`, and second
        dataset :math:`\mathcal{D}_2`:

        .. math::

            reference_data = [[0,0], [1,1], [2,2]]

            comparison_data = [[0,0], [1,1]]

        the pairwise square distances are given by the matrix:

        .. math::

            \begin{bmatrix}0 & 2 \\ 2 & 0 \\ 8 & 2 \end{bmatrix}

        which, summing across both axes, gives the result :math:`14`.
        """
        # Setup data
        reference_points = jnp.array([[0, 0], [1, 1], [2, 2]])
        comparison_points = jnp.array([[0, 0], [1, 1]])

        # Set expected output
        expected_output = 14

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = pairwise(squared_distance)

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute the sum of pairwise distances using the metric object
        output = metric.sum_pairwise_distances(
            reference_points=reference_points,
            comparison_points=comparison_points,
            block_size=2,
        )

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
        reference_points = jnp.array([[0, 0], [1, 1], [2, 2]])
        comparison_points = jnp.array([[0, 0], [1, 1]])

        # Set expected output
        expected_output = 14

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = pairwise(squared_distance)

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute the sum of pairwise distances using the metric object
        output = metric.sum_pairwise_distances(
            reference_points=reference_points,
            comparison_points=comparison_points,
            block_size=2000,
        )

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
            metric.sum_pairwise_distances(
                reference_points=self.reference_points,
                comparison_points=self.comparison_points,
                block_size=0,
            )
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
            metric.sum_pairwise_distances(
                reference_points=self.reference_points,
                comparison_points=self.comparison_points,
                block_size=-5,
            )
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
            metric.sum_pairwise_distances(
                reference_points=self.reference_points,
                comparison_points=self.comparison_points,
                block_size=2.0,
            )
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
        reference_points = jnp.array([[0, 0], [1, 1], [2, 2]])
        reference_data = coreax.data.Data(data=reference_points)
        comparison_points = jnp.array([[0, 0], [1, 1]])
        comparison_data = coreax.data.Data(data=comparison_points)

        # Define expected output
        expected_output = jnp.sqrt((3 - jnp.exp(-1) - 2 * jnp.exp(-4)) / 18)

        # Define a kernel object
        length_scale = 1.0
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute MMD block-wise
        mmd_block_test = metric.compute(
            reference_data=reference_data, comparison_data=comparison_data, block_size=2
        )

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
        for i1 in range(0, self.num_reference_points, self.block_size):
            for i2 in range(0, self.num_reference_points, self.block_size):
                kernel_nn += kernel.compute(
                    self.reference_points[i1 : i1 + self.block_size, :],
                    self.reference_points[i2 : i2 + self.block_size, :],
                ).sum()

        # Compute MMD term with y and itself
        kernel_mm = 0.0
        for j1 in range(0, self.num_comparison_points, self.block_size):
            for j2 in range(0, self.num_comparison_points, self.block_size):
                kernel_mm += kernel.compute(
                    self.comparison_points[j1 : j1 + self.block_size, :],
                    self.comparison_points[j2 : j2 + self.block_size, :],
                ).sum()

        # Compute MMD term with x and y
        kernel_nm = 0.0
        for i in range(0, self.num_reference_points, self.block_size):
            for j in range(0, self.num_comparison_points, self.block_size):
                kernel_nm += kernel.compute(
                    self.reference_points[i : i + self.block_size, :],
                    self.comparison_points[j : j + self.block_size, :],
                ).sum()

        # Compute expected output from MMD
        expected_output = (
            kernel_nn / self.num_reference_points**2
            + kernel_mm / self.num_comparison_points**2
            - 2 * kernel_nm / (self.num_reference_points * self.num_comparison_points)
        ) ** 0.5

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute MMD
        output = metric.maximum_mean_discrepancy_block(
            self.reference_points, self.comparison_points, block_size=self.block_size
        )

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
            float(
                metric.maximum_mean_discrepancy(
                    self.reference_points, self.comparison_points
                )
            ),
            float(
                metric.maximum_mean_discrepancy_block(
                    self.reference_points,
                    self.comparison_points,
                    block_size=self.block_size,
                )
            ),
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
            metric.maximum_mean_discrepancy_block(
                reference_points=self.reference_points,
                comparison_points=self.comparison_points,
                block_size=0,
            )
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
            metric.maximum_mean_discrepancy_block(
                reference_points=self.reference_points,
                comparison_points=self.comparison_points,
                block_size=-2,
            )
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
            metric.maximum_mean_discrepancy_block(
                reference_points=self.reference_points,
                comparison_points=self.comparison_points,
                block_size=1.0,
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "block_size must be a positive integer",
        )

    def test_sum_weighted_pairwise_distances(self) -> None:
        r"""
        Test sum_weighted_pairwise_distances, which calculates w^T*K*w matrices.

        Computations are done in blocks of size block_size.

        For the dataset of 3 points in 2 dimensions :math:`\mathcal{D}_1`, and second
        dataset :math:`\mathcal{D}_2`:

        .. math::

            reference_data = [[0,0], [1,1], [2,2]]

            comparison_data = [[0,0], [1,1]]

        the pairwise square distances are given by the matrix:

        .. math::

            k(\mathcal{D}_1, \mathcal{D}_2) = \begin{bmatrix}0 & 2 \\ 2 & 0 \\ 8 & 2
            \end{bmatrix}.

        Then, for weights vectors:

        .. math::

            w = [0.5, 0.5, 0],

            w_comparison_data = [1, 0]

        the product :math:`w^T*k(\mathcal{D}_1, \mathcal{D}_2)*w_comparison_data = 1`.
        """
        # Setup some data
        reference_points = jnp.array([[0, 0], [1, 1], [2, 2]])
        comparison_points = jnp.array([[0, 0], [1, 1]])
        reference_weights = jnp.array([0.5, 0.5, 0])
        comparison_weights = jnp.array([1, 0])

        # Define expected output
        expected_output = 1.0

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = pairwise(squared_distance)

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute sum of weighted pairwise distances
        output = metric.sum_weighted_pairwise_distances(
            reference_points=reference_points,
            comparison_points=comparison_points,
            reference_weights=reference_weights,
            comparison_weights=comparison_weights,
            block_size=2,
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
        reference_points = jnp.array([[0, 0], [1, 1], [2, 2]])
        comparison_points = jnp.array([[0, 0], [1, 1]])
        reference_weights = jnp.array([0.5, 0.5, 0])
        comparison_weights = jnp.array([1, 0])

        # Define expected output
        expected_output = 1.0

        # Define a kernel object and set pairwise computations to be the square distance
        kernel = MagicMock()
        kernel.compute = pairwise(squared_distance)

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute sum of weighted pairwise distances
        output = metric.sum_weighted_pairwise_distances(
            reference_points=reference_points,
            comparison_points=comparison_points,
            reference_weights=reference_weights,
            comparison_weights=comparison_weights,
            block_size=2000,
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
                reference_points=self.reference_points,
                comparison_points=self.comparison_points,
                reference_weights=self.reference_weights,
                comparison_weights=self.comparison_weights,
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
                reference_points=self.reference_points,
                comparison_points=self.comparison_points,
                reference_weights=self.reference_weights,
                comparison_weights=self.comparison_weights,
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
                reference_points=self.reference_points,
                comparison_points=self.comparison_points,
                reference_weights=self.reference_weights,
                comparison_weights=self.comparison_weights,
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

            reference_data = [[0,0], [1,1], [2,2]],

            comparison_data = [[0,0], [1,1]],

            w_1^T = [0.5, 0.5, 0],

            w_2^T = [1, 0],

        the weighted maximum mean discrepancy is given by:

        .. math::

            \text{WMMD}^2(\mathcal{D}_1, \mathcal{D}_2) =
            w_1^T k(\mathcal{D}_1, \mathcal{D}_1) w_1
            + w_2^T k(\mathcal{D}_2, \mathcal{D}_2) w_2
            - 2 w_1^T k(\mathcal{D}_1, \mathcal{D}_2) w_2

        which, when :math:`k(\mathcal{D}_1,\mathcal{D}_2)` is the RBF kernel,
        reduces to:

        .. math::

            \frac{1}{2} + \frac{e^{-1}}{2} + 1 - 2\times(\frac{1}{2} + \frac{e^{-1}}{2})

            = \frac{1}{2} - \frac{e^{-1}}{2}.
        """
        # Define some data
        reference_points = jnp.array([[0, 0], [1, 1], [2, 2]])
        comparison_points = jnp.array([[0, 0], [1, 1]])
        reference_weights = jnp.array([0.5, 0.5, 0])
        comparison_weights = jnp.array([1, 0])

        # Define expected output
        expected_output = jnp.sqrt(1 / 2 - jnp.exp(-1) / 2)

        # Define a kernel object
        length_scale = 1.0
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute weighted MMD block-wise
        output = metric.weighted_maximum_mean_discrepancy_block(
            reference_points=reference_points,
            comparison_points=comparison_points,
            reference_weights=reference_weights,
            comparison_weights=comparison_weights,
            block_size=2,
        )

        # Check output matches expected
        self.assertAlmostEqual(float(output), float(expected_output), places=5)

    def test_weighted_maximum_mean_discrepancy_block_equals_mmd(self) -> None:
        r"""
        Test weighted MMD computations with uniform weights.

        One expects that, when weights are uniform the output of
        weighted_maximum_mean_discrepancy_block should be the same as an unweighted
        computation.
        """
        # Define a kernel object
        length_scale = 1.0
        kernel = coreax.kernel.SquaredExponentialKernel(length_scale=length_scale)

        # Define a metric object
        metric = coreax.metrics.MMD(kernel=kernel)

        # Compute weighted MMD with uniform weights
        output = metric.weighted_maximum_mean_discrepancy_block(
            self.reference_points,
            self.comparison_points,
            reference_weights=jnp.ones(self.num_reference_points)
            / self.num_reference_points,
            comparison_weights=jnp.ones(self.num_comparison_points)
            / self.num_comparison_points,
            block_size=self.block_size,
        )

        # Check output matches expected
        self.assertAlmostEqual(
            float(output),
            float(
                metric.maximum_mean_discrepancy(
                    self.reference_points, self.comparison_points
                )
            ),
            places=5,
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
                reference_points=self.reference_points,
                comparison_points=self.comparison_points,
                reference_weights=self.reference_weights,
                comparison_weights=self.comparison_weights,
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
                reference_points=self.reference_points,
                comparison_points=self.comparison_points,
                reference_weights=self.reference_weights,
                comparison_weights=self.comparison_weights,
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
                reference_points=self.reference_points,
                comparison_points=self.comparison_points,
                reference_weights=self.reference_weights,
                comparison_weights=self.comparison_weights,
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
            metric.compute(
                reference_data=self.reference_data,
                comparison_data=self.comparison_data,
                block_size=0,
            )
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
            metric.compute(
                reference_data=self.reference_data,
                comparison_data=self.comparison_data,
                block_size=-2,
            )
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
            metric.compute(
                reference_data=self.reference_data,
                comparison_data=self.comparison_data,
                block_size=2.0,
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "block_size must be a positive integer",
        )


# pylint: enable=too-many-public-methods


if __name__ == "__main__":
    unittest.main()
