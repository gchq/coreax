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
from jax import random

import coreax.kernel as ck
import coreax.metrics as cm
import coreax.util as cu


class TestMetrics(unittest.TestCase):
    r"""
    Tests related to Metric abstract base class in metrics.py.
    """

    def test_metric_creation(self) -> None:
        r"""
        Test the class Metric initialises correctly.
        """
        # Patch the abstract methods of the Metric ABC, so it can be created
        p = patch.multiple(cm.Metric, __abstractmethods__=set())
        p.start()

        # Create a metric object
        metric = cm.Metric()

        # Check the compute method exists
        self.assertTrue(hasattr(metric, "compute"))


class TestMMD(unittest.TestCase):
    r"""
    Tests related to the maximum mean discrepancy (MMD) class in metrics.py.
    """

    def setUp(self):
        r"""
        Generate data for shared use across unit tests.

        Generate ``num_points_x`` random points in ``d`` dimensions from a uniform
        distribution [0, 1).
        Randomly select ``num_points_y`` points for second dataset ``Y``.
        Generate weights: ``weights_x`` for original data, ``weights_y`` for second
        dataset ``Y``.

        :num_points_x: Number of test data points
        :d: Dimension of data
        :m: Number of points to randomly select for second dataset ``Y``
        :max_size: Maximum number of points for block calculations
        """
        # Define data parameters
        self.num_points_x = 30
        self.dimension = 10
        self.num_points_y = 5
        self.max_size = 3

        # Define example datasets
        self.x = random.uniform(
            random.PRNGKey(0), shape=(self.num_points_x, self.dimension)
        )
        self.y = random.choice(random.PRNGKey(0), self.x, shape=(self.num_points_y,))
        self.weights_x = (
            random.uniform(random.PRNGKey(0), shape=(self.num_points_x,))
            / self.num_points_x
        )
        self.weights_y = (
            random.uniform(random.PRNGKey(0), shape=(self.num_points_y,))
            / self.num_points_y
        )

    def test_mmd_compare_same_data(self) -> None:
        r"""
        Test the MMD of a dataset with itself is zero, for several different kernels.
        """
        # Define a metric object using the SquaredExponentialKernel
        metric = cm.MMD(kernel=ck.SquaredExponentialKernel(length_scale=1.0))
        self.assertAlmostEqual(float(metric.compute(self.x, self.x)), 0.0)

        # Define a metric object using the LaplacianKernel
        metric = cm.MMD(kernel=ck.LaplacianKernel(length_scale=1.0))
        self.assertAlmostEqual(float(metric.compute(self.x, self.x)), 0.0)

        # Define a metric object using the PCIMQKernel
        metric = cm.MMD(kernel=ck.PCIMQKernel(length_scale=1.0))
        self.assertAlmostEqual(float(metric.compute(self.x, self.x)), 0.0)

    def test_mmd_ones(self):
        r"""
        Test mmd function with a small example dataset of ones and zeros.

        For the dataset of 4 points in 2 dimensions, :math:`X`, and another dataset
        :math:`Y`, given by:

        .. math::

            X = [[0,0], [1,1], [0,0], [1,1]]

            Y = [[0,0], [1,1]]

        the Gaussian (aka radial basis function) kernel,
        :math:`k(x,y) = \exp (-||x-y||^2/2\sigma^2)`, gives:

        .. math::

            k(X,X) = \exp(-\begin{bmatrix}0 & 2 & 0 & 2 \\ 2 & 0 & 2 & 0\\ 0 & 2 & 0 &
            2 \\ 2 & 0 & 2 & 0\end{bmatrix}/2\sigma^2).

            k(Y,Y) = \exp(-\begin{bmatrix}0 & 2  \\ 2 & 0 \end{bmatrix}/2\sigma^2).

            k(X,Y) = \exp(-\begin{bmatrix}0 & 2  \\ 2 & 0 \\0 & 2  \\ 2 & 0
            \end{bmatrix}/2\sigma^2).

        Then

        .. math::

            \text{MMD}^2(X,Y) = \mathbb{E}(k(X,X)) + \mathbb{E}(k(Y,Y)) -
            2\mathbb{E}(k(X,Y))

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
        metric = cm.MMD(kernel=ck.SquaredExponentialKernel(length_scale=length_scale))

        # Compute MMD using the metric object
        output = metric.compute(x=x, y=y)

        # Check output matches expected
        self.assertAlmostEqual(float(output), expected_output, places=5)

    def test_mmd_ints(self):
        r"""
        Test MMD function with a small example dataset of integers.

        For the dataset of 3 points in 2 dimensions, :math:`X`, and second dataset
        :math:`Y`, given by:

        .. math::

            X = [[0,0], [1,1], [2,2]]

            Y = [[0,0], [1,1]]

        the RBF kernel, :math:`k(x,y) = \exp (-||x-y||^2/2\sigma^2)`, gives:

        .. math::

            k(X,X) = \exp(-\begin{bmatrix}0 & 2 & 8 \\ 2 & 0 & 2 \\ 8 & 2 & 0
            \end{bmatrix}/2\sigma^2) = \begin{bmatrix}1 & e^{-1} & e^{-4} \\ e^{-1} &
            1 & e^{-1} \\ e^{-4} & e^{-1} & 1\end{bmatrix}.

            k(Y,Y) = \exp(-\begin{bmatrix}0 & 2 \\ 2 & 0 \end{bmatrix}/2\sigma^2) =
             \begin{bmatrix}1 & e^{-1}\\ e^{-1} & 1\end{bmatrix}.

            k(X,Y) =  \exp(-\begin{bmatrix}0 & 2 & 8 \\ 2 & 0 & 2 \end{bmatrix}
            /2\sigma^2) = \begin{bmatrix}1 & e^{-1} \\  e^{-1} & 1 \\ e^{-4} & e^{-1}
            \end{bmatrix}.

        Then

        .. math::

            \text{MMD}^2(X,Y) = \mathbb{E}(k(X,X)) + \mathbb{E}(k(Y,Y)) -
            2\mathbb{E}(k(X,Y))

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
        metric = cm.MMD(kernel=ck.SquaredExponentialKernel(length_scale=length_scale))

        # Compute MMD using the metric object
        output = metric.compute(x=x, y=y)

        # Check output matches expected
        self.assertAlmostEqual(float(output), float(expected_output), places=5)

    def test_mmd_rand(self):
        r"""
        Test that MMD computed from randomly generated test data agrees with mmd().
        """
        # Define a kernel object
        length_scale = 1.0
        kernel = ck.SquaredExponentialKernel(length_scale=length_scale)

        # Compute each term in the MMD formula
        kernel_nn = kernel.compute(self.x, self.x)
        kernel_mm = kernel.compute(self.y, self.y)
        kernel_nm = kernel.compute(self.x, self.y)

        # Compute overall MMD by
        expected_mmd = (
            kernel_nn.mean() + kernel_mm.mean() - 2 * kernel_nm.mean()
        ) ** 0.5

        # Define a metric object
        metric = cm.MMD(kernel=kernel)

        # Compute MMD using the metric object
        output = metric.compute(x=self.x, y=self.y)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_mmd, places=5)

    def test_wmmd_ints(self) -> None:
        r"""
        Test weighted MMD function wmmd() with a small example dataset of integers.

        wmmd is calculated if and only if weights_y are provided. When weights_y =
        None, the MMD class computes the standard, non-weighted mmd.

        For the dataset of 3 points in 2 dimensions :math:`X`, second dataset :math:`Y`,
        and weights for this second dataset :math:`w_y`, given by:

        .. math::

            X = [[0,0], [1,1], [2,2]]

            Y = [[0,0], [1,1]]

            w_y = [1,0]

        the weighted maximum mean discrepancy is calculated via:

        .. math::

            \text{WMMD}^2(X,Y) = \mathbb{E}(k(X,X)) + w_y^T k(Y,Y) w_y
             - 2\mathbb{E}_X(k(X,Y)) w_y

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
        metric = cm.MMD(kernel=ck.SquaredExponentialKernel(length_scale=length_scale))

        # Compute weighted mmd using the metric object
        output = metric.compute(x=x, y=y, weights_y=weights_y)

        # Check output matches expected
        self.assertAlmostEqual(float(output), float(expected_output), places=5)

    def test_wmmd_rand(self) -> None:
        r"""
        Test that WMMD computed from randomly generated test data agrees with wmmd().
        """
        # Define a kernel object
        length_scale = 1.0
        kernel = ck.SquaredExponentialKernel(length_scale=length_scale)

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
        metric = cm.MMD(kernel=kernel)

        # Compute weighted MMD using the metric object
        output = metric.compute(x=self.x, y=self.y, weights_y=self.weights_y)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=5)

    def test_wmmd_uniform_weights(self) -> None:
        r"""
        Test that wmmd = mmd if weights are uniform, :math:`w_y = 1/m`.
        """
        # Define a kernel object
        length_scale = 1.0
        kernel = ck.SquaredExponentialKernel(length_scale=length_scale)

        # Define a metric object
        metric = cm.MMD(kernel=kernel)

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

        For the dataset of 3 points in 2 dimensions :math:`X`, and second dataset
        :math:`Y`:

        .. math::

            X = [[0,0], [1,1], [2,2]]

            Y = [[0,0], [1,1]]

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
        kernel.compute = cu.sq_dist_pairwise

        # Define a metric object
        metric = cm.MMD(kernel=kernel)

        # Compute the sum of pairwise distances using the metric object
        output = metric.sum_pairwise_distances(x=x, y=y, max_size=2)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=5)

    def test_mmd_block_ints(self) -> None:
        r"""
        Test mmd_block calculation of MMD while limiting memory requirements.

        This test uses the same 2D, three-point dataset and second datset as
        test_mmd_ints().
        """
        # Setup data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])

        # Define expected output
        expected_output = jnp.sqrt((3 - jnp.exp(-1) - 2 * jnp.exp(-4)) / 18)

        # Define a kernel object
        length_scale = 1.0
        kernel = ck.SquaredExponentialKernel(length_scale=length_scale)

        # Define a metric object
        metric = cm.MMD(kernel=kernel)

        # Compute MMD block-wise
        mmd_block_test = metric.compute(x=x, y=y, max_size=2)

        # Check output matches expected
        self.assertAlmostEqual(float(mmd_block_test), float(expected_output), places=5)

    def test_mmd_block_rand(self) -> None:
        r"""
        Test that mmd block-computed for random test data equals mmd_block().
        """
        # Define a kernel object
        length_scale = 1.0
        kernel = ck.SquaredExponentialKernel(length_scale=length_scale)

        # Compute MMD term with x and itself
        kernel_nn = 0.0
        for i1 in range(0, self.num_points_x, self.max_size):
            for i2 in range(0, self.num_points_x, self.max_size):
                kernel_nn += kernel.compute(
                    self.x[i1 : i1 + self.max_size, :],
                    self.x[i2 : i2 + self.max_size, :],
                ).sum()

        # Compute MMD term with y and itself
        kernel_mm = 0.0
        for j1 in range(0, self.num_points_y, self.max_size):
            for j2 in range(0, self.num_points_y, self.max_size):
                kernel_mm += kernel.compute(
                    self.y[j1 : j1 + self.max_size, :],
                    self.y[j2 : j2 + self.max_size, :],
                ).sum()

        # Compute MMD term with x and y
        kernel_nm = 0.0
        for i in range(0, self.num_points_x, self.max_size):
            for j in range(0, self.num_points_y, self.max_size):
                kernel_nm += kernel.compute(
                    self.x[i : i + self.max_size, :], self.y[j : j + self.max_size, :]
                ).sum()

        # Compute expected output from MMD
        expected_output = (
            kernel_nn / self.num_points_x**2
            + kernel_mm / self.num_points_y**2
            - 2 * kernel_nm / (self.num_points_x * self.num_points_y)
        ) ** 0.5

        # Define a metric object
        metric = cm.MMD(kernel=kernel)

        # Compute MMD
        output = metric.compute(self.x, self.y, max_size=self.max_size)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=5)

    def test_mmd_equals_mmd_block(self) -> None:
        r"""
        Test that mmd() returns the same as mmd_block().
        """
        # Define a kernel object
        length_scale = 1.0
        kernel = ck.SquaredExponentialKernel(length_scale=length_scale)

        # Define a metric object
        metric = cm.MMD(kernel=kernel)

        # Check outputs are the same
        self.assertAlmostEqual(
            float(metric.compute(self.x, self.y)),
            float(metric.compute(self.x, self.y, max_size=self.max_size)),
            places=5,
        )

    def test_sum_weight_K(self) -> None:
        r"""
        Test sum_weight_K(), which calculates w^T*K*w matrices in blocks of max_size.

        For the dataset of 3 points in 2 dimensions :math:`X`, and second dataset
        :math:`Y`:

        .. math::

            X = [[0,0], [1,1], [2,2]]

            Y = [[0,0], [1,1]]

        the pairwise square distances are given by the matrix:

        .. math::

            k(X, Y) = \begin{bmatrix}0 & 2 \\ 2 & 0 \\ 8 & 2 \end{bmatrix}.

        Then, for weights vectors:

        .. math::

            w = [0.5, 0.5, 0],

            w_y = [1, 0]

        the product :math:`w^T*k(X, Y)*w_y = 1`.
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
        kernel.compute = cu.sq_dist_pairwise

        # Define a metric object
        metric = cm.MMD(kernel=kernel)

        # Compute sum of weighted pairwise distances
        output = metric.sum_weighted_pairwise_distances(
            x=x, y=y, weights_x=weights_x, weights_y=weights_y, max_size=2
        )

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=5)

    def test_mmd_weight_block_int(self) -> None:
        r"""
        Test the mmd_weight_block function.

        For
        .. math::

            X = [[0,0], [1,1], [2,2]],

            Y = [[0,0], [1,1]],

            w^T = [0.5, 0.5, 0],

            w_y^T = [1, 0],

        the weighted maximum mean discrepancy is given by:

        .. math::

            \text{WMMD}^2(X, Y) =
            w^T k(X,X) w + w_y^T k(Y, Y) w_y - 2 w^T k(X, Y) w_y

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
        kernel = ck.SquaredExponentialKernel(length_scale=length_scale)

        # Define a metric object
        metric = cm.MMD(kernel=kernel)

        # Compute weighted MMD block-wise
        output = metric.compute(
            x=x, y=y, weights_x=weights_x, weights_y=weights_y, max_size=2
        )

        # Check output matches expected
        self.assertAlmostEqual(float(output), float(expected_output), places=5)

    def test_mmd_weight_block_equals_mmd(self) -> None:
        r"""
        Test mmd_weight_block equals mmd when weights are uniform: w = 1/n, w_y = 1/m.
        """
        # Define a kernel object
        length_scale = 1.0
        kernel = ck.SquaredExponentialKernel(length_scale=length_scale)

        # Define a metric object
        metric = cm.MMD(kernel=kernel)

        # Compute weighted MMD with uniform weights
        output = metric.compute(
            self.x,
            self.y,
            weights_x=jnp.ones(self.num_points_x) / self.num_points_x,
            weights_y=jnp.ones(self.num_points_y) / self.num_points_y,
            max_size=self.max_size,
        )

        # Check output matches expected
        self.assertAlmostEqual(
            float(output), float(metric.compute(self.x, self.y)), places=5
        )


if __name__ == "__main__":
    unittest.main()
