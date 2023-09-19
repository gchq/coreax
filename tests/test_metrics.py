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

import numpy as np

from coreax.kernel import *
from coreax.metrics import *


def dummy_data(n, d, m):
    """
    Randomly generate dummy data for use in unit tests of metrics.py.

    :param n: Number of test data points
    :param d: Dimension of data
    :param m: Number of points to randomly select for coreset
    """

    # Generate n random points in d dimensions from a uniform distribution [0, 1)
    x = np.random.rand(n, d)
    # Randomly select m points
    sel = np.random.choice(x.shape[0], size=m, replace=False)
    x_c = x[sel]

    return x, x_c


def gaussian_kernel(A, B, Var):
    """
    Define Gaussian kernel for use in test functions
    """
    M = A.shape[0]
    N = B.shape[0]

    A_dots = (A * A).sum(axis=1).reshape((M, 1)) * np.ones(shape=(1, N))
    B_dots = (B * B).sum(axis=1) * np.ones(shape=(M, 1))
    dist_squared = A_dots + B_dots - 2 * A.dot(B.T)

    g_kern = np.exp(-dist_squared / (2 * Var))
    return g_kern


class TestMetrics(unittest.TestCase):
    """
    Tests related to metrics.py functions.
    """

    def test_mmd(self) -> None:
        """
        Test the maximum mean discrepancy (MMD) function.
        """

        # Set test data parameters:
        n = 50  # number of points
        d = 10  # dimensions
        m = 5  # number of points for coreset, m<n

        x, x_c = dummy_data(n, d, m)

        # Calculate kernel matrices
        var = 1
        Knn = gaussian_kernel(x, x, var)
        Kmm = gaussian_kernel(x_c, x_c, var)
        Knm = gaussian_kernel(x, x_c, var)

        # Calculate MMD
        mmd_rbf = (Knn.mean() + Kmm.mean() - 2 * Knm.mean()) ** 0.5
        # Get MMD from function being tested
        mmd_rbf_test = mmd(x, x_c, rbf_kernel)
        # Test for equality
        self.assertAlmostEqual(mmd_rbf_test, mmd_rbf, places=3)

        t_mmd = mmd(x, x, rbf_kernel)
        # Test that MMD of data with itself is zero
        self.assertAlmostEqual(np.zeros(1), t_mmd, places=3)

    def test_wmmd(self) -> None:
        """
        Test the weighted maximum mean discrepancy (wmmd) function.
        """

        # Set test data parameters:
        n = 50  # number of points
        d = 10  # dimensions
        m = 5  # number of points for coreset, m<n

        # Generate n random points in d dimensions from a uniform distribution [0, 1)
        x, x_c = dummy_data(n, d, m)

        # Generate random weights vector and normalise so it sums to 1
        weights = np.random.rand(m, 1)

        var = 1

        Knn = gaussian_kernel(x, x, var)
        Kmm = gaussian_kernel(x_c, x_c, var)
        Knm = gaussian_kernel(x, x_c, var)
        # Calculate weighted MMD
        wmmd_rbf = (
            np.mean(Knn)
            + np.dot(weights.T, np.dot(Kmm, weights))
            - 2 * np.dot(weights.T, Knm.mean(axis=0))
        ).item() ** 0.5
        # Get weighted MMD from function being tested
        wmmd_rbf_test = wmmd(x, x_c, rbf_kernel, weights)
        # Assert equality
        self.assertAlmostEqual(wmmd_rbf_test, wmmd_rbf, places=3)

        # Test equality with mmd() if weights = 1/m
        self.assertAlmostEqual(
            wmmd(x, x_c, rbf_kernel, np.ones(m) / m), mmd(x, x_c, rbf_kernel), places=3
        )

    def test_sum_K(self) -> None:
        """
        Test the sum_K function.
        """

        # Generate test data
        n = 10  # number of points
        d = 1  # dimensions
        m = 5  # number of points for coreset, m=<n
        x, x_c = dummy_data(n, d, m)

        # Test for Value Error when max_size exceeds m
        self.assertRaises(ValueError, sum_K, x, x_c, rbf_kernel, max_size=100)

        max_size = 5

        # Calculate kernel sum
        kernel_sum = 0.0
        for i in range(0, n, max_size):
            for j in range(0, m, max_size):
                kern_p = gaussian_kernel(x[i : i + max_size], x_c[j : j + max_size], 1)
                kernel_sum += kern_p.sum()

        # Get kernel sum from function being tested
        kernel_sum_test = sum_K(x[:, 0], x_c[:, 0], rbf_kernel, max_size)
        # Assert equality
        self.assertAlmostEqual(kernel_sum, kernel_sum_test, places=3)

    def test_mmd_block(self) -> None:
        """
        Test the mmd_block function, which calculates MMD while limiting memory
        requirements.
        """
        n = 50  # number of points
        d = 2  # dimensions
        m = 10  # number of points for coreset, m<n
        x, x_c = dummy_data(n, d, m)

        max_size = 5

        var = 1

        # Calculate K(x,x) matrix in blocks of max_size
        Knn = 0.0
        for i1 in range(0, n, max_size):
            for i2 in range(0, n, max_size):
                Knn += gaussian_kernel(
                    x[i1 : i1 + max_size, :], x[i2 : i2 + max_size, :], var
                ).sum()

        # Calculate K(x_c,x_c) matrix in blocks of max_size
        Kmm = 0.0
        for j1 in range(0, m, max_size):
            for j2 in range(0, m, max_size):
                Kmm += gaussian_kernel(
                    x_c[j1 : j1 + max_size, :], x_c[j2 : j2 + max_size, :], var
                ).sum()

        # Calculate K(x,x_c) matrix in blocks of max_size
        Knm = 0.0
        for i in range(0, n, max_size):
            for j in range(0, m, max_size):
                Knm += gaussian_kernel(
                    x[i : i + max_size, :], x_c[j : j + max_size, :], var
                ).sum()

        # Average over kernel matrices to calculate MMD
        mmd_block_rbf = (Knn / n**2 + Kmm / m**2 - 2 * Knm / (n * m)) ** 0.5
        # Get MMD from function being tested
        mmd_block_test = mmd_block(x, x_c, rbf_kernel, max_size)
        # Assert equality
        self.assertAlmostEqual(mmd_block_rbf, mmd_block_test, places=3)

        # Test equality with mmd()
        self.assertAlmostEqual(mmd_block_rbf, mmd(x, x_c, rbf_kernel), places=3)

    def test_sum_weight_K(self) -> None:
        """
        Test the sum_weight_K function.
        """

        n = 10  # number of points
        d = 1  # dimensions
        m = 5  # number of points for coreset, m=<n
        x, y = dummy_data(n, d, m)

        w_x = np.random.rand(n, 1)
        w_y = np.random.rand(m, 1)

        # Test for Value Error when max_size exceeds m
        self.assertRaises(
            ValueError, sum_weight_K, x, y, w_x, w_y, rbf_kernel, max_size=100
        )

        max_size = 5

        # Calculate weighted kernel sum
        weight_kernel_sum = 0.0
        for i in range(0, n, max_size):
            for j in range(0, m, max_size):
                kern_p = (
                    w_x[i : i + max_size, None]
                    * gaussian_kernel(x[i : i + max_size], y[j : j + max_size], 1)
                    * w_y[None, j : j + max_size]
                )

                weight_kernel_sum += kern_p.sum()

        # Get weighted kernel sum from function being tested
        sum_weight_K_test = sum_weight_K(x, y, w_x, w_y, rbf_kernel, max_size)
        # Assert equality
        self.assertAlmostEqual(weight_kernel_sum, sum_weight_K_test, places=3)

    def test_mmd_weight_block(self) -> None:
        """
        Test the mmd_weight_block function.
        """

        # Set test data parameters:
        n = 50  # number of points
        d = 10  # dimensions
        m = 5  # number of points for coreset, m<n

        # Generate n random points in d dimensions from a uniform distribution [0, 1)
        x, x_c = dummy_data(n, d, m)

        # Generate random weights vector and normalise so it sums to 1
        w = np.random.rand(n, 1)
        w_c = np.random.rand(m, 1)

        max_size = 5
        var = 1

        # Calculate w^T*K(x,x)*w matrix in blocks of max_size
        Knn = 0.0
        for i1 in range(0, n, max_size):
            for i2 in range(0, n, max_size):
                Knn += np.dot(
                    w[i1 : i1 + max_size].T,
                    np.dot(
                        gaussian_kernel(
                            x[i1 : i1 + max_size, :], x[i2 : i2 + max_size, :], var
                        ),
                        w[i2 : i2 + max_size],
                    ),
                ).sum()

        # Calculate w_c^T*K(x_c,x_c)*w_c matrix in blocks of max_size
        Kmm = 0.0
        for j1 in range(0, m, max_size):
            for j2 in range(0, m, max_size):
                Kmm += np.dot(
                    w_c[j1 : j1 + max_size].T,
                    np.dot(
                        gaussian_kernel(
                            x_c[j1 : j1 + max_size, :], x_c[j2 : j2 + max_size, :], var
                        ),
                        w_c[j2 : j2 + max_size],
                    ),
                ).sum()

        # Calculate w^T*K(x,x_c)*w_c matrix in blocks of max_size
        Knm = 0.0
        for i in range(0, n, max_size):
            for j in range(0, m, max_size):
                Knm += np.dot(
                    w[i : i + max_size].T,
                    np.dot(
                        gaussian_kernel(
                            x[i : i + max_size, :], x_c[j : j + max_size, :], var
                        ),
                        w_c[j : j + max_size],
                    ),
                ).sum()

        # Average over kernel matrices to calculate MMD
        mmd_weight_block_rbf = (Knn / n**2 + Kmm / m**2 - 2 * Knm / (n * m)) ** 0.5
        # Get MMD from function being tested
        mmd_weight_block_test = mmd_weight_block(x, x_c, w, w_c, rbf_kernel, max_size)
        # Assert equality
        self.assertAlmostEqual(mmd_weight_block_rbf, mmd_weight_block_test, places=3)

        # Test equality with wmmd() when original dataset weights = 1/n
        self.assertAlmostEqual(
            mmd_weight_block(x, x_c, np.ones(n) / n, w_c, rbf_kernel, max_size),
            wmmd(x, x_c, rbf_kernel, w_c),
            places=3,
        )


if __name__ == "__main__":
    unittest.main()
