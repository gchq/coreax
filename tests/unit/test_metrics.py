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


def gaussian_kernel(a: ArrayLike, b: ArrayLike, var: float = 1.0):
    r"""
    Define Gaussian kernel for use in test functions

    :param a: First set of vectors as a :math:`n \times d` array
    :param b: Second set of vectors as a :math:`m \times d` array
    :param var: Variance. Optional, defaults to 1
    """
    m = a.shape[0]
    n = b.shape[0]

    a_dots = (a * a).sum(axis=1).reshape((m, 1)) * np.ones(shape=(1, n))
    b_dots = (b * b).sum(axis=1) * np.ones(shape=(m, 1))
    dist_squared = a_dots + b_dots - 2 * a.dot(b.T)

    g_kern = np.exp(-dist_squared / (2 * var))
    return g_kern


class TestMetrics(unittest.TestCase):
    r"""
    Tests related to metrics.py functions.
    """

    def setUp(self):
        r"""
        Generate data for shared use across unit tests.
        Generate n random points in d dimensions from a uniform distribution [0, 1), and
        randomly select m points for coreset.
        Also generate weights: w, for original dataset, and w_c, for the coreset.

        : n: Number of test data points
        : d: Dimension of data
        : m: Number of points to randomly select for coreset
        : max_size: Maximum number of points for block calculations
        """

        self.n = 50
        self.d = 10
        self.m = 5
        self.max_size = 3

        self.x = np.random.rand(self.n, self.d)
        sel = np.random.choice(self.x.shape[0], size=self.m, replace=False)
        self.x_c = self.x[sel]

        self.w = np.random.rand(self.n, 1)
        # self.w_c = np.random.rand(self.m, 1)
        self.w_c = self.w[sel]

    def test_mmd(self) -> None:
        """
        Test the maximum mean discrepancy (MMD) function.

        Tests that MMD(X,X) = 0.

        Tests toy examples, with analytically determined results.

        Tests that MMD computed from randomly generated test data agrees with mmd().
        """

        self.assertAlmostEqual(mmd(self.x, self.x, rbf_kernel), 0.0, places=3)

        mmd_test = mmd(
            x=np.array([[0, 0], [1, 1], [0, 0], [1, 1]]),
            x_c=np.array([[0, 0], [1, 1]]),
            kernel=rbf_kernel,
        )
        self.assertAlmostEqual(mmd_test, 0.0, places=3)

        mmd_test = mmd(
            x=np.array([[0, 0], [1, 1], [2, 2]]),
            x_c=np.array([[0, 0], [1, 1]]),
            kernel=rbf_kernel,
        )
        self.assertAlmostEqual(
            mmd_test, (np.sqrt((3 - np.exp(-1) - 2 * np.exp(-4)) / 18)), places=5
        )

        Knn = gaussian_kernel(self.x, self.x)
        Kmm = gaussian_kernel(self.x_c, self.x_c)
        Knm = gaussian_kernel(self.x, self.x_c)

        mmd_rbf = (Knn.mean() + Kmm.mean() - 2 * Knm.mean()) ** 0.5
        mmd_rbf_test = mmd(self.x, self.x_c, rbf_kernel)

        self.assertAlmostEqual(mmd_rbf_test, mmd_rbf, places=3)

    def test_wmmd(self) -> None:
        """
        Test the weighted maximum mean discrepancy (wmmd) function.

        Tests toy example, with analytically determined result.

        Tests that WMMD computed from randomly generated test data agrees with wmmd().

        Tests that wmmd = mmd if weights = 1/m.
        """
        wmmd_test = wmmd(
            x=np.array([[0, 0], [1, 1], [2, 2]]),
            x_c=np.array([[0, 0], [1, 1]]),
            kernel=rbf_kernel,
            weights=np.array([1, 0]),
        )
        self.assertAlmostEqual(
            wmmd_test,
            (np.sqrt(2 / 3 - (2 / 9) * np.exp(-1) - (4 / 9) * np.exp(-4))),
            places=5,
        )

        Knn = gaussian_kernel(self.x, self.x)
        Kmm = gaussian_kernel(self.x_c, self.x_c)
        Knm = gaussian_kernel(self.x, self.x_c)

        wmmd_rbf = (
            np.mean(Knn)
            + np.dot(self.w_c.T, np.dot(Kmm, self.w_c))
            - 2 * np.dot(self.w_c.T, Knm.mean(axis=0))
        ).item() ** 0.5

        wmmd_rbf_test = wmmd(self.x, self.x_c, rbf_kernel, self.w_c)

        self.assertAlmostEqual(wmmd_rbf_test, wmmd_rbf, places=3)

        self.assertAlmostEqual(
            wmmd(self.x, self.x_c, rbf_kernel, np.ones(self.m) / self.m),
            mmd(self.x, self.x_c, rbf_kernel),
            places=3,
        )

    def test_sum_K(self) -> None:
        """
        Test the sum_K function.

        Tests for ValueError if max_size exceeds that of the data.

        Tests toy example, with analytically determined result.
        """

        self.assertRaises(ValueError, sum_K, self.x, self.x_c, rbf_kernel, max_size=100)

        kernel_sum_test = sum_K(
            x=np.array([[0, 0], [1, 1], [2, 2]]),
            y=np.array([[0, 0], [1, 1]]),
            k_pairwise=sq_dist_pairwise,
            max_size=2,
        )
        print(kernel_sum_test)
        self.assertAlmostEqual(kernel_sum_test, 14, places=3)

    def test_mmd_block(self) -> None:
        """
        Test the mmd_block function, which calculates MMD while limiting memory
        requirements.

        Tests toy example, with analytically determined result.

        Tests that MMD block-computed from randomly generated test data agrees with
        mmd_block().

        Tests that mmd() returns the same as mmd_block().
        """

        mmd_block_test = mmd_block(
            x=np.array([[0, 0], [1, 1], [2, 2]]),
            x_c=np.array([[0, 0], [1, 1]]),
            kernel=rbf_kernel,
            max_size=2,
        )
        self.assertAlmostEqual(
            mmd_block_test, np.sqrt((3 - np.exp(-1) - 2 * np.exp(-4)) / 18), places=3
        )

        Knn = 0.0
        for i1 in range(0, self.n, self.max_size):
            for i2 in range(0, self.n, self.max_size):
                Knn += gaussian_kernel(
                    self.x[i1 : i1 + self.max_size, :],
                    self.x[i2 : i2 + self.max_size, :],
                ).sum()

        Kmm = 0.0
        for j1 in range(0, self.m, self.max_size):
            for j2 in range(0, self.m, self.max_size):
                Kmm += gaussian_kernel(
                    self.x_c[j1 : j1 + self.max_size, :],
                    self.x_c[j2 : j2 + self.max_size, :],
                ).sum()

        Knm = 0.0
        for i in range(0, self.n, self.max_size):
            for j in range(0, self.m, self.max_size):
                Knm += gaussian_kernel(
                    self.x[i : i + self.max_size, :], self.x_c[j : j + self.max_size, :]
                ).sum()

        mmd_block_rbf = (
            Knn / self.n**2 + Kmm / self.m**2 - 2 * Knm / (self.n * self.m)
        ) ** 0.5

        mmd_block_test = mmd_block(self.x, self.x_c, rbf_kernel, self.max_size)

        self.assertAlmostEqual(mmd_block_rbf, mmd_block_test, places=3)

        self.assertAlmostEqual(
            mmd_block_rbf, mmd(self.x, self.x_c, rbf_kernel), places=3
        )

    def test_sum_weight_K(self) -> None:
        """
        Test the sum_weight_K function. Calculates w^T*K*w matrices in blocks of max_size.

        Tests for ValueError if max_size exceeds that of the data.

        Tests toy example, with analytically determined result.
        """

        self.assertRaises(
            ValueError,
            sum_weight_K,
            self.x,
            self.x_c,
            self.w,
            self.w_c,
            rbf_kernel,
            max_size=100,
        )

        sum_weight_K_test = sum_weight_K(
            x=np.array([[0, 0], [1, 1], [2, 2]]),
            y=np.array([[0, 0], [1, 1]]),
            w_x=np.asarray([0.5, 0.5, 0]),
            w_y=np.asarray([1, 0]),
            k_pairwise=sq_dist_pairwise,
            max_size=2,
        )

        self.assertAlmostEqual(sum_weight_K_test, 1.0, places=3)

    def test_mmd_weight_block(self) -> None:
        """
        Test the mmd_weight_block function.

        Tests toy example, with analytically determined result.

        Test equality with wmmd().

        Test equality with mmd() when weights = 1/n and coreset weights = 1/m.
        """

        mmd_weight_block_test = mmd_weight_block(
            x=np.array([[0, 0], [1, 1], [2, 2]]),
            x_c=np.array([[0, 0], [1, 1]]),
            w=np.asarray([0.5, 0.5, 0]),
            w_c=np.asarray([1, 0]),
            kernel=rbf_kernel,
            max_size=2,
        )
        self.assertAlmostEqual(
            mmd_weight_block_test, np.sqrt(1 / 2 - np.exp(-1) / 2), places=3
        )


if __name__ == "__main__":
    unittest.main()
