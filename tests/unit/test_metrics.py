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

from jax import random

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

    a_dots = (a * a).sum(axis=1).reshape((m, 1)) * jnp.ones(shape=(1, n))
    b_dots = (b * b).sum(axis=1) * jnp.ones(shape=(m, 1))
    dist_squared = a_dots + b_dots - 2 * a.dot(b.T)

    g_kern = jnp.exp(-dist_squared / (2 * var))
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

        self.n = 30
        self.d = 10
        self.m = 5
        self.max_size = 3

        self.x = random.uniform(random.PRNGKey(0), shape=(self.n, self.d))
        self.x_c = random.choice(random.PRNGKey(0), self.x, shape=(self.m,))
        self.w = random.uniform(random.PRNGKey(0), shape=(self.n,)) / self.n
        self.w_c = random.uniform(random.PRNGKey(0), shape=(self.m,)) / self.m

    def test_mmd(self) -> None:
        r"""
        Tests the maximum mean discrepancy (MMD) function.

        1. Tests that MMD(X,X) = 0.
        2 & 3. Tests toy examples, with analytically determined results.
        4. Tests that MMD computed from randomly generated test data agrees with mmd().
        """

        self.assertAlmostEqual(mmd(self.x, self.x, rbf_kernel), 0.0, places=3)

        mmd_test2 = mmd(
            x=jnp.array([[0, 0], [1, 1], [0, 0], [1, 1]]),
            x_c=jnp.array([[0, 0], [1, 1]]),
            kernel=rbf_kernel,
        )
        self.assertAlmostEqual(mmd_test2, 0.0, places=3)

        mmd_test3 = mmd(
            x=jnp.array([[0, 0], [1, 1], [2, 2]]),
            x_c=jnp.array([[0, 0], [1, 1]]),
            kernel=rbf_kernel,
        )
        self.assertAlmostEqual(
            mmd_test3, (jnp.sqrt((3 - jnp.exp(-1) - 2 * jnp.exp(-4)) / 18)), places=5
        )

        Knn = gaussian_kernel(self.x, self.x)
        Kmm = gaussian_kernel(self.x_c, self.x_c)
        Knm = gaussian_kernel(self.x, self.x_c)

        mmd_test4 = (Knn.mean() + Kmm.mean() - 2 * Knm.mean()) ** 0.5

        self.assertAlmostEqual(mmd_test4, mmd(self.x, self.x_c, rbf_kernel), places=3)

    def test_wmmd(self) -> None:
        r"""
        Tests the weighted maximum mean discrepancy (wmmd) function.

        1. Tests toy example, with analytically determined result.
        2. Tests that WMMD computed from randomly generated test data agrees with wmmd().
        3. Tests that wmmd = mmd if weights = 1/m.
        """
        wmmd_test1 = wmmd(
            x=jnp.array([[0, 0], [1, 1], [2, 2]]),
            x_c=jnp.array([[0, 0], [1, 1]]),
            kernel=rbf_kernel,
            weights=jnp.array([1, 0]),
        )
        self.assertAlmostEqual(
            wmmd_test1,
            (jnp.sqrt(2 / 3 - (2 / 9) * jnp.exp(-1) - (4 / 9) * jnp.exp(-4))),
            places=5,
        )

        Knn = gaussian_kernel(self.x, self.x)
        Kmm = gaussian_kernel(self.x_c, self.x_c)
        Knm = gaussian_kernel(self.x, self.x_c)

        wmmd_test2 = (
            jnp.mean(Knn)
            + jnp.dot(self.w_c.T, jnp.dot(Kmm, self.w_c))
            - 2 * jnp.dot(self.w_c.T, Knm.mean(axis=0))
        ).item() ** 0.5

        self.assertAlmostEqual(
            wmmd_test2, wmmd(self.x, self.x_c, rbf_kernel, self.w_c), places=3
        )

        wmmd_test3 = wmmd(self.x, self.x_c, rbf_kernel, jnp.ones(self.m) / self.m)
        self.assertAlmostEqual(wmmd_test3, mmd(self.x, self.x_c, rbf_kernel), places=3)

    def test_sum_K(self) -> None:
        r"""
        Tests the sum_K function.

        1. Tests toy example, with analytically determined result.
        """

        kernel_sum_test = sum_K(
            x=jnp.array([[0, 0], [1, 1], [2, 2]]),
            y=jnp.array([[0, 0], [1, 1]]),
            k_pairwise=sq_dist_pairwise,
            max_size=2,
        )

        self.assertAlmostEqual(kernel_sum_test, 14, places=3)

    def test_mmd_block(self) -> None:
        r"""
        Tests the mmd_block function, which calculates MMD while limiting memory
        requirements.

        1. Tests toy example, with analytically determined result.
        2. Tests that MMD block-computed from randomly generated test data agrees with
        mmd_block().
        3. Tests that mmd() returns the same as mmd_block().
        """

        mmd_block_test1 = mmd_block(
            x=jnp.array([[0, 0], [1, 1], [2, 2]]),
            x_c=jnp.array([[0, 0], [1, 1]]),
            kernel=rbf_kernel,
            max_size=2,
        )
        self.assertAlmostEqual(
            mmd_block_test1,
            jnp.sqrt((3 - jnp.exp(-1) - 2 * jnp.exp(-4)) / 18),
            places=3,
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

        mmd_block_test2 = (
            Knn / self.n**2 + Kmm / self.m**2 - 2 * Knm / (self.n * self.m)
        ) ** 0.5

        self.assertAlmostEqual(
            mmd_block_test2,
            mmd_block(self.x, self.x_c, rbf_kernel, self.max_size),
            places=3,
        )

        self.assertAlmostEqual(
            mmd_block_test2, mmd(self.x, self.x_c, rbf_kernel), places=3
        )

    def test_sum_weight_K(self) -> None:
        r"""
        Tests the sum_weight_K function. Calculates w^T*K*w matrices in blocks of max_size.

        1. Tests toy example, with analytically determined result.
        """

        sum_weight_K_test = sum_weight_K(
            x=jnp.array([[0, 0], [1, 1], [2, 2]]),
            y=jnp.array([[0, 0], [1, 1]]),
            w_x=jnp.array([0.5, 0.5, 0]),
            w_y=jnp.array([1, 0]),
            k_pairwise=sq_dist_pairwise,
            max_size=2,
        )

        self.assertAlmostEqual(sum_weight_K_test, 1.0, places=3)

    def test_mmd_weight_block(self) -> None:
        r"""
        Tests the mmd_weight_block function.

        1. Tests toy example, with analytically determined result.
        2. Test equality with mmd() when w = 1/n and coreset weights w_c = 1/m.

        """

        mmd_weight_block_test1 = mmd_weight_block(
            x=jnp.array([[0, 0], [1, 1], [2, 2]]),
            x_c=jnp.array([[0, 0], [1, 1]]),
            w=jnp.array([0.5, 0.5, 0]),
            w_c=jnp.array([1, 0]),
            kernel=rbf_kernel,
            max_size=2,
        )
        self.assertAlmostEqual(
            mmd_weight_block_test1, jnp.sqrt(1 / 2 - jnp.exp(-1) / 2), places=3
        )

        mmd_weight_block_test2 = mmd_weight_block(
            self.x,
            self.x_c,
            jnp.ones(self.n) / self.n,
            jnp.ones(self.m) / self.m,
            rbf_kernel,
        )

        self.assertAlmostEqual(
            mmd_weight_block_test2, mmd(self.x, self.x_c, rbf_kernel), places=3
        )


if __name__ == "__main__":
    unittest.main()
