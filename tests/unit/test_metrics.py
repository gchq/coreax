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

import jax.numpy as jnp
from jax import random

import coreax.kernel as ck
import coreax.metrics as cm


def gaussian_kernel(a: cm.ArrayLike, b: cm.ArrayLike, var: float = 1.0) -> cm.ArrayLike:
    r"""
    Define the Gaussian kernel (aka RBF kernel) for using in test functions.

    :param a: First set of vectors as a :math:`n \times d` array
    :param b: Second set of vectors as a :math:`m \times d` array
    :param var: Variance. Optional, defaults to 1
    :return: Gaussian kernel matrix, :math:`n \times m` array with entry :math:`i,j` as
     :math:`k(a_i, b_j)`, the kernel evaluated for points :math:`a_i, b_j`
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

        Generate n random points in d dimensions from a uniform distribution [0, 1).
        Randomly select m points for coreset.
        Generate weights: w for original data, w_c for coreset.

        :param n: Number of test data points
        :param d: Dimension of data
        :param m: Number of points to randomly select for coreset
        :param max_size: Maximum number of points for block calculations
        """

        self.n = 30
        self.d = 10
        self.m = 5
        self.max_size = 3

        self.x = random.uniform(random.PRNGKey(0), shape=(self.n, self.d))
        self.x_c = random.choice(random.PRNGKey(0), self.x, shape=(self.m,))
        self.w = random.uniform(random.PRNGKey(0), shape=(self.n,)) / self.n
        self.w_c = random.uniform(random.PRNGKey(0), shape=(self.m,)) / self.m

    def test_mmd_XX(self) -> None:
        r"""
        Test the MMD of a dataset with itself is zero, for several different kernels.
        """

        self.assertAlmostEqual(cm.mmd(self.x, self.x, ck.rbf_kernel), 0.0)

        self.assertAlmostEqual(cm.mmd(self.x, self.x, ck.laplace_kernel), 0.0)

        self.assertAlmostEqual(cm.mmd(self.x, self.x, ck.pc_imq), 0.0)

    def test_mmd_ones(self):
        r"""
        Test mmd function with a small example dataset of ones and zeros.

        For the dataset of 4 points in 2 dimensions, :math:`X`, and a coreset,
        :math:`X_c`, given by:

        .. math::

            X = [[0,0], [1,1], [0,0], [1,1]]

            X_c = [[0,0],[1,1]]

        the Gaussian (aka radial basis function) kernel, :math:`k(x,y) = \exp (-||
        x-y||^2/2\sigma^2)`, gives:

        .. math::

            k(X,X) = \exp(-\begin{bmatrix}0 & 2 & 0 & 2 \\ 2 & 0 & 2 & 0\\ 0 & 2 & 0 &
            2 \\ 2 & 0 & 2 & 0\end{bmatrix}/2\sigma^2).

            k(X_c,X_c) = \exp(-\begin{bmatrix}0 & 2  \\ 2 & 0 \end{bmatrix}/2\sigma^2).

            k(X,X_c) = \exp(-\begin{bmatrix}0 & 2  \\ 2 & 0 \\0 & 2  \\ 2 & 0
            \end{bmatrix}/2\sigma^2).

        Then

        .. math::

            \text{MMD}^2(X,X_c) = \mathbb{E}(k(X,X)) + \mathbb{E}(k(X_c,X_c)) -
            2\mathbb{E}(k(X,X_c))

            = \frac{1}{2} + e^{-1/2} + \frac{1}{2} + e^{-1/2} - 2\left(\frac{1}{2}
            + e^{-1/2}\right)

            = 0.
        """

        mmd_test = cm.mmd(
            x=jnp.array([[0, 0], [1, 1], [0, 0], [1, 1]]),
            x_c=jnp.array([[0, 0], [1, 1]]),
            kernel=ck.rbf_kernel,
        )
        self.assertAlmostEqual(mmd_test, 0.0, places=5)

    def test_mmd_ints(self):
        r"""
        Test MMD function with a small example dataset of integers.

        For the dataset of 3 points in 2 dimensions, :math:`X`, and coreset
        :math:`X_c`, given by:

        .. math::

            X = [[0,0], [1,1], [2,2]]

            X_c = [[0,0],[1,1]]

        the RBF kernel, :math:`k(x,y) = \exp (-||x-y||^2/2\sigma^2)`, gives:

        .. math::

            k(X,X) = \exp(-\begin{bmatrix}0 & 2 & 8 \\ 2 & 0 & 2 \\ 8 & 2 & 0
            \end{bmatrix}/2\sigma^2) = \begin{bmatrix}1 & e^{-1} & e^{-4} \\ e^{-1} &
            1 & e^{-1} \\ e^{-4} & e^{-1} & 1\end{bmatrix}.

            k(X_c,X_c) = \exp(-\begin{bmatrix}0 & 2 \\ 2 & 0 \end{bmatrix}/2\sigma^2) =
             \begin{bmatrix}1 & e^{-1}\\ e^{-1} & 1\end{bmatrix}.

            k(X,X_c) =  \exp(-\begin{bmatrix}0 & 2 & 8 \\ 2 & 0 & 2 \end{bmatrix}
            /2\sigma^2) = \begin{bmatrix}1 & e^{-1} \\  e^{-1} & 1 \\ e^{-4} & e^{-1}
            \end{bmatrix}.

        Then

        .. math::

            \text{MMD}^2(X,X_c) = \mathbb{E}(k(X,X)) + \mathbb{E}(k(X_c,X_c)) -
            2\mathbb{E}(k(X,X_c))

            = \frac{3+4e^{-1}+2e^{-4}}{9} + \frac{2 + 2e^{-1}}{2} -2 \times
            \frac{2 + 3e^{-1}+e^{-4}}{6}

            = \frac{3 - e^{-1} -2e^{-4}}{18}.
        """

        mmd_test = cm.mmd(
            x=jnp.array([[0, 0], [1, 1], [2, 2]]),
            x_c=jnp.array([[0, 0], [1, 1]]),
            kernel=ck.rbf_kernel,
        )
        self.assertAlmostEqual(
            mmd_test, (jnp.sqrt((3 - jnp.exp(-1) - 2 * jnp.exp(-4)) / 18)), places=5
        )

    def test_mmd_rand(self):
        r"""
        Test that MMD computed from randomly generated test data agrees with mmd().
        """

        Knn = gaussian_kernel(self.x, self.x)
        Kmm = gaussian_kernel(self.x_c, self.x_c)
        Knm = gaussian_kernel(self.x, self.x_c)

        mmd_test = (Knn.mean() + Kmm.mean() - 2 * Knm.mean()) ** 0.5

        self.assertAlmostEqual(
            mmd_test, cm.mmd(self.x, self.x_c, ck.rbf_kernel), places=5
        )

    def test_wmmd_ints(self) -> None:
        r"""
        Test wmmd() function with a small example dataset of integers.

        For the dataset of 3 points in 2 dimensions :math:`X`, coreset :math:`X_c`, and
        coreset weights :math:`w_c`, given by:

        .. math::

            X = [[0,0], [1,1], [2,2]]

            X_c = [[0,0],[1,1]]

            w_c = [1,0]

        the weighted maximum mean discrepancy is calculated via:

        .. math::

            \text{WMMD}^2(X,X_c) = \mathbb{E}(k(X,X)) + w_c^T k(X_c,X_c) w_c
             - 2\mathbb{E}_X(k(X,X_c)) w_c

            = \frac{3+4e^{-1}+2e^{-4}}{9} + 1 - 2 \times \frac{1 + e^{-1} + e^{-4}}{3}

            = \frac{2}{3} - \fracv{2}{9}e^{-1} - \frac{4}{9}e^{-4}.
        """

        wmmd_test = cm.wmmd(
            x=jnp.array([[0, 0], [1, 1], [2, 2]]),
            x_c=jnp.array([[0, 0], [1, 1]]),
            kernel=ck.rbf_kernel,
            weights=jnp.array([1, 0]),
        )
        self.assertAlmostEqual(
            wmmd_test,
            jnp.sqrt(2 / 3 - (2 / 9) * jnp.exp(-1) - (4 / 9) * jnp.exp(-4)),
            places=5,
        )

    def test_wmmd_rand(self) -> None:
        r"""
        Test that WMMD computed from randomly generated test data agrees with wmmd().
        """

        Knn = gaussian_kernel(self.x, self.x)
        Kmm = gaussian_kernel(self.x_c, self.x_c)
        Knm = gaussian_kernel(self.x, self.x_c)

        wmmd_test = (
            jnp.mean(Knn)
            + jnp.dot(self.w_c.T, jnp.dot(Kmm, self.w_c))
            - 2 * jnp.dot(self.w_c.T, Knm.mean(axis=0))
        ).item() ** 0.5

        self.assertAlmostEqual(
            wmmd_test, cm.wmmd(self.x, self.x_c, ck.rbf_kernel, self.w_c), places=5
        )

    def test_wmmd_uniform_weights(self) -> None:
        r"""
        Test that wmmd = mmd if weights are uniform, :math:`w_c = 1/m`.
        """

        wmmd_test = cm.wmmd(self.x, self.x_c, ck.rbf_kernel, jnp.ones(self.m) / self.m)
        self.assertAlmostEqual(
            wmmd_test, cm.mmd(self.x, self.x_c, ck.rbf_kernel), places=5
        )

    def test_sum_K(self) -> None:
        r"""
        Test sum_K() with a small integer example.

        For the dataset of 3 points in 2 dimensions :math:`X`, and coreset :math:`X_c`:

        .. math::

            X = [[0,0], [1,1], [2,2]]

            X_c = [[0,0],[1,1]]

        the pairwise square distances are given by the matrix:

        .. math::

            \begin{bmatrix}0 & 2 \\ 2 & 0 \\ 8 & 2 \end{bmatrix}

        which, summing across both axes, gives the result :math:`14`.
        """

        kernel_sum_test = cm.sum_K(
            x=jnp.array([[0, 0], [1, 1], [2, 2]]),
            y=jnp.array([[0, 0], [1, 1]]),
            k_pairwise=ck.sq_dist_pairwise,
            max_size=2,
        )

        self.assertAlmostEqual(kernel_sum_test, 14, places=5)

    def test_mmd_block_ints(self) -> None:
        r"""
        Test mmd_block calculation of MMD wile limiting memory requirements.

        This test uses the same 2D, three-point dataset and coreset as test_mmd_ints().
        """

        mmd_block_test = cm.mmd_block(
            x=jnp.array([[0, 0], [1, 1], [2, 2]]),
            x_c=jnp.array([[0, 0], [1, 1]]),
            kernel=ck.rbf_kernel,
            max_size=2,
        )
        self.assertAlmostEqual(
            mmd_block_test, jnp.sqrt((3 - jnp.exp(-1) - 2 * jnp.exp(-4)) / 18), places=5
        )

    def test_mmd_block_rand(self) -> None:
        r"""
        Test that mmd block-computed for random test data equals mmd_block().
        """

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

        mmd_block_test = (
            Knn / self.n**2 + Kmm / self.m**2 - 2 * Knm / (self.n * self.m)
        ) ** 0.5

        self.assertAlmostEqual(
            mmd_block_test,
            cm.mmd_block(self.x, self.x_c, ck.rbf_kernel, self.max_size),
            places=5,
        )

    def test_mmd_equals_mmd_block(self) -> None:
        r"""
        Test that mmd() returns the same as mmd_block().
        """

        mmd_test = cm.mmd(self.x, self.x_c, ck.rbf_kernel)
        mmd_block_test = cm.mmd_block(self.x, self.x_c, ck.rbf_kernel, self.max_size)

        self.assertAlmostEqual(mmd_test, mmd_block_test, places=5)

    def test_sum_weight_K(self) -> None:
        r"""
        Test sum_weight_K(), which calculates w^T*K*w matrices in blocks of max_size.

        For the dataset of 3 points in 2 dimensions :math:`X`, and coreset :math:`X_c`:

        .. math::

            X = [[0,0], [1,1], [2,2]]

            X_c = [[0,0],[1,1]]

        the pairwise square distances are given by the matrix:

        .. math::

            k(X, X_c) = \begin{bmatrix}0 & 2 \\ 2 & 0 \\ 8 & 2 \end{bmatrix}.

        Then, for weights vectors:

        .. math::

            w = [0.5, 0.5, 0],

            w_c = [1, 0]

        the product :math:`w^T*k(X, X_c)*w_c = 1`.
        """

        sum_weight_K_test = cm.sum_weight_K(
            x=jnp.array([[0, 0], [1, 1], [2, 2]]),
            y=jnp.array([[0, 0], [1, 1]]),
            w_x=jnp.array([0.5, 0.5, 0]),
            w_y=jnp.array([1, 0]),
            k_pairwise=ck.sq_dist_pairwise,
            max_size=2,
        )

        self.assertAlmostEqual(sum_weight_K_test, 1.0, places=5)

    def test_mmd_weight_block_int(self) -> None:
        r"""
        Test the mmd_weight_block function.

        For
        .. math::

            X = [[0,0], [1,1], [2,2]],

            X_c = [[0,0],[1,1]],

            w^T = [0.5, 0.5, 0],

            w_c^T = [1, 0],

        the weighted maximum mean discrepancy is given by:

        .. math::

            \text{WMMD}^2(X, X_c) =
            w^T k(X,X) w + w_c^T k(X_c, X_c) w_c - 2 w^T k(X, X_c) w_c

        which, when :math:`k(x,y)` is the RBF kernel, reduces to:

        .. math::

            \frac{1}{2} + \frac{e^{-1}}{2} + 1 - 2\times(\frac{1}{2} + \frac{e^{-1}}{2})

            = \frac{1}{2} - \frac{e^{-1}}{2}.
        """

        mmd_weight_block_test = cm.mmd_weight_block(
            x=jnp.array([[0, 0], [1, 1], [2, 2]]),
            x_c=jnp.array([[0, 0], [1, 1]]),
            w=jnp.array([0.5, 0.5, 0]),
            w_c=jnp.array([1, 0]),
            kernel=ck.rbf_kernel,
            max_size=2,
        )
        self.assertAlmostEqual(
            mmd_weight_block_test, jnp.sqrt(1 / 2 - jnp.exp(-1) / 2), places=5
        )

    def test_mmd_weght_block_equals_mmd(self) -> None:
        r"""
        Test mmd_weight_block equals mmd when weights are uniform: w = 1/n, w_c = 1/m.
        """

        mmd_weight_block_test = cm.mmd_weight_block(
            self.x,
            self.x_c,
            jnp.ones(self.n) / self.n,
            jnp.ones(self.m) / self.m,
            ck.rbf_kernel,
        )

        self.assertAlmostEqual(
            mmd_weight_block_test, cm.mmd(self.x, self.x_c, ck.rbf_kernel), places=5
        )


if __name__ == "__main__":
    unittest.main()
