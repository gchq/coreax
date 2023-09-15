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
from scipy.stats import ortho_group, norm
from jax import grad, vjp
from typing import Callable
from functools import partial
import numpy as np

from coreax.kernel import *


class TestKernels(unittest.TestCase):
    """
    Tests related to kernel.py functions.
    """

    def test_sq_dist(self) -> None:
        """
        Test square distance under float32.
        """
        m = ortho_group.rvs(dim=2)
        x = m[0]
        y = m[1]
        d = jnp.linalg.norm(x - y) ** 2
        td = sq_dist(x, y)
        self.assertAlmostEqual(d, td, places=3)
        td = sq_dist(x, x)
        self.assertAlmostEqual(0.0, td, places=3)
        td = sq_dist(y, y)
        self.assertAlmostEqual(0.0, td, places=3)

    def test_sq_dist_pairwise(self) -> None:
        """
        Test vmap version of sq distance.
        """
        # create an orthonormal matrix
        d = 3
        m = ortho_group.rvs(dim=d)
        tinner = sq_dist_pairwise(m, m)
        ans = np.ones((d, d)) * 2.0
        np.fill_diagonal(ans, 0.0)
        # Frobenius norm
        td = jnp.linalg.norm(tinner - ans)
        self.assertAlmostEqual(td, 0.0, places=3)

    def test_rbf_kernel(self) -> None:
        """
        Test the RBF kernel.

        Note that the bandwidth is the 'variance' of the sq exp.
        """
        bandwidth = np.float32(np.pi) / 2.0
        x = np.arange(10)
        y = x + 1.0
        ans = np.exp(-np.linalg.norm(x - y) ** 2 / (2.0 * bandwidth))
        tst = rbf_kernel(x, y, bandwidth)
        self.assertAlmostEqual(jnp.linalg.norm(ans - tst), 0.0, places=3)

    def test_laplace_kernel(self) -> None:
        """
        Test the Laplace kernel.

        Note that in this case, the norm isn't squared.
        """
        bandwidth = np.float32(np.pi) / 2.0
        x = np.arange(10)
        y = x + 1.0
        ans = np.exp(-np.linalg.norm(x - y) / (2.0 * bandwidth))
        tst = laplace_kernel(x, y, bandwidth)
        self.assertAlmostEqual(jnp.linalg.norm(ans - tst), 0.0, places=3)

    def test_pdiff(self) -> None:
        """
        Test the function pdiff.

        This test ensures efficient computation of pairwise differences.
        """
        m = 10
        n = 10
        d = 3
        X = np.random.random((n, d))
        Y = np.random.random((m, d))
        Z = []
        for x in X:
            row = []
            for y in Y:
                row.append(x - y)
            Z.append(list(row))
        Z = np.array(Z)
        tst = pdiff(X, Y)
        self.assertAlmostEqual(jnp.linalg.norm(tst - Z), 0.0, places=3)

    def test_gaussian_kernel(self) -> None:
        """
        Test the normalised RBF (Gaussian) kernel.
        """
        std_dev = np.e
        n = 10
        X = np.arange(n)
        Y = X + 1.0
        K = np.zeros((n, n))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i, j] = norm(y, std_dev).pdf(x)
        tst = normalised_rbf(X, Y, std_dev)
        self.assertAlmostEqual(jnp.linalg.norm(K - tst), 0.0, places=3)

    def test_pc_imq(self) -> None:
        """
        Test the function pc_imq (Inverse multi-quadric, pre-conditioned).

        Note that the bandwidth is the 'variance' of the sq exp.
        """
        std_dev = np.e
        n = 10
        X = np.arange(n)
        Y = X + 1.0
        K = np.zeros((n, n))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i, j] = 1.0 / np.sqrt(1.0 + ((x - y) / std_dev) ** 2 / 2.0)
        tst = pc_imq(X, Y, std_dev)
        self.assertAlmostEqual(jnp.linalg.norm(K - tst), 0.0, places=3)

    def test_rbf_f_x(self) -> None:
        r"""
        Test the kernel density estimation (KDE) PDF for a radial basis function.
        """
        std_dev = np.e
        n = 10
        X = np.arange(n)
        Y = X + 1.0
        K = np.zeros((n, n))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i, j] = norm(y, std_dev).pdf(x)
        tst_mean, tst_val = rbf_f_x(X, Y, std_dev)
        self.assertAlmostEqual(
            jnp.linalg.norm(K.mean(axis=1) - tst_mean), 0.0, places=3
        )
        self.assertAlmostEqual(jnp.linalg.norm(K - tst_val), 0.0, places=3)

    def test_grad_rbf_x(self) -> None:
        r"""
        Test the gradient of the RBF kernel (analytic).
        """
        bandwidth = 1 / np.sqrt(2)
        n = 10
        d = 2
        X = np.random.random((n, d))
        Y = np.random.random((n, d))
        K = np.zeros((n, n, d))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i, j] = (
                    -(x - y)
                    / bandwidth**3
                    * np.exp(-np.linalg.norm(x - y) ** 2 / (2 * bandwidth**2))
                    / (np.sqrt(2 * np.pi))
                )

        tst = grad_rbf_x(X, Y, bandwidth)
        self.assertAlmostEqual(jnp.linalg.norm(K - tst), 0.0, places=3)

    def test_grad_rbf_y(self) -> None:
        r"""
        Test the gradient of the RBF kernel (analytic).
        """
        bandwidth = 1 / np.sqrt(2)
        n = 10
        d = 2
        X = np.random.random((n, d))
        Y = np.random.random((n, d))
        K = np.zeros((n, n, d))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i, j] = (
                    (x - y)
                    / bandwidth**3
                    * np.exp(-np.linalg.norm(x - y) ** 2 / (2 * bandwidth**2))
                    / (np.sqrt(2 * np.pi))
                )

        tst = grad_rbf_y(X, Y, bandwidth)
        self.assertAlmostEqual(jnp.linalg.norm(K - tst), 0.0, places=3)

    def test_grad_pc_imq_x(self) -> None:
        r"""
        Test the gradient of the PC-IMQ kernel wrt x argument
        """
        bandwidth = 1 / np.sqrt(2)
        n = 10
        d = 2
        X = np.random.random((n, d))
        Y = np.random.random((n, d))
        K = np.zeros((n, n, d))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i, j] = -(x - y) / (1 + np.linalg.norm(x - y) ** 2) ** (3 / 2)
        tst = grad_pc_imq_x(X, Y, bandwidth)
        self.assertAlmostEqual(jnp.linalg.norm(K - tst), 0.0, places=3)

    def test_grad_pc_imq_y(self) -> None:
        r"""
        Test the gradient of the PC-IMQ kernel wrt x argument
        """
        bandwidth = 1 / np.sqrt(2)
        n = 10
        d = 2
        X = np.random.random((n, d))
        Y = np.random.random((n, d))
        K = np.zeros((n, n, d))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i, j] = (x - y) / (1 + np.linalg.norm(x - y) ** 2) ** (3 / 2)
        tst = grad_pc_imq_y(X, Y, bandwidth)
        self.assertAlmostEqual(jnp.linalg.norm(K - tst), 0.0, places=3)

    def test_rbf_grad_log_f_x(self) -> None:
        r"""
        Test the score function of an RBF
        """
        bandwidth = 1 / np.sqrt(2)
        n = 10
        d = 2
        X = np.random.random((n, d))
        kde_points = np.random.random((n, d))
        kde = lambda x: (
            np.exp(
                -np.linalg.norm(x - kde_points, axis=1)[:, None] ** 2
                / (2 * bandwidth**2)
            )
            / (np.sqrt(2 * np.pi) * bandwidth)
        ).mean(axis=0)
        J = np.zeros((n, d))
        for i, x in enumerate(X):
            J[i] = (
                -(x - kde_points)
                / bandwidth**3
                * np.exp(
                    -np.linalg.norm(x - kde_points, axis=1)[:, None] ** 2
                    / (2 * bandwidth**2)
                )
                / (np.sqrt(2 * np.pi))
            ).mean(axis=0) / (kde(x)[:, None])
        tst = rbf_grad_log_f_x(X, kde_points, bandwidth)
        self.assertAlmostEqual(jnp.linalg.norm(J - tst), 0.0, places=3)

    def test_stein_kernel_pc_imq_element(self) -> None:
        """
        Test the Stein kernel with PC-IMQ base and score fn -x
        """
        n = 10
        d = 2
        bandwidth = 1 / np.sqrt(2)
        score_fn = lambda x: -x
        beta = 0.5

        def k_x_y(x, y):
            norm_sq = np.linalg.norm(x - y) ** 2
            l = -3 * norm_sq / (1 + norm_sq) ** 2.5
            m = (
                2
                * beta
                * (d + np.dot(score_fn(x) - score_fn(y), x - y))
                / (1 + norm_sq) ** 1.5
            )
            r = np.dot(score_fn(x), score_fn(y)) / (1 + norm_sq) ** 0.5
            return l + m + r

        X = np.random.random((n, d))
        Y = np.random.random((n, d))
        K = np.zeros((n, n))
        K_ans = np.zeros((n, n))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K_ans[i, j] = k_x_y(x, y)
                K[i, j] = stein_kernel_pc_imq_element(
                    x, y, score_fn(x), score_fn(y), d, bandwidth
                )
        self.assertAlmostEqual(jnp.linalg.norm(K - K_ans), 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
