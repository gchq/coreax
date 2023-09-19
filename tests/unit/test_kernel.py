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
from scipy.stats import norm, ortho_group

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


if __name__ == "__main__":
    unittest.main()
