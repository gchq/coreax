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

        # Generate n random points in d dimensions from a uniform distribution [0, 1)
        x = np.random.rand(n, d)
        # Randomly select m points
        sel = np.random.choice(x.shape[0], size=m, replace=False)
        x_c = x[sel]

        def gaussian_kernel(A, B, Var):
            """Define Gaussian kernel"""
            M = A.shape[0]
            N = B.shape[0]

            A_dots = (A * A).sum(axis=1).reshape((M, 1)) * np.ones(shape=(1, N))
            B_dots = (B * B).sum(axis=1) * np.ones(shape=(M, 1))
            dist_squared = A_dots + B_dots - 2 * A.dot(B.T)

            g_kern = np.exp(-dist_squared / (2 * Var))
            return g_kern

        var = 1

        mmd_ = (
            np.mean(gaussian_kernel(x, x, var))
            + np.mean(gaussian_kernel(x_c, x_c, var))
            - 2 * np.mean(gaussian_kernel(x, x_c, var))
        ) ** 0.5

        mmd_test = mmd(x, x_c, rbf_kernel)
        self.assertAlmostEqual(mmd_test, mmd_, places=3)

        t_mmd = mmd(x, x, rbf_kernel)
        self.assertAlmostEqual(np.zeros(1), t_mmd, places=3)


if __name__ == "__main__":
    unittest.main()
