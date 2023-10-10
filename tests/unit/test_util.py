# © Crown Copyright GCHQ
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import unittest

import jax.numpy as jnp
import numpy as np
from scipy.stats import ortho_group

from coreax.util import pdiff, sq_dist, sq_dist_pairwise


class Test(unittest.TestCase):
    def test_sq_dist(self) -> None:
        """
        Test square distance under float32.
        """
        x, y = ortho_group.rvs(dim=2)
        d = jnp.linalg.norm(x - y) ** 2
        td = sq_dist(x, y)
        self.assertAlmostEqual(td, d, places=3)
        td = sq_dist(x, x)
        self.assertAlmostEqual(td, 0.0, places=3)
        td = sq_dist(y, y)
        self.assertAlmostEqual(td, 0.0, places=3)

    def test_sq_dist_pairwise(self) -> None:
        """
        Test vmap version of sq distance.
        """
        # create an orthonormal matrix
        d = 3
        m = ortho_group.rvs(dim=d)
        tinner = sq_dist_pairwise(m, m)
        # Use original numpy because Jax arrays are immutable
        ans = np.ones((d, d)) * 2.0
        np.fill_diagonal(ans, 0.0)
        # Frobenius norm
        td = jnp.linalg.norm(tinner - ans)
        self.assertEqual(td.ndim, 0)
        self.assertAlmostEqual(float(td), 0.0, places=3)

    def test_pdiff(self) -> None:
        """
        Test the function pdiff.

        This test ensures efficient computation of pairwise differences.
        """
        m = 10
        n = 10
        d = 3
        x_array = np.random.random((n, d))
        y_array = np.random.random((m, d))
        z_array = np.array([[x - y for y in y_array] for x in x_array])
        tst = pdiff(x_array, y_array)
        self.assertAlmostEqual(float(jnp.linalg.norm(tst - z_array)), 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
