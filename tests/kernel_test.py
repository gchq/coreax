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
from scipy.stats import ortho_group
import numpy as np

from coreax.kernel import *

class TestKernels(unittest.TestCase):

    def test_sq_dist(self):
        """Test square distance under float32
        """
        d = 2
        m = ortho_group.rvs(dim=2)
        x = m[0]
        y = m[1]
        d = jnp.linalg.norm(x - y)**2
        td = sq_dist(x, y)
        self.assertAlmostEqual(d, td, places=3)
        td = sq_dist(x, x)
        self.assertAlmostEqual(0., td, places=3)
        td = sq_dist(y, y)
        self.assertAlmostEqual(0., td, places=3)

    def test_sq_dist_pairwise(self):
        """Test vmap version of sq distance
        """
        # create an orthonormal matrix
        d = 3
        m = ortho_group.rvs(dim=d)
        tinner = sq_dist_pairwise(m, m)
        ans = np.ones((d, d)) * 2.
        np.fill_diagonal(ans, 0.)
        # Frobenius norm
        td = jnp.linalg.norm(tinner - ans)
        self.assertAlmostEqual(td, 0., places=3)

    def test_rbf_kernel(self):
        pass

if __name__ == "__main__":
    unittest.main()