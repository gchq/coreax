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

import coreax.util


class TestUtil(unittest.TestCase):
    """
    Tests for general utility functions.
    """

    def test_squared_distance_dist(self) -> None:
        """
        Test square distance under float32.
        """
        x, y = ortho_group.rvs(dim=2)
        expected_distance = jnp.linalg.norm(x - y) ** 2
        output_distance = coreax.util.squared_distance(x, y)
        self.assertAlmostEqual(output_distance, expected_distance, places=3)
        output_distance = coreax.util.squared_distance(x, x)
        self.assertAlmostEqual(output_distance, 0.0, places=3)
        output_distance = coreax.util.squared_distance(y, y)
        self.assertAlmostEqual(output_distance, 0.0, places=3)

    def test_squared_distance_dist_pairwise(self) -> None:
        """
        Test vmap version of sq distance.
        """
        # create an orthonormal matrix
        dimension = 3
        orthonormal_matrix = ortho_group.rvs(dim=dimension)
        inner_distance = coreax.util.squared_distance_pairwise(
            orthonormal_matrix, orthonormal_matrix
        )
        # Use original numpy because Jax arrays are immutable
        expected_output = np.ones((dimension, dimension)) * 2.0
        np.fill_diagonal(expected_output, 0.0)
        # Frobenius norm
        difference_in_distances = jnp.linalg.norm(inner_distance - expected_output)
        self.assertEqual(difference_in_distances.ndim, 0)
        self.assertAlmostEqual(float(difference_in_distances), 0.0, places=3)

    def test_pairwise_difference(self) -> None:
        """
        Test the function pairwise_difference.

        This test ensures efficient computation of pairwise differences.
        """
        num_points_x = 10
        num_points_y = 10
        dimension = 3
        x_array = np.random.random((num_points_x, dimension))
        y_array = np.random.random((num_points_y, dimension))
        expected_output = np.array([[x - y for y in y_array] for x in x_array])
        output = coreax.util.pairwise_difference(x_array, y_array)
        self.assertAlmostEqual(
            float(jnp.linalg.norm(output - expected_output)), 0.0, places=3
        )


if __name__ == "__main__":
    unittest.main()
