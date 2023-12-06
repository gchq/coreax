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
import numpy as np
from jax import random

import coreax.coreset
from coreax.data import DataReader


class TestRandomSample(unittest.TestCase):
    """
    Tests related to RandomSample class in coreset.py.
    """

    def setUp(self):
        r"""
        Generate data for use across unit tests.

        Generate n random points in d dimensions from a uniform distribution [0, 1).

        ``n``: Number of test data points
        ``d``: Dimension of data
        ``m``: Number of points to randomly select for second dataset Y
        ``max_size``: Maximum number of points for block calculations
        """
        # Define data parameters
        self.num_points_in_data = 30
        self.dimension = 10
        self.random_data_generation_key = 0
        self.num_points_in_coreset = 10
        self.random_sampling_key = 42

        # Define example dataset
        x = random.uniform(
            random.PRNGKey(self.random_data_generation_key),
            shape=(self.num_points_in_data, self.dimension),
        )

        data_obj = DataReader(original_data=x, pre_reduction_array=x)

        self.data_obj = data_obj

    def test_random_sample(self) -> None:
        """Test data reduction by uniform-randomly sampling a fixed number of points."""
        random_sample = coreax.coreset.RandomSample(
            data=self.data_obj,
            coreset_size=self.num_points_in_coreset,
            random_key=self.random_sampling_key,
        )
        random_sample.fit()

        # Assert the number of indices in the reduced data is as expected
        self.assertEqual(
            len(random_sample.reduction_indices), self.num_points_in_coreset
        )

        # Convert lists to set of tuples
        coreset_set = set(map(tuple, np.array(random_sample.coreset)))
        orig_data_set = set(
            map(tuple, np.array(random_sample.data.pre_reduction_array))
        )
        # Find common rows
        num_common_rows = len(coreset_set & orig_data_set)
        # Assert all rows in the coreset are in the original dataset
        self.assertEqual(len(coreset_set), num_common_rows)

    def test_random_sample_with_replacement(self) -> None:
        """
        Test reduction of datasets by uniform random sampling with replacement.

        For the purposes of this test, the random sampling behaviour is known for the
         seeds in setUp(). The parameters self.num_points_in_coreset = 10 and
        self.random_sampling_key = 42 ensure a repeated coreset point when unique=False.
        """
        random_sample = coreax.coreset.RandomSample(
            data=self.data_obj,
            coreset_size=self.num_points_in_coreset,
            random_key=self.random_sampling_key,
            unique=False,
        )
        random_sample.fit()

        unique_reduction_indices = jnp.unique(random_sample.reduction_indices)
        self.assertTrue(
            len(unique_reduction_indices) < len(random_sample.reduction_indices)
        )
