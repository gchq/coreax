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

        :n: Number of test data points
        :d: Dimension of data
        :m: Number of points to randomly select for second dataset Y
        :max_size: Maximum number of points for block calculations
        """
        # Define data parameters
        self.num_points_in_data = 30
        self.dimension = 10
        self.random_data_generation_key = 0
        self.num_points_in_coreset = 5
        self.random_sampling_key = 0

        # Define example dataset
        x = random.uniform(
            random.PRNGKey(self.random_data_generation_key),
            shape=(self.num_points_in_data, self.dimension),
        )

        data_obj = DataReader(
            original_data=x, pre_reduction_array=[], reduction_indices=[]
        )
        data_obj.pre_reduction_array = x

        self.data_obj = data_obj

    def test_random_sample(self) -> None:
        """
        Test data reduction by uniform-randomly sampling a fixed number of points.

        TODO:
        Test random sampling for known seed behaviour.
        Test unique flag True vs False.
        ...
        """

        random_sample = coreax.coreset.RandomSample(
            data=self.data_obj,
            coreset_size=self.num_points_in_coreset,
            random_key=self.random_sampling_key,
        )

        random_sample.fit()

        self.assertEqual(
            len(self.data_obj.reduction_indices), self.num_points_in_coreset
        )
