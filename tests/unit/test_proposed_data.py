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

"""
Tests for data classes and functionality.

The tests within this file verify that approaches to handling and processing data
produce the expected results on simple examples.
"""

import unittest

import jax.numpy as jnp

from coreax.proposed_data import SupervisedWeightedData, WeightedData


class TestWeightedData(unittest.TestCase):
    """Test operation of WeightedData class."""

    def test_uniform_weights(self):
        """Test that if no weights are given a uniform weight vector is made."""
        original_data = jnp.array([1, 2, 3])
        n = original_data.shape[0]

        data_class = WeightedData(data=original_data, weights=None)
        self.assertAlmostEqual(
            float(jnp.linalg.norm(data_class.weights - jnp.broadcast_to(1 / n, (n,)))),
            0.0,
        )

    def test_invalid_data_and_weight_dimensions(self):
        """
        Test that __check_init__ raises expected errors.
        """
        original_data = jnp.array([1, 2, 3])

        with self.assertRaises(ValueError) as error_raised:
            WeightedData(
                data=original_data, weights=jnp.ones(original_data.shape[0] + 1)
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "Leading dimensions of `weights` and `data` must be equal",
        )


class TestSupervisedWeightedData(unittest.TestCase):
    """Test operation of SupervisedWeightedData class."""

    def test_uniform_weights(self):
        """Test that if no weights are given a uniform weight vector is made."""
        original_data = jnp.array([1, 2, 3])
        original_supervision = jnp.array([4, 5, 6])
        n = original_data.shape[0]

        data_class = SupervisedWeightedData(
            data=original_data, supervision=original_supervision, weights=None
        )
        self.assertAlmostEqual(
            float(jnp.linalg.norm(data_class.weights - jnp.broadcast_to(1 / n, (n,)))),
            0.0,
        )

    def test_invalid_data_and_weight_dimensions(self):
        """
        Test that __check_init__ raises ValueError when weights array has bad dimension.
        """
        original_data = jnp.array([1, 2, 3])
        original_supervision = jnp.array([4, 5, 6])

        with self.assertRaises(ValueError) as error_raised:
            SupervisedWeightedData(
                data=original_data,
                supervision=original_supervision,
                weights=jnp.ones(original_data.shape[0] + 1),
            )
        self.assertEqual(
            error_raised.exception.args[0],
            "Leading dimensions of `weights` and `data` must be equal",
        )

    def test_invalid_data_and_supervision_dimensions(self):
        """
        Test that __check_init__ raises ValueError when supervision has bad dimension.
        """
        original_data = jnp.array([1, 2, 3])
        original_supervision = jnp.array([4, 5])

        with self.assertRaises(ValueError) as error_raised:
            SupervisedWeightedData(data=original_data, supervision=original_supervision)
        self.assertEqual(
            error_raised.exception.args[0],
            "Leading dimensions of `supervision` and `data` must be equal",
        )
