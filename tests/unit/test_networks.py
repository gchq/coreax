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
Tests for network implementations.

The tests within this file verify that the network classes and functionality throughout
the codebase produce the expected results on simple examples.
"""

import unittest

import jax.random as jr
import optax

import coreax.networks
import coreax.util


class TestTrainState(unittest.TestCase):
    """
    Tests related to :func:`create_train_state`.

    .. note::

        Learning rate is not tested as it is part of weight updates, not initialisation
        of the code written in this codebase.
    """

    def setUp(self):
        """
        Generate data for use across unit tests.
        """
        self.hidden_dimensions = [2, 2, 2]
        self.data_dimension = 5
        self.state_key = jr.key(1989)
        self.learning_rate = 0.1
        self.optimiser = optax.adamw

    def test_create_train_state_negative_hidden_dimension(self):
        """
        Test create_train_state with a negative hidden dimension size.
        """
        score_network = coreax.networks.ScoreNetwork([2, -2, 2], self.data_dimension)

        # Create a train state with this network - we expect optax to catch the invalid
        # hidden dimension argument and raise an appropriate error
        with self.assertRaisesRegex(TypeError, "Only non-negative indices"):
            coreax.networks.create_train_state(
                random_key=self.state_key,
                module=score_network,
                learning_rate=self.learning_rate,
                data_dimension=self.data_dimension,
                optimiser=self.optimiser,
            )

    def test_create_train_state_zero_hidden_dimension(self):
        """
        Test create_train_state with a zero size hidden dimension.
        """
        score_network = coreax.networks.ScoreNetwork([2, 0, 2], self.data_dimension)

        # Create a train state with this network - we expect optax to try and do a
        # division by hidden dimension size - which should give rise to a division error
        with self.assertRaisesRegex(ZeroDivisionError, "division by zero"):
            coreax.networks.create_train_state(
                random_key=self.state_key,
                module=score_network,
                learning_rate=self.learning_rate,
                data_dimension=self.data_dimension,
                optimiser=self.optimiser,
            )

    def test_create_train_state_float_hidden_dimension(self):
        """
        Test create_train_state with a float valued hidden dimension.
        """
        score_network = coreax.networks.ScoreNetwork([2, 2.0, 2], self.data_dimension)

        # Create a train state with this network - we expect optax to catch the invalid
        # hidden dimension argument and raise an appropriate error
        with self.assertRaisesRegex(TypeError, "Shapes must be"):
            coreax.networks.create_train_state(
                random_key=self.state_key,
                module=score_network,
                learning_rate=self.learning_rate,
                data_dimension=self.data_dimension,
                optimiser=self.optimiser,
            )

    def test_create_train_state_negative_output_dimension(self):
        """
        Test create_train_state with a negative output dimension size.
        """
        score_network = coreax.networks.ScoreNetwork(
            self.hidden_dimensions, -self.data_dimension
        )

        # Create a train state with this network - we expect optax to catch the invalid
        # data dimension argument and raise an appropriate error. We don't check the
        # exact message as it uses a word (non-negative without the dash) that would
        # mean adding an incorrect work to cspell.
        with self.assertRaises(TypeError):
            coreax.networks.create_train_state(
                random_key=self.state_key,
                module=score_network,
                learning_rate=self.learning_rate,
                data_dimension=-self.data_dimension,
                optimiser=self.optimiser,
            )

    def test_create_train_state_zero_output_dimension(self):
        """
        Test create_train_state with a zero size output dimension.
        """
        score_network = coreax.networks.ScoreNetwork(self.hidden_dimensions, 0)

        # Create a train state with this network - we expect optax to try and do a
        # division by data dimension size - which should give rise to a division error
        with self.assertRaisesRegex(ZeroDivisionError, "^division by zero$"):
            coreax.networks.create_train_state(
                random_key=self.state_key,
                module=score_network,
                learning_rate=self.learning_rate,
                data_dimension=0,
                optimiser=self.optimiser,
            )

    def test_create_train_state_float_output_dimension(self):
        """
        Test create_train_state with a float valued output dimension.
        """
        score_network = coreax.networks.ScoreNetwork(
            self.hidden_dimensions,
            1.0 * self.data_dimension,  # pyright: ignore[reportArgumentType]
        )

        # Create a train state with this network - we expect optax to catch the invalid
        # data dimension argument and raise an appropriate error
        # pylint: disable=line-too-long
        with self.assertRaisesRegex(TypeError, "Shapes must be"):
            coreax.networks.create_train_state(
                random_key=self.state_key,
                module=score_network,
                learning_rate=self.learning_rate,
                data_dimension=1.0 * self.data_dimension,  # pyright: ignore[reportArgumentType]
                optimiser=self.optimiser,
            )
        # pylint: enable=line-too-long

    def test_create_train_state_invalid_module(self):
        """
        Test create_train_state with an invalid network module.
        """
        score_network = coreax.util.InvalidKernel

        # Create a train state with this network - we expect optax to catch the invalid
        # module argument and raise an appropriate error
        with self.assertRaisesRegex(
            AttributeError, "type object 'InvalidKernel' has no attribute"
        ):
            coreax.networks.create_train_state(
                random_key=self.state_key,
                module=score_network,  # pyright: ignore[reportArgumentType]
                learning_rate=self.learning_rate,
                data_dimension=self.data_dimension,
                optimiser=self.optimiser,
            )
