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

"""TODO: Create top-level docstring."""

from collections.abc import Callable

from flax import linen as nn
from flax.linen import Module
from flax.training.train_state import TrainState
from jax import numpy as jnp
from jax.random import PRNGKey
from jax.typing import ArrayLike


class ScoreNetwork(nn.Module):
    """A feed-forward neural network for use in sliced score matching."""

    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x: ArrayLike) -> ArrayLike:
        r"""
        Forward pass through a three-layer network with softplus activations.

        :param x: Batch input data :math:`b \times m \times n`
        :return: Network output on batch :math:`b \times` self.output_dim
        """
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.softplus(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.softplus(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.softplus(x)
        x = nn.Dense(self.output_dim)(x)
        return x


def create_train_state(
    module: Module,
    rng: PRNGKey,
    learning_rate: float,
    dimension: int,
    optimiser: Callable,
) -> TrainState:
    """
    Create a flax :class:`~flax.training.train_state.TrainState` for learning with.

    :param module: Subclass of :class:`~flax.nn.Module`
    :param rng: Random number generator
    :param learning_rate: Optimiser learning rate
    :param dimension: Data dimension
    :param optimiser: optax optimiser, e.g. :class:`~optax.adam`
    :return: :class:`~flax.training.train_state.TrainState` object
    """
    params = module.init(rng, jnp.ones((1, dimension)))["params"]
    tx = optimiser(learning_rate)
    return TrainState.create(apply_fn=module.apply, params=params, tx=tx)
