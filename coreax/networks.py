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
Classes and associated functionality to define neural networks.

Neural networks are used throughout the codebase as functional approximators.
"""

from collections.abc import Callable, Sequence

import jax.numpy as jnp
from flax import linen as nn
from flax.linen import Module
from flax.training import train_state
from jax import Array
from jaxtyping import Shaped
from optax import GradientTransformation

from coreax.util import KeyArrayLike

_LearningRateOptimiser = Callable[[float], GradientTransformation]


class ScoreNetwork(nn.Module):
    """
    A feed-forward neural network for use in sliced score matching.

    See :class:`~coreax.score_matching.SlicedScoreMatching` for an example usage of this
    class.

    :param hidden_dims: Sequence of hidden dimension layer sizes. Each element of the
        sequence corresponds to one hidden layer.
    :param output_dim: Number of output layer nodes.
    """

    hidden_dims: Sequence
    output_dim: int

    @nn.compact
    def __call__(self, x: Shaped[Array, " b n d"]) -> Shaped[Array, " b output_dim"]:
        r"""
        Compute forward pass through a three-layer network with softplus activations.

        :param x: Batch input data :math:`b \times n \times d`
        :return: Network output on batch :math:`b \times` ``self.output_dim``
        """
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.softplus(x)

        return nn.Dense(self.output_dim)(x)


def create_train_state(
    random_key: KeyArrayLike,
    module: Module,
    learning_rate: float,
    data_dimension: int,
    optimiser: _LearningRateOptimiser,
) -> train_state.TrainState:
    """
    Create a flax :class:`~flax.training.train_state.TrainState` for learning with.

    :param random_key: Key for random number generation
    :param module: Subclass of :class:`~flax.linen.Module`
    :param learning_rate: Optimiser learning rate
    :param data_dimension: Data dimension
    :param optimiser: optax optimiser, e.g. :func:`~optax.adam`
    :return: :class:`~flax.training.train_state.TrainState` object
    """
    params = module.init(random_key, jnp.ones((1, data_dimension)))["params"]
    tx = optimiser(learning_rate)
    return train_state.TrainState.create(apply_fn=module.apply, params=params, tx=tx)
