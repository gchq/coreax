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
Classes for reading unsupervised and supervised input data.

In order to calculate a coreset, :meth:`~coreax.reduction.Coreset.fit` requires an
instance of a subclass of :class:`WeightedData`. It is necessary to use
:class:`WeightedData` because :class:`~coreax.reduction.Coreset` requires a
two-dimensional :class:`~jax.Array`. Data reductions are performed along the first
dimension.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Shaped


# pylint: disable=too-few-public-methods
class WeightedData(eqx.Module):
    """
    Class to apply pre-processing to unsupervised data.

    :param data: Array of data to be reduced to a coreset
    :param weights: Array of weight corresponding to data points
    """

    data: Shaped[Array, " n d"] = eqx.field(converter=jnp.atleast_2d)
    weights: Shaped[Array, " n"] = eqx.field(converter=jnp.atleast_1d)

    def __init__(
        self, data: Shaped[Array, " n d"], weights: Shaped[Array, " n"] | None = None
    ):
        """Initialise WeightedData class."""
        self.data = data
        if weights is None:
            n = data.shape[0]
            self.weights = jnp.broadcast_to(1 / n, (n,))
        else:
            self.weights = weights

    def __check_init__(self):
        """Check for valid __init__ inputs."""
        if self.weights.shape[0] != self.data.shape[0]:
            raise ValueError("Leading dimensions of `weights` and `data` must be equal")


class SupervisedWeightedData(WeightedData):
    """
    Class to apply pre-processing to supervised data.

    :param data: Array of data to be reduced to a coreset
    :param supervision: Array of supervision corresponding to data
    :param weights: Array of weight corresponding to data pairs
    """

    supervision: Shaped[Array, " n *p"] = eqx.field(converter=jnp.atleast_2d)

    def __init__(
        self,
        data: Shaped[Array, " n d"],
        supervision: Shaped[Array, " n *p"],
        weights: Shaped[Array, " n"] | None = None,
    ):
        """Initialise SupervisedWeightedData class."""
        self.supervision = supervision
        super().__init__(data, weights)

    def __check_init__(self):
        """Check for valid __init__ inputs."""
        if self.supervision.shape[0] != self.data.shape[0]:
            raise ValueError(
                "Leading dimensions of `supervision` and `data` must be equal"
            )


# pylint: enable=too-few-public-methods
