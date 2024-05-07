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
instance of a subclass of :class:`Data`. It is necessary to use
:class:`Data` because :class:`~coreax.reduction.Coreset` requires a
two-dimensional :class:`~jax.Array`. Data reductions are performed along the first
dimension.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Shaped


# pylint: disable=too-few-public-methods
class Data(eqx.Module):
    r"""
    Class for representing unsupervised data.

    :param data: An :math:`n \times d` array defining the unsupervised dataset
    :param weights: An :math:`n`-vector of weights with. Each element of the weights
        vector is associated with the data point at the corresponding index of the
        data array.
    """

    data: Shaped[Array, " n d"]
    weights: Shaped[Array, " n"]

    def __init__(
        self, data: Shaped[Array, " n d"], weights: Shaped[Array, " n"] | None = None
    ):
        """Initialise Data class."""
        self.data = data
        if weights is None:
            n = data.shape[0]
            self.weights = jnp.broadcast_to(1 / n, (n,))
        else:
            self.weights = weights

    def __check_init__(self):
        """Check leading dimensions of weights and data match."""
        if self.weights.shape[0] != self.data.shape[0]:
            raise ValueError("Leading dimensions of 'weights' and 'data' must be equal")


class SupervisedData(Data):
    r"""
    Class for representing supervised data.

    :param data: An :math:`n \times d` array defining the features of the supervised
        dataset paired with the responses.
    :param supervision: An :math:`n \times p` array defining the responses of the
        supervised dataset paired with the features.
    :param weights: An :math:`n`-vector of weights with. Each element of the weights
        vector is associated with the data pair at the corresponding index of the
        data and supervision arrays.
    """

    supervision: Shaped[Array, " n *p"]

    def __init__(
        self,
        data: Shaped[Array, " n d"],
        supervision: Shaped[Array, " n *p"],
        weights: Shaped[Array, " n"] | None = None,
    ):
        """Initialise SupervisedData class."""
        self.supervision = supervision
        super().__init__(data, weights)

    def __check_init__(self):
        """Check leading dimensions of supervision and data match."""
        if self.supervision.shape[0] != self.data.shape[0]:
            raise ValueError(
                "Leading dimensions of 'supervision' and 'data' must be equal"
            )


# pylint: enable=too-few-public-methods
