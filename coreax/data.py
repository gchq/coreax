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

"""Data-structures for representing weighted and/or supervised data."""

from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike, Shaped
from typing_extensions import Self


class Data(eqx.Module):
    r"""
    Class for representing unsupervised data.

    A dataset of size `n` consists of a set of pairs :math:`\{(x_i, w_i)\}_{i=1}^n`
    where :math`x_i` are the features or inputs and :math:`w_i` are weights.

    :param data: An :math:`n \times d` array defining the features of the unsupervised
        dataset
    :param weights: An :math:`n`-vector of weights where each element of the weights
        vector is paired with the corresponding index of the data array, forming the
        pair :math:`(x_i, w_i)`; if passed a scalar weight, it will be broadcast to an
        :math:`n`-vector. the default value of :data:`None` sets the weights to
        the ones vector (implies a scalar weight of one);
    """

    data: Shaped[Array, " n *d"]
    weights: Shaped[Array, " n"]

    def __init__(
        self,
        data: Shaped[ArrayLike, " n *d"],
        weights: Optional[Shaped[ArrayLike, " n"]] = None,
    ):
        """Initialise Data class."""
        self.data = jnp.asarray(data)
        n = self.data.shape[:1]
        self.weights = jnp.broadcast_to(1 if weights is None else weights, n)

    def __getitem__(self, key) -> Self:
        """Support Array style indexing of 'Data' objects."""
        return jtu.tree_map(lambda x: x[key], self)

    def __jax_array__(self) -> Shaped[ArrayLike, " n d"]:
        """Register ArrayLike behaviour - return value for `jnp.asarray(Data(...))`."""
        return self.data

    def __len__(self) -> int:
        """Return data length."""
        return len(self.data)

    def normalize(self, *, preserve_zeros: bool = False) -> Self:
        """
        Return a copy of 'self' with 'weights' that sum to one.

        :param preserve_zeros: If to preserve zero valued weights; when all weights are
            zero valued, the 'normalized' copy will **sum to zero, not one**.
        :return: A copy of 'self' with normalized 'weights'
        """
        normalized_weights = self.weights / jnp.sum(self.weights)
        if preserve_zeros:
            normalized_weights = jnp.nan_to_num(normalized_weights)
        return eqx.tree_at(lambda x: x.weights, self, normalized_weights)


class SupervisedData(Data):
    r"""
    Class for representing supervised data.

    A supervised dataset of size `n` consists of a set of triples
    :math:`\{(x_i, y_i, w_i)\}_{i=1}^n` where :math`x_i` are the features or inputs,
    :math:`y_i` are the responses or outputs, and :math:`w_i` are weights which
    correspond to the pairs :math:`(x_i, y_i)`.

    :param data: An :math:`n \times d` array defining the features of the supervised
        dataset paired with the corresponding index of the supervision
    :param supervision: An :math:`n \times p` array defining the responses of the
        supervised dataset paired with the corresponding index of the data
    :param weights: An :math:`n`-vector of weights where each element of the weights
        vector is is paired with the corresponding index of the data and supervision
        array, forming the triple :math:`(x_i, y_i, w_i)`; if passed a scalar weight,
        it will be broadcast to an :math:`n`-vector. the default value of :data:`None`
        sets the weights to the ones vector (implies a scalar weight of one);
    """

    supervision: Shaped[Array, " n *p"]

    def __init__(
        self,
        data: Shaped[Array, " n d"],
        supervision: Shaped[Array, " n *p"],
        weights: Optional[Shaped[Array, " n"]] = None,
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


def as_data(x: Any) -> Data:
    """Cast 'x' to a data instance."""
    return x if isinstance(x, Data) else Data(x)


def is_data(x: Any) -> bool:
    """Return boolean indicating if 'x' is an instance of 'coreax.data.Data'."""
    return isinstance(x, Data)
