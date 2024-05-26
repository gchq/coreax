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
Classes for reading different structures of input data.

In order to calculate a coreset, :meth:`~coreax.reduction.Coreset.fit` requires an
instance of a subclass of :class:`DataReader`. It is necessary to use
:class:`DataReader` because :class:`~coreax.reduction.Coreset` requires a
two-dimensional :class:`~jax.Array`. Data reductions are performed along the first
dimension.

The user should read in their data files using their preferred library that returns a
:class:`jax.Array` or :func:`numpy.array`. This array is passed to a
:meth:`load() <DataReader.load>` method. The user should not normally invoke
:class:`DataReader` directly. The user should select an appropriate subclass
of :class:`DataReader` to match the structure of the input array. The
:meth:`load() <DataReader.load>` method on the subclass will rearrange the original data
into the required two-dimensional format.

Various post-processing methods may be implemented if applicable to visualise or
restore a calculated coreset to match the format of the original data. To save a
copy of a coreset, call :meth:`format() <DataReader.format>` on a subclass to return an
:class:`~jax.Array`, which can be passed to the chosen IO library to write a file.
"""

# Support annotations with | in Python < 3.10
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Shaped

if TYPE_CHECKING:
    import coreax.reduction


class DataReader(ABC):
    """
    Class to apply pre- and post-processing to data.

    :param original_data: Array of data to be reduced to a coreset
    :param pre_coreset_array: Two-dimensional array already rearranged to be ready for
        calculating a coreset
    """

    def __init__(self, original_data: ArrayLike, pre_coreset_array: ArrayLike):
        """
        Initialise class.

        Should not normally be called by the user: use :meth:`load` instead.
        """
        self.original_data: Array = jnp.atleast_2d(original_data)
        """
        Original data
        """
        self.pre_coreset_array: Array = jnp.atleast_2d(pre_coreset_array)
        """
        Pre coreset array
        """

    @classmethod
    @abstractmethod
    def load(cls, original_data: ArrayLike) -> DataReader:
        """
        Construct :class:`DataReader` from an array of original data.

        This class method restructures the original data into the layout required for
        :class:`~coreax.reduction.Coreset`.

        The user should not normally initialise this class directly; instead use this
        constructor.

        :param original_data: Array of data to be reduced to a coreset
        :return: Populated instance of :class:`DataReader`
        """

    @abstractmethod
    def format(self, coreset: coreax.reduction.Coreset) -> Array:
        """
        Format coreset to match the shape of the original data.

        If the number of columns was reduced by :meth:`reduce_dimension`, it will be
        reverted by this method via a call to :meth:`restore_dimension`.

        :param coreset: Coreset to format
        :return: Array of coreset in format matching original data
        """

    def render(self, coreset: coreax.reduction.Coreset | None) -> None:
        """
        Plot coreset or original data interactively using :mod:`~matplotlib.pyplot`.

        This method is only implemented when applicable for the data type.

        :param coreset: Coreset to plot, or :data:`None` to plot original data
        """
        raise NotImplementedError

    def reduce_dimension(self, num_dimensions: int) -> None:
        """
        Reduce dimensionality of :attr:`~coreax.data.DataReader.pre_coreset_array`.

        Performed using principal component analysis (PCA).

        :attr:`~coreax.data.DataReader.pre_coreset_array` is updated in place. Metadata
        detailing the type of reduction are saved to this class to enable reconstruction
        later.

        :param num_dimensions: Target number of dimensions
        """
        raise NotImplementedError

    def restore_dimension(self, coreset: coreax.reduction.Coreset | None) -> Array:
        """
        Expand principal components into original number of columns in two dimensions.

        Some data will have been lost due to reduction in dimensionality, so the
        restored data will not exactly match the original data.

        Call :meth:`format` instead to restore to multiple dimensions if the original
        data format had more than two dimensions.

        :param coreset: Coreset to restore, or :data:`None` to restore original data
        :return: Array with the same number of columns as
            :attr:`~coreax.data.DataReader.pre_coreset_array` had prior to calling
            :meth:`reduce_dimension` and the same number of rows as the coreset
        """
        raise NotImplementedError


class ArrayData(DataReader):
    """
    Class to apply pre- and post-processing to two-dimensional array data.

    Data should already be in a format accepted by :class:`~coreax.reduction.Coreset`.
    Thus, if no dimensionality reduction is performed, this class is an identity
    wrapper and :attr:`~coreax.data.DataReader.pre_coreset_array` is equal to
    :attr:`~coreax.data.DataReader.original_data`.

    :param original_data: Array of data to be reduced to a coreset
    :param pre_coreset_array: Two-dimensional array already rearranged to be ready for
        calculating a coreset
    """

    @classmethod
    def load(cls, original_data: ArrayLike) -> ArrayData:
        """
        Construct :class:`ArrayData` from a two-dimensional array of data.

        This constructor does not modify the provided data.

        The user should not normally initialise this class directly; instead use this
        constructor.

        :param original_data: Array of data to be reduced to a coreset
        :return: Populated instance of :class:`ArrayData`
        """
        original_data = jnp.atleast_2d(original_data)
        return cls(original_data, original_data)

    def format(self, coreset: coreax.reduction.Coreset) -> Array:
        """
        Format coreset to match the shape of the original data.

        As the original data was already in the required format for
        :class:`~coreax.reduction.Coreset`, no reformatting takes place.

        If the number of columns was reduced by
        :meth:`~coreax.data.DataReader.reduce_dimension`, it will be reverted by this
        method via a call to :meth:`~coreax.data.DataReader.restore_dimension`.

        :param coreset: Coreset to format
        :return: Array of coreset in format matching original data
        """
        return coreset.coreset


class Data(eqx.Module):
    r"""
    Class for representing unsupervised data.

    A dataset of size `n` consists of a set of pairs :math:`\{(x_i, w_i)\}_{i=1}^n`
    where :math`x_i` are the features or inputs and :math:`w_i` are weights.

    :param data: An :math:`n \times d` array defining the features of the unsupervised
        dataset; d-vectors are converted to :math:`1 \times d` arrays
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
        weights: Shaped[ArrayLike, " n"] | None = None,
    ):
        """Initialise Data class."""
        self.data = jnp.asarray(data)
        n = self.data.shape[:1]
        self.weights = jnp.broadcast_to(1 if weights is None else weights, n)

    def __jax_array__(self) -> Shaped[ArrayLike, " n d"]:
        """Register ArrayLike behaviour - return value for `jnp.asarray(Data(...))`."""
        return self.data

    def __len__(self) -> int:
        """Return data length."""
        return len(self.data)

    def normalize(self, *, preserve_zeros: bool = False) -> Data:
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


def as_data(x: Any) -> Data:
    """Cast 'x' to a data instance."""
    return x if isinstance(x, Data) else Data(x)


def is_data(x: Any) -> bool:
    """Return boolean indicating if 'x' is an instance of 'coreax.data.Data'."""
    return isinstance(x, Data)


class SupervisedData(Data):
    r"""
    Class for representing supervised data.

    A supervised dataset of size `n` consists of a set of triples
    :math:`\{(x_i, y_i, w_i)\}_{i=1}^n` where :math`x_i` are the features or inputs,
    :math:`y_i` are the responses or outputs, and :math:`w_i` are weights which
    correspond to the pairs :math:`(x_i, y_i)`.

    :param data: An :math:`n \times d` array defining the features of the supervised
        dataset paired with the corresponding index of the supervision;  d-vectors are
        converted to :math:`1 \times d` arrays
    :param supervision: An :math:`n \times p` array defining the responses of the
        supervised paired with the corresponding index of the data; d-vectors are
        converted to :math:`1 \times d` arrays
    :param weights: An :math:`n`-vector of weights where each element of the weights
        vector is is paired with the corresponding index of the data and supervision
        array, forming the triple :math:`(x_i, y_i, w_i)`; if passed a scalar weight,
        it will be broadcast to an :math:`n`-vector. the default value of :data:`None`
        sets the weights to the ones vector (implies a scalar weight of one);
    """

    supervision: Shaped[Array, " n *p"] = eqx.field(converter=jnp.atleast_2d)

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
