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
:class:`DataReader` because :class:`~coreax.Coreset` requires a
two-dimensional :class:`~jax.Array`. Data reductions are performed along the first
dimension.

The user should read in their data files using their preferred library that returns a
:class:`jax.Array` or :func:`numpy.array`. This array is passed to a
:meth:`load() <DataReader.load>` method. The user should not normally call
:meth:`DataReader.__init__` directly. The user should select an appropriate subclass
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
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import coreax.validation

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
        self.original_data: Array = coreax.validation.cast_as_type(
            original_data, "original_data", jnp.atleast_2d
        )
        self.pre_coreset_array: Array = coreax.validation.cast_as_type(
            pre_coreset_array, "pre_coreset_array", jnp.atleast_2d
        )

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
        :return: Nothing
        """
        raise NotImplementedError

    def reduce_dimension(self, num_dimensions: int) -> None:
        """
        Reduce dimensionality of :attr:`pre_coreset_array`.

        Performed using principal component analysis (PCA).

        :attr:`pre_coreset_array` is updated in place. Metadata detailing the type of
        reduction are saved to this class to enable reconstruction later.

        :param num_dimensions: Target number of dimensions
        :return: Nothing
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
        :return: Array with the same number of columns as :attr:`pre_coreset_data` had
            prior to calling :meth:`reduce_dimension` and the same number of rows as the
            coreset
        """
        raise NotImplementedError


class ArrayData(DataReader):
    """
    Class to apply pre- and post-processing to two-dimensional array data.

    Data should already be in a format accepted by :class:`~coreax.reduction.Coreset`.
    Thus, if no dimensionality reduction is performed, this class is an identity
    wrapper and :attr:`pre_coreset_array` is equal to :attr:`original_data`.

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
        original_data = coreax.validation.cast_as_type(
            original_data, "original_data", jnp.atleast_2d
        )
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
