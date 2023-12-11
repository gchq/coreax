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

In order to calculate a coreset, an instance of a subclass of :class:`DataReader` is
passed to :class:`~coreax.ReductionStrategy`. It is necessary to use
:class:`DataReader` because :class:`~coreax.ReductionStrategy` requires a
two-dimensional :class:`~jax.Array`. Data reductions are performed along the first
dimension.

The user should read in their data files using their preferred library that returns a
:class:`jax.Array` or :class:`numpy.Array`. This array is passed to a
:meth:`DataReader.load` method. The user should not normally call
:meth:`DataReader.__init__` directly. The user should select an appropriate subclass
of :class:`DataReader` to match the structure of the input array. The :meth:`load`
method on the subclass will rearrange the original data into the required
two-dimensional format.

Various post-processing methods may be implemented if applicable to visualise or
restore a calculated coreset to match the format of the original data. To save a
copy of a coreset, call :meth:`format` to return an :class:`~jax.Array`, which can
be passed to the chosen IO library to write a file.
"""

# Support annotations with | in Python < 3.10
# TODO: Remove once no longer supporting old code
from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from coreax.reduction import Coreset, ReductionStrategy, reduction_strategy_factory
from coreax.util import create_instance_from_factory
from coreax.validation import cast_as_type


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
        self.original_data: Array = cast_as_type(original_data, jnp.asarray)
        self.pre_coreset_array: Array = cast_as_type(pre_coreset_array, jnp.atleast_2d)

    @classmethod
    @abstractmethod
    def load(cls, original_data: ArrayLike) -> DataReader:
        """
        Construct :class:`DataReader` from an array of original data.

        This class method restructures the original data into the layout required for
        :class:`Coreset`.

        The user should not normally initialise this class directly; instead use this
        constructor.

        :param original_data: Array of data to be reduced to a coreset
        :return: Populated instance of :class:`DataReader`
        """

    def reduce(
        self,
        coreset_method: str | type[Coreset],
        reduction_strategy: str | type[ReductionStrategy],
        **kwargs,
    ) -> Coreset:
        """
        Reduce original data stored in this class to a coreset.

        :param coreset_method: Type of coreset to generate, expressed either as a string
            name or uninstantiated class
        :param reduction_strategy: Reduction strategy to use when calculating this
            coreset, expressed either as a string name or uninstantiated class
        :param kwargs: Keyword arguments to be passed during initialisation of
            :class:`~coreax.reduction.Coreset` or
            :class:`~coreax.reduction.ReductionStrategy` as appropriate
        :return: Instance of :class:`~coreax.reduction.Coreset` containing the
            calculated coreset
        """
        return create_instance_from_factory(
            reduction_strategy_factory,
            reduction_strategy,
            coreset_method=coreset_method,
            **kwargs,
        ).reduce(self)

    @abstractmethod
    def format(self, coreset: Coreset) -> Array:
        """
        Format coreset to match the shape of the original data.

        If the number of columns was reduced by :meth:`reduce_dimension`, it will be
        reverted by this method via a call to :meth:`restore_dimension`.

        :param coreset: Coreset to format
        :return: Array of coreset in format matching original data
        """

    @abstractmethod
    def render(self, coreset: Coreset | None) -> None:
        """
        Plot coreset or original data interactively using :mod:`~matplotlib.pyplot`.

        :param coreset: Coreset to plot, or :data:`None` to plot original data
        :return: Nothing
        """

    def reduce_dimension(self, num_dimensions: int) -> None:
        """
        Reduce dimensionality of :attr:`pre_coreset_array`.

        Performed using pricipal component analysis (PCA).

        :attr:`pre_coreset_array` is updated in place. Meta data detailing the type of
        reduction are save to this class to enable reconstruction later.

        :param num_dimensions: Target number of dimensions
        :return: Nothing
        """
        raise NotImplementedError

    def restore_dimension(self, coreset: Coreset | None) -> Array:
        """
        Expand principle components into original dimensions.

        Some data will have been lost due to reduction in dimensionality, so the
        restored data will not exactly match the original data.

        :param coreset: Coreset to restore, or :data:`None` to restore original data
        :return: Array matching original number of columns of :attr:`pre_coreset_data`
        """
        raise NotImplementedError
