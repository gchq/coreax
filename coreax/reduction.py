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

r"""
This module reduces a large dataset down to a coreset.

To prepare data for reduction, convert it into a two-dimensional :math:`n \times d`
:class:`~jax.Array`. The resulting coreset will be :math:`m \times d` where
:math:`m \ll n` but still retain similar statistical properties.

When ready, the user selects a :class:`ReductionStrategy` to use to generate a chosen
type of :class:`Coreset`. As a simple example, the user may obtain a uniform random
sample of :math:`m` points by using the :class:`SizeReduce` strategy and a
:class:`~coreax.coresubset.RandomSample` coreset.

:class:`ReductionStrategy` and :class:`Coreset` are abstract base classes defining the
interface for which particular methods can be implemented.
"""

# Support annotations with | in Python < 3.10
# TODO: Remove once no longer supporting old code
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from multiprocessing.pool import ThreadPool
from typing import Self, TypeVar

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from sklearn.neighbors import KDTree

from coreax.data import ArrayData, DataReader
from coreax.kernel import Kernel
from coreax.metrics import Metric
from coreax.util import NotCalculatedError
from coreax.validation import cast_as_type, validate_in_range, validate_is_instance
from coreax.weights import WeightsOptimiser


class Coreset(ABC):
    r"""
    Methods for reducing data.

    Class for performing data reduction.

    :param weights: Type of weighting to apply, or :data:`None` if unweighted
    :param kernel: Kernel function
       :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`, or
       :data:`None` if not applicable
    """

    def __init__(self, weights: WeightsOptimiser | None, kernel: Kernel | None):
        """Initialise class and set internal attributes to defaults."""
        validate_is_instance(weights, "weights", (WeightsOptimiser, None))
        self.weights = weights
        validate_is_instance(kernel, "kernel", (Kernel, None))
        self.kernel = kernel
        self.original_data: DataReader | None = None  #: Data to be reduced
        self.coreset: Array | None = None  #: Calculated coreset
        self.coreset_indices: Array | None = None
        """Indices of :attr:`coreset` points in :attr:`original_data`, if applicable"""

    def clone_empty(self) -> Self:
        """
        Create an empty copy of this class with all data removed.

        Other parameters are retained.

        .. warning:: This copy is shallow so :attr:`weights` etc. still point to the
            original object.

        .. warning:: If any additional data attributes are added in a subclass, it
            should reimplement this method.

        :return: Copy of this class with data removed
        """
        new_obj = copy(self)
        new_obj.original_data = None
        new_obj.coreset = None
        new_obj.coreset_indices = None
        return new_obj

    def fit(self, original_data: DataReader, strategy: ReductionStrategy) -> None:
        """
        Compute coreset using a given reduction strategy.

        The resulting coreset is saved in-place to :attr:`coreset`.

        :param original_data: Instance of :class:`~coreax.data.DataReader` containing
            the data we wish to reduce
        :param strategy: Reduction strategy to use
        :return: Nothing
        """
        validate_is_instance(original_data, "original_data", DataReader)
        validate_is_instance(strategy, "strategy", ReductionStrategy)
        self.original_data = original_data
        strategy.reduce(self)

    @abstractmethod
    def fit_to_size(self, num_points: int) -> None:
        """
        Compute coreset for a fixed target size.

        .. note:: The user should not normally call this method directly; call
            :meth:`fit` instead.

        This method is equivalent to calling :meth:`fit` with strategy
        :class:`SizeReduce` with ``num_points`` except that it requires
        :attr:`original_data` to already be populated.

        The resulting coreset is saved in-place to :attr:`coreset`.

        If ``num_points`` is greater than the number of data points in
        :attr:`original_data`, the resulting coreset may be larger than the original
        data, if the coreset method permits. A :exc:`ValueError` is raised if it is not
        possible to generate a coreset of size ``num_points``.

        If ``num_points`` is equal to the number of data points in
        :attr:`original_data`, the resulting coreset is not necessarily equal to the
        original data, depending on the coreset method.

        :param num_points: Number of points to include in coreset
        :raises ValueError: When it is not possible to generate a coreset of size
            ``num_points``
        :return: Nothing
        """

    def solve_weights(self) -> Array:
        """
        Solve for optimal weighting of points in :attr:`coreset`.

        :return: Optimal weighting of points in :attr:`coreset` to represent the
            original data
        """
        self._validate_fitted("solve_weights")
        return self.weights.solve(self.original_data.pre_coreset_array, self.coreset)

    def compute_metric(self, metric: Metric, block_size: int | None = None) -> Array:
        """
        Compute metric comparing the coreset with the original data.

        The metric is computed unweighted. A weighted version may be implemented in
        future. For now, more options are available by calling the chosen
        :class:`~coreax.Metric` class directly.

        :param metric: Instance of :class:`~coreax.Metric` to use
        :param block_size: Size of matrix block to process, or :data:`None` to not split
            into blocks
        :return: Metric computed as a zero-dimensional array
        """
        validate_is_instance(metric, "metric", Metric)
        # block_size will be validated by metric.compute()
        return metric.compute(
            self.original_data.pre_coreset_array, self.coreset, block_size=block_size
        )

    def format(self) -> Array:
        """
        Format coreset to match the shape of the original data.

        :return: Array of formatted data
        """
        self._validate_fitted("format")
        return self.original_data.format(self)

    def render(self) -> None:
        """
        Plot coreset interactively using :mod:`~matplotlib.pyplot`.

        :return: Nothing
        """
        self._validate_fitted("render")
        return self.original_data.render(self)

    def copy_fit(self, other: Self, deep: bool = False) -> None:
        """
        Copy fitted coreset from other instance to this instance.

        The other coreset must be of the same type as this instance.

        :param other: :class:`Coreset` from which to copy calculated coreset
        :param deep: If :data:`True`, make a shallow copy of :attr:`coreset` and
            :attr:`coreset_indices`; otherwise, reference same objects
        :return: Nothing
        """
        validate_is_instance(other, "other", type[self])
        if deep:
            self.coreset = copy(other.coreset)
            self.coreset_indices = copy(other.coreset_indices)
        else:
            self.coreset = other.coreset
            self.coreset_indices = other.coreset_indices

    def _validate_fitted(self, caller_name: str) -> None:
        """
        Raise :exc:`~coreax.util.NotCalculatedError` if coreset has not been fitted yet.

        :param caller_name: Name of calling method to display in error message
        :raises NotCalculatedError: If :attr:`original_data` or :attr:`coreset` is
            :data:`None`
        :return: Nothing
        """
        if not isinstance(self.original_data, DataReader) or not isinstance(
            self.coreset, Array
        ):
            raise NotCalculatedError(f"Need to call fit before calling {caller_name}")


C = TypeVar("C", bound=Coreset)


class ReductionStrategy(ABC):
    """
    Define a strategy for how to construct a coreset for a given type of coreset.

    The strategy determines the size of the coreset, approximation strategies to aid
    memory management and other similar aspects that wrap around the type of coreset.
    """

    def __init__(self):
        """Initialise class."""

    @abstractmethod
    def reduce(self, coreset: Coreset) -> None:
        """
        Reduce a dataset to a coreset using this strategy.

        ``coreset`` is updated in place.

        :param coreset: :class:`Coreset` instance to populate in place
        :return: Nothing
        """


class SizeReduce(ReductionStrategy):
    """
    Calculate coreset containing a given number of points.

    :param num_points: Number of points to include in coreset
    """

    def __init__(self, num_points: int):
        """Initialise class."""
        super().__init__()

        num_points = cast_as_type(num_points, "num_points", int)
        validate_in_range(num_points, "num_points", True, lower_bound=0)
        self.num_points = num_points

    def reduce(self, coreset: Coreset) -> None:
        """
        Reduce a dataset to a coreset using this strategy.

        ``coreset`` is updated in place.

        :param coreset: :class:`Coreset` instance to populate in place
        :return: Nothing
        """
        coreset.fit_to_size(self.num_points)


class MapReduce(ReductionStrategy):
    r"""
    Calculate coreset of a given number of points using scalable reduction on blocks.

    This is a less memory-intensive alternative to :class:`SizeReduce`.

    It uses a :class:`~sklearn.neighbors.KDTree` to partition the original data
    into patches. Upon each of these a coreset of size :attr:`num_points` is calculated.
    These coresets are concatenated to produce a larger coreset covering the whole of
    the original data, which thus has size greater than :attr:`num_points`. This coreset
    is now treated as the original data and reduced recursively until its size is equal
    to :attr:`num_points`.

    :attr:`num_points` <= :attr:`partition_size` to prevent exponential growth of
    coreset size at each iteration. If for whatever reason you wish to break this
    restriction, use :class:`SizeReduce` instead.

    There is some intricate set-up:

    #.  :attr:`num_points` must be less than :attr:`partition_size`.
    #.  Unweighted coresets are calculated on each patch of roughly
        :attr:`partition_size` points and then concatenated. More specifically, each
        patch contains between :attr:`partition_size` and
        :math:`2 \times` :attr:`partition_size` points, inclusive.
    #.  Recursively calculate ever smaller coresets until a global coreset with size
        :attr:`num_points` is obtained.
    #.  If the input data on the final iteration is smaller than :attr:`num_points`, the
        whole input data is returned as the coreset and thus is smaller than the
        requested size.

    Let :math:`n_k` be the number of points after each recursion with :math:`n_0` equal
    to the size of the original data. Then, each recursion reduces the size of the
    coreset such that

    .. math::

        n_k <= \frac{n_{k - 1}}{\texttt{partition_size}} \texttt{num_points},

    so

    .. math::

        n_k <= \left( \frac{\texttt{num_points}}{\texttt{partition_size}} \right)^k n_0.

    Thus, the number of iterations required is roughly (find :math:`k` when
    :math:`n_k =` :attr:`num_points`)

    .. math::

        \frac{\log{\texttt{num_points}} - \log{\left( \text{original data size} \right)}}
        {\log{\texttt{num_points}} - \log{\texttt{partition_size}}} .

    :param num_points: Number of points to include in coreset
    :param partition_size: Approximate number of points to include in each partition;
        actual partition sizes vary non-strictly between :attr:`partition_size` and
        :math:`2 \times` :attr:`partition_size`; must be greater than :attr:`num_points`
    :param parallel: If :data:`True`, calculate coresets on partitions in parallel
    """

    def __init__(
        self,
        num_points: int,
        partition_size: int,
        parallel: bool = True,
    ):
        """Initialise class."""
        super().__init__()

        num_points = cast_as_type(num_points, "num_points", int)
        validate_in_range(num_points, "num_points", True, lower_bound=0)
        self.num_points = num_points

        partition_size = cast_as_type(partition_size, "partition_size", int)
        validate_in_range(
            partition_size, "partition_size", True, lower_bound=num_points
        )
        self.partition_size = partition_size

        self.parallel = cast_as_type(parallel, "parallel", bool)

    def reduce(self, coreset: Coreset) -> None:
        """
        Reduce a dataset to a coreset using scalable reduction.

        It is performed using recursive calls to :meth:`_reduce_recursive`.

        :param coreset: :class:`Coreset` instance to populate in place
        :return: Nothing
        """
        input_data = coreset.original_data.pre_coreset_array
        # _reduce_recursive returns a copy of coreset so need to transfer calculated
        # coreset fit into the original coreset object
        coreset.copy_fit(
            self._reduce_recursive(
                template=coreset,
                input_data=ArrayData.load(input_data),
                input_indices=jnp.array(range(input_data.shape[0])),
            )
        )

    def _reduce_recursive(
        self,
        template: C,
        input_data: ArrayData,
        input_indices: ArrayLike | None,
    ) -> C:
        r"""
        Recursively execute scalable reduction.

        :param template: Instance of :class:`Coreset` to duplicate
        :param input_data: Data to reduce on this iteration
        :param input_indices: Indices of ``input_data``, if applicable to ``template``
        :return: Copy of ``template`` containing fitted coreset
        """
        # Check if no partitions are required
        if input_data.pre_coreset_array.shape[0] <= self.partition_size:
            # Length of input_data < num_points is only possible if input_data is the
            # original data, so it is safe to request a coreset of size larger than the
            # original data (if of limited use)
            return self._coreset_copy_fit(template, input_data, input_indices)

        # Partitions required
        data_to_reduce = input_data.pre_coreset_array

        # Build a kdtree
        kdtree = KDTree(data_to_reduce, leaf_size=self.partition_size)
        _, node_indices, nodes, _ = kdtree.get_arrays()
        new_indices = [jnp.array(node_indices[nd[0] : nd[1]]) for nd in nodes if nd[2]]
        split_data = [data_to_reduce[n] for n in new_indices]

        # Generate a coreset on each partition
        if self.parallel:
            with ThreadPool() as pool:
                res = pool.map_async(
                    lambda args: self._coreset_copy_fit(template, *args),
                    zip(split_data, new_indices),
                )
                res.wait()
                partition_coresets: list[C] = res.get()
        else:
            partition_coresets = [
                self._coreset_copy_fit(template, sd, sd_indices)
                for sd, sd_indices in zip(split_data, new_indices)
            ]

        # Concatenate coresets
        full_coreset = jnp.concatenate([pc.coreset for pc in partition_coresets])
        if partition_coresets[0].corset_indices is None:
            full_indices = None
        else:
            full_indices = jnp.concatenate(
                [pc.coreset_indices for pc in partition_coresets]
            )

        # Recursively reduce large coreset
        # coreset_indices will be None if not applicable to the coreset method
        return self._reduce_recursive(
            template=template, input_data=full_coreset, input_indices=full_indices
        )

    def _coreset_copy_fit(
        self, template: C, input_data: ArrayData, input_indices: ArrayLike | None
    ) -> C:
        """
        Create a new instance of a :class:`Coreset` and fit to given data.

        If applicable to the coreset method, the coreset indices are overwritten using
        ``input_indices`` as the mapping to allow ``input_data`` to be a subset of the
        original data.

        :param template: Instance of :class:`Coreset` to duplicate
        :param input_data: Data to fit
        :param input_indices: Indices of ``input_data``, if applicable to ``template``
        :return: New instance :attr:`coreset_method` fitted to ``input_data``
        """
        coreset = template.clone_empty()
        coreset.original_data = input_data
        coreset.fit_to_size(self.num_points)
        # Update indices
        if coreset.coreset_indices is not None:
            # Should not reach here if input_indices is not populated
            assert input_indices is not None
            coreset.coreset_indices = input_indices[coreset.coreset_indices]
        return coreset
