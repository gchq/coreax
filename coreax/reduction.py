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
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Self, TypeVar

import jax.numpy as jnp
from jax import Array, tree_util
from sklearn.neighbors import KDTree

import coreax.coresubset as cc
import coreax.data as cd
import coreax.kernel as ck
import coreax.metrics as cm
import coreax.refine as cr
import coreax.util as cu

# import coreax.weights as cw
from coreax.data import ArrayData, DataReader
from coreax.kernel import Kernel
from coreax.metrics import Metric, metric_factory
from coreax.util import NotCalculatedError, create_instance_from_factory
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

    def __init__(self, weights: WeightsOptimiser | None, kernel: Kernel | None = None):
        """Initialise class and set internal attributes to defaults."""
        validate_is_instance(weights, "weights", (WeightsOptimiser, None))
        self.weights = weights
        validate_is_instance(kernel, "kernel", (Kernel, None))
        self.kernel = kernel
        self.original_data: DataReader | None = None
        self.coreset: Array | None = None

    def copy_empty(self) -> Self:
        """
        Create an empty copy of this class with all data removed.

        Other parameters are retained.

        .. warning::
            This copy is shallow so :attr:`weights` etc. still point to the original
            object.

        .. warning::
            If any additional data attributes are added in a subclass, it should
            reimplement this method.

        :return: Copy of this class with data removed
        """
        new_obj = copy(self)
        new_obj.original_data = None
        new_obj.coreset = None
        return new_obj

    @abstractmethod
    def fit(self, original_data: DataReader, num_points: int) -> None:
        """
        Compute coreset.

        The resulting coreset is saved in-place to :attr:`coreset`.

        Any concrete implementation should call this super method to set
        :attr:`original_data`.

        If ``num_points`` is greater than the number of data points in
        :attr:`original_data`, the resulting coreset may be larger than the original
        data, if the coreset method permits. A :exc:`ValueError` is raised if it is not
        possible to generate a coreset of size ``num_points``.

        If ``num_points`` is equal to the number of data points in
        :attr:`original_data`, the resulting coreset is not necessarily equal to the
        original data, depending on the coreset method.

        :param original_data: Instance of :class:`~coreax.data.DataReader` containing
            the data we wish to reduce
        :param num_points: Number of points to include in coreset
        :raises ValueError: When it is not possible to generate a coreset of size
            ``num_points``
        :return: Nothing
        """
        validate_is_instance(original_data, "original_data", DataReader)
        self.original_data = original_data

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
        """Format coreset to match the shape of the original data."""
        self._validate_fitted("format")
        return self.original_data.format(self)

    def render(self) -> None:
        """Plot coreset interactively using :mod:`~matplotlib.pyplot`."""
        self._validate_fitted("render")
        return self.original_data.render(self)

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


CoresetMethod = TypeVar("CoresetMethod", bound=Coreset)


class ReductionStrategy(ABC):
    """
    Define a strategy for how to construct a coreset for a given type of coreset.

    The strategy determines the size of the coreset, approximation strategies to aid
    memory management and other similar aspects that wrap around the type of coreset.

    :param coreset_method: :class:`Coreset` instance to populate
    """

    def __init__(self, coreset_method: CoresetMethod):
        """Initialise class."""
        self.coreset_method = coreset_method

    @abstractmethod
    def reduce(self, original_data: DataReader) -> CoresetMethod:
        """
        Reduce a dataset to a coreset.

        :param original_data: Data to be reduced
        :return: Coreset calculated according to chosen type and this reduction strategy
        """

    @classmethod
    def _tree_unflatten(cls, aux_data, children) -> type[Self]:
        """
        Reconstruct a pytree from the tree definition and the leaves.

        Arrays & dynamic values (children) and auxiliary data (static values) are
        reconstructed. A method to reconstruct the pytree needs to be specified to
        enable jit decoration of methods inside children of this class.

        :param aux_data: Auxiliary data
        :param children: Arrays and dynamic values
        :return: Pytree
        """
        return cls(*children, **aux_data)


class SizeReduce(ReductionStrategy):
    """
    Calculate coreset containing a given number of points.

    :param coreset_method: :class:`Coreset` instance to populate
    :param num_points: Number of points to include in coreset
    """

    def __init__(self, coreset_method: CoresetMethod, num_points: int):
        """Initialise class."""
        super().__init__(coreset_method)

        num_points = cast_as_type(num_points, "num_points", int)
        validate_in_range(num_points, "num_points", True, lower_bound=0)
        self.num_points = num_points

    def reduce(self, original_data: DataReader) -> CoresetMethod:
        """
        Reduce a dataset to a coreset.

        The coreset is saved to the instance provided in :attr:`coreset_method` and this
        instance is returned by this method.

        :param original_data: Data to be reduced
        :return: Coreset calculated according to chosen type and this reduction strategy
        """
        validate_is_instance(original_data, "original_data", DataReader)
        self.coreset_method.fit(original_data=original_data, num_points=self.num_points)
        return self.coreset_method

    def _tree_flatten(self) -> tuple[tuple, dict]:
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable jit decoration
        of methods inside this class.
        """
        children = ()
        aux_data = {
            "coreset_method": self.coreset_method,
            "num_points": self.num_points,
        }
        return children, aux_data


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

    :param coreset_method: :class:`Coreset` instance to populate
    :param num_points: Number of points to include in coreset
    :param partition_size: Approximate number of points to include in each partition;
        actual partition sizes vary non-strictly between :attr:`partition_size` and
        :math:`2 \times` :attr:`partition_size`; must be greater than :attr:`num_points`
    :param parallel: If :data:`True`, calculate coresets on partitions in parallel
    """

    def __init__(
        self,
        coreset_method: CoresetMethod,
        num_points: int,
        partition_size: int,
        parallel: bool = True,
    ):
        """Initialise class."""
        super().__init__(coreset_method)

        num_points = cast_as_type(num_points, "num_points", int)
        validate_in_range(num_points, "num_points", True, lower_bound=0)
        self.num_points = num_points

        partition_size = cast_as_type(partition_size, "partition_size", int)
        validate_in_range(
            partition_size, "partition_size", True, lower_bound=num_points
        )
        self.partition_size = partition_size

        self.parallel = cast_as_type(parallel, "parallel", bool)

    def reduce(self, original_data: DataReader) -> CoresetMethod:
        """
        Reduce a dataset to a coreset using scalable reduction.

        It is performed using recursive calls to :meth:`_reduce_recursive`.

        :param original_data: Data to be reduced
        :return: Coreset calculated according to chosen type and this reduction strategy
        """
        coreset = self._reduce_recursive(ArrayData.load(original_data.pre_coreset_data))
        # Redirect original_data on coreset to point to true original
        coreset.original_data = original_data
        return coreset

    def _reduce_recursive(
        self,
        input_data: ArrayData,
        w_function: ck.Kernel | None,
        block_size: int = 10_000,
        K_mean: Array | None = None,
        nu: float = 1.0,
        partition_size: int = 1000,
        parallel: bool = True,
    ) -> CoresetMethod:
        r"""
        Recursively execute scalable reduction.

        :param input_data: Data to reduce on this iteration

        :param kernel: Kernel function
                       :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
        :param w_function: Weights function. If unweighted, this is `None`
        :param block_size: Size of matrix blocks to process
        :param K_mean: Row sum of kernel matrix divided by `n`
        :param partition_size: Region size in number of points. Optional, defaults to `1000`
        :param parallel: Use multiprocessing. Optional, defaults to `True`
        :param kwargs: Keyword arguments to be passed to `function` after `X` and `n_core`
        :return: Coreset and weights, where weights is empty if unweighted
        """
        # Check if no partitions are required
        if input_data.pre_coreset_array.shape[0] <= self.partition_size:
            # Length of input_data < num_points is only possible if input_data is the
            # original data, so it is safe to request a coreset of size larger than the
            # original data (if of limited use)
            return self._coreset_copy_fit(input_data)

        # Partitions required
        data_to_reduce = input_data.pre_coreset_array

        # Build a kdtree
        kdtree = KDTree(data_to_reduce, leaf_size=partition_size)
        _, node_indices, nodes, _ = kdtree.get_arrays()
        new_indices = [jnp.array(node_indices[nd[0] : nd[1]]) for nd in nodes if nd[2]]
        split_data = [data_to_reduce[n] for n in new_indices]

        # Generate a coreset on each partition
        partition_coresets = []
        kwargs["self.size"] = self.size
        if parallel:
            with ThreadPool() as pool:
                res = pool.map_async(
                    partial(
                        self.data_reduction.fit,
                        self.data_reduction.kernel,
                        block_size,
                        K_mean,
                        unique,
                        nu,
                    ),
                    split_data,
                )
                res.wait()
                for herding_output, idx in zip(res.get(), new_indices):
                    c, _, _ = herding_output
                    coreset.append(idx[c])

        else:
            for X_, idx in zip(split_data, new_indices):
                c, _, _ = self.data_reduction.fit(
                    X_, self.data_reduction.kernel, block_size, K_mean, unique, nu
                )
                coreset.append(idx[c])

        input_len = input_data.pre_coreset_array.shape[0]

        # Fewer data points than requested coreset size so return all
        if input_len <= self.num_points:
            coreset = self.coreset_method.copy_empty()
            coreset.coreset = input_data

        # Coreset points < data points <= partition size, so no partitioning required
        elif self.num_points < input_len <= self.partition_size:
            coreset = self.coreset_method.copy_empty()
            coreset.fit(input_data, self.num_points)

        # Partitions required
        else:
            data_to_reduce = input_data.pre_coreset_array

            # Build a kdtree
            kdtree = KDTree(data_to_reduce, leaf_size=partition_size)
            _, node_indices, nodes, _ = kdtree.get_arrays()
            new_indices = [
                jnp.array(node_indices[nd[0] : nd[1]]) for nd in nodes if nd[2]
            ]
            split_data = [ArrayData.load(data_to_reduce[n]) for n in new_indices]

            # Generate a coreset on each partition
            partition_coresets = []
            kwargs["self.size"] = self.size
            if parallel:
                with ThreadPool() as pool:
                    res = pool.map_async(
                        partial(
                            self.data_reduction.fit,
                            self.data_reduction.kernel,
                            block_size,
                            K_mean,
                            unique,
                            nu,
                        ),
                        split_data,
                    )
                    res.wait()
                    for herding_output, idx in zip(res.get(), new_indices):
                        c, _, _ = herding_output
                        coreset.append(idx[c])

            else:
                for X_, idx in zip(split_data, new_indices):
                    c, _, _ = self.data_reduction.fit(
                        X_, self.data_reduction.kernel, block_size, K_mean, unique, nu
                    )
                    coreset.append(idx[c])

            coreset = jnp.concatenate(coreset)
            Xc = data_to_reduce[coreset]
            self.reduction_indices = self.reduction_indices[coreset]
            # recurse;
            coreset, weights = self.map_reduce(
                w_function,
                block_size,
                K_mean,
                unique,
                nu,
                partition_size,
                parallel,
            )

        return coreset, weights

    def _coreset_copy_fit(self, input_data: ArrayData) -> CoresetMethod:
        """
        Create a new instance of :attr:`coreset_method` and fit to given data.

        :param input_data: Data to fit
        :return: New instance :attr:`coreset_method` fitted to ``input_data``
        """
        coreset = self.coreset_method.copy_empty()
        coreset.fit(input_data, self.num_points)
        return coreset

    def _tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable jit decoration
        of methods inside this class.
        """
        children = ()
        aux_data = {
            "coreset_method": self.coreset_method,
            "num_points": self.num_points,
        }
        return children, aux_data


# Define the pytree node for the added class to ensure methods with jit decorators
# are able to run. This tuple must be updated when a new class object is defined.
reduction_classes = (SizeReduce, MapReduce)
for current_class in reduction_classes:
    tree_util.register_pytree_node(
        current_class, current_class._tree_flatten, current_class._tree_unflatten
    )
