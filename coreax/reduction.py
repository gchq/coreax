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

To prepare data for reduction, convert it into a :class:`~jax.Array` and pass to an
appropriate instance of :class:`~coreax.data.DataReader`. The class will convert the
data internally to be an :math:`n \times d` :class:`~jax.Array`. The resulting coreset
will be :math:`m \times d` where :math:`m \ll n` but still retain similar statistical
properties.

The user selects a method by choosing a :class:`Coreset` and a
:class:`ReductionStrategy`. For example, the user may obtain a uniform random
sample of :math:`m` points by using the :class:`SizeReduce` strategy and a
:class:`~coreax.coresubset.RandomSample` coreset. This may be implemented for
:class:`~coreax.data.ArrayData` by calling

.. code-block:: python

    original_data = ArrayData.load(input_data)
    coreset = RandomSample()
    coreset.reduce(original_data, SizeReduce(m))
    print(coreset.format())

:class:`ReductionStrategy` and :class:`Coreset` are abstract base classes defining the
interface for which particular methods can be implemented.
"""

# Support annotations with | in Python < 3.10
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from multiprocessing.pool import ThreadPool
from typing import TypeVar

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from sklearn.neighbors import KDTree
from typing_extensions import Self

import coreax.data
import coreax.kernel
import coreax.metrics
import coreax.refine
import coreax.util
import coreax.validation
import coreax.weights


class Coreset(ABC):
    r"""
    Class for reducing data to a coreset.

    :param weights_optimiser: :class:`~coreax.weights.WeightsOptimiser` object to
        determine weights for coreset points to optimise some quality metric, or
        :data:`None` (default) if unweighted
    :param kernel: :class:`~coreax.Kernel` instance implementing a kernel function
       :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`, or
       :data:`None` if not applicable
    :param refine_method: :class:`~coreax.refine.Refine` object to use, or :data:`None`
        (default) if no refinement is required
    """

    def __init__(
        self,
        *,
        weights_optimiser: coreax.weights.WeightsOptimiser | None = None,
        kernel: coreax.kernel.Kernel | None = None,
        refine_method: coreax.refine.Refine | None = None,
    ):
        """Initialise class and set internal attributes to defaults."""
        coreax.validation.validate_is_instance(
            weights_optimiser,
            "weights_optimiser",
            (coreax.weights.WeightsOptimiser, type(None)),
        )
        self.weights_optimiser = weights_optimiser
        coreax.validation.validate_is_instance(
            kernel, "kernel", (coreax.kernel.Kernel, type(None))
        )
        self.kernel = kernel
        coreax.validation.validate_is_instance(
            refine_method, "refine_method", (coreax.refine.Refine, type(None))
        )
        self.refine_method = refine_method

        # Data attributes not set in init
        self.original_data: coreax.data.DataReader | None = None
        """
        Data to be reduced
        """
        self.coreset: Array | None = None
        """
        Calculated coreset. The order of rows need not be monotonic with those in the
        original data (applicable only to coresubset).
        """
        self.coreset_indices: Array | None = None
        """
        Indices of :attr:`coreset` points in :attr:`original_data`, if applicable. The
        order matches the rows of :attr:`coreset`.
        """
        self.kernel_matrix_row_sum_mean: ArrayLike | None = None
        r"""
        Mean vector over rows for the Gram matrix, a :math:`1 \times n` array. If
        :meth:`fit_to_size` calculates this, it will be saved here automatically to save
        recalculating it in :meth:`refine`.
        """

    def clone_empty(self) -> Self:
        """
        Create an empty copy of this class with all data removed.

        Other parameters are retained.

        .. warning:: This copy is shallow so :attr:`weights_optimiser` etc. still point
            to the original object.

        .. warning:: If any additional data attributes are added in a subclass, it
            should reimplement this method.

        :return: Copy of this class with data removed
        """
        new_obj = copy(self)
        new_obj.original_data = None
        new_obj.coreset = None
        new_obj.coreset_indices = None
        new_obj.kernel_matrix_row_sum_mean = None
        return new_obj

    def fit(
        self, original_data: coreax.data.DataReader, strategy: ReductionStrategy
    ) -> None:
        """
        Compute coreset using a given reduction strategy.

        The resulting coreset is saved in-place to :attr:`coreset`.

        :param original_data: Instance of :class:`~coreax.data.DataReader` containing
            the data we wish to reduce
        :param strategy: Reduction strategy to use
        :return: Nothing
        """
        coreax.validation.validate_is_instance(
            original_data, "original_data", coreax.data.DataReader
        )
        coreax.validation.validate_is_instance(strategy, "strategy", ReductionStrategy)
        self.original_data = original_data
        strategy.reduce(self)

    @abstractmethod
    def fit_to_size(self, coreset_size: int) -> None:
        """
        Compute coreset for a fixed target size.

        .. note:: The user should not normally call this method directly; call
            :meth:`fit` instead.

        This method is equivalent to calling :meth:`fit` with strategy
        :class:`SizeReduce` with ``coreset_size`` except that it requires
        :attr:`original_data` to already be populated.

        The resulting coreset is saved in-place to :attr:`coreset`.

        If ``coreset_size`` is greater than the number of data points in
        :attr:`original_data`, the resulting coreset may be larger than the original
        data, if the coreset method permits. A :exc:`ValueError` is raised if it is not
        possible to generate a coreset of size ``coreset_size``.

        If ``coreset_size`` is equal to the number of data points in
        :attr:`original_data`, the resulting coreset is not necessarily equal to the
        original data, depending on the coreset method, metric and weighting.

        :param coreset_size: Number of points to include in coreset
        :raises ValueError: When it is not possible to generate a coreset of size
            ``coreset_size``
        :return: Nothing
        """

    def solve_weights(self) -> Array:
        """
        Solve for optimal weighting of points in :attr:`coreset`.

        :return: Optimal weighting of points in :attr:`coreset` to represent the
            original data
        """
        self.validate_fitted("solve_weights")
        return self.weights_optimiser.solve(
            self.original_data.pre_coreset_array, self.coreset
        )

    def compute_metric(
        self,
        metric: coreax.metrics.Metric,
        block_size: int | None = None,
        weights_x: ArrayLike | None = None,
        weights_y: ArrayLike | None = None,
    ) -> Array:
        r"""
        Compute metric comparing the coreset with the original data.

        The metric is computed unweighted unless ``weights_x`` and/or ``weights_y`` is
        supplied as an array. Further options are available by calling the chosen
        :class:`~coreax.Metric` class directly.

        :param metric: Instance of :class:`~coreax.metrics.Metric` to use
        :param block_size: Size of matrix block to process, or :data:`None` to not split
            into blocks
        :param weights_x: An :math:`1 \times n` array of weights for associated points
            in ``x``, or :data:`None` if not required
        :param weights_y: An :math:`1 \times m` array of weights for associated points
            in ``y``, or :data:`None` if not required
        :return: Metric computed as a zero-dimensional array
        """
        coreax.validation.validate_is_instance(metric, "metric", coreax.metrics.Metric)
        self.validate_fitted("compute_metric")
        # block_size will be validated by metric.compute()
        return metric.compute(
            self.original_data.pre_coreset_array,
            self.coreset,
            block_size=block_size,
            weights_x=weights_x,
            weights_y=weights_y,
        )

    def refine(self) -> None:
        """
        Refine coreset.

        Only applicable to coreset methods that generate coresubsets.

        :attr:`coreset` is updated in place.

        :raises TypeError: When :attr:`refine_method` is :data:`None`
        :return: Nothing
        """
        if self.refine_method is None:
            raise TypeError("Cannot refine without a refine_method")
        # Validate appropriate attributes are set on coreset inside refine_method.refine
        self.refine_method.refine(self)

    def format(self) -> Array:
        """
        Format coreset to match the shape of the original data.

        :return: Array of formatted data
        """
        self.validate_fitted("format")
        return self.original_data.format(self)

    def render(self) -> None:
        """
        Plot coreset interactively using :mod:`~matplotlib.pyplot`.

        :return: Nothing
        """
        self.validate_fitted("render")
        return self.original_data.render(self)

    def copy_fit(self, other: Self, deep: bool = False) -> None:
        """
        Copy fitted coreset from other instance to this instance.

        The other coreset must be of the same type as this instance and
        :attr:`original_data` must also be populated on ``other``. The user must ensure
        :attr:`original_data` is correctly populated on this instance.

        :param other: :class:`Coreset` from which to copy calculated coreset
        :param deep: If :data:`True`, make a shallow copy of :attr:`coreset` and
            :attr:`coreset_indices`; otherwise, reference same objects
        :return: Nothing
        """
        coreax.validation.validate_is_instance(other, "other", type(self))
        other.validate_fitted("copy_fit from another Coreset")
        if deep:
            self.coreset = copy(other.coreset)
            self.coreset_indices = copy(other.coreset_indices)
        else:
            self.coreset = other.coreset
            self.coreset_indices = other.coreset_indices

    def validate_fitted(self, caller_name: str) -> None:
        """
        Raise :exc:`~coreax.util.NotCalculatedError` if coreset has not been fitted yet.

        :param caller_name: Name of calling method to display in error message
        :raises NotCalculatedError: If :attr:`original_data` or :attr:`coreset` is
            :data:`None`
        :return: Nothing
        """
        if not isinstance(self.original_data, coreax.data.DataReader) or not isinstance(
            self.coreset, Array
        ):
            raise coreax.util.NotCalculatedError(
                f"Need to call fit before calling {caller_name}"
            )


C = TypeVar("C", bound=Coreset)


# pylint: disable=too-few-public-methods
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

    :param coreset_size: Number of points to include in coreset
    """

    def __init__(self, coreset_size: int):
        """Initialise class."""
        super().__init__()

        coreset_size = coreax.validation.cast_as_type(coreset_size, "coreset_size", int)
        coreax.validation.validate_in_range(
            coreset_size, "coreset_size", True, lower_bound=0
        )
        self.coreset_size = coreset_size

    def reduce(self, coreset: Coreset) -> None:
        """
        Reduce a dataset to a coreset using this strategy.

        ``coreset`` is updated in place.

        :param coreset: :class:`Coreset` instance to populate in place
        :return: Nothing
        """
        coreset.fit_to_size(self.coreset_size)


class MapReduce(ReductionStrategy):
    r"""
    Calculate coreset of a given number of points using scalable reduction on blocks.

    This is a less memory-intensive alternative to :class:`SizeReduce`.

    It uses a :class:`~sklearn.neighbors.KDTree` to partition the original data into
    patches. Upon each of these a coreset of size :attr:`coreset_size` is calculated.
    These coresets are concatenated to produce a larger coreset covering the whole of
    the original data, which thus has size greater than :attr:`coreset_size`. This
    coreset is now treated as the original data and reduced recursively until its
    size is equal to :attr:`coreset_size`.

    :attr:`coreset_size` < :attr:`leaf_size` to ensure the algorithm converges. If
    for whatever reason you wish to break this restriction, use :class:`SizeReduce`
    instead.

    There is some intricate set-up:

    #.  :attr:`coreset_size` must be less than :attr:`leaf_size`.
    #.  Unweighted coresets are calculated on each patch of roughly
        :attr:`leaf_size` points and then concatenated. More specifically, each
        patch contains between :attr:`leaf_size` and
        :math:`2 \,\,\times` :attr:`leaf_size` points, inclusive.
    #.  Recursively calculate ever smaller coresets until a global coreset with size
        :attr:`coreset_size` is obtained.
    #.  If the input data on the final iteration is smaller than :attr:`coreset_size`,
        the whole input data is returned as the coreset and thus is smaller than the
        requested size.

    Let :math:`n_k` be the number of points after each recursion with :math:`n_0` equal
    to the size of the original data. Then, each recursion reduces the size of the
    coreset such that

    .. math::

        n_k <= \frac{n_{k - 1}}{\texttt{leaf_size}} \texttt{coreset_size},

    so

    .. math::

        n_k <= \left( \frac{\texttt{coreset_size}}{\texttt{leaf_size}} \right)^k n_0.

    Thus, the number of iterations required is roughly (find :math:`k` when
    :math:`n_k =` :attr:`coreset_size`)

    .. math::

        \frac{
            \log{\texttt{coreset_size}} - \log{\left(\text{original data size}\right)}
        }{
            \log{\texttt{coreset_size}} - \log{\texttt{leaf_size}}
        } .

    :param coreset_size: Number of points to include in coreset
    :param leaf_size: Approximate number of points to include in each partition;
        corresponds to ``leaf_size`` in :class:`~sklearn.neighbors.KDTree`;
        actual partition sizes vary non-strictly between :attr:`leaf_size` and
        :math:`2 \,\times` :attr:`leaf_size`; must be greater than :attr:`coreset_size`
    :param parallel: If :data:`True`, calculate coresets on partitions in parallel
    """

    def __init__(
        self,
        coreset_size: int,
        leaf_size: int,
        parallel: bool = True,
    ):
        """Initialise class."""
        super().__init__()

        coreset_size = coreax.validation.cast_as_type(coreset_size, "coreset_size", int)
        coreax.validation.validate_in_range(
            coreset_size, "coreset_size", True, lower_bound=0
        )
        self.coreset_size = coreset_size

        leaf_size = coreax.validation.cast_as_type(leaf_size, "leaf_size", int)
        coreax.validation.validate_in_range(
            leaf_size, "leaf_size", True, lower_bound=coreset_size
        )
        self.leaf_size = leaf_size

        self.parallel = coreax.validation.cast_as_type(parallel, "parallel", bool)

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
                input_data=input_data,
                input_indices=jnp.array(range(input_data.shape[0])),
            )
        )

    def _reduce_recursive(
        self,
        template: C,
        input_data: ArrayLike,
        input_indices: ArrayLike | None = None,
    ) -> C:
        r"""
        Recursively execute scalable reduction.

        :param template: Instance of :class:`Coreset` to duplicate
        :param input_data: Data to reduce on this iteration
        :param input_indices: Indices of ``input_data``, if applicable to ``template``
        :return: Copy of ``template`` containing fitted coreset
        """
        # Check if no partitions are required
        if input_data.shape[0] <= self.leaf_size:
            # Length of input_data < coreset_size is only possible if input_data is the
            # original data, so it is safe to request a coreset of size larger than the
            # original data (if of limited use)
            return self._coreset_copy_fit(template, input_data, input_indices)

        # Partitions required

        # Build a kdtree
        kdtree = KDTree(input_data, leaf_size=self.leaf_size)
        _, node_indices, nodes, _ = kdtree.get_arrays()
        new_indices = [jnp.array(node_indices[nd[0] : nd[1]]) for nd in nodes if nd[2]]
        split_data = [input_data[n] for n in new_indices]

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
        if partition_coresets[0].coreset_indices is None:
            full_indices = None
        else:
            full_indices = jnp.concatenate(
                [input_indices[pc.coreset_indices] for pc in partition_coresets]
            )

        # Recursively reduce large coreset
        # coreset_indices will be None if not applicable to the coreset method
        return self._reduce_recursive(
            template=template, input_data=full_coreset, input_indices=full_indices
        )

    def _coreset_copy_fit(
        self, template: C, input_data: ArrayLike, input_indices: ArrayLike | None
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
        coreset.original_data = coreax.data.ArrayData.load(input_data)
        coreset.fit_to_size(self.coreset_size)
        # Update indices
        if coreset.coreset_indices is not None:
            # Should not reach here if input_indices is not populated
            assert input_indices is not None
            coreset.coreset_indices = input_indices[coreset.coreset_indices]
        return coreset


# pylint: enable=too-few-public-methods
