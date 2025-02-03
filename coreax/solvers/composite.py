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

"""Solvers that compose with other solvers."""

import math
import warnings
from typing import Generic, Optional, TypeVar, Union, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import Array
from jaxtyping import Integer
from sklearn.neighbors import BallTree, KDTree
from typing_extensions import TypeAlias, override

from coreax.coreset import AbstractCoreset, Coresubset
from coreax.data import Data, SupervisedData
from coreax.solvers.base import ExplicitSizeSolver, PaddingInvariantSolver, Solver
from coreax.util import tree_zero_pad_leading_axis

BinaryTree: TypeAlias = Union[KDTree, BallTree]
_Data = TypeVar("_Data", Data, SupervisedData)
_Coreset = TypeVar("_Coreset", bound=AbstractCoreset)
_State = TypeVar("_State")
_Indices = Integer[Array, "..."]


class CompositeSolver(
    Solver[_Coreset, _Data, _State], Generic[_Coreset, _Data, _State]
):
    """Base class for solvers that compose with/wrap other solvers."""

    base_solver: Solver[_Coreset, _Data, _State]


class MapReduce(
    CompositeSolver[_Coreset, _Data, _State],
    Generic[_Coreset, _Data, _State],
):
    r"""
    Calculate coreset of a given number of points using scalable reduction on blocks.

    Uses a :class:`~sklearn.neighbors.KDTree` or :class:`~sklearn.neighbors.BallTree` to
    partition the original data into patches. Upon each of these a coreset of size
    ``base_solver.coreset_size`` is calculated. These coresets are concatenated to
    produce a larger coreset covering the whole of the original data, which has size
    greater than ``coreset_size``. This coreset is now treated as the original data and
    reduced recursively until its size is equal to ``base_solver.coreset_size``.

    There is some intricate set-up:

    #.  ``base_solver.coreset_size`` must be less than ``leaf_size``.
    #.  Zero weighted and valued padding will be used to ensure each partition has the
        same size if ``len(dataset)`` is not an integer multiple of ``leaf_size``

    Let :math:`n_k` be the number of points after each recursion with :math:`n_0` equal
    to the size of the original data. Then, each recursion reduces the size of the
    coreset such that

    .. math::

        n_k <= \frac{n_{k - 1}}{\texttt{leaf_size}} \texttt{coreset_size},

    so

    .. math::

        n_k <= \left( \frac{\texttt{coreset_size}}{\texttt{leaf_size}} \right)^k n_0.

    Thus, the number of iterations required is roughly (find :math:`k` when
    :math:`n_k =` ``base_solver.coreset_size``)

    .. math::

        \frac{
            \log{\texttt{coreset_size}} - \log{\left(\text{original data size}\right)}
        }{
            \log{\texttt{coreset_size}} - \log{\texttt{leaf_size}}
        } .

    :param base_solver: Solver to compose with; full support is currently provided for
        :class:`coreax.solvers.KernelHerding` and :class:`coreax.solvers.SteinThinning`;
        the solver's result must be ignorant of/invariant to zero weighted padding.
    :param leaf_size: Number of points to include in each partition; corresponds to
        :code:`2*leaf_size` in ``sklearn.neighbors.BinaryTree``; must be greater
        than ``base_solver.coreset_size``
    :param tree_type: The type of binary tree based partitioning to use when
        splitting the dataset into smaller blocks.
    """

    leaf_size: int = eqx.field(converter=int)
    tree_type: type[BinaryTree] = KDTree

    def __check_init__(self):
        """Check 'leaf_size' is an integer larger than 'base_solver.coreset_size'."""
        if not isinstance(self.base_solver, ExplicitSizeSolver):
            raise ValueError("'base_solver' must be an 'ExplicitSizeSolver'")
        if self.leaf_size <= self.base_solver.coreset_size:
            raise ValueError(
                "'leaf_size' must be larger than 'base_solver.coreset_size'"
            )
        if not isinstance(self.base_solver, PaddingInvariantSolver):
            warnings.warn(
                "'base_solver' is not a 'PaddingInvariantSolver'; Zero-weighted padding"
                + " applied in 'MapReduce' may lead to undefined results.",
                stacklevel=2,
            )

    @override
    def reduce(
        self, dataset: _Data, solver_state: Optional[_State] = None
    ) -> tuple[_Coreset, _State]:
        # There is no obvious way to use state information here.
        del solver_state

        @overload
        def _reduce_coreset(
            data: _Data, _indices: _Indices
        ) -> tuple[_Coreset, _State, _Indices]: ...

        @overload
        def _reduce_coreset(
            data: _Data, _indices: None = None
        ) -> tuple[_Coreset, _State, None]: ...

        def _reduce_coreset(
            data: _Data, _indices: Optional[_Indices] = None
        ) -> tuple[_Coreset, _State, Optional[_Indices]]:
            if len(data) <= self.leaf_size:
                coreset, state = self.base_solver.reduce(data)
                if _indices is not None:
                    # TODO: can we avoid Nodes here?
                    _indices = _indices[coreset.nodes.data]
                return coreset, state, _indices

            def wrapper(partition: _Data) -> tuple[_Data, Array]:
                """
                Apply the `reduce` method of the base solver on a partition.

                This is a wrapper for `reduce()` for processing a single partition.
                The data is partitioned with `_jit_tree()`.
                The reduction is performed on each partition via ``vmap()``.
                """
                x, _ = self.base_solver.reduce(partition)
                # TODO: can we avoid `nodes` here?
                return x.points, x.nodes.data

            partitioned_dataset, partitioned_indices = _jit_tree(
                data, self.leaf_size, self.tree_type
            )
            # Reduce each partition and get indices from each
            coreset_ensemble, ensemble_indices = jax.vmap(wrapper)(partitioned_dataset)
            # Calculate the indices with respect to the original data
            concatenated_indices = jax.vmap(lambda x, index: x[index])(
                partitioned_indices, ensemble_indices
            )
            concatenated_indices = jnp.ravel(concatenated_indices)
            _coreset = jtu.tree_map(jnp.concatenate, coreset_ensemble)

            if _indices is not None:
                final_indices = _indices[concatenated_indices]
            else:
                final_indices = concatenated_indices
            return _reduce_coreset(_coreset, final_indices)

        (pre_coreset, output_solver_state, _indices) = _reduce_coreset(dataset)
        # Correct the pre-coreset data and the indices
        coreset = eqx.tree_at(lambda x: x.pre_coreset_data, pre_coreset, dataset)
        if _indices is not None:
            if isinstance(coreset, Coresubset):
                coreset = eqx.tree_at(lambda x: x.nodes.data, coreset, _indices)
        return coreset, output_solver_state


def _jit_tree(
    dataset: _Data, leaf_size: int, tree_type: type[BinaryTree]
) -> tuple[_Data, _Indices]:
    """
    Return JIT compatible BinaryTree partitioning of 'dataset'.

    :param dataset: Input dataset from which the tree partitioning is generated
    :param leaf_size: Size of the partitions to generate; If 'leaf_size' is not an
        integer multiple of :code:`len(dataset)`, padding will be used to ensure all
        partitions have equal size.
    :param tree_type: The type of BinaryTree to use for partitioning.
    :return: The partitioned dataset
    """
    # To enable JIT compatibility, we must know the number of points per-leaf a priori.
    # Unfortunately, when the dataset size is not an integer power of two, the KDTree
    # may produce variable sized leaves where the number of points in each leaf obeys:
    # leaf_size <= n_points <= 2 * leaf_size
    # It is not obvious in this case if 'n_points' can be determined a priori.
    #
    # A simple solution is to pad the dataset with zero valued and weighted points, up
    # to the smallest integer power of two. An additional benefit of this padding is
    # that all partitions will be perfectly balanced (of identical size).
    #
    # IMPORTANT: the base solver should ignore/be invariant to these padding values;
    # ignorance is weaker condition than invariance. For example, due to implementation
    # details, the RandomSample solver will select different values when padding is
    # present, even though the probability of selecting these padded values is zero.
    # Thus the solver ignores the padded values, but the result is not invariant (it is
    # equivalent to selecting a different random key in this example)

    # Divide by two so the leaf_size is the partition size.
    _leaf_size = max(1, leaf_size // 2)
    n_leaves = 2 ** max(0, math.floor(math.log2((len(dataset) - 1) / _leaf_size)))
    padding = math.ceil(len(dataset) / n_leaves) * n_leaves - len(dataset)
    padded_dataset = tree_zero_pad_leading_axis(dataset, padding)
    shape = (n_leaves, len(padded_dataset) // n_leaves)
    result_shape = jax.ShapeDtypeStruct(shape, jnp.int32)

    def _binary_tree(_input_data: Data) -> np.ndarray:
        _, node_indices, _, _ = tree_type(
            _input_data.data, leaf_size=_leaf_size, sample_weight=_input_data.weights
        ).get_arrays()
        return node_indices.reshape(n_leaves, -1).astype(np.int32)

    indices = jax.pure_callback(_binary_tree, result_shape, padded_dataset)
    return padded_dataset[indices], jnp.arange(len(dataset))[indices]
