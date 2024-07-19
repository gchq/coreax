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

"""Solvers for constructing coresubsets."""

from collections.abc import Callable
from typing import Optional, TypeVar, Union
from warnings import warn

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike
from typing_extensions import override

from coreax.coreset import Coresubset
from coreax.data import Data, SupervisedData, as_data
from coreax.inverses import (
    LeastSquareApproximator,
    RegularisedInverseApproximator,
)
from coreax.kernel import Kernel, TensorProductKernel
from coreax.score_matching import ScoreMatching, convert_stein_kernel
from coreax.solvers.base import (
    CoresubsetSolver,
    ExplicitSizeSolver,
    PaddingInvariantSolver,
    RefinementSolver,
)
from coreax.util import (
    KeyArrayLike,
    sample_batch_indices,
    tree_zero_pad_leading_axis,
)

_Data = TypeVar("_Data", bound=Data)
_SupervisedData = TypeVar("_SupervisedData", bound=SupervisedData)

INVALID_KERNEL_DATA_COMBINATION_MSG = (
    "Invalid combination of 'kernel' and 'dataset'; if compressing"
    + " 'SupervisedData', one must pass a 'TensorProductKernel', if"
    + " compressing 'Data', one must pass a child of 'Kernel'."
)


class HerdingState(eqx.Module):
    """
    Intermediate :class:`KernelHerding` solver state information.

    :param gramian_row_mean: Cached Gramian row-mean.
    """

    gramian_row_mean: Array


class RPCholeskyState(eqx.Module):
    """
    Intermediate :class:`RPCholesky` solver state information.

    :param gramian_diagonal: Cached Gramian diagonal.
    """

    gramian_diagonal: Array


class ConditionalKernelHerdingState(eqx.Module):
    """
    Intermediate :class:`ConditionalKernelHerding` solver state information.

    :param feature_gramian: Cached feature kernel gramian
    :param response_gramian: Cached response kernel gramian
    :param training_cme: Cached array of the  Conditional Mean Embedding evaluated at
        all possible pairs of data.
    """

    feature_gramian: Array
    response_gramian: Array
    training_cme: Array


MSG = "'coreset_size' must be less than 'len(dataset)' by definition of a coreset"


def _initial_coresubset(
    fill_value: int, coreset_size: int, dataset: _Data
) -> Coresubset[_Data]:
    """Generate a coresubset with zero valued and weighted indices."""
    initial_coresubset_indices = Data(
        jnp.full((coreset_size,), fill_value, dtype=jnp.int32), 0
    )
    try:
        return Coresubset(initial_coresubset_indices, dataset)
    except ValueError as err:
        if len(initial_coresubset_indices) > len(dataset):
            raise ValueError(MSG) from err
        raise


class RandomSample(CoresubsetSolver[_Data, None], ExplicitSizeSolver):
    """
    Reduce a dataset by randomly sampling a fixed number of points.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param weighted: If to use dataset weights as selection probabilities
    :param unique: If to sample without replacement
    """

    random_key: KeyArrayLike
    weighted: bool = False
    unique: bool = True

    @override
    def reduce(
        self, dataset: _Data, solver_state: None = None
    ) -> tuple[Coresubset, None]:
        selection_weights = dataset.weights if self.weighted else None
        try:
            random_indices = jr.choice(
                self.random_key,
                len(dataset),
                (self.coreset_size,),
                p=selection_weights,
                replace=not self.unique,
            )
            return Coresubset(random_indices, dataset), solver_state
        except ValueError as err:
            if self.coreset_size > len(dataset) and self.unique:
                raise ValueError(MSG) from err
            raise


def _greedy_kernel_selection(
    coresubset: Coresubset[Data],
    selection_function: Callable[[int, ArrayLike], Array],
    output_size: int,
    kernel: Union[Kernel, TensorProductKernel],
    unique: bool,
    block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]],
    unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]],
) -> Coresubset[Data]:
    """
    Iterative-greedy coresubset point selection loop.

    Primarily intended for use with :class`_GenericDataKernelHerding` and
    :class:`SteinThinning`.

    :param coresubset: The initialisation
    :param selection_function: Greedy selection function/objective
    :param output_size: The size of the resultant coresubset
    :param kernel: The kernel used to compute the penalty
    :param unique: If each index in the resulting coresubset should be unique
    :param block_size: Block size passed to :meth:`~coreax.kernel.Kernel.compute_mean`
    :param unroll: Unroll parameter passed to :meth:`~coreax.kernel.Kernel.compute_mean`
    :return: Coresubset generated by iterative-greedy selection
    """
    # If the initialisation coresubset is too small, pad its nodes up to 'output_size'
    # with zero valued and weighted indices.
    padding = max(0, output_size - len(coresubset))
    padded_indices = tree_zero_pad_leading_axis(coresubset.nodes, padding)
    padded_coresubset = eqx.tree_at(lambda x: x.nodes, coresubset, padded_indices)
    # The kernel similarity penalty must be computed for the initial coreset. If we did
    # not support refinement, the penalty could be initialised to the zeros vector, and
    # the result would be invariant to the initial coresubset.
    dataset = coresubset.pre_coreset_data
    weights = dataset.weights
    if isinstance(kernel, TensorProductKernel) and isinstance(dataset, SupervisedData):
        x, y = dataset.data, dataset.supervision
    elif isinstance(kernel, Kernel):
        x = dataset.data
    else:
        raise ValueError(INVALID_KERNEL_DATA_COMBINATION_MSG)
    kernel_similarity_penalty = kernel.compute_mean(
        dataset,
        padded_coresubset.coreset,
        axis=1,
        block_size=block_size,
        unroll=unroll,
    )

    def _greedy_body(i: int, val: tuple[Array, Array]) -> tuple[Array, ArrayLike]:
        coreset_indices, kernel_similarity_penalty = val
        valid_kernel_similarity_penalty = jnp.where(
            weights > 0, kernel_similarity_penalty, jnp.nan
        )
        updated_coreset_index = selection_function(i, valid_kernel_similarity_penalty)
        updated_coreset_indices = coreset_indices.at[i].set(updated_coreset_index)
        if isinstance(kernel, TensorProductKernel):
            penalty_update = kernel.compute(
                (x, y), (x[updated_coreset_index], y[updated_coreset_index])
            )
        else:
            penalty_update = kernel.compute(x, x[updated_coreset_index])

        updated_penalty = kernel_similarity_penalty + jnp.ravel(penalty_update)
        if unique:
            # Prevent the same 'updated_coreset_index' from being selected on a
            # subsequent iteration, by setting the penalty to infinity.
            updated_penalty = updated_penalty.at[updated_coreset_index].set(jnp.inf)
        return updated_coreset_indices, updated_penalty

    init_state = (padded_coresubset.unweighted_indices, kernel_similarity_penalty)
    output_state = jax.lax.fori_loop(0, output_size, _greedy_body, init_state)
    updated_coreset_indices = output_state[0][:output_size]
    return eqx.tree_at(lambda x: x.nodes, coresubset, as_data(updated_coreset_indices))


class _GenericDataKernelHerding(
    RefinementSolver[Union[_Data, _SupervisedData], HerdingState],
    ExplicitSizeSolver,
    PaddingInvariantSolver,
):
    r"""
    An implementation of Kernel Herding handling (un)supervised data types.

    .. note::
        :class:`_GenericDataKernelHerding` should not be used directly, if compressing
        unsupervised :class:`~coreax.data.Data`, use :class:`KernelHerding`,
        if compressing :class:`~coreax.data.SupervisedData`, use
        :class:`JointKernelHerding`.

    :param coreset_size: The desired size of the solved coreset
    :param kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`  if
        compressing unsupervised :class:`~coreax.data.Data`, else
        :class:`~coreax.kernel.TensorProductKernel` instance if compressing
        :class:`~coreax.data.SupervisedData`
    :param unique: Boolean that ensures the resulting coresubset will only contain
        unique elements
    :param block_size: Block size passed to :meth:`~coreax.kernel.Kernel.compute_mean`
    :param unroll: Unroll parameter passed to :meth:`~coreax.kernel.Kernel.compute_mean`
    """

    kernel: Union[Kernel, TensorProductKernel]
    unique: bool = True
    block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]] = None
    unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1

    @override
    def reduce(
        self,
        dataset: Union[Data, SupervisedData],
        solver_state: Optional[HerdingState] = None,
    ) -> tuple[Coresubset[Union[Data, SupervisedData]], HerdingState]:
        initial_coresubset = _initial_coresubset(0, self.coreset_size, dataset)
        return self.refine(initial_coresubset, solver_state)

    def refine(
        self,
        coresubset: Coresubset[Union[Data, SupervisedData]],
        solver_state: Optional[HerdingState] = None,
    ) -> tuple[Coresubset[Union[Data, SupervisedData]], HerdingState]:
        """
        Refine a coresubset according to the 'Kernel Herding' loss.

        We first compute the (tensor-product) kernel's Gramian row-mean if it is not
        given in the `solver_state`, and then iteratively swap points with the initial
        coreset,  balancing selecting points in high density regions with selecting
        points far from those already in the coreset.

        :param coresubset: The coresubset to refine
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :return: A refined coresubset and relevant intermediate solver state information
        """
        if solver_state is None:
            data, bs, un = coresubset.pre_coreset_data, self.block_size, self.unroll
            gramian_row_mean = self.kernel.gramian_row_mean(
                data, block_size=bs, unroll=un
            )
        else:
            gramian_row_mean = solver_state.gramian_row_mean

        def selection_function(i: int, _kernel_similarity_penalty: ArrayLike) -> Array:
            """Greedy selection criterion - Equation 8 of :cite:`chen2012herding`."""
            return jnp.nanargmax(
                gramian_row_mean - _kernel_similarity_penalty / (i + 1)
            )

        refined_coreset = _greedy_kernel_selection(
            coresubset,
            selection_function,
            self.coreset_size,
            self.kernel,
            self.unique,
            self.block_size,
            self.unroll,
        )
        return refined_coreset, HerdingState(gramian_row_mean)


class KernelHerding(
    RefinementSolver[_Data, HerdingState], ExplicitSizeSolver, PaddingInvariantSolver
):
    r"""
    Kernel Herding - an explicitly sized coresubset refinement solver.

    Solves the coresubset problem by taking a deterministic, iterative, and greedy
    approach to minimizing the (weighted) Maximum Mean Discrepancy (MMD) between the
    coresubset (the solution) and the problem dataset.

    .. note::
        :class:`KernelHerding` is suitable for compressing unsupervised
        :class:`~coreax.data.Data`, use :class:`JointKernelHerding`, if compressing
        :class:`~coreax.data.SupervisedData`.

    Given one has selected :math:`T` data points for their compressed representation of
    the original dataset, kernel herding selects the next point using Equation 8 of
    :cite:`chen2012herding`:

    .. math::

        x_{T+1} = \arg\max_{x} \left( \mathbb{E}[k(x, x')] -
            \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) \right)

    where :math:`k` is the kernel used, the expectation :math:`\mathbb{E}` is taken over
    the entire dataset, and the search is over the entire dataset. This can informally
    be seen as a balance between using points at which the underlying density is high
    (the first term) and exploration of distinct regions of the space (the second term).

    :param coreset_size: The desired size of the solved coreset
    :param kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param unique: Boolean that ensures the resulting coresubset will only contain
        unique elements
    :param block_size: Block size passed to :meth:`~coreax.kernel.Kernel.compute_mean`
    :param unroll: Unroll parameter passed to :meth:`~coreax.kernel.Kernel.compute_mean`
    """

    kernel: Kernel
    unique: bool = True
    block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]] = None
    unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1

    @override
    def reduce(
        self,
        dataset: Data,
        solver_state: Optional[HerdingState] = None,
    ) -> tuple[Coresubset[Data], HerdingState]:
        initial_coresubset = _initial_coresubset(0, self.coreset_size, dataset)
        return self.refine(initial_coresubset, solver_state)

    def refine(
        self,
        coresubset: Coresubset[Data],
        solver_state: Optional[HerdingState] = None,
    ) -> tuple[Coresubset[Data], HerdingState]:
        """
        Refine a coresubset with :class:`KernelHerding`.

        We first compute the kernel's Gramian row-mean if it is not given in the
        `solver_state`, and then iteratively swap points with the initial coreset,
        balancing selecting points in high density regions with selecting points far
        from those already in the coreset.

        :param coresubset: The coresubset to refine
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :return: A refined coresubset and relevant intermediate solver state information
        """
        unsupervised_solver = _GenericDataKernelHerding(
            coreset_size=self.coreset_size,
            kernel=self.kernel,
            unique=self.unique,
            block_size=self.block_size,
            unroll=self.unroll,
        )
        return unsupervised_solver.refine(coresubset, solver_state)


class JointKernelHerding(
    RefinementSolver[_SupervisedData, HerdingState],
    ExplicitSizeSolver,
    PaddingInvariantSolver,
):
    r"""
    Joint Kernel Herding - an explicitly sized coresubset refinement solver.

    Given a supervised dataset consisting of pairs of features
    :math:`x \in \mathbb{R}^d`and responses :math:`x \in \mathbb{R}^p` we solve the
    coresubset problem by taking a deterministic, iterative, and greedy approach to
    minimizing the (weighted) Joint Maximum Mean Discrepancy (JMMD) between the
    coresubset (the solution) and the problem supervised dataset.

    .. note::
        :class:`JointKernelHerding` is suitable for compressing
        :class:`~coreax.data.SupervisedData`, use :class:`KernelHerding`, if compressing
        unsupervised :class:`~coreax.data.Data`.

    Given one has selected :math:`T` data pairs for their compressed representation of
    the original dataset, joint kernel herding selects the next pair using:

    .. math::

        (x, y)_{T+1} = \arg\max_{(x, y)} \left(
            \mathbb{E}_{(x', y') \sim \mathbb{P}(X, Y)}[k((x, y), (x', y'))] -
            \frac{1}{T+1}\sum_{t=1}^T k((x, y), (x, y)_t) \right)

    where :math:`k` is the tensor-product kernel used, the expectation
    :math:`\mathbb{E}` is taken over the entire dataset, and the search is over the
    entire dataset. This can informally be seen as a balance between using points at
    which the underlying joint density is high (the first term) and exploration of
    distinct regions of the space (the second term).

    :param coreset_size: The desired size of the solved coreset
    :param feature_kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` on
        the feature space
    :param response_kernel: :class:`~coreax.kernel.Kernel` instance implementing a
        kernel function :math:`k: \mathbb{R}^p \times \mathbb{R}^p \rightarrow
        \mathbb{R}` on the response space
    :param unique: Boolean that ensures the resulting coresubset will only contain
        unique elements
    :param block_size: Block size passed to :meth:`~coreax.kernel.Kernel.compute_mean`
    :param unroll: Unroll parameter passed to :meth:`~coreax.kernel.Kernel.compute_mean`
    """

    kernel: TensorProductKernel
    unique: bool = True
    block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]] = None
    unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1

    def __init__(
        self,
        coreset_size: int,
        feature_kernel: Kernel,
        response_kernel: Kernel,
        unique: bool = True,
        block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]] = None,
        unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
    ):
        """Initialise `JointKernelHerding` class and build tensor-product kernel."""
        ExplicitSizeSolver.__init__(self, coreset_size)
        self.kernel = TensorProductKernel(
            feature_kernel=feature_kernel,
            response_kernel=response_kernel,
        )
        self.unique = unique
        self.block_size = block_size
        self.unroll = unroll

    @override
    def reduce(
        self,
        dataset: SupervisedData,
        solver_state: Optional[HerdingState] = None,
    ) -> tuple[Coresubset[SupervisedData], HerdingState]:
        initial_coresubset = _initial_coresubset(0, self.coreset_size, dataset)
        return self.refine(initial_coresubset, solver_state)

    def refine(
        self,
        coresubset: Coresubset[SupervisedData],
        solver_state: Optional[HerdingState] = None,
    ) -> tuple[Coresubset[SupervisedData], HerdingState]:
        """
        Refine a coresubset with :class:`JointKernelHerding`.

        We first compute the kernel's Gramian row-mean if it is not given in the
        `solver_state`, and then iteratively swap points with the initial coreset,
        balancing selecting points in high density regions with selecting points far
        from those already in the coreset.

        :param coresubset: The coresubset to refine
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :return: A refined coresubset and relevant intermediate solver state information
        """
        supervised_solver = _GenericDataKernelHerding(
            coreset_size=self.coreset_size,
            kernel=self.kernel,
            unique=self.unique,
            block_size=self.block_size,
            unroll=self.unroll,
        )
        return supervised_solver.refine(coresubset, solver_state)


class _GenericDataRPCholesky(
    CoresubsetSolver[Union[_Data, _SupervisedData], RPCholeskyState], ExplicitSizeSolver
):
    r"""
    An implementation of Randomly Pivoted Cholesky handling (un)supervised data types.

    Solves the coresubset problem by taking a stochastic, iterative, and greedy approach
    to approximating the Gramian of a given kernel, evaluated on the original dataset.

    .. note::
        :class:`_GenericDataRPCholesky` should not be used directly, if compressing
        unsupervised :class:`~coreax.data.Data`, use :class:`RPCholesky`, if compressing
        :class:`~coreax.data.SupervisedData`, use :class:`JointRPCholesky`.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` if
        compressing unsupervised :class:`~coreax.data.Data`, else
        :class:`~coreax.kernel.TensorProductKernel` instance if compressing
        :class:`~coreax.data.SupervisedData`
    :param unique: If each index in the resulting coresubset should be unique
    """

    random_key: KeyArrayLike
    kernel: Union[Kernel, TensorProductKernel]
    unique: bool = True

    def reduce(
        self,
        dataset: Union[Data, SupervisedData],
        solver_state: Optional[RPCholeskyState] = None,
    ) -> tuple[Coresubset[Union[Data, SupervisedData]], RPCholeskyState]:
        """
        Reduce 'dataset' to a :class:`~coreax.coreset.Coresubset`.

        This is done by first computing the (tensor-product) kernel Gram matrix of the
        original data, and isolating the diagonal of this. A 'pivot point' is then
        sampled, where sampling probabilities correspond to the size of the elements on
        this diagonal. The data-point corresponding to this pivot point is added to the
        coreset, and the diagonal of the Gram matrix is updated to add a repulsion term
        of sorts - encouraging the coreset to select a range of distinct points in the
        original data. The pivot sampling and diagonal updating steps are repeated until
        :math:`M` points have been selected.

        :param dataset: The dataset to reduce to a coresubset
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :return: a refined coresubset and relevant intermediate solver state information
        """
        # Setup the class to deal with supervised or unsupervised data
        if solver_state is not None:
            gramian_diagonal = solver_state.gramian_diagonal
        # pylint: disable=unidiomatic-typecheck
        if (
            isinstance(self.kernel, TensorProductKernel)
            and type(dataset) is SupervisedData
        ):
            x, y = dataset.data, dataset.supervision
            if solver_state is None:
                gramian_diagonal = jax.vmap(self.kernel.compute_elementwise)(
                    (x, y), (x, y)
                )
        elif isinstance(self.kernel, Kernel) and type(dataset) is Data:
            x = dataset.data
            if solver_state is None:
                gramian_diagonal = jax.vmap(self.kernel.compute_elementwise)(x, x)
        else:
            raise ValueError(INVALID_KERNEL_DATA_COMBINATION_MSG)
        # pylint: enable=unidiomatic-typecheck

        initial_coresubset = _initial_coresubset(0, self.coreset_size, dataset)
        coreset_indices = initial_coresubset.unweighted_indices
        num_data_points = len(x)

        def _greedy_body(
            i: int, val: tuple[Array, Array, Array]
        ) -> tuple[Array, Array, Array]:
            """RPCholesky Iteration - Algorithm 1 of :cite:`chen2023randomly`."""
            residual_diagonal, approximation_matrix, coreset_indices = val
            key = jr.fold_in(self.random_key, i)
            pivot_point = jr.choice(
                key, num_data_points, (), p=residual_diagonal, replace=False
            )
            updated_coreset_indices = coreset_indices.at[i].set(pivot_point)
            # Remove overlap with previously chosen columns
            if isinstance(self.kernel, TensorProductKernel):
                g = (
                    self.kernel.compute((x, y), (x[pivot_point], y[pivot_point]))
                    - (approximation_matrix @ approximation_matrix[pivot_point])[
                        :, None
                    ]
                )
            else:
                g = (
                    self.kernel.compute(x, x[pivot_point])
                    - (approximation_matrix @ approximation_matrix[pivot_point])[
                        :, None
                    ]
                )
            updated_approximation_matrix = approximation_matrix.at[:, i].set(
                jnp.ravel(g / jnp.sqrt(g[pivot_point]))
            )
            # Track diagonal of residual matrix and ensure it remains non-negative
            updated_residual_diagonal = jnp.clip(
                residual_diagonal - jnp.square(approximation_matrix[:, i]), min=0
            )
            if self.unique:
                # Ensures that index selected_pivot_point can't be drawn again in future
                updated_residual_diagonal = updated_residual_diagonal.at[
                    pivot_point
                ].set(0.0)
            return (
                updated_residual_diagonal,
                updated_approximation_matrix,
                updated_coreset_indices,
            )

        approximation_matrix = jnp.zeros((num_data_points, self.coreset_size))
        init_state = (gramian_diagonal, approximation_matrix, coreset_indices)
        output_state = jax.lax.fori_loop(0, self.coreset_size, _greedy_body, init_state)
        _, _, updated_coreset_indices = output_state
        updated_coreset = Coresubset(updated_coreset_indices, dataset)
        return updated_coreset, RPCholeskyState(gramian_diagonal)


class RPCholesky(CoresubsetSolver[_Data, RPCholeskyState], ExplicitSizeSolver):
    r"""
    Randomly Pivoted Cholesky - an explicitly sized coresubset refinement solver.

    Solves the coresubset problem by taking a stochastic, iterative, and greedy approach
    to approximating the Gramian of a given kernel, evaluated on the original dataset.

    .. note::
        :class:`RPCholesky` is suitable for compressing unsupervised
        :class:`~coreax.data.Data`, use :class:`JointRPCholesky`, if compressing
        :class:`~coreax.data.SupervisedData`.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param unique: If each index in the resulting coresubset should be unique
    """

    random_key: KeyArrayLike
    kernel: Kernel
    unique: bool = True

    def reduce(
        self, dataset: Data, solver_state: Optional[RPCholeskyState] = None
    ) -> tuple[Coresubset[Data], RPCholeskyState]:
        """
        Reduce `dataset` to a :class:`~coreax.coreset.Coresubset` with `RPCholesky`.

        This is done by first computing the kernel Gram matrix of the original data, and
        isolating the diagonal of this. A 'pivot point' is then sampled, where sampling
        probabilities correspond to the size of the elements on this diagonal. The
        data-point corresponding to this pivot point is added to the coreset, and the
        diagonal of the Gram matrix is updated to add a repulsion term of sorts -
        encouraging the coreset to select a range of distinct points in the original
        data. The pivot sampling and diagonal updating steps are repeated until
        :math:`M` points have been selected.

        :param dataset: The dataset to reduce to a coresubset
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :return: a refined coresubset and relevant intermediate solver state information
        """
        unsupervised_solver = _GenericDataRPCholesky(
            coreset_size=self.coreset_size,
            random_key=self.random_key,
            kernel=self.kernel,
            unique=self.unique,
        )
        return unsupervised_solver.reduce(dataset, solver_state)


class JointRPCholesky(
    CoresubsetSolver[_SupervisedData, RPCholeskyState], ExplicitSizeSolver
):
    r"""
    Joint Randomly Pivoted Cholesky - an explicitly sized coresubset refinement solver.

    Solves the coresubset problem by taking a stochastic, iterative, and greedy approach
    to approximating the Gramian of a tensor-product kernel, evaluated on the original
    supervised dataset.

    .. note::
        :class:`JointRPCholesky` is suitable for compressing
        :class:`~coreax.data.SupervisedData`, use :class:`RPCholesky`, if compressing
        unsupervised :class:`~coreax.data.Data`.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param feature_kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` on
        the feature space
    :param response_kernel: :class:`~coreax.kernel.Kernel` instance implementing a
        kernel function :math:`k: \mathbb{R}^p \times \mathbb{R}^p \rightarrow
        \mathbb{R}` on the response space
    :param unique: If each index in the resulting coresubset should be unique
    """

    random_key: KeyArrayLike
    kernel: TensorProductKernel
    unique: bool = True

    def __init__(
        self,
        coreset_size: int,
        random_key: KeyArrayLike,
        feature_kernel: Kernel,
        response_kernel: Kernel,
        unique: bool = True,
    ):
        """Initialise JointRPCholesky class and build tensor-product kernel."""
        ExplicitSizeSolver.__init__(self, coreset_size)
        self.random_key = random_key
        self.kernel = TensorProductKernel(
            feature_kernel=feature_kernel,
            response_kernel=response_kernel,
        )
        self.unique = unique

    def reduce(
        self, dataset: SupervisedData, solver_state: Optional[RPCholeskyState] = None
    ) -> tuple[Coresubset[SupervisedData], RPCholeskyState]:
        """
        Reduce `dataset` to :class:`~coreax.coreset.Coresubset` with `JointRPCholesky`.

        This is done by first computing the tensor-product kernel Gram matrix of the
        original supervised data pairs, and isolating the diagonal of this. A
        'pivot point' is then sampled, where sampling probabilities correspond to the
        size of the elements on this diagonal. The data-point corresponding to this
        pivot point is added to the coreset, and the diagonal of the Gram matrix is
        updated to add a repulsion term of sorts - encouraging the coreset to select a
        range of distinct pairs in the original data. The pivot sampling and diagonal
        updating steps are repeated until :math:`M` pairs have been selected.

        :param dataset: The dataset to reduce to a coresubset
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :return: a refined coresubset and relevant intermediate solver state information
        """
        supervised_solver = _GenericDataRPCholesky(
            coreset_size=self.coreset_size,
            random_key=self.random_key,
            kernel=self.kernel,
            unique=self.unique,
        )
        return supervised_solver.reduce(dataset, solver_state)


class SteinThinning(
    RefinementSolver[_Data, None], ExplicitSizeSolver, PaddingInvariantSolver
):
    r"""
    Stein Thinning - an explicitly sized coresubset refinement solver.

    Solves the coresubset problem by taking a deterministic, iterative, and greedy
    approach to minimizing the Kernelised Stein Discrepancy (KSD) between the empirical
    distribution of the coresubset (the solution) and the distribution of the problem
    dataset, as characterised by the score function of the Stein Kernel.

    Given one has selected :math:`T` data points for their compressed representation of
    the original dataset, (regularised) Stein thinning selects the next point using the
    equations in Section 3.1 of :cite:`benard2023kernel`:

    .. math::

        x_{T+1} = \arg\min_{x} \left( k_P(x, x) / 2 + \Delta^+ \log p(x) -
            \lambda T \log p(x) + \frac{1}{T+1}\sum_{t=1}^T k_P(x, x_t) \right)

    where :math:`k` is the Stein kernel induced by the supplied base kernel,
    :math:`\Delta^+` is the non-negative Laplace operator, :math:`\lambda` is a
    regularisation parameter, and the search is over the entire dataset.

    :param coreset_size: The desired size of the solved coreset
    :param kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`;
        if 'kernel' is a :class:`~coreax.kernel.SteinKernel` and :code:`score_matching
        is not None`, a new instance of the kernel will be generated where the score
        function is given by :code:`score_matching.match(...)`
    :param score_matching: Specifies/overwrite the score function of the implied/passed
       :class:`~coreax.kernel.SteinKernel`; if :data:`None`, default to
       :class:`~coreax.score_matching.KernelDensityMatching` unless 'kernel' is a
       :class:`~coreax.kernel.SteinKernel`, in which case the kernel's existing score
       function is used.
    :param unique: If each index in the resulting coresubset should be unique
    :param regularise: Boolean that enforces regularisation, see Section 3.1 of
        :cite:`benard2023kernel`.
    :param block_size: Block size passed to :meth:`~coreax.kernel.Kernel.compute_mean`
    :param unroll: Unroll parameter passed to :meth:`~coreax.kernel.Kernel.compute_mean`
    """

    kernel: Kernel
    score_matching: Optional[ScoreMatching] = None
    unique: bool = True
    regularise: bool = True
    block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]] = None
    unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1

    @override
    def reduce(
        self, dataset: _Data, solver_state: None = None
    ) -> tuple[Coresubset[_Data], None]:
        initial_coresubset = _initial_coresubset(0, self.coreset_size, dataset)
        return self.refine(initial_coresubset, solver_state)

    def refine(
        self, coresubset: Coresubset[_Data], solver_state: None = None
    ) -> tuple[Coresubset[_Data], None]:
        r"""
        Refine a coresubset with 'Stein thinning'.

        We first compute a score function, and then the Stein kernel. This is used to
        greedily choose points in the coreset to minimise kernel Stein discrepancy
        (KSD).

        :param coresubset: The coresubset to refine
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :return: a refined coresubset and relevant intermediate solver state information
        """
        x, w_x = jtu.tree_leaves(coresubset.pre_coreset_data)
        kernel = convert_stein_kernel(x, self.kernel, self.score_matching)
        stein_kernel_diagonal = jax.vmap(self.kernel.compute_elementwise)(x, x)
        if self.regularise:
            # Cannot guarantee that kernel.base_kernel has a 'length_scale' attribute
            bandwidth_method = getattr(kernel.base_kernel, "length_scale", None)
            kde = jsp.stats.gaussian_kde(x.T, weights=w_x, bw_method=bandwidth_method)
            # Use regularisation parameter suggested in :cite:`benard2023kernel`
            regulariser_lambda = 1 / len(coresubset)
            regularised_log_pdf = regulariser_lambda * kde.logpdf(x.T)

            @jax.vmap
            def _laplace_positive(x_):
                r"""Evaluate Laplace positive operator  :math:`\Delta^+ \log p(x)`."""
                hessian = jax.jacfwd(kernel.score_function)(x_)
                return jnp.clip(jnp.diag(hessian), min=0.0).sum()

            laplace_correction = _laplace_positive(x)
        else:
            laplace_correction, regularised_log_pdf = 0.0, 0.0

        def selection_function(i: int, _kernel_similarity_penalty: ArrayLike) -> Array:
            """
            Greedy selection criterion - Section 3.1 :cite:`benard2023kernel`.

            Argmin of the Laplace corrected and regularised Kernel Stein Discrepancy.
            """
            ksd = stein_kernel_diagonal + 2.0 * _kernel_similarity_penalty
            return jnp.nanargmin(ksd + laplace_correction - i * regularised_log_pdf)

        refined_coreset = _greedy_kernel_selection(
            coresubset,
            selection_function,
            self.coreset_size,
            self.kernel,
            self.unique,
            self.block_size,
            self.unroll,
        )
        return refined_coreset, solver_state


def _conditional_herding_loss(
    candidate_coresets: Array,
    feature_gramian: Array,
    response_gramian: Array,
    training_cme: Array,
    regularisation_parameter: float,
    identity: Array,
    inverse_approximator: RegularisedInverseApproximator,
) -> Array:
    """
    Given an array of candidate coreset indices, compute the CMMD loss for each.

    Primarily intended for use with :class`ConditionalKernelHerding.

    :param candidate_coresets: Array of indices representing all possible "next"
        coresets
    :param feature_gramian: Feature kernel gramian
    :param response_gramian: Response kernel gramian
    :param training_cme: CME evaluated at all possible pairs of training data
    :param regularisation_parameter: Regularisation parameter for stable inversion of
        array, negative values will be converted to positive
    :param identity: Block "identity" matrix
    :param inverse_approximator: Instance of
        :class:`coreax.inverses.RegularisedInverseApproximator`

    :return: ConditionalKernelHerding loss for each candidate coreset
    """
    # Extract all the possible "next" coreset arrays
    extract_indices = (candidate_coresets[:, :, None], candidate_coresets[:, None, :])
    coreset_feature_gramians = feature_gramian[extract_indices]
    coreset_response_gramians = response_gramian[extract_indices]
    coreset_cmes = training_cme[extract_indices]

    # Invert the coreset feature gramians
    inverse_coreset_feature_gramians = inverse_approximator.approximate_stack(
        coreset_feature_gramians,
        regularisation_parameter,
        identity,
    )

    # Compute the loss function. As the trace is a cyclic operation i.e. Tr(ABC) =
    # Tr(CAB) = Tr(BCA), the order of the matrix multiplications is chosen to reduce
    # numerical instability by avoiding taking a product of a matrix with its
    # own regularised inverse.
    term_2s = jnp.trace(
        inverse_coreset_feature_gramians
        @ coreset_response_gramians
        @ inverse_coreset_feature_gramians
        @ coreset_feature_gramians,
        axis1=1,
        axis2=2,
    )
    term_3s = jnp.trace(
        coreset_cmes @ inverse_coreset_feature_gramians, axis1=1, axis2=2
    )
    return term_2s - 2 * term_3s


# pylint: disable=too-many-locals
class ConditionalKernelHerding(
    RefinementSolver[_SupervisedData, ConditionalKernelHerdingState], ExplicitSizeSolver
):
    r"""
    Apply ConditionalKernelHerding to a supervised dataset.

    ConditionalKernelHerding is a deterministic, iterative and greedy approach to build
    a coreset.

    Given one has an original dataset :math:`\mathcal{D}^{(1)} = \{(x_i, y_i)\}_{i=1}^n`
    of ``n`` pairs with :math:`x\in\mathbb{R}^d` and :math:`y\in\mathbb{R}^p`, and one
    has selected :math:`m` data pairs :math:`\mathcal{D}^{(2)} = \{(\tilde{x}_i,
    \tilde{y}_i)\}_{i=1}^m` already for their compressed representation of the original
    dataset, ConditionalKernelHerding selects the next point to minimise the conditional
    maximum mean discrepancy (CMMD):

    .. math::

        \text{CMMD}^2(\mathcal{D}^{(1)}, \mathcal{D}^{(2)}) = ||\hat{\mu}^{(1)} -
        \hat{\mu}^{(2)}||^2_{\mathcal{H}_k \otimes \mathcal{H}_l}

    where :math:`\hat{\mu}^{(1)},\hat{\mu}^{(2)}` are the conditional mean embeddings
    estimated with :math:`\mathcal{D}^{(1)}` and :math:`\mathcal{D}^{(2)}` respectively,
    and :math:`\mathcal{H}_k,\mathcal{H}_l` are the RKHSs corresponding to the kernel
    functions :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` and
    :math:`l: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}` respectively.
    The search is performed over the entire dataset, or optionally over random batches
    at each iteration.

    This class works with all children of :class:`~coreax.kernel.Kernel`. Note that
    ConditionalKernelHerding does not support non-uniform weights and will only return
    coresubsets with uniform weights.

    .. note::
        When requesting a unique coresubset there is a non-vanishing chance that the
        returned coresubset will not be unique if `batch_size` is not large enough. In
        this case `batch_size` should be increased or a larger coresubset should be
        requested which can be made unique post-construction with `jnp.unique`.

    :param random_key: Key for random number generation
    :param feature_kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
        on the feature space
    :param response_kernel: :class:`~coreax.kernel.Kernel` instance implementing a
        kernel function :math:`k: \mathbb{R}^p \times \mathbb{R}^p \rightarrow
        \mathbb{R}` on the response space
    :param regularisation_parameter: Regularisation parameter for stable inversion of
        feature gram matrix
    :param unique: Boolean that enforces the resulting coreset will only contain
        unique elements
    :param batch_size: An integer representing the size of the batches of data pairs
        sampled at each iteration for consideration for adding to the coreset
    :param inverse_approximator: Instance of
        :class:`coreax.inverses.RegularisedInverseApproximator`, default value of
        :data:`None` uses :class:`coreax.inverses.LeastSquareApproximator`
    """

    random_key: KeyArrayLike
    feature_kernel: Kernel
    response_kernel: Kernel
    regularisation_parameter: float = 1e-6
    unique: bool = True
    batch_size: Union[int, None] = None
    inverse_approximator: Optional[RegularisedInverseApproximator] = None

    @override
    def reduce(
        self,
        dataset: _SupervisedData,
        solver_state: Union[ConditionalKernelHerdingState, None] = None,
    ) -> tuple[Coresubset[_SupervisedData], ConditionalKernelHerdingState]:
        initial_coresubset = _initial_coresubset(-1, self.coreset_size, dataset)
        return self.refine(initial_coresubset, solver_state)

    def refine(  # noqa: PLR0915
        self,
        coresubset: Coresubset[_SupervisedData],
        solver_state: Union[ConditionalKernelHerdingState, None] = None,
    ) -> tuple[Coresubset[_SupervisedData], ConditionalKernelHerdingState]:
        """
        Refine a coresubset with 'ConditionalKernelHerding'.

        We first compute the various factors if they are not given in the
        'solver_state', and then iteratively swap points with the initial coreset,
        selecting points which minimise the CMMD.

        :param coresubset: The coresubset to refine
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :return: A refined coresubset and relevant intermediate solver state information
        """
        # Handle default value of None
        if self.inverse_approximator is None:
            inverse_approximator = LeastSquareApproximator(self.random_key)
        else:
            inverse_approximator = self.inverse_approximator

        # If the initialisation coresubset is too small, pad its nodes up to
        # 'output_size' with -1 valued indices. If it is too large, raise a warning and
        # clip off the indices at the end.
        if self.coreset_size > len(coresubset):
            pad_size = max(0, self.coreset_size - len(coresubset))
            pad_indices = -1 * jnp.ones(pad_size, dtype=jnp.int32)
            coreset_indices = jnp.hstack((coresubset.unweighted_indices, pad_indices))
        elif self.coreset_size < len(coresubset):
            warn(
                "Requested coreset size is smaller than input 'coresubset', clipping"
                + " to the correct size and proceeding...",
                Warning,
                stacklevel=2,
            )
            coreset_indices = coresubset.unweighted_indices[: self.coreset_size]
        else:
            coreset_indices = coresubset.unweighted_indices

        dataset = coresubset.pre_coreset_data
        num_data_pairs = len(dataset)
        if solver_state is None:
            x, y = dataset.data, dataset.supervision

            feature_gramian = self.feature_kernel.compute(x, x)
            response_gramian = self.response_kernel.compute(y, y)

            inverse_feature_gramian = inverse_approximator.approximate(
                array=feature_gramian,
                regularisation_parameter=self.regularisation_parameter,
                identity=jnp.eye(num_data_pairs),
            )

            # Evaluate conditional mean embedding (CME) at all possible pairs of the
            # available training data.
            training_cme = feature_gramian @ inverse_feature_gramian @ response_gramian

            # Pad the gramians and training CME evaluations with zeros in an additional
            # column and row to allow us to extract sub-arrays and fill in elements with
            # zeros simultaneously.
            feature_gramian = jnp.pad(feature_gramian, [(0, 1)], mode="constant")
            response_gramian = jnp.pad(response_gramian, [(0, 1)], mode="constant")
            training_cme = jnp.pad(training_cme, [(0, 1)], mode="constant")

        else:
            feature_gramian = solver_state.feature_gramian
            response_gramian = solver_state.response_gramian
            training_cme = solver_state.training_cme

        # Sample the indices to be considered at each iteration ahead of time.
        if self.batch_size is not None and self.batch_size < num_data_pairs:
            batch_size = self.batch_size
        else:
            batch_size = num_data_pairs
        batch_indices = sample_batch_indices(
            random_key=self.random_key,
            max_index=num_data_pairs,
            batch_size=batch_size,
            num_batches=self.coreset_size,
        )

        # Insert coreset indices into the batch indices to avoid degrading the CMMD
        # (only has an effect when refining)
        if self.batch_size is not None:
            batch_indices = batch_indices.at[:, -1].set(coreset_indices)

        # Initialise an array that will let us extract and build rectangular arrays
        # which stack arrays corresponding to every possible candidate coreset.
        initial_candidate_coresets = (
            jnp.tile(coreset_indices, (batch_size, 1)).at[:, 0].set(batch_indices[0, :])
        )

        # Adaptively initialise an "identity matrix". For reduction we need this to
        # be a zeros matrix which we will iteratively add ones to on the diagonal. While
        # for refinement we need an actual identity matrix.
        identity_helper = jnp.hstack((jnp.ones(num_data_pairs), jnp.array([0])))
        identity = jnp.diag(identity_helper[coreset_indices])

        def _greedy_body(
            i: int, val: tuple[Array, Array, Array]
        ) -> tuple[Array, Array, Array]:
            """Execute main loop of ConditionalKernelHerding."""
            coreset_indices, identity, candidate_coresets = val

            # Update the identity matrix to allow for sub-array inversion in the
            # case of reduction (no effect when refining)
            updated_identity = identity.at[i, i].set(1)

            # Compute the loss corresponding to each candidate coreset. Note that we do
            # not compute the first term of the CMMD as is an invariant quantity wrt
            # the coreset.
            loss = _conditional_herding_loss(
                candidate_coresets,
                feature_gramian,
                response_gramian,
                training_cme,
                self.regularisation_parameter,
                updated_identity,
                inverse_approximator,
            )

            # Find the optimal replacement coreset index, ensuring we don't pick an
            # already chosen point if we want the indices to be unique.
            if self.unique:
                already_chosen_indices_mask = jnp.isin(
                    candidate_coresets[:, i],
                    coreset_indices.at[i].set(-1),
                )
                loss += jnp.where(already_chosen_indices_mask, jnp.inf, 0)
            index_to_include_in_coreset = candidate_coresets[loss.argmin(), i]

            # Repeat the chosen coreset index into the ith column of the array of
            # candidate coreset indices and replace the (i+1)th column with the next
            # batch of possible coreset indices.
            next_candidate_coresets = jnp.hstack(
                (
                    jnp.tile(index_to_include_in_coreset, (batch_size, 1)),
                    batch_indices[[i + 1], :].T,
                )
            )
            updated_candidate_coresets = candidate_coresets.at[:, [i, i + 1]].set(
                next_candidate_coresets
            )

            # Add the chosen coreset index to the current coreset indices
            updated_coreset_indices = coreset_indices.at[i].set(
                index_to_include_in_coreset
            )
            return updated_coreset_indices, updated_identity, updated_candidate_coresets

        # Greedily refine coreset points
        updated_coreset_indices, _, _ = jax.lax.fori_loop(
            lower=0,
            upper=self.coreset_size,
            body_fun=_greedy_body,
            init_val=(
                coreset_indices,
                identity,
                initial_candidate_coresets,
            ),
        )

        return Coresubset(
            updated_coreset_indices, dataset
        ), ConditionalKernelHerdingState(
            feature_gramian, response_gramian, training_cme
        )


# pylint: enable=too-many-locals
