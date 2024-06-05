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

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike
from typing_extensions import override

from coreax.coreset import Coresubset
from coreax.data import Data, as_data
from coreax.kernel import Kernel, SteinKernel
from coreax.score_matching import KernelDensityMatching, ScoreMatching
from coreax.solvers.base import (
    CoresubsetSolver,
    ExplicitSizeSolver,
    PaddingInvariantSolver,
    RefinementSolver,
)
from coreax.util import KeyArrayLike, tree_zero_pad_leading_axis

_Data = TypeVar("_Data", bound=Data)


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


MSG = "'coreset_size' must be less than 'len(dataset)' by definition of a coreset"


def _initial_coresubset(coreset_size: int, dataset: _Data) -> Coresubset[_Data]:
    """Generate a coresubset with zero valued and weighted indices."""
    initial_coresubset_indices = Data(jnp.zeros(coreset_size, dtype=jnp.int32), 0)
    try:
        return Coresubset(initial_coresubset_indices, dataset)
    except ValueError as err:
        if len(initial_coresubset_indices) > len(dataset):
            raise ValueError(MSG) from err
        raise


def _greedy_kernel_selection(
    coresubset: Coresubset[_Data],
    selection_function: Callable[[int, ArrayLike], Array],
    output_size: int,
    kernel: Kernel,
    unique: bool,
    block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]],
    unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]],
) -> Coresubset[_Data]:
    """
    Iterative-greedy coresubset point selection loop.

    Primarily intended for use with :class`KernelHerding` and :class:`SteinThinning`.

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
    data, weights = jtu.tree_leaves(coresubset.pre_coreset_data)
    kernel_similarity_penalty = kernel.compute_mean(
        data,
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
        penalty_update = jnp.ravel(kernel.compute(data, data[updated_coreset_index]))
        updated_penalty = kernel_similarity_penalty + penalty_update
        if unique:
            # Prevent the same 'updated_coreset_index' from being selected on a
            # subsequent iteration, by setting the penalty to infinity.
            updated_penalty = updated_penalty.at[updated_coreset_index].set(jnp.inf)
        return updated_coreset_indices, updated_penalty

    init_state = (padded_coresubset.unweighted_indices, kernel_similarity_penalty)
    output_state = jax.lax.fori_loop(0, output_size, _greedy_body, init_state)
    updated_coreset_indices = output_state[0][:output_size]
    return eqx.tree_at(lambda x: x.nodes, coresubset, as_data(updated_coreset_indices))


class KernelHerding(
    RefinementSolver[_Data, HerdingState], ExplicitSizeSolver, PaddingInvariantSolver
):
    r"""
    Kernel Herding - an explicitly sized coresubset refinement solver.

    Solves the coresubset problem by taking a deterministic, iterative, and greedy
    approach to minimizing the (weighted) Maximum Mean Discrepancy (MMD) between the
    coresubset (the solution) and the problem dataset.

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
        dataset: _Data,
        solver_state: Optional[HerdingState] = None,
    ) -> tuple[Coresubset[_Data], HerdingState]:
        initial_coresubset = _initial_coresubset(self.coreset_size, dataset)
        return self.refine(initial_coresubset, solver_state)

    def refine(
        self,
        coresubset: Coresubset[_Data],
        solver_state: Optional[HerdingState] = None,
    ) -> tuple[Coresubset[_Data], HerdingState]:
        """
        Refine a coresubset with 'Kernel Herding'.

        We first compute the kernel's Gramian row-mean if it is not given in the
        'solver_state', and then iteratively swap points with the initial coreset,
        balancing selecting points in high density regions with selecting points far
        from those already in the coreset.

        :param coresubset: The coresubset to refine
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :return: A refined coresubset and relevant intermediate solver state information
        """
        if solver_state is None:
            x, bs, un = coresubset.pre_coreset_data, self.block_size, self.unroll
            gramian_row_mean = self.kernel.gramian_row_mean(x, block_size=bs, unroll=un)
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


class RPCholesky(CoresubsetSolver[_Data, RPCholeskyState], ExplicitSizeSolver):
    r"""
    Randomly Pivoted Cholesky - an explicitly sized coresubset refinement solver.

    Solves the coresubset problem by taking a stochastic, iterative, and greedy approach
    to approximating the Gramian of a given kernel, evaluated on the original dataset.

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
        self, dataset: _Data, solver_state: Optional[RPCholeskyState] = None
    ) -> tuple[Coresubset[_Data], RPCholeskyState]:
        """
        Reduce 'dataset' to a :class:`~coreax.coreset.Coresubset` with 'RPCholesky'.

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
        x = dataset.data
        if solver_state is None:
            gramian_diagonal = jax.vmap(self.kernel.compute_elementwise)(x, x)
        else:
            gramian_diagonal = solver_state.gramian_diagonal
        initial_coresubset = _initial_coresubset(self.coreset_size, dataset)
        coreset_indices = initial_coresubset.unweighted_indices
        num_data_points = len(x)

        def _greedy_body(
            i: int, val: tuple[Array, Array, Array]
        ) -> tuple[Array, Array, Array]:
            """RPCholesky iteration - Algorithm 1 of :cite:`chen2023randomly`."""
            residual_diagonal, approximation_matrix, coreset_indices = val
            key = jr.fold_in(self.random_key, i)
            pivot_point = jr.choice(
                key, num_data_points, (), p=residual_diagonal, replace=False
            )
            updated_coreset_indices = coreset_indices.at[i].set(pivot_point)
            # Remove overlap with previously chosen columns
            g = (
                self.kernel.compute(x, x[pivot_point])
                - (approximation_matrix @ approximation_matrix[pivot_point])[:, None]
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


def _convert_stein_kernel(
    x: ArrayLike, kernel: Kernel, score_matching: Union[ScoreMatching, None]
) -> SteinKernel:
    r"""
    Convert the kernel to a :class:`~coreax.kernel.SteinKernel`.

    :param x: The data used to call `score_matching.match(x)`
    :param kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`;
        if 'kernel' is a :class:`~coreax.kernel.SteinKernel` and :code:`score_matching
        is not None`, a new instance of the kernel will be generated where the score
        function is given by :code:`score_matching.match(x)`
    :param score_matching: Specifies/overwrite the score function of the implied/passed
       :class:`~coreax.kernel.SteinKernel`; if :data:`None`, default to
       :class:`~coreax.score_matching.KernelDensityMatching` unless 'kernel' is a
       :class:`~coreax.kernel.SteinKernel`, in which case the kernel's existing score
       function is used.
    :return: The (potentially) converted/updated :class:`~coreax.kernel.SteinKernel`.
    """
    if isinstance(kernel, SteinKernel):
        if score_matching is not None:
            _kernel = eqx.tree_at(
                lambda x: x.score_function, kernel, score_matching.match(x)
            )
        _kernel = kernel
    else:
        if score_matching is None:
            length_scale = getattr(kernel, "length_scale", 1.0)
            score_matching = KernelDensityMatching(length_scale)
        _kernel = SteinKernel(kernel, score_function=score_matching.match(x))
    return _kernel


class SteinThinning(
    RefinementSolver[_Data, None], ExplicitSizeSolver, PaddingInvariantSolver
):
    r"""
    Stein Thinning - an explicitly sized coresubset refinement solver.

    Solves the coresubset problem by taking a deterministic, iterative, and greedy
    approach to minimizing the Kernelized Stein Discrepancy (KSD) between the empirical
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
        initial_coresubset = _initial_coresubset(self.coreset_size, dataset)
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
        kernel = _convert_stein_kernel(x, self.kernel, self.score_matching)
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
