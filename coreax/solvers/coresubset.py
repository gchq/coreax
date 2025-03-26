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

import math
from collections.abc import Callable
from typing import Literal, Optional, TypeVar, Union, overload
from warnings import warn

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import jax.tree_util as jtu
from jax import lax
from jaxtyping import Array, ArrayLike, Bool, Float, Scalar, Shaped
from typing_extensions import override

from coreax.coreset import Coresubset
from coreax.data import Data, SupervisedData, as_data
from coreax.kernels import ScalarValuedKernel
from coreax.least_squares import (
    MinimalEuclideanNormSolver,
    RegularisedLeastSquaresSolver,
)
from coreax.metrics import MMD
from coreax.score_matching import ScoreMatching, convert_stein_kernel
from coreax.solvers.base import (
    CoresubsetSolver,
    ExplicitSizeSolver,
    PaddingInvariantSolver,
    RefinementSolver,
)
from coreax.util import KeyArrayLike, sample_batch_indices, tree_zero_pad_leading_axis

_Data = TypeVar("_Data", Data, SupervisedData)
_SupervisedData = TypeVar("_SupervisedData", bound=SupervisedData)


class SizeWarning(Warning):
    """Custom warning to be raised when some parameter shape is too large."""


MSG = "'coreset_size' must be less than 'len(dataset)' by definition of a coreset"


def _initial_coresubset(
    fill_value: int, coreset_size: int, dataset: _Data
) -> Coresubset[_Data]:
    """
    Generate a coresubset with uniform indices, all zero-weighted.

    :param fill_value: Index of dataset to select as every element in coresubset.
    :param coreset_size: Size of coreset.
    :param dataset: Original dataset.
    :return: Coresubset of requested size with all indices set to ``fill_value``.
    """
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
            return Coresubset(Data(random_indices), dataset), solver_state
        except ValueError as err:
            if self.coreset_size > len(dataset) and self.unique:
                raise ValueError(MSG) from err
            raise


class HerdingState(eqx.Module):
    """
    Intermediate :class:`KernelHerding` solver state information.

    :param gramian_row_mean: Cached Gramian row-mean.
    """

    gramian_row_mean: Array


def _greedy_kernel_selection(
    coresubset: Coresubset[_Data],
    selection_function: Callable[[int, Shaped[Array, " n"], Scalar], Scalar],
    output_size: int,
    kernel: ScalarValuedKernel,
    unique: bool,
    block_size: Optional[Union[int, tuple[Optional[int], Optional[int]]]],
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
    :param block_size: Block size passed to
        :meth:`~coreax.kernels.ScalarValuedKernel.compute_mean`
    :param unroll: Unroll parameter passed to
        :meth:`~coreax.kernels.ScalarValuedKernel.compute_mean`
    :return: Coresubset generated by iterative-greedy selection
    """
    # If the initialisation coresubset is too small, pad its indices up to 'output_size'
    # with zero valued and weighted indices.
    padding = max(0, output_size - len(coresubset))
    padded_indices = tree_zero_pad_leading_axis(coresubset.indices, padding)
    padded_coresubset = eqx.tree_at(lambda x: x.indices, coresubset, padded_indices)

    # Calculate the actual size of the provided `coresubset` assuming 0-weighted
    # indices are not included
    coreset_weights = padded_coresubset.indices.weights
    init_coreset_size = jnp.sum(jnp.greater(coreset_weights, 0))

    # The kernel similarity penalty must be computed for the initial coreset. If we did
    # not support refinement, the penalty could be initialised to the zeroes vector, and
    # the result would be invariant to the initial coresubset.
    data, data_weights = jtu.tree_leaves(coresubset.pre_coreset_data)
    init_kernel_similarity_penalty = init_coreset_size * kernel.compute_mean(
        data,
        padded_coresubset.points,
        axis=1,
        block_size=block_size,
        unroll=unroll,
    )

    def _greedy_body(
        i: int,
        val: tuple[Shaped[Array, " coreset_size"], Shaped[Array, " n"], Scalar],
    ) -> tuple[Shaped[Array, " coreset_size"], Shaped[Array, " n"], Scalar]:
        coreset_indices, kernel_similarity_penalty, coreset_size = val

        # If the current coreset element is being replaced (non-zero weight),
        # subtract its contribution to the penalty
        penalty_update = jnp.ravel(kernel.compute(data, data[coreset_indices[i]]))
        kernel_similarity_penalty = jnp.where(
            coreset_weights[i] > 0,
            kernel_similarity_penalty - penalty_update,
            kernel_similarity_penalty,
        )

        # Increment the coreset size if we are adding new elements
        coreset_size = jnp.where(coreset_weights[i] > 0, coreset_size, coreset_size + 1)

        # Select the next coreset element and update penalty
        valid_kernel_similarity_penalty = jnp.where(
            data_weights > 0, kernel_similarity_penalty, jnp.nan
        )
        if unique:  # Temporarily exclude other coreset members from being selected
            valid_kernel_similarity_penalty = valid_kernel_similarity_penalty.at[
                coreset_indices
            ].set(
                jnp.where(
                    (coreset_indices != coreset_indices[i]) & (coreset_weights > 0),
                    jnp.inf,
                    valid_kernel_similarity_penalty[coreset_indices],
                )
            )
        updated_coreset_index = selection_function(
            i, valid_kernel_similarity_penalty, coreset_size
        )
        updated_coreset_indices = coreset_indices.at[i].set(updated_coreset_index)

        penalty_update = jnp.ravel(kernel.compute(data, data[updated_coreset_index]))
        updated_penalty = kernel_similarity_penalty + penalty_update
        if unique:
            # Prevent the same 'updated_coreset_index' from being selected on a
            # subsequent iteration, by setting the penalty to infinity.
            updated_penalty = updated_penalty.at[updated_coreset_index].set(jnp.inf)

        return updated_coreset_indices, updated_penalty, coreset_size

    init_state = (
        padded_coresubset.unweighted_indices,
        init_kernel_similarity_penalty,
        init_coreset_size,
    )
    output_state = jax.lax.fori_loop(0, output_size, _greedy_body, init_state)
    new_coreset_indices = output_state[0][:output_size]
    return eqx.tree_at(lambda x: x.indices, coresubset, as_data(new_coreset_indices))


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

    Optionally, the Kernel Herding procedure can be modified using the
    ``probabilistic`` and ``temperature`` parameters in the ``reduce`` and ``refine``
    methods. If ``probabilistic`` is ``True``, a single point :math:`x` at each
    iteration is selected with probability proportional to
    :math:`\text{softmax}(\frac{\text{KHMetric(x)}}{T})`, where :math:`\text{
    KHMetric}` is given above and :math:`T` is the ``temperature`` parameter. As
    :math:`T \rightarrow \infty`, the probabilities become uniform, resulting in a
    random sampling. As :math:`T \rightarrow 0`, the probabilities become
    concentrated at the point with the highest metric, recovering the original Kernel
    Herding procedure. This feature is experimental and does not come from the
    original paper (:cite:`chen2012herding`).

    :param coreset_size: The desired size of the solved coreset
    :param kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance implementing a
        kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param unique: Boolean that ensures the resulting coresubset will only contain
        unique elements
    :param block_size: Block size passed to
        :meth:`~coreax.kernels.ScalarValuedKernel.compute_mean`
    :param unroll: Unroll parameter passed to
        :meth:`~coreax.kernels.ScalarValuedKernel.compute_mean`
    :param probabilistic: If True, the elements are chosen probabilistically at each
        iteration. Otherwise, standard Kernel Herding is run.
    :param temperature: Temperature parameter, which controls how uniform the
        probabilities are for probabilistic selection.
    :param random_key: Key for random number generation, only used if probabilistic
    """

    kernel: ScalarValuedKernel
    unique: bool = True
    block_size: Optional[Union[int, tuple[Optional[int], Optional[int]]]] = None
    unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1
    probabilistic: bool = False
    temperature: Union[float, Scalar] = eqx.field(default=1.0)
    random_key: KeyArrayLike = eqx.field(default_factory=lambda: jax.random.key(0))

    @override
    def reduce(
        self,
        dataset: _Data,
        solver_state: Optional[HerdingState] = None,
    ) -> tuple[Coresubset[_Data], HerdingState]:
        initial_coresubset = _initial_coresubset(0, self.coreset_size, dataset)
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

        .. warning::

            If the input ``coresubset`` is smaller than the requested ``coreset_size``,
            it will be padded with zero-valued, zero-weighted indices. After the input
            ``coresubset`` has been refined, new indices will be chosen to fill the
            padding. If the input ``coresubset`` is larger than the requested
            ``coreset_size``, the extra indices will not be optimised and will be
            clipped from the return ``coresubset``.

        :param coresubset: The coresubset to refine.
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :return: A refined coresubset and relevant intermediate solver state
            information.
        """
        if solver_state is None:
            x, bs, un = coresubset.pre_coreset_data, self.block_size, self.unroll
            gramian_row_mean = self.kernel.gramian_row_mean(x, block_size=bs, unroll=un)
        else:
            gramian_row_mean = solver_state.gramian_row_mean

        def selection_function(
            i: int,
            kernel_similarity_penalty: Shaped[Array, " n"],
            coreset_size: Scalar,
        ) -> Shaped[Array, ""]:
            """Greedy selection criterion - Equation 8 of :cite:`chen2012herding`."""
            valid_residuals = (
                gramian_row_mean - kernel_similarity_penalty / coreset_size
            )

            # Apply softmax to the metric for probabilistic selection
            if self.probabilistic:
                probs = jax.nn.softmax(valid_residuals / self.temperature)
                key = jr.fold_in(self.random_key, i)
                return jr.choice(
                    key, gramian_row_mean.shape[0], (), p=probs, replace=False
                )
            # Otherwise choose the best candidate
            return jnp.nanargmax(valid_residuals)

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

    def reduce_iterative(
        self,
        dataset: _Data,
        solver_state: Optional[HerdingState] = None,
        num_iterations: int = 1,
        t_schedule: Optional[Shaped[Array, " {num_iterations}"]] = None,
    ) -> tuple[Coresubset[_Data], HerdingState]:
        """
        Reduce a dataset to a coreset by refining iteratively.

        :param dataset: Dataset to reduce.
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :param num_iterations: Number of iterations of the refine method to perform.
        :param t_schedule: An :class:`Array` of length `num_iterations`, where
            `t_schedule[i]` is the temperature parameter used for iteration i. If None,
            standard Kernel Herding is used.
        :return: A coresubset and relevant intermediate solver state information.
        """
        initial_coreset = _initial_coresubset(0, self.coreset_size, dataset)
        if solver_state is None:
            x, bs, un = initial_coreset.pre_coreset_data, self.block_size, self.unroll
            solver_state = HerdingState(
                self.kernel.gramian_row_mean(x, block_size=bs, unroll=un)
            )

        def refine_iteration(i: int, coreset: Coresubset) -> Coresubset:
            """
            Perform one iteration of the refine method.

            :param i: Iteration number.
            :param coreset: Coreset to be refined.
            """
            # Update the random key
            new_solver = eqx.tree_at(
                lambda x: x.random_key, self, jr.fold_in(self.random_key, i)
            )
            # If the temperature schedule is provided, update temperature too
            if t_schedule is not None:
                new_solver = eqx.tree_at(
                    lambda x: x.temperature, new_solver, t_schedule[i]
                )

            coreset, _ = new_solver.refine(coreset, solver_state)
            return coreset

        return (
            jax.lax.fori_loop(0, num_iterations, refine_iteration, initial_coreset),
            solver_state,
        )


class SteinThinning(
    RefinementSolver[_Data, None], ExplicitSizeSolver, PaddingInvariantSolver
):
    r"""
    Stein Thinning - an explicitly sized coresubset refinement solver.

    Solves the coresubset problem by taking a deterministic, iterative, and greedy
    approach to minimizing the Kernelised Stein Discrepancy (KSD) between the empirical
    distribution of the coresubset (the solution) and the distribution of the problem
    dataset, as characterised by the score function of the Stein Kernel.

    Given one has selected :math:`t-1` data points for their compressed representation
    of the original dataset, (regularised) Stein thinning selects the next point using
    the equations in Section 3.1 of :cite:`benard2023kernel`:

    .. math::

        x_{t} = \arg\min_{x} \left( k_p(x, x) + \Delta^+ \log p(x) -
            \lambda t \log p(x) + 2 \sum_{j=1}^{t-1} k_p(x, x_j) \right)

    where :math:`k` is the Stein kernel induced by the supplied base kernel,
    :math:`\Delta^+` is the non-negative Laplace operator, :math:`\lambda` is a
    regularisation parameter, and the search is over the entire dataset.

    :param coreset_size: The desired size of the solved coreset
    :param kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance implementing a
        kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`; if 'kernel'
        is a :class:`~coreax.kernels.SteinKernel` and
        :code:`score_matching is not`:data:`None`, a new instance of the kernel will be
        generated where the score function is given by :code:`score_matching.match(...)`
    :param score_matching: Specifies/overwrite the score function of the implied/passed
       :class:`~coreax.kernels.SteinKernel`; if :data:`None`, default to
       :class:`~coreax.score_matching.KernelDensityMatching` unless 'kernel' is a
       :class:`~coreax.kernels.SteinKernel`, in which case the kernel's existing score
       function is used.
    :param unique: If each index in the resulting coresubset should be unique
    :param regularise: Boolean that enforces regularisation, see Section 3.1 of
        :cite:`benard2023kernel`.
    :param regulariser_lambda: The entropic regularisation parameter, :math:`\lambda`.
        If :data:`None`, defaults to :math:`1/\text{coreset_size}` following
        :cite:`benard2023kernel`.
    :param block_size: Block size passed to
        :meth:`~coreax.kernels.ScalarValuedKernel.compute_mean`.
    :param unroll: Unroll parameter passed to
        :meth:`~coreax.kernels.ScalarValuedKernel.compute_mean`.
    :param kde_bw_method: Bandwidth method passed to `jax.scipy.stats.gaussian_kde`.
    """

    kernel: ScalarValuedKernel
    score_matching: Optional[ScoreMatching] = None
    unique: bool = True
    regularise: bool = True
    regulariser_lambda: Optional[float] = None
    block_size: Optional[Union[int, tuple[Optional[int], Optional[int]]]] = None
    unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1
    kde_bw_method: Optional[Union[str, int, Callable]] = None

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

        .. note::
            Only the score function, :math:`\nabla \log p(x)`, is provided to the
            solver. Since the lambda regularisation term relies on the density,
            :math:`p(x)`, directly, it is estimated using a Gaussian kernel density
            estimator using `jax.scipy.stats.gaussian_kde`. The bandwidth
            method for this can passed as kde_bw_method when initialising
            :class:`SteinThinning`.

        :param coresubset: The coresubset to refine
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :return: A refined coresubset and relevant intermediate solver state
            information.
        """
        x, w_x = jtu.tree_leaves(coresubset.pre_coreset_data)
        kernel = convert_stein_kernel(x, self.kernel, self.score_matching)
        stein_kernel_diagonal = jax.vmap(self.kernel.compute_elementwise)(x, x)
        if self.regularise:
            kde = jsp.stats.gaussian_kde(x.T, weights=w_x, bw_method=self.kde_bw_method)

            if self.regulariser_lambda is None:
                # Use regularisation parameter suggested in :cite:`benard2023kernel`
                regulariser_lambda = 1 / len(coresubset)
            else:
                regulariser_lambda = self.regulariser_lambda

            regularised_log_pdf = regulariser_lambda * kde.logpdf(x.T)

            @jax.vmap
            def _laplace_positive(x_):
                r"""Evaluate Laplace positive operator  :math:`\Delta^+ \log p(x)`."""
                hessian = jax.jacfwd(kernel.score_function)(x_)
                return jnp.clip(jnp.diag(hessian), min=0.0).sum()

            laplace_correction = _laplace_positive(x)
        else:
            laplace_correction, regularised_log_pdf = 0.0, 0.0

        def selection_function(
            i: int, _kernel_similarity_penalty: ArrayLike, _coreset_size: Scalar
        ) -> Array:
            """
            Greedy selection criterion - Section 3.1 :cite:`benard2023kernel`.

            Argmin of the Laplace corrected and regularised Kernel Stein Discrepancy.
            """
            ksd = stein_kernel_diagonal + 2.0 * _kernel_similarity_penalty
            return jnp.nanargmin(
                ksd + laplace_correction - (i + 1) * regularised_log_pdf
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
        return refined_coreset, solver_state


class RPCholeskyState(eqx.Module):
    """
    Intermediate :class:`RPCholesky` solver state information.

    :param gramian_diagonal: Cached Gramian diagonal.
    """

    gramian_diagonal: Array


class RPCholesky(CoresubsetSolver[_Data, RPCholeskyState], ExplicitSizeSolver):
    r"""
    Randomly Pivoted Cholesky - an explicitly sized coresubset refinement solver.

    Solves the coresubset problem by taking a stochastic, iterative, and greedy approach
    to approximating the Gramian of a given kernel, evaluated on the original dataset.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance implementing a
        kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param unique: If each index in the resulting coresubset should be unique
    """

    random_key: KeyArrayLike
    kernel: ScalarValuedKernel
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

        .. note::

            The RPCholesky algorithm sometimes converges before reaching
            ``self.coreset_size``. In this case, the other coreset points are chosen
            uniformly at random, avoiding repetition.


        :param dataset: Dataset to reduce to a coresubset.
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :return: Refined coresubset and relevant intermediate solver state information.
        """
        x = dataset.data
        num_data_points = len(x)
        if solver_state is None:
            gramian_diagonal = jax.vmap(self.kernel.compute_elementwise)(x, x)
        else:
            gramian_diagonal = solver_state.gramian_diagonal
        # Initialise with an index outside of the possible coreset indices
        initial_coresubset = _initial_coresubset(
            num_data_points + 1, self.coreset_size, dataset
        )
        initial_coreset_indices = initial_coresubset.unweighted_indices

        def _greedy_body(
            i: int, val: tuple[Array, Array, Array]
        ) -> tuple[Array, Array, Array]:
            """RPCholesky iteration - Algorithm 1 of :cite:`chen2023randomly`."""
            residual_diagonal, approximation_matrix, coreset_indices = val
            key = jr.fold_in(self.random_key, i)

            # Fall back on uniform probability if residual_diagonal is invalid
            available_mask = (
                jnp.ones(num_data_points, dtype=bool).at[coreset_indices].set(False)
            )
            uniform_probs = available_mask / jnp.sum(available_mask)
            valid_residuals = jnp.where(
                jnp.logical_or(
                    jnp.all(residual_diagonal == 0),
                    jnp.any(jnp.isnan(residual_diagonal)),
                ),
                uniform_probs,
                residual_diagonal,
            )
            pivot_point = jr.choice(
                key, num_data_points, (), p=valid_residuals, replace=False
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
                residual_diagonal - jnp.square(updated_approximation_matrix[:, i]),
                min=0,
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

        initial_approximation_matrix = jnp.zeros((num_data_points, self.coreset_size))
        init_state = (
            gramian_diagonal,
            initial_approximation_matrix,
            initial_coreset_indices,
        )
        output_state = jax.lax.fori_loop(0, self.coreset_size, _greedy_body, init_state)
        gramian_diagonal, _, new_coreset_indices = output_state
        updated_coreset: Coresubset[_Data] = Coresubset.build(
            new_coreset_indices, dataset
        )
        return updated_coreset, RPCholeskyState(gramian_diagonal)


class GreedyKernelPointsState(eqx.Module):
    """
    Intermediate :class:`GreedyKernelPoints` solver state information.

    :param feature_gramian: Cached feature kernel gramian matrix, should be padded with
        an additional row and column of zeroes.
    """

    feature_gramian: Shaped[Array, " n+1 n+1"]


# Overload for the case where we want to construct both the identity array and the
# loss_batch_indices array.
@overload
def _setup_batch_solver(  # pragma: no cover # pyright: ignore reportOverlappingOverload
    coreset_size: int,
    coresubset: Coresubset,
    num_data_pairs: int,
    candidate_batch_size: Optional[int],
    loss_batch_size: Optional[int],
    random_key: KeyArrayLike,
    setup_identity: Literal[True] = True,
    setup_loss_batch_indices: Literal[True] = True,
) -> tuple[Array, Array, Array, Array, Array]: ...


# Overload for the case where we want to construct neither the identity array or the
# loss_batch_indices array.
@overload
def _setup_batch_solver(  # pragma: no cover
    coreset_size: int,
    coresubset: Coresubset,
    num_data_pairs: int,
    candidate_batch_size: Optional[int],
    loss_batch_size: Optional[int],
    random_key: KeyArrayLike,
    setup_identity: Literal[False] = False,
    setup_loss_batch_indices: Literal[False] = False,
) -> tuple[Array, Array, None, Array, None]: ...


# Overload for the case where we want to construct just the identity array and not the
# loss_batch_indices array.
@overload
def _setup_batch_solver(  # pragma: no cover
    coreset_size: int,
    coresubset: Coresubset,
    num_data_pairs: int,
    candidate_batch_size: Optional[int],
    loss_batch_size: Optional[int],
    random_key: KeyArrayLike,
    setup_identity: Literal[True] = True,
    setup_loss_batch_indices: Literal[False] = False,
) -> tuple[Array, Array, None, Array, Array]: ...


# Overload for the case where we do not want to construct the identity array but we do
# the loss_batch_indices array.
@overload
def _setup_batch_solver(  # pragma: no cover
    coreset_size: int,
    coresubset: Coresubset,
    num_data_pairs: int,
    candidate_batch_size: Optional[int],
    loss_batch_size: Optional[int],
    random_key: KeyArrayLike,
    setup_identity: Literal[False] = False,
    setup_loss_batch_indices: Literal[True] = True,
) -> tuple[Array, Array, Array, Array, None]: ...


def _setup_batch_solver(
    coreset_size: int,
    coresubset: Coresubset,
    num_data_pairs: int,
    candidate_batch_size: Optional[int],
    loss_batch_size: Optional[int],
    random_key: KeyArrayLike,
    setup_identity: bool = False,
    setup_loss_batch_indices: bool = True,
) -> tuple[
    Shaped[Array, " n"],
    Shaped[Array, " n p"],
    Optional[Shaped[Array, " n p"]],
    Shaped[Array, " p n"],
    Optional[Shaped[Array, " n n"]],
]:
    """
    Set up the matrices required to initialise a batched solver.

    For use in :class:`GreedyKernelPoints` to reduce code duplication with future
    similar solvers.

    :param coreset_size: Requested coreset size.
    :param coresubset: Instance of :class:`~coreax.coreset.Coresubset` representing
        the coreset to be refined.
    :param  num_data_pairs: An integer representing the number of data pairs in the
        original dataset.
    :param candidate_batch_size: An integer representing the number of data pairs to
        randomly sample at each iteration to consider adding to the coreset.
    :param loss_batch_size: Number of data pairs to randomly sample at each iteration to
        consider computing the loss with respect to.
    :param random_key: Key for random number generation.
    :param setup_identity: Boolean indicating whether or not to construct and return
        an identity matrix used in coreset construction, defaults to :data:`False`.
    :param setup_loss_batch_indices: Boolean indicating whether or not to construct and
        return an array of batch indices for loss computation, defaults to :data:`True`.

    :return: Tuple of matrices, (``initial_coreset_indices``,
        ``loss_batch_indices``, ``candidate_batch_indices``,
        ``initial_candidate_coresets``, ``identity``), the first ``coreset_size``-vector
        which will be updated with the coreset index selected at each iteration. The
        second a ``coreset_size`` x ``candidate_batch_size`` matrix of sampled indices,
        each row represents an iteration of the algorithm and holds the indices we
        consider adding to the coreset. The third a ``coreset_size`` x
        ``candidate_batch_size`` matrix of sampled indices, each row represents an
        iteration of the algorithm and holds the indices we compute the loss with
        respect to. The fourth a ``candidate_batch_size`` x ``coreset_size`` matrix
        which will be iteratively updated such that each row represents a coreset under
        consideration at each iteration. Lastly an adaptively initialised
        "identity matrix". For reduction this is a zeroes matrix which we will
        iteratively add ones to on the diagonal, while for refinement this is an actual
        identity matrix. If ``setup_identity`` is :data:`False`, we return :data:`None`.
    """
    # If the initialisation coresubset is too small, pad its nodes up to
    # 'output_size' with -1 valued indices. If it is too large, raise a warning and
    # clip off the indices at the end. We fill with -1 valued indices as the -1 index
    # will point towards the zero-valued-padding of the arrays we wish to index. This
    # lets us work around JAX requiring static array sizes. Note that if we are reducing
    # coresubset.unweighted_indices will be a vector of -1's, whereas if we are refining
    # this will be a vector of actual coreset indices
    if coreset_size > len(coresubset):
        pad_size = max(0, coreset_size - len(coresubset))
        initial_coreset_indices = jnp.hstack(
            (
                coresubset.unweighted_indices,
                -1 * jnp.ones(pad_size, dtype=jnp.int32),
            )
        )
    elif coreset_size < len(coresubset):
        warn(
            "Requested coreset size is smaller than input 'coresubset', clipping"
            + " to the correct size and proceeding...",
            SizeWarning,
            stacklevel=3,
        )
        initial_coreset_indices = coresubset.unweighted_indices[:coreset_size]
    else:
        initial_coreset_indices = coresubset.unweighted_indices

    # Sample the indices to be considered at each iteration ahead of time. This is a
    # coreset_size x candidate_batch_size array. Each row corresponds to an iteration of
    # the algorithm; the nth row holds all the indices we will consider adding to the
    # coreset in the nth iteration.

    candidate_key, loss_key = jr.split(random_key)
    if candidate_batch_size is None or candidate_batch_size > num_data_pairs:
        candidate_batch_size = num_data_pairs
    candidate_batch_indices = sample_batch_indices(
        random_key=candidate_key,
        max_index=num_data_pairs,
        batch_size=candidate_batch_size,
        num_batches=coreset_size,
    )

    # Add coreset indices onto the batch indices to avoid degrading the loss (only
    # has an effect when refining). This extends the candidate_batch_indices array to a
    # coreset_size x candidate_batch_size + 1 array; we do this because when refining,
    # it may be the case that the current index is the best one, and we should not
    # replace it with any of the batch indices we sampled. Note that we increment
    # candidate_batch_size by 1.
    #
    # e.g. Assume we are reducing, coreset_size = 5, candidate_batch_size = 3 + 1, then
    # this will look something like (noticing that each row must have unique elements)
    #
    #       | 89  24 74 -1 |
    #       | 11  43 65 -1 |
    #       | 54  12 11 -1 |
    #       | 101 23 11 -1 |
    #       | 13  19 82 -1 |
    #
    # If we are refining, and the provided coreset is [ 0, 1, 18, 23, 100], then
    # it will look like
    #
    #       | 89  24 74 0   |
    #       | 11  43 65 1   |
    #       | 54  12 11 18  |
    #       | 101 23 11 23  |
    #       | 13  19 82 100 |
    #
    if candidate_batch_size is not None and candidate_batch_size < num_data_pairs:
        candidate_batch_indices = jnp.hstack(
            (candidate_batch_indices, initial_coreset_indices[:, jnp.newaxis])
        )
        candidate_batch_size += 1

    # Initialise an array that will let us extract arrays corresponding to every
    # possible candidate coreset. This is a candidate_batch_size x coreset_size array,
    # which is filled with -1's, except the first column, which we fill in with the
    # first set of batch indices. This means that each row is a possible candidate
    # coreset to be considered at the first iteration. At the second iteration, we
    # replace the first column by repeating the best coreset index of those batched
    # into it, and replace the next column (of -1's) with the next set of batch indices,
    # and so on... In the case of refinement, the initial_coreset_indices will not be
    # a vector of -1's, but instead a vector of actual indices. In this case the
    # initial_candidate_coresets array will consist of the initial_coreset_indices
    # vector repeated into each row, except the first column, which again we replace
    # with the first set of batched indices.
    #
    # e.g. If we are reducing, for candidate_batch_size = 3 + 1 and coreset_size = 5,
    # this will look something like
    #
    #       | 89 -1 -1 -1 -1 |
    #       | 24 -1 -1 -1 -1 |
    #       | 74 -1 -1 -1 -1 |
    #       | -1 -1 -1 -1 -1 |
    #
    # at the first iteration, and assuming that 24 is the best of those 3 batched
    # indices
    #
    #       | 24 11 -1 -1 -1 |
    #       | 24 43 -1 -1 -1 |
    #       | 24 65 -1 -1 -1 |
    #       | 24 -1 -1 -1 -1 |
    #
    # at the second iteration, and so on... If we are refining on the other hand,
    # and we have a coreset [ 0, 1, 18, 23, 100], then it will look like
    #
    #       | 89, 1, 18, 23, 100 |
    #       | 24, 1, 18, 23, 100 |
    #       | 74, 1, 18, 23, 100 |
    #       | 0,  1, 18, 23, 100 |
    #
    # at the first iteration and, assuming 0 is still the best coreset index, then
    #
    #       | 0, 11, 18, 23, 100 |
    #       | 0, 43, 18, 23, 100 |
    #       | 0, 65, 18, 23, 100 |
    #       | 0, 1,  18, 23, 100 |
    #
    # at the second iteration, and so on...
    initial_candidate_coresets = (
        jnp.tile(initial_coreset_indices, (candidate_batch_size, 1))
        .at[:, 0]
        .set(candidate_batch_indices[0, :])
    )

    # Adaptively initialise a coreset_size x coreset_size "identity matrix". For
    # reduction, we need this to be a zeroes matrix to which we will add ones on the
    # diagonal at each iteration of the algorithm. While for refinement we need an
    # actual identity matrix.
    identity = None
    if setup_identity:
        identity_helper = jnp.hstack((jnp.ones(num_data_pairs), jnp.array([0])))
        identity = jnp.diag(identity_helper[initial_coreset_indices])

    # Set up a coreset_size x loss_batch_size array holding indices that we compute
    # the loss with respect to. Each row corresponds to an iteration of the algorithm.
    # The cost of sampling these indices is non-zero, so we handle the option of not
    # sampling these at all for batched algorithms where this batch has no use.
    if setup_loss_batch_indices:
        if loss_batch_size is None or loss_batch_size > num_data_pairs:
            loss_batch_size = num_data_pairs
        loss_batch_indices = sample_batch_indices(
            random_key=loss_key,
            max_index=num_data_pairs,
            batch_size=loss_batch_size,
            num_batches=coreset_size,
        )
    else:
        loss_batch_indices = None

    return (
        initial_coreset_indices,
        candidate_batch_indices,
        loss_batch_indices,
        initial_candidate_coresets,
        identity,
    )


def _update_candidate_coresets_and_coreset_indices(
    i: int,
    unique: bool,
    candidate_coresets: Shaped[Array, " p n"],
    coreset_indices: Shaped[Array, " n"],
    loss: Shaped[Array, " p"],
    candidate_batch_indices: Shaped[Array, " n p"],
) -> tuple[Shaped[Array, " p n"], Shaped[Array, " n"]]:
    """
    Update the coreset indices and candidate coreset matrix following loss computation.

    For use with :class:`GreedyKernelPoints` to reduce code duplication with future
    coreset methods.

    On the ``i``th iteration, the ``i``th column of ``candidate_coresets`` contains the
    index we consider adding to the coreset. The preceding columns are fixed, containing
    indices already selected for the coreset. The following columns are -1.
    See :func:`_setup_batch_solver` for further details.

    :param i: Integer representing the current iteration number.
    :param unique: If each index in the resulting coresubset should be unique.
    :param candidate_coresets: Matrix of indices representing all possible "next"
        coresets; each row contains the indices of a candidate coreset.
    :param coreset_indices: Vector of the current coreset indices.
    :param loss: Vector of loss values corresponding to each potential next coreset
        point.
    :param candidate_batch_indices: Matrix of batch indices for consideration of adding
        to the coreset. Each column contains the candidate indices for each iteration.
    :return: Tuple of matrices (``updated_candidate_coresets`,
        ``updated_coreset_indices``), the first is  ``candidate_coresets`` with the
        chosen coreset index repeated into the ``i``th column, and the next batch of
        indices inserted into the (``i``:math:`+1`)th column. The second is the vector
        of ``coreset_indices`` with the chosen coreset index inserted as the ``i``th
        element.
    """
    # If we want the coreset indices to be unique, we add infinity to the value of
    # the loss corresponding to those indices that are already in the coreset. This
    # ensures it is not chosen. There is additional logic here to ensure that if
    # we are refining (i.e. not reducing), we are allowed to keep the index that
    # currently exists in the coreset if it is the best one.
    if unique:
        already_chosen_indices_mask = jnp.isin(
            candidate_coresets[:, i],
            coreset_indices.at[i].set(-1),
        )
        loss += jnp.where(already_chosen_indices_mask, jnp.inf, 0)
    index_to_include_in_coreset = candidate_coresets[loss.argmin(), i]

    # Repeat the chosen coreset index into the ith column of the array of
    # the candidate_batch_size x coreset_size candidate_coresets array. Replace the
    # (i + 1)th column with the next batch of possible coreset indices. This ensures
    # that each row of the updated_candidate_coresets array corresponds to a potential
    # coreset to be considered at the next iteration of the algorithm. See comments
    # inside _setup_batch_solver for an explicit example of this update procedure.
    updated_candidate_coresets = (
        candidate_coresets.at[:, i]
        .set(index_to_include_in_coreset)
        .at[:, i + 1]
        .set(candidate_batch_indices[i + 1, :])
    )

    # Add the chosen coreset index to the current coreset indices
    updated_coreset_indices = coreset_indices.at[i].set(index_to_include_in_coreset)
    return updated_candidate_coresets, updated_coreset_indices


def _greedy_kernel_points_loss(
    candidate_coresets: Shaped[Array, " batch_size coreset_size"],
    responses: Shaped[Array, " n+1 1"],
    feature_gramian: Shaped[Array, " n+1 n+1"],
    regularisation_parameter: float,
    identity: Shaped[Array, " coreset_size coreset_size"],
    least_squares_solver: RegularisedLeastSquaresSolver,
    loss_batch: Shaped[Array, " batch_size"],
) -> Shaped[Array, " batch_size"]:
    r"""
    Compute greedy kernel inducing points losses for a matrix of candidate coresets.

    Primarily intended for use with :class:`GreedyKernelPoints`.

    :param candidate_coresets: Matrix of indices representing all possible "next"
        coresets.
    :param responses: Matrix of responses.
    :param feature_gramian: Feature kernel Gramian.
    :param regularisation_parameter: Regularisation parameter :math:`\lambda` for
        stable inversion of the feature Gramian; negative values will be converted
        to positive.
    :param identity: Identity matrix used to regularise the feature Gramians
        corresponding to each coreset. For :meth:`GreedyKernelPoints.reduce`,
        this is a matrix of zeroes except for ones on the diagonal up to the
        current size of the coreset. For :meth:`GreedyKernelPoints.refine`, this
        is a standard identity matrix.
    :param loss_batch: Vector of batch indices, which we use to speed up computation
        of the loss.
    :param least_squares_solver: Instance of
        :class:`~coreax.least_squares.RegularisedLeastSquaresSolver`
    :param loss_batch: Batch of indices to make predictions on.
    :return: :class:`GreedyKernelPoints` loss for each candidate coreset
    """
    # For all of these arrays, the first dimension corresponds to each coreset under
    # consideration.

    # candidate_batch_size x coreset_size x coreset_size array holding the scalar-valued
    # kernel matrix computed on each coreset
    coreset_gramians = feature_gramian[
        (candidate_coresets[:, :, None], candidate_coresets[:, None, :])
    ]

    # candidate_batch_size x coreset_size x 1 (response dimension) array holding the
    # vector-valued responses corresponding to the coreset indices
    coreset_responses = responses[candidate_coresets]

    # coreset_cross_gramians is a candidate_batch_size x coreset_size x loss_batch_size
    # array containing the scalar-valued kernel matrices between each coreset and the
    # current loss batch of dataset.
    coreset_cross_gramians = feature_gramian[:, loss_batch][candidate_coresets, :]

    # Solve for the vector-valued kernel ridge regression coefficients for each possible
    # coreset. This is a candidate_batch_size x coreset_size x 1 (response dimension)
    # array.
    coefficients = least_squares_solver.solve_stack(
        arrays=coreset_gramians,
        regularisation_parameter=regularisation_parameter,
        targets=coreset_responses,
        identity=identity,
    )

    # Using each coreset model, make predictions of the training responses. The
    # predictions array is a candidate_batch_size x n (data size) x 1 (response
    # dimension) array representing the predictions (second and third dimensions) of
    # each model fitted on the candidate coresets (first dimension):
    # predictions = coreset_cross_gramians^T @ coreset_coefficients
    predictions = (coreset_cross_gramians * coefficients).sum(axis=1)

    # The loss array is a candidate_batch_size array representing the value of each
    # candidate coreset point; the smallest loss corresponds to the best coreset.
    # Sample the responses via batching and compute loss for each coreset. Note that
    # we do not compute the first term as it is invariant w.r.t. coreset.
    loss = ((predictions - 2 * responses[loss_batch, 0]) * predictions).sum(axis=1)
    return loss


class GreedyKernelPoints(
    RefinementSolver[_SupervisedData, GreedyKernelPointsState],
    ExplicitSizeSolver,
):
    r"""
    Apply Greedy Kernel Points to a supervised dataset.

    GreedyKernelPoints is a deterministic, iterative and greedy approach to
    build a coreset, adapted from the kernel inducing point (KIP) method developed in
    :cite:`nguyen2021meta`.

    Given one has an original dataset :math:`\mathcal{D}^{(1)} = \{(x_i, y_i)\}_{i=1}^n`
    of :math:`n` pairs with :math:`x \in \mathbb{R}^d` and :math:`y \in \mathbb{R}`,
    and one has selected :math:`m` data pairs
    :math:`\mathcal{D}^{(2)} = \{(\tilde{x}_i, \tilde{y}_i)\}_{i=1}^m` already for their
    compressed representation of the original dataset, Greedy Kernel Points selects
    the next point to minimise the loss

    .. math::

        L \left( \mathcal{D}^{(1)}, \mathcal{D}^{(2)} \right) = \left\|
        y^{(1)} - K^{(12)} \left( K^{(22)} + \lambda I_m \right)^{-1} y^{(2)}
        \right\|^2_{\mathbb{R}^n} ,

    where :math:`y^{(1)} \in \mathbb{R}^n` is the vector of responses from
    :math:`\mathcal{D}^{(1)}`, :math:`y^{(2)} \in \mathbb{R}^n` is the vector of
    responses from :math:`\mathcal{D}^{(2)}`,
    :math:`K^{(12)} \in \mathbb{R}^{n \times m}` is the cross-matrix of kernel
    evaluations between :math:`\mathcal{D}^{(1)}` and :math:`\mathcal{D}^{(2)}`,
    :math:`K^{(22)} \in \mathbb{R}^{m \times m}` is the kernel matrix on
    :math:`\mathcal{D}^{(2)}`, :math:`\lambda I_m \in \mathbb{R}^{m \times m}` is the
    identity matrix and :math:`\lambda \in \mathbb{R}_{>0}` is a regularisation
    parameter.

    We remark that :math:`\left( K^{(22)} + \lambda I_m \right)^{-1} y^{(2)}` are kernel
    ridge regression coefficients.

    This class works with all children of :class:`~coreax.kernels.ScalarValuedKernel`.

    .. warning::

        ``GreedyKernelPoints`` does not support non-uniform weights and will only
        return coresubsets with uniform weights.

    :param coreset_size: Desired size of the solved coreset.
    :param random_key: Key for random number generation.
    :param feature_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` on the
        feature space.
    :param regularisation_parameter: Regularisation parameter :math:`\lambda` for
        stable inversion of the feature Gramian; negative values will be converted
        to positive.
    :param unique: If :data:`False`, the resulting coresubset may contain the same point
        multiple times. If :data:`True` (default), the resulting coresubset will not
        contain any duplicate points.
    :param candidate_batch_size: Number of data pairs to randomly sample at each
        iteration to consider adding to the coreset. If :data:`None` (default), the
        search is performed over the entire dataset.
    :param loss_batch_size:  Number of data pairs to randomly sample at each iteration
        to compute the loss function with respect to. If :data:`None` (default), the
        loss is computed using the entire dataset.
    :param least_squares_solver: Instance of
        :class:`~coreax.least_squares.RegularisedLeastSquaresSolver`, default value of
        :data:`None` uses :class:`~coreax.least_squares.MinimalEuclideanNormSolver`.
    """

    random_key: KeyArrayLike
    feature_kernel: ScalarValuedKernel
    regularisation_parameter: float = 1e-6
    unique: bool = True
    candidate_batch_size: Optional[int] = None
    loss_batch_size: Optional[int] = None
    least_squares_solver: Optional[RegularisedLeastSquaresSolver] = None

    @override
    def reduce(
        self,
        dataset: SupervisedData,
        solver_state: Optional[GreedyKernelPointsState] = None,
    ) -> tuple[Coresubset[SupervisedData], GreedyKernelPointsState]:
        initial_coresubset = _initial_coresubset(-1, self.coreset_size, dataset)
        return self.refine(initial_coresubset, solver_state)

    def refine(  # noqa: PLR0915
        self,
        coresubset: Coresubset[SupervisedData],
        solver_state: Optional[GreedyKernelPointsState] = None,
    ) -> tuple[Coresubset[SupervisedData], GreedyKernelPointsState]:
        """
        Refine a coresubset with :class:`GreedyKernelPointsState`.

        We first compute the various factors if they are not given in the
        ``solver_state``, and then iteratively swap points with the initial coreset,
        selecting points that minimise the loss.

        .. warning::

            If the input ``coresubset`` is smaller than the requested ``coreset_size``,
            it will be padded with indices that are invariant with respect to the loss.
            After the input ``coresubset`` has been refined, new indices will be chosen
            to fill the padding. If the input ``coresubset`` is larger than the
            requested ``coreset_size``, the extra indices will not be optimised and will
            be clipped from the return ``coresubset``.

        :param coresubset: Coresubset to refine.
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :return: A refined coresubset; Relevant intermediate solver state information.
        """
        dataset = coresubset.pre_coreset_data
        num_data_pairs = len(dataset)

        # Handle default value of None
        if self.least_squares_solver is None:
            least_squares_solver = MinimalEuclideanNormSolver()
        else:
            least_squares_solver = self.least_squares_solver

        # See _setup_batch_solver for a detailed explanation of these arrays
        (
            initial_coreset_indices,
            candidate_batch_indices,
            loss_batch_indices,
            initial_candidate_coresets,
            initial_identity,
        ) = _setup_batch_solver(
            coreset_size=self.coreset_size,
            coresubset=coresubset,
            num_data_pairs=num_data_pairs,
            candidate_batch_size=self.candidate_batch_size,
            loss_batch_size=self.loss_batch_size,
            random_key=self.random_key,
            setup_identity=True,
            setup_loss_batch_indices=True,
        )

        # Pad the response array with an additional zero to allow us to
        # extract sub-arrays and fill in elements with zeroes simultaneously.
        x, y = dataset.data, dataset.supervision
        padded_responses = jnp.vstack((y, jnp.zeros(y.shape[1])))

        if solver_state is None:
            # Pad the Gramian with zeroes in an additional column and row
            padded_feature_gramian = jnp.pad(
                self.feature_kernel.compute(x, x), [(0, 1)], mode="constant"
            )
        else:
            padded_feature_gramian = solver_state.feature_gramian

        def _greedy_body(
            i: int,
            val: tuple[
                Shaped[Array, " n"], Shaped[Array, " n"], Shaped[Array, " batch_size n"]
            ],
        ) -> tuple[
            Shaped[Array, " n"], Shaped[Array, " n"], Shaped[Array, " batch_size n"]
        ]:
            """
            Execute main loop of Greedy Kernel Points.

            :param i: Integer representing iteration number.
            :param val: Tuple of arrays used to do one iteration of Greedy Kernel
                Points.
            :return: Updated ``val``.
            """
            coreset_indices, identity, candidate_coresets = val

            # Update the identity matrix to allow for sub-array inversion in the
            # case of reduction (no effect when refining).
            updated_identity = identity.at[i, i].set(1)

            # Compute the loss corresponding to each candidate coreset. Note that we do
            # not compute the first term as it is an invariant quantity wrt the coreset.
            loss = _greedy_kernel_points_loss(
                candidate_coresets,
                padded_responses,
                padded_feature_gramian,
                self.regularisation_parameter,
                updated_identity,
                least_squares_solver,
                loss_batch_indices[i],
            )

            # See _update_candidate_coresets_and_coreset_indices for an
            # explanation of this update procedure
            updated_candidate_coresets, updated_coreset_indices = (
                _update_candidate_coresets_and_coreset_indices(
                    i,
                    self.unique,
                    candidate_coresets,
                    coreset_indices,
                    loss,
                    candidate_batch_indices,
                )
            )
            return updated_coreset_indices, updated_identity, updated_candidate_coresets

        # Greedily refine coreset points
        new_coreset_indices, _, _ = jax.lax.fori_loop(
            lower=0,
            upper=self.coreset_size,
            body_fun=_greedy_body,
            init_val=(
                initial_coreset_indices,
                initial_identity,
                initial_candidate_coresets,
            ),
        )

        return (
            Coresubset.build(new_coreset_indices, dataset),
            GreedyKernelPointsState(padded_feature_gramian),
        )


class KernelThinning(CoresubsetSolver[_Data, None], ExplicitSizeSolver):
    r"""
    Kernel Thinning - a hierarchical coreset construction solver.

    `Kernel Thinning` is a hierarchical, and probabilistic algorithm for coreset
    construction. It builds a coreset by splitting the dataset into several candidate
    coresets by repeatedly halving the dataset and applying probabilistic swapping.
    The best of these candidates (the one with the lowest MMD) is chosen which is
    further refined to minimise the Maximum Mean Discrepancy (MMD) between the original
    dataset and the coreset. This implementation is a modification of the Kernel
    Thinning algorithm in :cite:`dwivedi2024kernelthinning` to make it an
    ExplicitSizeSolver.

    :param kernel: A `~coreax.kernels.ScalarValuedKernel` instance defining the primary
        kernel function used for choosing the best coreset and refining it.
    :param random_key: Key for random number generation, enabling reproducibility of
        probabilistic components in the algorithm.
    :param delta: A float between 0 and 1 used to compute the swapping probability
        during the splitting process. A recommended value is
        :math:`\frac{1}{n \log (\log n)}`, where :math:`n` is the length of the original
        dataset.
    :param sqrt_kernel: A `~coreax.kernels.ScalarValuedKernel` instance representing the
        square root kernel used for splitting the original dataset.
    """

    kernel: ScalarValuedKernel
    random_key: KeyArrayLike
    delta: float
    sqrt_kernel: ScalarValuedKernel

    def reduce(
        self, dataset: _Data, solver_state: None = None
    ) -> tuple[Coresubset[_Data], None]:
        """
        Reduce 'dataset' to a :class:`~coreax.coreset.Coresubset` with 'KernelThinning'.

        This is done by first computing the number of halving steps required, referred
        to as `m`. The original data is clipped so that it is divisible by a power of
        two. The kernel halving algorithm is then recursively applied to halve the data.

        Subsequently, a `baseline_coreset` is added to the ensemble of coresets. The
        best coreset is selected to minimise the Maximum Mean Discrepancy (MMD) and
        finally, it is refined further for optimal results. This final refinement step
        can reintroduce the clipped data dataset if they are found to reduce the MMD.

        :param dataset: The original dataset to be reduced.
        :param solver_state: The state of the solver.

        :return: A tuple containing the final coreset and the solver state (None).
        """
        if self.coreset_size > len(dataset):
            raise ValueError(MSG)
        n = len(dataset)
        m = math.floor(math.log2(n) - math.log2(self.coreset_size))
        clipped_original_dataset = dataset[: self.coreset_size * 2**m]

        partition = self.kt_half_recursive(clipped_original_dataset, m, dataset)
        baseline_coreset = self.get_baseline_coreset(dataset, self.coreset_size)
        partition.append(baseline_coreset)

        best_coreset_indices = self.kt_choose(partition, dataset)
        return self.kt_refine(Coresubset(Data(best_coreset_indices), dataset))

    def kt_half_recursive(
        self,
        current_coreset: Union[_Data, Coresubset[_Data]],
        m: int,
        original_dataset: _Data,
    ) -> list[Coresubset[_Data]]:
        """
        Recursively halve the original dataset into coresets.

        :param current_coreset: The current coreset or dataset being partitioned.
        :param m: The remaining depth of recursion.
        :param original_dataset: The original dataset.
        :return: Fully partitioned list of coresets.
        """
        # If m == 0, do not do anything just convert to original data to type Coresubset
        if m == 0:
            return [
                Coresubset(Data(jnp.arange(len(current_coreset))), original_dataset)
            ]

        # Recursively call self.kt_half on the coreset (or the dataset)
        if isinstance(current_coreset, Coresubset):
            subset1, subset2 = self.kt_half(current_coreset.points)
        else:
            subset1, subset2 = self.kt_half(current_coreset)

        # Update pre_coreset_data for both subsets to point to the original dataset
        subset1 = eqx.tree_at(lambda x: x.pre_coreset_data, subset1, original_dataset)
        subset2 = eqx.tree_at(lambda x: x.pre_coreset_data, subset2, original_dataset)

        # Update indices: map current subset's indices to original dataset
        if isinstance(current_coreset, Coresubset):
            parent_indices = current_coreset.indices.data  # Parent subset's indices
            subset1_indices = subset1.unweighted_indices  # Indices relative to parent
            subset2_indices = subset2.unweighted_indices  # Indices relative to parent

            # Map subset indices back to original dataset
            subset1_indices = parent_indices[subset1_indices]
            subset2_indices = parent_indices[subset2_indices]

            # Update the subsets with the remapped indices
            subset1 = eqx.tree_at(lambda x: x.indices.data, subset1, subset1_indices)
            subset2 = eqx.tree_at(lambda x: x.indices.data, subset2, subset2_indices)

        # Recurse for both subsets and concatenate results
        return self.kt_half_recursive(
            subset1, m - 1, original_dataset
        ) + self.kt_half_recursive(subset2, m - 1, original_dataset)

    def kt_half(self, dataset: _Data) -> tuple[Coresubset[_Data], Coresubset[_Data]]:
        """
        Partition the given dataset into two subsets.

        First, initialise two coresubsets, each of which will contain half the points of
        the original dataset. Divide the points of the original dataset into pairs and
        probabilistically decide which point of the pair should go to which of the
        coresets. This function uses variables such as `a`, `b`, `sigma`, and `delta`,
        they refer to the corresponding parameters in the paper
        :cite:`dwivedi2024kernelthinning`.

        :param dataset: The input dataset to be halved.
        :return: A list containing the two partitioned coresets.
        """
        n = len(dataset) // 2
        original_array = dataset.data
        first_coreset_indices = jnp.zeros(n, dtype=jnp.int32)
        second_coreset_indices = jnp.zeros(n, dtype=jnp.int32)

        original_array_masking = jnp.zeros(2 * n)
        coresets_masking = jnp.zeros(n)

        # Initialise parameter
        sigma = jnp.float32(0)
        k = self.sqrt_kernel.compute_elementwise

        def compute_kernel_distance(x1, x2):
            """
            Compute kernel distance between two data points.

            :param x1: The first data point.
            :param x2: The second data point.
            :return: The kernel distance between `x1` and `x2`.
            """
            return jnp.sqrt(k(x1, x1) + k(x2, x2) - 2 * k(x1, x2))

        def get_a_and_sigma(b, sigma):
            """Compute 'a' and new sigma parameter."""
            a = jnp.maximum(b * sigma * jnp.sqrt(2 * jnp.log(2 / self.delta)), b**2)

            # Update sigma
            new_sigma = jnp.sqrt(
                sigma**2 + jnp.maximum(b**2 * (1 + (b**2 - 2 * a) * sigma**2 / a**2), 0)
            )

            return a, new_sigma

        def get_alpha(
            x1: Float[Array, "1 d"],
            x2: Float[Array, "1 d"],
            i: int,
            current_first_coreset: Float[Array, "n d"],
            original_dataset_masking: Bool[Array, "2n d"],
            coreset_masking: Bool[Array, "n d"],
        ) -> tuple[Float[Array, ""], Bool[Array, "2n d"], Bool[Array, "n d"]]:
            r"""
            Calculate the value of alpha and update the boolean arrays.

            .. math::
              \\alpha(x, y, S, S1) =
              \\sum_{s \\in S}\\left(k(s, x) - k(s, y)\\right) - 2
              \\sum_{s \\in S1}\\left(k(s, x) - k(s, y)\\right), where S is the current
              data-points already considered and S1 is the current state of the first
              coreset.

            :param x1: The first data point in the kernel evaluation.
            :param x2: The second data point in the kernel evaluation.
            :param i: The current index in the iteration.
            :param current_first_coreset: Current first_coreset_indices.
            :param original_dataset_masking: A boolean array that tracks indices.
            :param coreset_masking: A boolean array that tracks indices.
            :return: A tuple containing:
                     - `alpha`: The computed value of alpha.
                     - `original_array_masking`: Updated boolean array for the dataset.
                     - `coresets_masking`: Updated boolean array for the coresets.
            """
            # Define the vectorised functions: k(.,x_1), k(.,x_2)
            k_vec_x1 = jax.vmap(lambda y: k(y, x1))
            k_vec_x2 = jax.vmap(lambda y: k(y, x2))

            # Define the indexed versions of the above functions were, we can pass
            # the index set k(original_array[], x_1) and k(original_array[], x_2)
            k_vec_x1_idx = jax.vmap(lambda y: k(original_array[y], x1))
            k_vec_x2_idx = jax.vmap(lambda y: k(original_array[y], x2))

            # Because the size of jax arrays are pre-fixed, we have to only sum the
            # first few elements and ignore the rest of elements, this is achieved by
            # dotting with a boolean array
            term1 = jnp.dot(
                (k_vec_x1(original_array) - k_vec_x2(original_array)),
                original_dataset_masking,
            )
            term2 = -2 * jnp.dot(
                (
                    k_vec_x1_idx(current_first_coreset)
                    - k_vec_x2_idx(current_first_coreset)
                ),
                coreset_masking,
            )
            # For original_array_masking, set 2i and 2i+1 positions to 1
            original_dataset_masking = original_dataset_masking.at[2 * i].set(1)
            original_dataset_masking = original_dataset_masking.at[2 * i + 1].set(1)
            # For coresets_masking, set i-th position to 1
            coreset_masking = coreset_masking.at[i].set(1)
            # Combine all terms
            alpha = term1 + term2
            return alpha, original_dataset_masking, coreset_masking

        def probabilistic_swap(
            i: int, a: jnp.ndarray, alpha: jnp.ndarray, random_key: KeyArrayLike
        ) -> tuple[tuple[int, int], KeyArrayLike]:
            """
            Perform a probabilistic swap based on the given parameters.

            :param i: The current index in the dataset.
            :param a: The swap threshold computed based on kernel parameters.
            :param alpha: The calculated value for probabilistic swapping.
            :param random_key: A random key for generating random numbers.
            :return: A tuple containing:
                     - A tuple of indices indicating the swapped values.
                     - The updated random key.
            """
            key1, key2 = jax.random.split(random_key)

            swap_probability = 1 / 2 * (1 - alpha / a)
            should_swap = jax.random.uniform(key1) <= swap_probability
            return lax.cond(
                should_swap,
                lambda _: (2 * i + 1, 2 * i),  # do swap: val1 = x2, val2 = x1
                lambda _: (2 * i, 2 * i + 1),  # don't swap: val1 = x1, val2 = x2
                None,
            ), key2

        def kernel_thinning_body_fun(
            i: int,
            state: tuple[
                Float[Array, "n"],  # first_coreset_indices
                Float[Array, "n"],  # second_coreset_indices
                Float[Array, "1"],  # sigma parameter
                Bool[Array, "2n"],  # original_array_masking
                Bool[Array, "n"],  # coresets_masking
                KeyArrayLike,
            ],
        ) -> tuple[
            Float[Array, "n"],
            Float[Array, "n"],
            Float[Array, "1"],
            Bool[Array, "2n"],
            Bool[Array, "n"],
            KeyArrayLike,
        ]:
            """
            Perform one iteration of the halving process.

            :param i: The current iteration index.
            :param state: A tuple containing:
                          - first_coreset_indices: The first array of indices.
                          - second_coreset_indices: The second array of indices.
                          - param: The sigma parameter.
                          - original_array_masking: Boolean array for masking.
                          - coresets_masking: Boolean array for masking coresets.
                          - random_key: A JAX random key.
            :return: The updated state tuple after processing the current iteration.
            """
            (
                first_coreset_indices,
                second_coreset_indices,
                sigma,
                original_array_masking,
                coresets_masking,
                random_key,
            ) = state
            # Step 1: Get values from original array
            x1 = original_array[i * 2]
            x2 = original_array[i * 2 + 1]
            # Step 2: Get a and new parameter
            a, new_sigma = get_a_and_sigma(compute_kernel_distance(x1, x2), sigma)
            # Step 3: Compute alpha
            alpha, new_bool_arr_1, new_bool_arr_2 = get_alpha(
                x1,
                x2,
                i,
                first_coreset_indices,
                original_array_masking,
                coresets_masking,
            )
            # Step 4: Get final values
            (val1, val2), new_random_key = probabilistic_swap(i, a, alpha, random_key)
            # Step 5: Update arrays
            updated_first_coreset_indices = first_coreset_indices.at[i].set(val1)
            updated_second_coreset_indices = second_coreset_indices.at[i].set(val2)
            return (
                updated_first_coreset_indices,
                updated_second_coreset_indices,
                new_sigma,
                new_bool_arr_1,
                new_bool_arr_2,
                new_random_key,
            )

        (final_arr1, final_arr2, _, _, _, _) = lax.fori_loop(
            0,  # start index
            n,  # end index
            kernel_thinning_body_fun,  # body function
            (
                first_coreset_indices,
                second_coreset_indices,
                sigma,
                original_array_masking,
                coresets_masking,
                self.random_key,
            ),
        )
        return (
            Coresubset.build(final_arr1, dataset),
            Coresubset.build(final_arr2, dataset),
        )

    def get_baseline_coreset(
        self, dataset: Data, baseline_coreset_size: int
    ) -> Coresubset[_Data]:
        """
        Generate a baseline coreset by randomly sampling from the dataset.

        :param dataset: The input dataset from which the baseline coreset is sampled.
        :param baseline_coreset_size: The number of dataset in the baseline coreset.
        :return: A randomly sampled baseline coreset with the specified size.
        """
        baseline_coreset, _ = RandomSample(
            coreset_size=baseline_coreset_size, random_key=self.random_key
        ).reduce(dataset)
        return baseline_coreset

    def kt_choose(
        self, candidate_coresets: list[Coresubset[_Data]], points: _Data
    ) -> Shaped[Array, " coreset_size"]:
        """
        Select the best coreset from a list of candidate coresets based on MMD.

        :param candidate_coresets: A list of candidate coresets to be evaluated.
        :param points: The original dataset against which the coresets are compared.
        :return: The coreset with the smallest MMD relative to the input dataset.
        """
        mmd = MMD(kernel=self.kernel)
        candidate_coresets_jax = jnp.array([c.points.data for c in candidate_coresets])
        candidate_coresets_indices = jnp.array([c.indices for c in candidate_coresets])
        mmd_values = jax.vmap(lambda c: mmd.compute(c, points))(candidate_coresets_jax)

        best_index = jnp.argmin(mmd_values)

        return candidate_coresets_indices[best_index]

    def kt_refine(
        self, candidate_coreset: Coresubset[_Data]
    ) -> tuple[Coresubset[_Data], None]:
        """
        Refine the selected candidate coreset.

        Use :meth:`~coreax.solvers.KernelHerding.refine` which achieves the result of
        looping through each element in coreset replacing that element with a point in
        the original dataset to minimise MMD in each step.

        :param candidate_coreset: The candidate coreset to be refined.
        :return: The refined coreset.
        """
        refined_coreset, _ = KernelHerding(
            coreset_size=self.coreset_size, kernel=self.kernel
        ).refine(candidate_coreset)
        return refined_coreset, None


class CompressPlusPlus(CoresubsetSolver[_Data, None], ExplicitSizeSolver):
    r"""
    Compress++ - A hierarchical coreset construction solver.

    `CompressPlusPlus` is an efficient method for building coresets without
    compromising performance significantly. It operates in two steps: recursively
    halving and thinning.

    In the recursive halving step, the dataset is partitioned into four subsets.
    Each subset is further divided into four subsets, repeated a predefined number
    of times. After this, each subset is halved using
    :class:`~coreax.solvers.KernelThinning`. The halved subsets are concatenated
    bottom-up to form a coreset.

    Finally, the resulting coreset is thinned again using
    :class:`~coreax.solvers.KernelThinning` to obtain a coreset of the desired size.
    This implementation is an adaptation of the Compress++ algorithm in
    :cite:`shetty2022compress` to make it an explicit sized solver.

    :param g: The oversampling factor.
    :param kernel: A :class:`~coreax.kernels.ScalarValuedKernel` for kernel thinning.
    :param random_key: A random number generator key for the kernel thinning solver,
        ensuring reproducibility of probabilistic components in the algorithm.
    :param delta: A float between 0 and 1, representing the swapping probability during
        the dataset splitting. A recommended value is :math:`1 / \log(\log(n))`, where
        :math:`n` is the size of the original dataset.
    :param sqrt_kernel: A :class:`~coreax.kernels.ScalarValuedKernel` instance defining
        the square root kernel used in the kernel thinning solver.
    """

    g: int
    kernel: ScalarValuedKernel
    random_key: KeyArrayLike
    delta: float
    sqrt_kernel: ScalarValuedKernel

    def reduce(
        self, dataset: _Data, solver_state: None = None
    ) -> tuple[Coresubset[_Data], None]:
        """
        Reduce a dataset to a coreset of the desired size using a hierarchical approach.

        This method reduces the dataset by recursively partitioning it and applying
        kernel thinning. The dataset is clipped to the nearest power of four before
        processing. The dataset is partitioned into four subsets. Each subset is further
        divided into four subsets, repeated until we have a coreset of predefined size.
        After this, the partitioned are recursively concatenated (4 at a time) and
        halved using :meth:`~coreax.solvers.KernelThinning.reduce`. Finally, the
        resulting coreset is thinned again using
        :meth:`~coreax.solvers.KernelThinning.reduce` to obtain a coreset of the desired
        size.

        :param dataset: The original dataset to be reduced.
        :param solver_state: The state of the solver.
        :return: A tuple containing the final coreset and the solver state (None).
        """
        n = len(dataset)
        if self.coreset_size > len(dataset):
            raise ValueError(MSG)

        # Check that depth and coreset_size are compatible
        nearest_power_of_4 = math.floor(math.log(n, 4))
        effective_data_size = 4**nearest_power_of_4
        if not 0 <= self.g <= nearest_power_of_4:
            raise ValueError(
                f"The over-sampling factor g should be between 0 "
                f"and {nearest_power_of_4}, inclusive."
            )

        if not self.coreset_size <= 2**self.g * math.sqrt(effective_data_size):
            raise ValueError(
                f"Coreset size and g are not compatible with the dataset size. Ensure "
                f"that the coreset size does not exceed "
                f"{2**self.g * math.sqrt(effective_data_size)} or increase g."
            )
        # Clip the dataset to the nearest power of 4
        clipped_indices = jax.random.choice(
            self.random_key, n, shape=(effective_data_size,), replace=False
        )

        def _compress_half(indices: Array) -> Array:
            """
            Compress the dataset to half its size using kernel thinning.

            Kernel thinning is used here but any other halving function could have been
            used.

            :param indices: The indices of current dataset with respect to the original
                dataset.
            :return: The indices of the halved dataset.
            """
            m = len(indices)
            data_to_half = dataset[indices]
            thinning_solver = KernelThinning(
                coreset_size=m // 2,
                delta=self.delta,
                kernel=self.kernel,
                sqrt_kernel=self.sqrt_kernel,
                random_key=self.random_key,
            )
            halved_coreset, _ = thinning_solver.reduce(data_to_half)
            return indices[halved_coreset.unweighted_indices]

        def _compress_thin(indices: Array) -> Array:
            """
            Compress the dataset to required size using kernel thinning.

            Kernel thinning is used here but any other thinning function could have been
            used.

            :param indices: The indices of current dataset with respect to the original
                dataset.
            :return: The indices of the thinned dataset.
            """
            data_to_half = dataset[indices]
            thinning_solver = KernelThinning(
                coreset_size=self.coreset_size,
                delta=self.delta,
                kernel=self.kernel,
                sqrt_kernel=self.sqrt_kernel,
                random_key=self.random_key,
            )
            halved_coreset, _ = thinning_solver.reduce(data_to_half)
            return indices[halved_coreset.unweighted_indices]

        def _compress(indices: Array) -> Array:
            """
            Apply the compress algorithm from :cite:`shetty2022compress`.

            :param indices: The indices of current dataset with respect to the original
                dataset.
            :return: The indices of the compressed dataset.
            """
            m = len(indices)
            # Base case: If m = 4^g, return the dataset
            if m == 4**self.g:
                return indices

            quarter_size = m // 4
            subsequences = jnp.reshape(indices, (4, quarter_size))
            # Recursive call to compress each subsequence
            compressed_subsequences = jax.vmap(_compress)(subsequences)
            # Concatenate the compressed subsequences
            concatenated = jnp.concatenate(compressed_subsequences)

            # Apply the halving function to the concatenated result
            return _compress_half(concatenated)

        def _compress_plus_plus(indices: Array) -> Array:
            """
            Apply the compress++ algorithm from :cite:`shetty2022compress`.

            :param indices: The indices of current dataset with respect to the original
                dataset.
            :return: The indices of the compressed dataset.
            """
            return _compress_thin(_compress(indices))

        plus_plus_indices = _compress_plus_plus(clipped_indices)
        return Coresubset(Data(plus_plus_indices), dataset), None
