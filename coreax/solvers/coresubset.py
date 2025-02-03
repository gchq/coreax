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
from typing import Optional, TypeVar, Union

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
from coreax.data import Data, SupervisedData, as_data, as_supervised_data
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

MSG = "'coreset_size' must be less than 'len(dataset)' by definition of a coreset"


def _ensure_positive(value: Union[float, int], name: str) -> float:
    """
    Ensure a value is positive and convert it to float.

    :param value: The value to validate
    :param name: Name of the parameter (for error message)
    :return: The validated value as float
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value)}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return float(value)


def _initial_coresubset(
    fill_value: int, coreset_size: int, dataset: _Data
) -> Coresubset[_Data]:
    """Generate a coresubset with `fill_value` valued and zero-weighted indices."""
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
    # not support refinement, the penalty could be initialised to the zeros vector, and
    # the result would be invariant to the initial coresubset.
    data, data_weights = jtu.tree_leaves(coresubset.pre_coreset_data)
    init_kernel_similarity_penalty = init_coreset_size * kernel.compute_mean(
        data,
        padded_coresubset.coreset,
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
    temperature: float = eqx.field(
        default=1.0,
        # Ensure temperature is positive to avoid degenerate behaviour
        converter=lambda x: _ensure_positive(x, "temperature"),
    )
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
        :meth:`~coreax.kernels.ScalarValuedKernel.compute_mean`
    :param unroll: Unroll parameter passed to
        :meth:`~coreax.kernels.ScalarValuedKernel.compute_mean`
    """

    kernel: ScalarValuedKernel
    score_matching: Optional[ScoreMatching] = None
    unique: bool = True
    regularise: bool = True
    regulariser_lambda: Optional[float] = None
    block_size: Optional[Union[int, tuple[Optional[int], Optional[int]]]] = None
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

        .. note::
            Only the score function, :math:`\nabla \log p(x)`, is provided to the
            solver. Since the lambda regularisation term relies on the density,
            :math:`p(x)`, directly, it is estimated using a Gaussian kernel density
            estimator.

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
        coreset_indices = initial_coresubset.unweighted_indices

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

        approximation_matrix = jnp.zeros((num_data_points, self.coreset_size))
        init_state = (gramian_diagonal, approximation_matrix, coreset_indices)
        output_state = jax.lax.fori_loop(0, self.coreset_size, _greedy_body, init_state)
        gramian_diagonal, _, updated_coreset_indices = output_state
        updated_coreset: Coresubset[_Data] = Coresubset.build(
            updated_coreset_indices, dataset
        )
        return updated_coreset, RPCholeskyState(gramian_diagonal)


class GreedyKernelPointsState(eqx.Module):
    """
    Intermediate :class:`GreedyKernelPoints` solver state information.

    :param feature_gramian: Cached feature kernel gramian matrix, should be padded with
        an additional row and column of zeros.
    """

    feature_gramian: Array


def _greedy_kernel_points_loss(
    candidate_coresets: Shaped[Array, " batch_size coreset_size"],
    responses: Shaped[Array, " n+1 1"],
    feature_gramian: Shaped[Array, " n+1 n+1"],
    regularisation_parameter: float,
    identity: Shaped[Array, " coreset_size coreset_size"],
    least_squares_solver: RegularisedLeastSquaresSolver,
) -> Shaped[Array, " batch_size"]:
    """
    Given an array of candidate coreset indices, compute the greedy KIP loss for each.

    Primarily intended for use with :class:`GreedyKernelPoints`.

    :param candidate_coresets: Array of indices representing all possible "next"
        coresets
    :param responses: Array of responses
    :param feature_gramian: Feature kernel gramian
    :param regularisation_parameter: Regularisation parameter for stable inversion of
        array, negative values will be converted to positive
    :param identity: Identity array used to regularise the feature gramians
        corresponding to each coreset. For :meth:`GreedyKernelPoints.reduce`
        this array is a matrix of zeros except for ones on the diagonal up to the
        current size of the coreset. For :meth:`GreedyKernelPoints.refine` this
        array is a standard identity array.
    :param least_squares_solver: Instance of
        :class:`coreax.least_squares.RegularisedLeastSquaresSolver`

    :return: :class`GreedyKernelPoints` loss for each candidate coreset
    """
    # Extract all the possible "next" coreset feature gramians, cross feature gramians
    # and coreset response vectors.
    coreset_gramians = feature_gramian[
        (candidate_coresets[:, :, None], candidate_coresets[:, None, :])
    ]
    coreset_cross_gramians = feature_gramian[candidate_coresets, :-1]
    coreset_responses = responses[candidate_coresets]

    # Solve for the kernel ridge regression coefficients for each possible coreset
    coefficients = least_squares_solver.solve_stack(
        arrays=coreset_gramians,
        regularisation_parameter=regularisation_parameter,
        targets=coreset_responses,
        identity=identity,
    )

    # Compute the loss function, making sure that we remove the padding on the responses
    predictions = (coreset_cross_gramians * coefficients).sum(axis=1)
    loss = ((predictions - 2 * responses[:-1, 0]) * predictions).sum(axis=1)
    return loss


class GreedyKernelPoints(
    RefinementSolver[SupervisedData, GreedyKernelPointsState],
    ExplicitSizeSolver,
):
    r"""
    Apply `GreedyKernelPoints` to a supervised dataset.

    `GreedyKernelPoints` is a deterministic, iterative and greedy approach to
    build a coreset adapted from the inducing point method developed in
    :cite:`nguyen2021meta`.

    Given one has an original dataset :math:`\mathcal{D}^{(1)} = \{(x_i, y_i)\}_{i=1}^n`
    of :math:`n` pairs with :math:`x\in\mathbb{R}^d` and :math:`y\in\mathbb{R}^p`, and
    one has selected :math:`m` data pairs :math:`\mathcal{D}^{(2)} = \{(\tilde{x}_i,
    \tilde{y}_i)\}_{i=1}^m` already for their compressed representation of the original
    dataset, `GreedyKernelPoints` selects the next point to minimise the loss

    .. math::

        L(\mathcal{D}^{(1)}, \mathcal{D}^{(2)}) = ||y^{(1)} -
        K^{(12)}(K^{(22)} + \lambda I_m)^{-1}y^{(2)} ||^2_{\mathbb{R}^n}

    where :math:`y^{(1)}\in\mathbb{R}^n` is the vector of responses from
    :math:`\mathcal{D}^{(1)}`, :math:`y^{(2)}\in\mathbb{R}^n` is the vector of responses
    from :math:`\mathcal{D}^{(2)}`,  :math:`K^{(12)} \in \mathbb{R}^{n\times m}` is the
    cross-matrix of kernel evaluations between :math:`\mathcal{D}^{(1)}` and
    :math:`\mathcal{D}^{(2)}`, :math:`K^{(22)} \in \mathbb{R}^{m\times m}` is the
    kernel matrix on :math:`\mathcal{D}^{(2)}`,
    :math:`\lambda I_m \in \mathbb{R}^{m \times m}` is the identity matrix and
    :math:`\lambda \in \mathbb{R}_{>0}` is a regularisation parameter.

    This class works with all children of :class:`~coreax.kernels.ScalarValuedKernel`.
    Note that `GreedyKernelPoints` does not support non-uniform weights and will only
    return coresubsets with uniform weights.

    :param random_key: Key for random number generation
    :param feature_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` on the
        feature space
    :param regularisation_parameter: Regularisation parameter for stable inversion of
        the feature Gramian
    :param unique: If :data:`False`, the resulting coresubset may contain the same point
        multiple times. If :data:`True` (default), the resulting coresubset will not
        contain any duplicate points
    :param batch_size: An integer representing the size of the batches of data pairs
        sampled at each iteration for consideration for adding to the coreset. If
        :data:`None` (default), the search is performed over the entire dataset
    :param least_squares_solver: Instance of
        :class:`coreax.least_squares.RegularisedLeastSquaresSolver`, default value of
        :data:`None` uses :class:`coreax.least_squares.MinimalEuclideanNormSolver`
    """

    random_key: KeyArrayLike
    feature_kernel: ScalarValuedKernel
    regularisation_parameter: float = 1e-6
    unique: bool = True
    batch_size: Optional[int] = None
    least_squares_solver: Optional[RegularisedLeastSquaresSolver] = None

    @override
    def reduce(
        self,
        dataset: SupervisedData,
        solver_state: Union[GreedyKernelPointsState, None] = None,
    ) -> tuple[Coresubset[SupervisedData], GreedyKernelPointsState]:
        initial_coresubset = _initial_coresubset(-1, self.coreset_size, dataset)
        return self.refine(initial_coresubset, solver_state)

    def refine(  # noqa: PLR0915
        self,
        coresubset: Coresubset[SupervisedData],
        solver_state: Union[GreedyKernelPointsState, None] = None,
    ) -> tuple[Coresubset[SupervisedData], GreedyKernelPointsState]:
        """
        Refine a coresubset with 'GreedyKernelPointsState'.

        We first compute the various factors if they are not given in the
        `solver_state`, and then iteratively swap points with the initial coreset,
        selecting points which minimise the loss.

        .. warning::

            If the input ``coresubset`` is smaller than the requested ``coreset_size``,
            it will be padded with indices that are invariant with respect to the loss.
            After the input ``coresubset`` has been refined, new indices will be chosen
            to fill the padding. If the input ``coresubset`` is larger than the
            requested ``coreset_size``, the extra indices will not be optimised and will
            be clipped from the return ``coresubset``.

        :param coresubset: The coresubset to refine
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :return: A refined coresubset and relevant intermediate solver state information
        """
        # Handle default value of None
        if self.least_squares_solver is None:
            least_squares_solver = MinimalEuclideanNormSolver()
        else:
            least_squares_solver = self.least_squares_solver

        # If the initialisation coresubset is too small, pad its nodes up to
        # 'output_size' with -1 valued indices. If it is too large, raise a warning and
        # clip off the indices at the end.
        if self.coreset_size > len(coresubset):
            pad_size = max(0, self.coreset_size - len(coresubset))
            coreset_indices = jnp.hstack(
                (
                    coresubset.unweighted_indices,
                    -1 * jnp.ones(pad_size, dtype=jnp.int32),
                )
            )
        elif self.coreset_size < len(coresubset):
            coreset_indices = coresubset.unweighted_indices[: self.coreset_size]
        else:
            coreset_indices = coresubset.unweighted_indices

        # Extract features and responses
        dataset = as_supervised_data(coresubset.pre_coreset_data)
        num_data_pairs = len(dataset)
        x, y = dataset.data, dataset.supervision

        # Pad the response array with an additional zero to allow us to
        # extract sub-arrays and fill in elements with zeros simultaneously.
        padded_responses = jnp.vstack((y, jnp.array([[0]])))

        if solver_state is None:
            # Pad the gramian with zeros in an additional column and row
            padded_feature_gramian = jnp.pad(
                self.feature_kernel.compute(x, x), [(0, 1)], mode="constant"
            )
        else:
            padded_feature_gramian = solver_state.feature_gramian

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

        # Initialise an array that will let us extract arrays corresponding to every
        # possible candidate coreset.
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
            """Execute main loop of GreedyKernelPoints."""
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
            )

            # Find the optimal replacement coreset index, ensuring we don't pick an
            # already chosen point if we want the indices to be unique. Note we
            # must set the ith index to -1 for refinement purposes, as we are happy for
            # the current index to be retained if it is the best.
            if self.unique:
                already_chosen_indices_mask = jnp.isin(
                    candidate_coresets[:, i],
                    coreset_indices.at[i].set(-1),
                )
                loss += jnp.where(already_chosen_indices_mask, jnp.inf, 0)
            index_to_include_in_coreset = candidate_coresets[loss.argmin(), i]

            # Repeat the chosen coreset index into the ith column of the array of
            # candidate coreset indices. Replace the (i+1)th column with the next batch
            # of possible coreset indices.
            updated_candidate_coresets = (
                candidate_coresets.at[:, i]
                .set(index_to_include_in_coreset)
                .at[:, i + 1]
                .set(batch_indices[i + 1, :])
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

        return Coresubset.build(
            updated_coreset_indices, dataset
        ), GreedyKernelPointsState(padded_feature_gramian)


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
            subset1_indices = (
                subset1.indices.data.flatten()
            )  # Indices relative to parent
            subset2_indices = (
                subset2.indices.data.flatten()
            )  # Indices relative to parent

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
        print(type(dataset))
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
