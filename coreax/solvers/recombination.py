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
Recombination solvers.

Take a dataset :math:`\{(x_i, w_i)\}_{i=1}^n`, where each node :math:`x_i \in \Omega`
is paired with a weight :math:`w_i \in \mathbb{R} \ge 0`, and the sum of all weights
is one, :math:`\sum_{i=1}^n w_i = 1` (a strict requirement for a probability measure).

.. note::
    Given any weighted dataset, we can use normalisation to satisfy the sum to
    one condition, providing :math:`\sum_{i=1}^n w_i \neq 0`.

Combined with :math:`m-1` test-functions :math:`\Phi^\prime = \{\phi_i\}_{i=1}^{m-1}`,
where :math:`\phi_i \colon \Omega \to \mathbb{R}`, that parametrise a set of :math:`m`
test-functions :math:`\Phi = \{x \mapsto 1\} \cup \Phi^\prime`, there exists a dataset
push-forward measure :math:`\mu_n := \Phi_* \nu_n`.

A recombination solver attempts to find a reduced measure (a coresubset)
:math:`\hat{\mu}_{m^\prime}`, which is given as a basic-feasible solution (BFS) to the
following linear-programming problem (with trivial objective)

.. math::
    \begin{align}
        \mathbf{Y} \mathbf{\hat{w}} &= \text{CoM}(\mu_n),\\
        \mathbf{\hat{w}} &\ge 0,
    \end{align}

where the system variables and "centre-of-mass" are defined as

.. math::
    \begin{gather}
    \mathbf{Y} := \left[\Phi(x_1), \dots, \Phi(x_n)\right] \in \mathbb{R}^{m \times n},\
    \mathbf{\hat{w}} \in \mathbb{R}^n \ge 0,\\
    \text{CoM}(\mu_n) := \sum_{i=1}^n w_i \Phi(x_i)
            = \left[ \sum_{i=1}^n w_i \phi_j(x_i) \right]_{j=1}^m \in \mathbb{R}^m.\\
    \end{gather}

.. note::
    The source dataset is, by definition, a solution to the linear-program that is not
    necessarily a BFS. Hence, one may consider the fundamental problem of recombination
    as that of finding a BFS given a solution that is not a BFS.

Basic feasible solutions to the linear-program above are of the form
:math:`\mathbf{\hat{w}} = \{\hat{w}_1, \dots, \hat{w}_{m^\prime}, \mathbf{0}\}`; I.E.
BFSs are feasible solutions with :math:`n-m^\prime` weights equal to zero. Given a BFS,
the reduced measure (the coresubset) can be constructed by explicitly removing the nodes
associated with each zero valued (implicitly removed) weight

.. math::
    \begin{gather}
    \hat{\nu}_{m^\prime} = \sum_{i \in I} \hat{w_i} \delta_{x_i},\\
    I = \{i \mid \hat{w_i} \neq 0\, \forall i \in \{1, \dots, n\}\}.
    \end{gather}

Due to Tchakaloff's theorem :cite:`tchakaloff1957,bayer2006tchakaloff`, which follows
from Caratheodory's convex hull theorem :cite:`caratheodory1907,loera2018caratheodory`,
we know there always exists a basic-feasible solution to the linear-program, with at
most :math:`m^\prime = \text{dim}(\text{span}(\Phi))` non-zero weights. Hence, we have
an upper bound on the size of a coresubset, controlled by the choice of test-functions.

.. note::
    A basic feasible solution (coresubset produced by recombination) is non-unique. In
    fact, there exists :math:`\binom{n}{m^\prime}` basic feasible solutions
    (coresubsets) for the described linear-program. In the context of Coreax, this means
    that a :class:`RecombinationSolver` is unlikely to ever be truly invariant to the
    presence of padding (see :class:`~coreax.solvers.PaddingInvariantSolver`). I.E. the
    padded problem may have an equivalent, but different BFS than the unpadded problem.

Canonically, recombination is used for reducing the support of a quadrature/cubature
measure, against which integration of any function :math:`f \in \text{span}(\Phi)`
is identical to integration against a "target" (potentially continuous) measure
:math:`\mu`.
"""

import math
from collections.abc import Callable
from typing import Generic, Literal, NamedTuple, Optional, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
from jaxtyping import Array, Bool, DTypeLike, Float, Integer, Real, Shaped
from typing_extensions import override

from coreax import Coresubset, Data
from coreax.solvers.base import CoresubsetSolver

_Data = TypeVar("_Data", bound=Data)
_State = TypeVar("_State")


class RecombinationSolver(CoresubsetSolver[_Data, _State], Generic[_Data, _State]):
    r"""
    Solver which returns a :class:`~coreax.coreset.Coresubset` via recombination.

    Given :math:`m-1` explicitly provided test-functions :math:`\Phi^\prime`, a
    recombination solver finds a coresubset with :math:`m^\prime \le m` points, whose
    push-forward :math:`\hat{\mu}_{m^\prime}` has the same "centre-of-mass" (CoM) as
    the dataset push-forward :math:`\mu_n := \Phi_* \nu_n`.

    :param test_functions: A callable that applies a set of specified test-functions
        :math:`\Phi^\prime = \{\phi_1,\dots,\phi_{m-1}\}` where each function is a map
        :math:`\phi_i \colon \Omega\to\mathbb{R}`; a value of :data:`None` implies the
        identity map :math:`\Phi^\prime \colon x \mapsto x`, and necessarily assumes
        that :math:`x \in \Omega \subseteq \mathbb{R}^{m-1}`
    :param mode: 'implicit-explicit' explicitly removes :math:`n - m` points, yielding
        a coreset of size :math:`m`, with :math:`m - m^\prime` zero-weighted (implicitly
        removed) points; 'implicit' explicitly removes no points, yielding a coreset of
        size :math:`n` with :math:`n - m^\prime` zero-weighted (implicitly removed)
        points; 'explicit' explicitly removes :math:`n - m^\prime` points, yielding a
        coreset of size :math:`m^\prime`, but unlike the other methods is not JIT
        compatible as the coreset size :math:`m^\prime` is unknown at compile time.
    """

    test_functions: Optional[Callable[[Array], Real[Array, " m-1"]]] = None
    mode: Literal["implicit-explicit", "implicit", "explicit"] = "implicit-explicit"

    def __check_init__(self):
        """Ensure a valid `self.mode` is specified."""
        if self.mode not in {"implicit-explicit", "implicit", "explicit"}:
            raise ValueError(
                "Invalid mode, expected 'implicit-explicit', 'implicit' or 'explicit'."
            )


class _EliminationState(NamedTuple):
    weights: Shaped[Array, " n"]
    nodes: Shaped[Array, "n m"]
    iteration: int


class CaratheodoryRecombination(RecombinationSolver[Data, None]):
    r"""
    Recombination via Caratheodory measure reduction (Gaussian-Elimination).

    Proposed in :cite:`tchernychova2016recombination` (see Chapter 1.3.3.3) as an
    alternative to the Simplex algorithm for solving the recombination problem.

    Unlike the Simplex method, with time complexity :math:`\mathcal{O}(m^3 n + m n^2)`,
    Caratheodory recombination has time complexity of only :math:`\mathcal{O}(m n^2)`.

    .. note::
        Given :math:`n = cm`, for a rational constant :math:`c`, the above complexities
        can be alternatively represented as :math:`\mathcal{O}(m^4)` for the Simplex
        method and :math:`\mathcal{O}(m^3)` for Caratheodory recombination.

    :param test_functions: A callable that applies a set of specified test-functions
        :math:`\Phi^\prime = \{\phi_1,\dots,\phi_{m-1}\}` where each function is a map
        :math:`\phi_i \colon \Omega \to \mathbb{R}`; a value of non implies the identity
        map :math:`\Phi^\prime \colon x \mapsto x`, and necessarily assumes that
        :math:`x \in \Omega \subseteq \mathbb{R}^{m-1}`
    :param mode: 'implicit-explicit' explicitly removes :math:`n - m` points, yielding
        a coreset of size :math:`m`, with :math:`m - m^\prime` zero-weighted (implicitly
        removed) points; 'implicit' explicitly removes no points, yielding a coreset of
        size :math:`n` with :math:`n - m^\prime` zero-weighted (implicitly removed)
        points; 'explicit' explicitly removes :math:`n - m^\prime` points, yielding a
        coreset of size :math:`m^\prime`, but unlike the other methods is not JIT
        compatible as the coreset size :math:`m^\prime` is unknown at compile time.
    :param rcond: A relative condition number; any singular value :math:`s` below the
        threshold :math:`\text{rcond} * \text{max}(s)` is treated as equal to zero; if
        rcond is :data:`None`, it defaults to `floating point eps * max(n, d)`
    """

    rcond: Optional[float] = None

    @override
    def reduce(
        self, dataset: Data, solver_state: None = None
    ) -> tuple[Coresubset, None]:
        nodes, weights = jtu.tree_leaves(dataset.normalize(preserve_zeros=True))
        push_forward_nodes = _push_forward(nodes, self.test_functions)
        # Handle pre-existing zero-weighted nodes (not handled by the base algorithm
        # described in :cite:`tchernychova2016recombination`)
        safe_push_forward_nodes, safe_weights, indices = _co_linearize(
            push_forward_nodes, weights
        )
        largest_null_space_basis, null_space_rank = _resolve_null_basis(
            safe_push_forward_nodes, self.rcond
        )

        def _eliminate_cond(state: _EliminationState) -> Bool[Array, ""]:
            """
            If to continue the iterative Gaussian-Elimination procedure.

            On each iteration, we eliminate a basis vector from the left null space. We
            repeat until all basis vectors have been eliminated (the dimension of the
            null space is zero); once the number of iterations is the same as the rank
            of the original null space.

            .. note::
                The reason for using a while loop, rather than scanning over the basis
                vectors, is due to the dimension of the null space being unknown at JIT
                compile time, preventing us from slicing the left singular vectors down
                to only those which form a basis for the left null space.

            :param state: Elimination state information
            :return: Boolean indicating if to continue/exit the elimination loop.
            """
            return state.iteration < null_space_rank

        def _eliminate(state: _EliminationState) -> _EliminationState:
            """
            Eliminate a basis from the left null space.

            At least one weight is zeroed (implicitly removed from the dataset), and one
            left null space basis vector eliminated on each iteration. The mass that is
            "lost" in weight zeroing/elimination is redistributed among the remaining
            non-zero weights to preserve the total mass/weight sum.

            If the procedure is repeated until all the left null space basis vectors
            are eliminated, the resulting weights (when combined with the original
            nodes) are a BFS to the recombination problem/linear-program.

            :param state: Elimination state information
            :return: Updated `state` information resulting from the elimination step.
            """
            # Algorithm 6 - Chapter 3.3 of :cite:`tchernychova2016recombination`
            # Our Notation -> Their Notation
            # - `basis_index` (loop iteration) -> i
            # - `elimination_index` -> k^{(i)}
            # - `elimination_rescaling_factor` -> \alpha_{(i)}
            # - `updated_weights` -> \underline\Beta^{(i)}
            # - `null_space_basis_update` -> d_{l+1}^{(i)}\phi_1^{(i-1)}
            # - `updated_null_space_basis` -> \Psi^{(i))
            _weights, null_space_basis, basis_index = state
            basis_vector = null_space_basis[basis_index]
            # Equation 3: Select the weight to eliminate.
            elimination_condition = jnp.where(
                basis_vector > 0, _weights / basis_vector, jnp.inf
            )
            elimination_index = jnp.argmin(elimination_condition)
            elimination_rescaling_factor = elimination_condition[elimination_index]
            # Equation 4: Eliminate the selected weight and redistribute its mass.
            # NOTE: Equation 5 is implicit from Equation 4 and is performed outside
            # of `_eliminate` via `_coresubset_nodes`.
            updated_weights = _weights - elimination_rescaling_factor * basis_vector
            updated_weights = updated_weights.at[elimination_index].set(0)
            # Equations 6, 7 and 8: Update the Null space basis.
            null_space_basis_update = jnp.tensordot(
                null_space_basis[:, elimination_index],
                basis_vector / basis_vector[elimination_index],
                axes=0,
            )
            updated_null_space_basis = null_space_basis - null_space_basis_update
            updated_null_space_basis = updated_null_space_basis.at[
                :, elimination_index
            ].set(0)
            return _EliminationState(
                updated_weights, updated_null_space_basis, basis_index + 1
            )

        in_state = _EliminationState(safe_weights, largest_null_space_basis, 0)
        out_weights, *_ = jax.lax.while_loop(_eliminate_cond, _eliminate, in_state)
        coresubset_nodes = _coresubset_nodes(
            safe_push_forward_nodes,
            out_weights,
            indices,
            self.mode,
            is_affine_augmented=True,
        )
        return Coresubset(coresubset_nodes, dataset), solver_state


def _push_forward(
    nodes: Shaped[Array, " n"],
    test_functions: Optional[Callable[[Array], Real[Array, " m-1"]]],
    augment: bool = True,
) -> Shaped[Array, "n m"]:
    r"""
    Push the 'nodes' forward through the 'test_functions'.

    :param nodes: The nodes to push-forward through the test-functions
    :param test_functions: A callable that applies a set of specified test-functions
        :math:`\Phi^\prime = \{\phi_1,\dots,\phi_{m-1}\}` where each function is a map
        :math:`\phi_i \colon \Omega \to \mathbb{R}`; a value of non implies the identity
        map :math:`\Phi^\prime \colon x \mapsto x`, and necessarily assumes that
        :math:`x \in \Omega \subseteq \mathbb{R}^{m-1}`
    :param augment: If to prepend the affine-augmentation test function
        :math:`\{x \mapsto 1\}` to the explicitly pushed forward nodes \Phi^\prime(x),
        to yield \Phi(x); default behaviour prepends the affine-augmentation function
    :return: The pushed-forward nodes.
    """
    if test_functions is None:
        push_forward_nodes = nodes
    else:
        push_forward_nodes = jax.vmap(test_functions, in_axes=0)(nodes)
    if augment:
        shape, dtype = push_forward_nodes.shape[0], push_forward_nodes.dtype
        affine_augmentation = jnp.ones((shape,), dtype)
        push_forward_nodes = jnp.c_[affine_augmentation, push_forward_nodes]
    return push_forward_nodes


def _co_linearize(
    nodes: Shaped[Array, "n m"], weights: Shaped[Array, " n"]
) -> tuple[Shaped[Array, "n m"], Shaped[Array, " n"], Shaped[Array, " n"]]:
    """
    Make zero-weighted nodes co-linear with the maximum weighted node.

    Due to the static shape requirements imposed by JAX, we implicitly remove nodes by
    setting their corresponding weight to zero. This is sufficient in the recombination
    algorithm for all but one scenario, the computation of the null space basis. Because
    the zero-weighted nodes still exist in the node matrix, they influence the SVD and
    yield an erroneous null space basis.

    We ameliorate this problem by setting the zero-weighted nodes equal (co-linear) to
    the largest weighted node (an arbitrary but consistent choice). Because the nodes
    are now co-linear to each other and the largest weighted node, we know that at least
    all but one of them can be safely eliminated by the recombination procedure. Thus,
    the nodes become effectively "invisible" to the elimination procedure.

    The only caveat is that we don't know which of the equal nodes will be retained post
    elimination. To handle this, we keep an index (reference) from the zero-weighted
    nodes to the largest weighted node, and we redistribute the largest weight equally
    over all the "co-linearized" nodes (preserving the CoM and allowing any node to be
    eliminated).

    :param nodes: The nodes to co-linearize
    :param weights: The weights to apply the co-linearization correction to
    :return: The co-linearized nodes, corrected weights, and co-linearized-to-original
        reference indices.
    """
    max_index = jnp.argmax(weights)
    non_zero_weights_mask = weights > 0
    zero_weights_mask = 1 - non_zero_weights_mask
    n_zeros = zero_weights_mask.sum()
    # Create a new set of indices that replace the zero-weighted node indices with the
    # maximum weighted node's index.
    indices = jnp.arange(weights.shape[0])
    indices *= non_zero_weights_mask
    indices += zero_weights_mask * max_index
    # Renormalize the maximum weight; ensures the weight sum is preserved under the new
    # (co-linearized) indices; prevents co-linearization from changing the weight sum.
    weights = weights.at[max_index].divide(n_zeros + 1)
    return nodes[indices], weights[indices], indices


# pylint: disable=line-too-long
# Credit: https://github.com/patrick-kidger/lineax/blob/9b923c8df6556551fedc7adeea7979b5c7b3ffb0/lineax/_solver/svd.py#L67  # noqa: E501
# for the rank determination code.
# pylint: enable=line-too-long
def _resolve_null_basis(
    nodes: Shaped[Array, "n m"],
    rcond: Union[float, None] = None,
) -> tuple[Shaped[Array, "n n"], Integer[Array, ""]]:
    r"""
    Resolve the largest left null space basis, and its rank, for passed the node matrix.

    By largest left null space basis, we mean the null space basis under the assumption
    that the rank of the null space is maximal (assumed to be ``n``). If the rank is not
    maximal, then only the first :math:`n - m^\prime` basis vectors will be actual basis
    vectors for the null space (where :math:`m^\prime` is the rank of the node matrix).
    The remaining "basis" vectors can, and should, be ignored in upstream computations
    by using the left null space rank value as a cut-off index.

    :param nodes: Matrix of nodes (m-vectors) whose null space is to be determined
    :param rcond: The relative condition number of the Matrix of nodes
    :return: The largest left null space basis and its rank, for the passed node matrix.
    """
    q, s, _ = jsp.linalg.svd(nodes, full_matrices=True)
    _rcond = _resolve_rcond(nodes.shape, s.dtype, rcond)
    if s.size > 0:
        _rcond *= jnp.max(s[0])
    mask = s > _rcond
    matrix_rank = sum(mask)
    null_space_rank = jnp.maximum(0, nodes.shape[0] - matrix_rank)
    largest_null_space_basis = q.T[::-1]
    return largest_null_space_basis, null_space_rank


# pylint: disable=line-too-long
# Credit: https://github.com/patrick-kidger/lineax/blob/9b923c8df6556551fedc7adeea7979b5c7b3ffb0/lineax/_misc.py#L34  # noqa: E501
# pylint: enable=line-too-long
def _resolve_rcond(
    shape: tuple[int, ...], dtype: DTypeLike, rcond: Optional[float] = None
) -> Float[Array, ""]:
    """
    Resolve the relative condition number (rcond).

    :param shape: The shape of the matrix whose relative condition number to resolved
    :param dtype: The element dtype of the matrix whose rcond is to be resolved
    :param rcond: The relative condition number of a given matrix; if ``None``,
        ``rcond = dtype_floating_point_eps * max(shape)``; else if negative,
        ``rcond = dtype_floating_point_eps``
    :return: The resolved relative condition number (rcond)
    """
    epsilon = jnp.asarray(jnp.finfo(dtype).eps, dtype)
    if rcond is None:
        return epsilon * max(shape)
    return jnp.where(rcond < jnp.asarray(0), epsilon, rcond)


def _coresubset_nodes(
    push_forward_nodes: Shaped[Array, "n m"],
    weights: Shaped[Array, " n"],
    indices: Shaped[Array, " n"],
    mode: Literal["implicit-explicit", "implicit", "explicit"],
    is_affine_augmented: bool = False,
) -> Data:
    r"""
    Determine the coresubset nodes based on the 'mode'.

    :param push_forward_nodes: The dataset push forward nodes
    :param weights: The coresubset weights
    :param mode: 'implicit-explicit' explicitly removes :math:`n - m` points, yielding
        a coreset of size :math:`m`, with :math:`m - m^\prime` zero-weighted (implicitly
        removed) points; 'implicit' explicitly removes no points, yielding a coreset of
        size :math:`n` with :math:`n - m^\prime` zero-weighted (implicitly removed)
        points; 'explicit' explicitly removes :math:`n - m^\prime` points, yielding a
        coreset of size :math:`m^\prime`, but unlike the other methods is not JIT
        compatible as the coreset size :math:`m^\prime` is unknown at compile time.
    :param is_affine_augmented: If the 'push_forward_nodes' include the :math:`\phi_1`
        affine-augmentation map.
    :return: The coresubset nodes as defined by the 'mode'.
    """
    n, m = push_forward_nodes.shape
    m = m if is_affine_augmented else m + 1
    if mode == "implicit-explicit":
        # Inside the JIT context we cannot explicitly remove all the non-zero
        # weights, because we don't know how many there will be a priori (`m^\prime`
        # is unknown until after the singular value decomposition is performed).
        # However, we do have an upper bound on the number of non-zero points
        # `min(n, m) \ge m^\prime`. Thus, we need only return the `min(n, m)` non-zero
        # weights where `min(n, m) - m^\prime` of these may be zero-weighted (implicitly
        # removed). The fill value is set to `argmin(weights)` to ensure we always index
        # a zero-weighted data point whenever the weight is zero.
        idx = jnp.flatnonzero(weights, size=min(n, m), fill_value=jnp.argmin(weights))
    elif mode == "implicit":
        idx = jnp.flatnonzero(weights, size=n, fill_value=jnp.argmin(weights))
    elif mode == "explicit":
        # Explicit mode is JIT incompatible
        try:
            idx = jnp.flatnonzero(weights)
        except jax.errors.ConcretizationTypeError as err:
            raise ValueError(
                "'explicit' mode is incompatible with transformations such as 'jax.jit'"
            ) from err
    else:
        # Should only get here if the `__check_init__`` has been skipped/avoided, or if
        # this function is called from an unexpected place.
        raise ValueError(
            "Invalid mode, expected 'implicit-explicit', 'implicit' or 'explicit'."
        )
    return Data(indices[idx], weights[idx])


class TreeRecombination(RecombinationSolver[Data, None]):
    r"""
    Tree recombination based coresubset solver.

    Based on Algorithm 7 Chapter 3.3 of :cite:`tchernychova2016recombination`, which
    is an order of magnitude more efficient than Algorithm 5 in Chapter 3.2, originally
    introduced in :cite:`litterer2012recombination`.

    The time complexity is of order :math:`\mathcal{O}(\log_2(\frac{n}{c_r m}) m^3)`,
    where :math`c_r` is the `tree_reduction_factor`. The time complexity can be
    equivalently expressed as :math:`\mathcal{O}(m^3)`, using the same arguments as used
    in :class:`CaratheodoryRecombination`.

    .. note::
        As the ratio of :math:`n / m` grows, the constant factor for the time complexity
        of :class:`TreeRecombination` increases at a logarithmic rate, rather than at a
        quadratic rate for plain :class:`CaratheodoryRecombination`. Hence, in general,
        we would expect :class:`TreeRecombination` to be the more efficient choice for
        all but the smallest values of :math:`n / m`.

    :param test_functions: the map :math:`\Phi^\prime = \{ \phi_1, \dots, \phi_{M-1} \}`
        where each :math:`\phi_i \colon \Omega \to \mathbb{R}` represents a linearly
        independent test-function; a value of :data:`None` implies the identity function
        (necessarily assuming :math:`\Omega \subseteq \mathbb{R}^{M-1}`)
    :param mode: 'implicit-explicit' explicitly removes :math:`n - m` points, yielding
        a coreset of size :math:`m`, with :math:`m - m^\prime` zero-weighted (implicitly
        removed) points; 'implicit' explicitly removes no points, yielding a coreset of
        size :math:`n` with :math:`n - m^\prime` zero-weighted (implicitly removed)
        points; 'explicit' explicitly removes :math:`n - m^\prime` points, yielding a
        coreset of size :math:`m^\prime`, but unlike the other methods is not JIT
        compatible as the coreset size :math:`m^\prime` is unknown at compile time.
    :param rcond: a relative condition number; any singular value :math:`s` below the
        threshold :math:`\text{rcond} * \text{max}(s)` is treated as equal to zero; if
        :code:`rcond is None`, it defaults to `floating point eps * max(n, d)`
    :param tree_reduction_factor: The factor by which each tree reduction step reduces
        the number of non-zero points; the remaining number of non-zero nodes, after
        performing recombination, is equal to `n_nodes / tree_reduction_factor`;
    """

    rcond: Union[float, None] = None
    tree_reduction_factor: int = 2

    @override
    def reduce(
        self, dataset: Data, solver_state: None = None
    ) -> tuple[Coresubset, None]:
        nodes, weights = jtu.tree_leaves(dataset.normalize(preserve_zeros=True))
        # Push the nodes forward through the test-functions \Phi^\prime.
        push_forward_nodes = _push_forward(nodes, self.test_functions, augment=False)
        n, m = push_forward_nodes.shape
        # We don't apply the affine-augmentation test-function \phi_1 here, instead
        # deferring it to `CaratheodoryRecombination.reduce`. Thus, we have to manually
        # correct the value for `m`.
        padding, count, depth = _prepare_tree(n, m + 1, self.tree_reduction_factor)
        car_recomb_solver = CaratheodoryRecombination(rcond=self.rcond, mode="implicit")

        def _tree_reduce(_, state: tuple[Array, Array]) -> tuple[Array, Array]:
            """
            Apply Tree-Based Caratheodory Recombination (Gaussian-Elimination).

            Partitions the dataset into 'count' clusters of size 'n / count' and then
            computes the cluster centroids. Caratheodory recombination is then performed
            on these centroids (rather than on the full dataset), with every node in the
            eliminated centroids' cluster being implicitly removed (given zero-weight).

            There are 'tree_reduction_factor * m' clusters, with each step reducing the
            number of remaining clusters down to 'm'. We can repeat the process until
            each cluster contains, at most, a single non-zero weighted point (at this
            point the recombination problem has been solved).

            :param _: Not used
            :param state: Tuple of node weights and indices; indices are passed to keep
                a correspondence between the original data indices and
            :return: Updated tuple of node weights and indices; weights are zeroed
                (implicitly removed) where appropriate; indices are shuffled to ensure
                balanced centroids in subsequent iterations (centroids are balanced when
                they are all constructed from subsets with as near to an equal number
                of non-zero weighted nodes as possible).
            """
            _weights, _indices = state
            # Index weights to a centroid; argsort ensures that centroids are balanced.
            centroid_indices = jnp.argsort(_weights).reshape(count, -1, order="F")
            centroid_nodes, centroid_weights = _centroid(
                push_forward_nodes[_indices[centroid_indices]],
                _weights[centroid_indices],
            )
            centroid_dataset = Data(centroid_nodes, centroid_weights)
            # Solve the measure reduction problem on the centroid dataset.
            centroid_coresubset, _ = car_recomb_solver.reduce(centroid_dataset)
            coresubset_indices = centroid_coresubset.unweighted_indices
            coresubset_weights = centroid_coresubset.points.weights
            # Propagate centroid coresubset weights to the underlying weights for each
            # centroid, as defined by `centroid_indices`.
            weight_update_indices = centroid_indices[coresubset_indices]
            weight_update = coresubset_weights / centroid_weights[coresubset_indices]
            updated_weights = _weights[weight_update_indices] * weight_update[..., None]
            # Maintain a correspondence between the original data indices and the sorted
            # indices, used to construct the balanced centroids.
            updated_indices = _indices[weight_update_indices.reshape(-1, order="F")]
            return updated_weights.reshape(-1, order="F"), updated_indices

        in_state = (jnp.pad(weights, (0, padding)), jnp.arange(n + padding))
        out_weights, indices = jax.lax.fori_loop(0, depth, _tree_reduce, in_state)
        coresubset_nodes = _coresubset_nodes(
            push_forward_nodes, out_weights, indices, self.mode
        )
        return Coresubset(coresubset_nodes, dataset), solver_state


def _prepare_tree(
    n: int, m: int, tree_reduction_factor: int = 2
) -> tuple[int, int, int]:
    r"""
    Compute and apply dataset padding and compute tree count and depth.

    :param n: Number of nodes
    :param m: Number of test-functions
    :param tree_reduction_factor: The factor by which each tree reduction step reduces
        the number of non-zero points; the remaining number of non-zero nodes, after
        performing recombination, is equal to `n_nodes / tree_reduction_factor`
    :return: The required amount of padding, to allow reshaping of the nodes into equal
        sized clusters), the tree_count (number of clusters), and the maximum tree depth
        (number of tree_reduction iterations required to complete tree recombination)
    """
    tree_count = tree_reduction_factor * m
    max_tree_depth = max(1, math.ceil(math.log(n / m, tree_reduction_factor)))
    padding = m * tree_reduction_factor**max_tree_depth - n
    return padding, tree_count, max_tree_depth


@jax.vmap
def _centroid(
    nodes: Shaped[Array, "tree_count n/tree_count m"],
    weights: Shaped[Array, "tree_count n/tree_count"],
) -> tuple[Shaped[Array, "n/tree_count m"], Shaped[Array, " n/tree_count"]]:
    """
    Compute the centroid mass and node centre (centre-of-mass).

    :param nodes: A set of clustered nodes where the leading axis indexes each cluster,
        which this function vmaps over, and the middle axis indexes each node within a
        given cluster.
    :param weights: A set of clustered weights associated with each node; has the same
        index layout as the nodes.
    :return: Cluster centroid (centre-of-mass) and total cluster mass for all clusters
    """
    centroid_nodes = jnp.nan_to_num(jnp.average(nodes, 0, weights))
    centroid_weights = jnp.sum(weights)
    return centroid_nodes, centroid_weights
