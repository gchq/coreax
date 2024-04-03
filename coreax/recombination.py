r"""
Recombination algorithms.

Algorithms to reduce discrete measures.
"""
from __future__ import annotations

import math
import warnings
from collections import namedtuple
from collections.abc import Callable
from typing import cast, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
import scipy
from jaxtyping import Array, ArrayLike, DTypeLike, Inexact, Int, Shaped

InexactScalarLike = Inexact[ArrayLike, ""]
IntScalar = Int[Array, ""] | int


# pylint: disable=line-too-long
# Credit: https://github.com/patrick-kidger/lineax/blob/9b923c8df6556551fedc7adeea7979b5c7b3ffb0/lineax/_misc.py#L34  # noqa: E501
# pylint: enable=line-too-long
def _resolve_rcond(
    rcond: InexactScalarLike | None, shape: tuple[int, ...], dtype: DTypeLike
) -> InexactScalarLike:
    """
    Resolve the relative condition number.

    :param rcond: the relative condition number of a given matrix; if ``None``,
        ``rcond = dtype_floating_point_eps * max(shape)``; else if negative,
        ``rcond = dtype_floating_point_eps``.
    :param shape: the shape of the matrix  whose relative is to be resolved
    :param dtype: the element dtype of the matrix whose rcond is to be resolved
    :return: the resolved relative condition number (rcond).
    """
    if rcond is None:
        return jnp.finfo(dtype).eps * max(shape)
    return jnp.where(rcond < jnp.asarray(0), jnp.finfo(dtype).eps, rcond)


# pylint: disable=line-too-long
# Credit: https://github.com/patrick-kidger/lineax/blob/9b923c8df6556551fedc7adeea7979b5c7b3ffb0/lineax/_solver/svd.py#L67  # noqa: E501
# pylint: enable=line-too-long
def reveal_left_null_space_rank(
    matrix_shape: tuple[int, ...],
    singular_values: Shaped[Array, " n"],
    rcond: InexactScalarLike | None = None,
) -> IntScalar:
    r"""
    Reveal the null space rank for a given matrix with given singular values.

    The rank of a matrix :math:`A` is the count of its non-zero singular values.
    More precisely, in finite-precision arithmetic, it is the count of singular values,
    :math:`s > \text{rcond}`. The relative condition number (rcond), multiplied by the
    largest singular value, sets an effective rounding threshold below which a singular
    value is treated as zero.

    The null space rank is the difference between the number of rows in the matrix,
    minus the matrix rank. I.E.

    .. math::

        \text{rank}(\text{null}(A)) = \text{max}(\text{shape}(A)) - \text{rank}(A).

    :param matrix_shape: the shape of an :math:`n \times d` matrix :math:`A`
    :param singular_values: an array with shape :math:`\text{min}(n, d)`
    :param rcond: a relative condition number. Any singular value :math:`s` below the
        threshold :math:`\text{rcond} * \text{max}(s)` is treated as equal to zero. If
        :code:`rcond is None`, it defaults to `floating point eps * max(n, d)`
    :return: the rank of the left null space of a matrix :math:`A` with given shape and
        given singular values.
    """
    rcond = _resolve_rcond(rcond, matrix_shape, singular_values.dtype)
    if singular_values.size > 0:
        rcond = rcond * jnp.max(singular_values[0])
    mask = singular_values > rcond
    matrix_rank = sum(mask)
    return jnp.maximum(0, matrix_shape[0] - matrix_rank)


class AbstractCoreSet(eqx.Module):
    r"""
    Abstract base class for coresets.

    TLDR: a coreset is a reduced set of :math:`\hat{n}` (potentially weighted) data
    points that, in some sense, best represent the "important" properties of a larger
    set of :math:`n > \hat{n}` (potentially weighted) data points.

    Given a dataset :math:`X = \{x_i\}_{i=1}^n, x \in \Omega`, where each node is paired
    with a non-negative (probability) weight :math:`w_i \in \mathbb{R} \ge 0`, there
    exists an implied discrete (probability) measure over :math:`\Omega`

    .. math:
        \eta_n = \sum_{i=1}^{n} w_i \delta_{x_i},

    If we then specify a set of test-functions :math:`\Phi = {\phi_1, \dots, \phi_M}`,
    where :math:`\phi_i \colon \Omega \to \mathbb{R}`, which somehow capture the
    "important" properties of the data, then there also exists an implied push-forward
    measure over :math:`\mathbb{R}^M`

    .. math:
        \mu_n = \sum_{i=1}^{n} w_i \delta_{\Phi(x_i)}.

    A coreset is simply a reduced measure containing :math:`\hat{n} < n` updated nodes
    :math:`\hat{x}_i` and weights :math:`\hat{w}_i`, such that the push-forward measure
    of the coreset :math:`\nu_\hat{n}` has (approximately for some algorithms) the same
    "centre-of-mass" as the push-forward measure for the original data :math:`\mu_n`

    .. math:
        \text{CoM}(\mu_n) = \text{CoM}(\nu_\hat{n}),
        \text{CoM}(\nu_\hat{n}) = \sum_{i=1}^\hat{n} \hat{w}_i \delta_{\Phi(\hat{x}_i)}.

    Note: depending on the algorithm, the test-functions may be explicitly specified by
    the user, or implicitly defined by the algorithm's specific objectives.
    """


class AbstractCoreSubset(AbstractCoreSet):
    r"""
    Abstract base class for coresubsets.

    A coresubset is identical to a coreset, except for the crucial additional condition
    that the support of the reduced measure (the coreset), must be a subset of the
    support of the original measure (the original data), such that

    .. math:
        \hat{x}_i = x_i, \forall i \in I,
        I \subset \{1, \dots, n\}, text{card}(I) = \hat{n}.

    Thus, a coresubset, unlike a corset, ensures that feasibility constraints on the
    support of the measure are maintained :cite:`litterer2012recombination`. This is
    vital if, for example, the test-functions are only defined on the support of the
    original measure/nodes, rather than all of :math:`\Omega`.

    In coresubsets, the measure reduction can be implicit (setting weights/nodes to
    zero for all :math:`i \in I \ {1, \dots, n}`) or explicit (removing entries from the
    weight/node arrays). The implicit approach is useful when input/output array shape
    stability is required (E.G. for some JAX transformations); the explicit approach is
    more similar to a standard coreset.
    """


class AbstractRecombination(AbstractCoreSubset):
    r"""
    Abstract base class for recombination based coresubsets.

    A special case in which the coresubset has cardinality of at most :math:`M+1`, given
    the explicitly provided test-functions :math:`\Phi = \{ \phi_1, \dots, \phi_M \}`,
    where :math:`\phi_i \colon \Omega \to \mathbb{R}`.

    Canonically, recombination is used for reducing the support of a quadrature/cubature
    measure, against which integration of any function :math:`f \in \text{span}(\Phi)`
    is identical to integration against the "target" (potentially continuous) measure
    :math:`\mu`,

    .. math:
        \int_\Omega f (\omega) d\eta_n(\omega) = \int_\Omega f(\omega) d\mu(\omega),

    where :math:`\eta_n` is a discrete (quadrature/cubature) measure with cardinality
    :math:`n \ge M+1`. Due to Tchakaloff's theorem, which follows from Caratheodory's
    convex hull theorem, we know there exists a reduced measure :math:`\eta_\hat{n}`
    with :math:`\text{supp}(\eta_\hat{n}) \subset \text{supp}(\eta_n)` and
    :math:`\text{card}(\eta_\hat{n}) \le M+1` (:math:`\hat{n} \le M+1`), that can
    be found via recombination, and that satisfies the above equation (when the original
    quadrature/cubature measure :math:`\eta_n` is replaced with :math:`\eta_\hat{n}`).
    """


class TreeRecombination(AbstractRecombination):
    """
    Tree recombination based coresubsets.

    Based on Algorithm 7 Chapter 3.3 of :cite:`tchernychova2016recombination`, which is
    an order of magnitude more efficient than Algorithm 5 Chapter 3.2, originally
    introduced in :cite:`litterer2012recombination`.
    """


def _compute_null_space(
    nodes: Shaped[Array, "n d"],
    rcond: InexactScalarLike | None = None,
    *,
    mode: Literal["svd", "qr"],
) -> tuple[Shaped[Array, "n mL.s"], IntScalar]:
    if mode == "svd":
        q, s, _ = jsp.linalg.svd(nodes, full_matrices=True)
    elif mode == "qr":
        # Not yet supported in JAX, see https://github.com/google/jax/issues/12897
        # For now we can fall back on scipy, noting that jit compilation will fail.
        q, r, *_ = scipy.linalg.qr(nodes, mode="full", pivoting=True)
        s = jnp.abs(jnp.diag(r))
    else:
        raise ValueError(f"Invalid mode specified; got {mode}, expected 'svd' or 'qr'")

    left_null_space_rank = reveal_left_null_space_rank(nodes.shape, s, rcond)
    largest_left_null_space_basis = q[:, ::-1]
    return largest_left_null_space_basis, left_null_space_rank


def caratheodory_measure_reduction(
    weights: Shaped[Array, " n"],
    nodes: Shaped[Array, "n d"],
    *,
    rcond: InexactScalarLike | None = None,
    mode: Literal["svd", "qr"] = "svd",
    assume_non_degenerate: bool = False,
) -> tuple[Shaped[Array, " hat_n"], Shaped[Array, "hat_n d"]]:
    r"""
    TO COMPLETE.

    :param weights: an array of shape :math:`n`, where each row is a weight :math:`w_i`
        for a discrete measure :math:`\eta_n=\sum_{i=1}^n w_i \delta_{\underline{y}_i}`
    :param nodes: an array of shape :math`n \times d`, where each row is a node/d-vector
        :math:`y_i`, that defines the augmented node :math:`\underline{y}_i = [1 | y_i]`
        for a discrete measure :math:`\eta_n = \sum_{i=1}^n w_i\delta_{\underline{y}_i}`
    :param rcond: a relative condition number. Any singular value :math:`s` below the
        threshold :math:`\text{rcond} * \text{max}(s)` is treated as equal to zero. If
        :code:`rcond is None`, it defaults to `floating point eps * max(n, d)`.
    :param mode: the mode used to compute the left null space basis of the augmented
        node matrix :math:`A = [\hat{y}_1, \cdots, \hat{y}_n]` where each node
        :math:`\hat{y_i} = [1 | y_i]`.
    :param assume_full_rank: ...
    :return: a (potentially reduced) set of (probability) weights :math:`\hat{w}` and
        nodes :math:`y_i`, which implicitly define a discrete probability measure
        :math:`\eta = \sum_{i \in I} \hat{w_i} y_i`, where :math:`I \subset {1,\dots,n}`
        with :math:`\text{card}(I) \le d + 1`.
    """
    abs_weights = jnp.abs(weights)
    probability_weights = abs_weights / jnp.sum(abs_weights)
    augmented_nodes = jnp.c_[jnp.ones_like(weights), nodes]

    output_weights, retained_indices = implicit_caratheodory_measure_reduction(
        probability_weights,
        augmented_nodes,
        rcond=rcond,
        mode=mode,
    )

    if retained_indices is None:
        return output_weights, nodes

    return output_weights[retained_indices], nodes[retained_indices]


EliminationState = namedtuple(
    "EliminationState", ["weights", "nodes", "indices", "iteration"]
)


# Can make use of eqxi.while to reduce memory usage here with checkpointing and buffers?
# Perhaps can replace with a palas kernel?
def implicit_caratheodory_measure_reduction(
    probability_weights: Shaped[Array, " n"],
    augmented_nodes: Shaped[Array, "n M"],
    *,
    rcond: InexactScalarLike | None = None,
    mode: Literal["svd", "qr"] = "svd",
) -> tuple[Shaped[Array, " hat_n"], Shaped[Array, " hat_n"] | None]:
    r"""
    Implicitly reduce the support of the implied discrete measure.

    Performs Gaussian-Elimination on the left null space basis for the augmented node
    matrix :math:`A = \[\underline{y}_1, \dots, \underline{y}_n \]`.
    Partial-Pivoting, etc...


    :param probability_weights: an array of shape :math:`n`, where each row is a weight
        :math:`w_i \ge 0` for the discrete measure
        :math:`\eta_n=\sum_{i=1}^n w_i \delta_{\underline{y}_i}`, where
        :math:`w_i \ge 0` and :math:`\sum_{i=1}^n w_i = 1`
    :param augmented_nodes: an array of shape :math`n \times d`, where each row is an
        augmented node/M-vector :math:`\underline{y}_i = [1 | y_i]` for a discrete
        measure :math:`\eta_n = \sum_{i=1}^n w_i\delta_{\underline{y}_i}`
    :param rcond: a relative condition number. Any singular value :math:`s` below the
        threshold :math:`\text{rcond} * \text{max}(s)` is treated as equal to zero. If
        :code:`rcond is None`, it defaults to `floating point eps * max(n, d)`.
    :param modes: the rank-revealing method for computing the left hand null space basis
        :math:`\Phi = {\phi_1, \dots, \phi_{N}}` of the augmented node matrix
        :math:`A = \[\underline{y}_1, \dots, \underline{y]_n \]`
    :param assume_non_degenerate: JIT compilation of this function is only possible when
        :code:`assume_non_degenerate=True`. See the module docstring of
        :mod:`coreax.recombination` for further information.
    :return:
    """
    largest_left_null_space_basis, left_null_space_rank = _compute_null_space(
        augmented_nodes, rcond, mode=mode
    )
    zero_tol = 1e-12

    def _cond(state: EliminationState):
        # TODO: add skip condition if null space is all zeros?
        *_, basis_index = state
        return basis_index < left_null_space_rank

    def _body(state: EliminationState):
        weights, left_null_space_basis, indices, basis_index = state
        basis_vector = left_null_space_basis[:, basis_index]

        # Do normal pivoting -> then pivot for the argmin?

        # This handles both possible cases, rather than just positive/negative.
        absolute_basis_vector = jnp.abs(basis_vector)
        _rescaled_weights = weights / absolute_basis_vector
        rescaled_weights = jnp.where(
            absolute_basis_vector > zero_tol, _rescaled_weights, jnp.inf
        )
        elimination_index = jnp.argmin(rescaled_weights)
        # pivot = (basis_index, elimination_index)
        # pivoted_indices = indices.at[pivot, ...].set(indices[pivot[::-1], ...])

        weights_update = rescaled_weights[elimination_index] * basis_vector
        update_sign = jnp.sign(basis_vector[elimination_index])
        updated_weights = weights - update_sign * weights_update
        # updated_weights = updated_weights.at[elimination_index].set(0.0)
        # updated_weights = weights.at[pivoted_indices].set(updated_weights)

        basis_update = jnp.tensordot(
            basis_vector / basis_vector[elimination_index],
            left_null_space_basis[elimination_index],
            axes=0,
        )
        updated_left_null_space_basis = left_null_space_basis - basis_update
        # updated_left_null_space_basis = left_null_space_basis.at[pivoted_indices].set(
        #     updated_left_null_space_basis
        # )
        # jax.debug.print("W: {W}", W=updated_weights)
        # jax.debug.breakpoint()
        return EliminationState(
            updated_weights,
            updated_left_null_space_basis,
            indices,
            basis_index + 1,
        )

    n, m = augmented_nodes.shape
    upper_bound_loop_count = n
    in_state = EliminationState(
        probability_weights,
        largest_left_null_space_basis,
        jnp.arange(upper_bound_loop_count),
        0,
    )
    out_state = jax.lax.while_loop(_cond, _body, in_state)
    output_weights, *_ = out_state
    _, retained_indices = jax.lax.top_k(output_weights, m)
    # jax.debug.breakpoint()
    jnp.set_printoptions(precision=2, linewidth=120)
    # q, s, vt = jsp.linalg.svd(augmented_nodes, full_matrices=True)
    # p1, l1, u1 = jsp.linalg.lu(largest_left_null_space_basis)
    # jax.debug.breakpoint()
    return output_weights, retained_indices


def recombination(
    weights: Shaped[Array, " n"],
    nodes: Shaped[Array, "n d"],
    test_functions: Callable[[Shaped[Array, " d"]], Shaped[Array, " M-1"]]
    | None = None,
    *,
    tree_reduction_factor: int = 2,
    mode: Literal["svd", "qr"] = "svd",
    rcond: InexactScalarLike | None = None,
    assume_non_degenerate: bool = False,
):
    # Pre-process the weights
    abs_weights = jnp.abs(weights)
    probability_weights = abs_weights / jnp.sum(abs_weights)

    # Push the nodes forward through the test functions.
    if test_functions is None:
        pushed_forward_nodes = nodes
    else:
        pushed_forward_nodes = jax.vmap(test_functions, in_axes=0)(nodes)
    augmented_pushed_forward_nodes = jnp.c_[
        jnp.ones_like(weights), pushed_forward_nodes
    ]

    # Perform tree recombination
    output_weights, retained_indices = _tree_recombination(
        probability_weights,
        augmented_pushed_forward_nodes,
        tree_reduction_factor=tree_reduction_factor,
        rcond=rcond,
        mode=mode,
    )
    return output_weights, nodes[retained_indices]


def _prepare_tree(weights, nodes, tree_reduction_factor):
    n, d = nodes.shape
    tree_count = tree_reduction_factor * d
    max_tree_depth = math.ceil(math.log(n / d, tree_reduction_factor))
    padding = d * tree_reduction_factor**max_tree_depth - n
    padding_sequence = (0, padding + 1)
    padded_weights = jnp.pad(weights, padding_sequence)
    padded_nodes = jnp.pad(nodes, (padding_sequence, (0, 0)))
    return padded_weights, padded_nodes, tree_count, max_tree_depth


def _tree_recombination(
    weights: Shaped[Array, " n"],
    nodes: Shaped[Array, "n M"],
    *,
    tree_reduction_factor: int = 2,
    rcond: InexactScalarLike | None = None,
    mode: Literal["svd", "qr"] = "svd",
) -> tuple[Shaped[Array, "L M/L"], Shaped[Array, "L M/L M"]]:
    # How to select an optimal value for tree_reduction_factor, L?
    padded_weights, padded_nodes, tree_count, max_tree_depth = _prepare_tree(
        weights, nodes, tree_reduction_factor
    )
    n, d = padded_nodes.shape
    padded_indices = jnp.arange(n - 1)

    caratheodory_measure_reduction = jtu.Partial(
        implicit_caratheodory_measure_reduction,
        rcond=rcond,
        mode=mode,
    )
    target_com = _centroid(weights[None, ...], nodes[None, ...])

    def _tree_reduction_step(_, state):
        weights, indices = state
        reshaped_indices = indices.reshape(tree_count, -1, order="F")
        centroid_weights, centroid_nodes = _centroid(
            weights[reshaped_indices], padded_nodes[reshaped_indices]
        )
        # TODO: have centering that only needs computing once.
        # Centre the centroids to have zero the dataset CoM at zero.
        centred_centroid_nodes = centroid_nodes.at[:, 1:].add(-target_com[-1][..., 1:])
        updated_centroid_weights, _ = caratheodory_measure_reduction(
            centroid_weights, centred_centroid_nodes
        )
        # Updated weights
        weight_update = updated_centroid_weights / centroid_weights
        # If weight update is 1, then nothing has happened.
        updated_weights = jnp.nan_to_num(
            weights.at[reshaped_indices].multiply(weight_update[..., None])
        )

        # Update indices
        # TODO: check degenerate possibilities for this.
        _, eliminated_indices = jax.lax.top_k(-updated_centroid_weights, d)
        _updated_indices = reshaped_indices.at[eliminated_indices].set(-1)
        updated_indices = jnp.partition(
            _updated_indices.reshape(-1), n // tree_reduction_factor
        )

        current_com = _centroid(updated_weights[None, ...], padded_nodes[None, ...])
        indexed_com = _centroid(
            updated_weights[updated_indices].reshape(1, -1),
            padded_nodes[updated_indices].reshape(1, -1, d),
        )
        com_diff = jtu.tree_map(
            lambda x, y: jnp.linalg.norm(x - y),
            (target_com, target_com, (centroid_weights.sum(), centroid_nodes)),
            (
                current_com,
                indexed_com,
                (updated_centroid_weights.sum(), centroid_nodes),
            ),
        )
        # jax.debug.print(
        #     "\nCOM DIFF\n--------\nMASKED: {x};\nINDEXED: {y};\nCENTROID: {z}",
        #     x=com_diff[0],
        #     y=com_diff[1],
        #     z=com_diff[2],
        # )
        # jax.debug.breakpoint()
        return updated_weights, updated_indices

    if n <= d:
        return caratheodory_measure_reduction(weights, nodes)

    in_state = (padded_weights, padded_indices)
    root_weights, _ = jax.lax.fori_loop(
        0, max_tree_depth, _tree_reduction_step, in_state
    )
    output_weights, retained_root_indices = jax.lax.top_k(root_weights, d)
    # jax.debug.breakpoint()
    return output_weights, retained_root_indices
    # leaf_weights, retained_leaf_indices = caratheodory_measure_reduction(
    #     root_weights[retained_root_indices], padded_nodes[retained_root_indices]
    # )
    # retained_indices = retained_root_indices[retained_leaf_indices]
    # return leaf_weights[retained_leaf_indices], retained_indices


@jax.vmap
def _centroid(weights, nodes):
    """Compute the centroid mass and node centre (centre of mass)."""
    centroid_weights = jnp.sum(weights)
    centroid_nodes = jnp.nan_to_num(jnp.average(nodes, 0, weights)).at[..., 0].set(1)
    return centroid_weights, centroid_nodes
