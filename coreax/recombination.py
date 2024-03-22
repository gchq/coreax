r"""
Recombination algorithms.

Algorithms to reduce discrete measures.
"""
from __future__ import annotations

import math
import warnings
from typing import Literal

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
def reveal_null_space_rank(
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

    The null space rank is the difference between either the number of columns, or the
    number of rows in the matrix (whichever is larger), minus the matrix rank. I.E.

    .. math::

        \text{rank}(\text{null}(A)) = \text{max}(\text{shape}(A)) - \text{rank}(A).

    :param matrix_shape: the shape of an :math:`n \times d` matrix :math:`A`
    :param singular_values: an array with shape :math:`\text{min}(n, d)`
    :param rcond: a relative condition number. Any singular value :math:`s` below the
        threshold :math:`\text{rcond} * \text{max}(s)` is treated as equal to zero. If
        :code:`rcond is None`, it defaults to `floating point eps * max(n, d)`
    :return: the rank of the null space of a matrix :math:`A` with given shape and given
        singular values.
    """
    rcond = _resolve_rcond(rcond, matrix_shape, singular_values.dtype)
    if singular_values.size > 0:
        rcond = rcond * jnp.max(singular_values[0])
    mask = singular_values > rcond
    matrix_rank = sum(mask)
    maximum_feasible_matrix_rank = max(matrix_shape)
    return maximum_feasible_matrix_rank - matrix_rank


class AbstractCoreSet(eqx.Module):
    r"""
    Abstract base class for coresets.

    TLDR: a coreset is a reduced set of :math:`\hat{n}` (potentially weighted) data
    points that, in some sense, best represent the "important" properties of a larger
    set of :math:`n` (potentially weighted) data points.

    Given a dataset :math:`X = \{x_i\}_{i=1}^n, x \in \Omega`, where each node is paired
    with a non-negative (probability) weight :math:`w_i \in \mathbb{R} \ge 0`, there
    exists an implied atomic (probability) measure over :math:`\Omega`

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
    is identical to integration against the "target" (potentially non-atomic) measure
    :math:`\mu`,

    .. math:
        \int_\Omega f (\omega) d\eta_n(\omega) = \int_\Omega f(\omega) d\mu(\omega),

    where :math:`\eta_n` is an atomic (quadrature/cubature) measure with cardinality
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
    assume_full_rank: bool = False,
) -> tuple[Shaped[Array, "n m"], IntScalar]:
    """
    Compute a left null space basis for the given nodes.

    JIT compilation is only possible when :code:`assume_full_rank=True`. See the module
    docstring of :mod:`coreax.recombination` for further information.
    """
    if mode == "svd":
        q, s, _ = jsp.linalg.svd(nodes, full_matrices=True)
    elif mode == "qr":
        # Not yet supported in JAX, see https://github.com/google/jax/issues/12897
        # For now we can fall back on scipy, noting that jit compilation will fail.
        q, r, *_ = scipy.linalg.qr(nodes, mode="full", pivoting=True)
        s = jnp.abs(jnp.diag(r))
    else:
        raise ValueError(f"Invalid mode specified; got {mode}, expected 'svd' or 'qr'")

    if assume_full_rank:
        null_space_rank = max(nodes.shape) - min(nodes.shape)
    else:
        null_space_rank = reveal_null_space_rank(nodes.shape, s, rcond)
    left_null_space_basis = q[:, -null_space_rank:]
    return left_null_space_basis, null_space_rank


def _explicit_delete(
    a: Shaped[Array, "n ..."], delete_indices: Shaped[Array, " m"], axis: int = 0
) -> Shaped[Array, "n-m ..."]:
    """
    Delete (assumed unique) leading entries from an array.

    :param a: an array :math:`A` whose leading entries are to be deleted
    :param delete_indices: an array of unique indices to delete from the leading axis of
        the array :math:`A`
    :return: the array :math:`A` with the specified leading entries deleted.
    """
    return jnp.delete(a, delete_indices, axis=axis, assume_unique_indices=True)


def trivial_measure_reduction(
    weights: Shaped[Array, " n"],
    nodes: Shaped[Array, "n d"],
    *,
    zero_tol: InexactScalarLike = 0.0,
) -> tuple[Shaped[Array, " m"], Shaped[Array, "m d"]]:
    r"""
    Perform trivial measure reduction and ensure the measure is a probability measure.

    Remove weights and/or nodes that are equal to zero, nodes that are co-linear, and
    convert the weights into valid probability weights (all positive and sum to one).
    A weight/node :math:`x` is equal to zero if :math:`\text{abs}(x) < \text{zero_tol}`,

    :param weights: an array of shape :math:`n`, where each row is a weight :math:`w_i`
        for the atomic measure :math:`\eta_n = \sum_{i=1}^n w_i \delta_{y_i}`
    :param nodes: an array of shape :math`n \times d`, where each row is a node/d-vector
        :math:`y_i` for the atomic measure :math:`\eta_n = \sum_{i=1}^n w_i\delta_{y_i}`
    :param zero_tol: a tolerance below which a value is considered equal to zero
    :return: a (potentially reduced) set of (probability) weights :math:`\hat{w}` and
        nodes :math:`y_i`, which implicitly define an atomic probability measure
        :math:`\eta = \sum_{i \in I} \hat{w_i} y_i`, where :math:`I \subset {1,\dots,n}`
        with :math:`\text{card}(I) = \hat{n} \le n` (with equality if no reduction is
        performed).
    """
    non_negative_weights = jnp.abs(weights)
    weighted_nodes = non_negative_weights[..., None] * nodes
    weighted_nodes_abs_coordinate_sum = jnp.abs(jnp.sum(weighted_nodes, axis=-1))

    # If the absolute weighted node co-ordinate sum, at a given row index :math:`i`, is
    # zero, the weight :math:`w_i = 0` and/or the corresponding node :math:`x_i = 0`. In
    # either case, both the weight and the node are redundant and can be removed from
    # the measure.
    redundant_indices = weighted_nodes_abs_coordinate_sum <= zero_tol

    _remove_redundant = jtu.Partial(_explicit_delete, delete_indices=redundant_indices)
    positive_weights = _remove_redundant(non_negative_weights)
    probability_weights = positive_weights / jnp.sum(positive_weights)
    non_zero_nodes = _remove_redundant(nodes)
    return probability_weights, non_zero_nodes


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
        for the atomic measure :math:`\eta_n=\sum_{i=1}^n w_i \delta_{\underline{y}_i}`
    :param nodes: an array of shape :math`n \times d`, where each row is a node/d-vector
        :math:`y_i`, that defines the augmented node :math:`\underline{y}_i = [1 | y_i]`
        for the atomic measure :math:`\eta_n = \sum_{i=1}^n w_i\delta_{\underline{y}_i}`
    :param rcond: a relative condition number. Any singular value :math:`s` below the
        threshold :math:`\text{rcond} * \text{max}(s)` is treated as equal to zero. If
        :code:`rcond is None`, it defaults to `floating point eps * max(n, d)`.
    :param mode: the mode used to compute the left null space basis of the augmented
        node matrix :math:`A = [\hat{y}_1, \cdots, \hat{y}_n]` where each node
        :math:`\hat{y_i} = [1 | y_i]`.
    :param assume_full_rank: ...
    :return: a (potentially reduced) set of (probability) weights :math:`\hat{w}` and
        nodes :math:`y_i`, which implicitly define an atomic probability measure
        :math:`\eta = \sum_{i \in I} \hat{w_i} y_i`, where :math:`I \subset {1,\dots,n}`
        with :math:`\text{card}(I) \le d + 1`.
    """
    abs_weights = jnp.abs(weights)
    probability_weights = abs_weights / jnp.sum(abs_weights)
    augmented_nodes = jnp.c_[jnp.ones_like(weights), nodes]

    zero_tol = None
    if not assume_non_degenerate:
        zero_tol = _resolve_rcond(rcond, augmented_nodes.shape, augmented_nodes.dtype)
        weights, augmented_nodes = trivial_measure_reduction(
            weights, augmented_nodes, zero_tol=zero_tol
        )

    output_weights, removed_indices = implicit_caratheodory_measure_reduction(
        probability_weights,
        augmented_nodes,
        rcond=rcond,
        mode=mode,
        assume_non_degenerate=assume_non_degenerate,
    )

    if not assume_non_degenerate and zero_tol is not None:
        removed_indices = output_weights <= zero_tol

    # Explicitly remove the redundant (zeroed) weights and their corresponding nodes.
    _remove_redundant = jtu.Partial(_explicit_delete, delete_indices=removed_indices)
    return _remove_redundant(output_weights), _remove_redundant(nodes)


def implicit_caratheodory_measure_reduction(
    probability_weights: Shaped[Array, " n"],
    augmented_nodes: Shaped[Array, "n-hat_n d+1"],
    *,
    rcond: InexactScalarLike | None = None,
    mode: Literal["svd", "qr"] = "svd",
    assume_non_degenerate: bool = False,
) -> tuple[Shaped[Array, " hat_n"], Shaped[Array, " hat_n"]]:
    r"""
    Implicitly reduce the support of the implied atomic measure.

    :param probability_weights: an array of shape :math:`n`, where each row is a weight
        :math:`w_i \ge 0` for the atomic measure
        :math:`\eta_n=\sum_{i=1}^n w_i \delta_{\underline{y}_i}`, where
        :math:`w_i \ge 0` and :math:`\sum_{i=1}^n w_i = 1`
    :param augmented_nodes: an array of shape :math`n \times d`, where each row is an
        augmented node/(d+1)-vector :math:`\underline{y}_i = [1 | y_i]` for the atomic
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
    left_null_space_basis, null_space_rank = _compute_null_space(
        augmented_nodes, rcond, mode=mode, assume_full_rank=assume_non_degenerate
    )
    if not assume_non_degenerate and augmented_nodes.shape[0] <= null_space_rank:
        warnings.warn(
            "Nothing to do; "
            "node count is already less than or equal to the null space rank",
            stacklevel=2,
        )
        return probability_weights, augmented_nodes

    def _reduction_step(
        state: tuple[Shaped[Array, " n"], Shaped[Array, "n-1 d+1"]],
        _,
    ) -> tuple[tuple[Shaped[Array, " n"], Shaped[Array, "n-1 d+1"]], Int[Array, ""]]:
        weights, left_null_space_basis = state
        basis_vector = left_null_space_basis[:, 0]

        rescaling_factor = weights / basis_vector
        rescaling_factor = jnp.where(rescaling_factor > 0.0, rescaling_factor, jnp.inf)
        argmin_index = jnp.argmin(rescaling_factor)
        updated_weights = weights - rescaling_factor[argmin_index] * basis_vector

        rescaled_basis_vector = basis_vector / basis_vector[argmin_index]
        left_null_space_basis -= jnp.tensordot(
            rescaled_basis_vector, left_null_space_basis[argmin_index], axes=0
        )
        updated_left_null_space_basis = jnp.roll(left_null_space_basis, -1, axis=1)
        return (updated_weights, updated_left_null_space_basis), argmin_index

    in_state = (probability_weights, left_null_space_basis)
    null_space_rank = left_null_space_basis.shape[-1]
    scan_vector = jnp.arange(null_space_rank)
    out_state, removed_indices = jax.lax.scan(_reduction_step, in_state, scan_vector)
    output_weights, _ = out_state

    return output_weights, removed_indices


def _tree_recombination(
    weights: Shaped[Array, " n"],
    nodes: Shaped[Array, "n d"],
    tree_count: int | None = None,
) -> tuple[Shaped[Array, "L n/L"], Shaped[Array, "L n/L d"]]:
    n, d = nodes.shape
    if tree_count is None:
        tree_count = 2 * d
    padding = tree_count ** math.ceil(math.log(n, tree_count)) - n

    padding_sequence = [(0, padding, 0)]
    weights = jax.lax.pad(weights, 0.0, padding_sequence)
    nodes = jax.lax.pad(nodes, 0.0, padding_sequence + [(0, 0, 0)])

    def _tree_reduction_step(state, _):
        weights, nodes = state
        reshaped_weights = weights.reshape(tree_count, -1, order="F")
        reshaped_nodes = nodes.reshape(tree_count, -1, d, order="F")
        centroid_masses, centroid_nodes = _centroid(reshaped_weights, reshaped_nodes)
        (
            updated_centroid_masses,
            redundant_centroids,
        ) = implicit_caratheodory_measure_reduction(centroid_masses, centroid_nodes)
        updated_weights = weights * (updated_centroid_masses / centroid_masses)
        out_state = updated_weights.reshape(-1), nodes.reshape(-1, d)
        return out_state, redundant_centroids

    assert jnp.count_nonzero(weights) == tree_count**tree_count

    in_state = (weights, nodes)
    scan_vector = jnp.arange(tree_count)
    out_state, redundant_centroids = jax.lax.scan(
        _tree_reduction_step, in_state, scan_vector
    )

    # Quite a lot of reshape acrobatics here; requires explanation.
    root_weights, root_nodes = out_state
    root_weights = root_weights.reshape(tree_count, -1)
    root_nodes = root_nodes.reshape(tree_count, -1, d)

    _prune_redundant = jtu.Partial(_explicit_delete, delete_indices=redundant_centroids)
    pruned_root_weights = _prune_redundant(root_weights).reshape(-1)
    pruned_root_nodes = _prune_redundant(root_nodes).reshape(-1, d)

    leaf_weights, redundant_leaves = implicit_caratheodory_measure_reduction(
        pruned_root_weights, pruned_root_nodes
    )
    _remove_redundant = jtu.Partial(_explicit_delete, delete_indices=redundant_leaves)
    return _remove_redundant(leaf_weights), _remove_redundant(pruned_root_nodes)


@jax.vmap
def _centroid(weights, nodes):
    """Compute the centroid mass and node centre (centre of mass)."""
    return jnp.sum(weights), jnp.average(nodes, -1, weights)
