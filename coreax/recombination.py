"""Recombination algorithms."""
from __future__ import annotations

import warnings
from typing import Literal

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
import scipy
from jaxtyping import Array, ArrayLike, DTypeLike, Inexact, Int, Shaped

InexactScalarLike = Inexact[ArrayLike, ""]


# pylint: disable=line-too-long
# Credit: https://github.com/patrick-kidger/lineax/blob/9b923c8df6556551fedc7adeea7979b5c7b3ffb0/lineax/_misc.py#L34  # noqa: E501
# pylint: enable=line-too-long
def _resolve_rcond(
    rcond: InexactScalarLike | None, shape: tuple[int, ...], dtype: DTypeLike
) -> InexactScalarLike:
    if rcond is None:
        return jnp.finfo(dtype).eps * max(shape)
    return jnp.where(rcond < jnp.asarray(0), jnp.finfo(dtype).eps, rcond)


# pylint: disable=line-too-long
# Credit: https://github.com/patrick-kidger/lineax/blob/9b923c8df6556551fedc7adeea7979b5c7b3ffb0/lineax/_solver/svd.py#L67  # noqa: E501
# pylint: enable=line-too-long
def reveal_null_space_rank(
    matrix: Shaped[Array, "n d"],
    singular_values: Shaped[Array, " n"],
    rcond: InexactScalarLike | None = None,
) -> int:
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

        \text{rank}(\text{Null}(A)) = \text{max}(A.shape) - \text{rank}(A).

    :param matrix: an array :math:`A` with shape :math:`n \times d`
    :param singular_values: an array with shape :math:`\text{min}(n, d)`
    :param rcond: a relative condition number. Any singular value :math:`s` below the
        threshold :math:`\text{rcond} * \text{max}(s)` is treated as equal to zero. If
        :code:`rcond is None`, it defaults to `floating point eps * max(n, d)`.
    :return: the rank of the null space of the given matrix :math:`A`.
    """
    rcond = _resolve_rcond(rcond, matrix.shape, singular_values.dtype)
    if singular_values.size > 0:
        rcond = rcond * jnp.max(singular_values[0])
    mask = singular_values > rcond
    rank_vt = sum(mask)
    return max(matrix.shape) - rank_vt


def _explicit_delete(
    x: Shaped[Array, "n ..."], removed_indices: Shaped[Array, " m"]
) -> Shaped[Array, "n-m ..."]:
    return jnp.delete(x, removed_indices, axis=0, assume_unique_indices=True)


# Not currently JIT compatible.
def pre_process_weights_and_nodes(
    weights: Shaped[Array, " n"],
    nodes: Shaped[Array, "n d"],
    *,
    tol: InexactScalarLike = 0.0,
) -> tuple[Shaped[Array, " m"], Shaped[Array, "m d"]]:
    r"""
    Perform trivial measure reduction and ensure the measure is a probability measure.

    Remove weights and/or nodes that are equal to zero, and convert the weights into
    valid probability weights (all positive and sum to one). A weight/node :math:`x` is
    considered equal to zero if :math:`\text{abs}(x) < \text{tol}`.

    :param weights: an array of shape :math:`n`, where each row is a weight :math:`w_i`
        for the atomic measure :math:`\eta_n = \sum_{i=1}^n w_i \delta_{y_i}`
    :param nodes: an array of shape :math`n \times d`, where each row is a node/d-vector
        :math:`y_i` for the atomic measure :math:`\eta_n = \sum_{i=1}^n w_i\delta_{y_i}`
    :param tol: a tolerance below which a value is consider as equal to zero
    :return: a (potentially reduced) set of (probability) weights :math:`\hat{w}` and
        nodes :math:`y_i`, which implicitly define an atomic probability measure
        :math:`\eta = \sum_{i \in I} \hat{w_i} y_i`, where :math:`I \subset {1,\dots,n}`
        with :math:`\text{card}(I) = \hat{n} \le n` (with equality if no reduction is
        performed).
    """
    non_negative_weights = jnp.abs(weights)
    weighted_nodes = non_negative_weights[..., None] * nodes
    weighted_nodes_abs_coordinate_sum = jnp.abs(jnp.sum(weighted_nodes, axis=-1))

    # If the weighted node absolute co-ordinate sum, at a given row index :math:`i`, is
    # zero, the weight :math:`w_i = 0` and/or the corresponding node :math:`y_i = 0`. In
    # either case, both the weight and the node are redundant and can be removed from
    # the measure.
    redundant_indices = jnp.nonzero(weighted_nodes_abs_coordinate_sum <= tol)

    _remove_redundant = jtu.Partial(_explicit_delete, removed_indices=redundant_indices)
    positive_weights = _remove_redundant(non_negative_weights)
    probability_weights = positive_weights / jnp.sum(positive_weights)
    non_zero_nodes = _remove_redundant(nodes)
    return probability_weights, non_zero_nodes


def _svd_reduction(
    nodes: Shaped[Array, "n d"], rcond: InexactScalarLike | None = None
) -> tuple[Shaped[Array, "n-d d"], int]:
    u, s, _ = jsp.linalg.svd(nodes, full_matrices=True)
    null_space_rank = reveal_null_space_rank(nodes, s, rcond)
    left_null_space_basis = u[:, -null_space_rank:]
    return left_null_space_basis, null_space_rank


# Not yet supported in JAX, see https://github.com/google/jax/issues/12897
# For now we can fall back on scipy, noting that jit compilation will fail.
def _qr_reduction(
    nodes: Shaped[Array, "n d"], rcond: InexactScalarLike | None = None
) -> tuple[Shaped[Array, "n-d d"], int]:
    q, r, *_ = scipy.linalg.qr(nodes, mode="full", pivoting=True)
    s = jnp.abs(jnp.diag(r))
    null_space_rank = reveal_null_space_rank(nodes, s, rcond)
    left_null_space_basis = q[:, -null_space_rank:]
    return left_null_space_basis, null_space_rank


SOLVERS = {"svd": _svd_reduction, "qr": _qr_reduction}


# Not currently JIT compatible. Requires careful consideration of the handling of the
# left null space basis, which is currently sliced by a dynamic value (preventing JIT)
def caratheodory_measure_reduction(
    weights: Shaped[Array, " n"],
    nodes: Shaped[Array, "n d"],
    *,
    rcond: InexactScalarLike | None = None,
    mode: Literal["svd", "qr"] = "svd",
) -> tuple[Shaped[Array, " rows=d+1"], Shaped[Array, "rows=d+1 d"]]:
    r"""
    Reduce the support of the implied atomic measure via Caratheodory measure reduction.

    Based on the algorithm detailed in :cite:`litterer2012recombination`, without the
    requirement of a centred measure (a measure with zero centre of mass). I.E Algorithm
    4 Chapter 3.2 of :cite:`tchernychova2016recombination`.

    Where the weights :math:`w \in \mathbb{R}` and nodes :math:`y in \mathbb{R^d}`
    define an :math:`n` point atomic probability measure

    .. math:

        \eta_n = \sum_{i=1}^{n} w_i \delta_{y_i},

    Caratheodory measure reduction allows one to determine a reduced measure, with at
    least :math:`d+1` points (unique weights and nodes), that preserves the centre of
    mass of the original measure, where :math:`I \subset {1, \dots, n}` and
    `\text{card}(I) = \hat{n} = d + 1`,

    .. math:

        \eta_{\hat{n}} = \sum{i \in I} \hat{w_i} y_i = \eta_n.

    Note that the weights :math:`\hat{w}` must be recomputed, while the remaining nodes
    are left unchanged. This ensures that feasibility constraints on the support of the
    measure are maintained. For example, given an atomic probability measure where each
    node represents a feasible category, transforming the weights simply changes the
    probability of the given categories, while transforming the nodes implicitly defines
    new (infeasible) "latent"-categories.

    :param weights: an array of shape :math:`n`, where each row is a weight :math:`w_i`
        for the atomic measure :math:`\eta_n = \sum_{i=1}^n w_i \delta_{y_i}`.
    :param nodes: an array of shape :math`n \times d`, where each row is a node/d-vector
        :math:`y_i` for the atomic measure :math:`\eta_n = \sum_{i=1}^n w_i\delta_{y_i}`
    :param rcond: a relative condition number. Any singular value :math:`s` below the
        threshold :math:`\text{rcond} * \text{max}(s)` is treated as equal to zero. If
        :code:`rcond is None`, it defaults to `floating point eps * max(n, d)`.
    :param mode: the mode used to compute the left null space basis of the augmented
        node matrix :math:`A = [\hat{y}_1, \cdots, \hat{y}_n]` where each node
        :math:`\hat{y_i} = [1 | y_i]`.
    :return: a (potentially reduced) set of (probability) weights :math:`\hat{w}` and
        nodes :math:`y_i`, which implicitly define an atomic probability measure
        :math:`\eta = \sum_{i \in I} \hat{w_i} y_i`, where :math:`I \subset {1,\dots,n}`
        with :math:`\text{card}(I) = d + 1`.
    """
    augmented_nodes = jnp.c_[jnp.ones_like(weights), nodes]
    left_null_space_basis, null_space_rank = SOLVERS[mode](augmented_nodes, rcond)

    if nodes.shape[0] <= null_space_rank:
        warnings.warn(
            "Nothing to do; node count already below reduction threshold",
            stacklevel=2,
        )
        return weights, nodes

    def _reduction_step(
        state: tuple[Shaped[Array, " n"], Shaped[Array, "n-m d"]],
        _,
    ) -> tuple[tuple[Shaped[Array, " n"], Shaped[Array, "n-m d"]], Int[Array, " n-m"]]:
        # Preserve the centre of mass of the atomic measure, subject to reducing the
        # number of nodes :math:`y` from :math:`n` to :math:`n-1`, by setting the k-th
        # weight :math:`w_k` to zero (implicitly removing the corresponding node
        # :math:`y_k`) and recomputing all other non-zero weights :math:`\hat{w}`.
        # I.E. :math:`\sum_{i=1}^{n} w_i y_i = \sum_{i=1}^{n} \hat{w}_i y_i` for all
        # :math:`i, w_i \ne 0.0`.
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

    in_state = (weights, left_null_space_basis)
    scan_vector = jnp.arange(0, null_space_rank)
    out_state, removed_indices = jax.lax.scan(_reduction_step, in_state, scan_vector)
    output_weights, _ = out_state

    # Explicitly remove the redundant (zero) weights and their corresponding nodes.
    _remove_redundant = jtu.Partial(_explicit_delete, removed_indices=removed_indices)
    return _remove_redundant(output_weights), _remove_redundant(nodes)
