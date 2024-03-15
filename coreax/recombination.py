"""
Recombination algorithms.

Note: these functions can not be JIT compiled if :code:`assume_non_degenerate=False`.
This results from the impact of two degeneracies:
    - Rank-defficiency in the node matrix :math:`A` which requires the null space
    rank to be dynamic (leading to a dynamicly sized slice and arange stop);
    - The centre of mass of the reduced measure lies on a face of the convex hull
    of its support, which requires a dynamic number of weights to be removed on
    each iteration/step (leading to shape instability in the scan step return).

We set the following challenge for interested contributors: without the use of
custom JAX primitives (or host_callback/pure_callback), can you handle the above
degenerates cases while still providing JIT compatiblity. I.E. can you make this
function JIT compatible when :code:`assume_non_degenerate=False`.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Literal

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
    matrix: Shaped[Array, "n d"],
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

        \text{rank}(\text{Null}(A)) = \text{max}(A.shape) - \text{rank}(A).

    :param matrix: an array :math:`A` with shape :math:`n \times d`
    :param singular_values: an array with shape :math:`\text{min}(n, d)`
    :param rcond: a relative condition number. Any singular value :math:`s` below the
        threshold :math:`\text{rcond} * \text{max}(s)` is treated as equal to zero. If
        :code:`rcond is None`, it defaults to `floating point eps * max(n, d)`
    :return: the rank of the null space of the given matrix :math:`A`.
    """
    rcond = _resolve_rcond(rcond, matrix.shape, singular_values.dtype)
    if singular_values.size > 0:
        rcond = rcond * jnp.max(singular_values[0])
    mask = singular_values > rcond
    matrix_rank = sum(mask)
    maximum_feasible_matrix_rank = max(matrix.shape)
    return maximum_feasible_matrix_rank - matrix_rank


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


# TODO:
# Not currently JIT compatible.
# Do we want to preserve the normalization factor if we have an un-normalised measure?
def trivial_measure_reduction(
    weights: Shaped[Array, " n"],
    nodes: Shaped[Array, "n d"],
    *,
    zero_tol: InexactScalarLike = 0.0,
    colinear_tol: InexactScalarLike = 0.0,
) -> tuple[Shaped[Array, " m"], Shaped[Array, "m d"]]:
    r"""
    Perform trivial measure reduction and ensure the measure is a probability measure.

    Remove weights and/or nodes that are equal to zero, nodes that are co-linear, and
    convert the weights into valid probability weights (all positive and sum to one).
    A weight/node :math:`x` is equal to zero if :math:`\text{abs}(x) < \text{tol}`.

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

    # Remove all colinear points <- evaluate pairwise and use cross product.
    # Add the evaluate pairwise function to utils.

    # If the weighted node absolute co-ordinate sum, at a given row index :math:`i`, is
    # zero, the weight :math:`w_i = 0` and/or the corresponding node :math:`y_i = 0`. In
    # either case, both the weight and the node are redundant and can be removed from
    # the measure.
    redundant_indices = jnp.nonzero(weighted_nodes_abs_coordinate_sum <= zero_tol)

    _remove_redundant = jtu.Partial(_explicit_delete, delete_indices=redundant_indices)
    positive_weights = _remove_redundant(non_negative_weights)
    probability_weights = positive_weights / jnp.sum(positive_weights)
    non_zero_nodes = _remove_redundant(nodes)
    return probability_weights, non_zero_nodes


def _compute_null_space(
    nodes: Shaped[Array, "n d"],
    rcond: InexactScalarLike | None = None,
    *,
    mode: Literal["svd", "qr"],
    assume_full_rank: bool = False,
) -> Shaped[Array, "n m"]:
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
        null_space_rank = reveal_null_space_rank(nodes, s, rcond)
    left_null_space_basis = q[:, -null_space_rank:]
    return left_null_space_basis


def _detect_degneracies(weights, nodes):
    msg = "Degeneracy detected in measure reduction step:"
    convex_remediation_step = "Perturb the nodes via a regularization constant"

    # Compute the value used in the trivial measure reduction.

    # Remedial steps; Trivial measure reduction a prior, regularization, etc...


def caratheodory_measure_reduction(
    weights: Shaped[Array, " n"],
    nodes: Shaped[Array, "n d"],
    *,
    rcond: InexactScalarLike | None = None,
    mode: Literal["svd", "qr"] = "svd",
    assume_non_degenerate: bool = False,
) -> tuple[Shaped[Array, " hat_n"], Shaped[Array, "hat_n d"]]:
    r"""
    Reduce the support of the implied atomic measure via Caratheodory measure reduction.

    Where the weights :math:`w \in \mathbb{R}`, nodes :math:`y in \mathbb{R}^d` and
    augmented nodes :math:`\underline{y}_i = [1 | y_i] \in \mathbb{R}^{d+1}` define an
    :math:`n` point atomic measure :math:`\eta_n` on :math:`\mathbb{R}^{d+1}`

    .. math:

        \eta_n = \sum_{i=1}^{n} w_i \delta_{\underline{y}_i},

    Caratheodory measure reduction allows one to determine a reduced measure, with at
    most :math:`d+1` points (unique weights and nodes), that preserves the centre of
    mass of the original measure,

    .. math:

        \eta_{\hat{n}} = \sum{i \in I} \hat{w_i} y_i = \eta_n,

    where :math:`I \subset {1, \dots, n}` and `\text{card}(I) = \hat{n} \le d + 1`.

    Note that the weights :math:`\hat{w}` must be recomputed, while the remaining nodes
    are left unchanged. This ensures that feasibility constraints on the support of the
    measure are maintained. For example, given an atomic probability measure where each
    node represents a feasible category, transforming the weights simply changes the
    probability of the given categories, while transforming the nodes implicitly defines
    new (infeasible) "latent"-categories.

    Further information on the underlying algorithm see :func:`implicit_caratheodory_measure_reduction`...

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
    # Augmenting the nodes elides the need to centre them (have a centred measure).
    augmented_nodes = jnp.c_[jnp.ones_like(weights), nodes]

    if not assume_non_degenerate:
        weights, augmented_nodes = trivial_measure_reduction(weights, augmented_nodes)

    output_weights, removed_indices = implicit_caratheodory_measure_reduction(
        weights,
        augmented_nodes,
        rcond=rcond,
        mode=mode,
        assume_non_degenerate=assume_non_degenerate,
    )

    if not assume_non_degenerate:
        ...

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

    Unlike the perfered interface, :func:`caratheodory_measure_reduction`, this function
    does not explictly remove any weights or nodes (providing input-output weight shape
    stability). Instead, redundant weights are zeroed and their indices returned.

    This implementation is based on Algorithm 6 Chapter 3.3 of
    :cite:`tchernychova2016recombination`, which is an order of magnitude more efficient
    (in the context of recombination) than Algorithm 4 Chapter 3.2 of the same thesis,
    originally introduced in :cite:`litterer2012recombination`.

    Unlike the prior art, here, the node count :math:`n` need not be a fixed multiple
    of the dimension of the augmented nodes :math:`d+1`. Hence, if math:`n = (d+1) + 1`,
    we recover a variant of Litterer's 1-Tree approach; if :math:`n = 2(d + 1)`, we
    recover a variant of Tchernychova's M-Tree approach, where :math:`M = d + 1`.

    Given an atomic probability measure on :math:`\mathbb{R}^{d+1}`

    .. math:
        \eta_n = \sum_{i=1}^{n} w_i \delta_{\underline{y}_i},

    implied by the given :math:`n` probability weights :math:`w_i` (non-negative weights
    that sum to one) and :math:`n` augmented nodes :math:`\hat{y_i} = [1 | y_i]`, the
    objective of this algorithm is to find a collection of :math:`\hat{n} \le d+1`
    weights :math:`w > 0` and :math:`n - \hat{n}` weights :math:`w = 0`) that are a
    basic feasible solution to the linear programming problem:

    .. math:
        \text{max}(c^T w) = 0,\ \forall w
        Aw = \text{CoM}(\eta_n) = \sum_{i=1}^n w_i \underline{y}_i
        w \ge 0

    where :math:`\text{CoM}(\eta_n)` is the centre of mass (weighted mean) of the atomic
    measure. The trivial objective function permits non-unique solutions which for the
    purposes considered here are equivalent.

    Such a problem is commonly solved via the Simplex method (or interior point methods
    that have been adapted to provide basic feasible solutions). However, in this
    algorithm, a slightly different but related approach is taken.

    Where :math:`\Phi = \{\phi_1, \dots, \phi_{N}}` is the left null space basis for the
    node matrix :math:`A`, and :math:`N = n - \hat{n}` is the rank of the null space,
    one can represent the centre of mass of the atomic measure as follows:

    .. math:
        \begin{aligned}
        \text{CoM}(\eta_n) &= \sum_{i=1}^n w_i \underline{y}_i,
        &\sum_{i=1}^{n} w_i\underline{y}_i - \alpha\sum_{i=1}^{n} \phi_i\underline{y}_i,
        &\sum_{i=1}^{n} (w_i - \alpha \phi_i) \underline{y}_i\),
        &\sum_{i=1}^{n} \hat{w}_i \underline{y}_i\),
        \end{aligned}

    where, by design,

    .. math:
        \alpha = \text{min}\{\frac{w_i}{\phi_i} | \phi_i > 0\} = \frac{w_k}{\phi_k}`,

    such that :math:`\hat{w}_k = (w_k - \alpha \phi_k) = 0` at the index :math:`i = k`,
    and :math:`\sum_{i=1}^n w_i = 1`.

    The vector :math:`\phi_1` can be (implicitly) removed from the null space basis, and
    the remaining vectors updated by taking the linear combination with :math:`\phi_1`

    .. math:
        \hat{\phi}_i = \phi_i - \frac{(\phi_i)_k}{(\phi_1)_k} \phi_k.

    The above process is repeated :math:`N` times, until only :math:`\hat{n}` non-zero
    weights remain, (equivilantly the :math:`N` indices of the implicitly removed
    weights are been determined).

    Important: the above presentation assumes no degerate behaviour, specifically:
        1. The node matrix is not rank-defficient (although this is implictly handled by
        computing, rather than assuming, the null space rank).
        2. The centre of mass of the reduced measure :math:`\eta_{\hat{n}}` lies on a
        face of the convex hull of its support. In this case, in a single step, there
        may exist multiple indices :math:`k` which are implictly set to zero, but only
        a single index will be recorded as an impliclty removed weight. Thankfully, this
        case can be easily handled (albeit in a JIT incompatible manner), by identifying
        the indices of these extra zeroed weights post reduction.

    :param probability_weights:
    :param augmented_nodes:
    :param rcond:
    :param modes: the rank-revealing method for computing the left hand null space basis
        :math:`\Phi = {\phi_1, \dots, \phi_{N}}` of the augmented node matrix :math:`A`
    :param assume_non_degenerate:
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
        # basis_vector is implicitly set to zero by the subtraction.
        # we then roll this zeroed vector to the back so that we select a new vector on
        # the next iteration.
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
    """ """
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


def recombination(
    weights: Shaped[Array, " n"],
    nodes: Shaped[Array, "n d"],
    push_forward_map: Callable[[Shaped[Array, " d"]], Shaped[Array, " M"]],
):
    r"""
    TO BE COMPLETED

    :param weights: an array of shape :math:`n`, where each row is a weight :math:`w_i`
        for the atomic measure :math:`\eta_n = \sum_{i=1}^n w_i \delta_{y_i}`.
    :param nodes: an array of shape :math`n \times d`, where each row is a node/d-vector
        :math:`y_i` for the atomic measure :math:`\eta_n = \sum_{i=1}^n w_i\delta_{y_i}`
    :param push_forward_map: an :math:`\eta_n\text{-integrable}` map,
        :math:`\Psi = {\psi_1, \dots, \psi_M} : K \to \mathbb{R}^M`, where
        :math:`\text{supp}(\eta_n) \subseteq K`, which defines the push forward measure
        :math:`(\Psi_{*}\eta_n) = \sum_{i=1}^n w_i \delta_{\Psi(y_i)} against which we
        will generate our :math:`\Psi\text{-generalised}` cubature measure.
    """
    pushed_forward_nodes = jax.vmap(push_forward_map, in_axes=0)(nodes)
    n_weights = weights.shape[-1]
    n_pushed_forward_nodes = pushed_forward_nodes.shape[-2]
    if n_weights != n_pushed_forward_nodes:
        msg = (
            f"Number of weights and nodes must be identical; got weights: {n_weights},"
            + f"pushed_forward_nodes: {n_pushed_forward_nodes}"
        )
        raise ValueError(msg)
