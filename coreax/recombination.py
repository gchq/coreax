"""Recombination algorithms."""
from __future__ import annotations

import warnings
from typing import Literal

import jax
import jax.numpy as jnp
import jax.scipy as jsp
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
    return jnp.where(rcond < 0, jnp.finfo(dtype).eps, rcond)


# pylint: disable=line-too-long
# Credit: https://github.com/patrick-kidger/lineax/blob/9b923c8df6556551fedc7adeea7979b5c7b3ffb0/lineax/_solver/svd.py#L67  # noqa: E501
# pylint: enable=line-too-long
def reveal_kernel_rank(array, singular_values, rcond):
    """Reveal array rank given singular values and a relative condition number."""
    rcond = _resolve_rcond(rcond, array.shape, singular_values.dtype)
    if singular_values.size > 0:
        rcond = rcond * singular_values[0]
    mask = singular_values > rcond
    rank_vt = jnp.sum(mask)
    return max(array.shape) - rank_vt, rcond


def caratheodory_measure_reduction(
    weights: Shaped[Array, " n"],
    nodes: Shaped[Array, "n d"],
    rcond: InexactScalarLike | None = None,
    mode: Literal["svd"] = "svd",
) -> tuple[Shaped[Array, " rows=d+1"], Shaped[Array, "rows=d+1 d"]]:
    """Reduce measure using Caratheodory measure reduction."""
    # These typehints won't strictly be true if the nodes are rank deficient/degenerate,
    # or if someone wanted a `coreset_size > d+1`, hence why they are documentation only
    # for the rows.
    augmented_nodes = jnp.c_[jnp.ones_like(weights), nodes]
    transposed_augmented_nodes = jnp.transpose(augmented_nodes)
    if mode == "svd":
        _, s, vt = jsp.linalg.svd(transposed_augmented_nodes, full_matrices=True)
        kernel_rank, rcond = reveal_kernel_rank(transposed_augmented_nodes, s, rcond)
        null_basis = vt[-kernel_rank:].T
    elif mode == "qr":
        # Not yet support in JAX, see https://github.com/google/jax/issues/12897
        q, r, _ = jsp.linalg.qr(augmented_nodes, mode="full", pivoting=True)
        s = jnp.abs(jnp.diag(r))
        kernel_rank, rcond = reveal_kernel_rank(
            augmented_nodes, jnp.abs(jnp.diag(r)), rcond
        )
        null_basis = q[:, -kernel_rank:]
    else:
        msg = f"`mode` must be one of 'svd' or 'qr'; got {mode}"
        raise ValueError(msg)

    if nodes.shape[0] <= kernel_rank:
        warnings.warn(
            "Nothing to do; node count already below reduction threshold",
            stacklevel=2,
        )
        return weights, nodes

    def _body_fn(
        state: tuple[Shaped[Array, " n"], Shaped[Array, "n-m d"]],
        _,
    ) -> tuple[tuple[Shaped[Array, " n"], Shaped[Array, "n-m d"]], Int[Array, " n-m"]]:
        weights, null_basis = state
        null_vector = null_basis[:, 0]

        rescaling_factor = jnp.where(
            null_vector > rcond, weights / null_vector, jnp.inf
        )
        argmin_index = jnp.argmin(rescaling_factor)
        updated_weights = weights - rescaling_factor[argmin_index] * null_vector
        updated_weights = updated_weights.at[argmin_index].set(jnp.inf)

        rescaled_null_vector = null_vector / null_vector[argmin_index]
        null_basis -= jnp.tensordot(
            rescaled_null_vector, null_basis[argmin_index], axes=0
        )
        null_basis = null_basis.at[argmin_index, :].set(0.0)
        updated_null_basis = jnp.roll(null_basis, -1, axis=1)
        return (updated_weights, updated_null_basis), argmin_index

    initial_state = (weights, null_basis)
    scan_vector = jnp.arange(0, kernel_rank)
    output_state, removed_indices = jax.lax.scan(_body_fn, initial_state, scan_vector)
    output_weights, _ = output_state

    def _explicit_delete(x: Array) -> Array:
        return jnp.delete(x, removed_indices, axis=0, assume_unique_indices=True)

    return _explicit_delete(output_weights), _explicit_delete(nodes)
