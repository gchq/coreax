"""Test recombination algorithms."""
import warnings
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import scipy
from jaxtyping import Array, ArrayLike, Inexact, Shaped

from coreax.recombination import caratheodory_measure_reduction, reveal_kernel_rank

jax.config.update("jax_enable_x64", True)
InexactScalarLike = Inexact[ArrayLike, ""]


# pylint: disable=line-too-long
# Credit: https://github.com/FraCose/Recombination_Random_Algos/blob/2ca4ff74279eb1604376723dcb00ee65ff7e519a/recombination.py#L914
# pylint: enable=line-too-long
# The below implementation provides additional support for a Rank-Revealing QR mode.
def _reference_caratheodory_measure_reduction(
    weights: Shaped[np.ndarray, " n"],
    nodes: Shaped[np.ndarray, "n d"],
    rcond: InexactScalarLike | None = None,
    mode: Literal["svd", "qr"] = "svd",
) -> tuple[Shaped[np.ndarray, " d+1"], Shaped[np.ndarray, "d+1 d"]]:
    augmented_nodes = np.insert(nodes, 0, 1.0, axis=1)
    if mode == "svd":
        _, s, vt = np.linalg.svd(augmented_nodes.T)
        kernel_rank, rcond = reveal_kernel_rank(augmented_nodes, s, rcond)
        null_basis = vt[-kernel_rank:, :].T
    elif mode == "qr":
        q, r, _ = scipy.linalg.qr(augmented_nodes, mode="full", pivoting=True)
        kernel_rank, rcond = reveal_kernel_rank(
            augmented_nodes, np.abs(np.diag(r)), rcond
        )
        null_basis = q[:, -kernel_rank:]
    else:
        msg = f"`mode` must be one of 'svd' or 'qr'; got {mode}"
        raise ValueError(msg)

    # pylint: disable=duplicate-code
    if nodes.shape[0] <= kernel_rank:
        warnings.warn(
            "Nothing to do; node count already below reduction threshold",
            stacklevel=2,
        )
        return weights, nodes
    # pylint: enable=duplicate-code

    for _ in range(kernel_rank):
        null_vector = null_basis[:, 0]
        # Ignore warnings from dividing zero by zero. These values are filtered out by
        # the ``np.where`` clause.
        with np.errstate(invalid="ignore"):
            rescaling_factor = np.where(
                null_vector > rcond, weights / null_vector, np.inf
            )
        argmin_index = np.argmin(rescaling_factor)
        weights = weights - rescaling_factor[argmin_index] * null_vector
        weights[argmin_index] = 0.0

        null_basis = np.delete(null_basis, 0, axis=1)
        rescaled_null_vector = null_vector / null_vector[argmin_index]
        null_basis -= np.tensordot(
            rescaled_null_vector, null_basis[argmin_index], axes=0
        )
        null_basis[argmin_index, :] = 0.0

    w_star = weights[weights > 0]
    nodes_star = nodes[weights > 0]
    return w_star, nodes_star


@pytest.fixture(params=[(100, 10), (10, 100), (50, 1), (1, 50)])
def shaped_array(request) -> Array:
    """Fixture generates random arrays of varying shape."""
    return jr.normal(jr.key(0), request.param)


@pytest.fixture(params=[None, np.asarray(1e-12), jnp.asarray([1e-12]), 1e-12])
def rcond(request):
    """Fixture generates relative condition numbers of varying type."""
    return request.param


def test_reveal_kernel_rank(shaped_array, rcond):
    """Test ``reveal_kernel_rank``."""
    rank = np.linalg.matrix_rank(shaped_array, tol=rcond)
    expected_kernel_rank = max(shaped_array.shape) - rank
    _, s, _ = scipy.linalg.svd(shaped_array)
    kernel_rank, _ = reveal_kernel_rank(shaped_array, s, rcond)
    assert kernel_rank == expected_kernel_rank


# Add "qr" to mode once support is available in JAX.
# see https://github.com/google/jax/issues/12897
@pytest.mark.parametrize("mode", ["svd"])
def test_caratheodory_measure_reduction_invariants(
    shaped_array, rcond, mode: Literal["svd"]
):
    """
    Test Caratheodory measure reduction preserves the centre-of-mass.
    """
    nodes = shaped_array
    positive_weights = jr.uniform(jr.key(1234), (nodes.shape[0],))
    probability_weights = positive_weights / np.sum(positive_weights)
    expected_com = np.average(nodes, 0, weights=probability_weights)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        reference_weights, reference_nodes = _reference_caratheodory_measure_reduction(
            np.asarray(probability_weights), np.asarray(nodes), rcond=rcond, mode=mode
        )
        caratheodory_weights, caratheodory_nodes = caratheodory_measure_reduction(
            probability_weights, nodes, rcond=rcond, mode=mode
        )

        if nodes.shape[0] < nodes.shape[1]:
            expected_warning_count = 2
            assert len(record) == expected_warning_count
            assert all("Nothing to do" in str(warn.message) for warn in record)
            expected_weights_shape = probability_weights.shape
            expected_nodes_shape = nodes.shape
        else:
            expected_weights_shape = (nodes.shape[-1] + 1,)
            expected_nodes_shape = (nodes.shape[-1] + 1, nodes.shape[-1])

    # Check reference implementation
    reference_com = np.average(reference_nodes, 0, weights=reference_weights)
    assert reference_weights.shape == expected_weights_shape
    assert reference_nodes.shape == expected_nodes_shape
    np.testing.assert_almost_equal(np.sum(reference_weights), 1.0)
    np.testing.assert_array_almost_equal(reference_com, expected_com)

    # Weak sense equivalence to the reference implementation
    recombined_com = np.average(caratheodory_nodes, 0, weights=caratheodory_weights)
    np.testing.assert_almost_equal(np.sum(caratheodory_weights), 1.0)
    np.testing.assert_array_almost_equal(recombined_com, reference_com)

    # Strong sense equivalence to the reference implementation
    # IMPORTANT: we only expect strong equivalence if the implementations are, for
    # all intents and purposes, identical. There exist many (non-unique) solutions
    # to the caratheodory measure reduction problem which, for the purposes
    # considered here, are equivalent; we want *any* basic feasible solution to
    # the underlying linear programming (optimisation) problem, where the objective
    # function is trivial. E.G. max c^T x, with c = 0.
    assert caratheodory_weights.shape == reference_weights.shape
    assert caratheodory_nodes.shape == reference_nodes.shape
    np.testing.assert_array_almost_equal(caratheodory_weights, reference_weights)
    np.testing.assert_array_almost_equal(caratheodory_nodes, reference_nodes)


# def test_rank_degenerate_invariants(self):
#     ...

# Test jit and differentiability in forward and reverse mode.
