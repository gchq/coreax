"""Test recombination algorithms."""
from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from typing import Literal, get_args

import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest
import scipy
from jaxtyping import Array, ArrayLike, Inexact

from coreax.recombination import (
    caratheodory_measure_reduction,
    reveal_left_null_space_rank,
)

jax.config.update("jax_enable_x64", True)
InexactScalarLike = Inexact[ArrayLike, ""]

N = 100
D = 10

DEGENERACIES = Literal[None, "shape", "null", "co-linear", "convex"]


@pytest.mark.parametrize("rcond", [None, 1e-6, 1e-12])
@pytest.mark.parametrize("shape", [(N, D), (D, D)])
class TestRecombinationCases:
    rng_seed = 0

    @pytest.fixture
    def atomic_measure_factory(self):
        """Return a generator for random atomic measures with a specified degeneracy."""
        key_generator = eqxi.GetKey(self.rng_seed)

        def random_atomic_measure_generator(
            shape: tuple[int, ...],
            rcond: float | None,
            *,
            n_repeats: int = 3,
            degeneracy: DEGENERACIES,
        ):
            """Yield a random atomic measure with a specified shape and degeneracy."""
            *_, n, d = shape

            iteration_count = 0
            while iteration_count < n_repeats:
                weight_key, node_key = jr.split(key_generator())
                weights = jr.uniform(weight_key, (shape[0],))
                weights /= jnp.sum(weights)
                assert jnp.all(weights >= 0) and jnp.isclose(jnp.sum(weights), 1.0)
                nodes = jr.normal(node_key, shape)

                _matrix_rank = jtu.Partial(jnp.linalg.matrix_rank, tol=rcond)
                is_degenerate = _matrix_rank(weights[..., None] * nodes) < d
                if degeneracy is None:
                    if is_degenerate:
                        if n < d:
                            raise ValueError(
                                "It is impossible to generate a non-degenerate"
                                f"(rank >= d) node matrix with shape '{shape}'"
                            )
                        continue
                    pass
                else:
                    if degeneracy == "shape":
                        weights = weights[: (d - 1)]
                        nodes = nodes[: (d - 1)]
                    elif degeneracy == "co-linear":
                        nodes = nodes.at[(d - 1) :].set(nodes.at[d - 1].get())
                    elif degeneracy == "convex":
                        com = np.average(nodes, axis=0, weights=weights)
                        # Generate two vectors that are co-linear w.r.t the com.
                        nodes = nodes.at[d - 1].set(com - 1)
                        nodes = nodes.at[d].set(com + 1)
                    elif degeneracy == "null":
                        nodes = nodes.at[(d - 1) :].set(0.0)
                    else:
                        raise ValueError(
                            f"Degeneracy must be one of {get_args(DEGENERACIES)}; "
                            f"got '{degeneracy}'"
                        )
                    is_rank_deficient = _matrix_rank(weights[..., None] * nodes) < d
                    is_other_degeneracy = degeneracy in {"co-linear", "convex"}
                    assert is_rank_deficient or is_other_degeneracy
                yield weights, nodes
                iteration_count += 1

        return random_atomic_measure_generator

    @pytest.mark.parametrize("degeneracy", get_args(DEGENERACIES))
    def test_reveal_left_null_space_rank(
        self,
        atomic_measure_factory: Callable[..., Iterator[tuple[Array, Array]]],
        shape: tuple[int, ...],
        rcond: float | None,
        degeneracy: DEGENERACIES,
    ):
        for _, nodes in atomic_measure_factory(shape, rcond, degeneracy=degeneracy):
            rank = np.linalg.matrix_rank(nodes, tol=rcond)
            expected_null_space_rank = max(0, nodes.shape[0] - rank)
            _, s, _ = scipy.linalg.svd(nodes)
            null_space_rank = reveal_left_null_space_rank(nodes.shape, s, rcond)
            assert null_space_rank == expected_null_space_rank

    @pytest.mark.parametrize("mode", ["svd", "qr"])
    @pytest.mark.parametrize(
        "assume_non_degenerate, degeneracy",
        [(False, i) for i in get_args(DEGENERACIES)] + [(True, None)],
    )
    def test_caratheodory_measure_reduction(
        self,
        atomic_measure_factory: Callable[..., Iterator[tuple[Array, Array]]],
        shape: tuple[int, ...],
        rcond: float | None,
        mode: Literal["svd", "qr"],
        assume_non_degenerate: bool,
        degeneracy: DEGENERACIES,
    ):
        case_iterator = atomic_measure_factory(shape, rcond, degeneracy=degeneracy)
        for weights, nodes in case_iterator:
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                result_weights, result_nodes = caratheodory_measure_reduction(
                    weights,
                    nodes,
                    rcond=rcond,
                    mode=mode,
                    assume_non_degenerate=assume_non_degenerate,
                )
            rank_plus_1 = jnp.linalg.matrix_rank(nodes, tol=rcond) + 1
            leading_shape = min(nodes.shape[0], rank_plus_1)
            if leading_shape == nodes.shape[0]:
                assert all("Nothing to do" in str(warn.message) for warn in record)
            else:
                assert len(record) == 0
            expected_weights_shape = (leading_shape,)
            expected_nodes_shape = (leading_shape, nodes.shape[-1])

            result_com = np.average(result_nodes, 0, weights=result_weights)
            expected_com = np.average(nodes, 0, weights=weights)

            assert result_weights.shape <= expected_weights_shape
            assert result_nodes.shape <= expected_nodes_shape
            np.testing.assert_almost_equal(np.sum(result_weights), 1.0)
            np.testing.assert_array_almost_equal(result_com, expected_com)
