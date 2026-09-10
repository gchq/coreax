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

"""Regression tests for truncated Hermitian eigendecomposition solves."""

from collections.abc import Callable

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from coreax.least_squares import RandomisedEigendecompositionSolver


class TestHermitianSolves:
    """Check spectral truncation and conjugation with and without compilation."""

    @pytest.mark.parametrize(
        "matrix",
        [
            np.zeros((3, 3)),
            np.diag([-4.0, -2.0, -1.0]),
            np.diag([-4.0, 0.0, 2.0]),
            np.array([[2.0, 1j, 0.0], [-1j, 3.0, 1j], [0.0, -1j, -2.0]]),
        ],
        ids=["zero", "negative-definite", "singular-indefinite", "complex"],
    )
    @pytest.mark.parametrize("regularisation", [0.0, 0.5, -0.5])
    def test_solve_matches_pseudo_inverse(
        self,
        matrix: np.ndarray,
        regularisation: float,
        jit_variant: Callable[[Callable], Callable],
    ) -> None:
        """Keep eigenvalue signs and solve complex systems against an SVD oracle."""
        array = jnp.asarray(matrix)
        identity = jnp.eye(3, dtype=array.dtype)
        target = jnp.asarray([[1.0, 2.0], [-2.0, 0.0], [3.0, -1.0]])
        solver = RandomisedEigendecompositionSolver(
            jr.key(2024), oversampling_parameter=3, rcond=1e-5
        )
        actual = jit_variant(solver.solve)(array, regularisation, target, identity)
        regularised_matrix = matrix + abs(regularisation) * np.eye(3)
        expected = np.linalg.pinv(regularised_matrix, rcond=1e-5) @ np.asarray(target)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    @pytest.mark.parametrize("rcond", [None, -1, 0.0, 1e-4])
    def test_zero_system_stays_finite(
        self, rcond: float | None, jit_variant: Callable[[Callable], Callable]
    ) -> None:
        """Never invert exact zero eigenvalues, including with a zero cut-off."""
        solver = RandomisedEigendecompositionSolver(jr.key(0), rcond=rcond)
        actual = jit_variant(solver.solve)(
            jnp.zeros((2, 2)), 0.0, jnp.eye(2), jnp.eye(2)
        )
        np.testing.assert_array_equal(actual, np.zeros((2, 2)))

    def test_rank_truncation_uses_magnitude(
        self, jit_variant: Callable[[Callable], Callable]
    ) -> None:
        """Use magnitude for the cut-off while retaining either eigenvalue sign."""
        eigenvalues = jnp.array([-8.0, -2.0, -0.5, 0.0, 0.5, 1.0])
        matrix = jnp.diag(eigenvalues)
        identity = jnp.eye(len(eigenvalues))
        solver = RandomisedEigendecompositionSolver(
            jr.key(5), oversampling_parameter=len(eigenvalues), rcond=0.0625
        )
        actual = jit_variant(solver.solve)(matrix, 0.0, identity, identity)
        expected = np.diag([-0.125, -0.5, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    @pytest.mark.parametrize("power_iterations", [0, 2])
    def test_low_rank_complex_reconstruction(
        self, power_iterations: int, jit_variant: Callable[[Callable], Callable]
    ) -> None:
        """Keep the Hermitian projection and orthonormal basis at reduced rank."""
        basis, _ = jnp.linalg.qr(
            jnp.array(
                [
                    [1.0, 1j],
                    [1j, 2.0],
                    [2.0, -1j],
                    [-1.0, 1.0],
                ]
            )
        )
        matrix = (basis * jnp.array([-4.0, 2.0])) @ jnp.conjugate(basis).T
        solver = RandomisedEigendecompositionSolver(
            jr.key(2024), oversampling_parameter=2, power_iterations=power_iterations
        )
        values, vectors = jit_variant(solver.randomised_eigendecomposition)(matrix)
        gram = jnp.conjugate(vectors).T @ vectors
        np.testing.assert_allclose(gram, np.eye(2), atol=2e-5)
        reconstructed = (vectors * values) @ jnp.conjugate(vectors).T
        np.testing.assert_allclose(reconstructed, matrix, atol=2e-5)
        inverse = jit_variant(solver.solve)(matrix, 0.0, jnp.eye(4), jnp.eye(4))
        np.testing.assert_allclose(matrix @ inverse @ matrix, matrix, atol=2e-5)
        np.testing.assert_allclose(inverse @ matrix @ inverse, inverse, atol=2e-5)
        np.testing.assert_allclose(inverse, jnp.conjugate(inverse).T, atol=2e-5)

    def test_batched_complex_systems(
        self, jit_variant: Callable[[Callable], Callable]
    ) -> None:
        """Apply the same spectral handling when solves are vectorised."""
        arrays = jnp.array([[[2.0, 1j], [-1j, -3.0]], [[0.0, 0.0], [0.0, 0.0]]])
        identities = jnp.stack([jnp.eye(2), jnp.eye(2)])
        solver = RandomisedEigendecompositionSolver(jr.key(3), oversampling_parameter=2)
        actual = jit_variant(solver.solve_stack)(arrays, 0.0, identities, jnp.eye(2))
        expected = np.linalg.pinv(np.asarray(arrays))
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
