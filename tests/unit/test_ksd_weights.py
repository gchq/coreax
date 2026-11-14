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

"""Regression tests for unit-sum Stein-kernel quadrature weights."""

from collections.abc import Callable
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

from coreax.coreset import Coresubset
from coreax.data import Data, SupervisedData
from coreax.kernels import SquaredExponentialKernel, SteinKernel
from coreax.metrics import KSD
from coreax.weights import (
    KSDWeightsOptimiser,
    _normalised_stein_weights,  # noqa: PLC2701
)


class TestNormalisedSteinWeights:
    """Compare singular and nonsingular systems with independent constrained solves."""

    @pytest.mark.parametrize(
        "matrix",
        [
            np.diag([1.0, 2.0, 4.0]),
            np.array([[1.0, 2.0], [2.0, 5.0]]),
            np.diag([0.0, 1.0, 2.0]),
            np.diag([1.0, 0.0, 0.0]),
            np.array([[1.0, 2.0], [2.0, 4.0]]),
            np.array([[1.0, -1.0], [-1.0, 1.0]]),
            np.ones((3, 3)),
            np.zeros((3, 3)),
            np.array([[1.0, 1.0, 2.0], [1.0, 1.0, 2.0], [2.0, 2.0, 5.0]]),
            np.array([[2.0]]),
        ],
        ids=[
            "diagonal",
            "negative-weight",
            "zero-row",
            "two-zero-rows",
            "singular-negative-weight",
            "null-space-unit-sum",
            "constant",
            "zero",
            "duplicates",
            "singleton",
        ],
    )
    def test_matches_constrained_solution(
        self, matrix: np.ndarray, jit_variant: Callable[[Callable], Callable]
    ) -> None:
        """Match a double-precision constrained least-squares solution."""
        count = len(matrix)
        constraint = np.ones((count, 1))
        system = np.block([[matrix, constraint], [constraint.T, np.zeros((1, 1))]])
        target = np.r_[np.zeros(count), 1.0]
        expected = np.linalg.lstsq(system, target, rcond=None)[0][:-1]
        actual = jit_variant(_normalised_stein_weights)(jnp.asarray(matrix))
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(jnp.sum(actual), 1.0, atol=2e-6)
        uniform = np.full(count, 1 / count)
        assert actual @ matrix @ actual <= uniform @ matrix @ uniform + 2e-5

    @pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
    def test_scale_invariance(
        self, scale: float, jit_variant: Callable[[Callable], Callable]
    ) -> None:
        """Changing kernel amplitude must not change the optimal weights."""
        matrix = jnp.array([[1.0, 2.0], [2.0, 5.0]]) * scale
        actual = jit_variant(_normalised_stein_weights)(matrix)
        np.testing.assert_allclose(actual, [1.5, -0.5], atol=2e-5)


class TestKSDWeightsOptimiser:
    """Use the public optimiser with real kernels and both execution modes."""

    @pytest.mark.parametrize("epsilon", [0.0, 1e-4, 0.5])
    def test_matches_inverse_formula(
        self, epsilon: float, jit_variant: Callable[[Callable], Callable]
    ) -> None:
        """Use only the Stein Gram matrix and retain the requested regularisation."""
        points = Data(jnp.array([[-2.0], [-0.5], [0.0], [1.0], [3.0]]))
        kernel = SteinKernel(
            SquaredExponentialKernel(), score_function=lambda x: -jnp.asarray(x)
        )
        optimiser = KSDWeightsOptimiser(kernel)
        gramian = np.array(kernel.compute(points.data, points.data), dtype=float)
        gramian += epsilon * np.eye(len(points))
        expected = np.linalg.solve(gramian, np.ones(len(points)))
        expected /= expected.sum()
        actual = jit_variant(optimiser.solve)(points, points, epsilon=epsilon)
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)

    def test_duplicate_points_stay_finite(
        self, jit_variant: Callable[[Callable], Callable]
    ) -> None:
        """A singular Gram matrix should yield a symmetric, unit-sum solution."""
        points = Data(jnp.array([[0.0], [0.0], [1.0], [1.0]]))
        kernel = SteinKernel(
            SquaredExponentialKernel(), score_function=lambda x: -jnp.asarray(x)
        )
        actual = jit_variant(KSDWeightsOptimiser(kernel).solve)(
            points, points, epsilon=0.0
        )
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual[0], actual[1], atol=2e-6)
        np.testing.assert_allclose(actual[2], actual[3], atol=2e-6)
        np.testing.assert_allclose(jnp.sum(actual), 1.0, atol=2e-6)

    def test_reduces_ksd_through_coreset_interface(
        self, jit_variant: Callable[[Callable], Callable]
    ) -> None:
        """The existing coreset wrapper applies the weights without losing indices."""
        data = Data(jnp.array([[-3.0], [-1.0], [0.0], [0.3], [2.0], [4.0]]))
        coreset = Coresubset(Data(jnp.array([0, 2, 5])), data)
        kernel = SteinKernel(
            SquaredExponentialKernel(), score_function=lambda x: -jnp.asarray(x)
        )
        metric = KSD(kernel)
        weighted = jit_variant(coreset.solve_weights)(KSDWeightsOptimiser(kernel))
        before = metric.compute(
            data, coreset.points, laplace_correct=False, regularise=False
        )
        after = metric.compute(
            data, weighted.points, laplace_correct=False, regularise=False
        )
        assert after < before
        np.testing.assert_array_equal(
            weighted.unweighted_indices, coreset.unweighted_indices
        )
        np.testing.assert_allclose(jnp.sum(weighted.points.weights), 1.0, atol=2e-6)

    def test_target_dataset_does_not_change_weights(self) -> None:
        """An explicit score defines the target; no empirical kernel mean is needed."""
        kernel = SteinKernel(
            SquaredExponentialKernel(), score_function=lambda x: -jnp.asarray(x)
        )
        optimiser = KSDWeightsOptimiser(kernel)
        coreset = Data(jnp.array([[-1.0], [2.0]]))
        first = optimiser.solve(Data(jnp.array([[0.0]])), coreset)
        second = optimiser.solve(Data(jnp.array([[100.0], [200.0]])), coreset)
        np.testing.assert_array_equal(first, second)

    def test_rejects_non_stein_kernel(self) -> None:
        """A generic kernel does not have the required zero mean embedding."""
        with pytest.raises(ValueError, match="SteinKernel"):
            KSDWeightsOptimiser(cast(SteinKernel, SquaredExponentialKernel()))

    @pytest.mark.parametrize("supervised", ["dataset", "coreset"])
    def test_rejects_supervised_data(self, supervised: str) -> None:
        """Match the input restrictions of the other weighting interfaces."""
        plain = Data(jnp.array([[0.0]]))
        labelled = SupervisedData(plain.data, plain.data)
        kernel = SteinKernel(
            SquaredExponentialKernel(), score_function=lambda x: -jnp.asarray(x)
        )
        with pytest.raises(ValueError, match="unsupervised"):
            KSDWeightsOptimiser(kernel).solve(
                labelled if supervised == "dataset" else plain,
                labelled if supervised == "coreset" else plain,
            )

    def test_rejects_empty_coreset(self) -> None:
        """An empty coreset cannot carry weights whose sum is one."""
        kernel = SteinKernel(
            SquaredExponentialKernel(), score_function=lambda x: -jnp.asarray(x)
        )
        with pytest.raises(ValueError, match="at least one"):
            KSDWeightsOptimiser(kernel).solve(
                Data(jnp.zeros((1, 1))), Data(jnp.zeros((0, 1)))
            )

    @pytest.mark.parametrize("epsilon", [-1.0, float("nan"), float("inf")])
    def test_rejects_invalid_regularisation(self, epsilon: float) -> None:
        """Invalid diagonal shifts must not silently produce invalid weights."""
        points = Data(jnp.array([[0.0]]))
        kernel = SteinKernel(
            SquaredExponentialKernel(), score_function=lambda x: -jnp.asarray(x)
        )
        with pytest.raises(RuntimeError, match="finite and non-negative"):
            KSDWeightsOptimiser(kernel).solve(points, points, epsilon=epsilon)

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_rejects_non_finite_gramian(self, value: float) -> None:
        """Invalid kernel values must be reported before eigendecomposition."""
        points = Data(jnp.array([[value]]))
        kernel = SteinKernel(
            SquaredExponentialKernel(), score_function=lambda x: -jnp.asarray(x)
        )
        with pytest.raises(RuntimeError, match="Gram matrix must be finite"):
            KSDWeightsOptimiser(kernel).solve(points, points)
