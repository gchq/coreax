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

"""
Tests for approximation approaches.

Approximations are used to reduce computational demand when computing coresets. The
tests within this file verify that these approximations produce the expected results on
simple examples.
"""

from abc import ABC, abstractmethod
from typing import Generic, NamedTuple, TypeVar

import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array
from typing_extensions import override

from coreax.kernels import SquaredExponentialKernel
from coreax.least_squares import (
    MinimalEuclideanNormSolver,
    RandomisedEigendecompositionSolver,
    RegularisedLeastSquaresSolver,
)

_RegularisedLeastSquaresSolver = TypeVar(
    "_RegularisedLeastSquaresSolver", bound=RegularisedLeastSquaresSolver
)


class _InverseProblem(NamedTuple):
    array: Array
    regularisation_parameter: float
    target: Array
    identity: Array
    expected_inverse: Array


class InverseApproximationTest(ABC, Generic[_RegularisedLeastSquaresSolver]):
    """Tests related to inverting kernel gramians via methods in least_squares.py."""

    @pytest.fixture
    @abstractmethod
    def approximator(self) -> _RegularisedLeastSquaresSolver:
        """Abstract pytest fixture which initialises a least-squares approximator."""

    @pytest.fixture
    @abstractmethod
    def problem(self) -> _InverseProblem:
        """Abstract fixture which returns a problem for least-squares approximation."""

    def test_solve(
        self, problem: _InverseProblem, approximator: _RegularisedLeastSquaresSolver
    ) -> None:
        """
        Verify approximator solve on toy problem.

        Here we focus on checking how accurately we can recover the inverse of arrays.
        """
        array, regularisation_parameter, target, identity, expected_inverse = problem

        approximate_inverse = approximator.solve(
            array=array,
            regularisation_parameter=regularisation_parameter,
            target=target,
            identity=identity,
        )
        abs_err = jnp.linalg.norm(expected_inverse - approximate_inverse)
        assert abs_err == pytest.approx(0.0, abs=1e-1)

    def test_solve_stack(
        self, problem: _InverseProblem, approximator: _RegularisedLeastSquaresSolver
    ) -> None:
        """
        Verify approximator stack solve on toy problem.

        Here we focus on checking how accurately we can recover the inverse of arrays.
        """
        array, regularisation_parameter, target, identity, expected_inverse = problem
        # Approximate stacks of kernel matrix inverses
        approximate_inverses = approximator.solve_stack(
            arrays=jnp.array((array, array)),
            regularisation_parameter=regularisation_parameter,
            targets=jnp.array((target, target)),
            identity=identity,
        )
        expected_inverses = jnp.array((expected_inverse, expected_inverse))
        abs_err = jnp.linalg.norm(expected_inverses - approximate_inverses)
        assert abs_err == pytest.approx(0.0, abs=1e-1)


class TestMinimalEuclideanNormSolver(
    InverseApproximationTest[MinimalEuclideanNormSolver]
):
    """Test `MinimalEuclideanNormSolver`."""

    @pytest.fixture(scope="class")
    def approximator(self) -> MinimalEuclideanNormSolver:
        """Pytest fixture returns an initialised `MinimalEuclideanNormSolver`."""
        return MinimalEuclideanNormSolver(rcond=None)

    @override
    @pytest.fixture(scope="class")
    def problem(self):
        regularisation_parameter = 1
        identity = jnp.eye(2)
        array = jnp.ones((2, 2))

        expected_inverse = jnp.array([[2 / 3, -1 / 3], [-1 / 3, 2 / 3]])
        return _InverseProblem(
            array,
            regularisation_parameter,
            identity,
            identity,
            expected_inverse,
        )


class TestRandomisedEigendecompositionSolver(
    InverseApproximationTest[RandomisedEigendecompositionSolver],
):
    """Test `RandomisedEigendecompositionSolver`."""

    SEED = 2_024

    @override
    @pytest.fixture(scope="class")
    def approximator(self) -> RandomisedEigendecompositionSolver:
        return RandomisedEigendecompositionSolver(
            random_key=jr.key(self.SEED),
            oversampling_parameter=50,
            power_iterations=1,
            rcond=None,
        )

    @override
    @pytest.fixture(scope="class")
    def problem(self):
        random_key = jr.key(self.SEED)
        num_data_points = 50
        dimension = 2
        identity = jnp.eye(num_data_points)
        regularisation_parameter = 1e-1

        # Compute kernel matrix from standard normal data
        x = jr.normal(random_key, (num_data_points, dimension))
        array = SquaredExponentialKernel().compute(x, x)
        # Compute "exact" inverse
        exact_inverter = MinimalEuclideanNormSolver(rcond=None)
        expected_inverse = exact_inverter.solve(
            array, regularisation_parameter, identity, identity
        )

        return _InverseProblem(
            array,
            regularisation_parameter,
            identity,
            identity,
            expected_inverse,
        )

    @pytest.mark.parametrize(
        "oversampling_parameter,power_iterations,rcond,msg",
        [
            (1.0, 1, -1, "'oversampling_parameter' must be a positive integer"),
            (0.0, 1, 1.0, "'oversampling_parameter' must be a positive integer"),
            (1, -1, 1.0, "'power_iterations' must be a non-negative integer"),
            (1, 1.0, -1, "'power_iterations' must be a non-negative integer"),
            (1, 1, -1.2, "'rcond' must be non-negative or -1"),
        ],
        ids=[
            "invalid_float_oversampling_parameter",
            "invalid_zero_oversampling_parameter",
            "invalid_float_power_iterations",
            "invalid_zero_power_iterations",
            "invalid_rcond",
        ],
    )
    def test_check_init(self, oversampling_parameter, power_iterations, rcond, msg):
        """Test the `__check_init__` magic of `RandomisedEigendecompositionSolver`."""
        with pytest.raises(ValueError, match=msg):
            RandomisedEigendecompositionSolver(
                random_key=jr.key(self.SEED),
                oversampling_parameter=oversampling_parameter,
                power_iterations=power_iterations,
                rcond=rcond,
            )

    def test_randomised_eigendecomposition_accuracy(self, problem) -> None:
        """Test that the `randomised_eigendecomposition` method is accurate."""
        # Unpack problem data
        array, _, _, _, _ = problem
        oversampling_parameter = 100
        power_iterations = 1
        solver = RandomisedEigendecompositionSolver(
            random_key=jr.key(self.SEED),
            oversampling_parameter=oversampling_parameter,
            power_iterations=power_iterations,
        )

        eigenvalues, eigenvectors = solver.randomised_eigendecomposition(array)
        assert jnp.linalg.norm(
            array - (eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T)
        ) == pytest.approx(0.0, abs=1)

    @pytest.mark.parametrize(
        "shape",
        [(2, 2, 2), (2, 3)],
        ids=["batched_array", "non_square_array"],
    )
    def test_randomised_eigendecomposition_invalid_inputs(
        self, shape: tuple[int, ...]
    ) -> None:
        """Test that `randomised_eigendecomposition` handles invalid inputs."""
        with pytest.raises(ValueError):
            solver = RandomisedEigendecompositionSolver(random_key=jr.key(self.SEED))
            array = jnp.ones(shape)
            solver.randomised_eigendecomposition(array)
