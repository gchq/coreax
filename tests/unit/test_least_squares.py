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
from contextlib import (
    AbstractContextManager,
    nullcontext as does_not_raise,
)
from typing import Generic, NamedTuple, TypeVar, Union

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array
from typing_extensions import override

from coreax.kernels import SquaredExponentialKernel
from coreax.least_squares import (
    MinimalEuclideanNormSolver,
    RandomisedEigendecompositionSolver,
    RegularisedLeastSquaresSolver,
)
from coreax.util import KeyArrayLike

_RegularisedLeastSquaresSolver = TypeVar(
    "_RegularisedLeastSquaresSolver", bound=RegularisedLeastSquaresSolver
)


class _InverseProblem(NamedTuple):
    random_key: KeyArrayLike
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

    def test_approximation_accuracy(
        self, problem: _InverseProblem, approximator: _RegularisedLeastSquaresSolver
    ) -> None:
        """
        Verify approximator performance on toy problem.

        Here we focus on checking how accurately we can recover the inverse of arrays.
        """
        _, array, regularisation_parameter, target, identity, expected_inverse = problem

        approximate_inverse = approximator.solve(
            array=array,
            regularisation_parameter=regularisation_parameter,
            target=target,
            identity=identity,
        )

        assert jnp.linalg.norm(expected_inverse - approximate_inverse) == pytest.approx(
            0.0, abs=1e-1
        )

        # Approximate stacks of kernel matrix inverses
        approximate_inverses = approximator.solve_stack(
            arrays=jnp.array((array, array)),
            regularisation_parameter=regularisation_parameter,
            targets=jnp.array((identity, identity)),
            identity=identity,
        )

        expected_inverses = jnp.array((expected_inverse, expected_inverse))
        assert jnp.linalg.norm(
            expected_inverses - approximate_inverses
        ) == pytest.approx(0.0, abs=1e-1)


class TestMinimalEuclideanNormSolver:
    """Test `MinimalEuclideanNormSolver`."""

    @pytest.fixture(scope="class")
    def approximator(self) -> MinimalEuclideanNormSolver:
        """Pytest fixture returns an initialised `MinimalEuclideanNormSolver`."""
        return MinimalEuclideanNormSolver(rcond=None)

    def test_approximator_accuracy(
        self, approximator: MinimalEuclideanNormSolver
    ) -> None:
        """
        Test `MinimalEuclideanNormSolver` is accurate via an analytical example.

        We focus on recovering the inverse of an array.
        """
        regularisation_parameter = 1
        identity = jnp.eye(2)
        array = jnp.ones((2, 2))

        expected_output = jnp.array([[2 / 3, -1 / 3], [-1 / 3, 2 / 3]])

        output = approximator.solve(
            array=array,
            regularisation_parameter=regularisation_parameter,
            target=identity,
            identity=identity,
        )
        assert jnp.linalg.norm(output - expected_output) == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.parametrize(
        "array, identity, rcond, context",
        [
            (jnp.eye(2), jnp.eye(2), 1e-6, does_not_raise()),
            (jnp.eye(2), jnp.eye(2), -1, does_not_raise()),
        ],
        ids=[
            "valid_rcond_not_negative_one_or_none",
            "valid_rcond_negative_one",
        ],
    )
    def test_approximator_inputs(
        self,
        array: Array,
        identity: Array,
        rcond: Union[int, float, None],
        context: AbstractContextManager,
    ) -> None:
        """Test `MinimalEuclideanNormSolver` handles inputs as expected."""
        with context:
            approximator = MinimalEuclideanNormSolver(rcond=rcond)
            approximator.solve(
                array=array,
                regularisation_parameter=1e-3,
                target=identity,
                identity=identity,
            )


class TestRandomisedEigendecompositionSolver(
    InverseApproximationTest[RandomisedEigendecompositionSolver],
):
    """Test `RandomisedEigendecompositionSolver`."""

    @override
    @pytest.fixture(scope="class")
    def approximator(self) -> RandomisedEigendecompositionSolver:
        random_seed = 2_024
        return RandomisedEigendecompositionSolver(
            random_key=jr.key(random_seed),
            oversampling_parameter=100,
            power_iterations=1,
            rcond=None,
        )

    @override
    @pytest.fixture(scope="class")
    def problem(self):
        random_key = jr.key(2_024)
        num_data_points = 2000
        dimension = 2
        identity = jnp.eye(num_data_points)
        regularisation_parameter = 1e-3

        # Compute kernel matrix from standard normal data
        x = jr.normal(random_key, (num_data_points, dimension))
        array = SquaredExponentialKernel().compute(x, x)

        # Compute "exact" inverse
        exact_inverter = MinimalEuclideanNormSolver(rcond=None)
        expected_inverse = exact_inverter.solve(
            array, regularisation_parameter, identity, identity
        )

        return _InverseProblem(
            random_key,
            array,
            regularisation_parameter,
            identity,
            identity,
            expected_inverse,
        )

    @pytest.mark.parametrize(
        "array, identity, rcond, context",
        [
            (jnp.eye(2), jnp.eye(2), -10, pytest.raises(ValueError)),
            (jnp.eye(2), jnp.eye(2), 1e-6, does_not_raise()),
            (jnp.eye(2), jnp.eye(2), -1, does_not_raise()),
        ],
        ids=[
            "rcond_negative_not_negative_one",
            "valid_rcond_not_negative_one_or_none",
            "valid_rcond_negative_one",
        ],
    )
    def test_approximator_inputs(
        self,
        array: Array,
        identity: Array,
        rcond: Union[int, float, None],
        context: AbstractContextManager,
    ) -> None:
        """Test `RandomisedEigendecompositionApproximator` handles invalid inputs."""
        with context:
            approximator = RandomisedEigendecompositionSolver(
                random_key=jr.key(0),
                oversampling_parameter=1,
                power_iterations=1,
                rcond=rcond,
            )
            approximator.solve(
                array=array,
                regularisation_parameter=1e-6,
                target=identity,
                identity=identity,
            )

    def test_randomised_eigendecomposition_accuracy(self, problem) -> None:
        """Test that the `randomised_eigendecomposition` method is accurate."""
        # Unpack problem data
        random_key, array, _, _, _, _ = problem
        oversampling_parameter = 100
        power_iterations = 1
        solver = RandomisedEigendecompositionSolver(
            random_key=random_key,
            oversampling_parameter=oversampling_parameter,
            power_iterations=power_iterations,
        )

        eigenvalues, eigenvectors = solver.randomised_eigendecomposition(array)
        assert jnp.linalg.norm(
            array - (eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T)
        ) == pytest.approx(0.0, abs=1)

    @pytest.mark.parametrize(
        "array, oversampling_parameter, power_iterations",
        [
            (jnp.zeros((2, 2, 2)), 1, 1),
            (jnp.zeros((2, 3)), 1, 1),
            (jnp.eye(2), 1.0, 1),
            (jnp.eye(2), -1, 1),
            (jnp.eye(2), 1, 1.0),
            (jnp.eye(2), 1, -1),
        ],
        ids=[
            "larger_than_two_d_array",
            "non_square_array",
            "float_oversampling_parameter",
            "negative_oversampling_parameter",
            "float_power_iterations",
            "negative_power_iterations",
        ],
    )
    def test_randomised_eigendecomposition_invalid_inputs(
        self,
        array: Array,
        oversampling_parameter: int,
        power_iterations: int,
    ) -> None:
        """Test that `randomised_eigendecomposition` handles invalid inputs."""
        with pytest.raises(ValueError):
            solver = RandomisedEigendecompositionSolver(
                random_key=jr.key(2_024),
                oversampling_parameter=oversampling_parameter,
                power_iterations=power_iterations,
            )

            solver.randomised_eigendecomposition(array)
