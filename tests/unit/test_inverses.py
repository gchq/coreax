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
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Generic, NamedTuple, TypeVar, Union

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array
from typing_extensions import override

from coreax.inverses import (
    LeastSquareApproximator,
    NystromApproximator,
    RandomisedEigendecompositionApproximator,
    RegularisedInverseApproximator,
)
from coreax.kernel import SquaredExponentialKernel, median_heuristic
from coreax.util import KeyArrayLike

_RegularisedInverseApproximator = TypeVar(
    "_RegularisedInverseApproximator", bound=RegularisedInverseApproximator
)


class _Problem(NamedTuple):
    random_key: KeyArrayLike
    array: Array
    regularisation_parameter: float
    identity: Array
    expected_inv: Array


class InverseApproximationTest(ABC, Generic[_RegularisedInverseApproximator]):
    """Tests related to kernel inverse approximations in approximation.py."""

    @abstractmethod
    def approximator(self) -> _RegularisedInverseApproximator:
        """Abstract pytest fixture which initialises an inverse approximator."""

    @abstractmethod
    def problem(self) -> _Problem:
        """Abstract pytest fixture which returns a problem for inverse approximation."""

    def test_approximation_accuracy(
        self, problem: _Problem, approximator: _RegularisedInverseApproximator
    ) -> None:
        """Verify approximator performance on toy problem."""
        _, array, regularisation_parameter, identity, expected_inverse = problem

        approximate_inverse = approximator.approximate(
            array=array,
            regularisation_parameter=regularisation_parameter,
            identity=identity,
        )

        assert jnp.linalg.norm(expected_inverse - approximate_inverse) == pytest.approx(
            0.0, abs=1
        )

        # Approximate stacks of kernel inverses
        approximate_inverses = approximator.approximate_stack(
            arrays=jnp.array((array, array)),
            regularisation_parameter=regularisation_parameter,
            identity=identity,
        )

        expected_inverses = jnp.array((expected_inverse, expected_inverse))
        assert jnp.linalg.norm(
            expected_inverses - approximate_inverses
        ) == pytest.approx(0.0, abs=1)


class TestLeastSquareApproximator:
    """Test LeastSquareApproximator."""

    @pytest.fixture(scope="class")
    def approximator(self) -> LeastSquareApproximator:
        """Abstract pytest fixture returns an initialised inverse approximator."""
        random_seed = 2_024
        return LeastSquareApproximator(random_key=jr.key(random_seed), rcond=None)

    def test_approximator_accuracy(self, approximator: LeastSquareApproximator) -> None:
        """Test LeastSquareApproximator is accurate with an analytical example."""
        regularisation_parameter = 1
        identity = jnp.eye(2)
        array = jnp.ones((2, 2))

        expected_output = jnp.array([[2 / 3, -1 / 3], [-1 / 3, 2 / 3]])

        output = approximator.approximate(
            array=array,
            regularisation_parameter=regularisation_parameter,
            identity=identity,
        )
        assert jnp.linalg.norm(output - expected_output) == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.parametrize(
        "array, identity, rcond, context",
        [
            (jnp.eye(2), jnp.eye(2), 1e-6, does_not_raise()),
            (jnp.eye(2), jnp.eye(2), -1, does_not_raise()),
            (jnp.eye(2), jnp.eye(3), None, pytest.raises(ValueError)),
        ],
        ids=[
            "valid_rcond_not_negative_one_or_none",
            "valid_rcond_negative_one",
            "invalid_array_shapes",
        ],
    )
    def test_approximator_inputs(
        self,
        array: Array,
        identity: Array,
        rcond: Union[int, float, None],
        context: AbstractContextManager,
    ) -> None:
        """Test LeastSquareApproximator handles inputs as expected."""
        with context:
            approximator = LeastSquareApproximator(
                random_key=jr.key(0),
                rcond=rcond,
            )
            approximator.approximate(
                array=array,
                regularisation_parameter=1e-6,
                identity=identity,
            )


class TestRandomisedEigendecompositionApproximator(
    InverseApproximationTest[RandomisedEigendecompositionApproximator],
):
    """Test RandomisedEigendecompositionApproximator."""

    random_seed = 2_024

    @override
    @pytest.fixture(scope="class")
    def approximator(self) -> RandomisedEigendecompositionApproximator:
        """Abstract pytest fixture returns an initialised inverse approximator."""
        return RandomisedEigendecompositionApproximator(
            random_key=jr.key(self.random_seed),
            oversampling_parameter=100,
            power_iterations=1,
            rcond=None,
        )

    @override
    @pytest.fixture(scope="class")
    def problem(self) -> _Problem:
        """Define data shared across tests."""
        random_key = jr.key(2_024)
        num_data_points = 1000
        dimension = 2
        identity = jnp.eye(num_data_points)
        regularisation_parameter = 1e-6

        # Compute kernel matrix from standard normal data
        x = jr.normal(random_key, (num_data_points, dimension))
        length_scale = median_heuristic(x)
        array = SquaredExponentialKernel(length_scale).compute(x, x)

        # Compute "exact" inverse
        exact_inverter = LeastSquareApproximator(random_key, rcond=None)
        expected_inverse = exact_inverter.approximate(
            array, regularisation_parameter, identity
        )

        return _Problem(
            random_key,
            array,
            regularisation_parameter,
            identity,
            expected_inverse,
        )

    @pytest.mark.parametrize(
        "array, identity, rcond, context",
        [
            (jnp.eye(2), jnp.eye(2), -10, pytest.raises(ValueError)),
            (jnp.eye(2), jnp.eye(3), None, pytest.raises(ValueError)),
            (jnp.eye(2), jnp.eye(2), 1e-6, does_not_raise()),
            (jnp.eye(2), jnp.eye(2), -1, does_not_raise()),
        ],
        ids=[
            "rcond_negative_not_negative_one",
            "unequal_array_dimensions",
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
            approximator = RandomisedEigendecompositionApproximator(
                random_key=jr.key(0),
                oversampling_parameter=1,
                power_iterations=1,
                rcond=rcond,
            )
            approximator.approximate(
                array=array,
                regularisation_parameter=1e-6,
                identity=identity,
            )

    def test_randomised_eigendecomposition_accuracy(
        self, problem: _Problem, approximator: RandomisedEigendecompositionApproximator
    ) -> None:
        """Test that the `randomised_eigendecomposition` method is accurate."""
        # Unpack problem data
        _, array, _, _, _ = problem

        eigenvalues, eigenvectors = approximator.randomised_eigendecomposition(
            array=array
        )
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
    def test_invalid_inputs(
        self,
        array: Array,
        oversampling_parameter: int,
        power_iterations: int,
    ) -> None:
        """Test that invalid inputs raise errors correctly."""
        with pytest.raises(ValueError):
            approximator = RandomisedEigendecompositionApproximator(
                random_key=jr.key(self.random_seed),
                oversampling_parameter=oversampling_parameter,
                power_iterations=power_iterations,
                rcond=None,
            )
            approximator.randomised_eigendecomposition(array=array)


class TestNystromApproximator(
    InverseApproximationTest[NystromApproximator],
):
    """Test NystromApproximator."""

    random_seed = 2_024

    @override
    @pytest.fixture(scope="class")
    def approximator(self) -> NystromApproximator:
        """Abstract pytest fixture returns an initialised inverse approximator."""
        return NystromApproximator(
            random_key=jr.key(self.random_seed),
            oversampling_parameter=100,
            power_iterations=1,
            rcond=None,
        )

    @override
    @pytest.fixture(scope="class")
    def problem(self) -> _Problem:
        """Define data shared across tests."""
        random_key = jr.key(2_024)
        num_data_points = 1000
        dimension = 2
        identity = jnp.eye(num_data_points)
        regularisation_parameter = 1e1

        # Compute kernel matrix from standard normal data
        x = jr.normal(random_key, (num_data_points, dimension))
        length_scale = median_heuristic(x)
        array = SquaredExponentialKernel(length_scale).compute(x, x)

        # Compute "exact" inverse
        exact_inverter = LeastSquareApproximator(random_key, rcond=None)
        expected_inverse = exact_inverter.approximate(
            array, regularisation_parameter, identity
        )

        return _Problem(
            random_key,
            array,
            regularisation_parameter,
            identity,
            expected_inverse,
        )

    @pytest.mark.parametrize(
        "array, identity, rcond, context",
        [
            (jnp.eye(2), jnp.eye(2), -10, pytest.raises(ValueError)),
            (jnp.eye(2), jnp.eye(3), None, pytest.raises(ValueError)),
            (jnp.eye(2), jnp.eye(2), 1e-6, does_not_raise()),
            (jnp.eye(2), jnp.eye(2), -1, does_not_raise()),
        ],
        ids=[
            "rcond_negative_not_negative_one",
            "unequal_array_dimensions",
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
        """Test `NystromApproximator` handles invalid inputs."""
        with context:
            approximator = NystromApproximator(
                random_key=jr.key(0),
                oversampling_parameter=1,
                power_iterations=1,
                rcond=rcond,
            )
            approximator.approximate(
                array=array,
                regularisation_parameter=1e-6,
                identity=identity,
            )

    def test_nystrom_eigendecomposition_accuracy(
        self, problem: _Problem, approximator: NystromApproximator
    ) -> None:
        """Test that the `nystrom_eigendecomposition` method is accurate."""
        # Unpack problem data
        _, array, _, _, _ = problem

        eigenvalues, eigenvectors = approximator.nystrom_eigendecomposition(array=array)
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
    def test_invalid_inputs(
        self,
        array: Array,
        oversampling_parameter: int,
        power_iterations: int,
    ) -> None:
        """Test that invalid inputs raise errors correctly."""
        with pytest.raises(ValueError):
            approximator = NystromApproximator(
                random_key=jr.key(self.random_seed),
                oversampling_parameter=oversampling_parameter,
                power_iterations=power_iterations,
                rcond=None,
            )
            approximator.nystrom_eigendecomposition(array=array)
