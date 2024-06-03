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
from typing import Generic, NamedTuple, TypeVar, Union

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike
from typing_extensions import override

from coreax.approximation import (
    ANNchorApproximateKernel,
    MonteCarloApproximateKernel,
    NystromApproximateKernel,
    RandomisedEigendecompositionApproximator,
    RandomRegressionKernel,
    RegularisedInverseApproximator,
    randomised_eigendecomposition,
)
from coreax.kernel import Kernel, SquaredExponentialKernel
from coreax.util import InvalidKernel, KeyArrayLike, invert_regularised_array

_RandomRegressionKernel = type[RandomRegressionKernel]


class _MeanProblem(NamedTuple):
    random_key: KeyArrayLike
    data: ArrayLike
    kernel: Kernel
    num_kernel_points: int
    num_train_points: int
    true_distances: ArrayLike


@pytest.mark.parametrize(
    "approximator",
    [
        MonteCarloApproximateKernel,
        ANNchorApproximateKernel,
        NystromApproximateKernel,
    ],
)
class TestRandomRegressionApproximations:
    """
    Tests related to ``RandomRegressionKernels`` in approximation.py .
    """

    @pytest.fixture
    def problem(self) -> _MeanProblem:
        r"""
        Define data shared across tests.

        We consider the data:

        .. math::

            x = [ [0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [-1.0, 0.0] ]

        and a SquaredExponentialKernel which is defined as
        :math:`k(x,y) = \text{output_scale}\exp(-||x-y||^2/2 * \text{length_scale}^2)`.
        For simplicity, we set ``length_scale`` to :math:`1.0/np.sqrt(2)`
        and ``output_scale`` to 1.0.

        The tests here ensure that approximations to the kernel's Gramian row-mean are
        valid. For a single row (data record), kernel's Gramian row-mean is computed by
        applying the kernel to this data record and all other data records. We then sum
        the results and divide by the number of data records. The first
        data-record ``[0, 0]`` in the data considered here therefore gives a result of:

        .. math::

              (1/4) * (
              exp(-((0.0 - 0.0)^2 + (0.0 - 0.0)^2)) +
              exp(-((0.0 - 0.5)^2 + (0.0 - 0.5)^2)) +
              exp(-((0.0 - 1.0)^2 + (0.0 - 0.0)^2)) +
              exp(-((0.0 - -1.0)^2 + (0.0 - 0.0)^2))
              )

        which evaluates to 0.5855723855138795.

        We can repeat the above but considering each data-point in ``x`` in turn and
        attain a set of true distances to use as the ground truth in the tests.
        """
        random_key = jr.key(10)

        # Setup a 'small' toy example that can be computed by hand
        data = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [-1.0, 0.0]])
        num_kernel_points = 3
        num_train_points = 3

        # Define a kernel object
        kernel = SquaredExponentialKernel(
            length_scale=1.0 / np.sqrt(2),
            output_scale=1.0,
        )

        # We can repeat the above, but changing the point with which we are comparing
        # to get:
        true_distances = np.array(
            [
                0.5855723855138795,
                0.5737865795122914,
                0.4981814349432025,
                0.3670700196710188,
            ]
        )
        return _MeanProblem(
            random_key,
            data,
            kernel,
            num_kernel_points,
            num_train_points,
            true_distances,
        )

    def test_approximation_accuracy(
        self, problem: _MeanProblem, approximator: _RandomRegressionKernel
    ) -> None:
        """
        Verify approximator performance on toy problem.

        This test verifies that, when the entire dataset is used to train the
        approximation, the result is very close to the true value. We further check if a
        subset of the data is used for training, that the approximation is still close
        to the true value, but less so than when using a larger training set.
        """
        # Define the approximator - full dataset used to fit the approximation
        random_key, data, kernel = problem.random_key, problem.data, problem.kernel
        true_distances = problem.true_distances
        full_kwargs = {
            "num_kernel_points": jnp.shape(data)[0],
            "num_train_points": jnp.shape(data)[0],
        }
        partial_kwargs = {
            "num_kernel_points": problem.num_kernel_points,
            "num_train_points": problem.num_train_points,
        }
        approximator_full = approximator(kernel, random_key, **full_kwargs)
        approximator_partial = approximator(kernel, random_key, **partial_kwargs)

        # Approximate the kernel row-mean using the full training set (so the
        # approximation should be very close to the true) and only part of the data for
        # training (so the error should grow)
        approximate_kernel_mean_full = approximator_full.gramian_row_mean(data)
        approximate_kernel_mean_partial = approximator_partial.gramian_row_mean(data)

        # Check the approximation is close to the true value
        np.testing.assert_array_almost_equal(
            approximate_kernel_mean_full, true_distances, decimal=0
        )
        np.testing.assert_array_almost_equal(
            approximate_kernel_mean_partial, true_distances, decimal=0
        )

        # Compute the approximation error and check if is better if we use more data
        full_delta = true_distances - approximate_kernel_mean_full
        approx_error_full = np.sum(np.square(full_delta))
        partial_delta = true_distances - approximate_kernel_mean_partial
        approx_error_partial = np.sum(np.square(partial_delta))
        assert approx_error_full <= approx_error_partial

    @pytest.mark.parametrize("num_kernel_points_multiplier", [-1, 0, 100])
    def test_degenerate_num_kernel_points(
        self,
        problem: _MeanProblem,
        approximator: _RandomRegressionKernel,
        num_kernel_points_multiplier: int,
    ) -> None:
        """
        Test approximators correctly handle degenerate cases of num_kernel_points.
        """
        random_key, data, kernel = problem.random_key, problem.data, problem.kernel
        true_distances = problem.true_distances
        num_kernel_points = problem.num_kernel_points
        num_train_points = problem.num_train_points
        kwargs = {
            "num_kernel_points": num_kernel_points * num_kernel_points_multiplier,
            "num_train_points": num_train_points,
        }
        if num_kernel_points_multiplier <= 0:
            expected_msg = "'num_kernel_points' must be a positive integer"
            with pytest.raises(ValueError, match=expected_msg):
                approximator(kernel, random_key, **kwargs)

        elif issubclass(approximator, ANNchorApproximateKernel):
            test_approximator = approximator(kernel, random_key, **kwargs)
            approximator_exact_num_data = ANNchorApproximateKernel(
                kernel,
                random_key,
                num_kernel_points=jnp.shape(data)[0],
                num_train_points=jnp.shape(data)[0],
            )
            result_exactly_num_data = approximator_exact_num_data.gramian_row_mean(data)
            result_more_than_num_data = test_approximator.gramian_row_mean(data)
            # Check the output is very close if we use all the data provided, or ask for
            # more than the number of points we have
            exact_delta = true_distances - result_exactly_num_data
            approx_error_exact_num_data = np.sum(np.square(exact_delta))
            more_than_delta = true_distances - result_more_than_num_data
            approx_error_more_than_num_data = np.sum(np.square(more_than_delta))
            assert approx_error_exact_num_data == pytest.approx(0, abs=1e-2)
            assert approx_error_more_than_num_data == pytest.approx(0, abs=1e-2)
        else:
            expected_msg = (
                "'num_kernel_points' must be no larger than the number of points in the"
                + " provided data"
            )
            with pytest.raises(ValueError, match=expected_msg):
                test_approximator = approximator(kernel, random_key, **kwargs)
                test_approximator.gramian_row_mean(data)

    @pytest.mark.parametrize("num_train_points_multiplier", [-1, 0, 100])
    def test_degenerate_num_train_points(
        self,
        problem: _MeanProblem,
        approximator: type[RandomRegressionKernel],
        num_train_points_multiplier: int,
    ) -> None:
        """
        Test approximators correctly handle degenerate cases of num_train_points.
        """
        random_key, data, kernel = problem.random_key, problem.data, problem.kernel
        kwargs = {
            "num_kernel_points": problem.num_kernel_points,
            "num_train_points": problem.num_train_points * num_train_points_multiplier,
        }

        if num_train_points_multiplier <= 0:
            expected_msg = "'num_train_points' must be a positive integer"
            with pytest.raises(ValueError, match=expected_msg):
                approximator(kernel, random_key, **kwargs)
        else:
            expected_msg = (
                "'num_train_points' must be no larger than the number of points in the"
                + " provided data"
            )
            with pytest.raises(ValueError, match=expected_msg):
                test_approximator = approximator(kernel, random_key, **kwargs)
                test_approximator.gramian_row_mean(data)

    def test_invalid_kernel(
        self, problem: _MeanProblem, approximator: type[RandomRegressionKernel]
    ) -> None:
        """
        Test approximators correctly handle invalid kernels.
        """
        random_key = problem.random_key
        kwargs = {
            "num_kernel_points": problem.num_kernel_points,
            "num_train_points": problem.num_train_points,
        }
        expected_msg = (
            "'base_kernel' must be an instance of "
            + f"'{Kernel.__module__}.{Kernel.__qualname__}'"
        )
        with pytest.raises(ValueError, match=expected_msg):
            approximator(InvalidKernel(0), random_key, **kwargs)


_RegularisedInverseApproximator = TypeVar(
    "_RegularisedInverseApproximator", bound=RegularisedInverseApproximator
)


class _InversionProblem(NamedTuple):
    random_key: KeyArrayLike
    kernel_gramian: Array
    regularisation_parameter: float
    identity: Array
    expected_inv: Array


class InverseApproximationTest(ABC, Generic[_RegularisedInverseApproximator]):
    """Tests related to kernel inverse approximations in approximation.py."""

    @abstractmethod
    def approximator(self) -> _RegularisedInverseApproximator:
        """Abstract pytest fixture which initialises an inverse approximator."""

    @abstractmethod
    def problem(self) -> _InversionProblem:
        """Abstract pytest fixture which returns a problem for inverse approximation."""

    def test_approximation_accuracy(
        self, problem: _InversionProblem, approximator: _RegularisedInverseApproximator
    ) -> None:
        """Verify approximator performance on toy problem."""
        # Extract problem settings
        _, kernel_gramian, regularisation_parameter, identity, expected_inverse = (
            problem
        )

        # Approximate the kernel inverse
        approximate_inverse = approximator.approximate(
            kernel_gramian=kernel_gramian,
            regularisation_parameter=regularisation_parameter,
            identity=identity,
        )

        # Check the approximation is close to the true value
        assert jnp.linalg.norm(expected_inverse - approximate_inverse) == pytest.approx(
            0.0, abs=1
        )


class TestRandomisedEigendecompositionApproximator(
    InverseApproximationTest[RandomisedEigendecompositionApproximator],
):
    """Test RandomisedEigendecompositionApproximator."""

    @override
    @pytest.fixture(scope="class")
    def approximator(self) -> RandomisedEigendecompositionApproximator:
        """Abstract pytest fixture returns an initialised inverse approximator."""
        random_seed = 2_024
        return RandomisedEigendecompositionApproximator(
            random_key=jr.key(random_seed),
            oversampling_parameter=100,
            power_iterations=1,
            rcond=None,
        )

    @override
    @pytest.fixture(scope="class")
    def problem(self):
        """Define data shared across tests."""
        random_key = jr.key(2_024)
        num_data_points = 1000
        dimension = 2
        identity = jnp.eye(num_data_points)
        regularisation_parameter = 1e-6

        # Compute kernel matrix from standard normal data
        x = jr.normal(random_key, (num_data_points, dimension))
        kernel_gramian = SquaredExponentialKernel().compute(x, x)

        # Compute "exact" inverse
        expected_inverse = invert_regularised_array(
            kernel_gramian, regularisation_parameter, identity, rcond=None
        )

        return _InversionProblem(
            random_key,
            kernel_gramian,
            regularisation_parameter,
            identity,
            expected_inverse,
        )

    @pytest.mark.parametrize(
        "kernel_gramian, identity, rcond",
        [
            (jnp.eye((2)), jnp.eye((2)), -10),
            (jnp.eye((2)), jnp.eye((3)), None),
        ],
        ids=[
            "rcond_negative_not_negative_one",
            "unequal_array_dimensions",
        ],
    )
    def test_approximator_invalid_inputs(
        self,
        kernel_gramian: Array,
        identity: Array,
        rcond: Union[int, float, None],
    ) -> None:
        """Test `RandomisedEigendecompositionApproximator` handles invalid inputs."""
        with pytest.raises(ValueError):
            approximator = RandomisedEigendecompositionApproximator(
                random_key=jr.key(0),
                oversampling_parameter=1,
                power_iterations=1,
                rcond=rcond,
            )

            approximator.approximate(
                kernel_gramian=kernel_gramian,
                regularisation_parameter=1e-6,
                identity=identity,
            )

    @pytest.mark.parametrize(
        "kernel_gramian, identity, rcond",
        [
            (jnp.eye((2)), jnp.eye((2)), 1e-6),
            (jnp.eye((2)), jnp.eye((2)), -1),
        ],
        ids=[
            "valid_rcond_not_negative_one_or_none",
            "valid_rcond_negative_one",
        ],
    )
    def test_approximator_valid_inputs(
        self,
        kernel_gramian: Array,
        identity: Array,
        rcond: Union[int, float, None],
    ) -> None:
        """
        Test `RandomisedEigendecompositionApproximator` handles valid `rcond`.

        Ensure that if we pass a valid `rcond` that is not None, no error is thrown.
        """
        approximator = RandomisedEigendecompositionApproximator(
            random_key=jr.key(0),
            oversampling_parameter=1,
            power_iterations=1,
            rcond=rcond,
        )

        approximator.approximate(
            kernel_gramian=kernel_gramian,
            regularisation_parameter=1e-6,
            identity=identity,
        )

    def test_randomised_eigendecomposition_accuracy(self, problem) -> None:
        """Test that the `randomised_eigendecomposition` is accurate."""
        # Unpack problem data
        random_key, kernel_gramian, _, _, _ = problem
        oversampling_parameter = 100
        power_iterations = 1

        eigenvalues, eigenvectors = randomised_eigendecomposition(
            random_key=random_key,
            array=kernel_gramian,
            oversampling_parameter=oversampling_parameter,
            power_iterations=power_iterations,
        )
        assert jnp.linalg.norm(
            kernel_gramian - (eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T)
        ) == pytest.approx(0.0, abs=1)

    @pytest.mark.parametrize(
        "kernel_gramian, oversampling_parameter, power_iterations",
        [
            (jnp.zeros((2, 2, 2)), 1, 1),
            (jnp.zeros((2, 3)), 1, 1),
            (jnp.eye((2)), 1.0, 1),
            (jnp.eye((2)), -1, 1),
            (jnp.eye((2)), 1, 1.0),
            (jnp.eye((2)), 1, -1),
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
        kernel_gramian: Array,
        oversampling_parameter: int,
        power_iterations: int,
    ) -> None:
        """Test that `randomised_eigendecomposition` handles invalid inputs."""
        with pytest.raises(ValueError):
            randomised_eigendecomposition(
                random_key=jr.key(0),
                array=kernel_gramian,
                oversampling_parameter=oversampling_parameter,
                power_iterations=power_iterations,
            )
