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
from jax.typing import ArrayLike
from typing_extensions import override

import coreax.kernel
import coreax.util
from coreax.approximation import (
    ANNchorApproximator,
    KernelInverseApproximator,
    KernelMeanApproximator,
    NystromApproximator,
    RandomApproximator,
    RandomisedEigendecompositionApproximator,
    randomised_eigendecomposition,
)

_KernelMeanApproximator = TypeVar(
    "_KernelMeanApproximator", bound=KernelMeanApproximator
)
_KernelInverseApproximator = TypeVar(
    "_KernelInverseApproximator", bound=KernelInverseApproximator
)


@pytest.mark.parametrize(
    "approximator",
    [RandomApproximator, ANNchorApproximator, NystromApproximator],
)
class TestKernelMeanApproximations:
    """
    Tests related to kernel mean approximations in approximation.py.
    """

    @pytest.fixture
    def problem(self):
        r"""
        Define data shared across tests.

        We consider the data:

        .. math::

            x = [ [0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [-1.0, 0.0] ]

        and a SquaredExponentialKernel which is defined as
        :math:`k(x,y) = \text{output_scale}\exp(-||x-y||^2/2 * \text{length_scale}^2)`.
        For simplicity, we set ``length_scale`` to :math:`1.0/np.sqrt(2)`
        and ``output_scale`` to 1.0.

        The tests here ensure that approximations to the kernel matrix row sum mean are
        valid. For a single row (data record), kernel matrix row sum mean is computed by
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
        kernel = coreax.kernel.SquaredExponentialKernel(
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
        return (
            random_key,
            data,
            kernel,
            num_kernel_points,
            num_train_points,
            true_distances,
        )

    def test_approximation_accuracy(
        self, problem, approximator: _KernelMeanApproximator
    ) -> None:
        """
        Verify approximator performance on toy problem.

        This test verifies that, when the entire dataset is used to train the
        approximation, the result is very close to the true value. We further check if a
        subset of the data is used for training, that the approximation is still close
        to the true value, but less so than when using a larger training set.
        """
        (
            random_key,
            data,
            kernel,
            num_kernel_points,
            num_train_points,
            true_distances,
        ) = problem
        # Define the approximator - full dataset used to fit the approximation
        if issubclass(approximator, NystromApproximator):
            full_kwargs = {"num_kernel_points": data.shape[0]}
            partial_kwargs = {"num_kernel_points": num_kernel_points}
        else:
            full_kwargs = {
                "num_kernel_points": data.shape[0],
                "num_train_points": data.shape[0],
            }
            partial_kwargs = {
                "num_kernel_points": num_kernel_points,
                "num_train_points": num_train_points,
            }
        approximator_full = approximator(random_key, kernel, **full_kwargs)
        approximator_partial = approximator(random_key, kernel, **partial_kwargs)

        # Approximate the kernel row mean using the full training set (so the
        # approximation should be very close to the true) and only part of the data for
        # training (so the error should grow)
        approximate_kernel_mean_full = approximator_full.approximate(data)
        approximate_kernel_mean_partial = approximator_partial.approximate(data)

        # Check the approximation is close to the true value
        np.testing.assert_array_almost_equal(
            approximate_kernel_mean_full, true_distances, decimal=0
        )
        np.testing.assert_array_almost_equal(
            approximate_kernel_mean_partial, true_distances, decimal=0
        )

        # Compute the approximation error and check if is better if we use more data
        approx_error_full = np.sum(
            np.square(true_distances - approximate_kernel_mean_full)
        )
        approx_error_partial = np.sum(
            np.square(true_distances - approximate_kernel_mean_partial)
        )
        assert approx_error_full <= approx_error_partial

    @pytest.mark.parametrize("num_kernel_points_multiplier", [-1, 0, 100])
    def test_degenerate_num_kernel_points(
        self, problem, approximator, num_kernel_points_multiplier
    ) -> None:
        """
        Test approximators correctly handle degenerate cases of num_kernel_points.
        """
        (
            random_key,
            data,
            kernel,
            num_kernel_points,
            num_train_points,
            true_distances,
        ) = problem
        if issubclass(approximator, NystromApproximator):
            kwargs = {
                "num_kernel_points": num_kernel_points * num_kernel_points_multiplier
            }
        else:
            kwargs = {
                "num_kernel_points": num_kernel_points * num_kernel_points_multiplier,
                "num_train_points": num_train_points,
            }
        test_approximator = approximator(random_key, kernel, **kwargs)

        if num_kernel_points_multiplier < 0:
            with pytest.raises(ValueError, match="num_kernel_points must be positive"):
                test_approximator.approximate(data)
        elif num_kernel_points_multiplier == 0:
            if isinstance(test_approximator, ANNchorApproximator):
                with pytest.raises(
                    ValueError, match="num_kernel_points must be positive and non-zero"
                ):
                    test_approximator.approximate(data)
            else:
                np.testing.assert_array_equal(
                    test_approximator.approximate(data),
                    jnp.zeros_like(true_distances),
                )
        elif isinstance(test_approximator, ANNchorApproximator):
            approximator_exact_num_data = ANNchorApproximator(
                random_key,
                kernel,
                num_kernel_points=data.shape[0],
                num_train_points=data.shape[0],
            )
            result_exactly_num_data = approximator_exact_num_data.approximate(data)
            result_more_than_num_data = test_approximator.approximate(data)

            # Check the output is very close if we use all the data provided, or ask for
            # more than the number of points we have
            approx_error_exact_num_data = np.sum(
                np.square(true_distances - result_exactly_num_data)
            )
            approx_error_more_than_num_data = np.sum(
                np.square(true_distances - result_more_than_num_data)
            )
            assert approx_error_exact_num_data == pytest.approx(0, abs=1e-2)
            assert approx_error_more_than_num_data == pytest.approx(0, abs=1e-2)
        else:
            with pytest.raises(
                ValueError,
                match="num_kernel_points must be no larger than the number of points"
                + " in the provided data",
            ):
                test_approximator.approximate(data)

    @pytest.mark.parametrize("num_train_points_multiplier", [-1, 0, 100])
    def test_degenerate_num_train_points(
        self, problem, approximator, num_train_points_multiplier
    ) -> None:
        """
        Test approximators correctly handle degenerate cases of num_train_points.
        """
        if issubclass(approximator, NystromApproximator):
            pytest.skip("Incompatible with NystromApproximator")
        (
            random_key,
            data,
            kernel,
            num_kernel_points,
            num_train_points,
            true_distances,
        ) = problem
        kwargs = {
            "num_kernel_points": num_kernel_points,
            "num_train_points": num_train_points * num_train_points_multiplier,
        }
        approximator = approximator(random_key, kernel, **kwargs)

        if num_train_points_multiplier < 0:
            with pytest.raises(ValueError, match="num_train_points must be positive"):
                approximator.approximate(data)
        elif num_train_points_multiplier == 0:
            np.testing.assert_array_equal(
                approximator.approximate(data), jnp.zeros_like(true_distances)
            )
        else:
            with pytest.raises(
                ValueError,
                match="num_train_points must be no larger than the number of points in"
                + " the provided data",
            ):
                approximator.approximate(data)

    def test_invalid_kernel(self, problem, approximator) -> None:
        """
        Test approximators correctly handle invalid kernels.
        """
        (
            random_key,
            data,
            _,
            num_kernel_points,
            num_train_points,
            _,
        ) = problem
        if issubclass(approximator, NystromApproximator):
            kwargs = {"num_kernel_points": num_kernel_points}
        else:
            kwargs = {
                "num_kernel_points": num_kernel_points,
                "num_train_points": num_train_points,
            }
        approximator = approximator(random_key, coreax.util.InvalidKernel(0), **kwargs)
        with pytest.raises(
            AttributeError,
            match="'InvalidKernel' object has no attribute 'compute'",
        ):
            approximator.approximate(data)


# Once we support only python 3.11+ this should be generic on _KernelInverseApproximator
class _Problem(NamedTuple):
    random_key: coreax.util.KeyArrayLike
    kernel_gramian: ArrayLike
    regularisation_parameter: float
    identity: ArrayLike
    expected_inverse: ArrayLike


class InverseApproximationTest(ABC, Generic[_KernelInverseApproximator]):
    """Tests related to kernel inverse approximations in approximation.py."""

    @abstractmethod
    def approximator(self) -> _KernelInverseApproximator:
        """Abstract pytest fixture which initialises an inverse approximator."""

    @abstractmethod
    def problem(self) -> _Problem:
        """Abstract pytest fixture which returns a problem for inverse approximation."""

    def test_approximation_accuracy(
        self, problem, approximator: _KernelInverseApproximator
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
    def approximator(self) -> _KernelInverseApproximator:
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
        r"""Define data shared across tests."""
        random_key = jr.key(2_024)
        num_data_points = 1000
        dimension = 2
        identity = jnp.eye(num_data_points)
        regularisation_parameter = 1e-6

        # Compute kernel matrix from standard normal data
        x = jr.normal(random_key, (num_data_points, dimension))
        kernel_gramian = coreax.kernel.SquaredExponentialKernel().compute(x, x)

        # Compute "exact" inverse
        expected_inverse = coreax.util.invert_regularised_array(
            kernel_gramian, regularisation_parameter, identity, rcond=None
        )

        return (
            random_key,
            kernel_gramian,
            regularisation_parameter,
            identity,
            expected_inverse,
        )

    @pytest.mark.parametrize(
        "kernel_gramian, identity, rcond",
        [(jnp.eye((2)), jnp.eye((2)), -10), (jnp.eye((2)), jnp.eye((3)), None)],
        ids=["rcond_negative_not_negative_one", "unequal_array_dimensions"],
    )
    def test_approximator_invalid_inputs(
        self,
        kernel_gramian: ArrayLike,
        identity: ArrayLike,
        rcond: Union[float, None],
    ) -> None:
        """Test that `randomised_eigendecomposition` handles invalid inputs."""
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
        ) == pytest.approx(0.0, abs=1e-3)

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
        kernel_gramian: ArrayLike,
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
