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

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import coreax.approximation
import coreax.kernel
import coreax.util


@pytest.mark.parametrize(
    "approximator",
    [
        coreax.approximation.RandomApproximator,
        coreax.approximation.ANNchorApproximator,
        coreax.approximation.NystromApproximator,
    ],
)
class TestApproximations:
    """
    Tests related to approximation.py classes & functions.
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
        self, problem, approximator: type[coreax.approximation.KernelMeanApproximator]
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
        if issubclass(approximator, coreax.approximation.NystromApproximator):
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
        if issubclass(approximator, coreax.approximation.NystromApproximator):
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
            if isinstance(test_approximator, coreax.approximation.ANNchorApproximator):
                with pytest.raises(
                    ValueError, match="num_kernel_points must be positive and non-zero"
                ):
                    test_approximator.approximate(data)
            else:
                np.testing.assert_array_equal(
                    test_approximator.approximate(data),
                    jnp.zeros_like(true_distances),
                )
        elif isinstance(test_approximator, coreax.approximation.ANNchorApproximator):
            approximator_exact_num_data = coreax.approximation.ANNchorApproximator(
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
        if issubclass(approximator, coreax.approximation.NystromApproximator):
            pytest.skip("Incompatible with coreax.approximation.NystromApproximator")
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
        if issubclass(approximator, coreax.approximation.NystromApproximator):
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
