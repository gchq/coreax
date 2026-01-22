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

from typing import NamedTuple

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jaxtyping import Array

from coreax.approximation import (
    ANNchorApproximateKernel,
    MonteCarloApproximateKernel,
    NystromApproximateKernel,
    RandomRegressionKernel,
)
from coreax.kernels import ScalarValuedKernel, SquaredExponentialKernel
from coreax.util import InvalidKernel, KeyArrayLike

_RandomRegressionKernel = type[RandomRegressionKernel]


class _Problem(NamedTuple):
    random_key: KeyArrayLike
    data: Array
    kernel: ScalarValuedKernel
    num_kernel_points: int
    num_train_points: int
    true_distances: Array


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
    def problem(self) -> _Problem:
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
        true_distances = jnp.array(
            [
                0.5855723855138795,
                0.5737865795122914,
                0.4981814349432025,
                0.3670700196710188,
            ]
        )
        return _Problem(
            random_key,
            data,
            kernel,
            num_kernel_points,
            num_train_points,
            true_distances,
        )

    def test_approximation_accuracy(
        self, problem: _Problem, approximator: _RandomRegressionKernel
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
        problem: _Problem,
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
        problem: _Problem,
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
        self, problem: _Problem, approximator: type[RandomRegressionKernel]
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
            + f"'{ScalarValuedKernel.__module__}.{ScalarValuedKernel.__qualname__}'"
        )
        with pytest.raises(TypeError, match=expected_msg):
            approximator(InvalidKernel(0), random_key, **kwargs)  # pyright: ignore
