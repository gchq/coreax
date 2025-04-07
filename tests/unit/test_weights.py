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
Tests for weighting approaches.

The tests within this file verify that various weighting approaches and optimisers
written produce the expected results on simple examples.
"""

import cmath
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Generic, NamedTuple, TypeVar

import jax.numpy as jnp
import jax.random as jr
import pytest
from typing_extensions import override

from coreax.coreset import AbstractCoreset, Coresubset
from coreax.data import Data, SupervisedData
from coreax.kernels import SquaredExponentialKernel
from coreax.metrics import MMD, Metric
from coreax.util import KeyArrayLike
from coreax.weights import (
    INVALID_KERNEL_DATA_COMBINATION,
    MMDWeightsOptimiser,
    SBQWeightsOptimiser,
    WeightsOptimiser,
    _prepare_kernel_system,  # noqa: PLC2701
    solve_qp,
)

_Data = TypeVar("_Data", bound=Data)


class _Problem(NamedTuple):
    coreset: AbstractCoreset
    optimiser: WeightsOptimiser
    target_metric: Metric | None


class BaseWeightsOptimiserTest(ABC, Generic[_Data]):
    """Test the common behaviour of `WeightsOptimiser`s."""

    random_key: KeyArrayLike = jr.key(2_024)
    data_shape: tuple = (100, 2)
    coreset_size: int = data_shape[0] // 10

    @pytest.fixture
    @abstractmethod
    def problem(self) -> _Problem:
        """Abstract pytest fixture which returns a weight optimisation problem."""

    @pytest.mark.parametrize(
        "epsilon",
        (0.0, -0.1, 2),
        ids=["zero_epsilon", "negative_epsilon", "large_epsilon"],
    )
    def test_solver_with_unexpected_epsilon(
        self,
        problem: _Problem,
        epsilon: float,
    ) -> None:
        """
        Test if `solve` method of `WeightsOptimiser` handles unexpected `epsilon`.

        The `solve` method should not throw any errors, even with odd choices of
        `epsilon`
        """
        coreset, optimiser, _ = problem
        optimiser.solve(coreset.pre_coreset_data, coreset.points, epsilon=epsilon)

    def test_solver_reduces_target_metric(
        self,
        jit_variant: Callable[[Callable], Callable],
        problem: _Problem,
    ) -> None:
        """
        Test if :meth:`~coreax.coreset.Coreset.solve_weights` reduces `target_metric`.
        """
        coreset, optimiser, target_metric = problem

        # Compute the value of the target metric before we weight the coreset, solve for
        # optimal weights, then compute the metric again, asserting that it has reduced
        if target_metric is not None:
            metric_before_weighting = coreset.compute_metric(target_metric)
            weighted_coreset = jit_variant(coreset.solve_weights)(
                optimiser, epsilon=1e-4
            )
            metric_after_weighting = weighted_coreset.compute_metric(target_metric)

            assert metric_after_weighting < metric_before_weighting


class TestSBQWeightsOptimiser(BaseWeightsOptimiserTest[_Data]):
    """
    Tests related to :meth:`~coreax.weights.SBQWeightsOptimiser`.
    """

    @override
    @pytest.fixture(scope="class")
    def problem(self) -> _Problem:
        # SBQ targets the MMD
        kernel = SquaredExponentialKernel()
        optimiser = SBQWeightsOptimiser(kernel)
        target_metric = MMD(kernel)

        pre_coreset_data = Data(jr.normal(self.random_key, self.data_shape))
        coreset_indices = Data(
            jr.choice(self.random_key, self.data_shape[0], (self.coreset_size,))
        )
        coreset = Coresubset(indices=coreset_indices, pre_coreset_data=pre_coreset_data)
        return _Problem(coreset, optimiser, target_metric)

    def test_analytic_case(self) -> None:
        r"""
        Test the calculation of weights via sequential Bayesian quadrature.

        For the simple dataset of 3 points in 2D :math:`X`, with coreset :math:`X_c`,
        given by:

        .. math::

            X = [[0,0], [1,1], [2,2]]

            X_c = [[0,0], [1,1]]

        the weights, calculated by sequential Bayesian quadrature, are the solution to
        the equation :math:`w = z^T K^{-1}`. Here, :math:`z` is the row-mean of the
        kernel matrix :math:`k(X_c, X)`, i.e., the mean in the :math:`X` direction. The
        matrix :math:`K = k(X_c, X_c)`.

        Choosing the SquaredExponentialKernel kernel,
        :math:`k(x,y) = \exp (-||x-y||^2/2\text{length_scale}^2)`,
        setting ``length_scale`` to 1.0, we have:

        .. math::

            z^T = [\frac{1 + e^{-1} + e^{-4}}{3}, \frac{1 + 2e^{-1}}{3}]

            K = [1, e^{-1}; e^{-1}, 1]

        Therefore

        .. math::

            K^{-1} = \frac{1}{1 - e^{-2}}[1, -e^{-1}; -e^{-1}, 1]

        and it follows that

        .. math::

            w = [1 - 2e^{-2} + e^{-4}, 1 + e^{-1} - e^{-2} - e^{-5}]/3(1 - e^{-2}).
        """
        # Setup data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])

        expected_output = jnp.asarray(
            [
                (1 - 2 * jnp.exp(-2) + jnp.exp(-4)) / (3 * (1 - jnp.exp(-2))),
                (1 + jnp.exp(-1) - jnp.exp(-2) - jnp.exp(-5)) / (3 * (1 - jnp.exp(-2))),
            ]
        )

        optimiser = SBQWeightsOptimiser(kernel=SquaredExponentialKernel())

        # Solve for the weights
        output = optimiser.solve(x, y)

        assert output == pytest.approx(expected_output, abs=1e-3)


class TestMMDWeightsOptimiser(BaseWeightsOptimiserTest[_Data]):
    """
    Tests related to :meth:`~coreax.weights.MMDWeightsOptimiser`.
    """

    @override
    @pytest.fixture(scope="class")
    def problem(self) -> _Problem:
        kernel = SquaredExponentialKernel()
        optimiser = MMDWeightsOptimiser(kernel)
        target_metric = MMD(kernel)

        pre_coreset_data = Data(jr.normal(self.random_key, self.data_shape))
        coreset_indices = Data(
            jr.choice(self.random_key, self.data_shape[0], (self.coreset_size,))
        )
        coreset = Coresubset(indices=coreset_indices, pre_coreset_data=pre_coreset_data)

        return _Problem(coreset, optimiser, target_metric)

    def test_analytic_case(self) -> None:
        r"""
        Test calculation of weights via the simplex method for quadratic programming.

        :meth:`~MMDWeightsOptimiser.solve` solves the equation:

        .. math::

            0.5 \mathbf{w}^{\mathrm{T}} \mathbf{K} \mathbf{w}
            + \mathbf{z}^{\mathrm{T}} \mathbf{w} = 0

        subject to

        .. math::

            \mathbf{Aw} = \mathbf{1}, \qquad \mathbf{Gw} \le 0.

        Here, :math:`z` is the row-mean of the kernel matrix :math:`k(X_c, X)`, i.e.,
        the mean in the :math:`X` direction. The matrix :math:`K = k(X_c, X_c)`.

        The constraints (see solve_qp() method in coreax/util.py), are imposed with
        :math:`\mathbf{A}=1` and :math:`\mathbf{G}=-I`, ensuring the weights sum to 1
        and are non-negative, respectively.

        For the simple dataset of 3 points in 2D :math:`X`, with coreset :math:`X_c`,
        given by:

        .. math::

            X = [[0,0], [1,1], [2,2]]

            X_c = [[0,0], [1,1]]

        and with the SquaredExponentialKernel kernel,
        :math:`k(x,y) = \exp (-||x-y||^2/2\text{length_scale}^2)`,
        setting ``length_scale`` to 1.0, we have:

        .. math::

            z^T = [\frac{1 + e^{-1} + e^{-4}}{3}, \frac{1 + 2e^{-1}}{3}]

            K = [1, e^{-1}; e^{-1}, 1]

        It follows that

        .. math::

            w_2 = (-1-2e^{3}+3e^{4}+\sqrt{1+4e^{3}-6e^4+28e^{6}-6e^{7}-21e^{8}})
            /(6(e^4 - e^3))

            w_1 = 1 - w_2
        """
        # Setup data
        x = jnp.array([[0, 0], [1, 1], [2, 2]])
        y = jnp.array([[0, 0], [1, 1]])

        w2 = (
            -1
            - 2 * jnp.exp(3)
            + 3 * jnp.exp(4)
            + cmath.sqrt(
                1
                + 4 * jnp.exp(3)
                - 6 * jnp.exp(4)
                + 28 * jnp.exp(6)
                - 6 * jnp.exp(7)
                - 21 * jnp.exp(8)
            )
        ) / (6 * (jnp.exp(4) - jnp.exp(3)))
        w2 = jnp.real(w2)
        w1 = 1 - w2
        expected_output = jnp.asarray([w1, w2])

        optimiser = MMDWeightsOptimiser(kernel=SquaredExponentialKernel())

        # Solve for the weights
        output = optimiser.solve(x, y)

        assert output == pytest.approx(expected_output, abs=1e-3)


class TestHelperFunctions:
    """
    Tests for the helper functions `solve_qp` and `_prepare_kernel_system`.
    """

    def test_solve_qp_invalid_kernel_mm(self) -> None:
        """
        Test how `solve_qp` handles invalid inputs of kernel_mm.

        The output of `solve_qp` is indirectly tested when testing the various weight
        optimisers that are used in this codebase. This test just ensures sensible
        behaviour occurs when unexpected inputs are passed to the function.
        """
        # Attempt to solve a QP with an input that cannot be converted to a JAX array -
        # this should error as no sensible result can be found in such a case.
        with pytest.raises(TypeError, match="not a valid JAX array type"):
            solve_qp(
                kernel_mm="invalid_kernel_mm",  # pyright: ignore
                gramian_row_mean=jnp.array([1, 2, 3]),
            )

    def test_solve_qp_invalid_gramian_row_mean(self) -> None:
        """
        Test how `solve_qp` handles invalid inputs of gramian_row_mean.

        The output of `solve_qp` is indirectly tested when testing the various weight
        optimisers that are used in this codebase. This test just ensures sensible
        behaviour occurs when unexpected inputs are passed to the function.
        """
        # Attempt to solve a QP with an input that cannot be converted to a JAX array -
        # this should error as no sensible result can be found in such a case.
        with pytest.raises(TypeError, match="not a valid JAX array type"):
            solve_qp(
                kernel_mm=jnp.array([1, 2, 3]),
                gramian_row_mean="invalid_gramian_row_mean",  # pyright: ignore
            )

    def test_prepare_kernel_system_invalid_kernel_data_combination(self) -> None:
        """
        Test `_prepare_kernel_system` handling of bad combinations of kernel and data.

        The output of `_prepare_kernel_system` is indirectly tested when testing the
        various weight optimisers that are used in this codebase. This test just ensures
        sensible behaviour occurs when unexpected inputs are passed to the function.
        """
        x = jnp.array([1])
        supervised_data = SupervisedData(x, x)
        kernel = SquaredExponentialKernel()

        with pytest.raises(ValueError, match=INVALID_KERNEL_DATA_COMBINATION):
            _prepare_kernel_system(
                kernel=kernel, dataset=supervised_data, coreset=supervised_data
            )
