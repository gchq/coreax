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
Tests for computations of metrics for coresets.

Metrics evaluate the quality of a coreset by some measure. The tests within this file
verify that metric computations produce the expected results on simple examples.
"""

from typing import Literal, NamedTuple

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import jax.tree_util as jtu
import pytest
from jax import Array, jacfwd, vmap

from coreax.data import Data
from coreax.kernel import (
    Kernel,
    LaplacianKernel,
    PCIMQKernel,
    RationalQuadraticKernel,
    SquaredExponentialKernel,
)
from coreax.metrics import KSD, MMD
from coreax.score_matching import convert_stein_kernel


class _MetricProblem(NamedTuple):
    reference_data: Data
    comparison_data: Data


class TestMMD:
    """
    Tests related to the maximum mean discrepancy (MMD) class in metrics.py.
    """

    @pytest.fixture
    def problem(self) -> _MetricProblem:
        """Generate an example problem for testing MMD."""
        dimension = 10
        num_points = 30, 5
        keys = tuple(jr.split(jr.key(0), 2))

        def _generate_data(_num_points: int, _key: Array) -> Data:
            point_key, weight_key = jr.split(_key, 2)
            points = jr.uniform(point_key, (_num_points, dimension))
            weights = jr.uniform(weight_key, (_num_points,))
            return Data(points, weights)

        reference_data, comparison_data = jtu.tree_map(_generate_data, num_points, keys)
        return _MetricProblem(reference_data, comparison_data)

    @pytest.mark.parametrize(
        "kernel", [SquaredExponentialKernel(), LaplacianKernel(), PCIMQKernel()]
    )
    def test_mmd_compare_same_data(self, problem: _MetricProblem, kernel: Kernel):
        """Check MMD of a dataset with itself is approximately zero."""
        x = problem.reference_data
        metric = MMD(kernel)
        assert metric.compute(x, x) == pytest.approx(0.0)

    def test_mmd_analytically_known(self):
        r"""
        Test MMD computation against an analytically derived solution.

        For the dataset of 3 points in 2 dimensions, :math:`\mathcal{D}_1`, and second
        dataset :math:`\mathcal{D}_2`, given by:

        .. math::

            reference_data = [[0,0], [1,1], [2,2]]

            comparison_data = [[0,0], [1,1]]

        the RBF kernel, :math:`k(x,y) = \exp (-||x-y||^2/2\sigma^2)`, gives:

        .. math::

            k(\mathcal{D}_1,\mathcal{D}_1) = \exp(-\begin{bmatrix}0 & 2 & 8 \\ 2 & 0 & 2
            \\ 8 & 2 & 0
            \end{bmatrix}/2\sigma^2) = \begin{bmatrix}1 & e^{-1} & e^{-4} \\ e^{-1} &
            1 & e^{-1} \\ e^{-4} & e^{-1} & 1\end{bmatrix}.

            k(\mathcal{D}_2,\mathcal{D}_2) = \exp(-\begin{bmatrix}0 & 2 \\ 2 & 0
            \end{bmatrix}/2\sigma^2) = \begin{bmatrix}1 & e^{-1}\\ e^{-1} & 1
            \end{bmatrix}.

            k(\mathcal{D}_1,\mathcal{D}_2) =  \exp(-\begin{bmatrix}0 & 2 & 8
            \\ 2 & 0 & 2 \end{bmatrix}
            /2\sigma^2) = \begin{bmatrix}1 & e^{-1} \\  e^{-1} & 1 \\ e^{-4} & e^{-1}
            \end{bmatrix}.

        Then

        .. math::

            \text{MMD}^2(\mathcal{D}_1,\mathcal{D}_2) =
            \mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_1)) +
            \mathbb{E}(k(\mathcal{D}_2,\mathcal{D}_2)) -
            2\mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_2))

            = \frac{3+4e^{-1}+2e^{-4}}{9} + \frac{2 + 2e^{-1}}{2} -2 \times
            \frac{2 + 3e^{-1}+e^{-4}}{6}

            = \frac{3 - e^{-1} -2e^{-4}}{18}.
        """
        reference_points = jnp.array([[0, 0], [1, 1], [2, 2]])
        reference_data = Data(data=reference_points)
        comparison_points = jnp.array([[0, 0], [1, 1]])
        comparison_data = Data(data=comparison_points)
        expected_output = jnp.sqrt((3 - jnp.exp(-1) - 2 * jnp.exp(-4)) / 18)
        # Compute MMD using the metric object
        metric = MMD(SquaredExponentialKernel())
        output = metric.compute(reference_data, comparison_data)
        assert output == pytest.approx(expected_output)

    def test_mmd_analytically_known_weighted(self) -> None:
        r"""
        Test MMD computation against an analytically derived weighted solution.

        Weighted mmd is calculated if and only if comparison_weights are provided. When
        `comparison_weights` = :data:`None`, the MMD class computes the standard,
        non-weighted MMD.

        For the dataset of 3 points in 2 dimensions :math:`\mathcal{D}_1`, second
        dataset :math:`\mathcal{D}_2`, and weights for this second dataset :math:`w_2`,
        given by:

        .. math::

            reference_data = [[0,0], [1,1], [2,2]]

            w_1 = [1,1,1]

            comparison_data = [[0,0], [1,1]]

            w_2 = [1,0]

        the weighted maximum mean discrepancy is calculated via:

        .. math::

            \text{WMMD}^2(\mathcal{D}_1,\mathcal{D}_2) =
            \frac{1}{||w_1||**2}w_1^T \mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_1)) w_1
             + \frac{1}{||w_2||**2}w_2^T k(\mathcal{D}_2,\mathcal{D}_2) w_2
             - \frac{2}{||w_1||||w_2||} w_1
             \mathbb{E}_x(k(\mathcal{D}_1,\mathcal{D}_2)) w_2

            = \frac{3+4e^{-1}+2e^{-4}}{9} + 1 - 2 \times \frac{1 + e^{-1} + e^{-4}}{3}

            = \frac{2}{3} - \frac{2}{9}e^{-1} - \frac{4}{9}e^{-4}.
        """
        reference_points = jnp.array([[0, 0], [1, 1], [2, 2]])
        reference_weights = jnp.array([1, 1, 1])
        reference_data = Data(reference_points, reference_weights)
        comparison_points = jnp.array([[0, 0], [1, 1]])
        comparison_weights = jnp.array([1, 0])
        comparison_data = Data(comparison_points, comparison_weights)
        expected_output = jnp.sqrt(
            2 / 3 - (2 / 9) * jnp.exp(-1) - (4 / 9) * jnp.exp(-4)
        )
        # Compute the weighted MMD using the metric object
        metric = MMD(SquaredExponentialKernel())
        output = metric.compute(reference_data, comparison_data)
        assert output == pytest.approx(expected_output)

    @pytest.mark.parametrize("mode", ["unweighted", "weighted"])
    def test_mmd_random_data(
        self, problem: _MetricProblem, mode: Literal["unweighted", "weighted"]
    ):
        r"""
        Test MMD computed from randomly generated test data agrees with method result.

        - "unweighted" parameterization checks that if the 'reference_data' and the
            'comparison_data' have the default 'None' weights, that the computed MMD is
            given by the means of the unweighted kernel matrices.
        - "weighted" parameterization checks that for arbitrarily weighted data, the
            computed MMD is given by the weighted average of the kernel matrices.
        """
        kernel = SquaredExponentialKernel()
        x, y = problem
        # Compute each term in the MMD formula to obtain an expected MMD.
        kernel_nn = kernel.compute(x.data, x.data)
        kernel_mm = kernel.compute(y.data, y.data)
        kernel_nm = kernel.compute(x.data, y.data)
        if mode == "weighted":
            weights_nn = x.weights[..., None] * x.weights[None, ...]
            weights_mm = y.weights[..., None] * y.weights[None, ...]
            weights_nm = x.weights[..., None] * y.weights[None, ...]
            expected_mmd = jnp.sqrt(
                jnp.average(kernel_nn, weights=weights_nn)
                + jnp.average(kernel_mm, weights=weights_mm)
                - 2 * jnp.average(kernel_nm, weights=weights_nm)
            )
        elif mode == "unweighted":
            x, y = Data(x.data), Data(y.data)
            expected_mmd = jnp.sqrt(
                jnp.mean(kernel_nn) + jnp.mean(kernel_mm) - 2 * jnp.mean(kernel_nm)
            )
        else:
            raise ValueError("Invalid mode parameterization")
        # Compute the MMD using the metric object
        metric = MMD(kernel=kernel)
        output = metric.compute(x, y)
        assert output == pytest.approx(expected_mmd, abs=1e-6)


class TestKSD:
    """
    Tests related to the kernel Stein discrepancy (KSD) class in metrics.py.
    """

    @pytest.fixture
    def problem(self) -> _MetricProblem:
        """Generate an example problem for testing KSD."""
        dimension = 1
        num_points = 2500, 1000
        keys = tuple(jr.split(jr.key(0), 2))

        def _generate_data(_num_points: int, _key: Array) -> Data:
            point_key, weight_key = jr.split(_key, 2)
            points = jr.normal(point_key, (_num_points, dimension))
            weights = jr.uniform(weight_key, (_num_points,))
            return Data(points, weights)

        reference_data, comparison_data = jtu.tree_map(_generate_data, num_points, keys)
        return _MetricProblem(reference_data, comparison_data)

    @pytest.mark.parametrize(
        "kernel", [SquaredExponentialKernel(), RationalQuadraticKernel(), PCIMQKernel()]
    )
    def test_ksd_compare_same_data(self, problem: _MetricProblem, kernel: Kernel):
        """Check KSD of a dataset with itself is approximately zero."""
        x = problem.reference_data
        metric = KSD(kernel)
        assert metric.compute(
            x, x, laplace_correct=False, regularise=False
        ) == pytest.approx(0.0, abs=1e-1)

    @pytest.mark.parametrize(
        "mode", ["unweighted", "weighted", "laplace-corrected", "regularised"]
    )
    def test_ksd_random_data(
        self,
        problem: _MetricProblem,
        mode: Literal["unweighted", "weighted", "laplace-corrected", "regularised"],
    ):
        r"""
        Test KSD computed from randomly generated test data agrees with method result.

        - "unweighted" parameterization checks that if the 'reference_data' and the
            'comparison_data' have the default 'None' weights, that the computed KSD
              is
            given by the means of the unweighted kernel matrices.
        - "weighted" parameterization checks that for arbitrarily weighted data, the
            computed MMD is given by the weighted average of the kernel matrices.
        """
        x, y = problem

        base_kernel = SquaredExponentialKernel()
        kernel = convert_stein_kernel(x.data, base_kernel, None)
        metric = KSD(kernel=kernel, score_matching=None)

        # Compute each term in the KSD formula to obtain an expected KSD.
        kernel_mm = kernel.compute(y.data, y.data)
        if mode == "weighted":
            weights_mm = y.weights[..., None] * y.weights[None, ...]
            expected_ksd = jnp.average(kernel_mm, weights=weights_mm)
            output = metric.compute(x, y, laplace_correct=False, regularise=False)
        elif mode == "unweighted":
            expected_ksd = jnp.mean(kernel_mm)
            output = metric.compute(
                Data(x.data), Data(y.data), laplace_correct=False, regularise=False
            )
        elif mode == "laplace-corrected":
            # pylint: disable=duplicate-code
            @vmap
            def _laplace_positive(x_: Array) -> Array:
                r"""Evaluate Laplace positive operator  :math:`\Delta^+ \log p(x)`."""
                hessian = jacfwd(kernel.score_function)(x_)
                return jnp.clip(jnp.diag(hessian), min=0.0).sum()

            laplace_correction = _laplace_positive(y.data).sum() / len(y) ** 2
            # pylint: enable=duplicate-code

            expected_ksd = jnp.mean(kernel_mm) + (laplace_correction / len(y) ** 2)
            output = metric.compute(
                Data(x.data), Data(y.data), laplace_correct=True, regularise=False
            )
        elif mode == "regularised":
            kde = jsp.stats.gaussian_kde(x.data.T, bw_method=base_kernel.length_scale)
            entropic_regularisation = kde.logpdf(y.data.T).mean() / len(y)
            expected_ksd = jnp.mean(kernel_mm) - entropic_regularisation
            output = metric.compute(
                Data(x.data), Data(y.data), laplace_correct=False, regularise=True
            )
        else:
            raise ValueError("Invalid mode parameterization")
        # Compute the KSD using the metric object
        assert output == pytest.approx(expected_ksd, abs=1e-6, rel=1e-3)
