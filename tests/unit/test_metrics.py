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
import jax.tree_util as jtu
import pytest
from jax import Array

from coreax.data import Data, SupervisedData
from coreax.inverses import (
    LeastSquareApproximator,
    RandomisedEigendecompositionApproximator,
    RegularisedInverseApproximator,
)
from coreax.kernel import (
    Kernel,
    LaplacianKernel,
    LinearKernel,
    PCIMQKernel,
    SquaredExponentialKernel,
    TensorProductKernel,
)
from coreax.metrics import CMMD, JMMD, MMD


class _MetricProblem(NamedTuple):
    reference_data: Data
    comparison_data: Data


class _SupervisedMetricProblem(NamedTuple):
    reference_data: SupervisedData
    comparison_data: SupervisedData


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


class TestJMMD:
    """
    Tests related to the joint maximum mean discrepancy (JMMD) class in metrics.py.
    """

    @pytest.fixture
    def problem(self) -> _SupervisedMetricProblem:
        """Generate an example problem for testing JMMD."""
        feature_dimension = 10
        response_dimension = 2
        num_points = 30, 5
        keys = tuple(jr.split(jr.key(0), 2))

        def _generate_data(_num_points: int, _key: Array) -> Data:
            point_key, weight_key = jr.split(_key, 2)
            data = jr.uniform(point_key, (_num_points, feature_dimension))
            supervision = jr.uniform(point_key, (_num_points, response_dimension))
            weights = jr.uniform(weight_key, (_num_points,))
            return SupervisedData(data, supervision, weights)

        reference_data, comparison_data = jtu.tree_map(_generate_data, num_points, keys)
        return _SupervisedMetricProblem(reference_data, comparison_data)

    @pytest.mark.parametrize(
        "kernel", [SquaredExponentialKernel(), LaplacianKernel(), PCIMQKernel()]
    )
    def test_jmmd_compare_same_data(
        self, problem: _SupervisedMetricProblem, kernel: Kernel
    ):
        """Check MMD of a dataset with itself is approximately zero."""
        x = problem.reference_data
        metric = JMMD(kernel, kernel)
        assert metric.compute(x, x) == pytest.approx(0.0)

    def test_jmmd_analytically_known(self):
        r"""
        Test JMMD computation against an analytically derived solution.

        For the dataset of 3 pairs with feature and response dimension of 1,
        :math:`\mathcal{D}_1`, and second dataset :math:`\mathcal{D}_2`, given by:

        .. math::

            reference_features = [[0], [1], [2]]
            reference_responses = [[0], [1], [2]]

            comparison_features = [[0], [1]]
            comparison_responses = [[0], [1]]

        with RBF feature kernel and response kernel given by
        :math:`k(x,y) = \exp (-||x-y||^2/2\sigma^2)`, gives:

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

            \text{JMMD}^2(\mathcal{D}_1,\mathcal{D}_2) =
            \mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_1)) +
            \mathbb{E}(k(\mathcal{D}_2,\mathcal{D}_2)) -
            2\mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_2))

            = \frac{3+4e^{-1}+2e^{-4}}{9} + \frac{2 + 2e^{-1}}{2} -2 \times
            \frac{2 + 3e^{-1}+e^{-4}}{6}

            = \frac{3 - e^{-1} -2e^{-4}}{18}.
        """
        reference_points = jnp.array([[0], [1], [2]])
        reference_supervision = jnp.array([[0], [1], [2]])
        reference_data = SupervisedData(reference_points, reference_supervision)

        comparison_points = jnp.array([[0], [1]])
        comparison_supervision = jnp.array([[0], [1]])
        comparison_data = SupervisedData(comparison_points, comparison_supervision)

        expected_output = jnp.sqrt((3 - jnp.exp(-1) - 2 * jnp.exp(-4)) / 18)
        # Compute JMMD using the metric object
        metric = JMMD(SquaredExponentialKernel(), SquaredExponentialKernel())
        output = metric.compute(reference_data, comparison_data)
        assert output == pytest.approx(expected_output)

    def test_jmmd_analytically_known_weighted(self) -> None:
        r"""
        Test JMMD computation against an analytically derived weighted solution.

        For the dataset of 3 pairs with feature and response dimension of 1,
        :math:`\mathcal{D}_1`, second dataset :math:`\mathcal{D}_2`, and weights for
        this second dataset :math:`w_2`, given by:

        .. math::

            reference_features = [[0], [1], [2]]
            reference_responses = [[0], [1], [2]]

            w_1 = [1,1,1]

            comparison_features = [[0], [1]]
            comparison_responses = [[0], [1]]

            w_2 = [1,0]

        the weighted maximum mean discrepancy is calculated via:

        .. math::

            \text{WJMMD}^2(\mathcal{D}_1,\mathcal{D}_2) =
            \frac{1}{||w_1||**2}w_1^T \mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_1)) w_1
             + \frac{1}{||w_2||**2}w_2^T k(\mathcal{D}_2,\mathcal{D}_2) w_2
             - \frac{2}{||w_1||||w_2||} w_1
             \mathbb{E}_x(k(\mathcal{D}_1,\mathcal{D}_2)) w_2

            = \frac{3+4e^{-1}+2e^{-4}}{9} + 1 - 2 \times \frac{1 + e^{-1} + e^{-4}}{3}

            = \frac{2}{3} - \frac{2}{9}e^{-1} - \frac{4}{9}e^{-4}.
        """
        reference_points = jnp.array([[0], [1], [2]])
        reference_supervision = jnp.array([[0], [1], [2]])
        reference_weights = jnp.array([1, 1, 1])
        reference_data = SupervisedData(
            reference_points, reference_supervision, reference_weights
        )

        comparison_points = jnp.array([[0], [1]])
        comparison_supervision = jnp.array([[0], [1]])
        comparison_weights = jnp.array([1, 0])
        comparison_data = SupervisedData(
            comparison_points, comparison_supervision, comparison_weights
        )

        expected_output = jnp.sqrt(
            2 / 3 - (2 / 9) * jnp.exp(-1) - (4 / 9) * jnp.exp(-4)
        )
        # Compute the weighted MMD using the metric object
        metric = JMMD(SquaredExponentialKernel(), SquaredExponentialKernel())
        output = metric.compute(reference_data, comparison_data)
        assert output == pytest.approx(expected_output)

    @pytest.mark.parametrize("mode", ["unweighted", "weighted"])
    def test_jmmd_random_data(
        self, problem: _SupervisedMetricProblem, mode: Literal["unweighted", "weighted"]
    ):
        r"""
        Test JMMD computed from randomly generated test data agrees with method result.

        - "unweighted" parameterization checks that if the 'reference_data' and the
            'comparison_data' have the default 'None' weights, that the computed MMD is
            given by the means of the unweighted kernel matrices.
        - "weighted" parameterization checks that for arbitrarily weighted data, the
            computed MMD is given by the weighted average of the kernel matrices.
        """
        base_kernel = SquaredExponentialKernel()
        kernel = TensorProductKernel(base_kernel, base_kernel)
        x, y = problem
        # Compute each term in the MMD formula to obtain an expected MMD.
        x_no_weights = SupervisedData(x.data, x.supervision)
        y_no_weights = SupervisedData(y.data, y.supervision)
        kernel_nn = kernel.compute(x_no_weights, x_no_weights)
        kernel_mm = kernel.compute(y_no_weights, y_no_weights)
        kernel_nm = kernel.compute(x_no_weights, y_no_weights)
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
            x, y = (
                SupervisedData(x.data, x.supervision),
                SupervisedData(y.data, y.supervision),
            )
            expected_mmd = jnp.sqrt(
                jnp.mean(kernel_nn) + jnp.mean(kernel_mm) - 2 * jnp.mean(kernel_nm)
            )
        else:
            raise ValueError("Invalid mode parameterization")
        # Compute the MMD using the metric object
        metric = JMMD(base_kernel, base_kernel)
        output = metric.compute(x, y)
        assert output == pytest.approx(expected_mmd, abs=1e-6)


class TestCMMD:
    """
    Tests for the conditional maximum mean discrepancy (CMMD) class in metrics.py.
    """

    @pytest.fixture
    def problem(self) -> _SupervisedMetricProblem:
        """Generate an example problem for testing CMMD."""
        feature_dimension = 1
        response_dimension = 1
        num_points = 500, 500
        keys = tuple(jr.split(jr.key(0), 2))

        def _generate_data(_num_points: int, _key: Array) -> Data:
            data_key, supervision_key = jr.split(_key, 2)

            x_shape = (_num_points, feature_dimension)
            x = jr.normal(data_key, shape=x_shape)

            y_shape = (_num_points, response_dimension)
            y = jr.normal(supervision_key, shape=y_shape)
            return SupervisedData(x, y)

        reference_data, comparison_data = jtu.tree_map(_generate_data, num_points, keys)
        return _SupervisedMetricProblem(reference_data, comparison_data)

    @pytest.mark.parametrize(
        "kernel", [SquaredExponentialKernel(), LinearKernel(), PCIMQKernel()]
    )
    def test_cmmd_compare_same_data(
        self, problem: _SupervisedMetricProblem, kernel: Kernel
    ):
        """Check CMMD of a dataset with itself is approximately zero."""
        x = problem.reference_data
        metric = CMMD(
            feature_kernel=kernel, response_kernel=kernel, regularisation_parameter=1e-6
        )
        assert metric.compute(x, x) == pytest.approx(0.0)

    def test_cmmd_analytically_known(self):
        r"""
        Test CMMD computation with a small dataset of integers with no regularisation.

        For the first dataset of 3 pairs in 2 feature dimensions and 1 response
        dimension, :math:`\mathcal{D}_1` given by:
        .. math::

            x_1 = [[0,0], [1,1]]
            y_1 = [[0], [1]]

        and a second dataset of 2 pairs in 2 feature dimensions and 1 response
        dimension, :math:`\mathcal{D}_2` given by:

        .. math::

            x_2 = [[1,1], [3,3]]
            y_2 = [[1], [3]]

        the Gaussian (aka radial basis function) kernels with length-scale equal to one,
        :math:`k(a,b) = l(a,b) = \exp (-||a-b||^2/2)` gives:

        .. math::

            k(x_1_x_1) := K_{11} = \begin{bmatrix}1 & e^{-1}\\ e^{-1} & 1\end{bmatrix},

            K_{11}^{-1} = \frac{1}{1 - e^{-2}}\begin{bmatrix}1 & -e^{-1}\\ -e^{-1} & 1
            \end{bmatrix},

            K_{22} = \begin{bmatrix}1 & e^{-4}\\ e^{-4} & 1\end{bmatrix},

            K_{22}^{-1} = \frac{1}{1 - e^{-8}}\begin{bmatrix}1 & -e^{-4}\\ -e^{-4} & 1
            \end{bmatrix},

            K_{21} = \begin{bmatrix}e^{-1} & 1\\ e^{-9} & e^{-4}\end{bmatrix},

            L_{11} = \begin{bmatrix}1 & e^{-1/2}\\ e^{-1/2} &1\end{bmatrix},

            L_{22} = \begin{bmatrix}1 & e^{-2}\\ e^{-2} &1\end{bmatrix},

            L_{12} = \begin{bmatrix}e^{-1/2} & e^{-9/2}\\ 1 & e^{-2}\end{bmatrix}.

        Then

        .. math::

            \text{CMMD}(\mathcal{D}_1, \mathcal{D}_2) = \sqrt{\text{Tr}(T_1) +
            \text{Tr}(T_2) - 2\text{Tr}(T_3)}

        where

        .. math::

            T_1 := K_{11}^{-1}L_{11}K_{11}^{-1}K_{11} = K_{11}^{-1}L_{11}

            = \frac{1}{1 - e^{-2}}\begin{bmatrix}1 & -e^{-1}\\ -e^{-1} & 1\end{bmatrix}
            \begin{bmatrix}1 & e^{-1/2}\\ e^{-1/2} &1\end{bmatrix}

            = \frac{1}{1 - e^{-2}}\begin{bmatrix}1 - e^{-3/2} & e^{-1/2} - e^{-1}\\
            e^{-1/2} - e^{-1} & 1 - e^{-3/2}\end{bmatrix}

            \implies \text{Tr}(T_1) = \frac{2(1 - e^{-3/2})}{1 - e^{-2}}


            T_2 := K_{22}^{-1}L_{22}K_{22}^{-1}K_{22} = K_{22}^{-1}L_{22}

            =\frac{1}{1 - e^{-8}}\begin{bmatrix}1 & -e^{-4}\\ -e^{-4} & 1\end{bmatrix}
            \begin{bmatrix}1 & e^{-2}\\ e^{-2} &1\end{bmatrix}

            = \frac{1}{1 - e^{-8}}\begin{bmatrix}1 - e^{-6} & e^{-2} - e^{-4}\\ e^{-2}
            - e^{-4} & 1 - e^{-6}\end{bmatrix}

            \implies \text{Tr}(T_2) = \frac{2(1 - e^{-6})}{1 - e^{-8}}


            T_3 := K_{21}^{-1}K_{11}^{-1}L_{12}K_{22}^{-1}

            = \frac{1}{(1 - e^{-2})(1 - e^{-8})}
            \begin{bmatrix}e^{-1} & 1\\ e^{-9} & e^{-4}\end{bmatrix}
            \begin{bmatrix}1 & -e^{-1}\\ -e^{-1} & 1\end{bmatrix}
            \begin{bmatrix}e^{-1/2} & e^{-9/2}\\ 1 & e^{-2}\end{bmatrix}
            \begin{bmatrix}1 & -e^{-4}\\ -e^{-4} & 1\end{bmatrix}

            = \frac{1}{(1 - e^{-2})(1 - e^{-8})}
            \begin{bmatrix}0 & 1 - e^{-2}\\ e^{-9} - e^{-5} &
            e^{-4} - e^{-10}\end{bmatrix}
            \begin{bmatrix}e^{-1/2} - e^{-17/2} & 0\\ 1 -e^{-6} &
            e^{-2} - e^{-4}\end{bmatrix}

            = \frac{1}{(1 - e^{-2})(1 - e^{-8})}\begin{bmatrix}(1 - e^{-2})(1 - e^{-6})
            & (1 - e^{-2})(e^{-2} - e^{-4})\\  (e^{-9} - e^{-5}) (e^{-1/2} - e^{-17/2})
            +  (e^{-4} - e^{-10}) (1 - e^{-6})
            & (e^{-2} - e^{-4})(e^{-4} - e^{-10})\end{bmatrix}

            \implies \text{Tr}(T_3) = \frac{(1 - e^{-2})(1 - e^{-6})  + (e^{-2} -
            e^{-4})(e^{-4} - e^{-10})}{(1 - e^{-2})(1 - e^{-8})}

        therefore

        .. math::

            \text{CMMD}(\mathcal{D}_1, \mathcal{D}_2) &= \sqrt{\text{Tr}(T_1) +
            \text{Tr}(T_2) - 2\text{Tr}(T_3)}

            =  \sqrt{ \frac{2(1 - e^{-3/2})}{1 - e^{-2}} + \frac{2(1 - e^{-6})}
            {1 - e^{-8}} -2 \frac{(1 - e^{-2})(1 - e^{-6})  + (e^{-2} - e^{-4})
            (e^{-4} - e^{-10})}{(1 - e^{-2})(1 - e^{-8})} }
        """
        # Setup data
        x1 = jnp.array([[0, 0], [1, 1]])
        y1 = jnp.array([[0], [1]])
        reference_data = SupervisedData(x1, y1)

        x2 = jnp.array([[1, 1], [3, 3]])
        y2 = jnp.array([[1], [3]])
        comparison_data = SupervisedData(x2, y2)

        # Set expected CMMD
        t1 = (2 * (1 - jnp.exp(-3 / 2))) / (1 - jnp.exp(-2))
        t2 = (2 * (1 - jnp.exp(-6))) / (1 - jnp.exp(-8))
        t3 = (
            ((1 - jnp.exp(-2)) * (1 - jnp.exp(-6)))
            + ((jnp.exp(-2) - jnp.exp(-4)) * (jnp.exp(-4) - jnp.exp(-10)))
        ) / ((1 - jnp.exp(-2)) * (1 - jnp.exp(-8)))
        expected_output = jnp.sqrt(t1 + t2 - 2 * t3)

        # Define a metric object using RBF kernels
        metric = CMMD(
            feature_kernel=SquaredExponentialKernel(length_scale=1.0),
            response_kernel=SquaredExponentialKernel(length_scale=1.0),
            regularisation_parameter=0,
        )

        # Compute CMMD using the metric object
        output = metric.compute(reference_data, comparison_data)

        assert output == pytest.approx(expected_output)

    # pylint: disable=too-many-locals
    def test_cmmd_random_data(self, problem: _SupervisedMetricProblem):
        r"""
        Test CMMD computed from randomly generated test data agrees with method result.
        """
        kernel = SquaredExponentialKernel()
        regularisation_parameter = 1e-6
        reference_data, comparison_data = problem

        # Compute each term in the CMMD formula to obtain an expected CMMD.
        x1 = reference_data.data
        y1 = reference_data.supervision
        x2 = comparison_data.data
        y2 = comparison_data.supervision

        # Compute feature kernel gramians
        feature_gramian_1 = kernel.compute(x1, x1)
        feature_gramian_2 = kernel.compute(x2, x2)

        inverse_approximator = LeastSquareApproximator(jr.key(2_024))
        inverse_feature_gramian_1 = inverse_approximator.approximate(
            array=feature_gramian_1,
            regularisation_parameter=regularisation_parameter,
            identity=jnp.eye(feature_gramian_1.shape[0]),
        )
        inverse_feature_gramian_2 = inverse_approximator.approximate(
            array=feature_gramian_2,
            regularisation_parameter=regularisation_parameter,
            identity=jnp.eye(feature_gramian_2.shape[0]),
        )

        # Compute each term in the CMMD
        term_1 = (
            inverse_feature_gramian_1
            @ kernel.compute(y1, y1)
            @ inverse_feature_gramian_1
            @ feature_gramian_1
        )
        term_2 = (
            inverse_feature_gramian_2
            @ kernel.compute(y2, y2)
            @ inverse_feature_gramian_2
            @ feature_gramian_2
        )
        term_3 = (
            inverse_feature_gramian_1
            @ kernel.compute(y1, y2)
            @ inverse_feature_gramian_2
            @ kernel.compute(x2, x1)
        )

        # Compute CMMD
        expected_cmmd = jnp.sqrt(
            jnp.trace(term_1) + jnp.trace(term_2) - 2 * jnp.trace(term_3)
        )

        # Compute the CMMD using the metric object
        metric = CMMD(
            feature_kernel=kernel,
            response_kernel=kernel,
            regularisation_parameter=regularisation_parameter,
        )
        output = metric.compute(reference_data, comparison_data)
        assert output == pytest.approx(expected_cmmd, abs=1e-6)

    # pylint: enable=too-many-locals

    @pytest.mark.parametrize(
        "approximator",
        [
            RandomisedEigendecompositionApproximator(
                jr.key(0), oversampling_parameter=100, power_iterations=2
            )
        ],
    )
    def test_approximate_cmmd_random_data(
        self,
        problem: _SupervisedMetricProblem,
        approximator: RegularisedInverseApproximator,
    ):
        r"""
        Test approximate CMMD is close to exact CMMD.
        """
        kernel = SquaredExponentialKernel()
        x, y = problem

        # Compute exact and approximate CMMD
        metric = CMMD(
            feature_kernel=kernel, response_kernel=kernel, regularisation_parameter=1e-6
        )
        output = metric.compute(x, y)

        approximate_metric = CMMD(
            feature_kernel=kernel,
            response_kernel=kernel,
            regularisation_parameter=1e-6,
            inverse_approximator=approximator,
        )
        approximate_output = approximate_metric.compute(x, y)

        assert output == pytest.approx(approximate_output, abs=1e-1)
