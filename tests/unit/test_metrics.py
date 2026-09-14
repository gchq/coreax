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

from typing import NamedTuple

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest
from jaxtyping import Array, ArrayLike

from coreax.data import Data, SupervisedData
from coreax.kernels import (
    LaplacianKernel,
    LinearKernel,
    PCIMQKernel,
    RationalQuadraticKernel,
    ScalarValuedKernel,
    SquaredExponentialKernel,
    SteinKernel,
)
from coreax.metrics import AMCMD, JMMD, KSD, MMD
from coreax.util import pairwise


class _MetricProblem(NamedTuple):
    reference_data: Data
    comparison_data: Data


class _SupervisedMetricProblem(NamedTuple):
    reference_data: SupervisedData
    comparison_data: SupervisedData
    weighting_data: Data | None = None


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
    def test_mmd_compare_same_data(
        self, problem: _MetricProblem, kernel: ScalarValuedKernel
    ):
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

    @pytest.mark.parametrize("weighted", [False, True])
    def test_mmd_known_properties(
        self, problem: _MetricProblem, weighted: bool
    ) -> None:
        """Check MMD symmetry and invariance to sample ordering."""
        reference_data, comparison_data = problem
        if not weighted:
            reference_data = Data(reference_data.data)
            comparison_data = Data(comparison_data.data)

        metric = MMD(SquaredExponentialKernel())
        expected = metric.compute(reference_data, comparison_data)

        assert metric.compute(comparison_data, reference_data) == pytest.approx(
            expected, abs=1e-6
        )

        permuted_reference = reference_data[::-1]
        permuted_comparison = comparison_data[::-1]

        assert metric.compute(permuted_reference, permuted_comparison) == pytest.approx(
            expected, abs=1e-6
        )

    @pytest.mark.parametrize("weighted", [False, True])
    def test_mmd_translation_invariance(
        self, problem: _MetricProblem, weighted: bool
    ) -> None:
        """Check MMD is unchanged by a common translation of both datasets."""
        reference_data, comparison_data = problem
        if not weighted:
            reference_data = Data(reference_data.data)
            comparison_data = Data(comparison_data.data)

        metric = MMD(SquaredExponentialKernel())
        expected = metric.compute(reference_data, comparison_data)
        translation = jnp.arange(reference_data.data.shape[1])
        translated_reference = Data(
            reference_data.data + translation, reference_data.weights
        )
        translated_comparison = Data(
            comparison_data.data + translation, comparison_data.weights
        )

        assert metric.compute(
            translated_reference, translated_comparison
        ) == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("weighted", [False, True])
    def test_mmd_rotation_invariance(
        self, problem: _MetricProblem, weighted: bool
    ) -> None:
        """Check MMD is unchanged by an orthogonal rotation for a 2-norm kernel."""
        reference_data, comparison_data = problem
        if not weighted:
            reference_data = Data(reference_data.data)
            comparison_data = Data(comparison_data.data)

        metric = MMD(SquaredExponentialKernel())
        expected = metric.compute(reference_data, comparison_data)
        rotation = jnp.eye(reference_data.data.shape[1])
        rotation = rotation.at[:2, :2].set(jnp.array([[0.0, -1.0], [1.0, 0.0]]))
        rotated_reference = Data(reference_data.data @ rotation, reference_data.weights)
        rotated_comparison = Data(
            comparison_data.data @ rotation, comparison_data.weights
        )

        assert metric.compute(rotated_reference, rotated_comparison) == pytest.approx(
            expected, abs=1e-6
        )

    @pytest.mark.parametrize("weighted", [False, True])
    def test_mmd_scale_length_scale_invariance(
        self, problem: _MetricProblem, weighted: bool
    ) -> None:
        """Check joint scaling of data and kernel length scale preserves MMD."""
        reference_data, comparison_data = problem
        if not weighted:
            reference_data = Data(reference_data.data)
            comparison_data = Data(comparison_data.data)

        length_scale = 0.75
        scale = 3.0
        expected = MMD(SquaredExponentialKernel(length_scale=length_scale)).compute(
            reference_data, comparison_data
        )
        scaled_reference = Data(reference_data.data * scale, reference_data.weights)
        scaled_comparison = Data(comparison_data.data * scale, comparison_data.weights)
        scaled_metric = MMD(SquaredExponentialKernel(length_scale=length_scale * scale))

        assert scaled_metric.compute(
            scaled_reference, scaled_comparison
        ) == pytest.approx(expected, abs=1e-6)


class TestKSD:
    """
    Tests related to the kernel Stein discrepancy (KSD) class in metrics.py.
    """

    @pytest.fixture
    def problem(self) -> _MetricProblem:
        """Generate an example problem for testing KSD."""
        dimension = 1
        num_points = 250, 100
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
    def test_ksd_compare_same_data(
        self, problem: _MetricProblem, kernel: ScalarValuedKernel
    ):
        """Check KSD of a dataset with itself is (very) approximately zero."""
        x = problem.reference_data
        metric = KSD(kernel)
        assert metric.compute(
            x, x, laplace_correct=False, regularise=False
        ) == pytest.approx(0.0, abs=1e0)

    def test_ksd_analytically_known_stein_kernel(self) -> None:
        r"""
        Test KSD computation against an analytically derived stein kernel solution.

        Given a standard multivariate normal distribution
        :math:`\mathbb{P} = \mathcal{N}(0, I)`, and the PCIMQ kernel function
        :math:`k: \mathbb{R}^2 \times \mathbb{R}^2 \rightarrow \mathbb{R}` such that
        :math:`k(x, y) = \frac{1}{\sqrt{1 + ||x-y||^2}}, one can analytically derive the
        induced Stein kernel (see Exercise 3, :cite:`oates2021uncertainty`):

        .. math::

            k_{\mathbb{P}}(x, y) = -\frac{3||x-y||^2}{(1 + ||x-y||^2)^{\frac{5}{2}}}
            + \frac{2 + (x-y)^T(y-x)}{(1 + ||x-y||^2)^{\frac{3}{2}}}
            + \frac{x^Ty}{(1 + ||x-y||^2)^{\frac{1}{2}}}.

        Then, given :math:`x_i \ sim \mathbb{Q}` where :math:`\mathbb{Q}` is some other
        distribution, the Kernelised Stein Discrepancy can be computed as

        .. math::

            KSD^2(\mathbb{P}, \mathbb{Q})
            =  \frac{1}{m^2}\sum_{i, j = 1}^m k_{\mathbb{P}}(x_i, x_j)
        """
        random_key = jr.key(2_024)
        dim = 2
        num_data_points = 100

        # Generate data from a uniform distribution Q with which to compute the expected
        # KSD.
        comparison_pts = jr.uniform(
            random_key,
            shape=(num_data_points, dim),
        )

        # Analytic stein kernel
        def _stein_kernel(x: ArrayLike, y: ArrayLike) -> Array:
            norm_sq = jnp.linalg.norm(jnp.subtract(x, y)) ** 2
            body = 1 + norm_sq
            first_term = 3 * norm_sq / body ** (5 / 2)
            second_term = (dim - norm_sq) / body ** (3 / 2)
            third_term = jnp.dot(x, y) / body ** (1 / 2)
            return -first_term + second_term + third_term

        stein_kernel_matrix = pairwise(_stein_kernel)(comparison_pts, comparison_pts)
        expected_output = jnp.sqrt(stein_kernel_matrix.sum() / num_data_points**2)

        # Compute the KSD using the metric object via a stein kernel induced by a
        # PCIMQ kernel, with the corresponding analytic score function and
        # reference data from a standard multivariate distribution P.
        reference_pts = jr.multivariate_normal(
            random_key,
            jnp.zeros(dim),
            jnp.eye(dim),
            shape=(num_data_points,),
        )
        base_kernel = PCIMQKernel(length_scale=1 / np.sqrt(2), output_scale=1)

        def _score_function(x: ArrayLike) -> Array:
            return -jnp.asarray(x)

        stein_kernel = SteinKernel(base_kernel, score_function=_score_function)
        metric = KSD(kernel=stein_kernel)
        output = metric.compute(Data(reference_pts), Data(comparison_pts))

        assert output == pytest.approx(expected_output)

    @pytest.mark.parametrize(
        ("weighted", "laplace_correct", "regularise"),
        [
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ],
    )
    def test_ksd_permutation_invariant(
        self,
        problem: _MetricProblem,
        weighted: bool,
        laplace_correct: bool,
        regularise: bool,
    ) -> None:
        """Check KSD is invariant to the ordering of empirical samples."""
        reference_data, comparison_data = problem
        if not weighted:
            reference_data = Data(reference_data.data)
            comparison_data = Data(comparison_data.data)

        metric = KSD(SquaredExponentialKernel())
        expected = metric.compute(
            reference_data,
            comparison_data,
            laplace_correct=laplace_correct,
            regularise=regularise,
        )

        permuted_reference = reference_data[::-1]
        permuted_comparison = comparison_data[::-1]

        actual = metric.compute(
            permuted_reference,
            permuted_comparison,
            laplace_correct=laplace_correct,
            regularise=regularise,
        )
        assert actual == pytest.approx(expected, abs=1e-6, rel=1e-3)
        assert jnp.isfinite(actual)
        assert actual >= 0


class TestAMCMD:
    """
    Tests for average maximum conditional mean discrepancy (AMCMD) class in metrics.py.
    """

    @pytest.fixture
    def problem(self) -> _SupervisedMetricProblem:
        """Generate an example problem for testing AMCMD."""
        num_points = 250, 125, 200
        keys = tuple(jr.split(jr.key(0), 3))

        def _generate_supervised_data(_num_points: int, _key: Array) -> SupervisedData:
            point_key, noise_key = jr.split(_key, 2)
            points = jr.uniform(point_key, (_num_points, 1))
            supervision = points + 0.1 * jr.normal(noise_key, (points.shape))
            return SupervisedData(points, supervision)

        reference_data, comparison_data, weighting_data = jtu.tree_map(
            _generate_supervised_data, num_points, keys
        )
        return _SupervisedMetricProblem(reference_data, comparison_data, weighting_data)

    @pytest.mark.parametrize(
        "feature_kernel", [SquaredExponentialKernel(), LaplacianKernel(), PCIMQKernel()]
    )
    @pytest.mark.parametrize(
        "response_kernel",
        [SquaredExponentialKernel(), LaplacianKernel(), PCIMQKernel()],
    )
    def test_amcmd_compare_same_data(
        self,
        problem: _SupervisedMetricProblem,
        feature_kernel: ScalarValuedKernel,
        response_kernel: ScalarValuedKernel,
    ):
        """Check AMCMD of a dataset with itself is approximately zero."""
        x, w = problem.reference_data, problem.weighting_data
        metric = AMCMD(feature_kernel, response_kernel, regularisation_parameter=1e-3)
        assert metric.compute(x, x, weighting_data=w) == pytest.approx(0.0)
        assert metric.compute(x, x) == pytest.approx(0.0)

    def test_amcmd_analytically_known(self):
        r"""
        Test AMCMD computation against an analytically derived solution.

        For the following datasets with features and responses each in 1 dimension:

        .. math::

            \mathcal{D}_1 = [(0,0)]

            \mathcal{D}_2 = [(1, 1), (2, 2)]

            \mathcal{D}_3 = [1]

        taking the RBF kernel, :math:`k(x,y) = \exp (-||x-y||^2)` for the feature
        and the linear kernel :math:`l(x,y) = xy` for the response, gives:

        .. math::

            k(\mathcal{D}_3,\mathcal{D}_1) = \begin{bmatrix} e^{-1} \end{bmatrix}.

            k(\mathcal{D}_1,\mathcal{D}_1) = \begin{bmatrix} 1 \end{bmatrix}.

            k(\mathcal{D}_3,\mathcal{D}_2) = \begin{bmatrix} 1 & e^{-1} \end{bmatrix}.

            k(\mathcal{D}_2,\mathcal{D}_2) =
                \begin{bmatrix}
                    1 & e^{-1} \\
                    e^{-1} & 1
                \end{bmatrix}.

            l(\mathcal{D}_1,\mathcal{D}_1) = \begin{bmatrix} 0 \end{bmatrix}.

            l(\mathcal{D}_2,\mathcal{D}_2) =
                \begin{bmatrix}
                    1 & 2 \\
                    2 & 4
                \end{bmatrix}.

            l(\mathcal{D}_1,\mathcal{D}_2) = \begin{bmatrix} 0 & 0 \end{bmatrix}.


        Then it is easy to see that the first and third terms in the AMCMD formula are
        zero. Then, using the inverse of K_22 given by

        .. math::

            R_22 = \frac{1}{1 - e^{-2}} \begin{bmatrix} 1 & -e^{-1} \\ -e^{-1} & 1
            \end{bmatrix},

        we have that the second term is given by:


        .. math::

            \mathrm{Tr}(K_32 R_22 L_22 R_22 K_23) = \frac{1}{(1 - e^{-2})^2}
            \mathrm{Tr}\left(\begin{bmatrix} 1 & e^{-1} \end{bmatrix}
            \begin{bmatrix} 1 & -e^{-1} \\ -e^{-1} & 1 \end{bmatrix}
            \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}
            \begin{bmatrix} 1 & -e^{-1} \\ -e^{-1} & 1 \end{bmatrix}
            \begin{bmatrix} 1 \\ e^{-1} \end{bmatrix}\right) = 1

        after a few lines of computation.

        """
        reference_data = SupervisedData(jnp.array([[0]]), jnp.array([[0]]))
        comparison_data = SupervisedData(jnp.array([[1], [2]]), jnp.array([[1], [2]]))
        weighting_data = Data(jnp.array([[1]]))
        expected_output = 1

        # Compute AMCMD using the metric object
        metric = AMCMD(
            SquaredExponentialKernel(1 / jnp.sqrt(2)),
            LinearKernel(),
            regularisation_parameter=0,
        )
        output = metric.compute(
            reference_data, comparison_data, weighting_data=weighting_data
        )
        assert output == pytest.approx(expected_output)

    def test_negative_regularisation_parameter(self):
        """Check that error message is raised for negative regularisation."""
        expected_msg = "'regularisation_parameter' should be non-negative"
        with pytest.raises(ValueError, match=expected_msg):
            AMCMD(
                feature_kernel=SquaredExponentialKernel(),
                response_kernel=SquaredExponentialKernel(),
                regularisation_parameter=-1,
            )


class TestJMMD:
    """
    Tests related to the joint maximum mean discrepancy (JMMD) class in metrics.py.
    """

    @pytest.fixture
    def problem(self) -> _SupervisedMetricProblem:
        """Generate an example problem for testing JMMD."""
        dimension = 2
        num_points = 30, 5
        keys = tuple(jr.split(jr.key(0), 2))

        def _generate_supervised_data(_num_points: int, _key: Array) -> SupervisedData:
            point_key, weight_key = jr.split(_key, 2)
            points = jr.uniform(point_key, (_num_points, dimension))
            supervision = points + 0.1 * jr.normal(point_key, (_num_points, dimension))
            weights = jr.uniform(weight_key, (_num_points,))
            return SupervisedData(points, supervision, weights)

        reference_data, comparison_data = jtu.tree_map(
            _generate_supervised_data, num_points, keys
        )
        return _SupervisedMetricProblem(reference_data, comparison_data)

    @pytest.mark.parametrize(
        "feature_kernel",
        [SquaredExponentialKernel(), LaplacianKernel(), PCIMQKernel()],
    )
    @pytest.mark.parametrize(
        "response_kernel",
        [SquaredExponentialKernel(), LaplacianKernel(), PCIMQKernel()],
    )
    def test_jmmd_compare_same_data(
        self,
        problem: _SupervisedMetricProblem,
        feature_kernel: ScalarValuedKernel,
        response_kernel: ScalarValuedKernel,
    ):
        """Check JMMD of a dataset with itself is approximately zero."""
        x = problem.reference_data
        metric = JMMD(feature_kernel, response_kernel)
        assert metric.compute(x, x) == pytest.approx(0.0)

    def test_jmmd_analytically_known(self):
        r"""
        Test JMMD computation against an analytically derived solution.

        For the following datasets with features and responses each in 1 dimension:

        .. math::

            \mathcal{D}_1 = [(-1, -1), (0, 0), (1, 1)]

            \mathcal{D}_2 = [(1, 1), (2, 2)]

        taking the RBF kernel, :math:`k(x,y) = \exp (-||x-y||^2)` for the feature
        and the linear kernel :math:`l(x,y) = xy` for the response, gives:

        .. math::

            k(\mathcal{D}_1,\mathcal{D}_1) =
                \begin{bmatrix}
                       1 & e^{-1} & e^{-4}
                    \\ e^{-1} & 1 & e^{-1}
                    \\ e^{-4} & e^{-1} & 1
                \end{bmatrix}.

            k(\mathcal{D}_1,\mathcal{D}_2)  =
                \begin{bmatrix}
                       e^{-4} & e^{-9}
                    \\ e^{-1} & e^{-4}
                    \\ 1 & e^{-1}
                \end{bmatrix}.

            k(\mathcal{D}_2,\mathcal{D}_2)  =
                \begin{bmatrix}
                       1 & e^{-1}
                    \\ e^{-1} & 1
                \end{bmatrix}.

            l(\mathcal{D}_1,\mathcal{D}_1) =
                \begin{bmatrix}
                       1 & 0 & -1
                    \\ 0 & 0 & 0
                    \\ -1 & 0 & 1
                \end{bmatrix}.

            l(\mathcal{D}_1,\mathcal{D}_2)  =
                \begin{bmatrix}
                       -1 & -2
                    \\ 0 & 0
                    \\ 1 & 2
                \end{bmatrix}

            l(\mathcal{D}_2,\mathcal{D}_2)  =
                \begin{bmatrix}
                       1 & 2
                    \\ 2 & 4
                \end{bmatrix}.

        Then

        .. math::

            \text{JMMD}^2(\mathcal{D}_1,\mathcal{D}_2) =
            \mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_1)
            \circ
            l(\mathcal{D}_1,\mathcal{D}_1))

            + \mathbb{E}(k(\mathcal{D}_2,\mathcal{D}_2)
            \circ
            l(\mathcal{D}_2,\mathcal{D}_2))

            - 2 \mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_2)
            \circ
            l(\mathcal{D}_1,\mathcal{D}_2))

            = \frac{5 + 4e^{-1}}{4}
            + \frac{2 - 2e^{-4}}{9}
            - \frac{1 + 2e^{-1} - e^{-4} - 2e^{-9}}{3}.
        """
        reference_data = jnp.array([[-1], [0], [1]])
        reference_supervision = jnp.array([[-1], [0], [1]])
        reference_dataset = SupervisedData(
            data=reference_data, supervision=reference_supervision
        )
        comparison_data = jnp.array([[1], [2]])
        comparison_supervision = jnp.array([[1], [2]])
        comparison_dataset = SupervisedData(
            data=comparison_data, supervision=comparison_supervision
        )
        expected_output = jnp.sqrt(
            (5 + 4 * jnp.e**-1) / 4
            - (1 + 2 * jnp.e**-1 - jnp.e**-4 - 2 * jnp.e**-9) / 3
            + (2 - 2 * jnp.e**-4) / 9
        )
        metric = JMMD(SquaredExponentialKernel(1 / jnp.sqrt(2)), LinearKernel())
        output = metric.compute(reference_dataset, comparison_dataset)
        assert output == pytest.approx(expected_output)

    def test_jmmd_analytically_known_weighted(self) -> None:
        r"""
        Test JMMD computation against an analytically derived weighted solution.

        For the following datasets with features and responses each in 1 dimension:

        .. math::

            \mathcal{D}_1 = [(-1, -1), (0, 0), (1, 1)]

            w_1 = [\frac{2}{3}, \frac{1}{3}, 1]

            \mathcal{D}_2 = [(1, 1), (2, 2)]

            w_2 = [\frac{1}{4}, \frac{3}{4}]

        we use the same kernels as in the previous test. Then, the weighted joint
        maximum mean discrepancy is calculated as:

        .. math::

            \text{WMMD}^2(\mathcal{D}_1,\mathcal{D}_2) =

            \frac{1}{||w_1||**2}w_1^T \mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_1)
            \circ
            l(\mathcal{D}_1,\mathcal{D}_1)) w_1

             + \frac{1}{||w_2||**2}w_2^T \mathbb{E}(k(\mathcal{D}_2,\mathcal{D}_2)
            \circ
            l(\mathcal{D}_2,\mathcal{D}_2)) w_2

             - \frac{2}{||w_1||||w_2||} w_1
            \mathbb{E}(k(\mathcal{D}_1,\mathcal{D}_2)
            \circ
            l(\mathcal{D}_1,\mathcal{D}_2)) w_2

            = (\frac{37}{16} + \frac{12}{16} * e^{-1})
            + (\frac{13}{36} - \frac{1}{3} * e^{-4})
            - 2 * (\frac{1}{8} + \frac{6}{8} * e^{-1} - \frac{1}{12} * e^{-4} -
            \frac{6}{12} * e^{-9})
        """
        reference_data = jnp.array([[-1], [0], [1]])
        reference_supervision = jnp.array([[-1], [0], [1]])
        reference_weights = jnp.array([2 / 3, 1 / 3, 1])
        reference_dataset = SupervisedData(
            data=reference_data,
            supervision=reference_supervision,
            weights=reference_weights,
        )

        comparison_data = jnp.array([[1], [2]])
        comparison_supervision = jnp.array([[1], [2]])
        comparison_weights = jnp.array([1 / 4, 3 / 4])
        comparison_dataset = SupervisedData(
            data=comparison_data,
            supervision=comparison_supervision,
            weights=comparison_weights,
        )

        expected_output = jnp.sqrt(
            (37 / 16 + 12 / 16 * jnp.e**-1)
            + (13 / 36 - 1 / 3 * jnp.e**-4)
            - 2 * (1 / 8 + 6 / 8 * jnp.e**-1 - 1 / 12 * jnp.e**-4 - 6 / 12 * jnp.e**-9)
        )
        metric = JMMD(SquaredExponentialKernel(1 / jnp.sqrt(2)), LinearKernel())
        output = metric.compute(reference_dataset, comparison_dataset)
        assert output == pytest.approx(expected_output)

    def test_jmmd_weights_zero_norm(self):
        """Check that zero norm weights return zero output."""
        reference_data = SupervisedData(
            data=jnp.array([[-1], [0], [1]]),
            supervision=jnp.array([[-1], [0], [1]]),
            weights=jnp.array([0, 0, 0]),
        )
        comparison_data = SupervisedData(
            data=jnp.array([[1], [2]]),
            supervision=jnp.array([[1], [2]]),
            weights=jnp.array([0, 0]),
        )
        assert 0 == pytest.approx(
            JMMD(SquaredExponentialKernel(), SquaredExponentialKernel()).compute(
                reference_data, comparison_data
            )
        )
