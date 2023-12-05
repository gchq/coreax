# © Crown Copyright GCHQ
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import unittest

import jax.random
import numpy as np
from flax import linen as nn
from jax.random import rademacher
from jax.scipy.stats import multivariate_normal, norm
from optax import sgd

import coreax.kernel as ck
import coreax.networks as cn
import coreax.score_matching as csm


class TestNetwork(nn.Module):
    """
    A simple neural network for use in testing of sliced score matching.
    """

    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x: csm.ArrayLike) -> csm.ArrayLike:
        x = nn.Dense(self.hidden_dim)(x)
        return x


class TestKernelDensityMatching(unittest.TestCase):
    """
    Tests related to the class in score_matching.py
    """

    def test_univariate_gaussian_score(self) -> None:
        """
        Test a simple univariate Gaussian with a known score function.
        """
        # Setup univariate Gaussian
        mu = 0.0
        std_dev = 1.0
        num_points = 500
        np.random.seed(0)
        samples = np.random.normal(mu, std_dev, size=(num_points, 1))

        def true_score(x_: csm.ArrayLike) -> csm.ArrayLike:
            return -(x_ - mu) / std_dev**2

        # Define data
        x = np.linspace(-2, 2).reshape(-1, 1)
        true_score_result = true_score(x)

        # Define a kernel density matching object
        kernel_density_matcher = csm.KernelDensityMatching(
            length_scale=ck.median_heuristic(samples), kde_data=samples
        )

        # Extract the score function (this is not really learned from the data, more
        # defined within the object)
        learned_score = kernel_density_matcher.match()
        score_result = learned_score(x)

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.5)

    def test_multivariate_gaussian_score(self) -> None:
        """
        Test a simple multivariate Gaussian with a known score function.
        """
        # Setup multivariate Gaussian
        dimension = 2
        mu = np.zeros(dimension)
        sigma_matrix = np.eye(dimension)
        lambda_matrix = np.linalg.pinv(sigma_matrix)
        num_points = 500
        np.random.seed(0)
        samples = np.random.multivariate_normal(mu, sigma_matrix, size=num_points)

        def true_score(x_: csm.ArrayLike) -> csm.ArrayLike:
            return np.array(list(map(lambda z: -lambda_matrix @ (z - mu), x_)))

        # Define data
        x, y = np.meshgrid(np.linspace(-2, 2), np.linspace(-2, 2))
        data_stacked = np.vstack([x.ravel(), y.ravel()]).T
        true_score_result = true_score(data_stacked)

        # Define a kernel density matching object
        kernel_density_matcher = csm.KernelDensityMatching(
            length_scale=ck.median_heuristic(samples), kde_data=samples
        )

        # Extract the score function (this is not really learned from the data, more
        # defined within the object)
        learned_score = kernel_density_matcher.match()
        score_result = learned_score(data_stacked)

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.75)

    def test_univariate_gmm_score(self):
        """
        Test a univariate Gaussian mixture model with a known score function.
        """
        # Define the univariate Gaussian mixture model
        mus = np.array([-4.0, 4.0])
        std_devs = np.array([1.0, 2.0])
        p = 0.7
        mix = np.array([1 - p, p])
        num_points = 1000
        np.random.seed(0)
        comp = np.random.binomial(1, p, size=num_points)
        samples = np.random.normal(mus[comp], std_devs[comp]).reshape(-1, 1)

        def egrad(g: csm.Callable) -> csm.Callable:
            def wrapped(x_, *rest):
                y, g_vjp = jax.vjp(lambda x__: g(x, *rest), x_)
                (x_bar,) = g_vjp(np.ones_like(y))
                return x_bar

            return wrapped

        def true_score(x_: csm.ArrayLike) -> csm.ArrayLike:
            log_pdf = lambda y: jax.numpy.log(norm.pdf(y, mus, std_devs) @ mix)
            return egrad(log_pdf)(x_)

        # Define data
        x = np.linspace(-10, 10).reshape(-1, 1)
        true_score_result = true_score(x)

        # Define a kernel density matching object
        kernel_density_matcher = csm.KernelDensityMatching(
            length_scale=ck.median_heuristic(samples), kde_data=samples
        )

        # Extract the score function (this is not really learned from the data, more
        # defined within the object)
        learned_score = kernel_density_matcher.match()
        score_result = learned_score(x)

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.5)

    def test_multivariate_gmm_score(self):
        """
        Test a multivariate Gaussian mixture model with a known score function.
        """
        # Define the multivariate Gaussian mixture model (we don't want to go much
        # higher than dimension=2)
        np.random.seed(0)
        dimension = 2
        k = 10
        mus = np.random.multivariate_normal(
            np.zeros(dimension), np.eye(dimension), size=k
        )
        sigmas = np.array(
            [np.random.gamma(2.0, 1.0) * np.eye(dimension) for _ in range(k)]
        )
        mix = np.random.dirichlet(np.ones(k))
        num_points = 500
        comp = np.random.choice(k, size=num_points, p=mix)
        samples = np.array(
            [np.random.multivariate_normal(mus[c], sigmas[c]) for c in comp]
        )

        def egrad(g: csm.Callable) -> csm.Callable:
            def wrapped(x_, *rest):
                y, g_vjp = jax.vjp(lambda x__: g(x_, *rest), x_)
                (x_bar,) = g_vjp(np.ones_like(y))
                return x_bar

            return wrapped

        def true_score(x_: csm.ArrayLike) -> csm.ArrayLike:
            def logpdf(y: csm.ArrayLike) -> csm.ArrayLike:
                lpdf = 0.0
                for k_ in range(k):
                    lpdf += multivariate_normal.pdf(y, mus[k_], sigmas[k_]) * mix[k_]
                return jax.numpy.log(lpdf)

            return egrad(logpdf)(x_)

        # Define data
        coords = np.meshgrid(*[np.linspace(-7.5, 7.5) for _ in range(dimension)])
        x_stacked = np.vstack([c.ravel() for c in coords]).T
        true_score_result = true_score(x_stacked)

        # Define a kernel density matching object
        kernel_density_matcher = csm.KernelDensityMatching(
            length_scale=ck.median_heuristic(samples), kde_data=samples
        )

        # Extract the score function (this is not really learned from the data, more
        # defined within the object)
        learned_score = kernel_density_matcher.match()
        score_result = learned_score(x_stacked)

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.5)


class TestSlicedScoreMatching(unittest.TestCase):
    """
    Tests related to the class SlicedScoreMatching in score_matching.py.
    """

    def test_analytic_objective_orthogonal(self) -> None:
        r"""
        Test the core objective function, analytic version.

        We consider two orthogonal vectors, u and v, and a score vector of ones. The
        analytic objective is given by:

        .. math::

            v' u + 0.5 * ||s||^2

        In the case of v and u being orthogonal, this reduces to:

        .. math::

            0.5 * ||s||^2

        which equals 1.0 in the case of s being a vector of ones.
        """
        # Define data
        u = np.array([0.0, 1.0])
        v = np.array([[1.0, 0.0]])
        s = np.ones(2, dtype=float)

        # Define expected output - orthogonal u and v vectors should give back
        # half-length squared s
        expected_output = 1.0

        # Define a sliced score matching object - with the analytic objective
        sliced_score_matcher = csm.SlicedScoreMatching(
            random_generator=rademacher, use_analytic=True
        )

        # Evaluate the analytic objective function
        output = sliced_score_matcher._objective_function(v, u, s)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=3)

    def test_analytic_objective(self) -> None:
        r"""
        Test the core objective function, analytic version.

        We consider the following vectors:

        .. math::

            u = [0, 1, 2]

            v = [3, 4, 5]

            s = [9, 10, 11]

        and the analytic objective

        .. math::

            v' u + 0.5 * ||s||^2

        Evaluating this gives a result of 165.0. We compare this to the general
        objective, which has the form:

        .. math::

            v' u + 0.5 * (v' s)^2

        which evaluates to 7456.0 when substituting in the given values of u, v and s.
        """
        # Define data
        u = np.arange(3, dtype=float)
        v = np.arange(3, 6, dtype=float)
        s = np.arange(9, 12, dtype=float)

        # Define expected outputs
        expected_output_analytic = 165.0
        expected_output_general = 7456.0

        # Define a sliced score matching object - with the analytic objective
        sliced_score_matcher = csm.SlicedScoreMatching(
            random_generator=rademacher, use_analytic=True
        )

        # Evaluate the analytic objective function
        output = sliced_score_matcher._objective_function(v, u, s)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output_analytic, places=3)

        # Mutate the objective, and check that the result changes
        sliced_score_matcher.use_analytic = False
        output = sliced_score_matcher._objective_function(v, u, s)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output_general, places=3)

    def test_general_objective_orthogonal(self) -> None:
        r"""
        Test the core objective function, non-analytic version.

        We consider the following vectors:

        .. math::

            u = [0, 1]

            v = [1, 0]

            s = [1, 1]


        The general objective has the form:

        .. math::

            v' u + 0.5 * (v' s)^2

        We consider orthogonal vectors v and u, meaning we only evaluate the second term
        to get the expected output.
        """
        # Define data - orthogonal u and v vectors should give back half squared dot
        # product of v and s
        u = np.array([0.0, 1.0])
        v = np.array([[1.0, 0.0]])
        s = np.ones(2, dtype=float)

        # Define expected outputs
        expected_output = 0.5

        # Define a sliced score matching object - with the non-analytic objective
        sliced_score_matcher = csm.SlicedScoreMatching(
            random_generator=rademacher, use_analytic=False
        )

        # Evaluate the analytic objective function
        output = sliced_score_matcher._objective_function(v, u, s)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=3)

    def test_general_objective(self) -> None:
        r"""
        Test the core objective function, non-analytic version.

        We consider the following vectors:

        .. math::

            u = [0, 1, 2]

            v = [3, 4, 5]

            s = [9, 10, 11]

        The general objective has the form:

        .. math::

            v' u + 0.5 * (v' s)^2

        Evaluating this gives a result of 7456.0.
        """
        # Define data
        u = np.arange(3, dtype=float)
        v = np.arange(3, 6, dtype=float)
        s = np.arange(9, 12, dtype=float)

        # Define expected outputs
        expected_output = 7456.0

        # Define a sliced score matching object - with the non-analytic objective
        sliced_score_matcher = csm.SlicedScoreMatching(
            random_generator=rademacher, use_analytic=False
        )

        # Evaluate the analytic objective function
        output = sliced_score_matcher._objective_function(v, u, s)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=3)

    def test_sliced_score_matching_loss_element_analytic(self) -> None:
        """
        Test the loss function elementwise.

        We use the analytic loss function in this example.
        """

        def score_function(y: csm.ArrayLike) -> csm.ArrayLike:
            """
            Basic score function, implicitly multivariate vector valued.

            :param y: point at which to evaluate the score function
            :return: score function (gradient of log density) evaluated at x
            """
            return y**2

        # Define an arbitrary input
        x = np.array([2.0, 7.0])
        s = score_function(x)

        # Defined the Hessian (grad of score function)
        hessian = 2.0 * np.diag(x)

        # Define some arbitrary random vector
        random_vector = np.ones(2, dtype=float)

        # Define a sliced score matching object
        sliced_score_matcher = csm.SlicedScoreMatching(
            random_generator=rademacher, use_analytic=True
        )

        # Determine the expected output - using the analytic objective function tested
        # elsewhere
        expected_output = sliced_score_matcher._objective_function(
            random_vector[None, :], hessian @ random_vector, s
        )

        # Evaluate the loss element
        output = sliced_score_matcher._loss_element(x, random_vector, score_function)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=3)

        # Call the loss element with a different objective function, and check that the
        # jit compilation recognises this change
        sliced_score_matcher.use_analytic = False
        output_changed_objective = sliced_score_matcher._loss_element(
            x, random_vector, score_function
        )
        self.assertNotAlmostEqual(output, output_changed_objective)

    def test_sliced_score_matching_loss_element_general(self) -> None:
        """
        Test the loss function elementwise.

        We use the non-analytic loss function in this example.
        """

        def score_function(x_: csm.ArrayLike) -> csm.ArrayLike:
            """
            Basic score function, implicitly multivariate vector valued.

            :param x_: point at which to evaluate the score function
            :return: score function (gradient of log density) evaluated at x
            """
            return x_**2

        # Define an arbitrary input
        x = np.array([2.0, 7.0])
        s = score_function(x)

        # Defined the Hessian (grad of score function)
        hessian = 2.0 * np.diag(x)

        # Define some arbitrary random vector
        random_vector = np.ones(2, dtype=float)

        # Define a sliced score matching object
        sliced_score_matcher = csm.SlicedScoreMatching(
            random_generator=rademacher, use_analytic=False
        )

        # Determine the expected output
        expected_output = sliced_score_matcher._objective_function(
            random_vector, hessian @ random_vector, s
        )

        # Evaluate the loss element
        output = sliced_score_matcher._loss_element(x, random_vector, score_function)

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=3)

    def test_sliced_score_matching_loss(self) -> None:
        """
        Test the vmapped loss function.
        """

        def score_function(x_: csm.ArrayLike) -> csm.ArrayLike:
            """
            Basic score function, implicitly multivariate vector valued.

            :param x_: point at which to evaluate the score function
            :return: score function (gradient of log density) evaluated at x
            """
            return x_**2

        # Define an arbitrary input
        x = np.tile(np.array([2.0, 7.0]), (10, 1))

        # Define arbitrary number of random vectors, 1 per input
        random_vectors = np.ones((10, 1, 2), dtype=float)

        # Set expected output
        expected_output = np.ones((10, 1), dtype=float) * 1226.5

        # Define a sliced score matching object
        sliced_score_matcher = csm.SlicedScoreMatching(
            random_generator=rademacher, use_analytic=True
        )
        output = sliced_score_matcher._loss(score_function)(x, random_vectors)

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    def test_train_step(self) -> None:
        """
        Test the basic training step.
        """
        # Define a simple linear model that we can compute the gradients for by hand
        score_network = TestNetwork(2, 2)

        # Define a sliced score matching object
        sliced_score_matcher = csm.SlicedScoreMatching(
            random_generator=rademacher,
            use_analytic=True,
            random_key=jax.random.PRNGKey(0),
        )

        # Create a train state. setting the PRNG with fixed seed means initialisation is
        # consistent for testing using SGD
        state = cn.create_train_state(
            score_network, jax.random.PRNGKey(0), 1e-3, 2, sgd
        )

        # Jax is row-based, so we have to work with the kernel transpose
        weights = state.params["Dense_0"]["kernel"].T
        bias = state.params["Dense_0"]["bias"]

        # Define input data
        x = np.array([2.0, 7.0])
        v = np.ones((1, 2), dtype=float)
        s = weights @ x.T + bias

        # Reformat for the vector mapped input to loss
        x_to_vector_map = np.array([x])
        v_to_vector_map = np.ones((1, 1, 2), dtype=float)

        # Compute these gradients by hand
        grad_weights = jax.numpy.outer(v, v) + jax.numpy.outer(s, x)
        grad_bias = s

        weights_ = weights - 1e-3 * grad_weights
        bias_ = bias - 1e-3 * grad_bias

        state, _ = sliced_score_matcher._train_step(
            state, x_to_vector_map, v_to_vector_map
        )

        # Jax is row based, so transpose W_
        np.testing.assert_array_almost_equal(
            weights_.T, state.params["Dense_0"]["kernel"], decimal=3
        )
        np.testing.assert_array_almost_equal(
            bias_, state.params["Dense_0"]["bias"], decimal=3
        )

    def test_univariate_gaussian_score(self):
        """
        Test a simple univariate Gaussian with a known score function.
        """
        # Setup univariate Gaussian
        mu = 0.0
        std_dev = 1.0
        num_points = 500
        np.random.seed(0)
        samples = np.random.normal(mu, std_dev, size=(num_points, 1))

        def true_score(x_: csm.ArrayLike) -> csm.ArrayLike:
            return -(x_ - mu) / std_dev**2

        # Define data
        x = np.linspace(-2, 2).reshape(-1, 1)
        true_score_result = true_score(x)

        # Define a sliced score matching object
        sliced_score_matcher = csm.SlicedScoreMatching(
            random_generator=jax.random.normal,
            use_analytic=True,
            hidden_dim=32,
            num_epochs=10,
        )

        # Learn score function with noise conditioning
        learned_score = sliced_score_matcher.match(samples)
        score_result_with_noise_conditioning = learned_score(x)
        self.assertLessEqual(
            np.abs(true_score_result - score_result_with_noise_conditioning).mean(), 1.0
        )

        # Learn score function without noise conditioning
        sliced_score_matcher.noise_conditioning = False
        learned_score = sliced_score_matcher.match(samples)
        score_result_without_noise_conditioning = learned_score(x)
        self.assertLessEqual(
            np.abs(true_score_result - score_result_without_noise_conditioning).mean(),
            1.0,
        )

    def test_multivariate_gaussian_score(self) -> None:
        """
        Test a simple multivariate Gaussian with a known score function.
        """
        # Setup multivariate Gaussian
        dimension = 2
        mu = np.zeros(dimension)
        sigma_matrix = np.eye(dimension)
        lambda_matrix = np.linalg.pinv(sigma_matrix)
        num_points = 500
        np.random.seed(0)
        samples = np.random.multivariate_normal(mu, sigma_matrix, size=num_points)

        def true_score(x_: csm.ArrayLike) -> csm.ArrayLike:
            return np.array(list(map(lambda z: -lambda_matrix @ (z - mu), x_)))

        # Define data
        x, y = np.meshgrid(np.linspace(-2, 2), np.linspace(-2, 2))
        data_stacked = np.vstack([x.ravel(), y.ravel()]).T
        true_score_result = true_score(data_stacked)

        # Define a sliced score matching object
        sliced_score_matcher = csm.SlicedScoreMatching(
            random_generator=jax.random.normal,
            use_analytic=True,
            hidden_dim=32,
            num_epochs=10,
        )

        # Learn score function with noise conditioning
        learned_score = sliced_score_matcher.match(samples)
        score_result_with_noise_conditioning = learned_score(data_stacked)
        self.assertLessEqual(
            np.abs(true_score_result - score_result_with_noise_conditioning).mean(), 1.0
        )

        # Learn score function without noise conditioning
        sliced_score_matcher.noise_conditioning = False
        learned_score = sliced_score_matcher.match(samples)
        score_result_with_noise_conditioning = learned_score(data_stacked)
        self.assertLessEqual(
            np.abs(true_score_result - score_result_with_noise_conditioning).mean(), 1.0
        )

    def test_univariate_gmm_score(self):
        """
        Test a univariate Gaussian mixture model with a known score function.
        """
        # Define the univariate Gaussian mixture model
        mus = np.array([-4.0, 4.0])
        std_devs = np.array([1.0, 2.0])
        p = 0.7
        mix = np.array([1 - p, p])
        num_points = 1000
        np.random.seed(0)
        comp = np.random.binomial(1, p, size=num_points)
        samples = np.random.normal(mus[comp], std_devs[comp]).reshape(-1, 1)

        def egrad(g: csm.Callable) -> csm.Callable:
            def wrapped(x_, *rest):
                y, g_vjp = jax.vjp(lambda x__: g(x, *rest), x_)
                (x_bar,) = g_vjp(np.ones_like(y))
                return x_bar

            return wrapped

        def true_score(x_: csm.ArrayLike) -> csm.ArrayLike:
            log_pdf = lambda y: jax.numpy.log(norm.pdf(y, mus, std_devs) @ mix)
            return egrad(log_pdf)(x_)

        # Define data
        x = np.linspace(-10, 10).reshape(-1, 1)
        true_score_result = true_score(x)

        # Define a sliced score matching object
        sliced_score_matcher = csm.SlicedScoreMatching(
            random_generator=jax.random.normal,
            use_analytic=True,
            hidden_dim=32,
            num_epochs=10,
        )

        # Learn score function with noise conditioning
        learned_score = sliced_score_matcher.match(samples)
        score_result_with_noise_conditioning = learned_score(x)
        self.assertLessEqual(
            np.abs(true_score_result - score_result_with_noise_conditioning).mean(), 1.0
        )

        # Learn score function without noise conditioning
        sliced_score_matcher.noise_conditioning = False
        learned_score = sliced_score_matcher.match(samples)
        score_result_with_noise_conditioning = learned_score(x)
        self.assertLessEqual(
            np.abs(true_score_result - score_result_with_noise_conditioning).mean(), 1.0
        )

    def test_multivariate_gmm_score(self):
        """
        Test a multivariate Gaussian mixture model with a known score function.
        """
        # Define the multivariate Gaussian mixture model (we don't want to go much
        # higher than dimension=2)
        np.random.seed(0)
        dimension = 2
        k = 10
        mus = np.random.multivariate_normal(
            np.zeros(dimension), np.eye(dimension), size=k
        )
        sigmas = np.array(
            [np.random.gamma(2.0, 1.0) * np.eye(dimension) for _ in range(k)]
        )
        mix = np.random.dirichlet(np.ones(k))
        num_points = 500
        comp = np.random.choice(k, size=num_points, p=mix)
        samples = np.array(
            [np.random.multivariate_normal(mus[c], sigmas[c]) for c in comp]
        )

        def egrad(g: csm.Callable) -> csm.Callable:
            def wrapped(x_, *rest):
                y, g_vjp = jax.vjp(lambda x__: g(x_, *rest), x_)
                (x_bar,) = g_vjp(np.ones_like(y))
                return x_bar

            return wrapped

        def true_score(x_: csm.ArrayLike) -> csm.ArrayLike:
            def logpdf(y: csm.ArrayLike) -> csm.ArrayLike:
                lpdf = 0.0
                for k_ in range(k):
                    lpdf += multivariate_normal.pdf(y, mus[k_], sigmas[k_]) * mix[k_]
                return jax.numpy.log(lpdf)

            return egrad(logpdf)(x_)

        # Define data
        coords = np.meshgrid(*[np.linspace(-7.5, 7.5) for _ in range(dimension)])
        x_stacked = np.vstack([c.ravel() for c in coords]).T
        true_score_result = true_score(x_stacked)

        # Define a sliced score matching object
        sliced_score_matcher = csm.SlicedScoreMatching(
            random_generator=jax.random.normal,
            use_analytic=True,
            hidden_dim=32,
            num_epochs=10,
        )

        # Learn score function with noise conditioning
        learned_score = sliced_score_matcher.match(samples)
        score_result_with_noise_conditioning = learned_score(x_stacked)
        self.assertLessEqual(
            np.abs(true_score_result - score_result_with_noise_conditioning).mean(), 1.0
        )

        # Learn score function without noise conditioning
        sliced_score_matcher.noise_conditioning = False
        learned_score = sliced_score_matcher.match(samples)
        score_result_with_noise_conditioning = learned_score(x_stacked)
        self.assertLessEqual(
            np.abs(true_score_result - score_result_with_noise_conditioning).mean(), 1.0
        )


if __name__ == "__main__":
    unittest.main()
