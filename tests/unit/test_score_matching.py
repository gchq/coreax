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

"""
Tests for score matching implementations.

Score matching fits models to data by ensuring the score function of the model matches
the score function of the data. The tests within this file verify that score matching
approaches used produce the expected results on simple examples.
"""

import unittest
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import random, vjp
from jax.scipy.stats import multivariate_normal, norm
from jax.typing import ArrayLike
from optax import sgd

import coreax.kernel
import coreax.networks
import coreax.score_matching


class SimpleNetwork(nn.Module):
    """
    A simple neural network for use in testing of sliced score matching.
    """

    num_hidden_dim: int
    num_output_dim: int

    @nn.compact
    def __call__(self, x: ArrayLike) -> ArrayLike:
        x = nn.Dense(self.num_hidden_dim)(x)
        return x


class TestKernelDensityMatching(unittest.TestCase):
    """
    Tests related to the class in score_matching.py
    """

    def test_tree_flatten(self) -> None:
        """
        Test the pytree flattens as expected.
        """
        # Setup a matching object
        kernel_density_matcher = coreax.score_matching.KernelDensityMatching(
            length_scale=0.25, kde_data=jnp.ones([2, 3])
        )

        # Flatten the pytree
        output_children, output_aux_data = kernel_density_matcher.tree_flatten()

        # Verify outputs are as expected
        self.assertEqual(len(output_children), 1)
        np.testing.assert_array_equal(output_children[0], jnp.ones([2, 3]))

        # We expect the kernel to be a SquaredExponentialKernel with output scale
        # defined such that it's a normalised (Gaussian) kernel
        self.assertListEqual(list(output_aux_data.keys()), ["kernel"])
        self.assertIsInstance(
            output_aux_data["kernel"], coreax.kernel.SquaredExponentialKernel
        )
        self.assertEqual(output_aux_data["kernel"].length_scale, 0.25)
        self.assertEqual(
            output_aux_data["kernel"].output_scale, 1.0 / (np.sqrt(2 * np.pi) * 0.25)
        )

    def test_univariate_gaussian_score(self) -> None:
        """
        Test a simple univariate Gaussian with a known score function.
        """
        # Setup univariate Gaussian
        mu = 0.0
        std_dev = 1.0
        num_data_points = 500
        generator = np.random.default_rng(1_989)
        samples = generator.normal(mu, std_dev, size=(num_data_points, 1))

        def true_score(x_: ArrayLike) -> ArrayLike:
            return -(x_ - mu) / std_dev**2

        # Define data
        x = np.linspace(-2, 2).reshape(-1, 1)
        true_score_result = true_score(x)

        # Define a kernel density matching object
        kernel_density_matcher = coreax.score_matching.KernelDensityMatching(
            length_scale=coreax.kernel.median_heuristic(samples), kde_data=samples
        )

        # Extract the score function (this is not really learned from the data, more
        # defined within the object)
        learned_score = kernel_density_matcher.match()
        score_result = learned_score(x)

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.5)

    def test_univariate_gaussian_score_1d_input(self) -> None:
        """
        Test a simple univariate Gaussian with a known score function, 1D input.
        """
        # Setup univariate Gaussian
        mu = 0.0
        std_dev = 1.0
        num_points = 500
        np.random.seed(0)
        samples = np.random.normal(mu, std_dev, size=(num_points, 1))

        def true_score(x_: ArrayLike) -> ArrayLike:
            return -(x_ - mu) / std_dev**2

        # Define data and select a specific single point to test
        test_index = 20
        x = np.linspace(-2, 2).reshape(-1, 1)
        true_score_result = true_score(x[test_index, 0])

        # Define a kernel density matching object
        kernel_density_matcher = coreax.score_matching.KernelDensityMatching(
            length_scale=coreax.kernel.median_heuristic(samples), kde_data=samples
        )

        # Extract the score function (this is not really learned from the data, more
        # defined within the object)
        learned_score = kernel_density_matcher.match()
        score_result = learned_score(x[test_index, 0])

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.5)

    # pylint: disable=too-many-locals
    def test_multivariate_gaussian_score(self) -> None:
        """
        Test a simple multivariate Gaussian with a known score function.
        """
        # Setup multivariate Gaussian
        dimension = 2
        mu = np.zeros(dimension)
        sigma_matrix = np.eye(dimension)
        lambda_matrix = np.linalg.pinv(sigma_matrix)
        num_data_points = 500
        generator = np.random.default_rng(1_989)
        samples = generator.multivariate_normal(mu, sigma_matrix, size=num_data_points)

        def true_score(x_: ArrayLike) -> ArrayLike:
            return np.array(list(map(lambda z: -lambda_matrix @ (z - mu), x_)))

        # Define data
        x, y = np.meshgrid(np.linspace(-2, 2), np.linspace(-2, 2))
        data_stacked = np.vstack([x.ravel(), y.ravel()]).T
        true_score_result = true_score(data_stacked)

        # Define a kernel density matching object
        kernel_density_matcher = coreax.score_matching.KernelDensityMatching(
            length_scale=coreax.kernel.median_heuristic(samples), kde_data=samples
        )

        # Extract the score function (this is not really learned from the data, more
        # defined within the object)
        learned_score = kernel_density_matcher.match()
        score_result = learned_score(data_stacked)

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.75)

    # pylint: enable=too-many-locals

    # pylint: disable=too-many-locals
    def test_univariate_gmm_score(self):
        """
        Test a univariate Gaussian mixture model with a known score function.
        """
        # Define the univariate Gaussian mixture model
        mus = np.array([-4.0, 4.0])
        std_devs = np.array([1.0, 2.0])
        p = 0.7
        mix = np.array([1 - p, p])
        num_data_points = 1000
        generator = np.random.default_rng(1_989)
        comp = generator.binomial(1, p, size=num_data_points)
        samples = generator.normal(mus[comp], std_devs[comp]).reshape(-1, 1)

        def e_grad(g: Callable) -> Callable:
            def wrapped(x_, *rest):
                y, g_vjp = vjp(lambda x__: g(x__, *rest), x_)
                (x_bar,) = g_vjp(np.ones_like(y))
                return x_bar

            return wrapped

        def true_score(x_: ArrayLike) -> ArrayLike:
            def log_pdf(y_: ArrayLike) -> ArrayLike:
                return jnp.log(norm.pdf(y_, mus, std_devs) @ mix)

            return e_grad(log_pdf)(x_)

        # Define data
        x = np.linspace(-5, 5).reshape(-1, 1)
        true_score_result = true_score(x)

        # Define a kernel density matching object
        kernel_density_matcher = coreax.score_matching.KernelDensityMatching(
            length_scale=coreax.kernel.median_heuristic(samples), kde_data=samples
        )

        # Extract the score function (this is not really learned from the data, more
        # defined within the object)
        learned_score = kernel_density_matcher.match()
        score_result = learned_score(x)

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.6)

    # pylint: enable=too-many-locals

    # pylint: disable=too-many-locals
    def test_multivariate_gmm_score(self):
        """
        Test a multivariate Gaussian mixture model with a known score function.
        """
        # Define the multivariate Gaussian mixture model (we don't want to go much
        # higher than dimension=2)
        dimension = 2
        k = 10
        num_data_points = 500
        generator = np.random.default_rng(0)
        mus = generator.multivariate_normal(
            np.zeros(dimension), np.eye(dimension), size=k
        )
        sigmas = np.array(
            [generator.gamma(2.0, 1.0) * np.eye(dimension) for _ in range(k)]
        )
        mix = generator.dirichlet(np.ones(k))
        comp = generator.choice(k, size=num_data_points, p=mix)
        samples = np.array(
            [generator.multivariate_normal(mus[c], sigmas[c]) for c in comp]
        )

        def e_grad(g: Callable) -> Callable:
            def wrapped(x_, *rest):
                y, g_vjp = vjp(lambda x__: g(x_, *rest), x_)
                (x_bar,) = g_vjp(np.ones_like(y))
                return x_bar

            return wrapped

        def true_score(x_: ArrayLike) -> ArrayLike:
            def log_pdf(y: ArrayLike) -> ArrayLike:
                l_pdf = 0.0
                for k_ in range(k):
                    l_pdf += multivariate_normal.pdf(y, mus[k_], sigmas[k_]) * mix[k_]
                return jnp.log(l_pdf)

            return e_grad(log_pdf)(x_)

        # Define data
        coords = np.meshgrid(*[np.linspace(-7.5, 7.5) for _ in range(dimension)])
        x_stacked = np.vstack([c.ravel() for c in coords]).T
        true_score_result = true_score(x_stacked)

        # Define a kernel density matching object
        kernel_density_matcher = coreax.score_matching.KernelDensityMatching(
            length_scale=10.0, kde_data=samples
        )

        # Extract the score function (this is not really learned from the data, more
        # defined within the object)
        learned_score = kernel_density_matcher.match()
        score_result = learned_score(x_stacked)

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.5)

    # pylint: enable=too-many-locals


class TestSlicedScoreMatching(unittest.TestCase):
    """
    Tests related to the class SlicedScoreMatching in score_matching.py.
    """

    def setUp(self):
        self.random_key = random.key(0)

    def test_analytic_objective_orthogonal(self) -> None:
        r"""
        Test the core objective function, analytic version.

        We consider two orthogonal vectors, ``u`` and ``v``, and a score vector of ones.
        The analytic objective is given by:

        .. math::

            v' u + 0.5 * ||s||^2

        In the case of ``v`` and ``u`` being orthogonal, this reduces to:

        .. math::

            0.5 * ||s||^2

        which equals 1.0 in the case of ``s`` being a vector of ones.
        """
        # Define data
        u = np.array([0.0, 1.0])
        v = np.array([[1.0, 0.0]])
        s = np.ones(2, dtype=float)

        # Define expected output - orthogonal u and v vectors should give back
        # half-length squared s
        expected_output = 1.0

        # Define a sliced score matching object - with the analytic objective
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            self.random_key, random_generator=random.rademacher, use_analytic=True
        )

        # Evaluate the analytic objective function
        # Disable pylint warning for protected-access as we are testing a single part of
        # the over-arching algorithm
        # pylint: disable=protected-access
        output = sliced_score_matcher._objective_function(v, u, s)
        # pylint: enable=protected-access

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

        which evaluates to 7456.0 when substituting in the given values of ``u``, ``v``
        and ``s``.
        """
        # Define data
        u = np.arange(3, dtype=float)
        v = np.arange(3, 6, dtype=float)
        s = np.arange(9, 12, dtype=float)

        # Define expected outputs
        expected_output_analytic = 165.0
        expected_output_general = 7456.0

        # Define a sliced score matching object - with the analytic objective
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            self.random_key, random_generator=random.rademacher, use_analytic=True
        )

        # Evaluate the analytic objective function
        # Disable pylint warning for protected-access as we are testing a single part of
        # the over-arching algorithm
        # pylint: disable=protected-access
        output = sliced_score_matcher._objective_function(v, u, s)
        # pylint: enable=protected-access

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output_analytic, places=3)

        # Mutate the objective, and check that the result changes
        sliced_score_matcher.use_analytic = False
        # Disable pylint warning for protected-access as we are testing a single part of
        # the over-arching algorithm
        # pylint: disable=protected-access
        output = sliced_score_matcher._objective_function(v, u, s)
        # pylint: enable=protected-access

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

        We consider orthogonal vectors ``v`` and ``u``, meaning we only evaluate the
        second term to get the expected output.
        """
        # Define data - orthogonal u and v vectors should give back half squared dot
        # product of v and s
        u = np.array([0.0, 1.0])
        v = np.array([[1.0, 0.0]])
        s = np.ones(2, dtype=float)

        # Define expected outputs
        expected_output = 0.5

        # Define a sliced score matching object - with the non-analytic objective
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            self.random_key, random_generator=random.rademacher, use_analytic=False
        )

        # Evaluate the analytic objective function
        # Disable pylint warning for protected-access as we are testing a single part of
        # the over-arching algorithm
        # pylint: disable=protected-access
        output = sliced_score_matcher._objective_function(v, u, s)
        # pylint: enable=protected-access

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
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            self.random_key, random_generator=random.rademacher, use_analytic=False
        )

        # Evaluate the analytic objective function
        # Disable pylint warning for protected-access as we are testing a single part of
        # the over-arching algorithm
        # pylint: disable=protected-access
        output = sliced_score_matcher._objective_function(v, u, s)
        # pylint: enable=protected-access

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=3)

    def test_sliced_score_matching_loss_element_analytic(self) -> None:
        """
        Test the loss function elementwise.

        We use the analytic loss function in this example.
        """

        def score_function(y: ArrayLike) -> ArrayLike:
            """
            Basic score function, implicitly multivariate vector valued.

            :param y: point at which to evaluate the score function
            :return: score function (gradient of log density) evaluated at ``y``
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
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            self.random_key, random_generator=random.rademacher, use_analytic=True
        )

        # Determine the expected output - using the analytic objective function tested
        # elsewhere
        # Disable pylint warning for protected-access as we are testing a single part of
        # the over-arching algorithm
        # pylint: disable=protected-access
        expected_output = sliced_score_matcher._objective_function(
            random_vector[None, :], hessian @ random_vector, s
        )

        # Evaluate the loss element
        output = sliced_score_matcher._loss_element(x, random_vector, score_function)
        # pylint: enable=protected-access

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=3)

        # Call the loss element with a different objective function, and check that the
        # JIT compilation recognises this change
        sliced_score_matcher.use_analytic = False
        # Disable pylint warning for protected-access as we are testing a single part of
        # the over-arching algorithm
        # pylint: disable=protected-access
        output_changed_objective = sliced_score_matcher._loss_element(
            x, random_vector, score_function
        )
        # pylint: enable=protected-access
        self.assertNotAlmostEqual(output, output_changed_objective)

    def test_sliced_score_matching_loss_element_general(self) -> None:
        """
        Test the loss function elementwise.

        We use the non-analytic loss function in this example.
        """

        def score_function(x_: ArrayLike) -> ArrayLike:
            """
            Basic score function, implicitly multivariate vector valued.

            :param x_: point at which to evaluate the score function
            :return: score function (gradient of log density) evaluated at ``x_``
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
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            self.random_key, random_generator=random.rademacher, use_analytic=False
        )

        # Determine the expected output
        # Disable pylint warning for protected-access as we are testing a single part of
        # the over-arching algorithm
        # pylint: disable=protected-access
        expected_output = sliced_score_matcher._objective_function(
            random_vector, hessian @ random_vector, s
        )

        # Evaluate the loss element
        output = sliced_score_matcher._loss_element(x, random_vector, score_function)
        # pylint: enable=protected-access

        # Check output matches expected
        self.assertAlmostEqual(output, expected_output, places=3)

    def test_sliced_score_matching_loss(self) -> None:
        """
        Test the loss function with vmap.
        """

        def score_function(x_: ArrayLike) -> ArrayLike:
            """
            Basic score function, implicitly multivariate vector valued.

            :param x_: point at which to evaluate the score function
            :return: score function (gradient of log density) evaluated at ``x_``
            """
            return x_**2

        # Define an arbitrary input
        x = np.tile(np.array([2.0, 7.0]), (10, 1))

        # Define arbitrary number of random vectors, 1 per input
        random_vectors = np.ones((10, 1, 2), dtype=float)

        # Set expected output
        expected_output = np.ones((10, 1), dtype=float) * 1226.5

        # Define a sliced score matching object
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            self.random_key, random_generator=random.rademacher, use_analytic=True
        )
        # Disable pylint warning for protected-access as we are testing a single part of
        # the over-arching algorithm
        # pylint: disable=protected-access
        output = sliced_score_matcher._loss(score_function)(x, random_vectors)
        # pylint: enable=protected-access

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    # pylint: disable=too-many-locals
    def test_train_step(self) -> None:
        """
        Test the basic training step.
        """
        # Define a simple linear model that we can compute the gradients for by hand
        score_network = SimpleNetwork(2, 2)
        score_key, state_key = random.split(self.random_key)

        # Define a sliced score matching object
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=random.rademacher,
            use_analytic=True,
        )

        # Create a train state. setting the PRNG with fixed seed means initialisation is
        # consistent for testing using SGD
        state = coreax.networks.create_train_state(
            state_key, score_network, 1e-3, 2, sgd
        )

        # Jax is row-based, so we have to work with the kernel transpose
        # Disable pylint warning for unsubscriptable-object as we are able to
        # subscript this and use this for testing purposes only
        # pylint: disable=unsubscriptable-object
        weights = state.params["Dense_0"]["kernel"].T
        bias = state.params["Dense_0"]["bias"]
        # pylint: enable=unsubscriptable-object

        # Define input data
        x = np.array([2.0, 7.0])
        v = np.ones((1, 2), dtype=float)
        s = weights @ x.T + bias

        # Reformat for the vector mapped input to loss
        x_to_vector_map = np.array([x])
        v_to_vector_map = np.ones((1, 1, 2), dtype=float)

        # Compute these gradients by hand
        grad_weights = jnp.outer(v, v) + jnp.outer(s, x)
        grad_bias = s

        weights_ = weights - 1e-3 * grad_weights
        bias_ = bias - 1e-3 * grad_bias

        # Disable pylint warning for protected-access as we are testing a single part of
        # the over-arching algorithm
        # pylint: disable=protected-access
        state, _ = sliced_score_matcher._train_step(
            state, x_to_vector_map, v_to_vector_map
        )
        # pylint: enable=protected-access

        # Jax is row based, so transpose W_
        np.testing.assert_array_almost_equal(
            weights_.T, state.params["Dense_0"]["kernel"], decimal=3
        )
        np.testing.assert_array_almost_equal(
            bias_, state.params["Dense_0"]["bias"], decimal=3
        )

    # pylint: enable=too-many-locals

    def test_univariate_gaussian_score(self):
        """
        Test a simple univariate Gaussian with a known score function.
        """
        # Setup univariate Gaussian
        mu = 0.0
        std_dev = 1.0
        num_data_points = 500
        generator = np.random.default_rng(1_989)
        samples = generator.normal(mu, std_dev, size=(num_data_points, 1))

        def true_score(x_: ArrayLike) -> ArrayLike:
            return -(x_ - mu) / std_dev**2

        # Define data
        x = np.linspace(-2, 2).reshape(-1, 1)
        true_score_result = true_score(x)

        # Define a sliced score matching object
        score_key, _ = random.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=random.rademacher,
            use_analytic=True,
        )

        # Extract the score function
        learned_score = sliced_score_matcher.match(samples)
        score_result = learned_score(x)

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.5)

    # pylint: disable=too-many-locals
    def test_multivariate_gaussian_score(self) -> None:
        """
        Test a simple multivariate Gaussian with a known score function.
        """
        # Setup multivariate Gaussian
        dimension = 2
        mu = np.zeros(dimension)
        sigma_matrix = np.eye(dimension)
        lambda_matrix = np.linalg.pinv(sigma_matrix)
        num_data_points = 500
        generator = np.random.default_rng(1_989)
        samples = generator.multivariate_normal(mu, sigma_matrix, size=num_data_points)

        def true_score(x_: ArrayLike) -> ArrayLike:
            return np.array(list(map(lambda z: -lambda_matrix @ (z - mu), x_)))

        # Define data
        x, y = np.meshgrid(np.linspace(-2, 2), np.linspace(-2, 2))
        data_stacked = np.vstack([x.ravel(), y.ravel()]).T
        true_score_result = true_score(data_stacked)

        # Define a sliced score matching object
        score_key, _ = random.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=random.rademacher,
            use_analytic=True,
        )

        # Extract the score function
        learned_score = sliced_score_matcher.match(samples)
        score_result = learned_score(data_stacked)

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.75)

    # pylint: enable=too-many-locals

    # pylint: disable=too-many-locals
    def test_univariate_gmm_score(self):
        """
        Test a univariate Gaussian mixture model with a known score function.
        """
        # Define the univariate Gaussian mixture model
        mus = np.array([-4.0, 4.0])
        std_devs = np.array([1.0, 2.0])
        p = 0.7
        mix = np.array([1 - p, p])
        num_data_points = 1000
        generator = np.random.default_rng(1_989)
        comp = generator.binomial(1, p, size=num_data_points)
        samples = generator.normal(mus[comp], std_devs[comp]).reshape(-1, 1)

        def e_grad(g: Callable) -> Callable:
            def wrapped(x_, *rest):
                y, g_vjp = vjp(lambda x__: g(x__, *rest), x_)
                (x_bar,) = g_vjp(np.ones_like(y))
                return x_bar

            return wrapped

        def true_score(x_: ArrayLike) -> ArrayLike:
            def log_pdf(y_: ArrayLike) -> ArrayLike:
                return jnp.log(norm.pdf(y_, mus, std_devs) @ mix)

            return e_grad(log_pdf)(x_)

        # Define data
        x = np.linspace(-5, 5).reshape(-1, 1)
        true_score_result = true_score(x)

        # Define a sliced score matching object
        score_key, _ = random.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=random.rademacher,
            use_analytic=True,
            num_epochs=50,
        )

        # Extract the score function
        learned_score = sliced_score_matcher.match(samples)
        score_result = learned_score(x)

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.5)

    # pylint: enable=too-many-locals

    # pylint: disable=too-many-locals
    def test_multivariate_gmm_score(self):
        """
        Test a multivariate Gaussian mixture model with a known score function.
        """
        # Define the multivariate Gaussian mixture model (we don't want to go much
        # higher than dimension=2)
        dimension = 2
        k = 10
        num_data_points = 500
        generator = np.random.default_rng(0)
        mus = generator.multivariate_normal(
            np.zeros(dimension), np.eye(dimension), size=k
        )
        sigmas = np.array(
            [generator.gamma(2.0, 1.0) * np.eye(dimension) for _ in range(k)]
        )
        mix = generator.dirichlet(np.ones(k))
        comp = generator.choice(k, size=num_data_points, p=mix)
        samples = np.array(
            [generator.multivariate_normal(mus[c], sigmas[c]) for c in comp]
        )

        def e_grad(g: Callable) -> Callable:
            def wrapped(x_, *rest):
                y, g_vjp = vjp(lambda x__: g(x__, *rest), x_)
                (x_bar,) = g_vjp(np.ones_like(y))
                return x_bar

            return wrapped

        def true_score(x_: ArrayLike) -> ArrayLike:
            def log_pdf(y: ArrayLike) -> ArrayLike:
                l_pdf = 0.0
                for k_ in range(k):
                    l_pdf += multivariate_normal.pdf(y, mus[k_], sigmas[k_]) * mix[k_]
                return jnp.log(l_pdf)

            return e_grad(log_pdf)(x_)

        # Define data
        coords = np.meshgrid(*[np.linspace(-7.5, 7.5) for _ in range(dimension)])
        x_stacked = np.vstack([c.ravel() for c in coords]).T
        true_score_result = true_score(x_stacked)

        # Define a sliced score matching object
        score_key, _ = random.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=random.rademacher,
            use_analytic=True,
            num_epochs=50,
        )

        # Extract the score function
        learned_score = sliced_score_matcher.match(samples)
        score_result = learned_score(x_stacked)

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.75)

    # pylint: enable=too-many-locals

    def test_sliced_score_matching_unexpected_num_random_vectors(self):
        """
        Test how SlicedScoreMatching handles unexpected inputs for num_random_vectors.
        """
        # Define example data
        mu = 0.0
        std_dev = 1.0
        num_data_points = 500
        generator = np.random.default_rng(1_989)
        samples = generator.normal(mu, std_dev, size=(num_data_points, 1))

        # Define a sliced score matching object with num_random_vectors set to 0. This
        # should get capped to a minimum of 1
        score_key, _ = random.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=random.rademacher,
            num_random_vectors=0,
            num_epochs=1,
            num_noise_models=1,
        )
        self.assertEqual(sliced_score_matcher.num_random_vectors, 1)

        # Define a sliced score matching object with num_random_vectors set to -4. This
        # should get capped to a minimum of 1
        score_key, _ = random.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=random.rademacher,
            num_random_vectors=-4,
            num_epochs=1,
            num_noise_models=1,
        )
        self.assertEqual(sliced_score_matcher.num_random_vectors, 1)

        # Define a sliced score matching object with num_random_vectors set to a float.
        # This should give rise to an error when indexing arrays with a float.
        score_key, _ = random.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=random.rademacher,
            num_random_vectors=1.0,
            num_epochs=1,
            num_noise_models=1,
        )
        with self.assertRaises(ValueError) as error_raised:
            sliced_score_matcher.match(samples)

        self.assertEqual(
            error_raised.exception.args[0],
            "num_random_vectors must be an integer",
        )

    def test_sliced_score_matching_unexpected_num_epochs(self):
        """
        Test how SlicedScoreMatching handles unexpected inputs for num_epochs.
        """
        # Define example data
        mu = 0.0
        std_dev = 1.0
        num_data_points = 500
        generator = np.random.default_rng(1_989)
        samples = generator.normal(mu, std_dev, size=(num_data_points, 1))

        # Define a sliced score matching object with num_epochs set to 0. This should
        # just give a randomly initialised neural network.
        score_key, _ = random.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=random.rademacher,
            num_epochs=0,
        )
        learned_score = sliced_score_matcher.match(samples)
        self.assertEqual(len(learned_score(samples)), num_data_points)

        # Define a sliced score matching object with num_epochs set to -5. This should
        # raise an error as we try to create an array with a negative dimension.
        score_key, _ = random.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=random.rademacher,
            num_epochs=-5,
        )
        with self.assertRaises(ValueError) as error_raised:
            sliced_score_matcher.match(samples)

        self.assertEqual(
            error_raised.exception.args[0],
            "num_epochs must be a positive integer",
        )

        # Define a sliced score matching object with num_epochs set to a float. This
        # should raise an error as we try to create an array with a non-integer
        # dimension.
        score_key, _ = random.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=random.rademacher,
            num_epochs=5.0,
        )
        with self.assertRaises(TypeError) as error_raised:
            sliced_score_matcher.match(samples)

        self.assertEqual(
            error_raised.exception.args[0],
            "num_epochs must be a positive integer",
        )

    def test_sliced_score_matching_unexpected_batch_size(self):
        """
        Test how SlicedScoreMatching handles unexpected inputs for batch_size.
        """
        # Define example data
        mu = 0.0
        std_dev = 1.0
        num_data_points = 500
        generator = np.random.default_rng(1_989)
        samples = generator.normal(mu, std_dev, size=(num_data_points, 1))

        # Define a sliced score matching object with batch_size set to 0. This should
        # just give a randomly initialised neural network that has not been updated.
        score_key, _ = random.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=random.rademacher,
            num_epochs=1,
            batch_size=0,
        )
        learned_score = sliced_score_matcher.match(samples)
        self.assertEqual(len(learned_score(samples)), num_data_points)

        # Define a sliced score matching object with batch_size set to -5. This should
        # raise an error as we try to create an array with a negative dimension.
        score_key, _ = random.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=random.rademacher,
            num_epochs=1,
            batch_size=-5,
        )
        with self.assertRaises(ValueError) as error_raised:
            sliced_score_matcher.match(samples)

        self.assertEqual(
            error_raised.exception.args[0],
            "batch_size must be a positive integer",
        )

        # Define a sliced score matching object with batch_size set to a float. This
        # should raise an error as we try to create an array with a non-integer
        # dimension.
        score_key, _ = random.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=random.rademacher,
            num_epochs=1,
            batch_size=5.0,
        )
        with self.assertRaises(TypeError) as error_raised:
            sliced_score_matcher.match(samples)

        self.assertEqual(
            error_raised.exception.args[0],
            "batch_size must be a positive integer",
        )

    def test_sliced_score_matching_unexpected_num_noise_models(self):
        """
        Test how SlicedScoreMatching handles unexpected inputs for num_noise_models.
        """
        # Define example data
        mu = 0.0
        std_dev = 1.0
        num_data_points = 500
        generator = np.random.default_rng(1_989)
        samples = generator.normal(mu, std_dev, size=(num_data_points, 1))

        # Define a sliced score matching object with num_noise_models set to 0. This
        # should get capped at 1 to allow the code to function.
        score_key, _ = random.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=random.rademacher,
            num_epochs=1,
            num_noise_models=0,
        )
        self.assertEqual(sliced_score_matcher.num_noise_models, 1)

        # Define a sliced score matching object with num_noise_models set to -5. This
        # should get capped at 1 to allow the code to function.
        score_key, _ = random.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=random.rademacher,
            num_epochs=1,
            num_noise_models=-5,
        )
        self.assertEqual(sliced_score_matcher.num_noise_models, 1)

        # Define a sliced score matching object with num_noise_models set to a float.
        # This should raise an error as we try to create an array with a non-integer
        # dimension - but hte internal JAX error is human-readable and the full text
        # highlights the variable in question.
        score_key, _ = random.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=random.rademacher,
            num_epochs=1,
            num_noise_models=5.0,
        )
        with self.assertRaises(TypeError) as error_raised:
            sliced_score_matcher.match(samples)

        # cSpell:disable
        self.assertEqual(
            error_raised.exception.args[0],
            "lower and upper arguments to fori_loop must have equal types, got "
            "int32 and float32",
        )
        # cSpell:enable


if __name__ == "__main__":
    unittest.main()
