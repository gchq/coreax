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
Tests for score matching implementations.

Score matching fits models to data by ensuring the score function of the model matches
the score function of the data. The tests within this file verify that score matching
approaches used produce the expected results on simple examples.
"""

import unittest
from collections.abc import Callable
from unittest.mock import MagicMock

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from flax import linen as nn
from jax.scipy.stats import multivariate_normal, norm
from jaxtyping import Array, ArrayLike
from optax import sgd
from typing_extensions import override

import coreax.networks
import coreax.score_matching
from coreax.kernels import (
    LaplacianKernel,
    LinearKernel,
    PCIMQKernel,
    ScalarValuedKernel,
    SquaredExponentialKernel,
    SteinKernel,
    median_heuristic,
)
from coreax.score_matching import KernelDensityMatching, convert_stein_kernel


class SimpleNetwork(nn.Module):
    """
    A simple neural network for use in testing of sliced score matching.
    """

    num_hidden_dim: int
    num_output_dim: int

    @nn.compact
    @override
    def __call__(self, x: ArrayLike) -> ArrayLike:
        x = nn.Dense(self.num_hidden_dim)(x)
        return x


class TestKernelDensityMatching(unittest.TestCase):
    """
    Tests related to the class in ``score_matching.py``.
    """

    def setUp(self):
        """Set up basic univariate Gaussian data."""
        self.random_key = jr.key(0)
        self.mu = 0.0
        self.std_dev = 1.0
        self.num_data_points = 50
        generator = np.random.default_rng(1_989)
        self.samples = jnp.asarray(
            generator.normal(self.mu, self.std_dev, size=(self.num_data_points, 1))
        )

    def test_univariate_gaussian_score(self) -> None:
        """
        Test a simple univariate Gaussian with a known score function.
        """

        def true_score(x_: ArrayLike) -> ArrayLike:
            return -(x_ - self.mu) / self.std_dev**2

        # Define data
        x = jnp.linspace(-2, 2).reshape(-1, 1)
        true_score_result = true_score(x)

        # Define a kernel density matching object
        kernel_density_matcher = KernelDensityMatching(
            length_scale=median_heuristic(self.samples).item()
        )

        # Extract the score function (this is not really learned from the data, more
        # defined within the object)
        learned_score = kernel_density_matcher.match(self.samples)
        score_result = learned_score(x)

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.5)

    def test_univariate_gaussian_score_1d_input(self) -> None:
        """
        Test a simple univariate Gaussian with a known score function, 1D input.
        """

        def true_score(x_: ArrayLike) -> ArrayLike:
            return -(x_ - self.mu) / self.std_dev**2

        # Define data and select a specific single point to test
        test_index = 20
        x = jnp.linspace(-2, 2).reshape(-1, 1)
        true_score_result = true_score(x[test_index, 0])

        # Define a kernel density matching object
        kernel_density_matcher = KernelDensityMatching(
            length_scale=median_heuristic(self.samples).item()
        )

        # Extract the score function (this is not really learned from the data, more
        # defined within the object)
        learned_score = kernel_density_matcher.match(self.samples)
        score_result = learned_score(x[test_index, 0])

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
        generator = np.random.default_rng(1_989)
        samples = jnp.asarray(
            generator.multivariate_normal(mu, sigma_matrix, size=self.num_data_points)
        )

        def true_score(x_: Array) -> Array:
            return jnp.array(list(map(lambda z: -lambda_matrix @ (z - mu), x_)))

        # Define data
        x, y = jnp.meshgrid(np.linspace(-2, 2), np.linspace(-2, 2))
        data_stacked = jnp.vstack([x.ravel(), y.ravel()]).T
        true_score_result = true_score(data_stacked)

        # Define a kernel density matching object
        kernel_density_matcher = KernelDensityMatching(
            length_scale=median_heuristic(samples).item()
        )

        # Extract the score function (this is not really learned from the data, more
        # defined within the object)
        learned_score = kernel_density_matcher.match(samples)
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
        generator = np.random.default_rng(1_989)
        comp = generator.binomial(1, p, size=self.num_data_points)
        samples = jnp.asarray(
            generator.normal(mus[comp], std_devs[comp]).reshape(-1, 1)
        )

        def e_grad(g: Callable) -> Callable:
            def wrapped(x_, *rest):
                y, g_vjp = jax.vjp(lambda x__: g(x__, *rest), x_)
                (x_bar,) = g_vjp(np.ones_like(y))
                return x_bar

            return wrapped

        def true_score(x_: ArrayLike) -> ArrayLike:
            def log_pdf(y_: ArrayLike) -> ArrayLike:
                return jnp.log(norm.pdf(y_, mus, std_devs) @ mix)

            return e_grad(log_pdf)(x_)

        # Define data
        x = jnp.linspace(-5, 5).reshape(-1, 1)
        true_score_result = true_score(x)

        # Define a kernel density matching object
        kernel_density_matcher = KernelDensityMatching(
            length_scale=median_heuristic(samples).item()
        )

        # Extract the score function (this is not really learned from the data, more
        # defined within the object)
        learned_score = kernel_density_matcher.match(samples)
        score_result = learned_score(x)

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.6)

    def test_multivariate_gmm_score(self):
        """
        Test a multivariate Gaussian mixture model with a known score function.
        """
        # Define the multivariate Gaussian mixture model (we don't want to go much
        # higher than dimension=2)
        dimension = 2
        k = 10
        generator = np.random.default_rng(0)
        mus = generator.multivariate_normal(
            np.zeros(dimension), np.eye(dimension), size=k
        )
        sigmas = np.array(
            [generator.gamma(2.0, 1.0) * np.eye(dimension) for _ in range(k)]
        )
        mix = generator.dirichlet(np.ones(k))
        comp = generator.choice(k, size=self.num_data_points, p=mix)
        samples = np.array(
            [generator.multivariate_normal(mus[c], sigmas[c]) for c in comp]
        )

        def e_grad(g: Callable) -> Callable:
            def wrapped(x_, *rest):
                y, g_vjp = jax.vjp(lambda x__: g(x_, *rest), x_)
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
        coords = jnp.meshgrid(*[np.linspace(-7.5, 7.5) for _ in range(dimension)])
        x_stacked = jnp.vstack([c.ravel() for c in coords]).T
        true_score_result = true_score(x_stacked)

        # Define a kernel density matching object
        kernel_density_matcher = KernelDensityMatching(length_scale=10.0)

        # Extract the score function (this is not really learned from the data, more
        # defined within the object)
        learned_score = kernel_density_matcher.match(samples)
        score_result = learned_score(x_stacked)

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.5)

    def test_handling_of_1d_input(self) -> None:
        """
        Verify how the score function behaves when given a one-dimensional input.
        """
        data = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        matcher = KernelDensityMatching(length_scale=1)
        score_function = matcher.match(data)

        # Define data to evaluate the score function at - we consider the same
        # data-point but given in two different shapes
        data_1d = jnp.array([1, 2])
        data_2d = jnp.array([[1, 2]])

        # When we evaluate the score function with a 1 dimensional input, we expect the
        # resulting score function to be 1 dimensional, which ensures compatibility with
        # Stein kernel usage
        np.testing.assert_array_equal(
            score_function(data_1d), jnp.array([0.03597286, 0.03597286])
        )

        # When we evaluate the score function with a 2 dimensional input that holds a
        # single data-point, we expect the resulting score function to be 2 dimensional,
        # holding exactly the same values as above, but with an extra dimension
        np.testing.assert_array_equal(
            score_function(data_2d), jnp.array([[0.03597286, 0.03597286]])
        )


class TestSlicedScoreMatching(unittest.TestCase):
    """
    Tests related to the class SlicedScoreMatching in score_matching.py.
    """

    def setUp(self):
        """Set up basic univariate Gaussian data."""
        self.random_key = jr.key(0)
        self.mu = 0.0
        self.std_dev = 1.0
        self.num_data_points = 250
        generator = np.random.default_rng(1_989)
        self.samples = generator.normal(
            self.mu, self.std_dev, size=(self.num_data_points, 1)
        )

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
        u = jnp.array([0.0, 1.0])
        v = jnp.array([[1.0, 0.0]])
        s = jnp.ones(2, dtype=float)

        # Define expected output - orthogonal u and v vectors should give back
        # half-length squared s
        expected_output = 1.0

        # Define a sliced score matching object - with the analytic objective
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            self.random_key, random_generator=jr.rademacher, use_analytic=True
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
        u = jnp.arange(3, dtype=float)
        v = jnp.arange(3, 6, dtype=float)
        s = jnp.arange(9, 12, dtype=float)

        # Define expected outputs
        expected_output_analytic = 165.0
        expected_output_general = 7456.0

        # Define a sliced score matching object - with the analytic objective
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            self.random_key, random_generator=jr.rademacher, use_analytic=True
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
        sliced_score_matcher = eqx.tree_at(
            lambda x: x.use_analytic, sliced_score_matcher, False
        )

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
        u = jnp.array([0.0, 1.0])
        v = jnp.array([[1.0, 0.0]])
        s = jnp.ones(2, dtype=float)

        # Define expected outputs
        expected_output = 0.5

        # Define a sliced score matching object - with the non-analytic objective
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            self.random_key, random_generator=jr.rademacher, use_analytic=False
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
        u = jnp.arange(3, dtype=float)
        v = jnp.arange(3, 6, dtype=float)
        s = jnp.arange(9, 12, dtype=float)

        # Define expected outputs
        expected_output = 7456.0

        # Define a sliced score matching object - with the non-analytic objective
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            self.random_key, random_generator=jr.rademacher, use_analytic=False
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

        def score_function(y: Array) -> Array:
            """
            Score function, implicitly multivariate vector valued.

            :param y: point at which to evaluate the score function
            :return: score function (gradient of log density) evaluated at ``y``
            """
            return y**2

        # Define an arbitrary input
        x = jnp.array([2.0, 7.0])
        s = score_function(x)

        # Defined the Hessian (grad of score function)
        hessian = 2.0 * jnp.diag(x)

        # Define some arbitrary random vector
        random_vector = jnp.ones(2, dtype=float)

        # Define a sliced score matching object
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            self.random_key, random_generator=jr.rademacher, use_analytic=True
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
        sliced_score_matcher = eqx.tree_at(
            lambda x: x.use_analytic, sliced_score_matcher, False
        )

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

        def score_function(x_: Array) -> Array:
            """
            Score function, implicitly multivariate vector valued.

            :param x_: point at which to evaluate the score function
            :return: score function (gradient of log density) evaluated at ``x_``
            """
            return x_**2

        # Define an arbitrary input
        x = jnp.array([2.0, 7.0])
        s = score_function(x)

        # Defined the Hessian (grad of score function)
        hessian = 2.0 * jnp.diag(x)

        # Define some arbitrary random vector
        random_vector = jnp.ones(2, dtype=float)

        # Define a sliced score matching object
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            self.random_key, random_generator=jr.rademacher, use_analytic=False
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
            Score function, implicitly multivariate vector valued.

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
            self.random_key, random_generator=jr.rademacher, use_analytic=True
        )
        # Disable pylint warning for protected-access as we are testing a single part of
        # the over-arching algorithm
        # pylint: disable=protected-access
        output = sliced_score_matcher._loss(score_function)(x, random_vectors)
        # pylint: enable=protected-access

        # Check output matches expected
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    def test_train_step(self) -> None:
        """
        Test the basic training step.
        """
        # Define a simple linear model that we can compute the gradients for by hand
        score_network = SimpleNetwork(2, 2)
        score_key, state_key = jr.split(self.random_key)

        # Define a sliced score matching object
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=jr.rademacher,
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
        weights = state.params["Dense_0"]["kernel"].T  # pyright: ignore[reportAttributeAccessIssue]
        bias = state.params["Dense_0"]["bias"]
        # pylint: enable=unsubscriptable-object

        # Define input data
        x = jnp.array([2.0, 7.0])
        v = jnp.ones((1, 2), dtype=float)
        s = weights @ x.T + bias

        # Reformat for the vector mapped input to loss
        x_to_vector_map = jnp.array([x])
        v_to_vector_map = jnp.ones((1, 1, 2), dtype=float)

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
            state.params["Dense_0"]["kernel"],  # pyright: ignore[reportArgumentType]
            weights_.T,
            decimal=3,
        )
        np.testing.assert_array_almost_equal(
            state.params["Dense_0"]["bias"],  # pyright: ignore[reportArgumentType]
            bias_,
            decimal=3,
        )

    def test_univariate_gaussian_score(self):
        """
        Test a simple univariate Gaussian with a known score function.
        """

        def true_score(x_: Array) -> Array:
            return -(x_ - self.mu) / self.std_dev**2

        # Define data
        x = jnp.linspace(-2, 2).reshape(-1, 1)
        true_score_result = true_score(x)

        # Define a sliced score matching object
        score_key, _ = jr.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=jr.rademacher,
            use_analytic=True,
        )

        # Extract the score function
        learned_score = sliced_score_matcher.match(self.samples)
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
        generator = np.random.default_rng(1_989)
        samples = generator.multivariate_normal(
            mu, sigma_matrix, size=self.num_data_points
        )

        def true_score(x_: Array) -> Array:
            return jnp.array(list(map(lambda z: -lambda_matrix @ (z - mu), x_)))

        # Define data
        x, y = jnp.meshgrid(np.linspace(-2, 2), np.linspace(-2, 2))
        data_stacked = jnp.vstack([x.ravel(), y.ravel()]).T
        true_score_result = true_score(data_stacked)

        # Define a sliced score matching object
        score_key, _ = jr.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=jr.rademacher,
            use_analytic=True,
        )

        # Extract the score function
        learned_score = sliced_score_matcher.match(samples)
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
        generator = np.random.default_rng(1_989)
        comp = generator.binomial(1, p, size=self.num_data_points)
        samples = generator.normal(mus[comp], std_devs[comp]).reshape(-1, 1)

        def e_grad(g: Callable) -> Callable:
            def wrapped(x_, *rest):
                y, g_vjp = jax.vjp(lambda x__: g(x__, *rest), x_)
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
        score_key, _ = jr.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=jr.rademacher,
            use_analytic=True,
            num_epochs=20,
        )

        # Extract the score function
        learned_score = sliced_score_matcher.match(samples)
        score_result = learned_score(x)

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.75)

    def test_multivariate_gmm_score(self):
        """
        Test a multivariate Gaussian mixture model with a known score function.
        """
        # Define the multivariate Gaussian mixture model (we don't want to go much
        # higher than dimension=2)
        dimension = 2
        k = 10
        generator = np.random.default_rng(0)
        mus = generator.multivariate_normal(
            np.zeros(dimension), np.eye(dimension), size=k
        )
        sigmas = np.array(
            [generator.gamma(2.0, 1.0) * np.eye(dimension) for _ in range(k)]
        )
        mix = generator.dirichlet(np.ones(k))
        comp = generator.choice(k, size=self.num_data_points, p=mix)
        samples = np.array(
            [generator.multivariate_normal(mus[c], sigmas[c]) for c in comp]
        )

        def e_grad(g: Callable) -> Callable:
            def wrapped(x_, *rest):
                y, g_vjp = jax.vjp(lambda x__: g(x__, *rest), x_)
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
        score_key, _ = jr.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key,
            random_generator=jr.rademacher,
            use_analytic=True,
            num_epochs=5,
        )

        # Extract the score function
        learned_score = sliced_score_matcher.match(samples)
        score_result = learned_score(x_stacked)

        # Check learned score and true score align
        self.assertLessEqual(np.abs(true_score_result - score_result).mean(), 0.8)

    def test_match_no_noise_conditioning(self):
        """Test  match does not raise an error with no 'noise_conditioning'."""
        score_key, _ = jr.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key, random_generator=jr.rademacher, noise_conditioning=False
        )
        sliced_score_matcher.match(self.samples)

    def test_match_zero_epochs_and_batch_size(self):
        """Test 'match' with zero valued 'num_epochs' and 'batch_size'."""
        score_key, _ = jr.split(self.random_key)
        sliced_score_matcher = coreax.score_matching.SlicedScoreMatching(
            score_key, random_generator=jr.rademacher, num_epochs=0, batch_size=0
        )
        sliced_score_matcher.match(self.samples)

    def test_check_init(self):
        """Test the `__check_init__` magic of `SlicedScoreMatching`."""
        score_key, _ = jr.split(self.random_key)
        # Test non-negative integer attributes
        coreax.score_matching.SlicedScoreMatching(
            score_key, random_generator=jr.rademacher, num_epochs=0, batch_size=0
        )
        for i in (-1, 1.0):
            with pytest.raises(
                ValueError, match="'num_epochs' must be a non-negative integer"
            ):
                coreax.score_matching.SlicedScoreMatching(
                    score_key,
                    random_generator=jr.rademacher,
                    num_epochs=i,  # type: ignore[reportArgumentType]
                )
            with pytest.raises(
                ValueError, match="'batch_size' must be a non-negative integer"
            ):
                coreax.score_matching.SlicedScoreMatching(
                    score_key,
                    random_generator=jr.rademacher,
                    batch_size=i,  # type: ignore[reportArgumentType]
                )
        # Test positive integer attributes
        for i in (-1, 0, 1.0):
            with pytest.raises(
                ValueError, match="'num_random_vectors' must be a positive integer"
            ):
                coreax.score_matching.SlicedScoreMatching(
                    score_key,
                    random_generator=jr.rademacher,
                    num_random_vectors=i,  # type: ignore[reportArgumentType]
                )
            with pytest.raises(
                ValueError, match="'num_noise_models' must be a positive integer"
            ):
                coreax.score_matching.SlicedScoreMatching(
                    score_key,
                    random_generator=jr.rademacher,
                    num_noise_models=i,  # type: ignore[reportArgumentType]
                )


class TestConvertSteinKernel:
    """Tests related to the function convert_stein_kernel in score_matching.py."""

    @pytest.mark.parametrize("score_matching", [None, MagicMock()])
    @pytest.mark.parametrize(
        "kernel",
        [LinearKernel(), SquaredExponentialKernel(), LaplacianKernel(), PCIMQKernel()],
    )
    def test_convert_stein_kernel(
        self,
        score_matching: MagicMock | KernelDensityMatching | None,
        kernel: ScalarValuedKernel,
    ) -> None:
        """Check handling of Stein kernels and standard kernels is consistent."""
        random_key = jr.key(2_024)
        dataset_shape = (50, 2)
        dataset = jr.uniform(random_key, dataset_shape)

        converted_kernel = convert_stein_kernel(dataset, kernel, score_matching)
        if isinstance(kernel, SteinKernel):
            if score_matching is not None:
                expected_kernel = eqx.tree_at(
                    lambda x: x.score_function,
                    kernel,
                    score_matching.match(dataset),
                )
            else:
                expected_kernel = kernel
        else:
            if score_matching is None:
                length_scale = getattr(kernel, "length_scale", 1.0)
                score_matching = KernelDensityMatching(length_scale)
            expected_kernel = SteinKernel(kernel, score_matching.match(dataset))

        assert eqx.tree_equal(converted_kernel.base_kernel, expected_kernel.base_kernel)
        # Score function hashes won't match; resort to checking identical
        # evaluation.
        assert eqx.tree_equal(
            converted_kernel.score_function(dataset),
            expected_kernel.score_function(dataset),
        )
