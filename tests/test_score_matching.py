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
import numpy as np
from jax import random, numpy as jnp
from jax.typing import ArrayLike
from typing import Callable
from jax.scipy.stats import norm, multivariate_normal
from optax import sgd

from coreax.score_matching import *
from flax import linen as nn


class TestNetwork(nn.Module):
    """A network for use in sliced score matching."""

    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x: ArrayLike) -> ArrayLike:
        x = nn.Dense(self.hidden_dim)(x)
        return x


class TestScoreMatching(unittest.TestCase):
    def test_analytic_objective(self) -> None:
        """Tests the core objective function, analytic version"""
        # orthogonal u and v vectors should give back half length squared s
        u = np.array([0.0, 1.0])
        v = np.array([1.0, 0.0])
        s = np.ones(2, dtype=float)
        out = analytic_obj(v, u, s)
        ans = 1.0
        self.assertAlmostEqual(out, ans, places=3)

        # basic test
        u = np.arange(3, dtype=float)
        v = np.arange(3, 6, dtype=float)
        s = np.arange(9, 12, dtype=float)
        out = analytic_obj(v, u, s)
        ans = 165.0
        self.assertAlmostEqual(out, ans, places=3)

    def test_general_objective(self) -> None:
        """Tests the core objective function."""
        # orthogonal u and v vectors should give back half squared dot product of v and
        # s
        u = np.array([0.0, 1.0])
        v = np.array([1.0, 0.0])
        s = np.ones(2, dtype=float)
        out = general_obj(v, u, s)
        ans = 0.5
        self.assertAlmostEqual(out, ans, places=3)

        # basic test
        u = np.arange(3, dtype=float)
        v = np.arange(3, 6, dtype=float)
        s = np.arange(9, 12, dtype=float)
        out = general_obj(v, u, s)
        ans = 7456.0
        self.assertAlmostEqual(out, ans, places=3)

    def test_sliced_score_matching_loss_element(self) -> None:
        """Tests the loss function elementwise."""

        def score_fn(x: ArrayLike) -> ArrayLike:
            """
            Basic score function, implicitly multivariate vector valued.

            :param x: point at which to evaluate the score function
            :return: score function (gradient of log density) evaluated at x
            """
            return x**2

        # arbitrary input
        x = np.array([2.0, 7.0])
        s = score_fn(x)
        # Hessian (grad of score function)
        H = 2.0 * np.diag(x)
        # arbitrary random vector
        v = np.ones(2, dtype=float)
        out = sliced_score_matching_loss_element(x, v, score_fn, analytic_obj)
        # assumes analytic_obj has passed test
        ans = analytic_obj(v, H @ v, s)
        self.assertAlmostEqual(out, ans, places=3)

    def test_sliced_score_matching_loss(self) -> None:
        """Tests the loss function vmapped function."""

        #
        def score_fn(x: ArrayLike) -> ArrayLike:
            """
            Basic score function, implicitly multivariate vector valued.

            :param x: point at which to evaluate the score function
            :return: score function (gradient of log density) evaluated at x
            """
            return x**2

        x = np.array([2.0, 7.0])
        # arbitrary number of inputs
        X = np.tile(x, (10, 1))
        # arbitrary number of randoms, 1 per input
        V = np.ones((10, 1, 2), dtype=float)
        out = sliced_score_matching_loss(score_fn, analytic_obj)(X, V)
        ans = np.ones((10, 1), dtype=float) * 1226.5
        self.assertAlmostEqual(np.linalg.norm(out - ans), 0.0, places=3)

    def test_train_step(self) -> None:
        """Tests the basic training step."""
        # simple linear model that we can compute the gradients for by hand
        score_network = TestNetwork(2, 2)

        # setting the PRNG with fixed seed means initialisation is consistent for
        # testing using SGD
        state = create_train_state(score_network, random.PRNGKey(0), 1e-3, 2, sgd)

        # Jax is row-based, so we have to work with the kernel transpose
        W = state.params["Dense_0"]["kernel"].T
        b = state.params["Dense_0"]["bias"]

        x = np.array([2.0, 7.0])
        v = np.ones((1, 2), dtype=float)
        s = W @ x.T + b

        # for the vector mapped input to loss
        X = np.array([x])
        V = np.ones((1, 1, 2), dtype=float)

        # we can compute these gradients by hand
        grad_W = jnp.outer(v, v) + jnp.outer(s, x)
        grad_b = s

        W_ = W - 1e-3 * grad_W
        b_ = b - 1e-3 * grad_b

        state, _ = sliced_score_matching_train_step(state, X, V, analytic_obj)

        # Jax is row based, so transpose W_
        self.assertAlmostEqual(
            np.linalg.norm(W_.T - state.params["Dense_0"]["kernel"]), 0.0, places=3
        )
        self.assertAlmostEqual(
            np.linalg.norm(b_ - state.params["Dense_0"]["bias"]), 0.0, places=3
        )

    def test_univariate_gaussian_score(self):
        """Tests a simple univariate Gaussian known score function."""
        mu = 0.0
        std_dev = 1.0
        N = 500
        np.random.seed(0)
        samples = np.random.normal(mu, std_dev, size=(N, 1))

        def true_score(x: ArrayLike) -> ArrayLike:
            return -(x - mu) / std_dev**2

        x = np.linspace(-2, 2).reshape(-1, 1)
        y_t = true_score(x)

        # noise conditioning
        learned_score = sliced_score_matching(
            samples,
            random.normal,
            hidden_dim=32,
            use_analytic=True,
            epochs=100,
        )
        y_l = learned_score(x)
        mae = np.abs(y_t - y_l).mean()
        self.assertLessEqual(mae, 1.0)

        # no noise conditioning
        learned_score = sliced_score_matching(
            samples,
            random.normal,
            noise_conditioning=False,
            hidden_dim=32,
            use_analytic=True,
            epochs=100,
        )
        y_l = learned_score(x)
        mae = np.abs(y_t - y_l).mean()
        self.assertLessEqual(mae, 1.0)

    def test_multivariate_gaussian_score(self) -> None:
        """Tests a simple multivariate Gaussian known score function."""
        d = 2
        mu = np.zeros(d)
        Sigma = np.eye(d)
        Lambda = np.linalg.pinv(Sigma)
        N = 500
        np.random.seed(0)
        samples = np.random.multivariate_normal(mu, Sigma, size=N)

        def true_score(x: ArrayLike) -> ArrayLike:
            y = np.array(list(map(lambda z: -Lambda @ (z - mu), x)))
            return y

        x, y = np.meshgrid(np.linspace(-2, 2), np.linspace(-2, 2))
        X = np.vstack([x.ravel(), y.ravel()]).T
        y_t = true_score(X)

        # noise conditioning
        learned_score = sliced_score_matching(
            samples,
            random.normal,
            hidden_dim=32,
            use_analytic=True,
            epochs=100,
        )
        y_l = learned_score(X)
        err = np.linalg.norm(y_t - y_l, axis=1).mean()
        self.assertLessEqual(err, 1.0)

        # no noise conditioning
        learned_score = sliced_score_matching(
            samples,
            random.normal,
            noise_conditioning=False,
            hidden_dim=32,
            use_analytic=True,
            epochs=100,
        )
        y_l = learned_score(X)
        err = np.linalg.norm(y_t - y_l, axis=1).mean()
        self.assertLessEqual(err, 1.0)

    def test_univariate_gmm_score(self):
        """Tests a univariate Gaussian mixture model known score function."""
        mus = np.array([-4.0, 4.0])
        std_devs = np.array([1.0, 2.0])
        p = 0.7
        mix = np.array([1 - p, p])
        N = 1000
        np.random.seed(0)
        comp = np.random.binomial(1, p, size=N)
        samples = np.random.normal(mus[comp], std_devs[comp]).reshape(-1, 1)

        def egrad(g: Callable) -> Callable:
            def wrapped(x, *rest):
                y, g_vjp = jax.vjp(lambda x: g(x, *rest), x)
                (x_bar,) = g_vjp(np.ones_like(y))
                return x_bar

            return wrapped

        def true_score(x: ArrayLike) -> ArrayLike:
            logpdf = lambda y: jnp.log(norm.pdf(y, mus, std_devs) @ mix)
            return egrad(logpdf)(x)

        x = np.linspace(-10, 10).reshape(-1, 1)
        y_t = true_score(x)

        # noise conditioning
        learned_score = sliced_score_matching(
            samples,
            random.normal,
            hidden_dim=128,
            use_analytic=True,
            batch_size=128,
            sigma=1.0,
            epochs=100,
        )
        y_l = learned_score(x)
        mae = np.abs(y_t - y_l).mean()
        self.assertLessEqual(mae, 1.0)

        # no noise conditioning
        learned_score = sliced_score_matching(
            samples,
            random.normal,
            noise_conditioning=False,
            hidden_dim=128,
            use_analytic=True,
            batch_size=128,
            epochs=100,
        )
        y_l = learned_score(x)
        mae = np.abs(y_t - y_l).mean()
        self.assertLessEqual(mae, 1.0)

    def test_multivariate_gmm_score(self):
        """Tests a multivariate Gaussian mixture model known socre function."""
        np.random.seed(0)
        # we don't want to go much higher than 2
        d = 2
        K = 10
        mus = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=K)
        Sigmas = np.array([np.random.gamma(2.0, 1.0) * np.eye(d) for _ in range(K)])
        mix = np.random.dirichlet(np.ones(K))
        N = 500
        comp = np.random.choice(K, size=N, p=mix)
        samples = np.array(
            [np.random.multivariate_normal(mus[c], Sigmas[c]) for c in comp]
        )

        def egrad(g: Callable) -> Callable:
            def wrapped(x, *rest):
                y, g_vjp = jax.vjp(lambda x: g(x, *rest), x)
                (x_bar,) = g_vjp(np.ones_like(y))
                return x_bar

            return wrapped

        def true_score(x: ArrayLike) -> ArrayLike:
            def logpdf(y: ArrayLike) -> ArrayLike:
                lpdf = 0.0
                for k in range(K):
                    lpdf += multivariate_normal.pdf(y, mus[k], Sigmas[k]) * mix[k]
                return jnp.log(lpdf)

            return egrad(logpdf)(x)

        coords = np.meshgrid(*[np.linspace(-7.5, 7.5) for _ in range(d)])
        X = np.vstack([c.ravel() for c in coords]).T
        y_t = true_score(X)

        # noise conditioning
        learned_score = sliced_score_matching(
            samples,
            random.normal,
            hidden_dim=128,
            use_analytic=True,
            batch_size=128,
            sigma=1.0,
            epochs=100,
        )
        y_l = learned_score(X)
        err = np.linalg.norm(y_t - y_l, axis=1).mean()
        self.assertLessEqual(err, d)

        # no noise conditioning
        learned_score = sliced_score_matching(
            samples,
            random.normal,
            noise_conditioning=False,
            hidden_dim=128,
            batch_size=128,
            use_analytic=True,
            epochs=100,
        )
        y_l = learned_score(X)
        err = np.linalg.norm(y_t - y_l, axis=1).mean()
        self.assertLessEqual(err, d)
        pass


if __name__ == "__main__":
    unittest.main()
