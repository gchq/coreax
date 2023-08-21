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
import unittest
import numpy as np
from jax import grad, jacfwd, random, numpy as jnp
from optax import sgd

from coreax.score_matching import *
from flax.training.train_state import TrainState
from flax import linen as nn
import matplotlib.pyplot as plt

class TestNetwork(nn.Module):
    """A network for use in sliced score matching."""
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        return x

class TestKernels(unittest.TestCase):

    def test_analytic_objective(self):
        """Tests the core objective function, analytic version
        """
        # orthogonal u and v vectors should give back half length squared s
        u = np.array([0., 1.])
        v = np.array([1., 0.])
        s = np.ones(2, dtype=float)
        out = analytic_obj(v, u, s)
        ans = 1.
        self.assertAlmostEqual(out, ans, places=3)

        # basic test
        u = np.arange(3, dtype=float)
        v = np.arange(3, 6, dtype=float)
        s = np.arange(9, 12, dtype=float)
        out = analytic_obj(v, u, s)
        ans = 165.
        self.assertAlmostEqual(out, ans, places=3)

    def test_general_objective(self):
        """Tests the core objective function.
        """
        # orthogonal u and v vectors should give back half squared dot product of v and s
        u = np.array([0., 1.])
        v = np.array([1., 0.])
        s = np.ones(2, dtype=float)
        out = general_obj(v, u, s)
        ans = .5
        self.assertAlmostEqual(out, ans, places=3)

        # basic test
        u = np.arange(3, dtype=float)
        v = np.arange(3, 6, dtype=float)
        s = np.arange(9, 12, dtype=float)
        out = general_obj(v, u, s)
        ans = 7456.
        self.assertAlmostEqual(out, ans, places=3)

    def test_sliced_score_matching_loss_element(self):
        # basic score function, implictly multivariate vector valued
        score_fn = lambda x: x**2
        # arbitrary input
        x = np.array([2., 7.]) 
        s = score_fn(x)
        # Hessian (grad of score function)
        H = 2. * np.diag(x)
        # arbitrary random vector
        v = np.ones(2, dtype=float)
        out = sliced_score_matching_loss_element(x, v, score_fn, analytic_obj)
        # assumes analytic_obj has passed test
        ans = analytic_obj(v, H @ v, s)
        self.assertAlmostEqual(out, ans, places=3)

    def test_sliced_score_matching_loss(self):
        # basic score function, implictly multivariate vector valued
        score_fn = lambda x: x**2
        x = np.array([2., 7.])
        # arbitrary number of inputs
        X = np.tile(x, (10, 1))
        # arbitrary number of randoms, 1 per input
        V = np.ones((10, 1, 2), dtype=float)
        out = sliced_score_matching_loss(score_fn, analytic_obj)(X, V)
        ans = np.ones((10, 1), dtype=float) * 1226.5
        self.assertAlmostEqual(np.linalg.norm(out - ans), 0., places=3)

    def test_train_step(self):
        # simple linear model that we can compute the gradients for by hand
        score_network = TestNetwork(2, 2)

        # setting the PRNG with fixed seed means initialisation is consistent for testing
        # using SGD
        state = create_train_state(score_network, random.PRNGKey(0), 1e-3, 2, sgd)

        # Jax is row-based, so we have to work with the kernel transpose
        W = state.params['Dense_0']['kernel'].T
        b = state.params['Dense_0']['bias']

        x = np.array([2., 7.])
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

        state = train_step(state, X, V, analytic_obj)

        # Jax is row based, so transpose W_
        self.assertAlmostEqual(np.linalg.norm(W_.T - state.params["Dense_0"]["kernel"]), 0., places=3)
        self.assertAlmostEqual(np.linalg.norm(b_ - state.params["Dense_0"]["bias"]), 0., places=3)

    def test_univariate_gaussian_score(self):
        mu = 6.
        std_dev = 1.6
        N = 1000
        np.random.seed(0)
        samples = np.random.normal(mu, std_dev, size=(N, 1))
        learned_score = sliced_score_matching(samples, random.normal, use_analytic=True, epochs=10)
        true_score = lambda x: -(x - mu) / std_dev**2
        x = np.linspace(-10, 10).reshape(-1, 1)
        y_t = true_score(x)
        y_l = learned_score(x)
        mae = np.abs(y_t - y_l).mean()
        self.assertLessEqual(mae, 1.)

    def test_multivariate_gaussian_score(self):
        pass

    def test_univariate_gmm_score(self):
        pass

    def test_multivariate_gmm_score(self):
        pass


if __name__ == "__main__":
    unittest.main()
