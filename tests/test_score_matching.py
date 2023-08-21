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
        # np.testing.assert_almost_equal(out, ans, decimal=3)

    def test_train_step(self):
        score_network = TestNetwork(2, 2)
        # setting the PRNG with fixed seed means initialisation is consistent for testing
        state = create_train_state(score_network, random.PRNGKey(0), 1e-3, 2, sgd)
        # assert that we know the initialisation from the PRNG
        W = jnp.array([[-0.27880254, -0.7407797], [-0.47987297,  0.2552868]])
        self.assertAlmostEqual(jnp.linalg.norm(state.params['Dense_0']['kernel'] - W), 0., 3)

        x = np.array([[2., 7.]])
        v = np.ones((1, 1, 2), dtype=float)
        s = W @ x[0, :]

        # to write
        grad_L = None
        new_params = W - 1e-3 * grad_L

        def loss(params): return sliced_score_matching_loss(
            lambda x: state.apply_fn({'params': params}, x), analytic_obj)(x, v).mean()
        print(loss(state.params))
        pass

    def test_sliced_score_matching(self):
        pass

    def test_univariate_gaussian_score(self):
        pass

    def test_multivariate_gaussian_score(self):
        pass

    def test_univariate_gmm_score(self):
        pass

    def test_multivariate_gmm_score(self):
        pass


if __name__ == "__main__":
    unittest.main()
