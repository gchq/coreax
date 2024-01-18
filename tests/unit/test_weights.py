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

import cmath
import unittest

import jax.numpy as jnp

import coreax.kernel
import coreax.weights


class TestWeights(unittest.TestCase):
    """
    Tests related to weights.py.
    """

    def test_calculate_BQ_weights(self) -> None:
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

        optimiser = coreax.weights.SBQ(kernel=coreax.kernel.SquaredExponentialKernel())

        # Solve for the weights
        output = optimiser.solve(x, y)

        self.assertTrue(jnp.allclose(output, expected_output))

    def test_simplex_weights(self) -> None:
        r"""
        Test calculation of weights via the simplex method for quadratic programming.

        The simplex_weights() method solves the equation:

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

        optimiser = coreax.weights.MMD(kernel=coreax.kernel.SquaredExponentialKernel())

        # Solve for the weights
        output = optimiser.solve(x, y)

        self.assertTrue(jnp.allclose(output, expected_output, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
