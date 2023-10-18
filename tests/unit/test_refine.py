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

import jax.numpy as jnp
from jax import random, vmap

import coreax.kernel as ck
import coreax.refine as cr


class TestMetrics(unittest.TestCase):
    r"""
    Tests related to refine.py functions.
    """

    def test_refine_ones(self) -> None:
        r"""
        Test that refining an optimal coreset, specified by indices S, returns the same
         indices.
        """

        x = jnp.asarray([[0, 0], [1, 1], [0, 0], [1, 1]])
        n = len(x)
        S = jnp.asarray([0, 1])

        K = vmap(
            vmap(ck.rbf_kernel, in_axes=(None, 0), out_axes=0),
            in_axes=(0, None),
            out_axes=0,
        )(x, x)
        K_mean = K.sum(axis=1) / n

        refine_test = cr.refine(x, S, ck.rbf_kernel, K_mean)
        self.assertCountEqual(refine_test, S)

    def test_refine_ints(self) -> None:
        r"""
        For a toy example, X = [[0,0], [1,1], [2,2]], the 2-point coreset that minimises
        the MMD is specified y the indices S = [0, 2], ie X_c =  [[0,0], [[2,2]].

        Test this example, for a random 2-point coreset, that refine returns [0, 2].
        """

        x = jnp.asarray([[0, 0], [1, 1], [2, 2]])
        n = len(x)

        key = random.PRNGKey(0)
        S1 = random.randint(key, (), 0, 2)
        S2 = random.randint(key, (), 0, 2)
        S = jnp.asarray([S1, S2])

        K = vmap(
            vmap(ck.rbf_kernel, in_axes=(None, 0), out_axes=0),
            in_axes=(0, None),
            out_axes=0,
        )(x, x)
        K_mean = K.sum(axis=1) / n

        refine_test = cr.refine(x, S, ck.rbf_kernel, K_mean)

        self.assertCountEqual(refine_test, jnp.asarray([0, 2]))

    def test_refine_rand(self):
        r"""
        For a toy example, X = [[0,0], [1,1], [2,2]], the 2-point coreset that minimises
        the MMD is specified y the indices S = [0, 2], ie X_c =  [[0,0], [[2,2]].

        Test, when given coreset indices [2,2], that refine_rand() returns [0, 2].
        """

        x = jnp.asarray([[0, 0], [1, 1], [2, 2]])
        n = len(x)

        S = jnp.asarray([2, 2])

        K = vmap(
            vmap(ck.rbf_kernel, in_axes=(None, 0), out_axes=0),
            in_axes=(0, None),
            out_axes=0,
        )(x, x)
        K_mean = K.sum(axis=1) / n

        refine_test = cr.refine_rand(x, S, ck.rbf_kernel, K_mean, p=1.0)

        self.assertCountEqual(refine_test, jnp.asarray([0, 2]))

    def test_refine_rev(self):
        r"""
        For a toy example, X = [[0,0], [1,1], [2,2]], the 2-point coreset that minimises
        the MMD is specified y the indices S = [0, 2], ie X_c =  [[0,0], [[2,2]].

        Test, when given coreset indices [1,2], that refine_rev() returns [0, 2].
        """

        x = jnp.asarray([[0, 0], [1, 1], [2, 2]])
        n = len(x)

        S = jnp.asarray([1, 2])

        K = vmap(
            vmap(ck.rbf_kernel, in_axes=(None, 0), out_axes=0),
            in_axes=(0, None),
            out_axes=0,
        )(x, x)
        K_mean = K.sum(axis=1) / n

        refine_test = cr.refine_rev(x, S, ck.rbf_kernel, K_mean)

        self.assertCountEqual(refine_test, jnp.asarray([0, 2]))
