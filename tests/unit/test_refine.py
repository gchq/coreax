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

import itertools
import unittest

import jax.numpy as jnp

import coreax.refine
from coreax.kernel import RBFKernel


class TestMetrics(unittest.TestCase):
    """
    Tests related to refine.py functions.
    """

    def test_refine_ones(self) -> None:
        """
        Test that refining an optimal coreset, specified by indices S, returns the same
         indices.
        """

        x = jnp.asarray([[0, 0], [1, 1], [0, 0], [1, 1]])

        best_indices = {0, 1}

        S = jnp.array(list(best_indices))

        rbf_kernel = RBFKernel(bandwidth=1.0)

        refine_regular = coreax.refine.RefineRegular(kernel=rbf_kernel)

        refine_test = refine_regular.refine(x=x, S=S)

        self.assertSetEqual(set(refine_test.tolist()), best_indices)

    def test_refine_ints(self) -> None:
        """
        For a toy example, X = [[0,0], [1,1], [2,2]], the 2-point coreset that minimises
        the MMD is specified by the indices S = [0, 2], ie X_c =  [[0,0], [[2,2]].

        Test this example, for any 2-point coreset, that refine returns [0, 2].
        """

        x = jnp.asarray([[0, 0], [1, 1], [2, 2]])

        best_indices = {0, 2}

        index_pairs = (set(combo) for combo in itertools.combinations(range(len(x)), 2))

        rbf_kernel = RBFKernel(bandwidth=1.0)
        refine_regular = coreax.refine.RefineRegular(kernel=rbf_kernel)

        for test_indices in index_pairs:
            S = jnp.array(list(test_indices))

            refine_test = refine_regular.refine(x, S)

            with self.subTest(test_indices):
                self.assertSetEqual(set(refine_test.tolist()), best_indices)

    def test_refine_rand(self):
        """
        For a toy example, X = [[0,0], [1,1], [2,2]], the 2-point coreset that minimises
        the MMD is specified by the indices S = [0, 2], ie X_c =  [[0,0], [[2,2]].

        Test, when given coreset indices [2,2], that refine_rand() returns [0, 2].
        """

        x = jnp.asarray([[0, 0], [1, 1], [2, 2]])

        best_indices = {0, 2}

        test_indices = [2, 2]

        S = jnp.array(test_indices)

        rbf_kernel = RBFKernel(bandwidth=1.0)

        refine_rand = coreax.refine.RefineRandom(kernel=rbf_kernel, p=1.0)

        refine_test = refine_rand.refine(x, S)

        self.assertSetEqual(set(refine_test.tolist()), best_indices)

    def test_refine_rev(self):
        """
        For a toy example, X = [[0,0], [1,1], [2,2]], the 2-point coreset that minimises
        the MMD is specified by the indices S = [0, 2], ie X_c =  [[0,0], [[2,2]].

        Test, for any 2-point coreset, that refine_rev() returns [0, 2].
        """

        x = jnp.asarray([[0, 0], [1, 1], [2, 2]])

        best_indices = {0, 2}

        index_pairs = (set(combo) for combo in itertools.combinations(range(len(x)), 2))

        rbf_kernel = RBFKernel(bandwidth=1.0)

        refine_rev = coreax.refine.RefineRev(kernel=rbf_kernel)

        for test_indices in index_pairs:
            S = jnp.array(list(test_indices))

            refine_test = refine_rev.refine(x, S)

            with self.subTest(test_indices):
                self.assertSetEqual(set(refine_test.tolist()), best_indices)
