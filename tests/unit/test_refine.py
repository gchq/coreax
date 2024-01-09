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
from coreax.kernel import SquaredExponentialKernel
from coreax.reduction import DataReduction


class TestRefine(unittest.TestCase):
    """
    Tests related to refine.py functions.
    """

    def test_refine_ones(self) -> None:
        """
        Test that refining an optimal coreset leaves ``coreset_indices`` unchanged.
        """
        x = jnp.asarray([[0, 0], [1, 1], [0, 0], [1, 1]])
        best_indices = {0, 1}
        coreset_indices = jnp.array(list(best_indices))

        data_reduction_obj = DataReduction(
            original_data=x, weight=None, kernel=SquaredExponentialKernel()
        )
        data_reduction_obj.reduction_indices = coreset_indices

        refine_regular = coreax.refine.RefineRegular()
        refine_regular.refine(data_reduction=data_reduction_obj)

        self.assertSetEqual(
            set(data_reduction_obj.reduction_indices.tolist()), best_indices
        )

    def test_refine_ints(self) -> None:
        """
        Test the regular refine method with a toy example.

        For a toy example, ``X = [[0,0], [1,1], [2,2]]``, the 2-point coreset that
        minimises the MMD is specified by the indices ``coreset_indices = [0, 2]``,
        i.e. ``X_c =  [[0,0], [[2,2]]``.

        Test this example, for every 2-point coreset, that refine() updates the coreset
        indices to [0, 2].
        """
        x = jnp.asarray([[0, 0], [1, 1], [2, 2]])
        best_indices = {0, 2}
        index_pairs = (set(combo) for combo in itertools.combinations(range(len(x)), 2))

        refine_regular = coreax.refine.RefineRegular()

        for test_indices in index_pairs:
            coreset_indices = jnp.array(list(test_indices))

            data_reduction_obj = DataReduction(
                original_data=x, weight=None, kernel=SquaredExponentialKernel()
            )
            data_reduction_obj.reduction_indices = coreset_indices

            refine_regular.refine(data_reduction=data_reduction_obj)

            with self.subTest(test_indices):
                self.assertSetEqual(
                    set(data_reduction_obj.reduction_indices.tolist()), best_indices
                )

    def test_refine_rand(self):
        """
        Test the random refine method with a toy example.

        For a toy example, ``X = [[0,0], [1,1], [2,2]]``, the 2-point coreset that
        minimises the MMD is specified by the indices ``coreset_indices = [0, 2]``,
        i.e. ``X_c =  [[0,0], [[2,2]]``.

        Test, when given ``coreset_indices=[2,2]``, that ``refine_rand()`` updates the
        coreset indices to ``[0, 2]``.
        """
        x = jnp.asarray([[0, 0], [1, 1], [2, 2]])
        best_indices = {0, 2}
        test_indices = [2, 2]
        coreset_indices = jnp.array(test_indices)

        data_reduction_obj = DataReduction(
            original_data=x, weight=None, kernel=SquaredExponentialKernel()
        )
        data_reduction_obj.reduction_indices = coreset_indices

        refine_rand = coreax.refine.RefineRandom(random_key=10, p=1.0)
        refine_rand.refine(data_reduction=data_reduction_obj)

        self.assertSetEqual(
            set(data_reduction_obj.reduction_indices.tolist()), best_indices
        )

    def test_refine_rand_negative_seed(self):
        """
        Test the random refine method with a toy example and a negative seed.

        For a toy example, ``X = [[0,0], [1,1], [2,2]]``, the 2-point coreset that
        minimises the MMD is specified by the indices ``coreset_indices = [0, 2]``,
        i.e. ``X_c =  [[0,0], [[2,2]]``.

        Test, when given ``coreset_indices=[2,2]``, that ``refine_rand()`` updates the
        coreset indices to ``[0, 2]``.
        """
        x = jnp.asarray([[0, 0], [1, 1], [2, 2]])
        best_indices = {0, 2}
        test_indices = [2, 2]
        coreset_indices = jnp.array(test_indices)

        data_reduction_obj = DataReduction(
            original_data=x, weight=None, kernel=SquaredExponentialKernel()
        )
        data_reduction_obj.reduction_indices = coreset_indices

        refine_rand = coreax.refine.RefineRandom(random_key=-10, p=1.0)
        refine_rand.refine(data_reduction=data_reduction_obj)

        self.assertSetEqual(
            set(data_reduction_obj.reduction_indices.tolist()), best_indices
        )

    def test_refine_reverse(self):
        """
        Test the reverse refine method with a toy example.

        For a toy example, ``X = [[0,0], [1,1], [2,2]]``, the 2-point coreset that
        minimises the MMD is specified by the indices ``coreset_indices = [0, 2]``,
        i.e. ``X_c =  [[0,0], [[2,2]]``.

        Test, for every 2-point coreset, that ``refine_rev()`` updates the coreset
        indices to ``[0, 2]``.
        """
        x = jnp.asarray([[0, 0], [1, 1], [2, 2]])
        best_indices = {0, 2}
        index_pairs = (set(combo) for combo in itertools.combinations(range(len(x)), 2))

        refine_rev = coreax.refine.RefineReverse()

        for test_indices in index_pairs:
            coreset_indices = jnp.array(list(test_indices))

            data_reduction_obj = DataReduction(
                original_data=x, weight=None, kernel=SquaredExponentialKernel()
            )
            data_reduction_obj.reduction_indices = coreset_indices

            refine_rev.refine(data_reduction=data_reduction_obj)

            with self.subTest(test_indices):
                self.assertSetEqual(
                    set(data_reduction_obj.reduction_indices.tolist()), best_indices
                )

    def test_kernel_mean_row_sum_approx(self):
        """
        Test for error when approximate_kernel_row_sum = True and no approximator given.
        """
        x = jnp.asarray([[0, 0], [1, 1], [0, 0], [1, 1]])
        best_indices = {0, 1}
        coreset_indices = jnp.array(list(best_indices))

        data_reduction_obj = DataReduction(
            original_data=x, weight=None, kernel=SquaredExponentialKernel()
        )
        data_reduction_obj.reduction_indices = coreset_indices

        refine_regular = coreax.refine.RefineRegular(approximate_kernel_row_sum=True)

        self.assertRaises(
            TypeError, refine_regular.refine, data_reduction=data_reduction_obj
        )
