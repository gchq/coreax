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

import coreax.approximation
import coreax.data
import coreax.kernel
import coreax.reduction
import coreax.refine
import coreax.util


class CoresetMock(coreax.reduction.Coreset):
    """Test version of :class:`Coreset` with all methods implemented."""

    def fit_to_size(self, coreset_size: int) -> None:
        raise NotImplementedError


class TestRefine(unittest.TestCase):
    """
    Tests related to refine.py functions.
    """

    def test_validate_coreset_ok(self) -> None:
        """Check validation passes with populated coresubset."""
        coreset = CoresetMock()
        coreset.original_data = coreax.data.ArrayData.load(1)
        coreset.coreset = jnp.array(1)
        coreset.coreset_indices = jnp.array(0)
        coreax.refine.Refine._validate_coreset(coreset)

    def test_validate_coreset_no_fit(self) -> None:
        """Check validation fails when coreset has not been calculated."""
        coreset = CoresetMock()
        coreset.original_data = coreax.data.ArrayData.load(1)
        self.assertRaises(
            coreax.util.NotCalculatedError,
            coreax.refine.Refine._validate_coreset,
            coreset,
        )

    def test_validate_coreset_not_coresubset(self) -> None:
        """Check validation raises TypeError when not a coresubset."""
        coreset = CoresetMock()
        coreset.original_data = coreax.data.ArrayData.load(1)
        coreset.coreset = jnp.array(1)
        self.assertRaises(TypeError, coreax.refine.Refine._validate_coreset, coreset)

    def test_refine_ones(self) -> None:
        """
        Test that refining an optimal coreset leaves ``coreset_indices`` unchanged.
        """
        original_array = jnp.asarray([[0, 0], [1, 1], [0, 0], [1, 1]])
        best_indices = {0, 1}
        coreset_indices = jnp.array(list(best_indices))

        coreset_obj = CoresetMock(
            weights_optimiser=None, kernel=coreax.kernel.SquaredExponentialKernel()
        )
        coreset_obj.coreset_indices = coreset_indices
        coreset_obj.original_data = coreax.data.ArrayData.load(original_array)
        coreset_obj.coreset = original_array[coreset_indices, :]

        refine_regular = coreax.refine.RefineRegular()
        refine_regular.refine(coreset=coreset_obj)

        self.assertSetEqual(set(coreset_obj.coreset_indices.tolist()), best_indices)

    def test_refine_ints(self) -> None:
        """
        Test the regular refine method with a toy example.

        For a toy example, ``X = [[0,0], [1,1], [2,2]]``, the 2-point coreset that
        minimises the MMD is specified by the indices ``coreset_indices = [0, 2]``,
        i.e. ``X_c =  [[0,0], [[2,2]]``.

        Test this example, for every 2-point coreset, that refine() updates the coreset
        indices to [0, 2].
        """
        original_array = jnp.asarray([[0, 0], [1, 1], [2, 2]])
        best_indices = {0, 2}
        index_pairs = (
            set(combo)
            for combo in itertools.combinations(range(len(original_array)), 2)
        )

        refine_regular = coreax.refine.RefineRegular()

        for test_indices in index_pairs:
            coreset_indices = jnp.array(list(test_indices))

            coreset_obj = CoresetMock(
                weights_optimiser=None,
                kernel=coreax.kernel.SquaredExponentialKernel(),
            )
            coreset_obj.coreset_indices = coreset_indices
            coreset_obj.original_data = coreax.data.ArrayData.load(original_array)
            coreset_obj.coreset = original_array[coreset_indices, :]

            refine_regular.refine(coreset=coreset_obj)

            with self.subTest(test_indices):
                self.assertSetEqual(
                    set(coreset_obj.coreset_indices.tolist()), best_indices
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
        original_array = jnp.asarray([[0, 0], [1, 1], [2, 2]])
        best_indices = {0, 2}
        test_indices = [2, 2]
        coreset_indices = jnp.array(test_indices)

        coreset_obj = CoresetMock(
            weights_optimiser=None, kernel=coreax.kernel.SquaredExponentialKernel()
        )
        coreset_obj.coreset_indices = coreset_indices
        coreset_obj.original_data = coreax.data.ArrayData.load(original_array)
        coreset_obj.coreset = original_array[coreset_indices, :]

        refine_rand = coreax.refine.RefineRandom(random_key=10, p=1.0)
        refine_rand.refine(coreset=coreset_obj)

        self.assertSetEqual(set(coreset_obj.coreset_indices.tolist()), best_indices)

    def test_refine_reverse(self):
        """
        Test the reverse refine method with a toy example.

        For a toy example, ``X = [[0,0], [1,1], [2,2]]``, the 2-point coreset that
        minimises the MMD is specified by the indices ``coreset_indices = [0, 2]``,
        i.e. ``X_c =  [[0,0], [[2,2]]``.

        Test, for every 2-point coreset, that ``refine_rev()`` updates the coreset
        indices to ``[0, 2]``.
        """
        original_array = jnp.asarray([[0, 0], [1, 1], [2, 2]])
        best_indices = {0, 2}
        index_pairs = (
            set(combo)
            for combo in itertools.combinations(range(len(original_array)), 2)
        )

        refine_rev = coreax.refine.RefineReverse()

        for test_indices in index_pairs:
            coreset_indices = jnp.array(list(test_indices))

            coreset_obj = CoresetMock(
                weights_optimiser=None,
                kernel=coreax.kernel.SquaredExponentialKernel(),
            )
            coreset_obj.coreset_indices = coreset_indices
            coreset_obj.original_data = coreax.data.ArrayData.load(original_array)
            coreset_obj.coreset = original_array[coreset_indices, :]

            refine_rev.refine(coreset=coreset_obj)

            with self.subTest(test_indices):
                self.assertSetEqual(
                    set(coreset_obj.coreset_indices.tolist()), best_indices
                )

    def test_kernel_mean_row_sum_approx_invalid(self):
        """
        Test for error when an invalid approximator is given.
        """
        original_array = jnp.asarray([[0, 0], [1, 1], [0, 0], [1, 1]])
        best_indices = {0, 1}
        coreset_indices = jnp.array(list(best_indices))

        coreset_obj = CoresetMock(
            weights_optimiser=None, kernel=coreax.kernel.SquaredExponentialKernel()
        )
        coreset_obj.coreset_indices = coreset_indices
        coreset_obj.original_data = coreax.data.ArrayData.load(original_array)
        coreset_obj.coreset = original_array[coreset_indices, :]
        refine_regular = coreax.refine.RefineRegular(approximator="not_an_approximator")

        self.assertRaises(AttributeError, refine_regular.refine, coreset=coreset_obj)

    def test_kernel_mean_row_sum_approx_valid(self):
        """
        Test for no-error when a valid approximator is given.
        """
        original_array = jnp.asarray([[0, 0], [1, 1], [0, 0], [1, 1]])
        best_indices = {0, 1}
        coreset_indices = jnp.array(list(best_indices))
        kernel = coreax.kernel.SquaredExponentialKernel()
        approximator = coreax.approximation.RandomApproximator(
            kernel=kernel, num_train_points=2, num_kernel_points=2
        )
        coreset_obj = CoresetMock(
            weights_optimiser=None, kernel=coreax.kernel.SquaredExponentialKernel()
        )
        coreset_obj.coreset_indices = coreset_indices
        coreset_obj.original_data = coreax.data.ArrayData.load(original_array)
        coreset_obj.coreset = original_array[coreset_indices, :]
        coreset_obj.kernel = kernel
        refine_regular = coreax.refine.RefineRegular(approximator=approximator)
        # This step passing is a test that the approximator does not cause an issue with
        # the code run
        refine_regular.refine(coreset=coreset_obj)
