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
from unittest.mock import patch

import jax.numpy as jnp

import coreax.reduction as re


class TestDataReduction(unittest.TestCase):
    """
    Tests related to kernel.py functions.
    """

    def setUp(self) -> None:
        self.original_data = jnp.array([0, 0.3, 0.75, 1])
        self.test_class = re.DataReduction(self.original_data, None)

    def test_solve_weights(self) -> None:
        """
        Test the solve_weights methods of :class:`coreax.reduction.DataReduction`.
        """
        # make reduced data distinguishable from original data without calling an actual reduction
        reduced_data = self.original_data[jnp.array([0, 1, 3])]
        self.test_class.reduced_data = reduced_data

        with patch('coreax.weights.simplex_weights') as mock_calc_weights_simp, \
                patch('coreax.weights.calculate_BQ_weights') as mock_calc_weights_bq:

            with self.subTest('MMD'):
                self.test_class.solve_weights('a_kernel', 'MMD')

                mock_calc_weights_simp.assert_called_once_with(self.original_data, reduced_data, 'a_kernel')
                mock_calc_weights_bq.assert_not_called()

                mock_calc_weights_simp.reset_mock()
                mock_calc_weights_bq.reset_mock()

            with self.subTest('SBQ'):
                self.test_class.solve_weights('a_kernel', 'SBQ')

                mock_calc_weights_simp.assert_not_called()
                mock_calc_weights_bq.assert_called_once_with(self.original_data, reduced_data, 'a_kernel')

                mock_calc_weights_simp.reset_mock()
                mock_calc_weights_bq.reset_mock()

            with self.subTest('Unexpected weighting'):
                self.assertRaisesRegex(ValueError,
                                       "weight type 'Unexpected weighting' not recognised.",
                                       self.test_class.solve_weights,
                                       'a_kernel',
                                       'Unexpected weighting'
                                       )

                mock_calc_weights_simp.assert_not_called()
                mock_calc_weights_bq.assert_not_called()

                mock_calc_weights_simp.reset_mock()
                mock_calc_weights_bq.reset_mock()
