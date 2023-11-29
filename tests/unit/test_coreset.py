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
from unittest.mock import patch, Mock, MagicMock

import numpy as np
import scipy.stats
from jax import numpy as jnp

from coreax.coreset import KernelHerding
from coreax.data import DataReader
from coreax.kernel import Kernel


class TestCoreset(unittest.TestCase):
    """
    Tests related to the Coreset class in coreset.py
    """


class TestKernelHerding(unittest.TestCase):
    """
    Tests related to the KernelHerding class defined in coreset.py
    """

    def test_fit(self) -> None:
        r"""
        Test the fit method of the KernelHerding class.

        Methods called by this method are mocked and assumed tested elsewhere.
        """

        with (patch('coreax.kernel.Kernel') as mock_kernel, \
              patch('coreax.data.DataReader') as mock_reader, \
                patch('jax.lax.fori_loop') as mock_loop):

            mock_gram_matrix = jnp.asarray([[1, 2, 3, 4],
                                            [5, 6, 7, 8],
                                            [9, 10, 11, 12]],
                                           dtype=jnp.float32)
            mock_loop.return_value = jnp.array(
                [3, 1, 2]), mock_gram_matrix, jnp.asarray([])

            # Define class
            test_class = KernelHerding(mock_reader, 'MMD', mock_kernel, size=3)

            with self.subTest("mean_supplied_no_refine"):
                # Mean supplied but no refine
                # Mean calculation should not happen and neither should refinement

                gram_matrix, Kbar = test_class.fit(K_mean=2.5)

                mock_kernel.calculate_kernel_matrix_row_sum_mean.assert_not_called()

                np.testing.assert_array_equal(test_class.reduction_indices,
                                              jnp.array([3, 1, 2]))

                np.testing.assert_array_equal(gram_matrix,
                                              jnp.asarray([[4, 2, 3],
                                                           [8, 6, 7],
                                                           [12, 10, 11]],
                                                          dtype=jnp.float32))

                np.testing.assert_array_equal(Kbar,
                                              jnp.asarray([2.5, 6.5, 10.5]))

            with self.subTest("mean_not_supplied_with_refine"):
                # Mean not supplied. Refine called for. No approximation.
                # TODO Mean calculation should happen and so should refinement but without approximation

                _, _ = test_class.fit(refine='RefineRegular')

                mock_kernel.calculate_kernel_matrix_row_sum_mean.assert_called_once()

            with self.subTest("mean_not_supplied_with_refine_approximate"):
                # Mean not supplied. Refine called for. With approximation.
                # TODO Mean calculation should happen and so should refinement and with approximation

                _, _ = test_class.fit(refine='RefineRegular', approximator='approximator')

                mock_kernel.calculate_kernel_matrix_row_sum_mean.assert_called_once()

    def test_greedy_body(self) -> None:
        r"""
        Test the _greedy_body method of the KernelHerding class.

        Methods called by this method are mocked and assumed tested elsewhere.
        """

        with (patch('coreax.kernel.Kernel') as mock_kernel, \
              patch('coreax.data.DataReader') as mock_reader):

            mock_reader.pre_reduced_array = jnp.asarray([[5, 4, 1, 2],
                                                         [3, 1, 4, 7],
                                                         [8, 4, 0, 3]])

            K_mean = jnp.asarray([3, 3.75, 3.75])

            mock_k_vec = MagicMock()
            mock_k_vec.return_value = jnp.asarray([0.5, 1, 1])

            # Define class
            test_class = KernelHerding(mock_reader, 'MMD', mock_kernel, size=3)

            S0 = jnp.zeros(2, dtype=jnp.int32)
            K0 = jnp.zeros((2, 3))
            K_t0 = jnp.zeros(3)

            with self.subTest("not_unique"):
                # Run the method twice to replicate the looping of it in higher methods
                # First and second runs it selects the second element.

                S1, K1, K_t1 = test_class._greedy_body(0, (S0, K0, K_t0), mock_k_vec,
                                                       K_mean=K_mean, unique=False)

                np.testing.assert_array_equal(S1, jnp.asarray([1, 0]))
                np.testing.assert_array_equal(K1, jnp.asarray([[0.5, 1, 1],
                                                              [0, 0, 0]]))
                np.testing.assert_array_equal(K_t1, jnp.asarray([0.5, 1, 1]))

                S2, K2, K_t2 = test_class._greedy_body(1, (S1, K1, K_t1), mock_k_vec,
                                                       K_mean=K_mean, unique=False)

                np.testing.assert_array_equal(S2, jnp.asarray([1, 1]))
                np.testing.assert_array_equal(K2, jnp.asarray([[0.5, 1, 1],
                                                              [0.5, 1, 1]]))
                np.testing.assert_array_equal(K_t2, jnp.asarray([1, 2, 2]))

            with self.subTest("unique"):
                # Unique elements. This time we don't pick the same index twice.

                S1, K1, K_t1 = test_class._greedy_body(0, (S0, K0, K_t0), mock_k_vec,
                                                       K_mean=K_mean, unique=True)

                np.testing.assert_array_equal(S1, jnp.asarray([1, 0]))
                np.testing.assert_array_equal(K1, jnp.asarray([[0.5, 1, 1],
                                                               [0, 0, 0]]))
                np.testing.assert_array_equal(K_t1, jnp.asarray([0.5, jnp.inf, 1]))

                S2, K2, K_t2 = test_class._greedy_body(1, (S1, K1, K_t1), mock_k_vec,
                                                       K_mean=K_mean, unique=True)

                np.testing.assert_array_equal(S2, jnp.asarray([1, 2]))
                np.testing.assert_array_equal(K2, jnp.asarray([[0.5, 1, 1],
                                                               [0.5, 1, 1]]))
                np.testing.assert_array_equal(K_t2, jnp.asarray([1, jnp.inf, jnp.inf]))


class MockKernel(Mock(spec=Kernel)):

    def __init__(self):
        self.compute = lambda x: jnp.exp(-x**2)

if __name__ == "__main__":
    unittest.main()
