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

"""
Performance tests for JIT compilation in coresubset implementations.
"""

import unittest

import jax.numpy as jnp
import numpy as np
from jax import random, vmap
from scipy.stats import ks_2samp

import coreax.coresubset
import coreax.kernel
import coreax.util

# Performance tests are split across several files for readability. As a result, ignore
# the pylint warnings for duplicated-code.
# pylint: disable=duplicate-code


class TestCoreSubset(unittest.TestCase):
    """
    Tests for performance of methods within coresubset.py.
    """

    def setUp(self) -> None:
        """
        Define data shared across tests.
        """
        # p-value threshold to pass/fail test
        self.threshold = 0.05
        # number of independent observations to generate (for each of the two samples)
        self.num_samples_to_generate = 10
        # number of data points per 'observation'
        self.num_data_points_per_observation = 10
        # dimensionality of observations
        self.dimension = 10
        # Size of the coreset to generate (if loop body called until completion)
        self.coreset_size = 1
        self.random_key = random.key(0)

    def test_greedy_body_kernel_herding(self) -> None:
        """
        Test the performance of the greedy loop body method for kernel herding.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        # Predefine the variables that are passed to the method
        coreset_indices_0 = jnp.zeros(self.coreset_size, dtype=jnp.int32)
        kernel_similarity_penalty_0 = jnp.zeros(self.num_data_points_per_observation)
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        kernel = coreax.kernel.SquaredExponentialKernel()

        # Create a kernel herding object
        herding_object = coreax.coresubset.KernelHerding(self.random_key, kernel=kernel)

        # Test performance
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            # pylint: disable=protected-access
            kernel_matrix_rsm = kernel.calculate_kernel_matrix_row_sum_mean(x[i])
            deltas = coreax.util.jit_test(
                herding_object._greedy_body,
                fn_kwargs={
                    "i": 0,
                    "val": (coreset_indices_0, kernel_similarity_penalty_0),
                    "x": x[i],
                    "kernel_vectorised": kernel.compute,
                    "kernel_matrix_row_sum_mean": kernel_matrix_rsm,
                    "unique": True,
                },
                jit_kwargs={"static_argnames": ["kernel_vectorised", "unique"]},
            )
            # pylint: enable=protected-access
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)

    def test_loop_body_rp_cholesky(self) -> None:
        """
        Test the performance of the loop body method for RP Cholesky.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        # Predefine the variables that are passed to the method
        coreset_indices_0 = jnp.zeros(self.coreset_size, dtype=jnp.int32)
        n = self.num_data_points_per_observation
        F_0 = jnp.zeros((n, self.coreset_size))
        _, key = random.split(self.random_key)
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        kernel = coreax.kernel.SquaredExponentialKernel()

        # Create a RPC object
        rpc_object = coreax.coresubset.RPCholesky(self.random_key, kernel=kernel)

        # Test performance
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            # pylint: disable=protected-access
            residual_diagonal = vmap(
                kernel.compute_elementwise, in_axes=(0, 0), out_axes=0
            )(x[i], x[i])

            deltas = coreax.util.jit_test(
                rpc_object._loop_body,
                fn_kwargs={
                    "i": 0,
                    "val": (residual_diagonal, F_0, coreset_indices_0, key),
                    "x": x[i],
                    "kernel_vectorised": kernel.compute,
                    "unique": True,
                },
                jit_kwargs={"static_argnames": ["kernel_vectorised", "unique"]},
            )
            # pylint: enable=protected-access
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)


# pylint: enable=duplicate-code


if __name__ == "__main__":
    unittest.main()
