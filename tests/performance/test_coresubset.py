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
import numpy as np
from jax import jit
from scipy.stats import ks_2samp

import coreax.coresubset as cs
import coreax.kernel as ck
from coreax.util import jit_test


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

    def test_greedy_body(self) -> None:
        """
        Test the performance of the greedy loop body method.

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
        kernel = ck.SquaredExponentialKernel()

        # Create a kernel herding object
        herding_object = cs.KernelHerding(kernel=kernel)

        # Test performance
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            deltas = jit_test(
                jit(
                    lambda *args, **kwargs: herding_object._greedy_body(
                        *args, **kwargs
                    ),
                    static_argnames=["kernel_vectorised", "unique"],
                ),
                i=0,
                val=(coreset_indices_0, kernel_similarity_penalty_0),
                x=x[i],
                kernel_vectorised=kernel.compute,
                kernel_matrix_row_sum_mean=kernel.calculate_kernel_matrix_row_sum_mean(
                    x[i]
                ),
                unique=True,
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)


if __name__ == "__main__":
    unittest.main()
