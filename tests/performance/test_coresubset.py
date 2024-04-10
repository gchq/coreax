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
from jax import random
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


class TestSupervisedCoreSubset(unittest.TestCase):
    """
    Tests for performance of supervised data coreset methods within coresubset.py.
    """

    def setUp(self) -> None:
        """
        Define data shared across tests.
        """
        # p-value threshold to pass/fail test
        self.threshold = 0.05
        # Number of independent observations to generate for the KS two-sample test
        self.num_samples_to_generate = 10
        # Number of data points per 'observation'
        self.num_data_points_per_observation = 10
        # Dimensionality of features and responses
        self.feature_dimension = 5
        self.response_dimension = 5
        # Size of the coreset to generate (if loop body called until completion)
        self.coreset_size = 1
        # Set random key
        self.random_key = random.key(0)
        # Set regularisation parameter
        self.regularisation_parameter = 1e-6

    def test_greedy_body(self) -> None:
        """
        Test the performance of the greedy loop body method.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        # Predefine the common variables that are passed to the method
        coreset_indices_0 = -1*jnp.ones(self.coreset_size, dtype=jnp.int32)
        coreset_identity_0 = jnp.zeros((self.coreset_size, self.coreset_size))
        feature_kernel = coreax.kernel.SquaredExponentialKernel()
        response_kernel = coreax.kernel.SquaredExponentialKernel()
        batch_indices = coreax.util.sample_batch_indices(
            random_key=self.random_key,
            data_size=self.num_data_points_per_observation,
            batch_size=self.num_data_points_per_observation,
            num_batches=self.coreset_size
        )
        all_possible_next_coreset_indices_0 = jnp.hstack(
            (
                batch_indices[:, [0]],
                jnp.tile(-1, (batch_indices.shape[0], self.coreset_size - 1) )
            )
        )      
        
        # Generate multi-output supervised dataset
        x = random.uniform(
            self.random_key,
            shape=(
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.feature_dimension
            )
        )
        coefficients = random.uniform(
            self.random_key,
            shape=(
                self.num_samples_to_generate,
                self.feature_dimension,
                self.response_dimension
            )
        )
        errors = random.normal(
            key,
            shape=(
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.response_dimension
            )
        )
        y = jnp.sin(10*x) @ coefficients + 0.1*errors
        
        # Create a GreedyCMMD object
        greedy_CMMD_object = GreedyCMMD(
            random_key=self.random_key,
            feature_kernel=feature_kernel,
            response_kernel=response_kernel,
            num_feature_dimensions=self.feature_dimension, 
            regularisation_parameter=regularisation_parameter,
            unique=True,
            batch_size=None,
            refine_method=None
        )

        # Test performance
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            # Compute _greedy_body inputs
            feature_gramian = feature_kernel.compute(x[i], x[i])
            response_gramian = response_kernel.compute(y[i], y[i])
            
            inverse_feature_gramian = coreax.util.invert_regularised_array(
                array=feature_gramian,
                regularisation_parameter=self.regularisation_parameter,
                identity=jnp.eye(self.num_data_points_per_observation)
            )
            
            training_CME = feature_gramian @ inverse_feature_gramian @ response_gramian
            
            feature_gramian = jnp.pad(feature_gramian, [(0, 1)], mode='constant')
            response_gramian = jnp.pad(response_gramian, [(0, 1)], mode='constant')
            training_CME = jnp.pad(training_CME, [(0, 1)], mode='constant')      
            
            deltas = coreax.util.jit_test(
                greedy_CMMD_object._greedy_body,
                fn_kwargs={
                    "i": 0,
                    "val": (coreset_indices_0, coreset_identity_0, all_possible_next_coreset_indices_0),
                    "feature_gramian": feature_gramian,
                    "response_gramian": response_gramian,
                    "training_CME": training_CME,
                    "batch_indices": batch_indices,
                    "regularisation_parameter": self.regularisation_parameter,
                    "unique": True
                },
                jit_kwargs={"static_argnames": ["regularisation_parameter", "unique"]},
            )
            # pylint: enable=protected-access
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold) 

# pylint: enable=duplicate-code

if __name__ == "__main__":
    unittest.main()
