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
Performance tests for JIT compilation in kernel implementations.
"""

import unittest

import numpy as np
from scipy.stats import ks_2samp

import coreax.kernel
import coreax.util

# Performance tests are split across several files for readability. As a result, ignore
# the pylint warnings for duplicated-code. Additionally, we wrap the method/function of
# interest in a lambda function to ensure no cached JIT code is re-used to make the test
# fair. As a result, ignore the pylint warnings for unnecessary-lambda.
# pylint: disable=duplicate-code


class TestKernel(unittest.TestCase):
    """
    Base class with common setUp for all kernels.
    """

    def setUp(self) -> None:
        # p-value threshold to pass/fail test
        self.threshold = 0.05
        # number of independent observations to generate (for each of the two samples)
        self.num_samples_to_generate = 10
        # number of data points per 'observation'
        self.num_data_points_per_observation = 10
        # dimensionality of observations
        self.dimension = 10


class TestSquaredExponentialKernel(TestKernel):
    """
    Tests related to the SquaredExponentialKernel defined in kernel.py
    """

    def test_compute(self) -> None:
        """
        Test the performance of the SquaredExponentialKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        y = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        kernel = coreax.kernel.SquaredExponentialKernel()
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            deltas = coreax.util.jit_test(kernel.compute, fn_args=(x[i], y[i]))
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)

    def test_grad_x(self):
        """
        Test the performance of the SquaredExponentialKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        y = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        kernel = coreax.kernel.SquaredExponentialKernel()
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            deltas = coreax.util.jit_test(kernel.grad_x, fn_args=(x[i], y[i]))
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)

    def test_grad_y(self):
        """
        Test the performance of the SquaredExponentialKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        y = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        kernel = coreax.kernel.SquaredExponentialKernel()
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            deltas = coreax.util.jit_test(kernel.grad_y, fn_args=(x[i], y[i]))
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)

    def test_div_x_grad_y(self):
        """
        Test the performance of the SquaredExponentialKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        y = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        kernel = coreax.kernel.SquaredExponentialKernel()
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            deltas = coreax.util.jit_test(
                kernel.divergence_x_grad_y, fn_args=(x[i], y[i])
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)


class TestLaplacianKernel(TestKernel):
    """
    Tests related to the LaplacianKernel defined in kernel.py
    """

    def test_compute(self) -> None:
        """
        Test the performance of the LaplacianKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        y = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        kernel = coreax.kernel.LaplacianKernel()
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            deltas = coreax.util.jit_test(kernel.compute, fn_args=(x[i], y[i]))
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)

    def test_grad_x(self):
        """
        Test the performance of the LaplacianKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        y = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        kernel = coreax.kernel.LaplacianKernel()
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            deltas = coreax.util.jit_test(kernel.grad_x, fn_args=(x[i], y[i]))
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)

    def test_grad_y(self):
        """
        Test the performance of the LaplacianKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        y = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        kernel = coreax.kernel.LaplacianKernel()
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            deltas = coreax.util.jit_test(kernel.grad_y, fn_args=(x[i], y[i]))
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)

    def test_div_x_grad_y(self):
        """
        Test the performance of the LaplacianKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        y = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        kernel = coreax.kernel.LaplacianKernel()
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            deltas = coreax.util.jit_test(
                kernel.divergence_x_grad_y, fn_args=(x[i], y[i])
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)


class TestPCIMQKernel(TestKernel):
    """
    Tests related to the PCIMQKernel defined in kernel.py
    """

    def test_compute(self) -> None:
        """
        Test the performance of the PCIMQKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        y = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        kernel = coreax.kernel.PCIMQKernel()
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            deltas = coreax.util.jit_test(kernel.compute, fn_args=(x[i], y[i]))
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)

    def test_grad_x(self):
        """
        Test the performance of the PCIMQKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        y = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        kernel = coreax.kernel.PCIMQKernel()
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            deltas = coreax.util.jit_test(kernel.grad_x, fn_args=(x[i], y[i]))
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)

    def test_grad_y(self):
        """
        Test the performance of the PCIMQKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        y = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        kernel = coreax.kernel.PCIMQKernel()
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            deltas = coreax.util.jit_test(kernel.grad_y, fn_args=(x[i], y[i]))
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)

    def test_div_x_grad_y(self):
        """
        Test the performance of the PCIMQKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        y = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        kernel = coreax.kernel.PCIMQKernel()
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            deltas = coreax.util.jit_test(
                kernel.divergence_x_grad_y, fn_args=(x[i], y[i])
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)


class TestSteinKernel(TestKernel):
    """
    Tests related to the SteinKernel defined in kernel.py
    """

    def test_compute(self) -> None:
        """
        Test the performance of the SteinKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        y = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )

        def score_function(x_):
            return -x_

        kernel = coreax.kernel.SteinKernel(
            base_kernel=coreax.kernel.PCIMQKernel(),
            score_function=score_function,
        )
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            deltas = coreax.util.jit_test(kernel.compute, fn_args=(x[i], y[i]))
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)

    def test_grad_x(self):
        """
        Test the performance of the SteinKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        y = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )

        def score_function(x_):
            return -x_

        kernel = coreax.kernel.SteinKernel(
            base_kernel=coreax.kernel.PCIMQKernel(),
            score_function=score_function,
        )
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            deltas = coreax.util.jit_test(kernel.grad_x, fn_args=(x[i], y[i]))
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)

    def test_grad_y(self):
        """
        Test the performance of the SteinKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        y = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )

        def score_function(x_):
            return -x_

        kernel = coreax.kernel.SteinKernel(
            base_kernel=coreax.kernel.PCIMQKernel(),
            score_function=score_function,
        )
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            deltas = coreax.util.jit_test(kernel.grad_y, fn_args=(x[i], y[i]))
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)

    def test_div_x_grad_y(self):
        """
        Test the performance of the SteinKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch suboptimal JIT tracing.
        """
        x = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )
        y = np.random.random(
            (
                self.num_samples_to_generate,
                self.num_data_points_per_observation,
                self.dimension,
            )
        )

        def score_function(x_):
            return -x_

        kernel = coreax.kernel.SteinKernel(
            base_kernel=coreax.kernel.PCIMQKernel(),
            score_function=score_function,
        )
        pre = []
        post = []
        for i in range(self.num_samples_to_generate):
            deltas = coreax.util.jit_test(
                kernel.divergence_x_grad_y,
                fn_args=(x[i], y[i]),
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        p_value = ks_2samp(pre, post).pvalue
        self.assertLessEqual(p_value, self.threshold)


# pylint: enable=unnecessary-lambda
# pylint: enable=duplicate-code


if __name__ == "__main__":
    unittest.main()
