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
import time
import unittest
from unittest.mock import patch

import numpy as np
import scipy.stats
from jax import jit
from jax import numpy as jnp
from scipy.stats import ks_2samp

import coreax.approximation as ca
import coreax.kernel as ck


def jit_test(fn, *args, **kwargs) -> tuple:
    """
    A before and after run of a function, catching timings of each for JIT testing.

    The function is called with supplied arguments twice, and timed for each run. These
    timings are returned in a 2-tuple

    :param fn: function callable to test
    :return: (first run time, second run time)
    """
    st = time.time()
    fn(*args, **kwargs)
    en = time.time()
    pre_delta = en - st
    st = time.time()
    fn(*args, **kwargs)
    en = time.time()
    post_delta = en - st
    return (pre_delta, post_delta)


class TestSquaredExponentialKernel(unittest.TestCase):
    """
    Tests related to the SquaredExponentialKernel defined in kernel.py
    """

    def test_compute(self) -> None:
        """
        Test the performance of the SquaredExponentialKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch sub-optimal JIT tracing.
        """
        # p-value threshold to pass/fail test
        thres = 0.05

        # number of independent observations to generate (for each of the two samples)
        N = 10

        # number of data points per 'observation'
        n = 10

        # data dimension
        d = 10
        x = np.random.random((N, n, d))
        y = np.random.random((N, n, d))
        kernel = ck.SquaredExponentialKernel()
        pre = []
        post = []
        for i in range(N):
            deltas = jit_test(
                jit(lambda *args, **kwargs: kernel.compute(*args, **kwargs)), x[i], y[i]
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        pval = ks_2samp(pre, post).pvalue
        self.assertLessEqual(pval, thres)

    def test_grad_x(self):
        """
        Test the performance of the SquaredExponentialKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch sub-optimal JIT tracing.
        """
        # p-value threshold to pass/fail test
        thres = 0.05

        # number of independent observations to generate (for each of the two samples)
        N = 10

        # number of data points per 'observation'
        n = 10

        # data dimension
        d = 10
        x = np.random.random((N, n, d))
        y = np.random.random((N, n, d))
        kernel = ck.SquaredExponentialKernel()
        pre = []
        post = []
        for i in range(N):
            deltas = jit_test(
                jit(lambda *args, **kwargs: kernel.grad_x(*args, **kwargs)), x[i], y[i]
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        pval = ks_2samp(pre, post).pvalue
        self.assertLessEqual(pval, thres)

    def test_grad_y(self):
        """
        Test the performance of the SquaredExponentialKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch sub-optimal JIT tracing.
        """
        # p-value threshold to pass/fail test
        thres = 0.05

        # number of independent observations to generate (for each of the two samples)
        N = 10

        # number of data points per 'observation'
        n = 10

        # data dimension
        d = 10
        x = np.random.random((N, n, d))
        y = np.random.random((N, n, d))
        kernel = ck.SquaredExponentialKernel()
        pre = []
        post = []
        for i in range(N):
            deltas = jit_test(
                jit(lambda *args, **kwargs: kernel.grad_y(*args, **kwargs)), x[i], y[i]
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        pval = ks_2samp(pre, post).pvalue
        self.assertLessEqual(pval, thres)

    def test_div_x_grad_y(self):
        """
        Test the performance of the SquaredExponentialKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch sub-optimal JIT tracing.
        """
        # p-value threshold to pass/fail test
        thres = 0.05

        # number of independent observations to generate (for each of the two samples)
        N = 10

        # number of data points per 'observation'
        n = 10

        # data dimension
        d = 10
        x = np.random.random((N, n, d))
        y = np.random.random((N, n, d))
        kernel = ck.SquaredExponentialKernel()
        pre = []
        post = []
        for i in range(N):
            deltas = jit_test(
                jit(
                    lambda *args, **kwargs: kernel.divergence_x_grad_y(*args, **kwargs)
                ),
                x[i],
                y[i],
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        pval = ks_2samp(pre, post).pvalue
        self.assertLessEqual(pval, thres)


class TestLaplacianKernel(unittest.TestCase):
    """
    Tests related to the LaplacianKernel defined in kernel.py
    """

    def test_compute(self) -> None:
        """
        Test the performance of the LaplacianKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch sub-optimal JIT tracing.
        """
        # p-value threshold to pass/fail test
        thres = 0.05

        # number of independent observations to generate (for each of the two samples)
        N = 10

        # number of data points per 'observation'
        n = 10

        # data dimension
        d = 10
        x = np.random.random((N, n, d))
        y = np.random.random((N, n, d))
        kernel = ck.LaplacianKernel()
        pre = []
        post = []
        for i in range(N):
            deltas = jit_test(
                jit(lambda *args, **kwargs: kernel.compute(*args, **kwargs)), x[i], y[i]
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        pval = ks_2samp(pre, post).pvalue
        self.assertLessEqual(pval, thres)

    def test_grad_x(self):
        """
        Test the performance of the LaplacianKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch sub-optimal JIT tracing.
        """
        # p-value threshold to pass/fail test
        thres = 0.05

        # number of independent observations to generate (for each of the two samples)
        N = 10

        # number of data points per 'observation'
        n = 10

        # data dimension
        d = 10
        x = np.random.random((N, n, d))
        y = np.random.random((N, n, d))
        kernel = ck.LaplacianKernel()
        pre = []
        post = []
        for i in range(N):
            deltas = jit_test(
                jit(lambda *args, **kwargs: kernel.grad_x(*args, **kwargs)), x[i], y[i]
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        pval = ks_2samp(pre, post).pvalue
        self.assertLessEqual(pval, thres)

    def test_grad_y(self):
        """
        Test the performance of the LaplacianKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch sub-optimal JIT tracing.
        """
        # p-value threshold to pass/fail test
        thres = 0.05

        # number of independent observations to generate (for each of the two samples)
        N = 10

        # number of data points per 'observation'
        n = 10

        # data dimension
        d = 10
        x = np.random.random((N, n, d))
        y = np.random.random((N, n, d))
        kernel = ck.LaplacianKernel()
        pre = []
        post = []
        for i in range(N):
            deltas = jit_test(
                jit(lambda *args, **kwargs: kernel.grad_y(*args, **kwargs)), x[i], y[i]
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        pval = ks_2samp(pre, post).pvalue
        self.assertLessEqual(pval, thres)

    def test_div_x_grad_y(self):
        """
        Test the performance of the LaplacianKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch sub-optimal JIT tracing.
        """
        # p-value threshold to pass/fail test
        thres = 0.05

        # number of independent observations to generate (for each of the two samples)
        N = 10

        # number of data points per 'observation'
        n = 10

        # data dimension
        d = 10
        x = np.random.random((N, n, d))
        y = np.random.random((N, n, d))
        kernel = ck.LaplacianKernel()
        pre = []
        post = []
        for i in range(N):
            deltas = jit_test(
                jit(
                    lambda *args, **kwargs: kernel.divergence_x_grad_y(*args, **kwargs)
                ),
                x[i],
                y[i],
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        pval = ks_2samp(pre, post).pvalue
        self.assertLessEqual(pval, thres)


class TestPCIMQKernel(unittest.TestCase):
    """
    Tests related to the PCIMQKernel defined in kernel.py
    """

    def test_compute(self) -> None:
        """
        Test the performance of the PCIMQKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch sub-optimal JIT tracing.
        """
        # p-value threshold to pass/fail test
        thres = 0.05

        # number of independent observations to generate (for each of the two samples)
        N = 10

        # number of data points per 'observation'
        n = 10

        # data dimension
        d = 10
        x = np.random.random((N, n, d))
        y = np.random.random((N, n, d))
        kernel = ck.PCIMQKernel()
        pre = []
        post = []
        for i in range(N):
            deltas = jit_test(
                jit(lambda *args, **kwargs: kernel.compute(*args, **kwargs)), x[i], y[i]
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        pval = ks_2samp(pre, post).pvalue
        self.assertLessEqual(pval, thres)

    def test_grad_x(self):
        """
        Test the performance of the PCIMQKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch sub-optimal JIT tracing.
        """
        # p-value threshold to pass/fail test
        thres = 0.05

        # number of independent observations to generate (for each of the two samples)
        N = 10

        # number of data points per 'observation'
        n = 10

        # data dimension
        d = 10
        x = np.random.random((N, n, d))
        y = np.random.random((N, n, d))
        kernel = ck.PCIMQKernel()
        pre = []
        post = []
        for i in range(N):
            deltas = jit_test(
                jit(lambda *args, **kwargs: kernel.grad_x(*args, **kwargs)), x[i], y[i]
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        pval = ks_2samp(pre, post).pvalue
        self.assertLessEqual(pval, thres)

    def test_grad_y(self):
        """
        Test the performance of the PCIMQKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch sub-optimal JIT tracing.
        """
        # p-value threshold to pass/fail test
        thres = 0.05

        # number of independent observations to generate (for each of the two samples)
        N = 10

        # number of data points per 'observation'
        n = 10

        # data dimension
        d = 10
        x = np.random.random((N, n, d))
        y = np.random.random((N, n, d))
        kernel = ck.PCIMQKernel()
        pre = []
        post = []
        for i in range(N):
            deltas = jit_test(
                jit(lambda *args, **kwargs: kernel.grad_y(*args, **kwargs)), x[i], y[i]
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        pval = ks_2samp(pre, post).pvalue
        self.assertLessEqual(pval, thres)

    def test_div_x_grad_y(self):
        """
        Test the performance of the PCIMQKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch sub-optimal JIT tracing.
        """
        # p-value threshold to pass/fail test
        thres = 0.05

        # number of independent observations to generate (for each of the two samples)
        N = 10

        # number of data points per 'observation'
        n = 10

        # data dimension
        d = 10
        x = np.random.random((N, n, d))
        y = np.random.random((N, n, d))
        kernel = ck.PCIMQKernel()
        pre = []
        post = []
        for i in range(N):
            deltas = jit_test(
                jit(
                    lambda *args, **kwargs: kernel.divergence_x_grad_y(*args, **kwargs)
                ),
                x[i],
                y[i],
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        pval = ks_2samp(pre, post).pvalue
        self.assertLessEqual(pval, thres)


class TestSteinKernel(unittest.TestCase):
    """
    Tests related to the SteinKernel defined in kernel.py
    """

    def test_compute(self) -> None:
        """
        Test the performance of the SteinKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch sub-optimal JIT tracing.
        """
        # p-value threshold to pass/fail test
        thres = 0.05

        # number of independent observations to generate (for each of the two samples)
        N = 10

        # number of data points per 'observation'
        n = 10

        # data dimension
        d = 10
        x = np.random.random((N, n, d))
        y = np.random.random((N, n, d))

        def score_function(x_):
            return -x_

        kernel = ck.SteinKernel(
            base_kernel=ck.PCIMQKernel(),
            score_function=score_function,
        )
        pre = []
        post = []
        for i in range(N):
            deltas = jit_test(
                jit(lambda *args, **kwargs: kernel.compute(*args, **kwargs)), x[i], y[i]
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        pval = ks_2samp(pre, post).pvalue
        self.assertLessEqual(pval, thres)

    def test_grad_x(self):
        """
        Test the performance of the SteinKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch sub-optimal JIT tracing.
        """
        # p-value threshold to pass/fail test
        thres = 0.05

        # number of independent observations to generate (for each of the two samples)
        N = 10

        # number of data points per 'observation'
        n = 10

        # data dimension
        d = 10
        x = np.random.random((N, n, d))
        y = np.random.random((N, n, d))

        def score_function(x_):
            return -x_

        kernel = ck.SteinKernel(
            base_kernel=ck.PCIMQKernel(),
            score_function=score_function,
        )
        pre = []
        post = []
        for i in range(N):
            deltas = jit_test(
                jit(lambda *args, **kwargs: kernel.grad_x(*args, **kwargs)), x[i], y[i]
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        pval = ks_2samp(pre, post).pvalue
        self.assertLessEqual(pval, thres)

    def test_grad_y(self):
        """
        Test the performance of the SteinKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch sub-optimal JIT tracing.
        """
        # p-value threshold to pass/fail test
        thres = 0.05

        # number of independent observations to generate (for each of the two samples)
        N = 10

        # number of data points per 'observation'
        n = 10

        # data dimension
        d = 10
        x = np.random.random((N, n, d))
        y = np.random.random((N, n, d))

        def score_function(x_):
            return -x_

        kernel = ck.SteinKernel(
            base_kernel=ck.PCIMQKernel(),
            score_function=score_function,
        )
        pre = []
        post = []
        for i in range(N):
            deltas = jit_test(
                jit(lambda *args, **kwargs: kernel.grad_y(*args, **kwargs)), x[i], y[i]
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        pval = ks_2samp(pre, post).pvalue
        self.assertLessEqual(pval, thres)

    def test_div_x_grad_y(self):
        """
        Test the performance of the SteinKernel computation.

        Runs a Kolmogorov-Smirnov two-sample test on the empirical CDFs of two
        sequential function calls, in order to catch sub-optimal JIT tracing.
        """
        # p-value threshold to pass/fail test
        thres = 0.05

        # number of independent observations to generate (for each of the two samples)
        N = 10

        # number of data points per 'observation'
        n = 10

        # data dimension
        d = 10
        x = np.random.random((N, n, d))
        y = np.random.random((N, n, d))

        def score_function(x_):
            return -x_

        kernel = ck.SteinKernel(
            base_kernel=ck.PCIMQKernel(),
            score_function=score_function,
        )
        pre = []
        post = []
        for i in range(N):
            deltas = jit_test(
                jit(
                    lambda *args, **kwargs: kernel.divergence_x_grad_y(*args, **kwargs)
                ),
                x[i],
                y[i],
            )
            pre.append(deltas[0])
            post.append(deltas[1])
        pval = ks_2samp(pre, post).pvalue
        self.assertLessEqual(pval, thres)


if __name__ == "__main__":
    unittest.main()
