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
Simple coreset performance tests.

These were adapted from the `test_herding_basic` integration test.
"""

import functools

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from sklearn.datasets import make_blobs

from coreax import (
    Data,
    KernelDensityMatching,
    PCIMQKernel,
    SquaredExponentialKernel,
    SteinKernel,
)
from coreax.kernels import median_heuristic
from coreax.solvers import KernelHerding, RPCholesky, SteinThinning
from coreax.util import JITCompilableFunction

CORESET_SIZE = 100
NUM_DATA_POINTS = 4_000
NUM_FEATURES = 2


@functools.cache
def _setup_dataset():
    num_data_points = NUM_DATA_POINTS
    num_features = NUM_FEATURES
    num_cluster_centers = 6
    random_seed = 1_989
    x, *_ = make_blobs(
        num_data_points,
        n_features=num_features,
        centers=num_cluster_centers,
        random_state=random_seed,
        return_centers=True,
    )
    x = jnp.asarray(x)

    # Setup the original data object
    data = Data(x)

    # Set the bandwidth parameter of the kernel using a median heuristic derived from at
    # most 1000 random samples in the data.
    num_samples_length_scale = min(num_data_points, 1_000)
    generator = np.random.default_rng(random_seed)
    idx = generator.choice(num_data_points, num_samples_length_scale, replace=False)
    length_scale = median_heuristic(x[idx])

    return data, length_scale


def setup_herding():
    """Set up a test to compute a coreset using kernel herding."""
    # Compute a coreset using kernel herding with a squared exponential kernel.
    data, length_scale = _setup_dataset()
    herding_solver = KernelHerding(
        CORESET_SIZE, SquaredExponentialKernel(length_scale=length_scale)
    )
    return JITCompilableFunction(
        fn=herding_solver.reduce,
        fn_args=(data,),
        fn_kwargs=None,
        jit_kwargs=None,
        name="basic_herding",
    )


def setup_stein():
    """Set up a test to compute a coreset using Stein thinning."""
    # Compute a coreset using Stein thinning with a PCIMQ base kernel.
    data, length_scale = _setup_dataset()
    # We use kernel density matching rather than sliced score matching as it's much
    # faster than the sliced score matching used in the original unit test
    matcher = KernelDensityMatching(length_scale=length_scale.item())
    stein_kernel = SteinKernel(
        PCIMQKernel(length_scale=length_scale),
        matcher.match(jnp.asarray(data)),
    )
    stein_solver = SteinThinning(CORESET_SIZE, kernel=stein_kernel)
    return JITCompilableFunction(
        fn=stein_solver.reduce,
        fn_args=(data,),
        fn_kwargs=None,
        jit_kwargs=None,
        name="basic_stein",
    )


def setup_rpc():
    """Set up a test to compute a coreset using Randomly Pivoted Cholesky."""
    # Compute a coreset using RPC with a squared exponential kernel.
    data, length_scale = _setup_dataset()
    rpc_solver = RPCholesky(
        CORESET_SIZE,
        jr.key(1_234),
        SquaredExponentialKernel(length_scale=length_scale),
    )
    return JITCompilableFunction(
        fn=rpc_solver.reduce,
        fn_args=(data,),
        fn_kwargs=None,
        jit_kwargs=None,
        name="basic_rpc",
    )
