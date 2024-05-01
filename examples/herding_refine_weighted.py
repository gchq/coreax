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
Example coreset generation using randomly generated point clouds.

This example showcases how a coreset can be generated from a dataset containing ``n``
points sampled from ``k`` clusters in space.

A coreset is generated using kernel herding, with a Squared Exponential kernel. This
coreset is then refined to improve quality.

The coreset generated from the above process is compared to a coreset generated via
uniform random sampling. Coreset quality is measured using maximum mean discrepancy
(MMD).
"""

# Support annotations with | in Python < 3.10
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from sklearn.datasets import make_blobs

from coreax import (
    MMD,
    ArrayData,
    KernelHerding,
    PCIMQKernel,
    RandomSample,
    RPCholesky,
    SizeReduce,
    SlicedScoreMatching,
    SquaredExponentialKernel,
    SteinThinning,
)
from coreax.kernel import median_heuristic
from coreax.refine import RefineRegular
from coreax.weights import MMDWeightsOptimiser


# Examples are written to be easy to read, copy and paste by users, so we ignore the
# pylint warnings raised that go against this approach
# pylint: disable=too-many-locals
# pylint: disable=duplicate-code
def main(out_path: Path | None = None) -> tuple[float, float]:
    """
    Run the kernel herding on tabular data with a refine post-processing step.

    Generate a set of points from distinct clusters in a plane. Generate a coreset via
    kernel herding. After generation, the coreset is improved by refining it (a greedy
    approach to replace points in the coreset for those that improve some measure of
    coreset quality). Compare results to coresets generated via uniform random sampling.
    Coreset quality is measured using maximum mean discrepancy (MMD).

    :param out_path: Path to save output to, if not :data:`None`, assumed relative to
        this module file unless an absolute path is given
    :return: Coreset MMD, random sample MMD
    """
    # pylint: disable=too-many-statements
    # Create some data. Here we'll use 10,000 points in 2D from 6 distinct clusters. 2D
    # for plotting below.
    num_data_points = 10_000
    num_features = 2
    num_cluster_centers = 6
    random_seed = 1_989
    x, _, _centers = make_blobs(
        num_data_points,
        n_features=num_features,
        centers=num_cluster_centers,
        random_state=random_seed,
        return_centers=True,
    )

    # Request 100 coreset points
    coreset_size = 100

    # Setup the original data object
    data = ArrayData.load(x)

    # Set the bandwidth parameter of the kernel using a median heuristic derived from at
    # most 1000 random samples in the data.
    num_samples_length_scale = min(num_data_points, 1_000)
    generator = np.random.default_rng(random_seed)
    idx = generator.choice(num_data_points, num_samples_length_scale, replace=False)
    length_scale = median_heuristic(x[idx])

    # Define a refinement object
    refiner = RefineRegular()
    kernel = SquaredExponentialKernel(length_scale=length_scale)
    print("Computing herding coreset...")
    weights_optimiser_herding = MMDWeightsOptimiser(kernel=kernel)
    # Compute a coreset using kernel herding with a squared exponential kernel.
    herding_key, sample_key, rp_key, stein_key = random.split(
        random.key(random_seed), num=4
    )
    herding_object = KernelHerding(
        herding_key,
        kernel=kernel,
        refine_method=refiner,
        weights_optimiser=weights_optimiser_herding,
    )
    herding_object.fit(
        original_data=data, strategy=SizeReduce(coreset_size=coreset_size)
    )
    weights_herding = herding_object.solve_weights()

    print("Computing Stein thinning coreset...")
    # Compute a coreset using Stein thinning with a PCIMQ base kernel.
    weights_optimiser_stein = MMDWeightsOptimiser(kernel=kernel)
    base_kernel = PCIMQKernel(length_scale=length_scale)
    _, subkey = random.split(stein_key)
    sliced_score_matcher = SlicedScoreMatching(
        subkey,
        random.rademacher,
        use_analytic=True,
        num_random_vectors=100,
        learning_rate=0.001,
        num_epochs=50,
    )
    stein_object = SteinThinning(
        stein_key,
        kernel=base_kernel,
        weights_optimiser=weights_optimiser_stein,
        score_method=sliced_score_matcher,
    )
    stein_object.fit(original_data=data, strategy=SizeReduce(coreset_size=coreset_size))
    weights_stein = stein_object.solve_weights()

    print("Computing RPC coreset...")
    weights_optimiser_rpc = MMDWeightsOptimiser(kernel=kernel)
    # Compute a coreset using RPC with a squared exponential kernel.
    rp_object = RPCholesky(
        rp_key,
        kernel=SquaredExponentialKernel(length_scale=length_scale),
        refine_method=refiner,
        weights_optimiser=weights_optimiser_rpc,
    )
    rp_object.fit(original_data=data, strategy=SizeReduce(coreset_size=coreset_size))
    weights_rpc = rp_object.solve_weights()

    print("Choosing random subset...")
    # Generate a coreset via uniform random sampling for comparison
    random_sample_object = RandomSample(sample_key, unique=True)
    random_sample_object.fit(
        original_data=data, strategy=SizeReduce(coreset_size=coreset_size)
    )

    # Define a reference kernel to use for comparisons of MMD. We'll use a normalised
    # SquaredExponentialKernel (which is also a Gaussian kernel)
    print("Computing MMD...")
    mmd_kernel = SquaredExponentialKernel(
        length_scale=length_scale,
        output_scale=1.0 / (length_scale * jnp.sqrt(2.0 * jnp.pi)),
    )

    # Compute the MMD between the original data and the coreset generated via herding
    metric_object = MMD(kernel=mmd_kernel)
    maximum_mean_discrepancy_herding = herding_object.compute_metric(
        metric_object, weights_y=weights_herding
    )

    # Compute the MMD between the original data and the coreset generated via RPC
    maximum_mean_discrepancy_rpc = rp_object.compute_metric(
        metric_object, weights_y=weights_rpc
    )

    # Compute the MMD between the original data and the coreset generated via ST
    maximum_mean_discrepancy_stein = stein_object.compute_metric(
        metric_object, weights_y=weights_stein
    )

    # Compute the MMD between the original data and the coreset generated via random
    # sampling
    maximum_mean_discrepancy_random = random_sample_object.compute_metric(metric_object)

    # Print the MMD values
    print(f"Random sampling coreset MMD: {maximum_mean_discrepancy_random}")
    print(f"Herding coreset MMD: {maximum_mean_discrepancy_herding}")
    print(f"Stein thinning coreset MMD: {maximum_mean_discrepancy_stein}")
    print(f"RPC coreset MMD: {maximum_mean_discrepancy_rpc}")

    # Produce some scatter plots (assume 2-dimensional data)
    plt.scatter(x[:, 0], x[:, 1], s=2.0, alpha=0.1)
    plt.scatter(
        herding_object.coreset[:, 0],
        herding_object.coreset[:, 1],
        s=10,
        color="red",
    )
    plt.axis("off")
    plt.title(
        f"Kernel herding, m={coreset_size}, "
        f"MMD={round(float(maximum_mean_discrepancy_herding), 6)}"
    )
    plt.show()

    plt.scatter(x[:, 0], x[:, 1], s=2.0, alpha=0.1)
    plt.scatter(
        rp_object.coreset[:, 0],
        rp_object.coreset[:, 1],
        s=10,
        color="red",
    )
    plt.axis("off")
    plt.title(
        f"RP Cholesky, m={coreset_size}, "
        f"MMD={round(float(maximum_mean_discrepancy_rpc), 6)}"
    )
    plt.show()

    plt.scatter(x[:, 0], x[:, 1], s=2.0, alpha=0.1)
    plt.scatter(
        stein_object.coreset[:, 0],
        stein_object.coreset[:, 1],
        s=10,
        color="red",
    )
    plt.axis("off")
    plt.title(
        f"Stein thinning, m={coreset_size}, "
        f"MMD={round(float(maximum_mean_discrepancy_stein), 6)}"
    )
    plt.show()

    plt.scatter(x[:, 0], x[:, 1], s=2.0, alpha=0.1)
    plt.scatter(
        random_sample_object.coreset[:, 0],
        random_sample_object.coreset[:, 1],
        s=10,
        color="red",
    )
    plt.title(
        f"Random, m={coreset_size}, "
        f"MMD={round(float(maximum_mean_discrepancy_random), 6)}"
    )
    plt.axis("off")

    if out_path is not None:
        if not out_path.is_absolute():
            out_path = Path(__file__).parent.joinpath(out_path)
        plt.savefig(out_path)

    plt.show()

    return (
        float(maximum_mean_discrepancy_herding),
        float(maximum_mean_discrepancy_rpc),
        float(maximum_mean_discrepancy_stein),
        float(maximum_mean_discrepancy_random),
    )


# pylint: enable=too-many-locals
# pylint: enable=duplicate-code


if __name__ == "__main__":
    main()
