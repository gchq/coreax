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

A coreset is generated using kernel herding, with a kernel that is created by adding and
taking products of Squared exponential, PCIMQ and Laplace kernels. This coreset is
compared to a coreset generated via uniform random sampling. Coreset quality is measured
using maximum mean discrepancy (MMD).
"""

from pathlib import Path
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from sklearn.datasets import make_blobs

from coreax import (
    MMD,
    Data,
    LaplacianKernel,
    PCIMQKernel,
    SquaredExponentialKernel,
)
from coreax.kernel import median_heuristic
from coreax.solvers import KernelHerding, RandomSample


# Examples are written to be easy to read, copy and paste by users, so we ignore the
# pylint warnings raised that go against this approach
# pylint: disable=too-many-locals
# pylint: disable=duplicate-code
def main(out_path: Optional[Path] = None) -> tuple[float, float, float, float]:
    """
    Run the basic herding on tabular data example.

    Generate a set of points from distinct clusters in a plane. Generate a coreset via
    kernel herding. Compare results to coresets generated via uniform random sampling.
    Coreset quality is measured using maximum mean discrepancy (MMD).

    :param out_path: Path to save output to, if not :data:`None`, assumed relative to
        this module file unless an absolute path is given
    :return: Coreset MMD, random sample MMD
    """
    # Create some data. Here we'll use 10,000 points in 2D from 6 distinct clusters. 2D
    # for plotting below.
    num_data_points = 10_000
    num_features = 2
    num_cluster_centers = 6
    random_seed = 1_989
    x, *_ = make_blobs(
        num_data_points,
        n_features=num_features,
        centers=num_cluster_centers,
        random_state=random_seed,
        return_centers=True,
    )

    # Request 100 coreset points
    coreset_size = 100

    # Setup the original data object
    data = Data(x)

    # Set the bandwidth parameter of the kernel using a median heuristic derived from at
    # most 1000 random samples in the data.
    num_samples_length_scale = min(num_data_points, 1_000)
    generator = np.random.default_rng(random_seed)
    idx = generator.choice(num_data_points, num_samples_length_scale, replace=False)
    length_scale = median_heuristic(x[idx])

    # Add together and take products of kernels
    kernel = SquaredExponentialKernel(length_scale=length_scale) + PCIMQKernel(
        length_scale=length_scale
    ) * LaplacianKernel(length_scale=length_scale)

    print("Computing herding coreset...")
    # Compute a coreset using kernel herding
    sample_key, _ = random.split(random.key(random_seed))
    herding_solver = KernelHerding(coreset_size, kernel)
    herding_coreset, _ = eqx.filter_jit(herding_solver.reduce)(data)

    print("Choosing random subset...")
    # Generate a coreset via uniform random sampling for comparison
    random_solver = RandomSample(coreset_size, sample_key, unique=True)
    random_coreset, _ = eqx.filter_jit(random_solver.reduce)(data)

    # Define a reference kernel to use for comparisons of MMD. We'll use a normalised
    # SquaredExponentialKernel (which is also a Gaussian kernel).
    print("Computing MMD...")
    mmd_kernel = SquaredExponentialKernel(
        length_scale=length_scale,
        output_scale=1.0 / (length_scale * jnp.sqrt(2.0 * jnp.pi)),
    )

    # Compute the MMD between the original data and the coreset generated via herding
    mmd_metric = MMD(kernel=mmd_kernel)
    herding_mmd = herding_coreset.compute_metric(mmd_metric)

    # Compute the MMD between the original data and the coreset generated via random
    # sampling
    random_mmd = random_coreset.compute_metric(mmd_metric)

    # Print the MMD values
    print(f"Random sampling coreset MMD: {random_mmd}")
    print(f"Herding coreset MMD: {herding_mmd}")

    # Produce some scatter plots (assume 2-dimensional data)
    plt.scatter(x[:, 0], x[:, 1], s=2.0, alpha=0.1)
    plt.scatter(
        herding_coreset.coreset.data[:, 0],
        herding_coreset.coreset.data[:, 1],
        s=10,
        color="red",
    )
    plt.axis("off")
    plt.title(
        f"Kernel herding, m={coreset_size}, " f"MMD={round(float(herding_mmd), 6)}"
    )
    plt.show()

    plt.scatter(x[:, 0], x[:, 1], s=2.0, alpha=0.1)
    plt.scatter(
        random_coreset.coreset.data[:, 0],
        random_coreset.coreset.data[:, 1],
        s=10,
        color="red",
    )
    plt.title(f"Random, m={coreset_size}, " f"MMD={round(float(random_mmd), 6)}")
    plt.axis("off")

    if out_path is not None:
        if not out_path.is_absolute():
            out_path = Path(__file__).parent.joinpath(out_path)
        plt.savefig(out_path)

    plt.show()

    return (
        float(herding_mmd),
        float(random_mmd),
    )


# pylint: enable=too-many-locals
# pylint: enable=duplicate-code


if __name__ == "__main__":
    main()
