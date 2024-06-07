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
Example coreset generation using Gaussian features and a non-linear response.

This example showcases how a coreset can be generated from a dataset containing ``n``
pairs of features sampled from a Gaussian distribution with corresponding responses
generated with a non-linear relationship to the features.

A coreset is generated using GreedyCMMD, with a Squared Exponential kernel for both the
features and the response. This coreset is compared to a coreset generated via uniform
random sampling. Coreset quality is measured using conditional maximum mean discrepancy
(CMMD).
"""

from pathlib import Path
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from sklearn.preprocessing import StandardScaler

from coreax import CMMD, SquaredExponentialKernel, SupervisedData
from coreax.kernel import median_heuristic
from coreax.solvers import GreedyCMMD, RandomSample


# Examples are written to be easy to read, copy and paste by users, so we ignore the
# pylint warnings raised that go against this approach
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=duplicate-code
def main(out_path: Optional[Path] = None) -> tuple[float, float]:
    """
    Run the basic GreedyCMMD on tabular data example.

    Generate a set of features from a Gaussian distribution, generate response with a
    non-linear relationship to the features and Gaussian errors. Generate a coreset via
    GreedyCMMD. Compare results to coresets generated via uniform random sampling.
    Coreset quality is measured using conditional maximum mean discrepancy (CMMD).

    :param out_path: Path to save output to, if not :data:`None`, assumed relative to
        this module file unless an absolute path is given
    :return: Coreset CMMD, random sample CMMD
    """
    print("Generating data...")
    # Generate features from normal distribution and produce response
    # with non-linear relationship to the features with normal errors.
    num_data_points = 1_000
    feature_sd = 20
    response_sd = 0.5
    random_seed = 2_024
    generator = np.random.default_rng(random_seed)

    x = generator.multivariate_normal(
        np.zeros(1), feature_sd * np.eye(1), num_data_points
    )
    epsilon = generator.multivariate_normal(
        np.zeros(1), response_sd * np.eye(1), num_data_points
    )
    y = 1 / 200 * x**3 + np.sin(x) + epsilon

    # Standardise the data and setup SupervisedData object
    feature_scaler = StandardScaler().fit(x)
    x = jnp.array(feature_scaler.transform(x))
    response_scaler = StandardScaler().fit(y)
    y = jnp.array(response_scaler.transform(y))
    supervised_data = SupervisedData(data=x, supervision=y)

    # Request 50 coreset points
    coreset_size = 50

    # Set the bandwidth parameter of the kernel using a median heuristic derived from at
    # most 1000 random samples in the data.
    num_samples_length_scale = min(num_data_points, 1_000)
    generator = np.random.default_rng(random_seed)
    idx = generator.choice(num_data_points, num_samples_length_scale, replace=False)

    feature_length_scale = median_heuristic(x[idx])
    feature_kernel = SquaredExponentialKernel(length_scale=feature_length_scale)
    response_length_scale = median_heuristic(y[idx])
    response_kernel = SquaredExponentialKernel(length_scale=response_length_scale)

    print("Computing GreedyCMMD coreset...")
    # Compute a coreset using GreedyCMMD with squared exponential kernels
    regularisation_parameter = 1e-6
    cmmd_key, sample_key = random.split(random.key(random_seed), num=2)
    cmmd_solver = GreedyCMMD(
        coreset_size=coreset_size,
        random_key=cmmd_key,
        feature_kernel=feature_kernel,
        response_kernel=response_kernel,
        regularisation_parameter=regularisation_parameter,
        unique=True,
    )
    cmmd_coreset, _ = eqx.filter_jit(cmmd_solver.reduce)(supervised_data)

    print("Choosing random subset...")
    # Generate a coreset via uniform random sampling for comparison
    random_solver = RandomSample(coreset_size, sample_key, unique=True)
    random_coreset, _ = eqx.filter_jit(random_solver.reduce)(supervised_data)

    # Define reference kernels to use for comparisons of CMMD. We'll use normalised
    # SquaredExponentialKernels (which is also a Gaussian kernel)
    print("Computing CMMD...")
    feature_cmmd_kernel = SquaredExponentialKernel(
        length_scale=feature_length_scale,
        output_scale=1.0 / (feature_length_scale * jnp.sqrt(2.0 * jnp.pi)),
    )
    response_cmmd_kernel = SquaredExponentialKernel(
        length_scale=response_length_scale,
        output_scale=1.0 / (response_length_scale * jnp.sqrt(2.0 * jnp.pi)),
    )

    # Compute CMMD between the original data and the coreset generated via GreedyCMMD
    cmmd_metric = CMMD(
        feature_kernel=feature_cmmd_kernel,
        response_kernel=response_cmmd_kernel,
        regularisation_parameter=regularisation_parameter,
    )
    greedy_cmmd = cmmd_coreset.compute_metric(cmmd_metric)

    # Compute the CMMD between the original data and the coreset generated via random
    # sampling
    random_cmmd = cmmd_coreset.compute_metric(cmmd_metric)

    # Print the CMMD values
    print(f"Random sampling coreset CMMD: {random_cmmd}")
    print(f"GreedyCMMD coreset CMMD: {greedy_cmmd}")

    # Produce some scatter plots (assume 1-dimensional features and response)
    plt.scatter(x[:, 0], y[:, 1], s=2.0, alpha=0.1)
    plt.scatter(
        cmmd_coreset.coreset.data[:, 0],
        cmmd_coreset.coreset.data[:, 1],
        s=10,
        color="red",
    )
    plt.axis("off")
    plt.title(f"GreedyCMMD, m={coreset_size}, " f"CMMD={round(float(greedy_cmmd), 6)}")
    plt.show()

    plt.scatter(x[:, 0], y[:, 1], s=2.0, alpha=0.1)
    plt.scatter(
        random_coreset.coreset.data[:, 0],
        random_coreset.coreset.data[:, 1],
        s=10,
        color="red",
    )
    plt.title(f"Random, m={coreset_size}, " f"CMMD={round(float(random_cmmd), 6)}")
    plt.axis("off")

    if out_path is not None:
        if not out_path.is_absolute():
            out_path = Path(__file__).parent.joinpath(out_path)
        plt.savefig(out_path)

    plt.show()

    return float(greedy_cmmd), float(random_cmmd)


# pylint: enable=too-many-locals
# pylint: enable=duplicate-code


if __name__ == "__main__":
    main()
