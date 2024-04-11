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

A coreset is generated using GreedyCMMD, with a Squared Exponential kernel for both the features
and the repsonse. This coreset is compared to a coreset generated via uniform random sampling.
Coreset quality is measured using conditional maximum mean discrepancy (CMMD).

To reduce computational requirements, a map reduce approach is used, splitting the
original dataset into distinct segments, with each segment handled on a different
process.
"""

# Support annotations with | in Python < 3.10
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

from coreax import (
    CMMD,
    ArrayData,
    GreedyCMMD,
    RandomSample,
    MapReduce,
    SizeReduce,
    SquaredExponentialKernel,
)
from coreax.kernel import median_heuristic

def main(out_path: Path | None = None) -> tuple[float, float]:
    """
    Run the basic GreedyCMMD on tabular data example.

    Generate a set of features from a Gaussian distribution, generate response with a non-linear 
    relationship to the features and Gaussian errors. Generate a coreset via
    GreedyCMMD. Compare results to coresets generated via uniform random sampling.
    Coreset quality is measured using conditional maximum mean discrepancy (CMMD).

    To reduce computational requirements, a map reduce approach is used, splitting the
    original dataset into distinct segments, with each segment handled on a different
    process.

    :param out_path: Path to save output to, if not :data:`None`, assumed relative to
        this module file unless an absolute path is given
    :return: Coreset CMMD, random sample CMMD
    """
    print("Generating data...")
    # Generate features from normal distribution and produce response 
    # with non-linear relationship to the features with normal erros.
    num_data_points = 1_000
    num_features = 1
    feature_sd = 20
    response_sd = 0.5
    random_seed = 2_000
    generator = np.random.default_rng(random_seed)
    
    x = generator.multivariate_normal(np.zeros(num_features), feature_sd * np.eye(num_features), num_data_points)
    epsilon = generator.multivariate_normal(np.zeros(1), response_sd * np.eye(1), num_data_points)
    y = 1/200*x**3 + np.sin(x) + epsilon
    
    # Standardise the data and stack it into one array
    x = ( x - x.mean() ) / x.std()
    y = ( y - y.mean() ) / y.std()
    D = jnp.hstack((x, y))
    
    # Request 50 coreset points
    coreset_size = 50

    # Setup the original data object
    data = ArrayData.load(D)

    # Set the bandwidth parameter of the feature and response kernels using a median 
    # heuristic derived from at most 1000 random samples in the data.
    num_samples_length_scale = min(num_data_points, 1_000)
    generator = np.random.default_rng(random_seed)
    idx = generator.choice(num_data_points, num_samples_length_scale, replace=False)
    feature_length_scale = median_heuristic(x[idx])
    response_length_scale = median_heuristic(y[idx])

    print("Computing coreset...")
    # Compute a coreset using GreedyCMMD with Squared exponential kernels.
    build_key, sample_key = random.split(random.key(random_seed))
    greedy_cmmd = GreedyCMMD(
        random_key=build_key,
        feature_kernel=SquaredExponentialKernel(length_scale=feature_length_scale),
        response_kernel = SquaredExponentialKernel(length_scale=response_length_scale),
        num_feature_dimensions = num_features
    )
    greedy_cmmd.fit(
        original_data=data,
        strategy=MapReduce(coreset_size=coreset_size, leaf_size=100)
    )

    print("Choosing random subset...")
    # Generate a coreset via uniform random sampling for comparison
    random_sample_object = RandomSample(sample_key, unique=True)
    random_sample_object.fit(
        original_data=data, strategy=SizeReduce(coreset_size=coreset_size)
    )

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

    # Compute the CMMD between the original data and the coreset generated via GreedyCMMD
    metric_object = normalised_metric_object = CMMD(
        feature_kernel=feature_cmmd_kernel,
        response_kernel=response_cmmd_kernel,
        num_feature_dimensions=1,
    )
    conditional_maximum_mean_discrepancy_greedy = greedy_cmmd.compute_metric(metric_object)

    # Compute the CMMD between the original data and the coreset generated via random
    # sampling
    conditional_maximum_mean_discrepancy_random = random_sample_object.compute_metric(metric_object)

    # Print the CMMD values
    print(f"Random sampling coreset CMMD: {conditional_maximum_mean_discrepancy_random}")
    print(f"GreedyCMMD coreset CMMD: {conditional_maximum_mean_discrepancy_greedy}")

    # Produce some scatter plots (assume 1-dimensional features and response)
    plt.scatter(x, y, s=2.0, alpha=0.5, color = 'black')
    plt.scatter(
        greedy_cmmd.coreset[:, 0],
        greedy_cmmd.coreset[:, 1],
        s=50,
        color="red",
        ec='black'
    )
    plt.axis("off")
    plt.title(
        f"GreedyCMMD, m={coreset_size}, "
        f"CMMD={round(conditional_maximum_mean_discrepancy_greedy.item(), 6)}"
    )
    plt.show()

    plt.scatter(x, y, s=2.0, alpha=0.5, color = 'black')
    plt.scatter(
        random_sample_object.coreset[:, 0],
        random_sample_object.coreset[:, 1],
        s=50,
        color="red",
        ec='black'
    )
    plt.title(
        f"Random, m={coreset_size}, "
        f"CMMD={round(conditional_maximum_mean_discrepancy_random.item(), 6)}"
    )
    plt.axis("off")

    if out_path is not None:
        if not out_path.is_absolute():
            out_path = Path(__file__).parent / out_path
        plt.savefig(out_path)

    plt.show()

    return (
        float(conditional_maximum_mean_discrepancy_greedy),
        float(conditional_maximum_mean_discrepancy_random),
    )

if __name__ == "__main__":
    main()