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

This example showcases how a coreset can be generated from a supervised dataset
containing ``n`` data pairs consisting of features sampled from a Gaussian distribution
with corresponding responses generated with a non-linear relationship to the features.

A coreset is generated using :class:`~coreax.solvers.JointKernelHerding` and
:class:`~coreax.solvers.JointRPCholesky`, with a
:class:`~coreax.kernel.SquaredExponentialKernel` for both the features and the response.
These coresets are compared to a coreset generated via uniform random sampling. Coreset
quality is measured using joint maximum mean discrepancy (JMMD).
"""

from pathlib import Path
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from sklearn.preprocessing import StandardScaler

from coreax import JMMD, SquaredExponentialKernel, SupervisedData
from coreax.kernel import median_heuristic
from coreax.solvers import JointKernelHerding, JointRPCholesky, RandomSample


# Examples are written to be easy to read, copy and paste by users, so we ignore the
# pylint warnings raised that go against this approach
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=duplicate-code
def main(out_path: Optional[Path] = None) -> tuple[float, float]:
    """
    Run the basic herding on tabular supervised data example.

    Generate a set of features from a Gaussian distribution, generate the response with
    a non-linear relationship to the features with Gaussian errors. Generate a coreset
    via `JointKernelHerding` and `JointRPCholesky`. Compare results to coresets
    generated via uniform random sampling. Coreset quality is measured using joint
    maximum mean discrepancy (JMMD).

    :param out_path: Path to save output to, if not :data:`None`, assumed relative to
        this module file unless an absolute path is given
    :return: Coreset JMMD, random sample JMMD
    """
    print("Generating data...")
    # Generate features from normal distribution and produce response
    # with non-linear relationship to the features with normal errors.
    num_data_points = 10000
    feature_sd = 1
    response_sd = 0.1
    random_seed = 2_024
    generator = np.random.default_rng(random_seed)

    x = generator.multivariate_normal(
        np.zeros(1), feature_sd * np.eye(1), num_data_points
    )
    epsilon = generator.multivariate_normal(
        np.zeros(1), response_sd * np.eye(1), num_data_points
    )
    y = 1 / 10 * x**3 + jnp.sin(2 * x) + epsilon

    # Standardise the data and setup SupervisedData object
    feature_scaler = StandardScaler().fit(x)
    x = jnp.array(feature_scaler.transform(x))
    response_scaler = StandardScaler().fit(y)
    y = jnp.array(response_scaler.transform(y))
    supervised_data = SupervisedData(data=x, supervision=y)

    # Request 250 coreset points
    coreset_size = 250

    # Set the bandwidth parameter of the kernel using a median heuristic derived from at
    # most 1000 random samples in the data.
    num_samples_length_scale = min(num_data_points, 1_000)
    generator = np.random.default_rng(random_seed)
    idx = generator.choice(num_data_points, num_samples_length_scale, replace=False)

    feature_length_scale = median_heuristic(x[idx])
    feature_kernel = SquaredExponentialKernel(length_scale=feature_length_scale)
    response_length_scale = median_heuristic(y[idx])
    response_kernel = SquaredExponentialKernel(length_scale=response_length_scale)

    print("Computing JointKernelHerding coreset...")
    # Compute a coreset using JointKernelHerding with squared exponential kernels
    cholesky_key, sample_key = random.split(random.key(random_seed), num=2)
    herding_solver = JointKernelHerding(
        coreset_size=coreset_size,
        feature_kernel=feature_kernel,
        response_kernel=response_kernel,
        unique=True,
    )
    herding_coreset, _ = eqx.filter_jit(herding_solver.reduce)(supervised_data)

    print("Computing JointRPCholesky coreset...")
    # Compute a coreset using JointRPCholesky with squared exponential kernels
    cholesky_solver = JointRPCholesky(
        coreset_size=coreset_size,
        random_key=cholesky_key,
        feature_kernel=feature_kernel,
        response_kernel=response_kernel,
        unique=True,
    )
    cholesky_coreset, _ = eqx.filter_jit(cholesky_solver.reduce)(supervised_data)

    print("Choosing random subset...")
    # Generate a coreset via uniform random sampling for comparison
    random_solver = RandomSample(coreset_size, sample_key, unique=True)
    random_coreset, _ = eqx.filter_jit(random_solver.reduce)(supervised_data)

    # Define reference kernels to use for comparisons of JMMD. We'll use normalised
    # SquaredExponentialKernels (which is also a Gaussian kernel)
    print("Computing JMMD...")
    feature_jmmd_kernel = SquaredExponentialKernel(
        length_scale=feature_length_scale,
        output_scale=1.0 / (feature_length_scale * jnp.sqrt(2.0 * jnp.pi)),
    )
    response_jmmd_kernel = SquaredExponentialKernel(
        length_scale=response_length_scale,
        output_scale=1.0 / (response_length_scale * jnp.sqrt(2.0 * jnp.pi)),
    )

    # Compute JMMD between the original data and the coreset generated via
    # JointKernelHerding
    jmmd_metric = JMMD(
        feature_kernel=feature_jmmd_kernel,
        response_kernel=response_jmmd_kernel,
    )
    herding_jmmd = eqx.filter_jit(herding_coreset.compute_metric)(jmmd_metric)

    # Compute JMMD between the original data and the coreset generated via
    # JointRPCholesky
    cholesky_jmmd = eqx.filter_jit(cholesky_coreset.compute_metric)(jmmd_metric)

    # Compute the JMMD between the original data and the coreset generated via random
    # sampling
    random_jmmd = eqx.filter_jit(random_coreset.compute_metric)(jmmd_metric)

    # Print the JMMD values
    print(f"Random sampling coreset JMMD: {random_jmmd}")
    print(f"JointKernelHerding coreset JMMD: {herding_jmmd}")
    print(f"JointRPCholesky coreset JMMD: {cholesky_jmmd}")

    # Produce some scatter plots (assume 1-dimensional features and response)
    plt.scatter(x[:, 0], y[:, 1], s=2.0, alpha=0.1)
    plt.scatter(
        herding_coreset.coreset.data[:, 0],
        herding_coreset.coreset.data[:, 1],
        s=10,
        color="red",
    )
    plt.axis("off")
    plt.title(
        f"JointKernelHerding, m={coreset_size}, " f"J={round(float(herding_jmmd), 6)}"
    )
    plt.show()

    plt.scatter(x[:, 0], y[:, 1], s=2.0, alpha=0.1)
    plt.scatter(
        cholesky_coreset.coreset.data[:, 0],
        cholesky_coreset.coreset.data[:, 1],
        s=10,
        color="red",
    )
    plt.axis("off")
    plt.title(
        f"JointRPCholesky, m={coreset_size}, " f"JMMD={round(float(cholesky_jmmd), 6)}"
    )
    plt.show()

    plt.scatter(x[:, 0], y[:, 1], s=2.0, alpha=0.1)
    plt.scatter(
        random_coreset.coreset.data[:, 0],
        random_coreset.coreset.data[:, 1],
        s=10,
        color="red",
    )
    plt.title(f"Random, m={coreset_size}, " f"JMMD={round(float(random_jmmd), 6)}")
    plt.axis("off")

    if out_path is not None:
        if not out_path.is_absolute():
            out_path = Path(__file__).parent.joinpath(out_path)
        plt.savefig(out_path)

    plt.show()

    return float(herding_jmmd), float(random_jmmd)


# pylint: enable=too-many-locals
# pylint: enable=duplicate-code


if __name__ == "__main__":
    main()
