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
Example coreset generation using an image of the statue of David.

This example showcases how a coreset can be generated from image data. In this context,
a coreset is a set of pixels that best capture the information in the original image.

The coreset is generated using scalable Stein kernel herding, with a PCIMQ base kernel.
The score function (gradient of the log-density function) for the Stein kernel is
estimated by applying kernel density estimation (KDE) to the data, and then taking
gradients.

The initial coreset generated from this procedure is then weighted, with weights
determined such that the weighted coreset achieves a better maximum mean discrepancy
when compared to the original dataset than the unweighted coreset.

To reduce computational requirements, a map reduce approach is used, splitting the
original dataset into distinct segments, with each segment handled on a different
process.

The coreset attained from Stein kernel herding is compared to a coreset generated via
uniform random sampling. Coreset quality is measured using maximum mean discrepancy
(MMD).
"""

# Support annotations with | in Python < 3.10
from __future__ import annotations

from pathlib import Path

import cv2
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import linen
from jax import random

from coreax import (
    MMD,
    ArrayData,
    KernelDensityMatching,
    KernelHerding,
    MapReduce,
    RandomSample,
    SizeReduce,
    SquaredExponentialKernel,
    SteinKernel,
)
from coreax.kernel import PCIMQKernel, median_heuristic
from coreax.weights import MMDWeightsOptimiser

MAX_8BIT = 255
MIN_LENGTH_SCALE = 1e-6


# Examples are written to be easy to read, copy and paste by users, so we ignore the
# pylint warnings raised that go against this approach
# pylint: disable=no-member
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=duplicate-code
def main(
    in_path: Path = Path("../examples/data/david_orig.png"),
    out_path: Path | None = None,
    downsampling_factor: int = 1,
) -> tuple[float, float]:
    """
    Run the 'david' example for image sampling.

    Take an image of the statue of David and then generate a coreset using
    scalable Stein kernel herding.

    The initial coreset generated from this procedure is then weighted, with weights
    determined such that the weighted coreset achieves a better maximum mean discrepancy
    when compared to the original dataset than the unweighted coreset.

    To reduce computational requirements, a map reduce approach is used, splitting the
    original dataset into distinct segments, with each segment handled on a different
    process.

    Compare the result from this to a coreset generated
    via uniform random sampling. Coreset quality is measured using maximum mean
    discrepancy (MMD).

    :param in_path: Path to input image, assumed relative to this module file unless an
        absolute path is given
    :param out_path: Path to save output to, if not :data:`None`, assumed relative to
        this module file unless an absolute path is given
    :param downsampling_factor: the window size to average (downsample) the images over.
    :return: Coreset MMD, random sample MMD
    """
    # Convert to absolute paths
    if not in_path.is_absolute():
        in_path = Path(__file__).parent.joinpath(in_path)
    if out_path is not None and not out_path.is_absolute():
        out_path = Path(__file__).parent.joinpath(out_path)

    # Path to original image
    original_data = cv2.imread(str(in_path))
    image_data = np.asarray(cv2.cvtColor(original_data, cv2.COLOR_BGR2GRAY))
    # Pool/downsample the image
    window_shape = (downsampling_factor, downsampling_factor)
    pooled_image_data = linen.avg_pool(
        image_data[..., None], window_shape, strides=window_shape
    )[..., 0]

    print(f"Image dimensions: {pooled_image_data.shape}")
    pre_coreset_data = np.column_stack(np.nonzero(pooled_image_data < MAX_8BIT))
    pixel_values = pooled_image_data[pooled_image_data < MAX_8BIT]
    pre_coreset_data = np.column_stack((pre_coreset_data, pixel_values)).astype(
        np.float32
    )
    num_data_points = pre_coreset_data.shape[0]

    # Request coreset points
    coreset_size = 8_000 // downsampling_factor

    # Setup the original data object
    data = ArrayData.load(pre_coreset_data)

    # Set the length_scale parameter of the kernel from at most 1000 samples
    num_samples_length_scale = min(num_data_points, 1000 // downsampling_factor)
    random_seed = 1_989
    generator = np.random.default_rng(random_seed)
    idx = generator.choice(num_data_points, num_samples_length_scale, replace=False)
    length_scale = median_heuristic(pre_coreset_data[idx].astype(float))
    if length_scale < MIN_LENGTH_SCALE:
        length_scale = 100.0

    # Learn a score function via kernel density estimation (this is required for
    # evaluation of the Stein kernel)
    kernel_density_score_matcher = KernelDensityMatching(
        length_scale=length_scale, kde_data=pre_coreset_data[idx, :]
    )
    score_function = kernel_density_score_matcher.match()

    # Define a kernel to use for herding
    herding_kernel = SteinKernel(
        PCIMQKernel(length_scale=length_scale),
        score_function=score_function,
    )

    # Define a weights optimiser to learn optimal weights for the coreset after creation
    weights_optimiser = MMDWeightsOptimiser(kernel=herding_kernel)

    print("Computing coreset...")
    # Compute a coreset using kernel herding with a Stein kernel. To reduce compute
    # time, we apply MapReduce, which partitions the input into blocks for independent
    # coreset solving. We also reduce memory requirements by specifying block size
    herding_key, sample_key = random.split(random.key(random_seed))
    herding_object = KernelHerding(
        herding_key,
        kernel=herding_kernel,
        weights_optimiser=weights_optimiser,
        block_size=1_000 // downsampling_factor,
    )
    herding_object.fit(
        original_data=data,
        strategy=MapReduce(
            coreset_size=coreset_size, leaf_size=10_000 // downsampling_factor
        ),
    )
    herding_weights = herding_object.solve_weights()

    print("Choosing random subset...")
    # Generate a coreset via uniform random sampling for comparison
    random_sample_object = RandomSample(sample_key, unique=True)
    random_sample_object.fit(
        original_data=data,
        strategy=SizeReduce(coreset_size=coreset_size),
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
        metric_object, block_size=1_000 // downsampling_factor
    )

    # Compute the MMD between the original data and the coreset generated via random
    # sampling
    maximum_mean_discrepancy_random = random_sample_object.compute_metric(
        metric_object, block_size=1_000 // downsampling_factor
    )

    # Print the MMD values
    print(f"Random sampling coreset MMD: {maximum_mean_discrepancy_random}")
    print(f"Herding coreset MMD: {maximum_mean_discrepancy_herding}")

    print("Plotting")
    # Plot the pre-coreset image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(pooled_image_data, cmap="gray")
    plt.title("Pre-Coreset")
    plt.axis("off")

    # Plot the coreset image and weight the points using a function of the coreset
    # weights
    plt.subplot(1, 3, 2)
    plt.scatter(
        herding_object.coreset[:, 1],
        -herding_object.coreset[:, 0],
        c=herding_object.coreset[:, 2],
        cmap="gray",
        s=np.exp(2.0 * coreset_size * herding_weights).reshape(1, -1),
        marker="h",
        alpha=0.8,
    )
    plt.axis("scaled")
    plt.title("Coreset")
    plt.axis("off")

    # Plot the image of randomly sampled points
    plt.subplot(1, 3, 3)
    plt.scatter(
        random_sample_object.coreset[:, 1],
        -random_sample_object.coreset[:, 0],
        c=random_sample_object.coreset[:, 2],
        s=1.0,
        cmap="gray",
        marker="h",
        alpha=0.8,
    )
    plt.axis("scaled")
    plt.title("Random")
    plt.axis("off")

    if out_path is not None:
        plt.savefig(out_path)

    plt.show()

    return (
        float(maximum_mean_discrepancy_herding),
        float(maximum_mean_discrepancy_random),
    )


# pylint: enable=no-member
# pylint: enable=too-many-locals
# pylint: enable=too-many-statements
# pylint: enable=duplicate-code


if __name__ == "__main__":
    main()
