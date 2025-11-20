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

from pathlib import Path

import cv2
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

from coreax import (
    MMD,
    Data,
    KernelDensityMatching,
    SquaredExponentialKernel,
    SteinKernel,
)
from coreax.kernels import PCIMQKernel, median_heuristic
from coreax.solvers import KernelHerding, MapReduce, RandomSample
from coreax.weights import MMDWeightsOptimiser

MAX_8BIT = 255
MIN_LENGTH_SCALE = 1e-6


# Examples are written to be easy to read, copy and paste by users, so we ignore the
# pylint warnings raised that go against this approach
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=duplicate-code


def downsample_opencv(image_path: str, downsampling_factor: int) -> np.ndarray:
    """
    Downsample an image using func:`~cv2.resize` and convert it to grayscale.

    :param image_path: Path to the input image file.
    :param downsampling_factor: Factor by which to downsample the image.
    :return: Grayscale image after downsampling.
    """
    img = cv2.imread(image_path)

    # Calculate new dimensions based on downsampling factor
    scale_factor = 1 / downsampling_factor
    if img is None:
        raise RuntimeError("'img' is unexpectedly 'None'.")
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)

    # Resize using INTER_AREA for better downsampling
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Convert to grayscale after resizing
    grayscale_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    return grayscale_resized


def main(
    in_path: Path = Path("../examples/data/david_orig.png"),
    out_path: Path | None = None,
    downsampling_factor: int = 1,
) -> tuple[float, float]:
    """
    Run the 'David' example for image sampling.

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
    original_data = downsample_opencv(str(in_path), downsampling_factor)

    block_size = 1_000 // (downsampling_factor**2)

    print(f"Image dimensions: {original_data.shape}")
    pre_coreset_data = np.column_stack(np.nonzero(original_data < MAX_8BIT))
    pixel_values = original_data[original_data < MAX_8BIT]
    pre_coreset_data = jnp.column_stack((pre_coreset_data, pixel_values)).astype(
        jnp.float32
    )
    num_data_points = pre_coreset_data.shape[0]

    # Request coreset points
    coreset_size = 8_000 // (downsampling_factor**2)

    # Setup the original data object
    data = Data(pre_coreset_data)

    # Set the length_scale parameter of the kernel from at most 1000 samples
    num_samples_length_scale = min(num_data_points, 1000 // (downsampling_factor**2))
    random_seed = 1_989
    generator = np.random.default_rng(random_seed)
    idx = generator.choice(num_data_points, num_samples_length_scale, replace=False)
    length_scale = float(median_heuristic(pre_coreset_data[idx]))
    if length_scale < MIN_LENGTH_SCALE:
        length_scale = 100.0

    # Learn a score function via kernel density estimation (this is required for
    # evaluation of the Stein kernel)
    kernel_density_score_matcher = KernelDensityMatching(length_scale=length_scale)
    score_function = kernel_density_score_matcher.match(pre_coreset_data[idx, :])

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
    _, sample_key = random.split(random.key(random_seed))
    herding_solver = KernelHerding(
        coreset_size,
        kernel=herding_kernel,
        block_size=block_size,
    )
    mapped_herding_solver = MapReduce(
        herding_solver, leaf_size=16_000 // (downsampling_factor**2)
    )
    herding_coreset, _ = eqx.filter_jit(mapped_herding_solver.reduce)(data)
    herding_weights = weights_optimiser.solve(data, herding_coreset.points)

    print("Choosing random subset...")
    # Generate a coreset via uniform random sampling for comparison
    random_solver = RandomSample(coreset_size, sample_key, unique=True)
    random_coreset, _ = eqx.filter_jit(random_solver.reduce)(data)

    # Define a reference kernel to use for comparisons of MMD. We'll use a normalised
    # SquaredExponentialKernel (which is also a Gaussian kernel)
    print("Computing MMD...")
    mmd_kernel = SquaredExponentialKernel(
        length_scale=length_scale,
        output_scale=1.0 / (length_scale * float(jnp.sqrt(2.0 * jnp.pi))),
    )

    # Compute the MMD between the original data and the coreset generated via herding
    mmd_metric = MMD(kernel=mmd_kernel)
    herding_mmd = herding_coreset.compute_metric(mmd_metric, block_size=block_size)

    # Compute the MMD between the original data and the coreset generated via random
    # sampling
    random_mmd = random_coreset.compute_metric(mmd_metric, block_size=block_size)

    # Print the MMD values
    print(f"Random sampling coreset MMD: {random_mmd}")
    print(f"Herding coreset MMD: {herding_mmd}")

    def transform_marker_size(
        weights: jax.Array,
        scale_factor: int = 15,
        min_size: int = 4 * downsampling_factor,
    ) -> np.ndarray:
        """
        Transform coreset weights to marker sizes for plotting.

        :param weights: Array of coreset weights to be transformed.
        :param scale_factor: Ratio of the largest and the smallest marker sizes.
        :param min_size: Smallest marker size.
        :return: Array of transformed marker sizes for plotting.
        """
        # Define threshold percentiles
        lower_percentile, upper_percentile = 1, 99

        # Clip weights to reduce the effect of outliers
        clipped_weights = np.clip(
            weights,
            np.percentile(weights, lower_percentile),
            np.percentile(weights, upper_percentile),
        )

        # Normalize weights to a [0, 1] range
        normalized_weights = (clipped_weights - clipped_weights.min()) / (
            clipped_weights.max() - clipped_weights.min()
        )

        # Apply exponential scaling to get the desired spread
        transformed_sizes = min_size + (scale_factor**normalized_weights - 1) * min_size

        return transformed_sizes

    print("Plotting")
    # Plot the pre-coreset image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_data, cmap="gray")
    plt.title("Pre-Coreset")
    plt.axis("off")

    # Plot the coreset image and weight the points using a function of the coreset
    # weights
    plt.subplot(1, 3, 2)
    plt.scatter(
        herding_coreset.points.data[:, 1],
        -herding_coreset.points.data[:, 0],
        c=herding_coreset.points.data[:, 2],
        cmap="gray",
        s=(transform_marker_size(herding_weights)).reshape(1, -1),
        marker="h",
        alpha=0.8,
    )
    plt.axis("scaled")
    plt.title("Coreset")
    plt.axis("off")

    # Plot the image of randomly sampled points
    plt.subplot(1, 3, 3)
    plt.scatter(
        random_coreset.points.data[:, 1],
        -random_coreset.points.data[:, 0],
        c=random_coreset.points.data[:, 2],
        s=25 * downsampling_factor,
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
        float(herding_mmd),
        float(random_mmd),
    )


# pylint: enable=too-many-locals
# pylint: enable=too-many-statements
# pylint: enable=duplicate-code


if __name__ == "__main__":
    main(out_path=Path("data/david_coreset_2.png"))
