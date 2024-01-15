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
Example coreset generation using a video of a pouncing cat.

This example showcases how a coreset can be generated from video data. In this context,
a coreset is a set of frames that best capture the information in the original video.

Firstly, principal component analysis (PCA) is applied to the video data to reduce
dimensionality. Then, a coreset is generated using Stein kernel herding, with a PCIMQ
base kernel. The score function (gradient of the log-density function) for the Stein
kernel is estimated by applying kernel density estimation (KDE) to the data, and then
taking gradients.

The coreset attained from Stein kernel herding is compared to a coreset generated via
uniform random sampling. Coreset quality is measured using maximum mean discrepancy
(MMD).
"""

from pathlib import Path

import imageio
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import coreax.refine
from coreax.coresubset import KernelHerding, RandomSample
from coreax.data import ArrayData
from coreax.kernel import SquaredExponentialKernel, SteinKernel, median_heuristic
from coreax.metrics import MMD
from coreax.reduction import MapReduce
from coreax.score_matching import KernelDensityMatching


def main(directory: Path = Path("../examples/data/pounce")) -> tuple[float, float]:
    """
    Run the 'pounce' example for video sampling with Stein kernel herding.

    Take a video of a pouncing cat, apply PCA and then generate a coreset using
    Stein kernel herding. Compare the result from this to a coreset generated
    via uniform random sampling. Coreset quality is measured using maximum mean
    discrepancy (MMD).

    :param directory: Path to directory containing input video, assumed relative to this
        module file unless an absolute path is given
    :return: Coreset MMD, random sample MMD
    """
    # Convert directory to absolute path
    if not directory.is_absolute():
        directory = Path(__file__).parent / directory

    # Define path to directory containing video as sequence of images
    file_name = Path("pounce.gif")
    coreset_dir = directory / Path("coreset_map_reduce")
    coreset_dir.mkdir(exist_ok=True)

    # Read in the data as a video. Frame 0 is missing A from RGBA.
    raw_data = np.array(imageio.v2.mimread(directory / file_name)[1:])
    raw_data_reshaped = raw_data.reshape(raw_data.shape[0], -1)

    # Run PCA to reduce the dimension of the images whilst minimising effects on some of
    # the statistical properties, i.e. variance.
    num_principle_components = 25
    pca = PCA(num_principle_components)
    principle_components_data = pca.fit_transform(raw_data_reshaped)

    # Setup the original data object
    data = coreax.data.ArrayData(
        original_data=principle_components_data,
        pre_coreset_array=principle_components_data,
    )

    # Request a 10 frame summary of the video
    coreset_size = 10

    # Set the length_scale parameter of the underlying RBF kernel
    num_points_length_scale_selection = min(principle_components_data.shape[0], 1000)
    idx = np.random.choice(
        principle_components_data.shape[0],
        num_points_length_scale_selection,
        replace=False,
    )
    length_scale = median_heuristic(principle_components_data[idx])

    # Learn a score function via kernel density estimation
    kernel_density_score_matcher = KernelDensityMatching(
        length_scale=length_scale, kde_data=principle_components_data[idx, :]
    )
    score_function = kernel_density_score_matcher.match()

    # Run kernel herding with a Stein kernel in block mode to avoid GPU memory issues
    herding_object = KernelHerding(
        kernel=SteinKernel(
            SquaredExponentialKernel(length_scale=length_scale),
            score_function=score_function,
        )
    )
    herding_object.fit(
        original_data=data,
        strategy=MapReduce(coreset_size=coreset_size, leaf_size=1000),
    )

    # Get and sort the coreset indices ready for producing the output video
    coreset_indices_herding = jnp.sort(herding_object.coreset_indices)

    # Define a reference kernel to use for comparisons of MMD. We'll use a normalised
    # SquaredExponentialKernel (which is also a Gaussian kernel)
    mmd_kernel = SquaredExponentialKernel(
        length_scale=length_scale,
        output_scale=1.0 / (length_scale * jnp.sqrt(2.0 * jnp.pi)),
    )

    # Compute the MMD between the original data and the coreset generated via herding
    metric_object = MMD(kernel=mmd_kernel)
    maximum_mean_discrepancy_herding = metric_object.compute(
        data.original_data, herding_object.coreset
    )

    # Generate a coreset via uniform random sampling for comparison
    random_sample_object = RandomSample(unique=True)
    random_sample_object.fit(
        original_data=data,
        strategy=MapReduce(coreset_size=coreset_size, leaf_size=1000),
    )
    # Compute the MMD between the original data and the coreset generated via random
    # sampling
    maximum_mean_discrepancy_random = metric_object.compute(
        data.original_data, random_sample_object.coreset
    )

    # Print the MMD values
    print(f"Random sampling coreset MMD: {maximum_mean_discrepancy_random}")
    print(f"Herding coreset MMD: {maximum_mean_discrepancy_herding}")

    # Save a new video. Y_ is the original sequence with dimensions preserved
    coreset_images = raw_data[coreset_indices_herding]
    imageio.mimsave(coreset_dir / Path("coreset_map_reduce.gif"), coreset_images)

    # Plot to visualise which frames were chosen from the sequence action frames are
    # where the "pounce" occurs
    action_frames = np.arange(63, 85)
    x = np.arange(num_points_length_scale_selection)
    y = np.zeros(num_points_length_scale_selection)
    y[coreset_indices_herding] = 1.0
    z = np.zeros(num_points_length_scale_selection)
    z[jnp.intersect1d(coreset_indices_herding, action_frames)] = 1.0
    plt.figure(figsize=(20, 3))
    plt.bar(x, y, alpha=0.5)
    plt.bar(x, z)
    plt.xlabel("Frame")
    plt.ylabel("Chosen")
    plt.tight_layout()
    plt.savefig(directory / "coreset_map_reduce" / "frames_map_reduce.png")
    plt.close()

    return (
        float(maximum_mean_discrepancy_herding),
        float(maximum_mean_discrepancy_random),
    )


if __name__ == "__main__":
    main()