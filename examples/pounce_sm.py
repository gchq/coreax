# © Crown Copyright GCHQ
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""TODO: Write module docstring."""

from pathlib import Path

import imageio
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.random import rademacher
from sklearn.decomposition import PCA

from coreax.kernel import median_heuristic, rbf_kernel, stein_kernel_pc_imq_element
from coreax.kernel_herding import stein_kernel_herding_block
from coreax.metrics import mmd_block
from coreax.score_matching import sliced_score_matching


def main(directory: Path = Path("../examples/data/pounce")) -> tuple[float, float]:
    """
    Run the 'pounce' example for video sampling with score matching.

    Take a video of a pouncing cat, apply PCA and then generate a coreset using
    score matching, in which we train a neural network to approximate the score functon
    of the underlying distribution. Compare the result from this to a coreset generated
    via uniform random sampling. Coreset quality is measured using maximum mean
    discrepancy (MMD).

    :param directory: Path to directory containing input video, assumed relative to this
        module file unless an absolute path is given
    :return: Coreset MMD, random sample MMD
    """
    # Convert directory to absolute path
    if not directory.is_absolute():
        directory = Path(__file__).parent / directory

    # path to directory containing video as sequence of images
    fn = Path("pounce.gif")
    coreset_dir = directory / Path("coreset_sm")
    coreset_dir.mkdir(exist_ok=True)

    # read in as video. Frame 0 is missing A from RGBA.
    Y_ = np.array(imageio.v2.mimread(directory / fn)[1:])
    Y = Y_.reshape(Y_.shape[0], -1)

    # run PCA to reduce the dimension of the images whilst minimising effects on some of
    # the statistical properties, i.e. variance.
    p = 25
    pca = PCA(p)
    X = pca.fit_transform(Y)

    # request a 10 frame summary of the video
    C = 10

    # set the bandwidth parameter of the underlying RBF kernel
    N = min(X.shape[0], 1000)
    idx = np.random.choice(X.shape[0], N, replace=False)
    nu = median_heuristic(X[idx])

    # learn a score function
    score_function = sliced_score_matching(
        X, rademacher, use_analytic=True, epochs=100, L=100, sigma=1.0
    )

    # run Stein kernel herding in block mode to avoid GPU memory issues
    coreset, Kc, Kbar = stein_kernel_herding_block(
        X, C, stein_kernel_pc_imq_element, score_function, nu=nu, max_size=1000, sm=True
    )

    # sort the coreset ready for producing the output video
    coreset = jnp.sort(coreset)
    print("Coreset:", coreset)

    # define a reference kernel to use for comparisons of MMD. We'll use an RBF
    def k(x, y):
        return rbf_kernel(x, y, jnp.float32(nu) ** 2) / (nu * jnp.sqrt(2.0 * jnp.pi))

    # compute the MMD between X and the coreset
    m = mmd_block(X, X[coreset], k, max_size=1000)

    # get a random sample of points to compare against
    rsample = np.random.choice(N, size=C, replace=False)
    # compute the MMD between X and the random sample
    rm = mmd_block(X, X[rsample], k, max_size=1000).item()

    # print the MMDs
    print(f"Random MMD: {rm}")
    print(f"Coreset MMD: {m}")

    # Save a new video. Y_ is the original sequence with dimensions preserved
    coreset_images = Y_[coreset]
    imageio.mimsave(coreset_dir / Path("coreset_sm.gif"), coreset_images)

    # plot to visualise which frames were chosen from the sequence action frames are
    # where the "pounce" occurs
    action_frames = np.arange(63, 85)
    x = np.arange(N)
    y = np.zeros(N)
    y[coreset] = 1.0
    z = np.zeros(N)
    z[jnp.intersect1d(coreset, action_frames)] = 1.0
    plt.figure(figsize=(20, 3))
    plt.bar(x, y, alpha=0.5)
    plt.bar(x, z)
    plt.xlabel("Frame")
    plt.ylabel("Chosen")
    plt.tight_layout()
    plt.savefig(directory / "coreset_sm" / "frames_sm.png")
    plt.close()

    return m, rm


if __name__ == "__main__":
    main()
