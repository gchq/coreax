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

"""TODO: Write module docstring."""

# Support annotations with | in Python < 3.10
# TODO: Remove once no longer supporting old code
from __future__ import annotations

from pathlib import Path

import cv2
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from coreax.kernel import median_heuristic, rbf_kernel
from coreax.kernel_herding import (
    scalable_herding,
    scalable_rbf_grad_log_f_X,
    scalable_stein_kernel_pc_imq_element,
    stein_kernel_herding_block,
)
from coreax.metrics import mmd_block
from coreax.util import solve_qp


def main(
    in_path: Path = Path("../examples/data/david_orig.png"),
    out_path: Path | None = None,
) -> tuple[float, float]:
    """
    Run the 'david' example for image sampling.

    Take an image of the statue of David and then generate a coreset using
    scalable Stein kernel herding. Compare the result from this to a coreset generated
    via uniform random sampling. Coreset quality is measured using maximum mean
    discrepancy (MMD).

    :param in_path: Path to input image, assumed relative to this module file unless an
        absolute path is given
    :param out_path: Path to save output to, if not :data:`None`, assumed relative to
        this module file unless an absolute path is given
    :return: Coreset MMD, random sample MMD
    """
    # Convert to absolute paths
    if not in_path.is_absolute():
        in_path = Path(__file__).parent / in_path
    if out_path is not None and not out_path.is_absolute():
        out_path = Path(__file__).parent / out_path

    # path to original image
    orig = cv2.imread(str(in_path))
    img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    print(f"Image dimensions: {img.shape}")
    X_ = np.column_stack(np.where(img < 255))
    vals = img[img < 255]
    X = np.column_stack((X_, vals)).astype(np.float32)
    n = X.shape[0]

    # request 8000 coreset points
    C = 8000

    # set the bandwidth parameter of the kernel from at most 1000 samples
    N = min(n, 1000)
    idx = np.random.choice(n, N, replace=False)
    nu = median_heuristic(X[idx].astype(float))
    if nu == 0.0:
        nu = 100.0

    indices = np.arange(n)

    print("Computing coreset...")
    # use scalable Stein kernel herding. Here size=10000 partitions the input into size
    # 10000 blocks for independent coreset solving. grad_log_f_X is the score function.
    # We use an explicit function derived from a KDE, but this can be any score function
    # approximation, e.g. score matching. max size is for block processing Gram matrices
    # to avoid memory issues
    coreset, weights = scalable_herding(
        X,
        indices,
        C,
        stein_kernel_herding_block,
        solve_qp,
        size=10000,
        kernel=scalable_stein_kernel_pc_imq_element,
        grad_log_f_X=scalable_rbf_grad_log_f_X,
        nu=nu,
        max_size=1000,
    )

    print("Choosing random subset...")
    # choose a random subset of C points from the original image
    rand_points = np.random.choice(n, C, replace=False)

    # define a reference kernel to use for comparisons of MMD. We'll use an RBF
    def k(x, y):
        return rbf_kernel(x, y, jnp.float32(nu) ** 2) / (nu * jnp.sqrt(2.0 * jnp.pi))

    # compute the MMD between X and the coreset
    m = mmd_block(X, X[coreset], k, max_size=1000)

    # compute the MMD between X and the random sample
    rm = mmd_block(X, X[rand_points], k, max_size=1000).item()

    # print the MMDs
    print("Random MMD")
    print(rm)
    print("Coreset MMD")
    print(m)

    print("Plotting")
    # plot the original image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # plot the coreset image and weight the points using a function of the coreset
    # weights
    plt.subplot(1, 3, 2)
    plt.scatter(
        X[coreset, 1],
        -X[coreset, 0],
        c=X[coreset, 2],
        cmap="gray",
        s=np.exp(2.0 * C * weights).reshape(1, -1),
        marker="h",
        alpha=0.8,
    )
    plt.axis("scaled")
    plt.title("Coreset")
    plt.axis("off")

    # plot the image of randomly sampled points
    plt.subplot(1, 3, 3)
    plt.scatter(
        X[rand_points, 1],
        -X[rand_points, 0],
        c=X[rand_points, 2],
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

    return float(m), float(rm)


if __name__ == "__main__":
    main()
