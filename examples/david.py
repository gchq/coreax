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
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import cv2

from coreax.weights import qp
from coreax.kernel import rbf_kernel, median_heuristic
from coreax.kernel_herding import stein_kernel_herding_block, scalable_herding, scalable_rbf_grad_log_f_X, \
    scalable_stein_kernel_pc_imq_element


def main(inpath="./examples/data/david_orig.png", outpath=None):

    # path to original image
    orig = cv2.imread(inpath)
    img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    print("Image dimensions:")
    print(img.shape)
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
    if nu == 0.:
        nu = 100.

    indices = np.arange(n)

    print("Computing coreset...")
    # use scalable Stein kernel herding. Here size=10000 partitions the input into size 10000 blocks for
    # independent coreset solving. grad_log_f_X is the score function. We use an explicit function derived from a
    # KDE, but this can be any score function approximation, e.g. score matching.
    # max size is for block processing Gram matrices to avoid memory issues
    coreset, weights = \
        scalable_herding(X, indices, C, stein_kernel_herding_block, qp, size=10000,
                         kernel=scalable_stein_kernel_pc_imq_element, grad_log_f_X=scalable_rbf_grad_log_f_X,
                         nu=nu, max_size=1000)

    print("Choosing random subset...")
    # choose a random subset of C points from the original image
    rpoints = np.random.choice(n, C, replace=False)

    print("Plotting")
    # plot the original image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title('Original')
    plt.axis('off')

    # plot the coreset image and weight the points using a function of the coreset weights
    plt.subplot(1, 3, 2)
    plt.scatter(X[coreset, 1], -X[coreset, 0], c=X[coreset, 2], cmap="gray",
                s=np.exp(2. * C * weights).reshape(1, -1), marker="h", alpha=.8)
    plt.axis('scaled')
    plt.title('Coreset')
    plt.axis('off')

    # plot the image of randomly sampled points
    plt.subplot(1, 3, 3)
    plt.scatter(X[rpoints, 1], -X[rpoints, 0], c=X[rpoints, 2], s=1., cmap="gray", marker="h", alpha=.8)
    plt.axis('scaled')
    plt.title('Random')
    plt.axis('off')

    if outpath is not None:
        plt.savefig(outpath)

    plt.show()


if __name__ == '__main__':
    main()
