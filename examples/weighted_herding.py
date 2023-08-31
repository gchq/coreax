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
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import make_blobs
from pathlib import Path

from coreax.weights import qp
from coreax.kernel import rbf_kernel, median_heuristic, stein_kernel_pc_imq_element, rbf_grad_log_f_x
from coreax.kernel_herding import stein_kernel_herding_block
from coreax.metrics import mmd_block, mmd_weight_block


def main(out_path: Path = None, weighted: bool = True):
    """
    Run the 'weighted_herding' example for weighted and unweighted herding.

    Args:
        out_path: path to save output to, if not None. Default None.
        weighted: boolean flag for whether to use weighted or unweighted herding

    Returns:
        coreset MMD, random sample MMD

    """

    # create some data. Here we'll use 10,000 points in 2D from 6 distinct clusters. 2D for plotting below.
    N = 10000
    X, _ = make_blobs(N, n_features=2, centers=6, random_state=32)

    # ask for 100 coreset points
    C = 100

    # set the bandwidth parameter of the kernel using a median heuristic derived from at most 1000 random samples
    # in the data.
    n = min(N, 1000)
    idx = np.random.choice(N, n, replace=False)
    nu = median_heuristic(X[idx])

    # define a kernel. We'll use an RBF
    def k(x, y): return rbf_kernel(x, y, jnp.float32(nu)**2) / \
        (nu * jnp.sqrt(2. * jnp.pi))

    # Find a C-sized coreset using -- in this case -- Stein kernel herding (block mode).
    # Stein kernel herding uses the Stein kernel derived from the RBF above.
    # Block mode processes the Gram matrix in blocks to avoid GPU memory issues.
    # rbf_grad_log_f_X is the score function derived from a KDE. This could be replaced by any score-function
    # approximation, e.g. score matching.
    # max_size sets the block processing size

    # returns the indices for the coreset points, the coreset Gram matrix (Kc) and the coreset Gram mean (Kbar)
    coreset, Kc, Kbar = stein_kernel_herding_block(
        X, C, stein_kernel_pc_imq_element, rbf_grad_log_f_X, nu=nu, max_size=1000)

    # get a random sample of points to compare against
    rsample = np.random.choice(N, size=C, replace=False)

    # the weighted bool turns the coreset weights on or off. If on, a quadratic program is invoked to solve
    # the weights' vector. This buys some increase in integration error, but at a computational cost. Likely
    # to most effective in lower dimensions.
    if weighted:
        # find the weights. Solves a QP
        weights = qp(Kc + 1e-10, Kbar)
        # check minimum weight is not negative by more than a reasonable tolerance
        if weights.min() < -1e-4:
            raise ValueError(f"Minimum weight was {weights.min()} but should have been >=0")
        # compute the MMD between X and the coreset, weighted version
        m = mmd_weight_block(X, X[coreset], jnp.ones(N), weights, k, max_size=1000)
    else:
        # equal weights
        weights = jnp.ones(C)
        # compute the MMD between X and the coreset, unweighted version
        m = mmd_block(X, X[coreset], k, max_size=1000)
    m = m.item()

    # compute the MMD between X and the random sample
    rm = mmd_block(X, X[rsample], k, max_size=1000).item()

    # nudge the weights to avoid negative entries for plotting
    if weights.min() < 0:
        weights -= weights.min()

    # produce some scatter plots
    plt.scatter(X[:, 0], X[:, 1], s=2., alpha=.1)
    plt.scatter(X[coreset, 0], X[coreset, 1], s=weights*1000, color="red")
    plt.axis('off')
    plt.title('Stein kernel herding, m=%d, MMD=%.6f' % (C, m))
    plt.show()

    plt.scatter(X[:, 0], X[:, 1], s=2., alpha=.1)
    plt.scatter(X[rsample, 0], X[rsample, 1], s=10, color="red")
    plt.title('Random, m=%d, MMD=%.6f' % (C, rm))
    plt.axis('off')

    if out_path is not None:
        plt.savefig(out_path)

    plt.show()

    # print the MMDs
    print(f"Random MMD: {rm}")
    print(f"Coreset MMD: {m}")

    return m, rm


if __name__ == '__main__':
    main()
