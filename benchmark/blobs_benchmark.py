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
Performance of different coreset algorithms on the synthetic dataset.

We generate synthetic data using the scipy's make_blob function
and generate coresets using different coreset algorithms
"""

import json
import time
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from coreax import Data, SlicedScoreMatching
from coreax.kernels import (
    SquaredExponentialKernel,
    SteinKernel,
    median_heuristic,
)
from coreax.metrics import KSD, MMD
from coreax.solvers import (
    KernelHerding,
    RandomSample,
    RPCholesky,
    SteinThinning,
)
from coreax.weights import MMDWeightsOptimiser


def setup_kernel(x: np.ndarray) -> SquaredExponentialKernel:
    """Set up kernel using median heuristic."""
    num_samples_length_scale = min(300, 1000)
    random_seed = 45
    generator = np.random.default_rng(random_seed)
    idx = generator.choice(300, num_samples_length_scale, replace=False)
    length_scale = median_heuristic(x[idx])
    return SquaredExponentialKernel(length_scale=length_scale)


def setup_stein_kernel(
    sq_exp_kernel: SquaredExponentialKernel, dataset: Data
) -> SteinKernel:
    """Set up SteinKernel."""
    sliced_score_matcher = SlicedScoreMatching(
        jax.random.PRNGKey(45),
        jax.random.rademacher,
        use_analytic=True,
        num_random_vectors=100,
        learning_rate=0.001,
        num_epochs=50,
    )
    return SteinKernel(
        sq_exp_kernel,
        sliced_score_matcher.match(jnp.asarray(dataset.data)),
    )


def setup_solvers(
    coreset_size: int,
    sq_exp_kernel: SquaredExponentialKernel,
    stein_kernel: SteinKernel,
) -> List[Tuple[str, Any]]:
    """Define solvers."""
    random_key = jax.random.PRNGKey(42)
    return [
        (
            "KernelHerding",
            KernelHerding(coreset_size=coreset_size, kernel=sq_exp_kernel),
        ),
        (
            "RandomSample",
            RandomSample(coreset_size=coreset_size, random_key=random_key),
        ),
        (
            "RPCholesky",
            RPCholesky(
                coreset_size=coreset_size,
                kernel=sq_exp_kernel,
                random_key=random_key,
            ),
        ),
        (
            "SteinThinning",
            SteinThinning(
                coreset_size=coreset_size,
                kernel=stein_kernel,
                regularise=False,
            ),
        ),
    ]


def compute_metrics(
    solvers: List[Tuple[str, Any]],
    dataset: Data,
    mmd_metric: MMD,
    ksd_metric: KSD,
    weights_optimiser: MMDWeightsOptimiser,
) -> Dict[str, Dict[str, float]]:
    """Compute coresubsets and metrics."""
    results = {}
    for name, solver in solvers:
        start_time = time.time()
        coresubset, _ = solver.reduce(dataset)

        # Unweighted metrics
        unweighted_mmd = float(mmd_metric.compute(dataset, coresubset.coreset))
        unweighted_ksd = float(ksd_metric.compute(dataset, coresubset.coreset))

        # Weighted metrics
        weighted_coresubset = coresubset.solve_weights(weights_optimiser)
        weighted_mmd = float(weighted_coresubset.compute_metric(mmd_metric))
        weighted_ksd = float(weighted_coresubset.compute_metric(ksd_metric))

        end_time = time.time()
        elapsed_time = end_time - start_time

        results[name] = {
            "unweighted_mmd": unweighted_mmd,
            "unweighted_ksd": unweighted_ksd,
            "weighted_mmd": weighted_mmd,
            "weighted_ksd": weighted_ksd,
            "time": elapsed_time,
        }

    return results


def visualize_results(
    results: Dict[str, Dict[str, float]],
    dataset: Data,
    coreset_size: int,
) -> None:
    """Visualize results for each solver."""
    plt.figure(figsize=(20, 15))
    for i, (name, metrics) in enumerate(results.items()):
        plt.subplot(2, 2, i + 1)
        plt.scatter(
            dataset.data[:, 0],
            dataset.data[:, 1],
            alpha=0.3,
            label="Original Data",
        )
        plt.title(
            f"{name} (size: {coreset_size})\n"
            f"Unweighted MMD: {metrics['unweighted_mmd']:.6f}, "
            f"KSD: {metrics['unweighted_ksd']:.6f}\n"
            f"Weighted MMD: {metrics['weighted_mmd']:.6f}, "
            f"KSD: {metrics['weighted_ksd']:.6f}"
        )
        plt.legend()
        plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"coreset_comparison_{coreset_size}.png")
    plt.close()


def main() -> None:
    """Perform the benchmark."""
    # Generate data
    x, *_ = make_blobs(n_samples=1000, n_features=2, centers=10, random_state=45)
    dataset = Data(jnp.array(x))

    # Set up kernel
    sq_exp_kernel = setup_kernel(x)

    # Set up SteinKernel
    stein_kernel = setup_stein_kernel(sq_exp_kernel, dataset)

    # Set up metrics
    mmd_metric = MMD(kernel=sq_exp_kernel)
    ksd_metric = KSD(kernel=sq_exp_kernel)

    # Set up weights optimizer
    weights_optimiser = MMDWeightsOptimiser(kernel=sq_exp_kernel)

    # Define coreset sizes
    coreset_sizes = [10, 50, 100, 200]

    all_results = {}

    for size in coreset_sizes:
        # Define solvers
        solvers = setup_solvers(size, sq_exp_kernel, stein_kernel)

        # Compute metrics
        results = compute_metrics(
            solvers, dataset, mmd_metric, ksd_metric, weights_optimiser
        )
        all_results[size] = results

        visualize_results(results, dataset, size)

    # Save results to JSON file
    with open("coreset_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
