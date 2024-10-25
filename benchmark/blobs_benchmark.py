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
Benchmark performance of different coreset algorithms on a synthetic dataset.

The benchmarking process follows these steps:
1. Generate a synthetic dataset of 1000 two-dimensional points using
   :func:`sklearn.datasets.make_blobs`.
2. Generate coresets of varying sizes: 10, 50, 100, and 200 points using different
   coreset algorithms.
3. Compute two metrics to evaluate the coresets' quality:
   - Maximum Mean Discrepancy (MMD)
   - Kernel Stein Discrepancy (KSD)
4. Optimize weights for the coresets to minimize the MMD score and recompute both
   the MMD and KSD metrics.
5. Measure and report the time taken for each step of the benchmarking process.
"""

import json
import os
import time
from typing import Any

import jax
import jax.numpy as jnp
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
    """
    Set up a squared exponential kernel using the median heuristic.

    :param x: Input data array used to compute the kernel length scale.
    :return: A SquaredExponentialKernel with the computed length scale.
    """
    num_samples_length_scale = min(300, 1000)
    random_seed = 45
    generator = np.random.default_rng(random_seed)
    idx = generator.choice(300, num_samples_length_scale, replace=False)
    length_scale = median_heuristic(x[idx])
    return SquaredExponentialKernel(length_scale=length_scale)


def setup_stein_kernel(
    sq_exp_kernel: SquaredExponentialKernel, dataset: Data
) -> SteinKernel:
    """
    Set up a Stein Kernel for Stein Thinning.

    :param sq_exp_kernel: A SquaredExponential base kernel for the Stein Kernel.
    :param dataset: Dataset for score matching.
    :return: A SteinKernel object.
    """
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
) -> list[tuple[str, Any]]:
    """
    Set up and return a list of solver configurations for reducing a dataset.

    :param coreset_size: The size of the coresets to be generated by the solvers.
    :param sq_exp_kernel: A Squared Exponential kernel for KernelHerding and RPCholesky.
    :param stein_kernel: A Stein kernel object used for the SteinThinning solver.

    :return: A list of tuples, where each tuple contains the name of the solver
             and the corresponding solver object.
    """
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


def compute_solver_metrics(
    solver: Any,
    dataset: Data,
    mmd_metric: MMD,
    ksd_metric: KSD,
    weights_optimiser: MMDWeightsOptimiser,
) -> dict[str, float]:
    """
    Compute weighted and unweighted MMD and KSD metrics for a given solver.

    :param name: Name of the solver being evaluated.
    :param solver: Solver object used to reduce the dataset.
    :param dataset: The dataset.
    :param mmd_metric: MMD metric object to compute MMD.
    :param ksd_metric: KSD metric object to compute KSD.
    :param weights_optimiser: Optimizer to compute weights for the coresubset.

    :return: A dictionary with unweighted and weighted metrics (MMD, KSD) and
             the time taken for the computation.
    """
    start_time = time.perf_counter()  # Using perf_counter for higher precision timing
    coresubset, _ = solver.reduce(dataset)

    # Unweighted metrics
    unweighted_mmd = float(mmd_metric.compute(dataset, coresubset.coreset))
    unweighted_ksd = float(ksd_metric.compute(dataset, coresubset.coreset))

    # Weighted metrics
    weighted_coresubset = coresubset.solve_weights(weights_optimiser)
    weighted_mmd = float(weighted_coresubset.compute_metric(mmd_metric))
    weighted_ksd = float(weighted_coresubset.compute_metric(ksd_metric))

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return {
        "unweighted_mmd": unweighted_mmd,
        "unweighted_ksd": unweighted_ksd,
        "weighted_mmd": weighted_mmd,
        "weighted_ksd": weighted_ksd,
        "time": elapsed_time,
    }


def compute_metrics(
    solvers: list[tuple[str, Any]],
    dataset: Data,
    mmd_metric: MMD,
    ksd_metric: KSD,
    weights_optimiser: MMDWeightsOptimiser,
) -> dict[str, dict[str, float]]:
    """
    Compute the coresubsets and corresponding metrics for each solver in a given list.

    :param solvers: A list of tuples containing solver names and their
                    respective solver objects.
    :param dataset: The dataset.
    :param mmd_metric: The MMD metric object for computing Maximum Mean Discrepancy.
    :param ksd_metric: The KSD metric object for computing Kernel Stein Discrepancy.
    :param weights_optimiser: The optimizer object for weights for the coresubset.

    :return: A dictionary where the keys are the solver names, and the values are
             dictionaries of computed metrics (unweighted/weighted MMD and KSD, and
             computation time).
    """
    return {
        name: compute_solver_metrics(
            solver, dataset, mmd_metric, ksd_metric, weights_optimiser
        )
        for name, solver in solvers
    }


def main() -> None:
    """
    Benchmark different algorithms against on a synthetic dataset.

    Compare the performance of different coreset algorithms using a synthetic dataset,
    generated using :func:`sklearn.datasets.make_blobs`. We set up various solvers,
    generate coresets of multiple sizes, and compute performance metrics (MMD and KSD)
    for each solver at each coreset size. Results are saved to a JSON file.
    """
    # Generate data
    x, *_ = make_blobs(n_samples=1000, n_features=2, centers=10, random_state=45)
    dataset = Data(jnp.array(x))

    # Set up kernel
    sq_exp_kernel = setup_kernel(x)

    # Set up Stein Kernel
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
        solvers = setup_solvers(size, sq_exp_kernel, stein_kernel)

        # Compute metrics
        results = compute_metrics(
            solvers, dataset, mmd_metric, ksd_metric, weights_optimiser
        )
        all_results[size] = results

    # Save results to JSON file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(base_dir, "blobs_benchmark_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
