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
4. Optimise weights for the coresets to minimise the MMD score and recompute both
   the MMD and KSD metrics.
5. Measure and report the time taken for each step of the benchmarking process.
"""

import json
import os
import time
from typing import TypeVar

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
    KernelThinning,
    RandomSample,
    RPCholesky,
    Solver,
    SteinThinning,
)
from coreax.weights import MMDWeightsOptimiser

_Solver = TypeVar("_Solver", bound=Solver)


def setup_kernel(x: jnp.array, random_seed: int = 45) -> SquaredExponentialKernel:
    """
    Set up a squared exponential kernel using the median heuristic.

    :param x: Input data array used to compute the kernel length scale.
    :param random_seed: An integer seed for the random number generator.
    :return: A SquaredExponentialKernel with the computed length scale.
    """
    num_data_points = len(x)
    num_samples_length_scale = min(num_data_points, 1000)
    generator = np.random.default_rng(random_seed)
    idx = generator.choice(num_data_points, num_samples_length_scale, replace=False)
    length_scale = median_heuristic(x[idx])
    return SquaredExponentialKernel(length_scale=length_scale)


def setup_stein_kernel(
    sq_exp_kernel: SquaredExponentialKernel, dataset: Data, random_seed: int = 45
) -> SteinKernel:
    """
    Set up a Stein Kernel for Stein Thinning.

    :param sq_exp_kernel: A SquaredExponential base kernel for the Stein Kernel.
    :param dataset: Dataset for score matching.
    :param random_seed: An integer seed for the random number generator.
    :return: A SteinKernel object.
    """
    sliced_score_matcher = SlicedScoreMatching(
        jax.random.PRNGKey(random_seed),
        jax.random.rademacher,
        use_analytic=True,
        num_random_vectors=100,
        learning_rate=0.001,
        num_epochs=50,
    )
    return SteinKernel(
        base_kernel=sq_exp_kernel,
        score_function=sliced_score_matcher.match(jnp.asarray(dataset.data)),
    )


def setup_solvers(
    coreset_size: int,
    sq_exp_kernel: SquaredExponentialKernel,
    stein_kernel: SteinKernel,
    delta: float,
    random_seed: int = 45,
) -> list[tuple[str, _Solver]]:
    """
    Set up and return a list of solver configurations for reducing a dataset.

    :param coreset_size: The size of the coresets to be generated by the solvers.
    :param sq_exp_kernel: A Squared Exponential kernel for KernelHerding and RPCholesky.
        The square root kernel for KernelThinning is also derived from this kernel.
    :param stein_kernel: A Stein kernel object used for the SteinThinning solver.
    :param delta: The delta parameter for KernelThinning solver.
    :param random_seed: An integer seed for the random number generator.

    :return: A list of tuples, where each tuple contains the name of the solver
             and the corresponding solver object.
    """
    random_key = jax.random.PRNGKey(random_seed)
    sqrt_kernel = sq_exp_kernel.get_sqrt_kernel(dim=2)
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
        (
            "KernelThinning",
            KernelThinning(
                coreset_size=coreset_size,
                kernel=sq_exp_kernel,
                random_key=random_key,
                delta=delta,
                sqrt_kernel=sqrt_kernel,
            ),
        ),
    ]


def compute_solver_metrics(
    solver: _Solver,
    dataset: Data,
    mmd_metric: MMD,
    ksd_metric: KSD,
    weights_optimiser: MMDWeightsOptimiser,
) -> dict[str, float]:
    """
    Compute weighted and unweighted MMD and KSD metrics for a given solver.

    :param solver: Solver object used to reduce the dataset.
    :param dataset: The dataset.
    :param mmd_metric: MMD metric object to compute MMD.
    :param ksd_metric: KSD metric object to compute KSD.
    :param weights_optimiser: Optimiser to compute weights for the coresubset.

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
        "Unweighted_MMD": unweighted_mmd,
        "Unweighted_KSD": unweighted_ksd,
        "Weighted_MMD": weighted_mmd,
        "Weighted_KSD": weighted_ksd,
        "Time": elapsed_time,
    }


def compute_metrics(
    solvers: list[tuple[str, _Solver]],
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
    :param weights_optimiser: The optimiser object for weights for the coresubset.

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


def main() -> None:  # pylint: disable=too-many-locals
    """Benchmark various algorithms on a synthetic dataset over multiple seeds."""
    n_samples = 1_000
    seeds = [42, 45, 46, 47, 48]  # List of seeds to average over
    coreset_sizes = [25, 50, 100, 200]

    # Initialize storage for aggregated results
    aggregated_results = {size: {} for size in coreset_sizes}

    for seed in seeds:
        # Generate data for this seed
        x, *_ = make_blobs(
            n_samples=n_samples, n_features=2, centers=10, random_state=seed
        )
        dataset = Data(jnp.array(x))

        # Set up kernel
        sq_exp_kernel = setup_kernel(jnp.array(x))

        # Set up Stein Kernel
        stein_kernel = setup_stein_kernel(sq_exp_kernel, dataset)

        # Set up metrics
        mmd_metric = MMD(kernel=sq_exp_kernel)
        ksd_metric = KSD(kernel=sq_exp_kernel)

        # Set up weights optimiser
        weights_optimiser = MMDWeightsOptimiser(kernel=sq_exp_kernel)

        for size in coreset_sizes:
            solvers = setup_solvers(size, sq_exp_kernel, stein_kernel, seed)

            # Compute metrics for this size and seed
            results = compute_metrics(
                solvers, dataset, mmd_metric, ksd_metric, weights_optimiser
            )

            # Aggregate results across seeds
            for solver_name, metrics in results.items():
                if solver_name not in aggregated_results[size]:
                    aggregated_results[size][solver_name] = {
                        metric: [] for metric in metrics
                    }

                for metric, value in metrics.items():
                    aggregated_results[size][solver_name][metric].append(value)

    # Average results across seeds
    final_results = {"n_samples": n_samples}
    for size, solvers in aggregated_results.items():
        final_results[size] = {}
        for solver_name, metrics in solvers.items():
            final_results[size][solver_name] = {
                metric: sum(values) / len(values) for metric, values in metrics.items()
            }

    # Save final results to JSON file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(base_dir, "blobs_benchmark_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(final_results, f, indent=2)


if __name__ == "__main__":
    main()
