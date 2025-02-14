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
Benchmark time taken to generate coresets from a large dataset.

The benchmarking process follows these steps:
1. Start with the MNIST dataset, which consists of 60_000 training images and 10_000
   test images.
2. To reduce dimensionality, apply density preserving UMAP to project the 28x28 60_000
   images into 16 components before applying coreset algorithms.
3. Generate coresets of different sizes using various coreset algorithms and record the
   time taken.

The benchmark is run on amazon g4dn.12xlarge instance with 4 nvidia t4 tensor core
GPUs, 48 virtual CPUs and 192 GiB memory.
"""

import json
import os
import time

import equinox as eqx
import jax
from mnist_benchmark import (
    density_preserving_umap,
    prepare_datasets,
)

from coreax import Data
from coreax.benchmark_util import initialise_solvers


def save_results(results: dict) -> None:
    """
    Save benchmark results to a JSON file for algorithm performance visualisation.

    :param results: A dictionary of results structured as follows:
                    {
                        "algorithm_name": {
                            "coreset_size_1": {
                                "run_1": time_taken,
                                "run_2": time_taken,
                                ...
                            },
                            "coreset_size_2": {
                                "run_1": time_taken,
                                "run_2": time_taken,
                                ...
                            },
                            ...
                        },
                        "another_algorithm_name": {
                            "coreset_size_1": {
                                "run_1": time_taken,
                                "run_2": time_taken,
                                ...
                            },
                            ...
                        },
                        ...
                    }
                    Each algorithm contains coreset sizes as keys, with values being
                    dictionaries of time taken from different runs.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = "mnist_time_results.json"
    with open(os.path.join(base_dir, file_name), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Data has been saved to {file_name}")


def main() -> None:
    """
    Perform the benchmark for multiple solvers, coreset sizes, and random seeds.

    The function follows these steps:
    1. Prepare and load the MNIST dataset (training set).
    2. Perform dimensionality reduction on the training data using UMAP.
    3. Initialise solvers for data reduction.
    4. For each solver and coreset size, reduce the dataset and store the time taken.
    """
    train_data_jax, _, _, _ = prepare_datasets()
    train_data_umap = Data(density_preserving_umap(train_data_jax))

    coreset_times = {}

    # Run the experiment with 5 different random keys
    # pylint: disable=duplicate-code
    for i in range(5):
        print(f"Run {i + 1} of 5:")
        key = jax.random.PRNGKey(i)
        solver_factories = initialise_solvers(
            train_data_umap, key, g=7, leaf_size=15_000
        )
        for solver_name, solver_creator in solver_factories.items():
            for size in [25, 50, 100, 500, 1_000]:
                solver = solver_creator(size)
                start_time = time.perf_counter()
                # pylint: enable=duplicate-code
                _, _ = eqx.filter_jit(solver.reduce)(train_data_umap)
                time_taken = time.perf_counter() - start_time

                # Ensure that there is a dictionary for this solver
                # If not, initialise with an empty dictionary
                if solver_name not in coreset_times:
                    coreset_times[solver_name] = {}

                # Populate the dictionary created above with coreset_size as keys
                # The values themselves will be dictionaries, so initialise with an
                # empty dictionary
                if size not in coreset_times[solver_name]:
                    coreset_times[solver_name][size] = {}

                # Store time taken result in nested structure
                coreset_times[solver_name][size][i] = time_taken

    # Save or print results
    save_results(coreset_times)


if __name__ == "__main__":
    main()
