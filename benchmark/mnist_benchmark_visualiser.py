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

"""Visualise the results of ``mnist_benchmark.py``."""

import json
import os

import numpy as np
from matplotlib import pyplot as plt


def main() -> None:
    """Load benchmark results and visualise the algorithm performance."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(base_dir, "mnist_benchmark_results.json"), "r", encoding="utf-8"
    ) as file:
        # Load the JSON data into a Python object
        data_by_solver = json.load(file)

    # Prepare data for bar plotting
    algorithms = []
    coreset_sizes = sorted(
        {size for sizes in data_by_solver.values() for size in sizes.keys()}
    )
    means = {}
    mins = {}
    maxs = {}

    for algo, sizes in data_by_solver.items():
        algorithms.append(algo)
        means[algo] = []
        mins[algo] = []
        maxs[algo] = []

        for size in coreset_sizes:
            if size in sizes:
                accuracies = list(sizes[size].values())
                means[algo].append(np.mean(accuracies))
                mins[algo].append(np.min(accuracies))
                maxs[algo].append(np.max(accuracies))
            else:
                means[algo].append(np.nan)
                mins[algo].append(np.nan)
                maxs[algo].append(np.nan)

    # Create the bar plot with error bars
    plt.figure(figsize=(12, 6))
    bar_width = 0.2
    index = np.arange(len(coreset_sizes))

    for i, algo in enumerate(algorithms):
        plt.bar(
            index + i * bar_width,
            means[algo],
            bar_width,
            label=algo,
            yerr=[
                np.array(means[algo]) - np.array(mins[algo]),
                maxs[algo] - np.array(means[algo]),
            ],
            capsize=5,
        )

    plt.xlabel("Coreset Size")
    plt.ylabel("Performance (Accuracy)")
    plt.title("Algorithm Performance for different Coreset Sizes")
    plt.xticks(index + bar_width / 2, coreset_sizes)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    # Add annotation to indicate "Higher is Better"
    plt.text(
        x=0.5,
        y=1.05,
        s="(Higher is Better)",
        fontsize=12,
        ha="center",
        va="bottom",
        transform=plt.gca().transAxes,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
