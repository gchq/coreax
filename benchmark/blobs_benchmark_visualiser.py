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

"""Visualise the results of ``blobs_benchmark.py``."""

import json
import os

from matplotlib import pyplot as plt


def plot_benchmarking_results(data):
    """
    Visualise the benchmarking results.

    :param data: A dictionary where the first key is the original sample size
                 and the rest of the keys are the coreset sizes (as strings) and values
                 that are dictionaries containing the metrics for each algorithm.

    Example:
                 {
                    'n_samples': 1000,
                     '100': {
                         'KernelHerding': {
                             'unweighted_mmd': 0.12345678,
                             'unweighted_ksd': 0.23456789,
                             'weighted_mmd': 0.34567890,
                             'weighted_ksd': 0.45678901,
                             'time': 0.123
                         },
                         'Algorithm B': { ... },
                         ...
                     },
                     '1000': { ... },
                     ...
                 }

    """
    # Extract n_samples
    n_samples = data.pop("n_samples")

    first_coreset_size = next(iter(data.keys()))
    first_algorithm = next(
        iter(data[first_coreset_size].values())
    )  # Get one example algorithm
    metrics = list(first_algorithm.keys())
    n_metrics = len(metrics)

    n_rows = (n_metrics + 1) // 2
    fig, axs = plt.subplots(n_rows, 2, figsize=(14, 6 * n_rows))
    fig.delaxes(axs[2, 1])
    axs = axs.flatten()

    # Iterate over each metric and create its subplot
    for i, metric in enumerate(metrics):
        ax = axs[i]
        ax.set_title(
            f"{metric.replace('_', ' ').title()} vs "
            f"Coreset Size (n_samples = {n_samples})",
            fontsize=14,
        )

        # For each algorithm, plot its performance across different subset sizes
        for algo in data[list(data.keys())[0]].keys():  # Iterating through algorithms
            # Create lists of subset sizes (10, 50, 100, 200)
            coreset_sizes = sorted(map(int, data.keys()))
            metric_values = [
                data[str(subset_size)][algo].get(metric, float("nan"))
                for subset_size in coreset_sizes
            ]

            ax.plot(coreset_sizes, metric_values, marker="o", label=algo)

        ax.set_xlabel("Coreset Size")
        ax.set_ylabel(f"{metric.replace('_', ' ').title()}")
        ax.set_yscale("log")  # log scale for better visualization
        ax.legend()

    # Adjust layout to avoid overlap
    plt.subplots_adjust(hspace=15.0, wspace=1.0)
    plt.tight_layout(pad=3.0, rect=(0.0, 0.0, 1.0, 0.96))
    plt.show()


# Function to print metrics table for each sample size
def print_metrics_table(data: dict, coreset_size: str) -> None:
    """
    Print a table for the given sample size with methods as rows and metrics as columns.

    :param data: A dictionary where the first key is the original sample size
                 and the rest of the keys are the coreset sizes (as strings) and values
                 that are dictionaries containing the metrics for each algorithm.

    Example:
                 {
                    'n_samples': 1000,
                     '100': {
                         'KernelHerding': {
                             'unweighted_mmd': 0.12345678,
                             'unweighted_ksd': 0.23456789,
                             'weighted_mmd': 0.34567890,
                             'weighted_ksd': 0.45678901,
                             'time': 0.123
                         },
                         'Algorithm B': { ... },
                         ...
                     },
                     '1000': { ... },
                     ...
                 }
    :param coreset_size: The coreset size for which to print the table.

    """
    # Define header
    methods = data[coreset_size]
    header = ["Method"] + list(next(iter(methods.values())).keys())
    formatted_header = " | ".join(f"{method:15}" for method in header)
    separator = "-" * len(formatted_header)

    # Print table for the current coreset size
    print(f"\nCoreset Size: {coreset_size} (Original Sample Size: {data['n_samples']})")
    print(separator)
    print(formatted_header)
    print(separator)

    for method, metrics in methods.items():
        row = [method] + [f"{value:.6f}" for value in metrics.values()]
        formatted_row = " | ".join(f"{r:^15}" for r in row)
        print(formatted_row)

    print(separator)


def main() -> None:
    """Load the data and print metrics in table format per sample size."""
    # Load the JSON data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(base_dir, "blobs_benchmark_results.json"), encoding="utf-8"
    ) as f:
        data = json.load(f)

    # Print tables for each coreset size
    for coreset_size in data:
        if coreset_size == "n_samples":
            continue
        print_metrics_table(data, coreset_size)

    plot_benchmarking_results(data)


if __name__ == "__main__":
    main()
