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
    Visualise the benchmarking results in five separate plots.

    :param data: A dictionary where keys are the coreset sizes (as strings) and values
                 are dictionaries containing the metrics for each algorithm.

    Example:
                 {
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
    title_size = 22
    label_size = 20
    tick_size = 18
    legend_size = 18

    first_coreset_size = next(iter(data.keys()))
    first_algorithm = next(iter(data[first_coreset_size].values()))
    metrics = list(first_algorithm.keys())

    for metric in metrics:
        plt.figure(figsize=(15, 10))
        plt.title(
            f"{metric.replace('_', ' ').title()} vs Coreset Size",
            fontsize=title_size,
            fontweight="bold",
        )

        for algo in data[first_coreset_size].keys():
            coreset_sizes = sorted(map(int, data.keys()))
            metric_values = [
                data[str(size)][algo].get(metric, float("nan"))
                for size in coreset_sizes
            ]

            plt.plot(
                coreset_sizes,
                metric_values,
                marker="o",
                markersize=8,
                linewidth=2.5,
                label=algo,
            )

        plt.xlabel("Coreset Size", fontsize=label_size, fontweight="bold")
        plt.ylabel(
            f"{metric.replace('_', ' ').title()}",
            fontsize=label_size,
            fontweight="bold",
        )
        plt.yscale("log")
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)

        plt.legend(fontsize=legend_size, loc="best", frameon=True)

        plt.grid(True, linestyle="--", alpha=0.7)

        plt.savefig(
            f"../examples/benchmarking_images/blobs_{metric.lower()}.png",
            bbox_inches="tight",
        )


# Function to print metrics table for each sample size
def print_metrics_table(data: dict, coreset_size: str) -> None:
    """
    Print a table for the given sample size with methods as rows and metrics as columns.

    :param data: A dictionary where keys are the coreset sizes (as strings) and values
                 that are dictionaries containing the metrics for each algorithm.

    Example:
                 {
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
    print(f"\nCoreset Size: {coreset_size}")
    print(separator)
    print(formatted_header)
    print(separator)

    for method, metrics in methods.items():
        row = [method] + [f"{value:.6f}" for value in metrics.values()]
        formatted_row = " | ".join(f"{r:^15}" for r in row)
        print(formatted_row)

    print(separator)


def print_rst_metrics_table(data: dict, original_sample_size: int) -> None:
    """
    Print metrics tables in reStructuredText format with highlighted best values.

    :param data: Dictionary with coreset sizes as keys and nested metrics data
    :param original_sample_size: The size of the original sample to display
    """
    metrics = [
        "Unweighted_MMD",
        "Unweighted_KSD",
        "Weighted_MMD",
        "Weighted_KSD",
        "Time",
    ]

    for coreset_size, methods_data in sorted(data.items(), key=lambda x: int(x[0])):
        if coreset_size == "n_samples":  # Skip the sample size entry
            continue

        print(
            f".. list-table:: Coreset Size {coreset_size} "
            f"(Original Sample Size {original_sample_size:,})"
        )
        print("   :header-rows: 1")
        print("   :widths: 20 15 15 15 15 15")
        print()
        print("   * - Method")
        for metric in metrics:
            print(f"     - {metric}")

        # Find best (minimum) values for each metric
        best_values = {
            metric: min(methods_data[method][metric] for method in methods_data)
            for metric in metrics
        }

        for method in methods_data:
            print(f"   * - {method}")
            for metric in metrics:
                value = methods_data[method][metric]
                if value == best_values[metric]:
                    print(f"     - **{value:.6f}**")  # Highlight best value
                else:
                    print(f"     - {value:.6f}")
            print()


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

    print_rst_metrics_table(data, original_sample_size=1024)
    plot_benchmarking_results(data)


if __name__ == "__main__":
    main()
