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

import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_data(filename: str) -> dict:
    """
    Load benchmark data from a specified JSON file.

    :param filename: The path to the JSON file containing benchmark data.
    :return: A dictionary containing the benchmark data, or an empty dictionary
             if there was an error loading the file.
    """
    try:
        with open(filename, encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load benchmark data: {e}") from e


def compute_statistics(
    data_by_solver: dict, coreset_sizes: list[int]
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]]]:
    """
    Compute statistical summary (mean, min, max).

    The parameter data_by_solver contains time and accuracy data for different solvers
    for different seeds. The data for the first run is skipped because it is much slower
    due to JIT compilation.

    :param data_by_solver: A dictionary where each key is an algorithm name,
                           and each value is a dictionary mapping coreset size
                           to benchmark results. Benchmark results include multiple
                           runs with 'accuracy' and 'time_taken'.
    :param coreset_sizes: A list of integer coreset sizes to evaluate.
    :return: A tuple containing two dictionaries:
             - The first dictionary maps each algorithm name to its accuracy statistics,
               with keys 'means', 'min', 'max' and 'points'.
             - The second dictionary maps each algorithm name to its time statistics,
               also with keys 'means', 'min', 'max' and 'points'.
    """
    accuracy_stats = {
        algo: {
            "means": [],
            "min": [],
            "max": [],
            "points": {size: [] for size in coreset_sizes},
        }
        for algo in data_by_solver
    }
    time_stats = {
        algo: {
            "means": [],
            "min": [],
            "max": [],
            "points": {size: [] for size in coreset_sizes},
        }
        for algo in data_by_solver
    }

    for algo, sizes in data_by_solver.items():
        for size in coreset_sizes:
            size_str = str(size)
            accuracies, times = [], []
            if size_str in sizes:
                # Skip the first run as it is much slower due to JIT compilation
                run_list = list(sizes[size_str].values())[1:]
                for run_data in run_list:
                    accuracies.append(run_data["accuracy"])
                    times.append(run_data["time_taken"])
                    accuracy_stats[algo]["points"][size].append(run_data["accuracy"])
                    time_stats[algo]["points"][size].append(run_data["time_taken"])

            accuracy_stats[algo]["means"].append(np.mean(accuracies))
            accuracy_stats[algo]["min"].append(np.min(accuracies))
            accuracy_stats[algo]["max"].append(np.max(accuracies))

            time_stats[algo]["means"].append(np.mean(times))
            time_stats[algo]["min"].append(np.min(times))
            time_stats[algo]["max"].append(np.max(times))

    return accuracy_stats, time_stats


def compute_time_statistics(data: dict, coreset_sizes: list[int]) -> dict:
    """
    Compute statistical summary (mean, min, max) for standalone time data.

    The data for the first run is skipped because it is much slower due to JIT
    compilation.

    :param data: A dictionary containing time data for different algorithms
                 and coreset sizes.
    :param coreset_sizes: A list of integer coreset sizes to evaluate.
    :return: A dictionary containing the statistics (mean, min, max)
             and raw points for each algorithm.
    """
    stats = {
        algo: {
            "means": [],
            "min": [],
            "max": [],
            "points": {size: [] for size in coreset_sizes},
        }
        for algo in data
    }

    for algo, sizes in data.items():
        for size in coreset_sizes:
            size_str = str(size)
            times = []
            if size_str in sizes:
                # Skip the first run as it is much slower due to JIT compilation
                run_list = list(sizes[size_str].values())[1:]
                for time in run_list:
                    times.append(time)
                    stats[algo]["points"][size].append(time)

            stats[algo]["means"].append(np.mean(times))
            stats[algo]["min"].append(np.min(times))
            stats[algo]["max"].append(np.max(times))

    return stats


def plot_performance(
    stats: dict,
    coreset_sizes: list[int],
    ylabel: str,
    title: str,
    log_scale: bool = False,
) -> None:
    """
    Plot performance statistics for each algorithm over varying coreset sizes.

    :param stats: A dictionary containing statistics for algorithms, with keys 'means'
                  and 'points'.
    :param coreset_sizes: List of coreset sizes to display on the x-axis.
    :param ylabel: Label for the y-axis.
    :param title: Title of the plot.
    :param log_scale: Option to use a logarithmic scale for the y-axis.
    """
    n_algorithms = len(stats)
    bar_width = 0.8 / n_algorithms  # Divide available space for bars
    index = np.arange(len(coreset_sizes))  # x positions for coreset sizes

    plt.figure(figsize=(12, 8))  # Bigger figure size

    for i, algo in enumerate(stats):
        # Plot the bars for mean values
        plt.bar(
            index + i * bar_width,
            stats[algo]["means"],
            bar_width,
            label=algo,
            color=f"C{i}",
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add error bars with a larger capsize
        plt.errorbar(
            index + i * bar_width,
            stats[algo]["means"],
            yerr=[
                np.array(stats[algo]["means"]) - np.array(stats[algo]["min"]),
                np.array(stats[algo]["max"]) - np.array(stats[algo]["means"]),
            ],
            fmt="none",
            ecolor="black",
            capsize=7,  # Larger cap size for better visibility
            alpha=0.9,
            elinewidth=2,  # Thicker error bars
        )

        # Overlay individual points as larger dots
        for j, size in enumerate(coreset_sizes):
            x_positions = (
                index[j]
                + i * bar_width
                + np.random.uniform(
                    -0.02 * bar_width,
                    0.02 * bar_width,
                    len(stats[algo]["points"][size]),
                )
            )
            plt.scatter(
                x_positions,
                stats[algo]["points"][size],
                color=f"C{i}",
                s=40,  # Larger dots for better visibility
                edgecolor="black",
                linewidth=0.8,
                alpha=0.8,
            )

    # Add labels, titles, and formatting
    plt.ylabel(ylabel, fontsize=20, fontweight="bold")
    plt.xlabel("Coreset Size", fontsize=20, fontweight="bold")

    if log_scale:
        plt.yscale("log")

    plt.title(title, fontsize=24, fontweight="bold")
    plt.xticks(
        index + bar_width * (n_algorithms - 1) / 2,
        [str(size) for size in coreset_sizes],
        fontsize=18,
    )
    plt.yticks(fontsize=18)

    # Enhanced legend styling
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=(n_algorithms + 1) // 2,
        fontsize=18,
        frameon=True,
        edgecolor="black",
    )

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()


def main() -> None:
    """Load benchmark results and visualise the algorithm performance."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_by_solver = load_benchmark_data(
        os.path.join(base_dir, "mnist_benchmark_results.json")
    )

    time_only_data = load_benchmark_data(
        os.path.join(base_dir, "mnist_time_results.json")
    )

    coreset_sizes = sorted(
        {int(size) for sizes in data_by_solver.values() for size in sizes.keys()}
    )
    accuracy_stats, time_stats = compute_statistics(data_by_solver, coreset_sizes)

    coreset_sizes_time_only = sorted(
        {int(size) for sizes in time_only_data.values() for size in sizes.keys()}
    )

    time_stats_time_only = compute_time_statistics(
        time_only_data, coreset_sizes_time_only
    )

    # Plot accuracy results
    plt.figure(figsize=(12, 6))
    plot_performance(
        accuracy_stats,
        coreset_sizes,
        "Performance (Accuracy)",
        "Algorithm Performance (Accuracy) for Different Coreset Sizes",
    )

    plt.show()

    # Plot time taken results
    plt.figure(figsize=(12, 6))
    plot_performance(
        time_stats,
        coreset_sizes,
        "Time Taken (seconds)",
        "Algorithm Performance (Time Taken) for Different Coreset Sizes",
    )

    plt.figtext(
        0.5,
        0.91,
        "Plot showing the mean time taken to generate coresets and train MNIST"
        "classifier with coreset sizes with error bars representing min-max ranges",
        wrap=True,
        horizontalalignment="center",
        fontsize=12,
    )

    plt.show()

    # Plot time taken results
    plt.figure(figsize=(12, 6))
    plot_performance(
        time_stats_time_only,
        coreset_sizes_time_only,
        "Time Taken (seconds)",
        "Time Taken to generate coresets for Different Coreset Sizes",
        True,
    )

    plt.figtext(
        0.5,
        0.91,
        "Plot showing the mean time taken to generate coresets of different"
        " coreset sizes with error bars representing min-max ranges",
        wrap=True,
        horizontalalignment="center",
        fontsize=12,
    )

    plt.show()


if __name__ == "__main__":
    main()
