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
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading the file: {e}")
        return {}


def compute_statistics(data_by_solver: dict, coreset_sizes: list[int]) -> tuple[dict]:
    """
    Compute statistical summaries (mean, minimum, and maximum).

    :param data_by_solver: A dictionary where each key is an algorithm name,
                           and each value is a dictionary mapping coreset size
                           to benchmark results. Benchmark results include multiple
                           runs with 'accuracy' and 'time_taken'.
    :param coreset_sizes: A list of integer coreset sizes to evaluate.
    :return: A tuple containing two dictionaries:
             - The first dictionary maps each algorithm name to its accuracy statistics,
               with keys 'means', 'mins', and 'maxs'.
             - The second dictionary maps each algorithm name to its time statistics,
               also with keys 'means', 'mins', and 'maxs'.
    """
    accuracy_stats = {
        algo: {"means": [], "mins": [], "maxs": []} for algo in data_by_solver
    }
    time_stats = {
        algo: {"means": [], "mins": [], "maxs": []} for algo in data_by_solver
    }

    for algo, sizes in data_by_solver.items():
        for size in coreset_sizes:
            size_str = str(size)
            accuracies, times = [], []
            if size_str in sizes:
                for run_data in sizes[size_str].values():
                    accuracies.append(run_data["accuracy"])
                    times.append(run_data["time_taken"])

            accuracy_stats[algo]["means"].append(np.mean(accuracies))
            accuracy_stats[algo]["mins"].append(np.min(accuracies))
            accuracy_stats[algo]["maxs"].append(np.max(accuracies))
            time_stats[algo]["means"].append(np.mean(times))
            time_stats[algo]["mins"].append(np.min(times))
            time_stats[algo]["maxs"].append(np.max(times))

    return accuracy_stats, time_stats


def plot_performance(
    index: list[float],
    bar_width: float,
    data_dict: dict[str, dict[str, list[float]]],
    coreset_sizes: list[int],
    ylabel: str,
    title: str,
    log_scale: bool = True,
) -> None:
    """
    Plot performance statistics for each algorithm over varying coreset sizes.

    :param index: Array of positions for bar locations.
    :param bar_width: Width of each bar in the plot.
    :param data_dict: Dictionary of data statistics for each algorithm.
    :param coreset_sizes: List of coreset sizes to display on the x-axis.
    :param ylabel: Label for the y-axis.
    :param title: Title of the plot.
    :param log_scale: Option to use a logarithmic scale for the y-axis.
    """
    for i, algo in enumerate(data_dict):
        plt.bar(
            index + i * bar_width,
            data_dict[algo]["means"],
            bar_width,
            label=f"{algo} ({ylabel})",
            yerr=[
                np.array(data_dict[algo]["means"]) - np.array(data_dict[algo]["mins"]),
                np.array(data_dict[algo]["maxs"]) - np.array(data_dict[algo]["means"]),
            ],
            capsize=5,
        )

    plt.xlabel("Coreset Size")
    plt.ylabel(ylabel)
    if log_scale:
        plt.yscale("log")
    plt.title(title)
    plt.xticks(index + bar_width * (len(data_dict) - 1) / 2, coreset_sizes)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()


def main() -> None:
    """Load benchmark results and visualise the algorithm performance."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_by_solver = load_benchmark_data(
        os.path.join(base_dir, "mnist_benchmark_results.json")
    )
    if not data_by_solver:
        return

    coreset_sizes = sorted(
        {int(size) for sizes in data_by_solver.values() for size in sizes.keys()}
    )
    accuracy_stats, time_stats = compute_statistics(data_by_solver, coreset_sizes)

    # Plot accuracy results
    plt.figure(figsize=(12, 6))
    plot_performance(
        np.arange(len(coreset_sizes)),
        0.2,
        accuracy_stats,
        coreset_sizes,
        "Performance (Accuracy)",
        "Algorithm Performance (Accuracy) for Different Coreset Sizes",
    )

    plt.show()

    # Plot time taken results
    plt.figure(figsize=(12, 6))
    plot_performance(
        np.arange(len(coreset_sizes)),
        0.2,
        time_stats,
        coreset_sizes,
        "Time Taken (seconds)",
        "Algorithm Performance (Time Taken) for Different Coreset Sizes",
    )

    plt.show()
