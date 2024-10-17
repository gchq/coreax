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

from matplotlib import pyplot as plt


# Open the JSON file
def main() -> None:
    """Load benchmark results and visualize the algorithm performance."""
    with open("benchmark_results.json", "r", encoding="utf-8") as file:
        # Load the JSON data into a Python object
        data_by_solver = json.load(file)

    # Visualization code
    plt.figure(figsize=(12, 6))

    for algo, data in data_by_solver.items():
        plt.plot(data["coreset_size"], data["accuracy"], "o-", label=algo)

    plt.xlabel("Iteration")
    plt.ylabel("Performance Metric")
    plt.title("Algorithm Performance Across Iterations")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.show()


if __name__ == "__main__":
    main()
