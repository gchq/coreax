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


# Function to print metrics table for each sample size
def print_metrics_table(data: dict, sample_size: str) -> None:
    """
    Print a table for the given sample size with methods as rows and metrics as columns.

    :param data: A dictionary where keys are sample sizes (as strings) and values are
                 dictionaries containing the metrics for each algorithm. Each method's
                 dictionary contains the following keys:
                 - 'unweighted_mmd': Unweighted maximum mean discrepancy (MMD).
                 - 'unweighted_ksd': Unweighted kernel Stein discrepancy (KSD).
                 - 'weighted_mmd': Weighted maximum mean discrepancy (MMD).
                 - 'weighted_ksd': Weighted kernel Stein discrepancy (KSD).
                 - 'time': Time taken to compute the coreset and metrics (in seconds).
                 Example format:
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
    :param sample_size: The sample size for which to print the table.
    """
    # Define header
    header = (
        f"| {'Method':^15} | {'unweighted_mmd':^15} | {'unweighted_ksd':^15} | "
        f"{'weighted_mmd':^15} | {'weighted_ksd':^15} | {'time':^10} |"
    )
    separator = "-" * len(header)

    # Print table for the current sample size
    print(f"\nSample Size: {sample_size}")
    print(separator)
    print(header)
    print(separator)

    for method, metrics in data[sample_size].items():
        print(
            f"| {method:^15} | {metrics['unweighted_mmd']:^15.8f} | "
            f"{metrics['unweighted_ksd']:^15.8f} | {metrics['weighted_mmd']:^15.8f} | "
            f"{metrics['weighted_ksd']:^15.8f} | {metrics['time']:^10.3f} |"
        )

    print(separator)


def main() -> None:
    """Load the data and print metrics in table format per sample size."""
    # Load the JSON data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(base_dir, "blobs_benchmark_results.json"), encoding="utf-8"
    ) as f:
        data = json.load(f)

    # Print tables for each sample size
    for sample_size in data.keys():
        print_metrics_table(data, sample_size)


if __name__ == "__main__":
    main()
