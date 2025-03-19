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
Benchmark performance of different coreset algorithms on pixel data from an image.

The benchmarking process follows these steps:
1. Load an input image and downsample (downsampling reduces the image resolution,
   i.e., the number of points in the dataset, to reduce the computational load on the
   machine this script is run on. A downsampling factor of 1 corresponds to no
   downsampling).
2. Convert to grayscale and extract pixel locations and values.
3. Generate coresets with different algorithms.
4. Plot the original image alongside coresets generated by each algorithm.
5. Save the resulting plot as an output file.

Each coreset algorithm is timed to measure and report the time taken for each step.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Pylint and Ruff are fighting over the position of the local import
# pylint: disable=wrong-import-position
import math
import time
from pathlib import Path
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import random

from coreax import Data
from coreax.benchmark_util import initialise_solvers
from examples.david_map_reduce_weighted import downsample_opencv

# pylint: enable=wrong-import-position

MAX_8BIT = 255


# pylint: disable=too-many-locals
# ruff: noqa: PLR0914, PLR0915
# (disable line too long and too many statements ruff)
def benchmark_coreset_algorithms(
    in_path: Path = Path("../examples/data/david_orig.png"),
    out_path: Optional[Path] = Path(
        "../examples/benchmarking_images/david_benchmark_results.png"
    ),
    downsampling_factor: int = 1,
):
    """
    Benchmark the performance of coreset algorithms on a downsampled greyscale image.

    Downsample an input image, extract pixel data, and obtain coresets using various
    algorithms. The original and coreset data are plotted and saved, and the execution
    time is printed.

    :param in_path: Path to the input image file.
    :param out_path: Path to save the output benchmark plot image.
    :param downsampling_factor: Factor by which to downsample the image.
    """
    # Base directory of the current script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Convert to absolute paths using os.path.join
    if not in_path.is_absolute():
        in_path = Path(os.path.join(base_dir, in_path))
    if out_path is not None and not out_path.is_absolute():
        out_path = Path(os.path.join(base_dir, out_path))

    original_data = downsample_opencv(str(in_path), downsampling_factor)
    pre_coreset_data = np.column_stack(np.nonzero(original_data < MAX_8BIT))
    pixel_values = original_data[original_data < MAX_8BIT]
    pre_coreset_data = np.column_stack((pre_coreset_data, pixel_values)).astype(
        np.float32
    )
    # Set up the original data object and coreset parameters
    data = Data(jnp.asarray(pre_coreset_data))
    over_sampling_factor = math.floor(math.log(data.shape[0], 4))
    coreset_size = 8_000 // (downsampling_factor**2)
    # Initialize each coreset solver
    key = random.PRNGKey(0)
    solver_factories = initialise_solvers(
        data, key, cpp_oversampling_factor=over_sampling_factor
    )

    # Dictionary to store coresets generated by each method
    coresets = {}
    solver_times = {}

    for solver_name, solver_creator in solver_factories.items():
        solver = solver_creator(coreset_size)
        start_time = time.perf_counter()
        coreset, _ = eqx.filter_jit(solver.reduce)(data)
        duration = time.perf_counter() - start_time
        coresets[solver_name] = coreset.points.data
        solver_times[solver_name] = duration

        # Save individual plots for each coreset method
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            coreset.points.data[:, 1],
            -coreset.points.data[:, 0],
            c=coreset.points.data[:, 2],
            cmap="gray",
            s=10.0 * downsampling_factor**2,
            marker="h",
            alpha=0.8,
        )
        ax.axis("off")

        individual_plot_path = f"{solver_name}_coreset_plot.png"
        plt.savefig(individual_plot_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"Saved individual plot for {solver_name} at {individual_plot_path}")

    original_plot_path = "original_image_plot.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(original_data, cmap="gray")
    ax.axis("off")
    plt.savefig(original_plot_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved original image plot at {original_plot_path}")

    # Save the combined benchmark plot
    if out_path:
        plt.figure(figsize=(15, 10))
        plt.subplot(4, 3, 1)
        plt.imshow(original_data, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        for i, (solver_name, coreset_data) in enumerate(coresets.items(), start=2):
            plt.subplot(4, 3, i)
            plt.scatter(
                coreset_data[:, 1],
                -coreset_data[:, 0],
                c=coreset_data[:, 2],
                cmap="gray",
                s=10.0 * downsampling_factor**2,
                marker="h",
                alpha=0.8,
            )
            plt.title(f"{solver_name} ({solver_times[solver_name]:.4f} s)")
            plt.axis("scaled")
            plt.axis("off")

        plt.savefig(out_path)
        print(f"Benchmark plot saved to {out_path}")


if __name__ == "__main__":
    benchmark_coreset_algorithms()
