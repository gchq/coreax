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
1. Load an input image and downsample it using a specified factor.
2. Convert to grayscale and extract non-zero pixel locations and values.
3. Generate coresets with varying algorithms.
4. Plot the original image alongside coresets generated by each algorithm.
5. Save the resulting plot as an output file.

Each coreset algorithm is timed to measure and report the time taken for each step.
"""

import os
import time
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from jax import random
from mnist_benchmark import initialise_solvers

from coreax import Data

MAX_8BIT = 255


def downsample_opencv(image_path: str, downsampling_factor: int) -> np.ndarray:
    """
    Downsample an image using OpenCV resize and convert it to grayscale.

    :param image_path: Path to the input image file.
    :param downsampling_factor: Factor by which to downsample the image.
    :return: Grayscale image after downsampling.
    """
    img = cv2.imread(image_path)

    # Calculate new dimensions based on downsampling factor
    scale_factor = 1 / downsampling_factor
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dim = (width, height)

    # Resize using INTER_AREA for better downsampling
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # Convert to grayscale after resizing
    grayscale_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    return grayscale_resized


def benchmark_coreset_algorithms(
    in_path: Path = Path("../examples/data/david_orig.png"),
    out_path: Optional[Path] = Path("david_benchmark_results.png"),
    downsampling_factor: int = 2,
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
    data = Data(pre_coreset_data)
    coreset_size = 8_000 // (downsampling_factor**2)

    # Initialize each coreset solver
    key = random.PRNGKey(0)
    solvers = initialise_solvers(pre_coreset_data, key)

    # Dictionary to store coresets generated by each method
    coresets = {}

    # Print header for timing results
    print("\nCoreset Generation Times:")
    print(f"{'Coreset Algorithm':<25} {'Generation Time (seconds)':<30}")
    print("-" * 55)

    # Generate coresets for each method and time each solver
    for get_solver in solvers:
        solver_name, solver = get_solver(coreset_size)
        start_time = time.perf_counter()
        coreset, _ = solver.reduce(data)
        duration = time.perf_counter() - start_time
        coresets[solver_name] = coreset.coreset.data

        print(f"{solver_name:<25} {duration:<30.4f}")

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(original_data, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    # Plot each coreset method
    for i, (solver_name, coreset_data) in enumerate(coresets.items(), start=2):
        plt.subplot(2, 3, i)
        plt.scatter(
            coreset_data[:, 1],
            -coreset_data[:, 0],
            c=coreset_data[:, 2],
            cmap="gray",
            s=5.0 * downsampling_factor**2,  # Set a constant marker size
            marker="h",
            alpha=0.8,
        )
        plt.title(solver_name)
        plt.axis("scaled")
        plt.axis("off")

    # Save plot to file instead of showing
    if out_path:
        plt.savefig(out_path)
        print(f"Benchmark plot saved to {out_path}")


# Call the function to run the benchmark and save results
if __name__ == "__main__":
    benchmark_coreset_algorithms()
