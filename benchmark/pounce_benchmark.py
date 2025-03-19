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
Benchmark performance of different coreset algorithms on frames of a video.

The benchmarking process follows these steps:
1. Load an input GIF and preprocess its frames.
2. Reshape the frame data and apply UMAP for dimensionality reduction.
3. Generate coresets using each algorithm and save the selected frames as GIFs.
4. Print the time taken to generate each coreset.
"""

import time
from pathlib import Path

import imageio
import jax.numpy as jnp
import numpy as np
import umap
from jax import random
from matplotlib import pyplot as plt

from coreax.benchmark_util import initialise_solvers
from coreax.data import Data


def plot_selected_frames(
    umap_data, selected_indices, action_frames, solver_name, out_dir
):
    """
    Plot the selected frames and action frames on a bar chart.

    :param umap_data: The UMAP-transformed data.
    :param selected_indices: Indices of the selected frames.
    :param action_frames: Indices of the action frames.
    :param solver_name: The name of the solver used.
    :param out_dir: The output directory for saved plots.
    """
    x = np.arange(len(umap_data))
    y = np.zeros(len(umap_data))
    y[selected_indices] = 1.0

    z = np.zeros(len(umap_data))
    z[jnp.intersect1d(selected_indices, action_frames)] = 1.0

    plt.figure(figsize=(20, 3))
    plt.bar(x, y, alpha=0.5, label="Selected Frames")
    plt.bar(x, z, label="Action Frames")
    plt.xlabel("Frame", fontsize=18)
    plt.ylabel("Chosen", fontsize=18)
    plt.title(f"Selected Frames for {solver_name}", fontsize=24, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    output_frames_path = out_dir / f"frames_{solver_name.replace(' ', '_')}.png"
    plt.savefig(output_frames_path)


def benchmark_coreset_algorithms(
    in_path: Path = Path("../examples/data/pounce/pounce.gif"),
    out_dir: Path = Path("../examples/benchmarking_images/pounce"),
    coreset_size: int = 10,
):
    """
    Benchmark coreset algorithms by processing a video GIF.

    :param in_path: Path to the input GIF file, relative to the script's location.
    :param out_dir: Directory to save the output GIFs for each coreset algorithm,
        relative to the script's location.
    :param coreset_size: The size of the coreset.
    """
    base_dir = Path(__file__).resolve().parent

    # Ensure paths are absolute and output directory exists
    in_path = (base_dir / in_path).resolve()
    out_dir = (base_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess video frames
    _, *image_data = imageio.v2.mimread(in_path)
    raw_data = np.asarray(image_data)
    reshaped_data = raw_data.reshape(raw_data.shape[0], -1)

    umap_model = umap.UMAP(densmap=True, n_components=25)
    umap_data = jnp.asarray(umap_model.fit_transform(reshaped_data))
    print("umap_data_shape", umap_data.shape)

    solver_factories = initialise_solvers(
        Data(umap_data), random.PRNGKey(45), cpp_oversampling_factor=3
    )
    for solver_name, solver_creator in solver_factories.items():
        solver = solver_creator(coreset_size)
        data = Data(umap_data)

        start_time = time.perf_counter()
        coreset, _ = solver.reduce(data)
        duration = time.perf_counter() - start_time

        selected_indices = np.sort(np.asarray(coreset.unweighted_indices))

        coreset_frames = raw_data[selected_indices]
        output_gif_path = out_dir / f"{solver_name.replace(' ', '_')}_coreset.gif"
        imageio.v3.imwrite(output_gif_path, coreset_frames, loop=0)
        print(f"Saved {solver_name} coreset GIF to {output_gif_path}")
        print(f"time taken: {solver_name:<25} {duration:<30.4f}")

        plot_selected_frames(
            umap_data=umap_data,
            selected_indices=selected_indices,
            action_frames=np.arange(63, 85),
            solver_name=solver_name,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    benchmark_coreset_algorithms()
