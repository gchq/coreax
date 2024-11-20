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
Benchmark time taken to generate coresets from a large dataset.

The benchmarking process follows these steps:
1. Start with the MNIST dataset, which consists of 60_000 training images and 10_000
   test images.
2. Use a simple MLP neural network with a single hidden layer of 64 nodes to classify
   the images. The images are flattened into vectors.
3. To reduce dimensionality, apply PCA to project the 28x28 images into 16 components
   before applying coreset algorithms.
4. Generate coresets of different sizes using various coreset algorithms and record the
   time taken.
"""

import json
import os
import time
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from coreax import Data
from coreax.kernels import SquaredExponentialKernel, SteinKernel, median_heuristic
from coreax.score_matching import KernelDensityMatching
from coreax.solvers import (
    KernelHerding,
    MapReduce,
    RandomSample,
    RPCholesky,
    SteinThinning,
)


# Convert PyTorch dataset to JAX arrays
def convert_to_jax_arrays(pytorch_data: Dataset) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert a PyTorch dataset to JAX arrays.

    :param pytorch_data: PyTorch dataset to convert.
    :return: Tuple of JAX arrays (data, targets).
    """
    # Load all data in one batch
    data_loader = DataLoader(pytorch_data, batch_size=len(pytorch_data))
    # Grab the first batch, which is all data
    _data, _targets = next(iter(data_loader))
    # Convert to NumPy first, then JAX array
    data_jax = jnp.array(_data.numpy())
    targets_jax = jnp.array(_targets.numpy())
    return data_jax, targets_jax


def pca(x: jnp.ndarray, n_components: int = 16) -> jnp.ndarray:
    """
    Perform Principal Component Analysis (PCA) on the dataset x.

    This function computes the principal components of the input data matrix
    and returns the projected data.

    :param x: The input data matrix of shape (n_samples, n_features).
    :param n_components: The number of principal components to return.
    :return: The projected data of shape (n_samples, n_components).
    """
    # Center the data
    x_centred = x - jnp.mean(x, axis=0)

    # Compute the covariance matrix
    cov_matrix = jnp.cov(x_centred.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov_matrix)

    # Sort eigenvectors by descending eigenvalues
    _idx = jnp.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, _idx]

    # Select top n_components eigenvectors
    components = eigenvectors[:, :n_components]

    # Project the data onto the new subspace
    x_pca = jnp.dot(x_centred, components)

    return x_pca


def prepare_datasets() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Prepare and return training and test datasets.

    :return: A tuple containing training and test datasets in JAX arrays:
             (train_data_jax, train_targets_jax, test_data_jax, test_targets_jax).
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_data_jax, train_targets_jax = convert_to_jax_arrays(train_dataset)
    test_data_jax, test_targets_jax = convert_to_jax_arrays(test_dataset)

    return train_data_jax, train_targets_jax, test_data_jax, test_targets_jax


def initialise_solvers(train_data_pca: Data, key: jax.random.PRNGKey) -> list[Callable]:
    """
    Initialise and return a list of solvers for various coreset algorithms.

    Set up solvers for Kernel Herding, Stein Thinning, Random Sampling, and Randomised
    Cholesky methods. Each solver has different parameter requirements. Some solvers
    can utilise MapReduce, while others cannot,and some require specific kernels.
    This setup allows them to be called by passing only the coreset size,
    enabling easy integration in a loop for benchmarking.

    :param train_data_pca: The PCA-transformed training data used for
                           length scale estimation for ``SquareExponentialKernel``.
    :param key: The random key for initialising random solvers.

    :return: A list of solvers functions for different coreset algorithms.
    """
    # Set up kernel using median heuristic
    num_data_points = len(train_data_pca)
    num_samples_length_scale = min(num_data_points, 300)
    random_seed = 45
    generator = np.random.default_rng(random_seed)
    idx = generator.choice(num_data_points, num_samples_length_scale, replace=False)
    length_scale = median_heuristic(train_data_pca[idx])
    kernel = SquaredExponentialKernel(length_scale=length_scale)

    def _get_herding_solver(_size: int) -> MapReduce:
        """
        Set up KernelHerding to use ``MapReduce``.

        Create a KernelHerding solver with the specified size and return
        it along with a MapReduce object for reducing a large dataset like
        MNIST dataset.

        :param _size: The size of the coreset to be generated.
        :return: A tuple containing the solver name and the MapReduce solver.
        """
        herding_solver = KernelHerding(_size, kernel)
        return MapReduce(herding_solver, leaf_size=3 * _size)

    def _get_stein_solver(_size: int) -> MapReduce:
        """
        Set up Stein Thinning to use ``MapReduce``.

        Create a SteinThinning solver with the specified  coreset size,
        using ``KernelDensityMatching`` score function for matching on
        a subset of the dataset.

        :param _size: The size of the coreset to be generated.
        :return: A tuple containing the solver name and the MapReduce solver.
        """
        # Generate small dataset for ScoreMatching for Stein Kernel

        score_function = KernelDensityMatching(length_scale=length_scale).match(
            train_data_pca[idx]
        )
        stein_kernel = SteinKernel(kernel, score_function)
        stein_solver = SteinThinning(coreset_size=_size, kernel=stein_kernel)
        return MapReduce(stein_solver, leaf_size=3 * _size)

    def _get_random_solver(_size: int) -> RandomSample:
        """
        Set up Random Sampling to generate a coreset.

        :param _size: The size of the coreset to be generated.
        :return: A tuple containing the solver name and the RandomSample solver.
        """
        random_solver = RandomSample(_size, key)
        return random_solver

    def _get_rp_solver(_size: int) -> RPCholesky:
        """
        Set up Randomised Cholesky solver.

        :param _size: The size of the coreset to be generated.
        :return: A tuple containing the solver name and the RPCholesky solver.
        """
        rp_solver = RPCholesky(coreset_size=_size, kernel=kernel, random_key=key)
        return rp_solver

    return [_get_random_solver, _get_rp_solver, _get_herding_solver, _get_stein_solver]


def save_results(results: dict) -> None:
    """
    Save benchmark results to a JSON file for algorithm performance visualisation.

    :param results: A dictionary of results structured as follows:
                    {
                        "algorithm_name": {
                            "coreset_size_1": {
                                "run_1": accuracy_value,
                                "run_2": accuracy_value,
                                ...
                            },
                            "coreset_size_2": {
                                "run_1": accuracy_value,
                                "run_2": accuracy_value,
                                ...
                            },
                            ...
                        },
                        "another_algorithm_name": {
                            "coreset_size_1": {
                                "run_1": accuracy_value,
                                "run_2": accuracy_value,
                                ...
                            },
                            ...
                        },
                        ...
                    }
                    Each algorithm contains coreset sizes as keys, with values being
                    dictionaries of accuracy results from different runs.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = "mnist_time_results.json"
    with open(os.path.join(base_dir, file_name), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Data has been saved to {file_name}")


# pylint: disable=too-many-locals
def main() -> None:
    """
    Perform the benchmark for multiple solvers, coreset sizes, and random seeds.

    The function follows these steps:
    1. Prepare and load the MNIST datasets (training and test).
    2. Perform dimensionality reduction on the training data using PCA.
    3. Initialise solvers for data reduction.
    4. For each solver and coreset size, reduce the dataset and train the model
       on the reduced set.
    5. Train the model and evaluate its performance on the test set.
    6. Save the results, which include test accuracy for each solver and coreset size.
    """
    (train_data_jax, _, _, _) = prepare_datasets()
    train_data_pca = Data(pca(train_data_jax))

    coreset_times = {}

    # Run the experiment with 5 different random keys
    for i in range(5):
        print(f"Run {i + 1} of 5:")
        key = jax.random.PRNGKey(i)
        solvers = initialise_solvers(train_data_pca, key)
        for getter in solvers:
            for size in [25, 50, 100, 500, 1_000]:
                solver = getter(size)
                solver_name = (
                    solver.base_solver.__class__.__name__
                    if solver.__class__.__name__ == "MapReduce"
                    else solver.__class__.__name__
                )
                start_time = time.perf_counter()
                _, _ = eqx.filter_jit(solver.reduce)(train_data_pca)
                time_taken = time.perf_counter() - start_time

                # Ensure that there is a dictionary for this solver
                # If not, initialise with an empty dictionary
                if solver_name not in coreset_times:
                    coreset_times[solver_name] = {}

                # Populate the dictionary created above with coreset_size as keys
                # The values themselves will be dictionaries, so initialise with an
                # empty dictionary
                if size not in coreset_times[solver_name]:
                    coreset_times[solver_name][size] = {}

                # Store time taken result in nested structure
                coreset_times[solver_name][size][i] = time_taken

    # Save or print results
    save_results(coreset_times)


if __name__ == "__main__":
    main()
