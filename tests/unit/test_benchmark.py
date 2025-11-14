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
Tests for Benchmarking.

This module contains unit tests for the functions in the`benchmark.mnist_benchmark`
module. It tests the conversion of PyTorch datasets to JAX arrays and the training and
evaluation of a neural network model using these datasets.
"""

from contextlib import nullcontext as does_not_warn

import jax.numpy as jnp
import pytest
import torch
from jax import random
from torchvision.datasets import VisionDataset

from benchmark.mnist_benchmark import (
    MLP,
    DataSet,
    convert_to_jax_arrays,
    train_and_evaluate,
)
from coreax import Data, SquaredExponentialKernel
from coreax.benchmark_util import (
    IterativeKernelHerding,
    calculate_delta,
    initialise_solvers,
)
from coreax.coreset import Coresubset
from coreax.solvers import (
    CompressPlusPlus,
    KernelHerding,
    KernelThinning,
    MapReduce,
    PaddingInvariantSolver,
    RandomSample,
    RPCholesky,
    SteinThinning,
)


class MockDataset(VisionDataset):
    """Mock dataset class for testing purposes."""

    def __init__(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Initialise the MockDataset.

        Set up the dataset with features and labels.

        :param data: A tensor containing the dataset features.
        :param labels: A tensor containing the corresponding labels.
        """
        super().__init__(root="", transform=None, target_transform=None)
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        :return: The total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a sample from the dataset.

        :param idx: The index of the sample to retrieve.
        :return: A tuple containing the data and its corresponding label.
        """
        return self.data[idx], self.labels[idx]


def test_convert_to_jax_arrays() -> None:
    """
    Test the :func:`convert_to_jax_arrays`.

    Verify that the conversion from a PyTorch dataset to JAX arrays
    is performed correctly. Check if the output matches the expected
    JAX arrays.
    """
    data = torch.tensor([[1, 2], [3, 4], [5, 6]])
    labels = torch.tensor([0, 1, 0])
    pytorch_dataset = MockDataset(data, labels)

    data_jax, targets_jax = convert_to_jax_arrays(pytorch_dataset)

    assert jnp.array_equal(data_jax, jnp.array([[1, 2], [3, 4], [5, 6]]))
    assert jnp.array_equal(targets_jax, jnp.array([0, 1, 0]))


def test_train_and_evaluate() -> None:
    """
    Test the :func:`train_and_evaluate`.

    Test the training and evaluation of the MLP model using a
    synthetic dataset. Ensure that the function returns the expected
    keys and verify that the final test accuracy is within the range
    [0.0, 1.0].
    """
    train_data = jnp.array([[1.0] * 784, [0.5] * 784, [0.2] * 784])
    train_labels = jnp.array([0, 1, 0])
    test_data = jnp.array([[1.0] * 784, [0.5] * 784])
    test_labels = jnp.array([0, 1])

    train_set = DataSet(features=train_data, labels=train_labels)
    test_set = DataSet(features=test_data, labels=test_labels)

    rng = random.PRNGKey(0)
    model = MLP(2)

    config = {
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 2,
        "epochs": 5,
        "patience": 2,
        "min_delta": 0.01,
    }

    result = train_and_evaluate(train_set, test_set, model, rng, config)

    assert "final_test_loss" in result
    assert "final_test_accuracy" in result
    assert 0.0 <= result["final_test_accuracy"] <= 1.0


def test_initialise_solvers() -> None:
    """
    Test the :func:`initialise_solvers`.

    Verify that the returned dictionary contains callable functions that produce
    valid solver instances.
    """
    # Create a mock dataset (UMAP-transformed) with arbitrary values
    mock_data = Data(jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]))
    key = random.PRNGKey(42)
    cpp_oversampling_factor = 1

    # Initialise solvers
    solvers = initialise_solvers(mock_data, key, cpp_oversampling_factor)

    # Ensure solvers is a dictionary with the expected keys
    expected_solver_keys = [
        "Random Sample",
        "RP Cholesky",
        "Kernel Herding",
        "Stein Thinning",
        "Kernel Thinning",
        "Compress++",
        "Iterative Herding",
        "Iterative Probabilistic Herding (constant)",
        "Iterative Probabilistic Herding (cubic)",
    ]
    assert set(solvers.keys()) == set(expected_solver_keys)


def test_solver_instances() -> None:
    """
    Test :func:`initialise_solvers` returns an instance of the expected solver type.
    """
    mock_data = Data(jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]))
    key = random.PRNGKey(42)
    cpp_oversampling_factor = 1
    # Case 1: When leaf_size is not provided
    expected_solver_types_no_leaf = {
        "Random Sample": RandomSample,
        "RP Cholesky": RPCholesky,
        "Kernel Herding": KernelHerding,
        "Stein Thinning": SteinThinning,
        "Kernel Thinning": KernelThinning,
        "Compress++": CompressPlusPlus,
        "Iterative Herding": IterativeKernelHerding,
        "Iterative Probabilistic Herding (constant)": IterativeKernelHerding,
        "Iterative Probabilistic Herding (cubic)": IterativeKernelHerding,
    }
    solvers_no_leaf = initialise_solvers(mock_data, key, cpp_oversampling_factor)
    for solver_name, solver_function in solvers_no_leaf.items():
        solver_instance = solver_function(1)
        assert isinstance(solver_instance, expected_solver_types_no_leaf[solver_name])

    # Case 2: When leaf_size is provided
    leaf_size = 2
    expected_solver_types_with_leaf = {
        "Random Sample": RandomSample,
        "RP Cholesky": RPCholesky,
        "Kernel Herding": MapReduce,
        "Stein Thinning": MapReduce,
        "Kernel Thinning": MapReduce,
        "Compress++": CompressPlusPlus,
        "Iterative Herding": MapReduce,
        "Iterative Probabilistic Herding (constant)": MapReduce,
        "Iterative Probabilistic Herding (cubic)": MapReduce,
    }
    solvers_with_leaf = initialise_solvers(
        mock_data, key, cpp_oversampling_factor, leaf_size
    )
    for solver_name, solver_function in solvers_with_leaf.items():
        expected_solver_type = expected_solver_types_with_leaf[solver_name]
        context = does_not_warn()
        if expected_solver_type == MapReduce:
            base_solver = expected_solver_types_no_leaf[solver_name]
            if not issubclass(base_solver, PaddingInvariantSolver):
                context = pytest.warns(UserWarning)
        with context:
            solver_instance = solver_function(1)
        assert isinstance(solver_instance, expected_solver_types_with_leaf[solver_name])

    # For SteinThinning, run reduce to make sure the score function works
    stein_solver = solvers_no_leaf["Stein Thinning"](1)
    coreset, _ = stein_solver.reduce(mock_data)
    assert isinstance(coreset, Coresubset)


@pytest.mark.parametrize("n", [1, 2, 100])
def test_calculate_delta(n):
    """
    Test the `calculate_delta` function.

    Ensure that the function produces a positive delta value for different values of n.
    """
    delta = calculate_delta(n)
    assert delta > 0


def test_iterative_kernel_herding_reduce() -> None:
    """Check that `IterativeKernelHerding.reduce` = `KernelHerding.reduce_iterative`."""
    random_key = random.key(0)
    dataset = Data(random.uniform(random_key, (100, 5)))

    solver_params = {
        "coreset_size": 10,
        "kernel": SquaredExponentialKernel(),
        "probabilistic": True,
        "temperature": 0.001,
        "random_key": random_key,
    }
    iter_params = {"num_iterations": 5, "t_schedule": jnp.ones(5) * 0.0001}
    solver_kh = KernelHerding(**solver_params)
    solver_ikh = IterativeKernelHerding(**solver_params, **iter_params)

    coreset_kh, state = solver_kh.reduce_iterative(dataset, **iter_params)
    coreset_ikh, _ = solver_ikh.reduce(dataset, solver_state=state)

    assert jnp.array_equal(
        coreset_kh.unweighted_indices, coreset_ikh.unweighted_indices
    )


if __name__ == "__main__":
    pytest.main()
