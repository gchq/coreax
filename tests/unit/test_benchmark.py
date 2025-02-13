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
from coreax import Data
from coreax.benchmark_util import calculate_delta, get_solver_name, initialise_solvers
from coreax.kernels.scalar_valued import SquaredExponentialKernel
from coreax.solvers import (
    KernelHerding,
    MapReduce,
    RandomSample,
    RPCholesky,
)


class MockDataset(VisionDataset):
    """Mock dataset class for testing purposes."""

    # We deliberately don't call super().__init__(), as this is a mock class
    def __init__(self, data: torch.Tensor, labels: torch.Tensor) -> None:  # pylint: disable=super-init-not-called
        """
        Initialise the MockDataset.

        Set up the dataset with features and labels.

        :param data: A tensor containing the dataset features.
        :param labels: A tensor containing the corresponding labels.
        """
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

    Verify that the returned list contains callable functions that produce
    valid solver instances.
    """
    # Create a mock dataset (UMAP-transformed) with arbitrary values
    mock_data = Data(jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]))
    key = random.PRNGKey(42)

    solvers = initialise_solvers(mock_data, key)
    for solver in solvers:
        solver_instance = solver(1)  # Instantiate with a coreset size of 1
        assert isinstance(solver_instance, (MapReduce, RandomSample, RPCholesky)), (
            f"Unexpected solver type: {type(solver_instance)}"
        )


def test_get_solver_name():
    """
    Test `get_solver_name` function to ensure it returns correct solver names.
    """
    # Create a KernelHerding solver
    herding_solver = KernelHerding(coreset_size=5, kernel=SquaredExponentialKernel())

    # Wrap it in MapReduce
    map_reduce_solver = MapReduce(base_solver=herding_solver, leaf_size=15)

    assert get_solver_name(lambda _: herding_solver) == "KernelHerding", (
        "Expected 'KernelHerding' but got something else."
    )

    assert get_solver_name(lambda _: map_reduce_solver) == "KernelHerding", (
        "Expected 'KernelHerding' from MapReduce solver but got something else."
    )


@pytest.mark.parametrize("n", [10, 100, 1000])
def test_calculate_delta(n):
    """
    Test the `calculate_delta` function.

    Ensure that the function produces a positive delta value for different values of n.
    """
    delta = calculate_delta(n)
    assert delta > 0, f"Delta should be positive but got {delta} for n={n}"


if __name__ == "__main__":
    pytest.main()
