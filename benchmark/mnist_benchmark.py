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
Benchmark performance of coreset algorithms on the MNIST dataset with a neural network.

The benchmarking process follows these steps:
1. Start with the MNIST dataset, which consists of 60_000 training images and 10_000
   test images.
2. Use a simple MLP neural network with a single hidden layer of 64 nodes to classify
   the images. The images are flattened into vectors.
3. To reduce dimensionality, apply UMAP to project the 28x28 images into 16 components
   before applying coreset algorithms.
4. Generate coresets of different sizes using various coreset algorithms.
   - For Kernel Herding and Stein Thinning, use MapReduce to handle larger-scale data.
5. Use the coreset indices to select the original images from the training set, and
   train the model on these selected coresets.
6. Evaluate the model's accuracy on the test set of 10_000 images.
7. Due to the inherent randomness in both coreset algorithms and the machine learning
   training process, repeat the experiment 5 times with different random seeds.
8. Store the results from each run and visualise them using
   `coreset.benchmark.mnist_benchmark_visualiser.py`, which plots error bars (min,
   max, mean) for accuracy across different coreset sizes.

The benchmark is run on amazon g4dn.12xlarge instance with 4 nvidia t4 tensor core
GPUs, 48 virtual CPUs and 192 GiB memory.
"""

import json
import os
import time
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import torchvision
import umap
from equinox.nn import State
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset

from coreax import Data
from coreax.benchmark_util import initialise_solvers
from coreax.util import KeyArrayLike


# Convert PyTorch dataset to JAX arrays
def convert_to_jax_arrays(
    pytorch_data: VisionDataset,
) -> tuple[jnp.ndarray, jnp.ndarray]:
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


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """
    Compute cross-entropy loss.

    :param logits: Logits predicted by the model.
    :param labels: Ground truth labels as an array of integers.
    :return: The cross-entropy loss.
    """
    return jnp.mean(
        optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, num_classes=10))
    )


def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """
    Compute loss and accuracy metrics.

    :param logits: Logits predicted by the model.
    :param labels: Ground truth labels as an array of integers.
    :return: A dictionary containing 'loss' and 'accuracy' as keys.
    """
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {"loss": loss, "accuracy": accuracy}


class MLP(eqx.Module):
    """
    Multi-layer perceptron with batch normalisation and dropout.

    :param random_key: The random key.
    :param input_size: Number of dimensions in input space.
    :param hidden_size: Number of units in the hidden layer.
    :param output_size: Number of output units.
    :param dropout_rate: Dropout rate to use during training.
    """

    linear_1: eqx.nn.Linear
    linear_2: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    batch_norm: eqx.nn.BatchNorm

    def __init__(
        self,
        random_key: KeyArrayLike,
        input_size: int,
        hidden_size: int,
        output_size: int = 10,
        dropout_rate: float = 0.2,
    ) -> None:
        """Initialise MLP."""
        key_1, key_2 = jr.split(random_key)
        self.linear_1 = eqx.nn.Linear(input_size, hidden_size, key=key_1)
        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.batch_norm = eqx.nn.BatchNorm(
            input_size=hidden_size, axis_name="batch", mode="batch"
        )
        self.linear_2 = eqx.nn.Linear(hidden_size, output_size, key=key_2)

    def __call__(
        self, x: jnp.ndarray, state: State, key: KeyArrayLike | None = None
    ) -> tuple[jnp.ndarray, State]:
        """
        Forward pass of the MLP.

        :param x: Input data.
        :return: Output logits of the network.
        """
        x = self.linear_1(x)
        x = self.dropout(x, key=key)
        x, state = self.batch_norm(x, state)
        x = jax.nn.relu(x)
        x = self.linear_2(x)
        return x, state


def compute_loss(
    model: MLP,
    state: State,
    batch_data: jnp.ndarray,
    batch_labels: jnp.ndarray,
    key: KeyArrayLike,
):
    """
    Compute cross-entropy.

    :param model: The current model.
    :param state: The current state of the model.
    :param batch_data: Batch of input data.
    :param batch_labels: Batch of ground truth labels.
    :param key: Random key for dropout.

    :return: cross-entropy, (model state, logits)
    """
    keys = jr.split(key, batch_data.shape[0])
    batch_model = jax.vmap(
        model, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None)
    )
    logits, state = batch_model(batch_data, state, keys)
    loss = cross_entropy_loss(logits, batch_labels)
    return loss, (state, logits)


@eqx.filter_jit
# pylint: disable-next=too-many-positional-arguments
def train_step(
    model: MLP,
    state: State,
    optimiser: optax.GradientTransformation,
    opt_state: optax.OptState,
    batch_data: jnp.ndarray,
    batch_labels: jnp.ndarray,
    key: KeyArrayLike,
) -> tuple[MLP, State, optax.OptState, jnp.ndarray]:
    """
    Make a training step.

    :param model: The current model.
    :param state: The current state of the model.
    :param optimiser: The optimiser.
    :param opt_state: The current state of the optimiser.
    :param batch_data: Batch of input data.
    :param batch_labels: Batch of ground truth labels.
    :param key: Random key for dropout.
    :return: Updated model, updated model state, updated optimiser state, and logits.
    """
    grads, (state, logits) = eqx.filter_grad(compute_loss, has_aux=True)(
        model, state, batch_data, batch_labels, key
    )
    updates, opt_state = optimiser.update(
        grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, logits


@eqx.filter_jit
def eval_step(
    model: MLP, state: State, batch_data: jnp.ndarray, batch_labels: jnp.ndarray
) -> dict[str, jnp.ndarray]:
    """
    Perform a single evaluation step.

    :param model: The current model.
    :param state: The current state of the model.
    :param batch_data: Batch of input data.
    :param batch_labels: Batch of ground truth labels.
    :return: A dictionary of evaluation metrics (loss and accuracy).
    """
    inference_model = eqx.Partial(eqx.nn.inference_mode(model), state=state)
    logits, _ = jax.vmap(inference_model)(batch_data)
    return compute_metrics(logits, batch_labels)


# pylint: disable-next=too-many-positional-arguments
def train_epoch(
    model: MLP,
    state: State,
    optimiser: optax.GradientTransformation,
    opt_state: optax.OptState,
    train_data: jnp.ndarray,
    train_labels: jnp.ndarray,
    batch_size: int,
    key: KeyArrayLike,
) -> tuple[MLP, State, optax.OptState, dict[str, jnp.ndarray]]:
    """
    Train for one epoch and return updated state and metrics.

    :param model: The current model.
    :param state: The current state of the model.
    :param optimiser: The optimiser.
    :param opt_state: The current state of the optimiser.
    :param train_data: Training input data.
    :param train_labels: Training labels.
    :param batch_size: Size of each training batch.
    :param key: Random key for dropout.
    :return: Updated model, model state, optimiser state, and a dictionary containing
        'loss' and 'accuracy'.
    """
    num_batches = train_data.shape[0] // batch_size
    total_loss, total_accuracy = jnp.array(0.0), jnp.array(0.0)

    for batch_idx in range(num_batches):
        key, subkey = jr.split(key)
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_data = train_data[start_idx:end_idx]
        batch_labels = train_labels[start_idx:end_idx]
        model, state, opt_state, logits = train_step(
            model, state, optimiser, opt_state, batch_data, batch_labels, subkey
        )
        metrics = compute_metrics(logits, batch_labels)
        total_loss += metrics["loss"]
        total_accuracy += metrics["accuracy"]

    return (
        model,
        state,
        opt_state,
        {
            "loss": total_loss / num_batches,
            "accuracy": total_accuracy / num_batches,
        },
    )


def evaluate(
    model: MLP,
    state: State,
    test_data: jnp.ndarray,
    test_labels: jnp.ndarray,
    batch_size: int,
) -> dict[str, jnp.ndarray]:
    """
    Evaluate the model on given data and return metrics.

    :param model: The current model.
    :param state: The current state of the model.
    :param test_data: Test input data.
    :param test_labels: Test labels.
    :param batch_size: Size of each training batch.
    :return: A dictionary containing 'loss' and 'accuracy' metrics.
    """
    num_batches = test_data.shape[0] // batch_size
    total_loss, total_accuracy = jnp.array(0.0), jnp.array(0.0)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_data = test_data[start_idx:end_idx]
        batch_labels = test_labels[start_idx:end_idx]
        metrics = eval_step(model, state, batch_data, batch_labels)
        total_loss += metrics["loss"]
        total_accuracy += metrics["accuracy"]

    return {
        "loss": total_loss / num_batches,
        "accuracy": total_accuracy / num_batches,
    }


class DataSet(NamedTuple):
    """Represents a dataset with features and labels."""

    features: jnp.ndarray
    labels: jnp.ndarray


# pylint: disable-next=too-many-positional-arguments
def train_and_evaluate(
    train_set: DataSet,
    test_set: DataSet,
    model: MLP,
    state: State,
    key: KeyArrayLike,
    config: dict[str, Any],
) -> dict[str, jnp.ndarray]:
    """
    Train and evaluate the model with early stopping.

    :param train_set: The training dataset containing features and labels.
    :param test_set: The test dataset containing features and labels.
    :param model: The model to be trained.
    :param state: The initial state of the model to be trained.
    :param key: Random key for dropout.
    :param config: A dictionary of training configuration parameters, including:
                   - "learning_rate": Learning rate for the optimiser.
                   - "weight_decay": Weight decay for the optimiser.
                   - "batch_size": Number of samples per training batch.
                   - "epochs": Total number of training epochs.
                   - "patience": Early stopping patience.
                   - "min_delta": Minimum change in accuracy to qualify as improvement.
    :return: A dictionary containing the final test loss and accuracy after training.
    """
    best_accuracy, best_state, best_model = 0.0, None, None
    patience_counter = 0

    optimiser = optax.adamw(
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    opt_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))

    for epoch in range(config["epochs"]):
        key, subkey = jr.split(key)
        model, state, opt_state, _ = train_epoch(
            model,
            state,
            optimiser,
            opt_state,
            train_set.features,
            train_set.labels,
            config["batch_size"],
            subkey,
        )
        test_metrics = evaluate(
            model,
            state,
            test_set.features,
            test_set.labels,
            config["batch_size"],
        )

        if test_metrics["accuracy"] > best_accuracy + config["min_delta"]:
            best_accuracy = test_metrics["accuracy"]
            best_state = state
            best_model = model
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    final_state = best_state or state
    final_model = best_model or model
    final_metrics = evaluate(
        final_model,
        final_state,
        test_set.features,
        test_set.labels,
        config["batch_size"],
    )
    print(
        f"Final Test Loss: {final_metrics['loss']:.4f},"
        f" Final Test Accuracy: {final_metrics['accuracy']:.4f}"
    )

    return {
        "final_test_loss": final_metrics["loss"],
        "final_test_accuracy": final_metrics["accuracy"],
    }


def density_preserving_umap(x: jnp.ndarray, n_components: int = 16) -> jnp.ndarray:
    """
    Perform Density-Preserving UMAP to reduce dimensionality.

    :param x: The input data matrix of shape (n_samples, n_features).
    :param n_components: The number of components to return.
    :return: The projected data of shape (n_samples, n_components).
    """
    # Convert jax array to numpy array for UMAP compatibility
    x_np = np.array(x)

    # Initialize UMAP with density-preserving option
    umap_model = umap.UMAP(
        densmap=True, n_components=n_components, random_state=0, n_jobs=1
    )

    # Fit and transform the data
    x_umap = umap_model.fit_transform(x_np)

    # Convert the result back to jax array (optional)
    return jnp.array(x_umap)


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


def train_model(
    data_bundle: dict[str, jnp.ndarray],
    key: KeyArrayLike,
    config: dict[str, int | float],
) -> dict[str, jnp.ndarray]:
    """
    Train the model and return the results.

    :param data_bundle: A dictionary containing the following keys:
                        - "data": Training input data.
                        - "targets": Training labels.
                        - "test_data": Test input data.
                        - "test_targets": Test labels.
    :param key: Random number generator key for model initialisation and dropout.
    :param config: A dictionary of training configuration parameters, including:
                   - "learning_rate": Learning rate for the optimiser.
                   - "weight_decay": Weight decay for the optimiser.
                   - "batch_size": Number of samples per training batch.
                   - "epochs": Total number of training epochs.
                   - "patience": Early stopping patience.
                   - "min_delta": Minimum change in accuracy to qualify as improvement.
    :return: A dictionary containing the final test loss and accuracy after training.
    """
    model, state = eqx.nn.make_with_state(MLP)(  # pylint: disable=assignment-from-no-return
        key, 784, hidden_size=64
    )

    # Access the values from the data_bundle dictionary
    data = data_bundle["data"]
    targets = data_bundle["targets"]
    test_data = data_bundle["test_data"]
    test_targets = data_bundle["test_targets"]

    result = train_and_evaluate(
        DataSet(data, targets),
        DataSet(test_data, test_targets),
        model,
        state,
        key,
        config,
    )

    return result


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
    file_name = "mnist_benchmark_results.json"
    with open(os.path.join(base_dir, file_name), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Data has been saved to {file_name}")


# pylint: disable=too-many-locals
def main() -> None:
    """
    Perform the benchmark for multiple solvers, coreset sizes, and random seeds.

    The function follows these steps:
    1. Prepare and load the MNIST datasets (training and test).
    2. Perform dimensionality reduction on the training data using UMAP.
    3. Initialise solvers for data reduction.
    4. For each solver and coreset size, reduce the dataset and train the model
       on the reduced set.
    5. Train the model and evaluate its performance on the test set.
    6. Save the results, which include test accuracy for each solver and coreset size.
    """
    train_data_jax, train_targets_jax, test_data_jax, test_targets_jax = (
        prepare_datasets()
    )
    train_data_umap = Data(density_preserving_umap(train_data_jax))

    all_results = {}

    config = {
        "epochs": 100,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "patience": 5,
        "min_delta": 0.001,
    }

    # Run the experiment with 5 different random keys
    # pylint: disable=duplicate-code
    for i in range(5):
        print(f"Run {i + 1} of 5:")
        key = jax.random.PRNGKey(i)
        solver_factories = initialise_solvers(
            train_data_umap, key, cpp_oversampling_factor=7, leaf_size=15_000
        )
        for solver_name, solver_creator in solver_factories.items():
            for size in [25, 50, 100, 500, 1_000, 5_000]:
                solver = solver_creator(size)
                start_time = time.perf_counter()
                # pylint: enable=duplicate-code
                coreset, _ = eqx.filter_jit(solver.reduce)(train_data_umap)

                coreset_indices = coreset.unweighted_indices

                train_data_coreset = train_data_jax[coreset_indices]
                train_targets_coreset = train_targets_jax[coreset_indices]

                # Adjust batch size based on size
                config["batch_size"] = min(len(coreset_indices) // 2, 64)

                data_bundle = {
                    "data": train_data_coreset,
                    "targets": train_targets_coreset,
                    "test_data": test_data_jax,
                    "test_targets": test_targets_jax,
                }
                # Train the model and get the evaluation metrics for this run
                run_metrics = train_model(data_bundle, key, config)

                # Ensure that there is a dictionary for this solver
                # If not, initialise with an empty dictionary
                if solver_name not in all_results:
                    all_results[solver_name] = {}

                # Populate the dictionary created above with coreset_size as keys
                # The values themselves will be dictionaries, so initialise with an
                # empty dictionary
                if size not in all_results[solver_name]:
                    all_results[solver_name][size] = {}

                # Store accuracy result in nested structure
                all_results[solver_name][size][i] = {
                    "accuracy": float(run_metrics["final_test_accuracy"]),
                    "time_taken": time.perf_counter() - start_time,
                }

    save_results(all_results)


if __name__ == "__main__":
    main()
