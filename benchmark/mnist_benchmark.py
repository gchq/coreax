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
from typing import Any, NamedTuple, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torchvision
import umap
from flax import linen as nn
from flax.training import train_state
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from coreax import Data
from coreax.benchmark_util import get_solver_name, initialise_solvers


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
    _accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {"loss": loss, "accuracy": _accuracy}


class MLP(nn.Module):
    """
    Multi-layer perceptron with optional batch normalisation and dropout.

    :param hidden_size: Number of units in the hidden layer.
    :param output_size: Number of output units.
    :param use_batchnorm: Whether to apply batch norm.
    :param dropout_rate: Dropout rate to use during training.
    """

    hidden_size: int
    output_size: int = 10
    use_batchnorm: bool = True
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Forward pass of the MLP.

        :param x: Input data.
        :param training: Whether the model is in training mode (default is True).
        :return: Output logits of the network.
        """
        x = nn.Dense(self.hidden_size)(x)
        if training:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=False)(x)
        if self.use_batchnorm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_size)(x)
        return x


class TrainState(train_state.TrainState):
    """Custom train state with batch statistics and dropout RNG."""

    batch_stats: Optional[dict[str, jnp.ndarray]] = None
    dropout_rng: Optional[jnp.ndarray] = None


class Metrics(NamedTuple):
    """Represents evaluation metrics."""

    loss: float
    accuracy: float


def create_train_state(
    rng: jnp.ndarray, _model: nn.Module, learning_rate: float, weight_decay: float
) -> TrainState:
    """
    Create and initialise the train state.

    :param rng: Random number generator key.
    :param _model: The model to initialise.
    :param learning_rate: Learning rate for the optimiser.
    :param weight_decay: Weight decay for the optimiser.
    :return: The initialised TrainState.
    """
    dropout_rng, params_rng = jax.random.split(rng)
    params = _model.init(
        {"params": params_rng, "dropout": dropout_rng},
        jnp.ones([1, 784]),
        training=False,
    )
    tx = optax.adamw(learning_rate, weight_decay=weight_decay)
    return TrainState.create(
        apply_fn=_model.apply,
        params=params["params"],
        tx=tx,
        batch_stats=params["batch_stats"],
        dropout_rng=dropout_rng,
    )


@jax.jit
def train_step(
    state: TrainState, batch_data: jnp.ndarray, batch_labels: jnp.ndarray
) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    """
    Perform a single training step.

    :param state: The current state of the model and optimiser.
    :param batch_data: Batch of input data.
    :param batch_labels: Batch of ground truth labels.
    :return: Updated TrainState and a dictionary of metrics (loss and accuracy).
    """
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_fn(
        params: dict[str, jnp.ndarray],
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, dict[str, jnp.ndarray]]]:
        """
        Compute the cross-entropy loss for the given batch.

        :param params: Model parameters.
        :return: Tuple containing the loss and a tuple of (logits, updated model state).
        """
        variables = {"params": params, "batch_stats": state.batch_stats}
        logits, new_model_state = state.apply_fn(
            variables,
            batch_data,
            training=True,
            mutable=["batch_stats"],
            rngs={"dropout": dropout_rng},
        )
        loss = cross_entropy_loss(logits, batch_labels)
        return loss, (logits, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (logits, new_model_state)), grads = grad_fn(state.params)
    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state["batch_stats"],
        dropout_rng=new_dropout_rng,
    )
    metrics = compute_metrics(logits, batch_labels)
    return state, metrics


@jax.jit
def eval_step(
    state: TrainState, batch_data: jnp.ndarray, batch_labels: jnp.ndarray
) -> dict[str, jnp.ndarray]:
    """
    Perform a single evaluation step.

    :param state: The current state of the model.
    :param batch_data: Batch of input data.
    :param batch_labels: Batch of ground truth labels.
    :return: A dictionary of evaluation metrics (loss and accuracy).
    """
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    logits = state.apply_fn(
        variables, batch_data, training=False, rngs={"dropout": state.dropout_rng}
    )
    return compute_metrics(logits, batch_labels)


def train_epoch(
    state: TrainState,
    train_data: jnp.ndarray,
    train_labels: jnp.ndarray,
    batch_size: int,
) -> tuple[TrainState, dict[str, float]]:
    """
    Train for one epoch and return updated state and metrics.

    :param state: The current state of the model and optimiser.
    :param train_data: Training input data.
    :param train_labels: Training labels.
    :param batch_size: Size of each training batch.
    :return: Updated TrainState and a dictionary containing 'loss' and 'accuracy'.
    """
    num_batches = train_data.shape[0] // batch_size
    total_loss, total_accuracy = 0.0, 0.0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_data = train_data[start_idx:end_idx]
        batch_labels = train_labels[start_idx:end_idx]
        state, metrics = train_step(state, batch_data, batch_labels)
        total_loss += metrics["loss"]
        total_accuracy += metrics["accuracy"]

    return state, {
        "loss": total_loss / num_batches,
        "accuracy": total_accuracy / num_batches,
    }


def evaluate(
    state: TrainState, _data: jnp.ndarray, labels: jnp.ndarray, batch_size: int
) -> dict[str, float]:
    """
    Evaluate the model on given data and return metrics.

    :param state: The current state of the model.
    :param _data: Input data for evaluation.
    :param labels: Ground truth labels for evaluation.
    :param batch_size: Size of each evaluation batch.
    :return: A dictionary containing 'loss' and 'accuracy' metrics.
    """
    num_batches = _data.shape[0] // batch_size
    total_loss, total_accuracy = 0.0, 0.0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        batch_data = _data[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        metrics = eval_step(state, batch_data, batch_labels)
        total_loss += metrics["loss"]
        total_accuracy += metrics["accuracy"]

    return {"loss": total_loss / num_batches, "accuracy": total_accuracy / num_batches}


class DataSet(NamedTuple):
    """Represents a dataset with features and labels."""

    features: jnp.ndarray
    labels: jnp.ndarray


def train_and_evaluate(
    train_set: DataSet,
    test_set: DataSet,
    _model: nn.Module,
    rng: jnp.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    """
    Train and evaluate the model with early stopping.

    :param train_set: The training dataset containing features and labels.
    :param test_set: The test dataset containing features and labels.
    :param _model: The model to be trained.
    :param rng: Random number generator key for parameter initialisation and dropout.
    :param config: A dictionary of training configuration parameters, including:
                   - "learning_rate": Learning rate for the optimiser.
                   - "weight_decay": Weight decay for the optimiser.
                   - "batch_size": Number of samples per training batch.
                   - "epochs": Total number of training epochs.
                   - "patience": Early stopping patience.
                   - "min_delta": Minimum change in accuracy to qualify as improvement.
    :return: A dictionary containing the final test loss and accuracy after training.
    """
    state = create_train_state(
        rng, _model, config["learning_rate"], config["weight_decay"]
    )
    best_accuracy, best_state = 0.0, None
    patience_counter = 0

    for epoch in range(config["epochs"]):
        state, _ = train_epoch(
            state, train_set.features, train_set.labels, config["batch_size"]
        )
        test_metrics = evaluate(
            state, test_set.features, test_set.labels, config["batch_size"]
        )

        if test_metrics["accuracy"] > best_accuracy + config["min_delta"]:
            best_accuracy = test_metrics["accuracy"]
            best_state = state
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    final_state = best_state or state
    final_metrics = evaluate(
        final_state, test_set.features, test_set.labels, config["batch_size"]
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
    umap_model = umap.UMAP(densmap=True, n_components=n_components)

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
    key: jax.random.PRNGKey,
    config: dict[str, Union[int, float]],
) -> dict[str, float]:
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
    model = MLP(hidden_size=64)

    # Access the values from the data_bundle dictionary
    data = data_bundle["data"]
    targets = data_bundle["targets"]
    test_data = data_bundle["test_data"]
    test_targets = data_bundle["test_targets"]

    result = train_and_evaluate(
        DataSet(data, targets),
        DataSet(test_data, test_targets),
        model,
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
    (train_data_jax, train_targets_jax, test_data_jax, test_targets_jax) = (
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
        solver_factories = initialise_solvers(train_data_umap, key)
        for solver_creator in solver_factories:
            for size in [25, 50, 100, 500, 1_000, 5_000]:
                solver = solver_creator(size)
                solver_name = get_solver_name(solver_creator)
                start_time = time.perf_counter()
                # pylint: enable=duplicate-code
                coreset, _ = eqx.filter_jit(solver.reduce)(train_data_umap)

                coreset_indices = coreset.nodes.data

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
