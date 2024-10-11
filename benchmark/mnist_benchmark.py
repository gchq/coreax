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
Performance of different coreset algorithms on the MNIST dataset.

We generate coresubsets of the MNIST training dataset using different
coreset algorithms and train a MLP classifier and compare the accuracy
on the testing dataset.
"""

import json
from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torchvision
from flax import linen as nn
from flax.training import train_state
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
def convert_to_jax_arrays(pytorch_data: Dataset) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert a PyTorch dataset to JAX arrays.

    :param pytorch_data: PyTorch dataset to convert
    :return: Tuple of JAX arrays (data, targets)
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
    """Compute cross-entropy loss."""
    return jnp.mean(
        optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, num_classes=10))
    )


def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Compute loss and accuracy metrics."""
    loss = cross_entropy_loss(logits, labels)
    _accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {"loss": loss, "accuracy": _accuracy}


class MLP(nn.Module):
    """Multi-layer perceptron with optional batch normalization and dropout."""

    hidden_size: int
    output_size: int = 10
    use_batchnorm: bool = True
    dropout_rate: float = 0.5

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Forward pass of the MLP."""
        x = nn.Dense(self.hidden_size)(x)
        if self.use_batchnorm:
            x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(self.output_size)(x)
        return x


class TrainState(train_state.TrainState):
    """Custom train state with batch statistics and dropout RNG."""

    batch_stats: Optional[Dict[str, jnp.ndarray]] = None
    dropout_rng: Optional[jnp.ndarray] = None


class Metrics(NamedTuple):
    """Represents evaluation metrics."""

    loss: float
    accuracy: float


def create_train_state(
    rng: jnp.ndarray, _model: nn.Module, learning_rate: float, weight_decay: float
) -> TrainState:
    """Create and initialise the train state."""
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
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Perform a single training step."""
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_fn(params):
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
) -> Dict[str, jnp.ndarray]:
    """Perform a single evaluation step."""
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
) -> Tuple[TrainState, Dict[str, float]]:
    """Train for one epoch and return updated state and metrics."""
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
) -> Dict[str, float]:
    """Evaluate the model on given data and return metrics."""
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
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Train and evaluate the model with early stopping."""
    state = create_train_state(
        rng, _model, config["learning_rate"], config["weight_decay"]
    )
    best_accuracy, best_state = 0.0, None
    patience_counter = 0

    for epoch in range(config["epochs"]):
        state, train_metrics = train_epoch(
            state, train_set.features, train_set.labels, config["batch_size"]
        )
        test_metrics = evaluate(
            state, test_set.features, test_set.labels, config["batch_size"]
        )

        if epoch % 8 == 0:
            print(
                f"Epoch {epoch}, Train Loss: {train_metrics['loss']:.4f},"
                f" Train Accuracy: {train_metrics['accuracy']:.4f}"
            )
            print(
                f"Epoch {epoch}, Test Loss: {test_metrics['loss']:.4f},"
                f" Test Accuracy: {test_metrics['accuracy']:.4f}"
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


def pca(x: jnp.ndarray, n_components: int = 16) -> jnp.ndarray:
    """
    Perform Principal Component Analysis (PCA) on the dataset x.

    This function computes the principal components of the input data matrix
    and returns the projected data.

    :param x: The input data matrix of shape (n_samples, n_features)
    :param n_components: The number of principal components to return
    :return: The projected data of shape (n_samples, n_components)
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


def prepare_datasets() -> tuple:
    """Prepare and return training and test datasets."""
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

    return (train_data_jax, train_targets_jax, test_data_jax, test_targets_jax)


def initialise_solvers(
    train_data_pca: jnp.ndarray, key: jax.random.PRNGKey, small_dataset: jnp.ndarray
) -> list:
    """
    Initialise and return a list of solvers for various coreset algorithms.

    Set up solvers for Kernel Herding, Stein Thinning, Random Sampling, and Randomised
    Cholesky methods. Each solver has different parameter requirements.
    Some solvers can utilise MapReduce, while others cannot,
    and some require specific kernels.
     This setup allows them to be called by passing only the coreset size,
    enabling easy integration in a loop for benchmarking.

    :param train_data_pca: The PCA-transformed training data used for
                           length scale estimation for ``SquareExponentialKernel``.
    :param key: The random key for initialising random solvers.
    :param small_dataset: A subset of data used for score function
                          matching in the Stein solver.

    :return: A list of solvers functions for different coreset algorithms.
    """
    # Set up kernel using median heuristic
    num_samples_length_scale = min(300, 1000)
    random_seed = 45
    generator = np.random.default_rng(random_seed)
    idx = generator.choice(300, num_samples_length_scale, replace=False)
    length_scale = median_heuristic(train_data_pca[idx])
    kernel = SquaredExponentialKernel(length_scale=length_scale)

    def _get_herding_solver(_size: int) -> Tuple[str, MapReduce]:
        """
        Set up KernelHerding to use ``MapReduce``.

        Create a KernelHerding solver with the specified size and return
        it along with a MapReduce object for reducing a large dataset like
        MNIST dataset.

        :param _size: The size of the coreset to be generated.
        :return: A tuple containing the solver name and the MapReduce solver.
        """
        herding_solver = KernelHerding(_size, kernel, block_size=64)
        return "KernelHerding", MapReduce(herding_solver, leaf_size=2 * _size)

    def _get_stein_solver(_size: int) -> Tuple[str, MapReduce]:
        """
        Set up Stein Thinning to use ``MapReduce``.

        Create a SteinThinning solver with the specified  coreset size,
        using ``KernelDensityMatching`` score function for matching on
        a subset of the dataset.

        :param _size: The size of the coreset to be generated.
        :return: A tuple containing the solver name and the MapReduce solver.
        """
        score_function = KernelDensityMatching(length_scale=length_scale).match(
            small_dataset
        )
        stein_kernel = SteinKernel(kernel, score_function)
        stein_solver = SteinThinning(
            coreset_size=_size, kernel=stein_kernel, block_size=64
        )
        return "SteinThinning", MapReduce(stein_solver, leaf_size=2 * _size)

    def _get_random_solver(_size: int) -> Tuple[str, MapReduce]:
        """
        Set up Random Sampling to generate a coreset.

        :param _size: The size of the coreset to be generated.
        :return: A tuple containing the solver name and the RandomSample solver.
        """
        random_solver = RandomSample(_size, key)
        return "RandomSample", random_solver

    def _get_rp_solver(_size: int) -> Tuple[str, MapReduce]:
        """
        Set up Randomised Cholesky solver.

        :param _size: The size of the coreset to be generated.
        :return: A tuple containing the solver name and the RPCholesky solver.
        """
        rp_solver = RPCholesky(coreset_size=_size, kernel=kernel, random_key=key)
        return "RPCholesky", rp_solver

    return [_get_random_solver, _get_rp_solver, _get_herding_solver, _get_stein_solver]


def train_model(data_bundle: dict, key, config) -> dict:
    """Train the model and return the results."""
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


def train_and_save_results(results: list) -> None:
    """Save results to JSON."""
    data_by_solver = {}
    for solver, coreset_size, accuracy in results:
        if solver not in data_by_solver:
            data_by_solver[solver] = {"coreset_size": [], "accuracy": []}
        data_by_solver[solver]["coreset_size"].append(coreset_size)
        data_by_solver[solver]["accuracy"].append(float(accuracy))

    # Dump data to JSON
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(data_by_solver, f, indent=4)

    print("Data has been saved to 'benchmark_results.json'")


def main() -> None:
    """Perform the benchmark."""
    key = jax.random.PRNGKey(0)

    # Prepare datasets
    (train_data_jax, train_targets_jax, test_data_jax, test_targets_jax) = (
        prepare_datasets()
    )

    # Perform PCA
    train_data_pca = pca(train_data_jax)

    results = []

    # Generate small dataset for ScoreMatching for SteinKernel
    small_dataset = train_data_pca[
        jax.random.choice(key, train_data_pca.shape[0], shape=(1000,), replace=False)
    ]

    # Initialise solvers
    solvers = initialise_solvers(train_data_pca, key, small_dataset)

    config = {
        "epochs": 100,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "patience": 5,
        "min_delta": 0.001,
    }

    for getter in solvers:
        for size in [25, 26, 50, 100]:
            name, solver = getter(size)
            print("This much works")
            subset, _ = solver.reduce(Data(train_data_pca))
            print(name, subset)

            indices = subset.nodes.data

            data = train_data_jax[indices]
            targets = train_targets_jax[indices]

            # Adjust batch size based on size
            config["batch_size"] = min(len(indices) // 2, 64)

            data_bundle = {
                "data": data,
                "targets": targets,
                "test_data": test_data_jax,
                "test_targets": test_targets_jax,
            }

            result = train_model(data_bundle, key, config)
            print(result)
            results.append((name, size, result["final_test_accuracy"]))

    # Save results to JSON
    train_and_save_results(results)


if __name__ == "__main__":
    main()
