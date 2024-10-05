import json
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torchvision
from flax import linen as nn
from flax.training import train_state
from matplotlib import pyplot as plt
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

key = jax.random.PRNGKey(0)  # for reproducibility
n_components = 16  # for PCA

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


# Convert PyTorch dataset to JAX arrays
def convert_to_jax_arrays(dataset: Dataset) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert a PyTorch dataset to JAX arrays.

    This function takes a PyTorch dataset, loads all the data at once using a DataLoader,
    and converts it to JAX arrays. It's designed to work with datasets that can fit into memory.

    Args:
        dataset (Dataset): A PyTorch dataset to be converted.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing two JAX arrays:
            - The first array contains the data.
            - The second array contains the targets (labels).

    """
    data_loader = DataLoader(dataset, batch_size=len(dataset))
    data, targets = next(iter(data_loader))  # Load all data at once
    data_jax = jnp.array(data.numpy())       # Convert to NumPy first, then JAX array
    targets_jax = jnp.array(targets.numpy())
    return data_jax, targets_jax


# Usage remains the same
train_data_jax, train_targets_jax = convert_to_jax_arrays(train_dataset)
test_data_jax, test_targets_jax = convert_to_jax_arrays(test_dataset)


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Compute cross-entropy loss."""
    return jnp.mean(optax.softmax_cross_entropy(
        logits,
        jax.nn.one_hot(labels, num_classes=10)
    ))


def compute_metrics(
    logits: jnp.ndarray,
    labels: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """Compute loss and accuracy metrics."""
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}


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

    batch_stats: Any = None
    dropout_rng: jnp.ndarray = None


def create_train_state(
    rng: jnp.ndarray,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float
) -> TrainState:
    """Create and initialize the train state."""
    dropout_rng, params_rng = jax.random.split(rng)
    params = model.init(
        {'params': params_rng, 'dropout': dropout_rng},
        jnp.ones([1, 784]),
        training=False
    )
    tx = optax.adamw(learning_rate, weight_decay=weight_decay)
    return TrainState.create(
        apply_fn=model.apply,
        params=params['params'],
        tx=tx,
        batch_stats=params['batch_stats'],
        dropout_rng=dropout_rng
    )


@jax.jit
def train_step(
    state: TrainState,
    batch_data: jnp.ndarray,
    batch_labels: jnp.ndarray
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Perform a single training step."""
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logits, new_model_state = state.apply_fn(
            variables,
            batch_data,
            training=True,
            mutable=['batch_stats'],
            rngs={'dropout': dropout_rng}
        )
        loss = cross_entropy_loss(logits, batch_labels)
        return loss, (logits, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_model_state)), grads = grad_fn(state.params)
    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state['batch_stats'],
        dropout_rng=new_dropout_rng
    )
    metrics = compute_metrics(logits, batch_labels)
    return state, metrics


@jax.jit
def eval_step(
    state: TrainState,
    batch_data: jnp.ndarray,
    batch_labels: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """Perform a single evaluation step."""
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(
        variables,
        batch_data,
        training=False,
        rngs={'dropout': state.dropout_rng}
    )
    return compute_metrics(logits, batch_labels)


def train_and_evaluate(
    train_data_jax: jnp.ndarray,
    train_labels_jax: jnp.ndarray,
    test_data_jax: jnp.ndarray,
    test_labels_jax: jnp.ndarray,
    model: nn.Module,
    rng: jnp.ndarray,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 5,
    min_delta: float = 0.001
) -> Dict[str, float]:
    """Train and evaluate the model with early stopping."""
    state = create_train_state(rng, model, learning_rate, weight_decay)
    num_train_batches = train_data_jax.shape[0] // batch_size
    num_test_batches = test_data_jax.shape[0] // batch_size

    best_accuracy = 0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Shuffle data
        rng, input_rng = jax.random.split(rng)
        perm = jax.random.permutation(input_rng, train_data_jax.shape[0])
        train_data_shuffled = train_data_jax[perm]
        train_labels_shuffled = train_labels_jax[perm]

        # Training loop
        for batch_idx in range(num_train_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            batch_data = train_data_shuffled[start_idx:end_idx]
            batch_labels = train_labels_shuffled[start_idx:end_idx]
            state, metrics = train_step(state, batch_data, batch_labels)

        if epoch % 8 == 0:
            print(
                f"Epoch {epoch}, "
                f"Loss: {metrics['loss']:.4f}, "
                f"Accuracy: {metrics['accuracy']:.4f}"
            )

        # Evaluation loop
        total_metrics = {'loss': 0.0, 'accuracy': 0.0}
        for batch_idx in range(num_test_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            batch_data = test_data_jax[start_idx:end_idx]
            batch_labels = test_labels_jax[start_idx:end_idx]
            metrics = eval_step(state, batch_data, batch_labels)
            total_metrics['loss'] += metrics['loss']
            total_metrics['accuracy'] += metrics['accuracy']

        avg_loss = total_metrics['loss'] / num_test_batches
        avg_accuracy = total_metrics['accuracy'] / num_test_batches
        if epoch % 8 == 0:
            print(
                f"Epoch {epoch}, "
                f"Test Loss: {avg_loss:.4f}, "
                f"Test Accuracy: {avg_accuracy:.4f}"
            )

        # Early stopping logic
        if avg_accuracy > best_accuracy + min_delta:
            best_accuracy = avg_accuracy
            patience_counter = 0
            best_state = state
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # If training completed without early stopping, use the final state
    if best_state is None:
        best_state = state

    # Final evaluation using the best state
    total_metrics = {'loss': 0.0, 'accuracy': 0.0}
    for batch_idx in range(num_test_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        batch_data = test_data_jax[start_idx:end_idx]
        batch_labels = test_labels_jax[start_idx:end_idx]
        metrics = eval_step(best_state, batch_data, batch_labels)
        total_metrics['loss'] += metrics['loss']
        total_metrics['accuracy'] += metrics['accuracy']

    final_avg_loss = total_metrics['loss'] / num_test_batches
    final_avg_accuracy = total_metrics['accuracy'] / num_test_batches
    print(
        f"Final Test Loss: {final_avg_loss:.4f}, "
        f"Final Test Accuracy: {final_avg_accuracy:.4f}"
    )

    return {
        'final_test_loss': final_avg_loss,
        'final_test_accuracy': final_avg_accuracy
    }


def pca(X: jnp.ndarray, n_components: int) -> jnp.ndarray:
    """
    Perform Principal Component Analysis (PCA) on the dataset X.

    Args:
        X (jnp.ndarray): The input data matrix of shape (n_samples, n_features).
        n_components (int): The number of principal components to return.

    Returns:
        jnp.ndarray: The projected data of shape (n_samples, n_components).
    """
    # Center the data
    X_centered = X - jnp.mean(X, axis=0)

    # Compute the covariance matrix
    cov_matrix = jnp.cov(X_centered.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov_matrix)

    # Sort eigenvectors by descending eigenvalues
    idx = jnp.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Select top n_components eigenvectors
    components = eigenvectors[:, :n_components]

    # Project the data onto the new subspace
    X_pca = jnp.dot(X_centered, components)

    return X_pca


# Perform PCA
train_data_pca = pca(train_data_jax, n_components)
dataset = Data(train_data_pca)

results = []

# Set up different solvers
# Set up kernel using median heuristic
num_samples_length_scale = min(300, 1000)
random_seed = 45
generator = np.random.default_rng(random_seed)
idx = generator.choice(300, num_samples_length_scale, replace=False)
length_scale = median_heuristic(train_data_pca[idx])
kernel = SquaredExponentialKernel(length_scale=length_scale)

# Generate small dataset for ScoreMatching for SteinKernel
indices = jax.random.choice(key, train_data_pca.shape[0], shape=(1000,), replace=False)
small_dataset = train_data_pca[indices]


def _get_herding_solver(coreset_size):
    herding_solver = KernelHerding(coreset_size, kernel, block_size=64)

    return 'KernelHerding', MapReduce(herding_solver, leaf_size=2 * coreset_size)


def _get_stein_solver(coreset_size):
    score_function = KernelDensityMatching(length_scale=length_scale).match(small_dataset)
    stein_kernel = SteinKernel(kernel, score_function)
    stein_solver = SteinThinning(coreset_size=coreset_size, kernel=stein_kernel, block_size=64)
    return 'SteinThinning', MapReduce(stein_solver, leaf_size=2 * coreset_size)


def _get_random_solver(coreset_size):
    random_solver = RandomSample(coreset_size, key)
    return "RandomSample", random_solver


def _get_rp_solver(coreset_size):
    rp_solver = RPCholesky(coreset_size=coreset_size, kernel=kernel, random_key=key)
    return "RPCholesky", rp_solver


getters = [_get_stein_solver, _get_rp_solver, _get_random_solver, _get_herding_solver]
for getter in getters:
    for size in [25, 26]:
        name, solver = getter(size)
        subset, _ = solver.reduce(Data(small_dataset))
        print(name, subset)

        indices = subset.nodes.data

        data = train_data_jax[indices]
        targets = train_targets_jax[indices]

        if size <= 100:
            batch_size = 8
        else:
            batch_size = 64

        model = MLP(hidden_size=64)

        result = train_and_evaluate(data, targets, test_data_jax, test_targets_jax, model, key,
                                    epochs=100, batch_size=batch_size)
        print(result)
        results.append((name, size, result['final_test_accuracy']))




data_by_solver = {}
for solver, coreset_size, accuracy in results:
    if solver not in data_by_solver:
        data_by_solver[solver] = {'coreset_size': [], 'accuracy': []}
    data_by_solver[solver]['coreset_size'].append(coreset_size)
    data_by_solver[solver]['accuracy'].append(float(accuracy))  # Convert np.float32 to Python float

# Dump data to JSON
with open('benchmark_results.json', 'w') as f:
    json.dump(data_by_solver, f, indent=4)

print("Data has been saved to 'benchmark_results.json'")

# Visualization code (same as before)
plt.figure(figsize=(12, 6))

for algo, data in data_by_solver.items():
    plt.plot(data['coreset_size'], data['accuracy'], 'o-', label=algo)

plt.xlabel('Iteration')
plt.ylabel('Performance Metric')
plt.title('Algorithm Performance Across Iterations')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()
