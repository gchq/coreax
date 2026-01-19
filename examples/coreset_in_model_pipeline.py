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
Integration test showing how coresets can be included in a machine learning pipeline.

A coreset is generated using Kernel Herding and a Kernel Ridge Regression model is fit
using it. It is then compared in terms of time and test mean squared error to the
full dataset and a random subset.
"""

from pathlib import Path
from time import time

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from coreax.data import Data
from coreax.kernels import SquaredExponentialKernel, median_heuristic
from coreax.solvers import KernelHerding, RandomSample


# Examples are written to be easy to read, copy and paste by users, so we ignore the
# pylint warnings raised that go against this approach
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=duplicate-code
def main(
    out_path: Path | None = None,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Run machine learning pipeline example.

    A 1-dimensional set of features is generated from the standard Gaussian
    distribution a set of paired responses is generated with a non-linear relationship
    to the features. A coreset is then generated using Kernel Herding and a Kernel Ridge
    Regression model is fit using it. It is then compared in terms of time to fit the
    model and test mean squared error to the full dataset and a random subset.

    :param out_path: Path to save output to, if not :data:`None`, assumed relative to
        this module file unless an absolute path is given
    :return: ((full data test MSE, fitting time), (coreset test MSE, fitting time),
        (random test MSE, fitting time))
    """
    # Create some data. Here we'll use 25,000 points in 1D from a uniform distribution
    seed = 1_234
    num_data_pairs = 25_000
    feature_key, error_key = jr.split(jr.key(seed), 2)
    x = jr.normal(feature_key, shape=(num_data_pairs, 1))

    def _generating_func(x: Array) -> Array:
        """Encode non-linear relationship between features and response."""
        return (
            x
            + x**2
            + 5 * jnp.sin(x)
            + 100 * jnp.exp(-((x - 3) ** 2))
            + 100 * jnp.exp(-((x + 2) ** 2))
        )

    y = _generating_func(x) + jr.normal(error_key, shape=(num_data_pairs, 1))
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.1, random_state=seed)

    # Scale the features and responses individually
    feature_scaler = StandardScaler().fit(x)
    x = jnp.asarray(feature_scaler.transform(x))
    x_test = jnp.asarray(feature_scaler.transform(x_test))

    response_scaler = StandardScaler().fit(y)
    y = jnp.asarray(response_scaler.transform(y))
    y_test = jnp.asarray(response_scaler.transform(y_test))

    # Request 200 coreset points
    coreset_size = 200

    # Setup the original data object
    data = Data(x)

    # Set the bandwidth parameter of the kernel using a median heuristic derived from at
    # most 1000 random samples in the data.
    num_samples_length_scale = min(num_data_pairs, 1_000)
    generator = np.random.default_rng(seed)
    idx = generator.choice(num_data_pairs, num_samples_length_scale, replace=False)
    length_scale = median_heuristic(x[idx]).item()
    kernel = SquaredExponentialKernel(length_scale=length_scale)

    print("Estimating kernel ridge regression model with full data...")
    full_data_fit_time = time()
    full_data_model = KernelRidge(
        kernel="rbf",
        gamma=length_scale,
        # Pyright is wrong here - this can be a float, not just an int
        alpha=1e-1,  # pyright: ignore[reportArgumentType]
    ).fit(x, y)
    full_data_fit_time = time() - full_data_fit_time
    full_data_rmse = jnp.sqrt(
        ((y_test - full_data_model.predict(x_test)) ** 2).sum() / num_data_pairs
    )

    print("Computing herding coreset...")
    # Compute a coreset using kernel herding with a squared exponential kernel.
    coreset_solver = KernelHerding(coreset_size, kernel)

    coreset_build_time = time()
    coreset, _ = eqx.filter_jit(coreset_solver.reduce)(data)
    coreset_build_time = time() - coreset_build_time

    print("Estimating kernel ridge regression model with coreset...")
    coreset_indices = coreset.unweighted_indices
    coreset_fit_time = time()
    coreset_model = KernelRidge(
        kernel="rbf",
        gamma=float(median_heuristic(x[coreset_indices])),
        # Pyright is wrong here - this can be a float, not just an int
        alpha=1e-1,  # pyright: ignore[reportArgumentType]
    ).fit(x[coreset_indices], y[coreset_indices])
    coreset_fit_time = time() - coreset_fit_time
    coreset_overall_time = coreset_build_time + coreset_fit_time
    coreset_rmse = jnp.sqrt(
        ((y_test - coreset_model.predict(x_test)) ** 2).sum() / num_data_pairs
    )

    print("Estimating kernel ridge regression model with random sample of data...")
    solver = RandomSample(coreset_size, jr.key(seed))
    random_set, _ = eqx.filter_jit(solver.reduce)(data)
    random_indices = random_set.unweighted_indices
    random_fit_time = time()
    random_model = KernelRidge(
        kernel="rbf",
        gamma=float(median_heuristic(x[random_indices])),
        # Pyright is wrong here - this can be a float, not just an int
        alpha=1e-1,  # pyright: ignore[reportArgumentType]
    ).fit(x[random_indices], y[random_indices])
    random_fit_time = time() - random_fit_time
    random_rmse = jnp.sqrt(
        ((y_test - random_model.predict(x_test)) ** 2).sum() / num_data_pairs
    )

    # Produce some scatter plots
    x_plot = jnp.linspace(x.min(), x.max(), 100).reshape(-1, 1)

    # Plot of full data model
    plt.scatter(x, y, s=20, color="black", alpha=0.25, label="Data")
    plt.plot(
        x_plot,
        full_data_model.predict(x_plot),
        color="red",
        linewidth=3,
        label="Model estimate",
    )
    plt.legend()
    plt.title("Model trained on the full dataset")
    plt.show()

    plt.scatter(x, y, s=20, color="black", alpha=0.25, label="Data")
    plt.scatter(
        x[coreset_indices],
        y[coreset_indices],
        s=100,
        color="yellow",
        ec="black",
        label="Coreset",
    )

    # Plot of coreset model
    plt.plot(
        x_plot,
        coreset_model.predict(x_plot),
        color="red",
        linewidth=3,
        label="Model estimate",
    )
    plt.legend()
    plt.title("Model trained on the coreset")
    plt.show()

    # Plot of random subset model
    plt.scatter(x, y, s=20, color="black", alpha=0.25, label="Data")
    plt.scatter(
        x[random_indices],
        y[random_indices],
        s=100,
        color="yellow",
        ec="black",
        label="Random sample",
    )

    plt.plot(
        x_plot,
        random_model.predict(x_plot),
        color="blue",
        linewidth=3,
        label="Model estimate",
    )
    plt.legend()
    plt.title("Model trained on the random sample")
    plt.show()

    # Plot comparing each model on one graph
    plt.scatter(x, y, s=20, color="black", alpha=0.25, label="Data")
    plt.plot(
        x_plot,
        full_data_model.predict(x_plot),
        color="green",
        linewidth=3,
        label=f"Full estimate, RMSE = {round(float(full_data_rmse), 3)},"
        + f" Time = {round(float(full_data_fit_time), 3)}s",
    )
    plt.plot(
        x_plot,
        coreset_model.predict(x_plot),
        color="red",
        linewidth=3,
        label=f"Coreset estimate, RMSE = {round(float(coreset_rmse), 3)},"
        + f" Time = {round(float(coreset_overall_time), 3)}s",
    )
    plt.plot(
        x_plot,
        random_model.predict(x_plot),
        color="blue",
        linewidth=3,
        label=f"Random estimate, RMSE = {round(float(random_rmse), 3)},"
        + f" Time = {round(float(random_fit_time), 3)}s",
    )
    plt.legend()
    plt.title("All models compared")
    plt.show()

    if out_path is not None:
        if not out_path.is_absolute():
            out_path = Path(__file__).parent.joinpath(out_path)
        plt.savefig(out_path)

    return (
        (float(full_data_rmse), float(full_data_fit_time)),
        (float(coreset_rmse), float(coreset_overall_time)),
        (float(random_rmse), float(random_fit_time)),
    )


# pylint: enable=too-many-locals
# pylint: enable=duplicate-code
# pylint: enable=too-many-statements

if __name__ == "__main__":
    main()
