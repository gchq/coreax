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
Example coreset generation using probabilistic KernelHerding iteratively.

Use `iterative_refine_experiment` to run `KernelHerding.refine` iteratively for a
given number of iterations, `N`. Optionally, you can supply `t_schedule`, an array of
length `N`, in which case the iteration `i` is run with the probabilistic version of
Kernel Herding with the temperature of `t_schedule[i]`.

Generally, standard (deterministic) Kernel Herding converges very quickly so that
`refine` keeps producing the same coreset after a few iterations (i.e., it finds a
fixed point). Introducing probabilistic selection perturbs the procedure and helps
find new coresets. The temperature parameter can help balance this trade-off: high
values tend to produce random coresets, while low values approximate standard Kernel
Herding and converge faster.
"""

import time
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Shaped
from sklearn.datasets import make_blobs

from coreax import MMD, Coresubset, Data, SquaredExponentialKernel
from coreax.kernels import median_heuristic
from coreax.solvers import KernelHerding
from coreax.solvers.coresubset import _initial_coresubset  # noqa: PLC2701


def make_data(
    num_data_points: int = 10_000,
    num_features: int = 2,
    num_cluster_centers: int = 10,
    random_seed: int = 123,
) -> Data:
    """
    Create dataset using sklearn.datasets.make_blobs.

    :param num_data_points: The total number of data points to generate
    :param num_features: The number of features (dimensions) for each data point
    :param num_cluster_centers: The number of cluster centers to generate
    :param random_seed: Random seed
    :return: A :class:`Data` instance containing the generated data
    """
    x, *_ = make_blobs(
        n_samples=num_data_points,
        n_features=num_features,
        centers=num_cluster_centers,
        random_state=random_seed,
        return_centers=True,
        cluster_std=2,
    )
    return Data(jnp.asarray(x))


def get_kernels(
    data: Data, random_seed: int = 0
) -> tuple[SquaredExponentialKernel, SquaredExponentialKernel]:
    """
    Get a SquaredExponentialKernel based on the data using the median heuristic.

    :param data: A :class:`Data` instance
    :param random_seed: Random seed
    :return: A tuple `(kernel, mmd_kernel)` of :class:`SquaredExponentialKernel`
    """
    num_data_points = len(data)
    num_samples_length_scale = min(num_data_points, 1_000)
    generator = np.random.default_rng(random_seed)
    idx = generator.choice(num_data_points, num_samples_length_scale, replace=False)
    length_scale = median_heuristic(data.data[idx])
    return (
        SquaredExponentialKernel(length_scale=length_scale),
        SquaredExponentialKernel(
            length_scale=length_scale,
            output_scale=1.0 / (length_scale * jnp.sqrt(2.0 * jnp.pi)),
        ),
    )


def iterative_refine_experiment(
    data: Data,
    coreset_size: int,
    n_iter: int,
    t_schedule: Optional[Shaped[Array, " {n_iter}"]] = None,
    seed: int = 0,
) -> tuple[Array, Coresubset]:
    """
    Perform an experiment by iteratively refining the coreset using Kernel Herding.

    :param data: A :class:`Data` instance
    :param coreset_size: The desired size of the coreset
    :param n_iter: The number of refine iterations to perform
    :param t_schedule: A :class:`Array` of length `n_iter`, where `t_schedule[i]` is the
        temperature parameter used for iteration i. If None, standard Kernel Herding is
        used.
    :param seed: Random seed
    """
    random_key = jax.random.key(seed)
    kernel, mmd_kernel = get_kernels(data)
    mmd_metric = MMD(mmd_kernel)
    probabilistic = True
    if t_schedule is None:
        t_schedule = jnp.ones(n_iter)
        probabilistic = False

    mmd_data = jnp.zeros(n_iter)  # store experiment data
    initial_coreset = _initial_coresubset(0, coreset_size, data)

    def run_experiment(
        i: int, state: tuple[Coresubset, Array]
    ) -> tuple[Coresubset, Array]:
        """Perform a single iteration of refining the coreset."""
        coreset, mmd_data = state
        key = jax.random.fold_in(random_key, i)

        solver = KernelHerding(
            coreset_size=coreset_size,
            kernel=kernel,
            probabilistic=probabilistic,
            temperature=t_schedule[i],
            random_key=key,
        )
        coreset, _ = solver.refine(coreset)
        mmd_data = mmd_data.at[i].set(coreset.compute_metric(mmd_metric))
        return coreset, mmd_data

    coreset, mmd_data = jax.lax.fori_loop(
        0, n_iter, run_experiment, (initial_coreset, mmd_data)
    )
    return mmd_data, coreset


def visualise_results(mmd_data_prob: dict[str, Array], mmd_data_base: Array) -> None:
    """
    Visualise the results of the experiment.

    :param mmd_data_prob: A dictionary of labelled probabilistic experiment runs
    :param mmd_data_base: An array containing data for a standard run
    """
    baseline = mmd_data_base[0].item()
    plt.plot(mmd_data_base, label="Standard KH")
    plt.axhline(baseline, c="k", ls="--", label="Standard KH reduce")

    for label, mmd_data in mmd_data_prob.items():
        plt.plot(mmd_data, label=f"Probabilistic: {label}")

    plt.grid(ls=":")
    plt.xlabel("Refinement iteration")
    plt.ylabel("MMD")
    plt.legend()
    plt.ylim([0, baseline * 3])  # zoom in on the relevant scale
    plt.title("Iterative refinement with probabilistic selection")
    plt.show()


def main(seed_exp: int) -> None:
    """
    Run standard and probabilistic Kernel Herding iteratively and visualise the results.

    The steps are as follows:
        1. Generate data using :meth:`make_data`.
        2. Choose the number of iterations and the desired coreset size.
        3. Define temperature schedules (arrays for size `n_iter`) in a dictionary.
        4. Run the experiment for standard KH and for each defined temperature schedule.
        5. Plot the resulting data.

    :param seed_exp: Seed for the probabilistic experiment
    """
    start = time.time()

    dataset = make_data(
        # Data parameters
        num_data_points=1000,
        num_features=4,
        num_cluster_centers=5,
        random_seed=412,
    )

    # Number of iterations and desired coreset size
    n_iter = 100
    coreset_size = 100

    # Temperature schedules which define the temperature parameter for each iteration -
    # feel free to experiment!
    t_schedules = {
        "const0.001": jnp.ones(n_iter) * 0.001,
        "const0.0001": jnp.ones(n_iter) * 0.0001,
        "inverse_cubic": 1 / jnp.linspace(1, 100, n_iter) ** 3,
    }

    # Standard refinement as a baseline
    results_base, _ = iterative_refine_experiment(dataset, coreset_size, n_iter)

    # Probabilistic refinement
    results_prob = {}
    for schedule_key, schedule in t_schedules.items():
        results_prob[schedule_key], _ = iterative_refine_experiment(
            dataset, coreset_size, n_iter, schedule, seed=seed_exp
        )

    visualise_results(results_prob, results_base)

    end = time.time()
    print(f"Time taken: {end - start:.6f} seconds")


if __name__ == "__main__":
    # Random seed for the experiment - change to get a different run
    SEED_EXP = 83569

    main(seed_exp=SEED_EXP)
