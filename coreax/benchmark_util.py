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
Core functionality for setting up and managing coreset solvers.

This module provides functions to initialise solvers Kernel Thinning, Kernel Herding,
Stein Thinning, Random Sampling, and Randomised Cholesky. It also defines helper
functions for computing solver parameters and retrieving solver names.
"""

from collections.abc import Callable
from typing import Optional, Union

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from coreax import Data
from coreax.kernels import SquaredExponentialKernel, SteinKernel, median_heuristic
from coreax.score_matching import KernelDensityMatching
from coreax.solvers import (
    CompressPlusPlus,
    IterativeKernelHerding,
    KernelHerding,
    KernelThinning,
    MapReduce,
    RandomSample,
    RPCholesky,
    Solver,
    SteinThinning,
)
from coreax.util import KeyArrayLike


def calculate_delta(n: int) -> Float[Array, "1"]:
    r"""
    Calculate the delta parameter for kernel thinning.

    This function evaluates the following cases:

    1. If :math:`\\log n` is positive:
       - Further evaluates :math:`\\log (\\log n)`.
         * If this is also positive, returns :math:`\frac{1}{n \\log (\\log n)}`.
         * Otherwise, returns :math:`\frac{1}{n \\log n}`.
    2. If :math:`\\log n` is negative:
       - Returns :math:`\frac{1}{n}`.

    The recommended value is :math:`\frac{1}{n \\log (\\log n)}`, but for small
    values of :math:`n`, this may be negative or even undefined. Therefore,
    alternative values are used in such cases.

    :param n: The size of the dataset we wish to reduce.
    :return: The calculated delta value based on the described conditions.
    """
    log_n = jnp.log(n)
    if log_n > 0:
        log_log_n = jnp.log(log_n)
        if log_log_n > 0:
            return 1 / (n * log_log_n)
        return 1 / (n * log_n)
    return jnp.array(1 / n)


def initialise_solvers(  # noqa: C901
    train_data_umap: Data,
    key: KeyArrayLike,
    cpp_oversampling_factor: int,
    leaf_size: Optional[int] = None,
) -> dict[str, Callable[[int], Solver]]:
    """
    Initialise and return a list of solvers for various coreset algorithms.

    Set up solvers for Kernel Herding, Stein Thinning, Random Sampling, and Randomised
    Cholesky methods. Each solver has different parameter requirements. Some solvers
    can utilise MapReduce, while others cannot,and some require specific kernels.
    This setup allows them to be called by passing only the coreset size,
    enabling easy integration in a loop for benchmarking.

    :param train_data_umap: The UMAP-transformed training data used for
        length scale estimation for ``SquareExponentialKernel``.
    :param key: The random key for initialising random solvers.
    :param cpp_oversampling_factor: The oversampling factor for `Compress++`.
    :param leaf_size: The leaf size to be used in `MapReduce` solvers. If not provided
    (i.e., `None`), `MapReduce` solvers will not be used.
    :return: A dictionary where the keys are solver names and the values are
        corresponding solver functions for different coreset algorithms.
    """
    # Set up kernel using median heuristic
    num_data_points = len(train_data_umap)
    num_samples_length_scale = min(num_data_points, 300)
    random_seed = 45
    generator = np.random.default_rng(random_seed)
    idx = generator.choice(num_data_points, num_samples_length_scale, replace=False)
    length_scale = median_heuristic(jnp.asarray(train_data_umap[idx]))
    kernel = SquaredExponentialKernel(length_scale=length_scale)
    sqrt_kernel = kernel.get_sqrt_kernel(16)

    def _get_thinning_solver(_size: int) -> Union[KernelThinning, MapReduce]:
        """
        Set up kernel thinning solver.

        If the `leaf_size` is provided, the solver uses ``MapReduce`` to reduce
        datasets.

        :param _size: The size of the coreset to be generated.
        :return: A `KernelThinning` solver if `leaf_size` is `None`, otherwise a
            `MapReduce` solver with `KernelThinning` as the base solver.
        """
        thinning_solver = KernelThinning(
            coreset_size=_size,
            kernel=kernel,
            random_key=key,
            delta=calculate_delta(num_data_points).item(),
            sqrt_kernel=sqrt_kernel,
        )
        if leaf_size is None:
            return thinning_solver
        return MapReduce(thinning_solver, leaf_size=leaf_size)

    def _get_herding_solver(_size: int) -> Union[KernelHerding, MapReduce]:
        """
        Set up kernel herding solver.

        If the `leaf_size` is provided, the solver uses ``MapReduce`` to reduce
        datasets.

        :param _size: The size of the coreset to be generated.
        :return: A `KernelHerding` solver if `leaf_size` is `None`, otherwise a
             `MapReduce` solver with `KernelHerding` as the base solver.
        """
        herding_solver = KernelHerding(_size, kernel)
        if leaf_size is None:
            return herding_solver
        return MapReduce(herding_solver, leaf_size=leaf_size)

    def _get_stein_solver(_size: int) -> Union[SteinThinning, MapReduce]:
        """
        Set up Stein thinning solver.

        If the `leaf_size` is provided, the solver uses ``MapReduce`` to reduce
        datasets.

        :param _size: The size of the coreset to be generated.
        :return: A `SteinThinning` solver if `leaf_size` is `None`, otherwise a
             `MapReduce` solver with `SteinThinning` as the base solver.
        """
        # Generate small dataset for ScoreMatching for Stein Kernel

        score_function = KernelDensityMatching(length_scale=length_scale.item()).match(
            train_data_umap[idx]
        )
        stein_kernel = SteinKernel(kernel, score_function)
        stein_solver = SteinThinning(
            coreset_size=_size, kernel=stein_kernel, regularise=False
        )
        if leaf_size is None:
            return stein_solver
        return MapReduce(stein_solver, leaf_size=leaf_size)

    def _get_random_solver(_size: int) -> RandomSample:
        """
        Set up Random Sampling to generate a coreset.

        :param _size: The size of the coreset to be generated.
        :return: A RandomSample solver.
        """
        random_solver = RandomSample(_size, key)
        return random_solver

    def _get_rp_solver(_size: int) -> RPCholesky:
        """
        Set up Randomised Cholesky solver.

        :param _size: The size of the coreset to be generated.
        :return: An RPCholesky solver.
        """
        rp_solver = RPCholesky(coreset_size=_size, kernel=kernel, random_key=key)
        return rp_solver

    def _get_compress_solver(_size: int) -> CompressPlusPlus:
        """
        Set up Compress++ solver.

        :param _size: The size of the coreset to be generated.
        :return: A Compress++ solver.
        """
        compress_solver = CompressPlusPlus(
            coreset_size=_size,
            kernel=kernel,
            random_key=key,
            delta=calculate_delta(num_data_points).item(),
            sqrt_kernel=sqrt_kernel,
            g=cpp_oversampling_factor,
        )
        return compress_solver

    def _get_probabilistic_herding_solver(
        _size: int,
    ) -> Union[IterativeKernelHerding, MapReduce]:
        """
        Set up KernelHerding with probabilistic selection.

        If the `leaf_size` is provided, the solver uses ``MapReduce`` to reduce
        datasets.

        :param _size: The size of the coreset to be generated.
        :return: An `IterativeKernelHerding` solver if `leaf_size` is `None`, otherwise
            a `MapReduce` solver with `IterativeKernelHerding` as the base solver.
        """
        herding_solver = IterativeKernelHerding(
            coreset_size=_size,
            kernel=kernel,
            probabilistic=True,
            temperature=0.001,
            random_key=key,
            num_iterations=5,
        )
        if leaf_size is None:
            return herding_solver
        return MapReduce(herding_solver, leaf_size=leaf_size)

    def _get_iterative_herding_solver(
        _size: int,
    ) -> Union[IterativeKernelHerding, MapReduce]:
        """
        Set up KernelHerding with probabilistic selection.

        If the `leaf_size` is provided, the solver uses ``MapReduce`` to reduce
        datasets.

        :param _size: The size of the coreset to be generated.
        :return: An `IterativeKernelHerding` solver if `leaf_size` is `None`, otherwise
            a `MapReduce` solver with `IterativeKernelHerding` as the base solver.
        """
        herding_solver = IterativeKernelHerding(
            coreset_size=_size,
            kernel=kernel,
            probabilistic=False,
            temperature=0.001,
            random_key=key,
            num_iterations=5,
        )
        if leaf_size is None:
            return herding_solver
        return MapReduce(herding_solver, leaf_size=leaf_size)

    return {
        "Random Sample": _get_random_solver,
        "RP Cholesky": _get_rp_solver,
        "Kernel Herding": _get_herding_solver,
        "Stein Thinning": _get_stein_solver,
        "Kernel Thinning": _get_thinning_solver,
        "Compress++": _get_compress_solver,
        "Probabilistic Iterative Herding": _get_probabilistic_herding_solver,
        "Iterative Herding": _get_iterative_herding_solver,
    }
