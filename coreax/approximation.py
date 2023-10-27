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

"""TODO: Create top-level docstring."""

from abc import ABC, abstractmethod
from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax, random, vmap
from jax.typing import ArrayLike

from coreax.util import KernelFunction


class KernelMeanApproximator(ABC):
    r"""
    Base class for approximation methods to kernel means.

    Define an approximator to the mean of a kernel distance matrix. When a dataset is
    very large, computing the mean distance between a given point and all other points
    can be time-consuming. Instead, this property can be approximated by various
    methods. :class:`~coreax.approximation.KernelMeanApproximator` is the base class
    for implementing these approximation methods.

    :param kernel_evaluation: Kernel function
            :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param random_key: Key for random number generation
    :param num_kernel_points: Number of kernel evaluation points
    """

    def __init__(
        self,
        kernel_evaluation: KernelFunction,
        random_key: random.PRNGKeyArray = random.PRNGKey(0),
        num_kernel_points: int = 10_000,
    ):
        """Construct class instance."""
        self.kernel_evaluation = kernel_evaluation
        self.random_key = random_key
        self.num_kernel_points = num_kernel_points

    @abstractmethod
    def approximate(self, data: ArrayLike) -> Array:
        r"""
        Approximate kernel row mean.

        :param data: The original :math:`n \times d` data
        :return: Approximation of the kernel matrix row sum divided by :math:`n`
        """


class RandomApproximator(KernelMeanApproximator):
    r"""
    Approximation to kernel mean through regression on random sampled points.

    Approximate kernel row mean by regression on points selected randomly. Here, the
    kernel row mean is the matrix row sum divided by :math:`n`.

    :param kernel_evaluation: Kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param random_key: Key for random number generation
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    """

    def __init__(
        self,
        kernel_evaluation: KernelFunction,
        random_key: random.PRNGKeyArray = random.PRNGKey(0),
        num_kernel_points: int = 10_000,
        num_train_points: int = 10_000,
    ):
        """Construct class instance."""
        self.num_train_points = num_train_points

        # Initialise parent
        super().__init__(
            kernel_evaluation=kernel_evaluation,
            random_key=random_key,
            num_kernel_points=num_kernel_points,
        )

    def approximate(
        self,
        data: ArrayLike,
    ) -> Array:
        r"""
        Compute approximate kernel row mean by regression on randomly selected points.

        :param data: The original :math:`n \times d` data
        :return: Approximation of the kernel matrix row sum divided by :math:`n`
        """
        # Ensure data is the expected type
        data = jnp.asarray(data)
        num_data_points = len(data)

        # Define function to compute pairwise distances with the kernel
        k_pairwise = jit(
            vmap(
                vmap(self.kernel_evaluation, in_axes=(None, 0), out_axes=0),
                in_axes=(0, None),
                out_axes=0,
            )
        )

        # Randomly select points for kernel regression
        key, subkey = random.split(self.random_key)
        features_idx = random.choice(
            subkey, num_data_points, (self.num_kernel_points,), replace=False
        )
        features = k_pairwise(data, data[features_idx])

        # Select training points
        train_idx = random.choice(
            key, num_data_points, (self.num_train_points,), replace=False
        )
        target = k_pairwise(data[train_idx], data).sum(axis=1) / num_data_points

        # Solve regression problem.
        params, _, _, _ = jnp.linalg.lstsq(features[train_idx], target)

        return features @ params


class ANNchorApproximator(KernelMeanApproximator):
    r"""
    Approximation method to kernel mean through regression on ANNchor selected points.

    Here, the kernel row mean is the matrix row sum divided by :math:`n`. The ANNchor
    implementation used can be found `here <https://github.com/gchq/annchor>`_.

    :param kernel_evaluation: Kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param random_key: Key for random number generation
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    """

    def __init__(
        self,
        kernel_evaluation: KernelFunction,
        random_key: random.PRNGKeyArray = random.PRNGKey(0),
        num_kernel_points: int = 10_000,
        num_train_points: int = 10_000,
    ):
        """Construct class instance."""
        self.num_train_points = num_train_points

        # Initialise parent
        super().__init__(
            kernel_evaluation=kernel_evaluation,
            random_key=random_key,
            num_kernel_points=num_kernel_points,
        )

    def approximate(
        self,
        data: ArrayLike,
    ) -> Array:
        r"""
        Compute approximate kernel row mean by regression on ANNchor selected points.

        :param data: The original :math:`n \times d` data
        :return: Approximation of the kernel matrix row sum divided by :math:`n`
        """
        # Ensure data is the expected type
        data = jnp.asarray(data)
        n = len(data)

        # Define function to compute pairwise distances with the kernel
        kernel_pairwise = jit(
            vmap(
                vmap(self.kernel_evaluation, in_axes=(None, 0), out_axes=0),
                in_axes=(0, None),
                out_axes=0,
            )
        )

        # kernel_vector is a function R^d x R^d \to R^d
        kernel_vector = jit(vmap(self.kernel_evaluation, in_axes=(0, None)))

        # Select point for kernel regression using ANNchor construction
        features = jnp.zeros((n, self.num_kernel_points))
        features = features.at[:, 0].set(kernel_vector(data, data[0]))
        body = partial(anchor_body, data=data, kernel_function=kernel_vector)
        features = lax.fori_loop(1, self.num_kernel_points, body, features)

        train_idx = random.choice(
            self.random_key, n, (self.num_train_points,), replace=False
        )
        target = kernel_pairwise(data[train_idx], data).sum(axis=1) / n

        # solve regression problem
        params, _, _, _ = jnp.linalg.lstsq(features[train_idx], target)

        return features @ params


class NystromApproximator(KernelMeanApproximator):
    r"""
    Approximate kernel row mean by using Nystrom approximation.

    Here, the kernel row mean is the matrix row sum divided by :math:`n`. Further
    details for Nystrom kernel mean embeddings can be found in
    :cite:p:`chatalic2022nystrom`.

    :param kernel_evaluation: Kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param random_key: Key for random number generation
    :param num_kernel_points: Number of kernel evaluation points
    """

    def __init__(
        self,
        kernel_evaluation: KernelFunction,
        random_key: random.PRNGKeyArray = random.PRNGKey(0),
        num_kernel_points: int = 10_000,
    ):
        """Construct class instance."""
        # Initialise parent
        super().__init__(
            kernel_evaluation=kernel_evaluation,
            random_key=random_key,
            num_kernel_points=num_kernel_points,
        )

    def approximate(
        self,
        data: ArrayLike,
    ) -> Array:
        r"""
        Compute approximate kernel row mean by regression on ANNchor selected points.

        :param data: The original :math:`n \times d` data
        :return: Approximation of the kernel matrix row sum divided by :math:`n`
        """
        # Ensure data is the expected type
        data = jnp.asarray(data)
        num_data_points = len(data)

        # Define function to compute pairwise distances with the kernel
        kernel_pairwise = jit(
            vmap(
                vmap(self.kernel_evaluation, in_axes=(None, 0), out_axes=0),
                in_axes=(0, None),
                out_axes=0,
            )
        )

        # Randomly select points for kernel regression
        sample_points = random.choice(
            self.random_key, num_data_points, (self.num_kernel_points,)
        )

        # Solve for kernel distances
        k_mn = kernel_pairwise(data[sample_points], data)
        k_mm = kernel_pairwise(data[sample_points], data[sample_points])
        alpha = (jnp.linalg.pinv(k_mm) @ k_mn).sum(axis=1) / num_data_points

        return k_mn.T @ alpha


@partial(jit, static_argnames=["kernel_function"])
def anchor_body(
    idx: int,
    features: ArrayLike,
    data: ArrayLike,
    kernel_function: KernelFunction,
) -> Array:
    r"""
    Execute main loop of the ANNchor construction.

    :param idx: Loop counter
    :param features: Loop updateables
    :param data: Original :math:`n \times d` dataset
    :param kernel_function: Vectorised kernel function on pairs `(X,x)`:
        :math:`k: \mathbb{R}^{n \times d} \times \mathbb{R}^d \rightarrow \mathbb{R}^n`
    :return: Updated loop variables ``features``
    """
    features = jnp.asarray(features)
    data = jnp.asarray(data)

    max_entry = features.max(axis=1).argmin()
    features = features.at[:, idx].set(kernel_function(data, data[max_entry]))

    return features
