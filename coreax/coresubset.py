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

r"""
Classes and associated functionality to construct coresets.

Given a :math:`n \times d` dataset, one may wish to construct a compressed
:math:`m \times d` dataset representation of this dataset, where :math:`m << n`. This
module contains implementations of approaches to do such a construction using coresets.
Coresets are a type of data reduction, and these inherit from
:class:`~coreax.reduction.Coreset`. The aim is to select a small set of indices
that represent the key features of a larger dataset.

The abstract base class is :class:`CoreSubset`. Concrete implementations are:

*   :class:`KernelHerding` defines the kernel herding method for both regular and Stein
    kernels.
*   :class:`RandomSample` selects points for the coreset using random sampling. It is
    typically only used for benchmarking against other coreset methods.

**:class:`KernelHerding`**
Kernel herding is a deterministic, iterative and greedy approach to determine this
compressed representation.

Given one has selected ``T`` data points for their compressed representation of the
original dataset, kernel herding selects the next point as:

.. math::

    x_{T+1} = \argmax_{x} \left( \mathbb{E}[k(x, x')] -
        \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) \right)

where ``k`` is the kernel used, the expectation :math:`\mathbb{E}` is taken over the
entire dataset, and the search is over the entire dataset. This can informally be seen
as a balance between using points at which the underlying density is high (the first
term) and exploration of distinct regions of the space (the second term).
"""

# Support annotations with | in Python < 3.10
# TODO: Remove once no longer supporting old code
from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING

import jax.lax as lax
import jax.numpy as jnp
from jax import Array, jit, random
from jax.typing import ArrayLike

from coreax.approximation import KernelMeanApproximator, approximator_factory
from coreax.kernel import Kernel
from coreax.reduction import Coreset, coreset_factory
from coreax.refine import Refine, refine_factory
from coreax.util import KernelFunction, create_instance_from_factory
from coreax.weights import WeightsOptimiser

if TYPE_CHECKING:
    from coreax.data import DataReader


class CoreSubset(Coreset):
    """Abstract base class for a method to construct a coreset."""

    def __init__(
        self,
        original_data: DataReader,
        weights: str | WeightsOptimiser,
        kernel: Kernel,
        size: int,
    ):
        """
        TODO: write docstring.

        :param size: Number of coreset points to calculate
        """
        self.coreset_size = size
        super().__init__(original_data, weights, kernel)

        self.coreset_indices = jnp.asarray(
            range(original_data.pre_coreset_array.shape[0])
        )

    @abstractmethod
    def fit(
        self,
        X: Array,
        kernel: Kernel,
    ) -> None:
        """
        Fit...TODO once children implemented.

        :param X: data
        :param kernel: kernel
        """


class KernelHerding(CoreSubset):
    """
    Apply kernel herding to a dataset.

    This class works with all kernels, including Stein kernels.
    """

    def __init__(
        self,
        original_data: DataReader,
        weights: str | WeightsOptimiser,
        kernel: Kernel,
        size: int,
    ):
        """
        TODO: write docstring.

        :param size: Number of coreset points to calculate
        """
        # Initialise Coreset parent
        super().__init__(original_data, weights, kernel, size)

    def fit(
        self,
        block_size: int = 10000,
        K_mean: Array | None = None,
        unique: bool = True,
        refine: str | Refine | None = None,
        approximator: str | KernelMeanApproximator | None = None,
        random_key: random.PRNGKeyArray = random.PRNGKey(0),
        num_kernel_points: int = 10_000,
    ) -> tuple[Array, Array]:
        r"""
        Execute kernel herding algorithm with Jax.

        :param block_size: Size of matrix blocks to process
        :param K_mean: Row sum of kernel matrix divided by `n`
        :param unique: Flag for enforcing unique elements
        :param refine: Refine method to use or None (default) if no refinement required
        :param approximator: coreax KernelMeanApproximator object for the kernel mean
            approximation method. If None (default) then calculation is exact.
        :param random_key: Key for random number generation
        :param num_kernel_points: Number of kernel evaluation points for approximation
        :returns: coreset Gram matrix and coreset Gram mean
        """
        n = len(self.original_data.pre_coreset_array)
        if K_mean is None:
            # TODO: for the reviewer, the issue ticket says we should "incorporate the caching of K_mean from
            #  kernel_herding_refine into KernelHerding" but the mean is needed here before being calculated in Refine.refine
            print(type(self.kernel))
            K_mean = self.kernel.calculate_kernel_matrix_row_sum_mean(
                x=self.original_data.pre_coreset_array, max_size=block_size
            )

        # Initialise loop updateables
        K_t = jnp.zeros(n)
        S = jnp.zeros(self.coreset_size, dtype=jnp.int32)
        print("S: ", S)
        K = jnp.zeros((self.coreset_size, n))

        # Greedly select coreset points
        body = partial(
            self._greedy_body, k_vec=self.kernel.compute, K_mean=K_mean, unique=unique
        )
        # self.coreset_indices, K, _ = lax.fori_loop(
        #     lower=0, upper=self.coreset_size, body_fun=body, init_val=(S, K, K_t)
        # )
        self.coreset_indices, K_t = lax.fori_loop(
            lower=0, upper=self.coreset_size, body_fun=body, init_val=(S, K_t)
        )
        print("self.coreset_indices: ", self.coreset_indices)
        Kbar = K.mean(axis=1)
        # gram_matrix = K[:, self.coreset_indices]
        gram_matrix = []

        # TODO: for reviewer, this whole block seems clunky...
        if refine is not None:
            if approximator is not None:
                # Create an approximator object
                approximator_instance = create_instance_from_factory(
                    approximator_factory,
                    approximator,
                    random_key=random_key,
                    num_kernel_points=num_kernel_points,
                )
            else:
                approximator_instance = None
            # Create a Refine object
            refine_instance = create_instance_from_factory(
                refine_factory,
                refine,
                approximate_kernel_row_sum=False if approximator is None else True,
                approximator=approximator_instance,
            )
            # refine
            refine_instance.refine(data_reduction=self)

        # self.coreset_indices = coreset_indices
        self.coreset = self.original_data.pre_coreset_array[self.coreset_indices, :]
        return gram_matrix, Kbar  # delete this

    @partial(jit, static_argnames=["self", "k_vec", "unique"])
    def _greedy_body(
        self,
        i: int,
        val: tuple[ArrayLike, ArrayLike],
        # S: ArrayLike,
        k_vec: KernelFunction,
        K_mean: ArrayLike,
        unique: bool,
    ) -> tuple[Array, Array]:  # tuple[Array, Array, Array]:
        r"""
        Execute main loop of greedy kernel herding.

        :param i: Loop counter
        :param val: Loop updatables
        :param k_vec: Vectorised kernel function on pairs `(X,x)`:
                      :math:`k: \mathbb{R}^{n \times d} \times \mathbb{R}^d \rightarrow \mathbb{R}^n`
        :param K_mean: Mean vector over rows for the Gram matrix, a :math:`1 \times n` array
        :param unique: Flag for enforcing unique elements
        :returns: Updated loop variables (`coreset indices`, `Gram matrix`, `objective`)
        """
        # print("val[0]: ", val[0])
        # S, K, K_t = val
        S, K_t = val
        S = jnp.asarray(S)
        # K = jnp.asarray(K)
        j = (K_mean - K_t / (i + 1)).argmax()
        # kv = k_vec(self.original_data.pre_coreset_array[S,:], self.original_data.pre_coreset_array[S[j],:])
        # K_t = K_t + kv
        S = S.at[i].set(j)
        print(S)
        # K = K.at[i].set(kv)
        # if unique:
        #     K_t = K_t.at[j].set(jnp.inf)

        return S, K_t  # , K, K_t


class RandomSample(CoreSubset):
    r"""
    Reduce a dataset by uniformly randomly sampling a fixed number of points.

    :param original_data: The :math:`n \times d` dataset to be reduced
    :param coreset_size: Number of  points to randomly sample from ``data``
    :param random_key: Pseudo-random number generator key for sampling
    :param unique: If :data:`True`, this flag enforces unique elements, i.e. sampling
        without replacement
    """

    def __init__(
        self,
        original_data: DataReader,
        coreset_size: int,
        random_key: ArrayLike = 0,
        unique: bool = True,
    ):
        """Initialise a random sampling object."""
        self.random_key = random_key
        self.unique_flag = unique

        # Initialise Coreset parent
        super().__init__(
            original_data=original_data, weights=None, kernel=None, size=coreset_size
        )

    def fit(self) -> None:
        r"""
        Reduce a dataset by uniformly randomly sampling a fixed number of points.

        This class is updated in-place. The randomly sampled points are stored in the
        ``reduction_indices`` attribute.

        :return: Nothing, the coreset is assigned to :attr:`coreset`
        """
        key = random.PRNGKey(self.random_key)
        orig_data = self.original_data.pre_coreset_array
        n = len(orig_data)

        random_indices = random.choice(
            key,
            a=jnp.arange(0, n),
            shape=(self.coreset_size,),
            replace=not self.unique_flag,
        )
        self.coreset_indices = random_indices
        self.coreset = orig_data[random_indices]


coreset_factory.register("kernel_herding", KernelHerding)
coreset_factory.register("random_sample", RandomSample)
