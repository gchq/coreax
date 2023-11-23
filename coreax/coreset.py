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
Classes and associated functionality to construct coresets.

Given a :math:`n \times d` dataset, one may wish to construct a compressed
:math:`m \times d` dataset representation of this dataset, where :math:`m << n`. This
module contains implementations of approaches to do such a construction using coresets.
Coresets are a type of data reduction, so these inherit from
:class:`~coreax.reduction.DataReduction`. The aim is to select a samll set of indices
that represent the key features of a larger dataset.

The abstract base class is :class:`Coreset`. Concrete implementations are:

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

from abc import abstractmethod
from functools import partial
from multiprocessing.pool import ThreadPool

import jax.lax as lax
import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike
from sklearn.neighbors import KDTree

from coreax.data import DataReader
from coreax.kernel import Kernel
from coreax.reduction import DataReduction, data_reduction_factory
from coreax.util import KernelFunction
from coreax.weights import WeightsOptimiser


class Coreset(DataReduction):
    """Abstract base class for a method to construct a coreset."""

    def __init__(
            self,
            data: DataReader,
            weight: str | WeightsOptimiser,
            kernel: Kernel,
            size: int
    ):
        """

        :param size: Number of coreset points to calculate
        """

        self.coreset_size = size
        super().__init__(data, weight, kernel)

        self.reduction_indices = jnp.asarray(range(data.pre_reduction_data.shape[0]))

    @abstractmethod
    def fit(self, X: Array, kernel: Kernel,) -> None:
        """
        Fit...TODO once children implemented
        """


class KernelHerding(Coreset):
    """
    Apply kernel herding to a dataset.

    This class works with all kernels, including Stein kernels.
    """

    def __init__(
            self,
            data: DataReader,
            weight: str | WeightsOptimiser,
            kernel: Kernel,
            size: int):
        """

        :param size: Number of coreset points to calculate
        """

        # Initialise Coreset parent
        super().__init__(data, weight, kernel, size)

    def fit_by_partition(
            self,
            X: Array,
            w_function: Kernel | None,
            block_size: int = 10_000,
            K_mean: Array | None = None,
            unique: bool = True,
            nu: float = 1.0,
            partition_size: int = 1000,
            parallel: bool = True
    ) -> tuple[Array, Array]:
        r"""
        Execute scalable kernel herding.

        This uses a `kd-tree` to partition `X`-space into patches. Upon each of these a
        kernel herding problem is solved.

        There is some intricate setup:

            #.  Parameter `n_core` must be less than `size`.
            #.  If we have :math:`n` points, unweighted herding is executed recursively on
                each patch of :math:`\lceil \frac{n}{size} \rceil` points.
            #.  If :math:`r` is the recursion depth, then we recurse unweighted for
                :math:`r` iterations where

                .. math::

                         r = \lfloor \log_{frac{n_core}{size}}(\frac{n_core}{n})\rfloor

                Each recursion gives :math:`n_r = C \times k_{r-1}` points. Unpacking the
                recursion, this gives
                :math:`n_r \approx n_0 \left( \frac{n_core}{n_size}\right)^r`.
            #.  Once :math:`n_core < n_r \leq size`, we run a final weighted herding (if
                weighting is requested) to give :math:`n_core` points.

        :param X: :math:`n \times d` dataset to find a coreset from
        :param kernel: Kernel function
                       :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
        :param w_function: Weights function. If unweighted, this is `None`
        :param block_size: Size of matrix blocks to process
        :param K_mean: Row sum of kernel matrix divided by `n`
        :param unique: Flag for enforcing unique elements
        :param partition_size: Region size in number of points. Optional, defaults to `1000`
        :param parallel: Use multiprocessing. Optional, defaults to `True`
        :param kwargs: Keyword arguments to be passed to `function` after `X` and `n_core`
        :return: Coreset and weights, where weights is empty if unweighted
        """
        # check parameters to see if we need to invoke the kd-tree and recursion.
        if self.coreset_size >= partition_size:
            raise OverflowError(
                f"Number of coreset points requested {self.coreset_size} is larger than the region size {partition_size}. "
                f"Try increasing the size argument, or reducing the number of coreset points."
            )
        n = X.shape[0]
        weights = None

        # fewer data points than requested coreset points so return all
        if n <= self.coreset_size:
            coreset = self.reduction_indices
            if w_function is not None:
                _, Kc, Kbar = self.fit(X, self.kernel, block_size, K_mean, unique, nu)
                weights = w_function(Kc, Kbar)

        # coreset points < data points <= partition size, so no partitioning required
        elif self.coreset_size < n <= partition_size:
            c, Kc, Kbar = self.fit(X, self.kernel, block_size, K_mean, unique, nu)
            coreset = self.reduction_indices[c]
            if w_function is not None:
                weights = w_function(Kc, Kbar)

        # partitions required
        else:
            # build a kdtree
            kdtree = KDTree(X, leaf_size=partition_size)
            _, nindices, nodes, _ = kdtree.get_arrays()
            new_indices = [jnp.array(nindices[nd[0]: nd[1]]) for nd in nodes if nd[2]]
            split_data = [X[n] for n in new_indices]

            # generate a coreset on each partition
            coreset = []
            kwargs["self.size"] = self.coreset_size
            if parallel:
                with ThreadPool() as pool:
                    res = pool.map_async(partial(self.fit, self.kernel, block_size, K_mean, unique, nu), split_data)
                    res.wait()
                    for herding_output, idx in zip(res.get(), new_indices):
                        c, _, _ = herding_output
                        coreset.append(idx[c])

            else:
                for X_, idx in zip(split_data, new_indices):
                    c, _, _ = self.fit(X_, self.kernel, block_size, K_mean, unique, nu)
                    coreset.append(idx[c])

            coreset = jnp.concatenate(coreset)
            Xc = X[coreset]
            self.reduction_indices = self.reduction_indices[coreset]
            # recurse;
            coreset, weights = self.fit_by_partition(
                Xc,
                w_function,
                block_size,
                K_mean,
                unique,
                nu,
                partition_size,
                parallel,
            )

        return coreset, weights

    def fit(
            self,
            X: Array,
            block_size: int = 10_000,
            K_mean: Array | None = None,
            unique: bool = True,
            nu: float = 1.0,
    ) -> tuple[Array, Array, Array]:
        r"""
        Execute kernel herding algorithm with Jax.

        :param X: :math:`n \times d` dataset to find a coreset from
        :param block_size: Size of matrix blocks to process
        :param K_mean: Row sum of kernel matrix divided by `n`
        :param unique: Flag for enforcing unique elements
        :returns: Coreset point indices, coreset Gram matrix and coreset Gram mean
        """

        n = len(X)
        if K_mean is None:
            K_mean = kernel.calculate_kernel_matrix_row_sum_mean(X, max_size=block_size)

        # Initialise loop updateables
        K_t = jnp.zeros(n)
        S = jnp.zeros(self.coreset_size, dtype=jnp.int32)
        K = jnp.zeros((self.coreset_size, n))

        # Greedly select coreset points
        body = partial(self._greedy_body, k_vec=self.kernel.compute, K_mean=K_mean, unique=unique)
        S, K, _ = lax.fori_loop(0, self.coreset_size, body, (S, K, K_t))
        Kbar = K.mean(axis=1)
        gram_matrix = K[:, S]

        return S, gram_matrix, Kbar

    @partial(jit, static_argnames=["k_vec", "unique"])
    def _greedy_body(
            self,
            X: Array,
            i: int,
            val: tuple[ArrayLike, ArrayLike, ArrayLike],
            k_vec: KernelFunction,
            K_mean: ArrayLike,
            unique: bool,
    ) -> tuple[Array, Array, Array]:
        r"""
        Execute main loop of greedy kernel herding.

        :param X: :math:`n \times d` dataset to find a coreset from
        :param i: Loop counter
        :param val: Loop updatables
        :param k_vec: Vectorised kernel function on pairs `(X,x)`:
                      :math:`k: \mathbb{R}^{n \times d} \times \mathbb{R}^d \rightarrow \mathbb{R}^n`
        :param K_mean: Mean vector over rows for the Gram matrix, a :math:`1 \times n` array
        :param unique: Flag for enforcing unique elements
        :returns: Updated loop variables (`coreset`, `Gram matrix`, `objective`)
        """
        S, K, K_t = val
        S = jnp.asarray(S)
        K = jnp.asarray(K)
        j = (K_mean - K_t / (i + 1)).argmax()
        kv = k_vec(X, X[j])
        K_t = K_t + kv
        S = S.at[i].set(j)
        K = K.at[i].set(kv)
        if unique:
            K_t = K_t.at[j].set(jnp.inf)

        return S, K, K_t

data_reduction_factory.register("kernel_herding", KernelHerding)
