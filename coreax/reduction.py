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

# Support annotations with | in Python < 3.10
# TODO: Remove once no longer supporting old code
from __future__ import annotations

import inspect
import sys
import jax.numpy as jnp

from abc import ABC, abstractmethod
from functools import partial
from jax import Array
from jax import tree_util
from multiprocessing.pool import ThreadPool
from sklearn.neighbors import KDTree

import coreax.coreset as cc
import coreax.data as cd
import coreax.kernel as ck
import coreax.metrics as cm
import coreax.refine as cr
import coreax.util as cu
import coreax.weights as cw


class DataReduction(ABC):
    """
    Methods for reducing data.

    Class for performing data reduction.

    :param data: DataReader instance with original data before reduction
    :param weight: Type of weighting to apply
    :param kernel: Kernel function
       :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    """

    def __init__(
            self,
            data: cd.DataReader,
            weight: str | cw.WeightsOptimiser,
            kernel: cu.KernelFunction):

        self.data = data
        self.weight = weight
        self.kernel = kernel
        self.reduction_indices = jnp.asarray(range(data.pre_reduction_array.shape[0]))
        self.reduced_data = data.pre_reduction_array

    def solve_weights(
            self,
    ) -> Array:
        """
        Solve for weights.

        :return: TODO once OOPedweights.py is implemented
        """
        # Create a weights optimiser object
        weights_instance = cu.create_instance_from_factory(
            cw.WeightsOptimiser,
            self.weight,
        )
        return weights_instance.solve(self.data, self.reduced_data, self.kernel)

    @abstractmethod
    def fit(self) -> None:
        """
        Fit...TODO once coreset.py implemented
        """

    def refine(
        self,
        refine_name: str | type[cr.Refine],
    ) -> Array:
        """
        Compute the refined coreset, of m points in d dimensions.

        The refinement procedure replaces elements with points most reducing maximum mean
        discrepancy (MMD). The iteration is carried out over points in original_data.

        :param refine_name: Name of the refine type to use, or an uninstantiated
            class object
        :return: :math:`m` refined coreset point indices
        """
        # Create a refine object
        refiner = cu.create_instance_from_factory(
            cr.refine_factory,
            refine_name,
            kernel=self.kernel
        )
        return refiner.refine(self.data, self.reduction_indices, kernel_mean)  # TODO compute kernel mean here or in refine.py?

    def compute_metric(
        self,
        metric_name: str | type[cm.Metric],
    ) -> Array:
        """
        Compute metrics...TODO: once OOPed metrics.py is implemented

        :param metric_name: Name of the metric type to use, or an uninstantiated
            class object
        :return: TODO: once OOPed metrics.py is implemented
        """
        # Create a metric object
        metric_instance = cu.create_instance_from_factory(
            cr.metric_factory,
            metric_name,
            kernel=self.kernel,
            weight=self.weight,
        )

        return metric_instance.compute(self.data, self.reduced_data)

    def format(self):
        """
        TODO: once data.py is implemented
        """
        return self.data.format()

    def render(self):
        """
        TODO: once data.py is implemented
        """
        return self.data.render()

class ReductionStrategy(ABC):
    """
    TODO
    """

    def __init__(self, reduction_method: DataReduction):
        """
        TODO
        """

        self.data_reduction = reduction_method

    def reduce(self, original_data, weight, kernel):
        """
        TODO
        """

        return self.data_reduction.__init__(original_data, weight, kernel)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """
        Reconstructs a pytree from the tree definition and the leaves.

        Arrays & dynamic values (children) and auxiliary data (static values) are
        reconstructed. A method to reconstruct the pytree needs to be specified to
        enable jit decoration of methods inside children of this class.
        """
        return cls(*children, **aux_data)


class SizeReduce(ReductionStrategy):

    def __init__(self, n, reduction_method: DataReduction):

        super().__init__(reduction_method)
        self.n = n

    def generate_coreset(self):
        # TODO: return Coreset from coreset.py when ready
        return NotImplementedError

    def _tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable jit decoration
        of methods inside this class.
        """
        children = (self.data_reduction, self.n,)
        aux_data = {}
        return children, aux_data


class ErrorReduce(ReductionStrategy):

    def __init__(self, eps, reduction_method: DataReduction):

        super().__init__(reduction_method)
        self.eps = eps

    def reduce_error(self):
        # TODO: vary n to meet probabilistically...
        return NotImplementedError

    def _tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable jit decoration
        of methods inside this class.
        """
        children = (self.data_reduction, self.eps,)
        aux_data = {}
        return children, aux_data


class MapReduce(ReductionStrategy):

    def __init__(self, size, reduction_method: DataReduction):

        super().__init__(reduction_method)
        self.size = size

    def map_reduce(
            self,
            w_function: ck.Kernel | None,
            block_size: int = 10_000,
            K_mean: Array | None = None,
            unique: bool = True,
            nu: float = 1.0,
            partition_size: int = 1000,
            parallel: bool = True
    ) -> tuple[Array, Array]:
        r"""
        Execute scalable reduction.

        This uses a `kd-tree` to partition `X`-space into patches. Upon each of these a
        reduction problem is solved.

        There is some intricate setup:

        TODO: review for herding references

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
        # use reduced data in case this is not the first reduction applied
        data_to_reduce = self.data_reduction.data.reduced_data

        # check parameters to see if we need to invoke the kd-tree and recursion.
        if self.size >= partition_size:
            raise OverflowError(
                f"Number of coreset points requested {self.size} is larger than the region size {partition_size}. "
                f"Try increasing the size argument, or reducing the number of coreset points."
            )
        n = data_to_reduce.shape[0]
        weights = None

        # fewer data points than requested coreset points so return all
        if n <= self.size:
            coreset = self.reduction_indices
            if w_function is not None:
                _, Kc, Kbar = self.data_reduction.fit(data_to_reduce, self.data_reduction.kernel, block_size, K_mean, unique, nu)
                weights = w_function(Kc, Kbar)

        # coreset points < data points <= partition size, so no partitioning required
        elif self.size < n <= partition_size:
            c, Kc, Kbar = self.data_reduction.fit(data_to_reduce, self.data_reduction.kernel, block_size, K_mean, unique, nu)
            coreset = self.reduction_indices[c]
            if w_function is not None:
                weights = w_function(Kc, Kbar)

        # partitions required
        else:
            # build a kdtree
            kdtree = KDTree(data_to_reduce, leaf_size=partition_size)
            _, nindices, nodes, _ = kdtree.get_arrays()
            new_indices = [jnp.array(nindices[nd[0]: nd[1]]) for nd in nodes if nd[2]]
            split_data = [data_to_reduce[n] for n in new_indices]

            # generate a coreset on each partition
            coreset = []
            kwargs["self.size"] = self.size
            if parallel:
                with ThreadPool() as pool:
                    res = pool.map_async(partial(self.data_reduction.fit, self.data_reduction.kernel, block_size, K_mean, unique, nu), split_data)
                    res.wait()
                    for herding_output, idx in zip(res.get(), new_indices):
                        c, _, _ = herding_output
                        coreset.append(idx[c])

            else:
                for X_, idx in zip(split_data, new_indices):
                    c, _, _ = self.data_reduction.fit(X_, self.data_reduction.kernel, block_size, K_mean, unique, nu)
                    coreset.append(idx[c])

            coreset = jnp.concatenate(coreset)
            Xc = data_to_reduce[coreset]
            self.reduction_indices = self.reduction_indices[coreset]
            # recurse;
            coreset, weights = self.map_reduce(
                w_function,
                block_size,
                K_mean,
                unique,
                nu,
                partition_size,
                parallel,
            )

        return coreset, weights

    def _tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable jit decoration
        of methods inside this class.
        """
        children = (self.data_reduction, self.n,)
        aux_data = {}
        return children, aux_data


# Define the pytree node for the added class to ensure methods with jit decorators
# are able to run. This tuple must be updated when a new class object is defined.
reduction_classes = (SizeReduce, ErrorReduce, MapReduce)
for current_class in reduction_classes:
    tree_util.register_pytree_node(
        current_class, current_class._tree_flatten, current_class._tree_unflatten
    )

# Set up class factories
data_reduction_factory = cu.ClassFactory(DataReduction)
reduction_strategy_factory = cu.ClassFactory(ReductionStrategy)
reduction_strategy_factory.register("size", SizeReduce)
reduction_strategy_factory.register("error", ErrorReduce)
reduction_strategy_factory.register("map", MapReduce)
