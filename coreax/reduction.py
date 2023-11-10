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
from abc import ABC
from collections.abc import Callable
from functools import partial
from multiprocessing.pool import ThreadPool

import jax.numpy as jnp
from jax import Array, tree_util
from jax.typing import ArrayLike
from sklearn.neighbors import KDTree

import coreax.coreset as cc
import coreax.metrics as cm
import coreax.refine as cr
import coreax.util as cu
import coreax.weights as cw


class DataReduction(ABC):
    """
    Methods for reducing data.
    """

    def __init__(
        self,
        original_data: ArrayLike,
        weight: str | cw.WeightsOptimiser,
        kernel: cu.KernelFunction,
    ):
        r"""
        Class for performing data reduction.

        :param original_data: Original data before reduction
        :param weight: Type of weighting to apply
        :param kernel: Kernel function
           :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
        """
        self.original_data = jnp.asarray(original_data)
        self.reduced_data = original_data.copy()
        self.weight = weight
        self.kernel = kernel
        self.reduction_indices = jnp.asarray(range(original_data.shape[0]))

    def solve_weights(
        self,
    ) -> Array:
        """
        Solve for weights.

        :return: TODO once OOPedweights.py is implemented
        """
        # Create a weights optimiser object
        weights_instance = self._create_instance_from_factory(
            cw.WeightsOptimiser,
            self.weight,
        )
        return weights_instance.solve(
            self.original_data, self.reduced_data, self.kernel
        )

    def fit(
        self,
        coreset_name: str | type[cc.Coreset],
    ) -> Array:
        """
        Fit...TODO once coreset.py implemented

        :param coreset_name: Name of the coreset method to use, or an uninstantiated
            class object
        :return: TODO once OOPed coreset.py is implemented
        """
        # Create a coreset object
        coreset_instance = self._create_instance_from_factory(
            cc.coreset_factory, coreset_name
        )
        return coreset_instance.fit(self.original_data, self.kernel)

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
        :return: :math:`m` Refined coreset point indices
        """
        # Create a refine object
        refiner = self._create_instance_from_factory(
            cr.refine_factory, refine_name, kernel=self.kernel
        )
        return refiner.refine(
            self.original_data, self.reduction_indices, kernel_mean
        )  # TODO compute kernel mean here or in refine.py?

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
        metric_instance = self._create_instance_from_factory(
            cr.metric_factory,
            metric_name,
            kernel=self.kernel,
            weight=self.weight,
        )

        return metric_instance.compute(self.original_data, self.reduced_data)

    @staticmethod
    def _create_instance_from_factory(
        factory_obj: cu.ClassFactory,
        class_type: str
        | type[cc.Coreset]
        | type[cm.Metric]
        | type[cr.Refine]
        | type[cw.WeightsOptimiser],
        **kwargs,
    ) -> cc.Coreset | cm.Metric | cr.Refine | cw.WeightsOptimiser:
        """
        Create a refine object for use with the fit method.

        :param class_type: The name of a class to use, or the uninstantiated class
            directly as a dependency injection
        :return: Class instance of the requested type
        """
        class_obj = factory_obj.get(class_type)

        # Initialise, accounting for different classes having different numbers of
        # parameters
        return cu.call_with_excess_kwargs(
            class_obj,
            **kwargs,
        )

    def render(self):
        """
        TODO: once data.py is implemented
        """
        return self.original_data.render_reduction()

    def save(self):
        """
        TODO: once data.py is implemented
        """
        return self.original_data.save_reduction()


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
        children = (
            self.data_reduction,
            self.n,
        )
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
        children = (
            self.data_reduction,
            self.eps,
        )
        aux_data = {}
        return children, aux_data


class MapReduce(ReductionStrategy):
    def __init__(self, n, reduction_method: DataReduction):
        super().__init__(reduction_method)
        self.n = n

    def map_reduce(
        self,
        indices: ArrayLike,
        n_core: int,
        function: Callable[..., tuple[Array, Array, Array]],
        w_function: cu.KernelFunction | None,
        size: int = 1000,
        parallel: bool = True,
        **kwargs,
    ) -> tuple[Array, Array]:
        r"""
        # TODO: note for review - this function is largely copy-pasted from kernel_herding, I have not tried to change it other than fit it into the class structure.

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

        :param n_core: Number of coreset points to calculate
        :param function: Kernel herding function to call on each block
        :param w_function: Weights function. If unweighted, this is `None` # TODO: is this self.kernel or actually 'weights'
        :param size: Region size in number of points. Optional, defaults to `1000`
        :param parallel: Use multiprocessing. Optional, defaults to `True`
        :param kwargs: Keyword arguments to be passed to `function` after `X` and `n_core`
        :return: Coreset and weights, where weights is empty if unweighted
        """
        # check parameters to see if we need to invoke the kd-tree and recursion.
        if n_core >= size:
            raise OverflowError(
                f"Number of coreset points requested {n_core} is larger than the region size {size}. "
                f"Try increasing the size argument, or reducing the number of coreset points"
            )

        n = self.reduced_data.shape[0]
        weights = None
        if n <= n_core:
            coreset = indices
            if w_function is not None:
                _, Kc, Kbar = function(X=self.reduced_data, n_core=n_core, **kwargs)
                weights = w_function(Kc, Kbar)
        elif n_core < n <= size:
            # Tail case
            c, Kc, Kbar = function(X=self.reduced_data, n_core=n_core, **kwargs)
            coreset = indices[c]
            if w_function is not None:
                weights = w_function(Kc, Kbar)
        else:
            # build a kdtree
            kdtree = KDTree(self.reduced_data, leaf_size=size)
            _, nindices, nodes, _ = kdtree.get_arrays()
            new_indices = [jnp.array(nindices[nd[0] : nd[1]]) for nd in nodes if nd[2]]
            split_data = [self.reduced_data[n] for n in new_indices]

            # run k coreset problems
            coreset = []
            kwargs["n_core"] = n_core
            if parallel:
                with ThreadPool() as pool:
                    res = pool.map_async(partial(function, **kwargs), split_data)
                    res.wait()
                    for herding_output, idx in zip(res.get(), new_indices):
                        # different herding algorithms return different things
                        if isinstance(herding_output, tuple):
                            c, _, _ = herding_output
                        else:
                            c = herding_output
                        coreset.append(idx[c])

            else:
                for X_, idx in zip(split_data, new_indices):
                    c, _, _ = function(X_, **kwargs)
                    coreset.append(idx[c])

            coreset = jnp.concatenate(coreset)
            self.reduction_indices = self.reduction_indices[coreset].copy()
            # recurse; n_core is already in kwargs
            coreset, weights = self.map_reduce(
                function=function,
                w_function=w_function,
                size=size,
                parallel=parallel,
                **kwargs,
            )

        return coreset, weights

    def _tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable jit decoration
        of methods inside this class.
        """
        children = (
            self.data_reduction,
            self.n,
        )
        aux_data = {}
        return children, aux_data


# Define the pytree node for the added class to ensure methods with jit decorators
# are able to run. We rely on the naming convention that all child classes of
# DataReduction include the sub-string DataReduction inside of them.
for name, current_class in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    if "ReductionStrategy" in name and name != "ReductionStrategy":
        tree_util.register_pytree_node(
            current_class, current_class._tree_flatten, current_class._tree_unflatten
        )

# Set up class factories
data_reduction_factory = cu.ClassFactory(DataReduction)
reduction_strategy_factory = cu.ClassFactory(ReductionStrategy)
reduction_strategy_factory.register("size", SizeReduce)
reduction_strategy_factory.register("error", ErrorReduce)
reduction_strategy_factory.register("map", MapReduce)
