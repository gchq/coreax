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
from jax import Array
from jax.typing import ArrayLike
from jax import tree_util

import coreax.coreset as cc
import coreax.metrics as cm
import coreax.refine as cr
import coreax.util as cu
import coreax.weights as cw


class DataReduction(ABC):
    """
    Methods for reducing data.
    """

    def __init__(self, original_data: ArrayLike, weighting: str):
        r"""
        Define a ... TODO
        """
        self.original_data = original_data
        self.reduced_data = original_data.copy()
        self.weighting = weighting

    def solve_weights(
            self,
            kernel: cu.KernelFunction
    ) -> Array | None:
        """
        Solve for weights. Currently implemented are MMD and SBQ weights.

        TODO: update when weights.py is OOPed.
        """

        if self.weighting is None:
            return None
        elif self.weighting == 'MMD':
            return cw.simplex_weights(
                self.original_data,
                self.reduced_data,
                kernel
            )
        elif self.weighting == 'SBQ':
            return cw.calculate_BQ_weights(
                self.original_data,
                self.reduced_data,
                kernel
            )
        else:
            raise ValueError(f"weight type '{self.weighting}' not recognised.")

    def fit(
        self,
        coreset_name: str | type[cc.Coreset],
        kernel: cu.KernelFunction,
    ) -> Array:
        """
        Fit...TODO

        :param coreset_name: Name of the coreset method to use, or an uninstantiated
            class object
        :param kernel: Kernel function
           :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
        :return: Approximation to the kernel matrix row sum
        """
        # Create an approximator object
        coreset_instance = self._create_instance_from_factory(
            cc.coreset_factory,
            coreset_name
        )
        return coreset_instance.fit(self.original_data, kernel)

    def refine(
        self,
        x: ArrayLike,
        refine_name: str | type[cr.Refine],
    ) -> Array:
        """
        Refine...TODO

        :param x: Data matrix, :math:`n \times d`
        :param refine_name: Name of the refine type to use, or an uninstatiated
            class object
        :return: Approximation to the kernel matrix row sum
        """
        # Create an approximator object
        refiner = self._create_instance_from_factory(
            cr.refine_factory,
            refine_name
        )
        return refiner.refine(x)  # TODO check what refine actually needs to do here...

    def compute_metric(
        self,
        x: ArrayLike,
        metric_name: str | type[cm.Metric],
    ) -> Array:
        """
        Refine...TODO

        :param x: Data matrix, :math:`n \times d`
        :param refine_name: Name of the refine type to use, or an uninstatiated
            class object
        :return: Approximation to the kernel matrix row sum
        """
        # Create an approximator object
        metric_instance = self._create_instance_from_factory(
            cr.metric_factory,
            metric_name
        )
        return metric_instance.refine(x)

    def _create_instance_from_factory(
        self,
        factory_obj: cu.ClassFactory,
        class_type: str | type[cc.Corset] | type[cm.Metric] | type[cr.Refine] | type[cw.WeightsOptimiser],
    ) -> cc.Corset | cm.Metric | cr.Refine | cw.WeightsOptimiser:
        """
        Create a refine object for use with the fit method.

        :param class_type: The name of a class to use, or the uninstantiated class
            directly as a dependency injection
        :return: Refine object
        """
        class_obj = factory_obj.get(class_type)

        # Initialise, accounting for different classes having different numbers of
        # parameters
        return cu.call_with_excess_kwargs(
            class_obj,
            kernel_evaluation=self._compute_elementwise,
        )

    def render(self):
        """
        TODO
        """
        return self.original_data.render_reduction

    def save(self):
        """
        TODO
        """
        return self.original_data.save_reduction


class ReductionStrategy(ABC):
    """
    TODO
    """

    def __init__(self, reduction_method: DataReduction):
        """
        TODO
        """

        self.data_reduction = reduction_method

    def reduce(self, original_data, weight):
        """
        TODO
        """

        return self.data_reduction.__init__(original_data, weight)


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

    def __init__(self, n, reduction_method: DataReduction):

        super().__init__(reduction_method)
        self.n = n

    # TODO: functionality mostly in scalable_herding

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
