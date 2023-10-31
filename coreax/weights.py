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

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

import coreax.kernel as ck
from coreax.util import ClassFactory, solve_qp


class WeightsOptimiser(ABC):
    """
    Base class for calculating weights.
    """

    def __init__(self, kernel: ck.Kernel) -> None:
        r"""

        :param kernel: Kernel function
               :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
        """
        self.kernel = kernel

    @abstractmethod
    def solve(self, x: ArrayLike, y: ArrayLike) -> Array:
        """
        Calculate the weights.
        """


class SBQ(WeightsOptimiser):
    def __init__(self):
        # initialise parent
        super().__init__()

    def solve(self, x: ArrayLike, x_c: ArrayLike) -> Array:
        r"""
        Calculate weights from Sequential Bayesian Quadrature (SBQ).

        References for this technique can be found in
        [huszar2016optimallyweighted]_. These are equivalent to the unconstrained weighted
        maximum mean discrepancy (MMD) optimum.

        :param x: The original :math:`n \times d` data
        :param x_c: :math:`m times d` coreset
        :return: Optimal weights
        """
        x = jnp.asarray(x)
        x_c = jnp.asarray(x_c)
        kernel_nm = self.kernel.compute(x_c, x).sum(axis=1) / len(x)
        kernel_mm = self.kernel.compute(x_c, x_c) + 1e-10 * jnp.identity(len(x_c))
        return jnp.linalg.solve(kernel_mm, kernel_nm)


class MMD(WeightsOptimiser):
    def __init__(self):
        # initialise parent
        super().__init__()

    def solve(self, x: ArrayLike, x_c: ArrayLike) -> Array:
        r"""
        Compute optimal weights given the simplex constraint.

        :param x: The original :math:`n \times d` data
        :param x_c: :math:`m times d` coreset
        :return: Optimal weights
        """
        x = jnp.asarray(x)
        x_c = jnp.asarray(x_c)
        kernel_nm = self.kernel.compute(x_c, x).sum(axis=1) / len(x)
        kernel_mm = self.kernel.compute(x_c, x_c) + 1e-10 * jnp.identity(len(x_c))
        sol = solve_qp(kernel_mm, kernel_nm)
        return sol


# Set up class factory
weights_factory = ClassFactory(WeightsOptimiser)
weights_factory.register("SBQ", SBQ)
weights_factory.register("MMD", MMD)
