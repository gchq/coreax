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

import warnings
from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import coreax.kernel as ck
from coreax.util import ClassFactory, solve_qp


class WeightsOptimiser(ABC):
    """
    Base class for calculating weights.

    :param kernel: Kernel object
    """

    def __init__(self, kernel: ck.Kernel) -> None:
        r"""
        Initilise a weights optimiser class.

        # TODO: Does this need to take in a DataReduction object that has kernel attached to it?
        """
        self.kernel = kernel

    @abstractmethod
    def solve(self, x: ArrayLike, y: ArrayLike) -> Array:
        """
        Calculate the weights.

        :param x: The original :math:`n \times d` data
        :param y: :math:`m times d` representation of ``x``, e.g. a coreset
        :return: Optimal weighting of points in ``y`` to represent ``x``
        """

    def solve_approximate(self, x: ArrayLike, y: ArrayLike) -> Array:
        """
        Calculate approximate weights.
        """
        warnings.warn(
            "solve_approximate() not yet implemented. "
            "Calculating exact solution via solve()"
        )
        return self.solve(x, y)


class SBQ(WeightsOptimiser):
    """
    Define the Sequential Bayesian Quadrature (SBQ) optimiser class.

    References for this technique can be found in :cite:p:`huszar2016optimallyweighted`.
    Weighted determined by SBQ are equivalent to the unconstrained weighted maximum mean
    discrepancy (MMD) optimum.

    :param kernel: Kernel object
    """

    def __init__(self, kernel: ck.Kernel) -> None:
        """
        Initilise a Sequential Bayesian Quadrature (SBQ) optimiser class.
        """
        # initialise parent
        super().__init__(kernel)

    def solve(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Calculate weights from Sequential Bayesian Quadrature (SBQ).

        References for this technique can be found in
        :cite:p:`huszar2016optimallyweighted`. These are equivalent to the unconstrained
        weighted maximum mean discrepancy (MMD) optimum.

        Note that weights determined through SBQ do not need to sum to 1, and can be
        negative.

        Optimal weights in this sense

        :param x: The original :math:`n \times d` data
        :param y: :math:`m times d` representation of ``x``, e.g. a coreset
        :return: Optimal weighting of points in ``y`` to represent ``x``
        """
        # Format data
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        # Compute the components of the kernel matrix. Note that to ensure the solver
        # can numerically compute the result, we add a small perturbation to the kernel
        # matrix.
        kernel_nm = self.kernel.compute(y, x).sum(axis=1) / len(x)
        kernel_mm = self.kernel.compute(y, y) + 1e-10 * jnp.identity(len(y))

        # Solve for the optimal weights
        return jnp.linalg.solve(kernel_mm, kernel_nm)


class MMD(WeightsOptimiser):
    """
    Define the MMD weights optimiser class.

    This optimser solves a simplex weight problem of the form:

    .. math::

        \mathbf{w}^{\mathrm{T}} \mathbf{k} \mathbf{w} + \bar{\mathbf{k}}^{\mathrm{T}} \mathbf{w} = 0

    subject to

    .. math::

        \mathbf{Aw} = \mathbf{1}, \qquad \mathbf{Gx} \le 0.

    using the OSQP quadratic programming solver.

    :param kernel: Kernel object
    """

    def __init__(self, kernel: ck.Kernel) -> None:
        """
        Initilise a Sequential Bayesian Quadrature (SBQ) optimiser class.
        """
        # initialise parent
        super().__init__(kernel)

    def solve(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Compute optimal weights given the simplex constraint.

        :param x: The original :math:`n \times d` data
        :param y: :math:`m times d` representation of ``x``, e.g. a coreset
        :return: Optimal weighting of points in ``y`` to represent ``x``
        """
        # Format data
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        # Compute the components of the kernel matrix. Note that to ensure the solver
        # can numerically compute the result, we add a small perturbation to the kernel
        # matrix.
        kernel_nm = self.kernel.compute(y, x).sum(axis=1) / len(x)
        kernel_mm = self.kernel.compute(y, y) + 1e-10 * jnp.identity(len(y))

        # Call the QP solver
        sol = solve_qp(kernel_mm, kernel_nm)

        return sol


# Set up class factory
weights_factory = ClassFactory(WeightsOptimiser)
weights_factory.register("SBQ", SBQ)
weights_factory.register("MMD", MMD)
