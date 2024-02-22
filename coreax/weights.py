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
Classes and associated functionality to optimise weighted representations of data.

Several aspects of this codebase take a :math:`n \times d` dataset and generate an
alternative representation of it, for example a coreset. The quality of this alternative
representation in approximating the original dataset can be assessed using some metric
of interest, for example see :class:`~coreax.metrics.Metric`.

One can improve the quality of the representation generated by weighting the individual
elements of it. These weights are determined by optimising the metric of interest, which
compares the original :math:`n \times d` dataset and the generated representation of it.

This module provides functionality to calculate such weights, through various methods.
All methods implement :class:`WeightsOptimiser` and must have a
:meth:`~WeightsOptimiser.solve` method that, given two datasets, returns an array of
weights such that a metric of interest is optimised when these weights are applied to
the dataset.
"""

import warnings
from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import coreax.kernel
import coreax.util
import coreax.validation


class WeightsOptimiser(ABC):
    """
    Base class for calculating weights.

    :param kernel: :class:`~coreax.kernel.Kernel` object
    """

    def __init__(self, kernel: coreax.kernel.Kernel) -> None:
        """Initialise a weights optimiser class."""
        coreax.validation.validate_is_instance(
            x=kernel, object_name="kernel", expected_type=coreax.kernel.Kernel
        )
        self.kernel = kernel

    @abstractmethod
    def solve(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Calculate the weights.

        :param x: The original :math:`n \times d` data
        :param y: :math:`m \times d` representation of ``x``, e.g. a coreset
        :return: Optimal weighting of points in ``y`` to represent ``x``
        """

    def solve_approximate(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Calculate approximate weights.

        :param x: The original :math:`n \times d` data
        :param y: :math:`m \times d` representation of ``x``, e.g. a coreset
        :return: Approximately optimal weighting of points in ``y`` to represent ``x``
        """
        warnings.warn(
            "solve_approximate() not yet implemented. "
            "Calculating exact solution via solve()"
        )
        return self.solve(x, y)


class SBQ(WeightsOptimiser):
    r"""
    Define the Sequential Bayesian Quadrature (SBQ) optimiser class.

    References for this technique can be found in :cite:`huszar2016optimally`.
    Weights determined by SBQ are equivalent to the unconstrained weighted maximum mean
    discrepancy (MMD) optimum.

    The Bayesian quadrature estimate of the integral

    .. math::

        \int f(x) p(x) dx

    can be viewed as a  weighted version of kernel herding. The Bayesian quadrature
    weights, :math:`w_{BQ}`, are given by

    .. math::

        w_{BQ}^{(n)} = \sum_m z_m^T K_{mn}^{-1}

    for a dataset :math:`x` with :math:`n` points, and coreset :math:`y` of :math:`m`
    points. Here, for given kernel :math:`k`, we have :math:`z = \int k(x, y)p(x) dx`
    and :math:`K = k(y, y)` in the above expression. See equation 20 in
    :cite:`huszar2016optimally` for further detail.

    :param kernel: :class:`~coreax.kernel.Kernel` object
    """

    def solve(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Calculate weights from Sequential Bayesian Quadrature (SBQ).

        References for this technique can be found in
        :cite:`huszar2016optimally`. These are equivalent to the unconstrained
        weighted maximum mean discrepancy (MMD) optimum.

        Note that weights determined through SBQ do not need to sum to 1, and can be
        negative.

        :param x: The original :math:`n \times d` data
        :param y: :math:`m \times d` representation of ``x``, e.g. a coreset
        :return: Optimal weighting of points in ``y`` to represent ``x``
        """
        # Validate inputs
        x = coreax.validation.cast_as_type(
            x=x, object_name="x", type_caster=jnp.atleast_2d
        )
        y = coreax.validation.cast_as_type(
            x=y, object_name="y", type_caster=jnp.atleast_2d
        )

        # Compute the components of the kernel matrix. Note that to ensure the solver
        # can numerically compute the result, we add a small perturbation to the kernel
        # matrix.
        kernel_nm = self.kernel.compute(y, x).sum(axis=1) / len(x)
        kernel_mm = self.kernel.compute(y, y) + 1e-10 * jnp.identity(len(y))

        # Solve for the optimal weights
        return jnp.linalg.solve(kernel_mm, kernel_nm)


class MMD(WeightsOptimiser):
    r"""
    Define the MMD weights optimiser class.

    This optimiser solves a simplex weight problem of the form:

    .. math::

        \mathbf{w}^{\mathrm{T}} \mathbf{k} \mathbf{w} +
        \bar{\mathbf{k}}^{\mathrm{T}} \mathbf{w} = 0

    subject to

    .. math::

        \mathbf{Aw} = \mathbf{1}, \qquad \mathbf{Gx} \le 0.

    using the OSQP quadratic programming solver.

    :param kernel: :class:`~coreax.kernel.Kernel` object
    """

    def solve(self, x: ArrayLike, y: ArrayLike, epsilon: float = 1e-10) -> Array:
        r"""
        Compute optimal weights given the simplex constraint.

        :param x: The original :math:`n \times d` data
        :param y: :math:`m \times d` representation of ``x``, e.g. a coreset
        :param epsilon: Small positive value to add to the kernel Gram matrix to aid
            numerical solver computations
        :return: Optimal weighting of points in ``y`` to represent ``x``
        """
        # Validate inputs
        x = coreax.validation.cast_as_type(
            x=x, object_name="x", type_caster=jnp.atleast_2d
        )
        y = coreax.validation.cast_as_type(
            x=y, object_name="y", type_caster=jnp.atleast_2d
        )
        epsilon = coreax.validation.cast_as_type(
            x=epsilon, object_name="epsilon", type_caster=float
        )
        coreax.validation.validate_in_range(
            x=epsilon, object_name="epsilon", strict_inequalities=False, lower_bound=0
        )

        # Compute the components of the kernel matrix. Note that to ensure the solver
        # can numerically compute the result, we add a small perturbation to the kernel
        # matrix.
        kernel_nm = self.kernel.compute(y, x).sum(axis=1) / len(x)
        kernel_mm = self.kernel.compute(y, y) + epsilon * jnp.identity(len(y))

        # Call the QP solver
        sol = coreax.util.solve_qp(kernel_mm, kernel_nm)

        return sol
