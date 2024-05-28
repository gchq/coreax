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

from abc import ABC, abstractmethod
from typing import Union

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from typing_extensions import deprecated

import coreax.kernel
import coreax.util
from coreax.data import Data


def _prepare_kernel_system(
    kernel: coreax.kernel.Kernel,
    x: Union[ArrayLike, Data],
    y: Union[ArrayLike, Data],
    epsilon: float = 1e-10,
    *,
    block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]] = None,
    unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
) -> tuple[Array, Array]:
    r"""
    Return the row mean of k(y,x) and the Gramian k(y,y).

    :param x: The original :math:`n \times d` data
    :param y: :math:`m \times d` representation of ``x``, e.g. a coreset
    :param epsilon: Small positive value to add to the kernel Gram matrix to aid
        numerical solver computations
    :param block_size: Block size passed to the ``self.kernel.compute_mean``
    :param unroll: Unroll parameter passed to ``self.kernel.compute_mean``
    :return: The row mean of k(y,x) and the epsilon perturbed Gramian k(y,y)
    """
    x = jnp.atleast_2d(jnp.asarray(x))
    y = jnp.atleast_2d(jnp.asarray(y))
    kernel_yx = kernel.compute_mean(y, x, axis=1, block_size=block_size, unroll=unroll)
    kernel_yy = kernel.compute(y, y) + epsilon * jnp.identity(len(y))
    return kernel_yx, kernel_yy


class WeightsOptimiser(ABC):
    """
    Base class for calculating weights.

    :param kernel: :class:`~coreax.kernel.Kernel` object
    """

    def __init__(self, kernel: coreax.kernel.Kernel) -> None:
        """Initialise a weights optimiser class."""
        self.kernel = kernel

    @abstractmethod
    def solve(self, x: Union[ArrayLike, Data], y: Union[ArrayLike, Data]) -> Array:
        r"""
        Calculate the weights.

        :param x: The original :math:`n \times d` data
        :param y: :math:`m \times d` representation of ``x``, e.g. a coreset
        :return: Optimal weighting of points in ``y`` to represent ``x``
        """


class SBQWeightsOptimiser(WeightsOptimiser):
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

    def solve(
        self,
        x: Union[ArrayLike, Data],
        y: Union[ArrayLike, Data],
        epsilon: float = 1e-10,
        *,
        block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]] = None,
        unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
    ) -> Array:
        r"""
        Calculate weights from Sequential Bayesian Quadrature (SBQ).

        References for this technique can be found in
        :cite:`huszar2016optimally`. These are equivalent to the unconstrained
        weighted maximum mean discrepancy (MMD) optimum.

        Note that weights determined through SBQ do not need to sum to 1, and can be
        negative.

        :param x: The original :math:`n \times d` data
        :param y: :math:`m \times d` representation of ``x``, e.g. a coreset
        :param epsilon: Small positive value to add to the kernel Gram matrix to aid
            numerical solver computations
        :param block_size: Block size passed to the ``self.kernel.compute_mean``
        :param unroll: Unroll parameter passed to ``self.kernel.compute_mean``
        :return: Optimal weighting of points in ``y`` to represent ``x``
        """
        kernel_yx, kernel_yy = _prepare_kernel_system(
            self.kernel, x, y, epsilon, block_size=block_size, unroll=unroll
        )
        return jnp.linalg.solve(kernel_yy, kernel_yx)


class MMDWeightsOptimiser(WeightsOptimiser):
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

    def solve(
        self,
        x: Union[ArrayLike, Data],
        y: Union[ArrayLike, Data],
        epsilon: float = 1e-10,
        *,
        block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]] = None,
        unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
    ) -> Array:
        r"""
        Compute optimal weights given the simplex constraint.

        :param x: The original :math:`n \times d` data
        :param y: :math:`m \times d` representation of ``x``, e.g. a coreset
        :param epsilon: Small positive value to add to the kernel Gram matrix to aid
            numerical solver computations
        :param block_size: Block size passed to the ``self.kernel.compute_mean``
        :param unroll: Unroll parameter passed to ``self.kernel.compute_mean``
        :return: Optimal weighting of points in ``y`` to represent ``x``
        """
        kernel_yx, kernel_yy = _prepare_kernel_system(
            self.kernel, x, y, epsilon, block_size=block_size, unroll=unroll
        )
        return coreax.util.solve_qp(kernel_yy, kernel_yx)


@deprecated("Renamed to SBQWeightsOptimiser; will be removed in version 0.3.0")
class SBQ(SBQWeightsOptimiser):
    """
    Deprecated reference to :class:`~coreax.weights.SBQWeightsOptimiser`.

    Will be removed in version 0.3.0
    """


@deprecated("Renamed to `MMDWeightsOptimiser`; will be removed in version 0.3.0")
class MMD(MMDWeightsOptimiser):
    """
    Deprecated reference to :class:`~coreax.weights.MMDWeightsOptimiser`.

    Will be removed in version 0.3.0
    """
