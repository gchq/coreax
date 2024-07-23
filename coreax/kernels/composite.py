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

"""Compositions for kernels."""

import equinox as eqx
from jax import Array
from jax.typing import ArrayLike
from typing_extensions import override

from coreax.kernels.base import ScalarValuedKernel


class UniCompositeKernel(eqx.Module):
    """
    Abstract base class for kernels that compose/wrap one scalar-valued kernel.

    :param base_kernel: kernel to be wrapped/used in composition
    """

    base_kernel: ScalarValuedKernel

    def __check_init__(self):
        """Check that 'base_kernel' is of the required type."""
        if not isinstance(self.base_kernel, ScalarValuedKernel):
            raise ValueError(
                "'base_kernel' must be an instance of "
                + f"'{ScalarValuedKernel.__module__}.{ScalarValuedKernel.__qualname__}'"
            )


class DuoCompositeKernel(eqx.Module):
    """
    Abstract base class for kernels that compose/wrap two scalar-valued kernels.

    :param first_kernel: Instance of :class:`ScalarValuedKernel`
    :param second_kernel: Instance of :class:`ScalarValuedKernel`
    """

    first_kernel: ScalarValuedKernel
    second_kernel: ScalarValuedKernel

    def __check_init__(self):
        """Ensure attributes are instances of Kernel class."""
        if not (
            isinstance(self.first_kernel, ScalarValuedKernel)
            and isinstance(self.second_kernel, ScalarValuedKernel)
        ):
            raise ValueError(
                "'first_kernel'and `second_kernel` must be an instance of "
                + f"'{ScalarValuedKernel.__module__}.{ScalarValuedKernel.__qualname__}'"
            )


class AdditiveKernel(DuoCompositeKernel, ScalarValuedKernel):
    r"""
    Define a kernel which is a summation of two kernels.

    Given kernel functions :math:`k:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}` and
    :math:`l:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}`, define the additive
    kernel :math:`p:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}` where
    :math:`p(x,y) := k(x,y) + l(x,y)`

    :param first_kernel: Instance of :class:`Kernel`
    :param second_kernel: Instance of :class:`Kernel`
    """

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return self.first_kernel.compute_elementwise(
            x, y
        ) + self.second_kernel.compute_elementwise(x, y)

    @override
    def grad_x_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return self.first_kernel.grad_x_elementwise(
            x, y
        ) + self.second_kernel.grad_x_elementwise(x, y)

    @override
    def grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return self.first_kernel.grad_y_elementwise(
            x, y
        ) + self.second_kernel.grad_y_elementwise(x, y)

    @override
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return self.first_kernel.divergence_x_grad_y_elementwise(
            x, y
        ) + self.second_kernel.divergence_x_grad_y_elementwise(x, y)


class ProductKernel(DuoCompositeKernel, ScalarValuedKernel):
    r"""
    Define a kernel which is a product of two kernels.

    Given kernel functions :math:`k:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}` and
    :math:`l:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}`, define the product kernel
    :math:`p:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}` where
    :math:`p(x,y) = k(x,y)l(x,y)`

    :param first_kernel: Instance of :class:`Kernel`
    :param second_kernel: Instance of :class:`Kernel`
    """

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        if self.first_kernel == self.second_kernel:
            return self.first_kernel.compute_elementwise(x, y) ** 2
        return self.first_kernel.compute_elementwise(
            x, y
        ) * self.second_kernel.compute_elementwise(x, y)

    @override
    def grad_x_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        if self.first_kernel == self.second_kernel:
            return (
                2
                * self.first_kernel.grad_x_elementwise(x, y)
                * self.first_kernel.compute_elementwise(x, y)
            )
        return self.first_kernel.grad_x_elementwise(
            x, y
        ) * self.second_kernel.compute_elementwise(
            x, y
        ) + self.second_kernel.grad_x_elementwise(
            x, y
        ) * self.first_kernel.compute_elementwise(x, y)

    @override
    def grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        if self.first_kernel == self.second_kernel:
            return (
                2
                * self.first_kernel.grad_y_elementwise(x, y)
                * self.first_kernel.compute_elementwise(x, y)
            )
        return self.first_kernel.grad_y_elementwise(
            x, y
        ) * self.second_kernel.compute_elementwise(
            x, y
        ) + self.second_kernel.grad_y_elementwise(
            x, y
        ) * self.first_kernel.compute_elementwise(x, y)

    @override
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        if self.first_kernel == self.second_kernel:
            return 2 * (
                self.first_kernel.grad_x_elementwise(x, y).dot(
                    self.first_kernel.grad_y_elementwise(x, y)
                )
                + self.first_kernel.compute_elementwise(x, y)
                * self.first_kernel.divergence_x_grad_y_elementwise(x, y)
            )
        return (
            self.first_kernel.grad_x_elementwise(x, y).dot(
                self.second_kernel.grad_y_elementwise(x, y)
            )
            + self.first_kernel.grad_y_elementwise(x, y).dot(
                self.second_kernel.grad_x_elementwise(x, y)
            )
            + self.first_kernel.compute_elementwise(x, y)
            * self.second_kernel.divergence_x_grad_y_elementwise(x, y)
            + self.second_kernel.compute_elementwise(x, y)
            * self.first_kernel.divergence_x_grad_y_elementwise(x, y)
        )


# pylint: disable=abstract-method
class MagicScalarValuedKernel(ScalarValuedKernel):
    """Abstract base class for scalar-valued kernels with implemented magic methods."""

    def __add__(self, addition: ScalarValuedKernel) -> AdditiveKernel:
        """Overload `+` operator."""
        if isinstance(addition, MagicScalarValuedKernel):
            return AdditiveKernel(self, addition)
        return NotImplemented

    def __radd__(self, addition: ScalarValuedKernel) -> AdditiveKernel:
        """Overload right `+` operator, order is mathematically irrelevant."""
        return self.__add__(addition)

    def __mul__(self, product: ScalarValuedKernel) -> ProductKernel:
        """Overload `*` operator."""
        if isinstance(product, MagicScalarValuedKernel):
            return ProductKernel(self, product)
        return NotImplemented

    def __rmul__(self, product: ScalarValuedKernel) -> ProductKernel:
        """Overload right `*` operator, order is mathematically irrelevant."""
        return self.__mul__(product)

    def __pow__(self, power: int) -> ProductKernel:
        """Overload `**` operator."""
        min_power = 2
        power = int(power)
        if power < min_power:
            raise ValueError("'power' must be an integer greater than or equal to 2.")

        first_kernel = self
        second_kernel = self

        # Ensure the first and second kernels are symmetric for even powers to make use
        # of reduced computation capabilities in ProductKernel. For example,
        # :meth:`~divergence_x_grad_y_elementwise` can be computed more efficiently if
        # the first and second kernels are recognised as the same.
        for i in range(min_power, power):
            if i % 2 == 0:
                first_kernel = ProductKernel(first_kernel, self)
            else:
                second_kernel = ProductKernel(second_kernel, self)

        return ProductKernel(first_kernel, second_kernel)


# pylint: enable=abstract-method
