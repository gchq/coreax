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

"""Scalar-valued kernel functions."""

from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from typing_extensions import override

from coreax.kernels.base import ProductKernel, ScalarValuedKernel, UniCompositeKernel
from coreax.util import squared_distance


class LinearKernel(ScalarValuedKernel):
    r"""
    Define a linear kernel.

    Given :math:`\rho`=`'output_scale` and :math:`c =`'constant',  the linear kernel is
    defined as :math:`k: \mathbb{R}^d\times \mathbb{R}^d \to \mathbb{R}`,
    :math:`k(x, y) = \rho x^Ty + c`.

    :param output_scale: Kernel normalisation constant, :math:`\rho`
    :param constant: Additive constant, :math:`c`
    """

    output_scale: float = eqx.field(default=1.0, converter=float)
    constant: float = eqx.field(default=0.0, converter=float)

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return self.output_scale * jnp.dot(x, y) + self.constant

    @override
    def grad_x_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return self.output_scale * jnp.asarray(y)

    @override
    def grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return self.output_scale * jnp.asarray(x)

    @override
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        d = len(jnp.asarray(x))
        return jnp.array(self.output_scale * d)


class PolynomialKernel(ScalarValuedKernel):
    r"""
    Define a polynomial kernel.

    Given :math:`\rho =``output_scale`, :math:`c =`'constant', and :math:`d=`'degree',
    the polynomial kernel is defined as
    :math:`k: \mathbb{R}^d\times \mathbb{R}^d \to \mathbb{R}`,
    :math:`k(x, y) = \rho (x^Ty + c)^d`.

    :param output_scale: Kernel normalisation constant, :math:`\rho`
    :param constant: Additive constant, :math:`c`
    :param degree: Degree of kernel, must be a positive integer greater than 1.
    """

    output_scale: float = 1.0
    constant: float = 0.0
    degree: int = 2

    def __check_init__(self):
        """Ensure degree is an integer greater than 1."""
        min_degree = 2
        if not isinstance(self.degree, int) or self.degree < min_degree:
            raise ValueError("'degree' must be a positive integer greater than 1")

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return self.output_scale * (jnp.dot(x, y) + self.constant) ** self.degree

    @override
    def grad_x_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return (
            self.output_scale
            * self.degree
            * jnp.asarray(y)
            * (jnp.dot(x, y) + self.constant) ** (self.degree - 1)
        )

    @override
    def grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return (
            self.output_scale
            * self.degree
            * jnp.asarray(x)
            * (jnp.dot(x, y) + self.constant) ** (self.degree - 1)
        )

    @override
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        dot = jnp.dot(x, y)
        body = dot + self.constant
        d = len(jnp.asarray(x))

        return (
            self.output_scale
            * self.degree
            * (
                ((self.degree - 1) * dot * body ** (self.degree - 2))
                + (d * body ** (self.degree - 1))
            )
        )


class SquaredExponentialKernel(ScalarValuedKernel):
    r"""
    Define a squared exponential kernel.

    Given :math:`\lambda =``length_scale` and :math:`\rho =``output_scale`, the squared
    exponential kernel is defined as
    :math:`k: \mathbb{R}^d\times \mathbb{R}^d \to \mathbb{R}`,
    :math:`k(x, y) = \rho * \exp(-\frac{||x-y||^2}{2 \lambda^2})` where
    :math:`||\cdot||` is the usual :math:`L_2`-norm.

    :param length_scale: Kernel smoothing/bandwidth parameter, :math:`\lambda`
    :param output_scale: Kernel normalisation constant, :math:`\rho`
    """

    length_scale: float = eqx.field(default=1.0, converter=float)
    output_scale: float = eqx.field(default=1.0, converter=float)

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return self.output_scale * jnp.exp(
            -squared_distance(x, y) / (2 * self.length_scale**2)
        )

    @override
    def grad_x_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return -self.grad_y_elementwise(x, y)

    @override
    def grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return (
            jnp.subtract(x, y) / self.length_scale**2 * self.compute_elementwise(x, y)
        )

    @override
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        k = self.compute_elementwise(x, y)
        scale = 1 / self.length_scale**2
        d = len(jnp.asarray(x))
        return scale * k * (d - scale * squared_distance(x, y))


class ExponentialKernel(ScalarValuedKernel):
    r"""
    Define an exponential kernel.

    Given :math:`\lambda =``length_scale` and :math:`\rho =``output_scale`, the
    exponential kernel is defined as
    :math:`k: \mathbb{R}^d\times \mathbb{R}^d \to \mathbb{R}`,
    :math:`k(x, y) = \rho * \exp(-\frac{||x-y||}{2 \lambda^2})` where
    :math:`||\cdot||` is the usual :math:`L_2`-norm.

    .. note::
        Note that the Exponential kernel is not differentiable when :math:`x=y`.

    :param length_scale: Kernel smoothing/bandwidth parameter, :math:`\lambda`
    :param output_scale: Kernel normalisation constant, :math:`\rho`
    """

    length_scale: float = 1.0
    output_scale: float = 1.0

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return self.output_scale * jnp.exp(
            -jnp.linalg.norm(jnp.subtract(x, y)) / (2 * self.length_scale**2)
        )

    @override
    def grad_x_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return -self.grad_y_elementwise(x, y)

    @override
    def grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        sub = jnp.subtract(x, y)
        dist = jnp.linalg.norm(sub)
        factor = 2 * self.length_scale**2
        return self.output_scale * sub * jnp.exp(-dist / factor) / (factor * dist)

    @override
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        d = len(jnp.asarray(x))
        sub = jnp.subtract(x, y)
        dist = jnp.linalg.norm(sub)
        factor = 2 * self.length_scale**2
        exp = jnp.exp(-dist / factor)

        first_term = (-exp * sub / dist**2) * ((1 / dist) + 1 / factor)
        second_term = exp / dist

        return (self.output_scale / factor) * (
            jnp.dot(first_term, sub) + d * second_term
        )


class RationalQuadraticKernel(ScalarValuedKernel):
    r"""
    Define a rational quadratic kernel.

    Given :math:`\lambda =``length_scale`,  :math:`\rho =``output_scale`, and
    :math:`\alpha =``relative_weighting`, the rational quadratic kernel is defined as
    :math:`k: \mathbb{R}^d\times \mathbb{R}^d \to \mathbb{R}`,
    :math:`k(x, y) = \rho * (1 + \frac{||x-y||^2}{2 \alpha \lambda^2})^{-\alpha}` where
    :math:`||\cdot||` is the usual :math:`L_2`-norm.

    :param length_scale: Kernel smoothing/bandwidth parameter, :math:`\lambda`
    :param output_scale: Kernel normalisation constant, :math:`\rho`
    :param relative_weighting: Parameter controlling the relative weighting of
        large-scale and small-scale variations, :math:`\alpha`. As
        :math:`alpha \to \infty` the rational quadratic kernel is identical to the
        squared exponential kernel.
    """

    length_scale: float = 1.0
    output_scale: float = 1.0
    relative_weighting: float = 1.0

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return (
            self.output_scale
            * (
                1
                + squared_distance(x, y)
                / (2 * self.relative_weighting * self.length_scale**2)
            )
            ** -self.relative_weighting
        )

    @override
    def grad_x_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return -self.grad_y_elementwise(x, y)

    @override
    def grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return (self.output_scale * jnp.subtract(x, y) / self.length_scale**2) * (
            1
            + squared_distance(x, y)
            / (2 * self.relative_weighting * self.length_scale**2)
        ) ** (-self.relative_weighting - 1)

    @override
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        d = len(jnp.asarray(x))
        sq_dist = squared_distance(x, y)
        power = self.relative_weighting + 1
        div = self.relative_weighting * self.length_scale**2
        body = 1 + sq_dist / (2 * div)
        factor = self.output_scale / self.length_scale**2

        first_term = factor * body**-power
        second_term = -(factor * power * sq_dist / div) * body ** -(power + 1)
        return d * first_term + second_term


class PeriodicKernel(ScalarValuedKernel):
    r"""
    Define a periodic kernel.

    Given :math:`\lambda =``length_scale`,  :math:`\rho =``output_scale`, and
    :math:`\p =``periodicity`, the periodic kernel is defined as
    :math:`k: \mathbb{R}^d\times \mathbb{R}^d \to \mathbb{R}`,
    :math:`k(x, y) = \rho * \exp(\frac{-2 \sin^2(\pi ||x-y||/p)}{\lambda^2})` where
    :math:`||\cdot||` is the usual :math:`L_2`-norm.

    .. note::
        Note that the Periodic kernel is not differentiable when :math:`x=y`.

    :param length_scale: Kernel smoothing/bandwidth parameter, :math:`\lambda`
    :param output_scale: Kernel normalisation constant, :math:`\rho`
    :param periodicity: Parameter controlling the periodicity of the kernel. :\math: `p`
    """

    length_scale: float = 1.0
    output_scale: float = 1.0
    periodicity: float = 1.0

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return self.output_scale * (
            jnp.exp(
                -2
                * jnp.sin(
                    jnp.pi * jnp.linalg.norm(jnp.subtract(x, y)) / self.periodicity
                )
                ** 2
                / self.length_scale**2
            )
        )

    @override
    def grad_x_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return -self.grad_y_elementwise(x, y)

    @override
    def grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        dist = jnp.linalg.norm(jnp.subtract(x, y))
        body = jnp.pi * dist / self.periodicity
        return (
            (
                4
                * jnp.subtract(x, y)
                * self.output_scale
                * jnp.pi
                / (dist * self.periodicity * self.length_scale**2)
            )
            * jnp.sin(body)
            * jnp.cos(body)
            * jnp.exp(-(2 / self.length_scale**2) * jnp.sin(body) ** 2)
        )

    @override
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        d = len(jnp.asarray(x))
        sub = jnp.subtract(x, y)
        dist = jnp.linalg.norm(sub)
        factor = jnp.pi / self.periodicity
        func_body = factor * dist
        grad_factor = sub / dist
        output_factor = 4 * factor * self.output_scale / self.length_scale**2

        func_1 = 1 / dist
        func_2 = jnp.sin(func_body)
        func_3 = jnp.cos(func_body)
        func_4 = jnp.exp(-(2 / self.length_scale**2) * func_2**2)

        first_term = func_1 * func_2 * func_3 * func_4
        second_term = (
            -grad_factor * func_1**2 * func_2 * func_3 * func_4
            - grad_factor * factor * func_1 * func_2**2 * func_4
            + grad_factor * factor * func_1 * func_3**2 * func_4
            - (output_factor * sub * func_1**2 * func_2**2 * func_3**2 * func_4)
        )

        return output_factor * (d * first_term + jnp.dot(second_term, sub))


class LaplacianKernel(ScalarValuedKernel):
    r"""
    Define a Laplacian kernel.

    Given :math:`\lambda =``length_scale` and :math:`\rho =``output_scale`, the
    Laplacian kernel is defined as
    :math:`k: \mathbb{R}^d\times \mathbb{R}^d \to \mathbb{R}`,
    :math:`k(x, y) = \rho * \exp(-\frac{||x-y||_1}{2 \lambda^2})`  where
    :math:`||\cdot||_1` is the :math:`L_1`-norm.

    :param length_scale: Kernel smoothing/bandwidth parameter, :math:`\lambda`
    :param output_scale: Kernel normalisation constant, :math:`\rho`
    """

    length_scale: float = eqx.field(default=1.0, converter=float)
    output_scale: float = eqx.field(default=1.0, converter=float)

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return self.output_scale * jnp.exp(
            -jnp.linalg.norm(jnp.subtract(x, y), ord=1) / (2 * self.length_scale**2)
        )

    @override
    def grad_x_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return -self.grad_y_elementwise(x, y)

    @override
    def grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return (
            jnp.sign(jnp.subtract(x, y))
            / (2 * self.length_scale**2)
            * self.compute_elementwise(x, y)
        )

    @override
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        k = self.compute_elementwise(x, y)
        d = len(jnp.asarray(x))
        return -d * k / (4 * self.length_scale**4)


class PCIMQKernel(ScalarValuedKernel):
    r"""
    Define a pre-conditioned inverse multi-quadric (PCIMQ) kernel.

    Given :math:`\lambda =``length_scale` and :math:`\rho =``output_scale`, the
    PCIMQ kernel is defined as
    :math:`k: \mathbb{R}^d\times \mathbb{R}^d \to \mathbb{R}`,
    :math:`k(x, y) = \frac{\rho}{\sqrt{1 + \frac{||x-y||^2}{2 \lambda^2}}}
    where :math:`||\cdot||` is the usual :math:`L_2`-norm.

    :param length_scale: Kernel smoothing/bandwidth parameter, :math:`\lambda`
    :param output_scale: Kernel normalisation constant, :math:`\rho`
    """

    length_scale: float = eqx.field(default=1.0, converter=float)
    output_scale: float = eqx.field(default=1.0, converter=float)

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        scaling = 2 * self.length_scale**2
        mq_array = squared_distance(x, y) / scaling
        return self.output_scale / jnp.sqrt(1 + mq_array)

    @override
    def grad_x_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return -self.grad_y_elementwise(x, y)

    @override
    def grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return (
            self.output_scale
            * jnp.subtract(x, y)
            / (2 * self.length_scale**2)
            * (self.compute_elementwise(x, y) / self.output_scale) ** 3
        )

    @override
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        k = self.compute_elementwise(x, y) / self.output_scale
        scale = 2 * self.length_scale**2
        d = len(jnp.asarray(x))
        return (
            self.output_scale
            / scale
            * (d * k**3 - 3 * k**5 * squared_distance(x, y) / scale)
        )


class LocallyPeriodicKernel(ProductKernel):
    r"""
    Define a locally periodic kernel.

    The periodic kernel is defined as
    :math:`k: \mathbb{R}^d\times \mathbb{R}^d \to \mathbb{R}`,
    :math:`k(x, y) = r(x,y)l(x,y)` where :math:`r` is the periodic kernel and
    :math:`l` is the squared exponential kernel.

    .. note::
        Note that the Periodic kernel is not differentiable when :math:`x=y`.

    :param periodic_length_scale: Periodic kernel smoothing/bandwidth parameter
    :param periodic_output_scale: Periodic kernel normalisation constant
    :param periodicity: Parameter controlling the periodicity of the Periodic kernel
    :param squared_exponential_length_scale: SquaredExponential kernel
        smoothing/bandwidth parameter]
    :param squared_exponential_output_scale: SquaredExponential Kernel normalisation
        constant
    """

    def __init__(
        self,
        periodic_length_scale: float = 1.0,
        periodic_output_scale: float = 1.0,
        periodicity: float = 1.0,
        squared_exponential_length_scale: float = 1.0,
        squared_exponential_output_scale: float = 1.0,
    ):
        """Initialise LocallyPeriodicKernel with ProductKernel attributes."""
        self.first_kernel = PeriodicKernel(
            length_scale=periodic_length_scale,
            output_scale=periodic_output_scale,
            periodicity=periodicity,
        )

        self.second_kernel = SquaredExponentialKernel(
            length_scale=squared_exponential_length_scale,
            output_scale=squared_exponential_output_scale,
        )


class SteinKernel(UniCompositeKernel):
    r"""
    Define the Stein kernel, i.e. the application of the Stein operator.

    .. math::

        \mathcal{A}_\mathbb{P}(g(\mathbf{x})) := \nabla_\mathbf{x} g(\mathbf{x})
        + g(\mathbf{x}) \nabla_\mathbf{x} \log f_X(\mathbf{x})^\intercal

    w.r.t. probability measure :math:`\mathbb{P}` to the base kernel
    :math:`k(\mathbf{x}, \mathbf{y})`. Here, differentiable vector-valued
    :math:`g: \mathbb{R}^d \to \mathbb{R}^d`, and
    :math:`\nabla_\mathbf{x} \log f_X(\mathbf{x})` is the *score function* of measure
    :math:`\mathbb{P}`.

    :math:`\mathbb{P}` is assumed to admit a density function :math:`f_X` w.r.t.
    d-dimensional Lebesgue measure. The score function is assumed to be Lipschitz.

    The key property of a Stein operator is zero expectation under
    :math:`\mathbb{P}`, i.e.
    :math:`\mathbb{E}_\mathbb{P}[\mathcal{A}_\mathbb{P} f(\mathbf{x})]`, for
    positive differentiable :math:`f_X`.

    The Stein kernel for base kernel :math:`k(\mathbf{x}, \mathbf{y})` is defined as

    .. math::

        k_\mathbb{P}(\mathbf{x}, \mathbf{y}) = \nabla_\mathbf{x} \cdot
        \nabla_\mathbf{y}
        k(\mathbf{x}, \mathbf{y}) + \nabla_\mathbf{x} \log f_X(\mathbf{x})
        \cdot \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y}) + \nabla_\mathbf{y} \log
        f_X(\mathbf{y}) \cdot \nabla_\mathbf{x} k(\mathbf{x}, \mathbf{y}) +
        (\nabla_\mathbf{x} \log f_X(\mathbf{x}) \cdot \nabla_\mathbf{y} \log
        f_X(\mathbf{y})) k(\mathbf{x}, \mathbf{y}).

    This kernel requires a 'base' kernel to evaluate. The base kernel can be any
    other implemented subclass of the Kernel abstract base class; even another Stein
    kernel.

    The score function
    :math:`\nabla_\mathbf{x} \log f_X: \mathbb{R}^d \to \mathbb{R}^d` can be any
    suitable Lipschitz score function, e.g. one that is learned from score matching
    (:class:`~coreax.score_matching.ScoreMatching`), computed explicitly from a density
    function, or known analytically.

    :param base_kernel: Initialised kernel object with which to evaluate
        the Stein kernel
    :param score_function: A vector-valued callable defining a score function
        :math:`\mathbb{R}^d \to \mathbb{R}^d`
    """

    score_function: Callable[[ArrayLike], Array]

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        k = self.base_kernel.compute_elementwise(x, y)
        div = self.base_kernel.divergence_x_grad_y_elementwise(x, y)
        gkx = self.base_kernel.grad_x_elementwise(x, y)
        gky = self.base_kernel.grad_y_elementwise(x, y)
        score_x = self.score_function(x)
        score_y = self.score_function(y)
        return div + gkx @ score_y + gky @ score_x + k * score_x @ score_y