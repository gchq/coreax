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
Classes and associated functionality to use kernel functions.

A kernel is a non-negative, real-valued integrable function that can take two inputs,
``x`` and ``y``, and returns a value that decreases as ``x`` and ``y`` move further away
in space from each other. Note that *further* here may account for cyclic behaviour in
the data, for example.

In this library, we often use kernels as a smoothing tool: given a dataset of distinct
points, we can reconstruct the underlying data generating distribution through smoothing
of the data with kernels.

Some kernels are parameterizable and may represent other well known kernels when given
appropriate parameter values. For example, the :class:`SquaredExponentialKernel`,

.. math::

    k(x,y) = \text{output_scale} * \exp (-||x-y||^2/2 * \text{length_scale}^2),

which if parameterized with an ``output_scale`` of
:math:`\frac{1}{\sqrt{2\pi} \,*\, \text{length_scale}}`, yields the well known
Gaussian kernel.

A :class:`~coreax.kernel.Kernel` must implement a
:meth:`~coreax.kernel.Kernel.compute_elementwise` method, which returns the floating
point value after evaluating the kernel on two floats, ``x`` and ``y``. Additional
methods, such as :meth:`Kernel.grad_x_elementwise`, can optionally be overridden to
improve performance. The canonical example is when a suitable closed-form representation
of a higher-order gradient can be used to avoid the expense of performing automatic
differentiation.

Such an example can be seen in :class:`SteinKernel`, where the analytic forms of
:meth:`Kernel.divergence_x_grad_y` are significantly cheaper to compute that the
automatic differentiated default.
"""

# Support annotations with | in Python < 3.10
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, grad, jacrev, jit
from jax.typing import ArrayLike
from typing_extensions import override

from coreax.util import pairwise, squared_distance

T = TypeVar("T")


@jit
def median_heuristic(x: ArrayLike) -> Array:
    """
    Compute the median heuristic for setting kernel bandwidth.

    Analysis of the performance of the median heuristic can be found in
    :cite:`garreau2018median`.

    :param x: Input array of vectors
    :return: Bandwidth parameter, computed from the median heuristic, as a
        zero-dimensional array
    """
    # Format inputs
    x = jnp.atleast_2d(x)
    # Calculate square distances as an upper triangular matrix
    square_distances = jnp.triu(pairwise(squared_distance)(x, x), k=1)
    # Calculate the median of the square distances
    median_square_distance = jnp.median(
        square_distances[jnp.triu_indices_from(square_distances, k=1)]
    )

    return jnp.sqrt(median_square_distance / 2.0)


class Kernel(eqx.Module):
    """Abstract base class for kernels."""

    @eqx.filter_jit
    def compute(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Evaluate the kernel on input data ``x`` and ``y``.

        The 'data' can be any of:
            * floating numbers (so a single data-point in 1-dimension)
            * zero-dimensional arrays (so a single data-point in 1-dimension)
            * a vector (a single-point in multiple dimensions)
            * array (multiple vectors).

        Evaluation is always vectorised.

        :param x: An :math:`n \times d` dataset (array) or a single value (point)
        :param y: An :math:`m \times d` dataset (array) or a single value (point)
        :return: Kernel evaluations between points in ``x`` and ``y``. If ``x`` = ``y``,
            then this is the Gram matrix corresponding to the RKHS inner product.
        """
        return pairwise(self.compute_elementwise)(x, y)

    @abstractmethod
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Evaluate the kernel on individual input vectors ``x`` and ``y``, not-vectorised.

        Vectorisation only becomes relevant in terms of computational speed when we
        have multiple ``x`` or ``y``.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Kernel evaluated at (``x``, ``y``)
        """

    def grad_x(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Evaluate the gradient (Jacobian) of the kernel function w.r.t. ``x``.

        The function is vectorised, so ``x`` or ``y`` can be any of:
            * floating numbers (so a single data-point in 1-dimension)
            * zero-dimensional arrays (so a single data-point in 1-dimension)
            * a vector (a single-point in multiple dimensions)
            * array (multiple vectors).

        :param x: An :math:`n \times d` dataset (array) or a single value (point)
        :param y: An :math:`m \times d` dataset (array) or a single value (point)
        :return: An :math:`n \times m \times d` array of pairwise Jacobians
        """
        return pairwise(self.grad_x_elementwise)(x, y)

    def grad_y(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Evaluate the gradient (Jacobian) of the kernel function w.r.t. ``y``.

        The function is vectorised, so ``x`` or ``y`` can be any of:
            * floating numbers (so a single data-point in 1-dimension)
            * zero-dimensional arrays (so a single data-point in 1-dimension)
            * a vector (a single-point in multiple dimensions)
            * array (multiple vectors).

        :param x: An :math:`n \times d` dataset (array) or a single value (point)
        :param y: An :math:`m \times d` dataset (array) or a single value (point)
        :return: An :math:`m \times n \times d` array of pairwise Jacobians
        """
        return pairwise(self.grad_y_elementwise)(x, y)

    def grad_x_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Evaluate the element-wise gradient of the kernel function w.r.t. ``x``.

        The gradient (Jacobian) of the kernel function w.r.t. ``x``.

        Only accepts single vectors ``x`` and ``y``, i.e. not arrays.
        :meth:`coreax.kernel.Kernel.grad_x` provides a vectorised version of this method
        for arrays.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Jacobian
            :math:`\nabla_\mathbf{x} k(\mathbf{x}, \mathbf{y}) \in \mathbb{R}^d`
        """
        return grad(self.compute_elementwise, 0)(x, y)

    def grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Evaluate the element-wise gradient of the kernel function w.r.t. ``y``.

        The gradient (Jacobian) of the kernel function w.r.t. ``y``.

        Only accepts single vectors ``x`` and ``y``, i.e. not arrays.
        :meth:`coreax.kernel.Kernel.grad_y` provides a vectorised version of this method
        for arrays.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`.
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`.
        :return: Jacobian
            :math:`\nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y}) \in \mathbb{R}^d`
        """
        return grad(self.compute_elementwise, 1)(x, y)

    def divergence_x_grad_y(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Evaluate the divergence operator w.r.t. ``x`` of Jacobian w.r.t. ``y``.

        :math:`\nabla_\mathbf{x} \cdot \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.
        This function is vectorised, so it accepts vectors or arrays.

        This is the trace of the 'pseudo-Hessian', i.e. the trace of the Jacobian matrix
        :math:`\nabla_\mathbf{x} \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.

        :param x: First vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Second vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Array of Laplace-style operator traces :math:`n \times m` array
        """
        return pairwise(self.divergence_x_grad_y_elementwise)(x, y)

    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Evaluate the element-wise divergence w.r.t. ``x`` of Jacobian w.r.t. ``y``.

        :math:`\nabla_\mathbf{x} \cdot \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.
        Only accepts vectors ``x`` and ``y``. A vectorised version for arrays is
        computed in :meth:`~coreax.kernel.Kernel.divergence_x_grad_y`.

        This is the trace of the 'pseudo-Hessian', i.e. the trace of the Jacobian matrix
        :math:`\nabla_\mathbf{x} \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.

        :param x: First vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Second vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Trace of the Laplace-style operator; a real number
        """
        pseudo_hessian = jacrev(self.grad_y_elementwise, 0)(x, y)
        return pseudo_hessian.trace()

    def gramian_row_mean(
        self,
        x: ArrayLike,
        *,
        block_size: int | None = None,
        unroll: tuple[int | bool, int | bool] = (1, 1),
    ) -> Array:
        r"""
        Compute the (blocked) row-wise mean of the kernel's Gramian matrix.

        The Gramian is a symmetric matrix :math:`G_{ij} = K(x_i, x_j)`, whose 'row-mean'
        is given by the vector :math:`\frac{1}{n}\sum_{i=1}^{n} G_{ij}`, where ``K`` is
        a :class:`~coreax.kernel.Kernel` and ``x`` is a :math:`n \times d` data matrix.

        .. note:
            Because the Gramian is symmetric, the 'row-mean' and 'column-mean' are
            equivalent. Use of the name 'row-mean' here is purley by convention.

        To avoid materializing the entire Gramian (memory cost :math:`\mathcal{O}(n^2)),
        we accumulate the mean in blocks (memory cost :math:`\mathcal{O}(B^2)`, where
        ``B`` is a user-specified block size).

        :param x: Data matrix, :math:`n \times d`
        :param block_size: Size of Gramian blocks to process; a value of `None` implies
            a block size equal to :math:`n`; a value that is not an integer divisor of
            :math:`n` yields :math:`\text{floor}(n / B)` blocks of size ``B`` and a
            final block of size `n - \text{floor}(n / B)`; to reduce overheads, select
            the largest integer multiple block size which does not exhaust the available
            memory resources
        :param unroll: Unrolling parameter for the outer and inner :func:`jax.lax.scan`
            calls, allows for trade-offs between compilation and runtime cost; consult
            the JAX docs for further information
        :return: Gramian 'row/column-mean', :math:`\frac{1}{n}\sum_{i=1}^{n} G_{ij}`.
        """
        x = jnp.atleast_2d(x)
        num_data_points = x.shape[0]
        if block_size is None:
            _block_size = num_data_points
        else:
            # Clamp 'block_size' to [1, num_data_points]. Explicit cast will raise an
            # error if 'block_size' cannot be interpreted as an int.
            _block_size = min(max(1, abs(int(block_size))), num_data_points)
        num_blocks = num_data_points // _block_size
        split_point = num_blocks * _block_size
        block_iterable_x, trailing_x = x[:split_point], x[split_point:]
        try:
            block_x = block_iterable_x.reshape(num_blocks, -1, x.shape[1])
        except ZeroDivisionError as err:
            if x.size == 0:
                raise ValueError("'x' must not be empty") from err
            raise
        outer_unroll, inner_unroll = unroll
        requires_trailing_block = split_point < num_data_points

        def outer_loop(_: T, outer_x: Array) -> tuple[T, Array]:
            """Block the outer argument of :math:`K(x, x_{b_2})`."""

            def inner_loop(_: T, inner_x: Array) -> tuple[T, Array]:
                """Block the inner argument of :math:`K(x_{b_1}, x_{b_2})`."""
                gramian_block = self.compute(inner_x, outer_x)
                return _, jnp.sum(gramian_block, axis=0)

            _, row_sum = jax.lax.scan(inner_loop, _, block_x, unroll=inner_unroll)
            if requires_trailing_block:
                _, trailing_row_sum = inner_loop(_, trailing_x)
                row_sum = jnp.r_[row_sum, trailing_row_sum[None, ...]]
            return _, jnp.sum(row_sum, axis=0)

        # Compute the Gramian row sum over the 'block' iterable part of 'x'.
        _, block_row_sum = jax.lax.scan(outer_loop, None, block_x, unroll=outer_unroll)
        # If 'block_size' is not an integer divisor of 'num_data_points', there remains
        # a differently sized 'trailing block' which must be handled outside the scan.
        if requires_trailing_block:
            _, trailing_block_row_sum = outer_loop(_, trailing_x)
            block_row_sum = jnp.r_[block_row_sum.reshape(-1), trailing_block_row_sum]
        row_sum = jnp.sum(block_row_sum.reshape(-1, num_data_points), axis=0)
        return row_sum / num_data_points


class LinearKernel(Kernel):
    r"""
    Define a linear kernel.

    The linear kernel is defined as :math:`k: \mathbb{R}^d\times \mathbb{R}^d
    \to \mathbb{R}`, :math:`k(x, y) = x^Ty`.
    """

    length_scale: float = 1.0
    output_scale: float = 1.0

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return jnp.dot(x, y)

    @override
    def grad_x_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return jnp.asarray(y)

    @override
    def grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return jnp.asarray(x)

    @override
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return jnp.asarray(x).shape[0]


class SquaredExponentialKernel(Kernel):
    r"""
    Define a squared exponential kernel.

    Given :math:`\lambda =`'length_scale' and :math:`\rho =`'output_scale', the squared
    exponential kernel is defined as
    :math:`k: \mathbb{R}^d\times \mathbb{R}^d \to \mathbb{R}`,
    :math:`k(x, y) = \rho \exp(\frac{||x-y||^2}{2 \lambda^2})` where
    :math:`||\cdot||` is the usual :math:`L_2`-norm.

    :param length_scale: Kernel smoothing/bandwidth parameter, :math:`\lambda`
    :param output_scale: Kernel normalisation constant, :math:`\rho`
    """

    length_scale: float = 1.0
    output_scale: float = 1.0

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
        return (x - y) / self.length_scale**2 * self.compute_elementwise(x, y)

    @override
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        k = self.compute_elementwise(x, y)
        scale = 1 / self.length_scale**2
        d = len(x)
        return scale * k * (d - scale * squared_distance(x, y))


class LaplacianKernel(Kernel):
    r"""
    Define a Laplacian kernel.

    Given :math:`\lambda =`'length_scale' and :math:`\rho =`'output_scale', the
    Laplacian kernel is defined as
    :math:`k: \mathbb{R}^d\times \mathbb{R}^d \to \mathbb{R}`,
    :math:`k(x, y) = \rho * \exp(\frac{||x-y||_1}{2 \lambda^2})`  where
    :math:`||\cdot||_1` is the :math:`L_1`-norm.

    :param length_scale: Kernel smoothing/bandwidth parameter, :math:`\lambda`
    :param output_scale: Kernel normalisation constant, :math:`\rho`
    """

    length_scale: float = 1.0
    output_scale: float = 1.0

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return self.output_scale * jnp.exp(
            -jnp.linalg.norm(x - y, ord=1) / (2 * self.length_scale**2)
        )

    @override
    def grad_x_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return -self.grad_y_elementwise(x, y)

    @override
    def grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return (
            jnp.sign(x - y)
            / (2 * self.length_scale**2)
            * self.compute_elementwise(x, y)
        )

    @override
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        k = self.compute_elementwise(x, y)
        d = len(x)
        return -d * k / (4 * self.length_scale**4)


class PCIMQKernel(Kernel):
    r"""
    Define a pre-conditioned inverse multi-quadric (PCIMQ) kernel.

    Given :math:`\lambda =`'length_scale' and :math:`\rho =`'output_scale', the
    PCIMQ kernel is defined as
    :math:`k: \mathbb{R}^d\times \mathbb{R}^d \to \mathbb{R}`,
    :math:`k(x, y) = \frac{\rho}{\sqrt{1 + \frac{||x-y||^2}{2 \lambda^2}}}
    where :math:`||\cdot||` is the usual :math:`L_2`-norm.

    :param length_scale: Kernel smoothing/bandwidth parameter, :math:`\lambda`
    :param output_scale: Kernel normalisation constant, :math:`\rho`
    """

    length_scale: float = 1.0
    output_scale: float = 1.0

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
            * (x - y)
            / (2 * self.length_scale**2)
            * (self.compute_elementwise(x, y) / self.output_scale) ** 3
        )

    @override
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        k = self.compute_elementwise(x, y) / self.output_scale
        scale = 2 * self.length_scale**2
        d = len(x)
        return (
            self.output_scale
            / scale
            * (d * k**3 - 3 * k**5 * squared_distance(x, y) / scale)
        )


class CompositeKernel(Kernel):
    """
    Abstract base class for kernels that compose/wrap another kernel.

    :param base_kernel: kernel to be wrapped/used in composition
    """

    base_kernel: Kernel

    def __check_init__(self):
        """Check that 'base_kernel' is of the required type."""
        if not isinstance(self.base_kernel, Kernel):
            raise ValueError(
                "'base_kernel' must be an instance of "
                + f"'{Kernel.__module__}.{Kernel.__qualname__}'"
            )


class SteinKernel(CompositeKernel):
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
    :param output_scale: Kernel normalisation constant
    """

    score_function: Callable[[ArrayLike], Array]
    output_scale: float = 1.0

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        k = self.base_kernel.compute_elementwise(x, y)
        div = self.base_kernel.divergence_x_grad_y_elementwise(x, y)
        gkx = self.base_kernel.grad_x_elementwise(x, y)
        gky = self.base_kernel.grad_y_elementwise(x, y)
        score_x = self.score_function(x)
        score_y = self.score_function(y)
        return (
            div
            + jnp.dot(gkx, score_y)
            + jnp.dot(gky, score_x)
            + k * jnp.dot(score_x, score_y)
        )
