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
from math import ceil
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array, grad, jacrev, jit
from jax.typing import ArrayLike
from typing_extensions import override

from coreax.data import Data, is_data
from coreax.util import pairwise, squared_distance, tree_leaves_repeat

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

    def __add__(self, addition: Kernel | int | float):
        """Overload `+` operator."""
        if not isinstance(addition, (Kernel, int, float)):
            return NotImplemented
        if isinstance(addition, Kernel):
            return AdditiveKernel(self, addition)
        if isinstance(addition, (int, float)):
            return AdditiveKernel(self, LinearKernel(0, addition))
        return None

    def __radd__(self, addition: Kernel | int | float):
        """Overload right `+` operator, order is mathematically irrelevant."""
        return self.__add__(addition)

    def __mul__(self, product: Kernel | int | float):
        """Overload `*` operator."""
        if not isinstance(product, (Kernel, int, float)):
            return NotImplemented
        if isinstance(product, Kernel):
            return ProductKernel(self, product)
        if isinstance(product, (int, float)):
            return ProductKernel(self, LinearKernel(0, product))
        return None

    def __rmul__(self, product: Kernel | int | float):
        """Overload right `*` operator, order is mathematically irrelevant."""
        return self.__mul__(product)

    def __pow__(self, power: int):
        """
        Overload `**` operator.

        I ensure the first and second kernels are symmetric for even powers to make use
        of reduced computation capabilities in ProductKernel.
        """
        min_power = 2
        if not isinstance(power, int) or power < min_power:
            raise ValueError("'power' must be an integer greater than 1.")

        first_kernel = self
        second_kernel = self
        for i in range(min_power, power):
            if i % 2 == 0:
                first_kernel = ProductKernel(first_kernel, self)
            else:
                second_kernel = ProductKernel(second_kernel, self)

        return ProductKernel(first_kernel, second_kernel)

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
        x: ArrayLike | Data,
        *,
        block_size: int | None = None,
        unroll: tuple[int | bool, int | bool] = (1, 1),
    ):
        r"""
        Compute the (blocked) row-mean of the kernel's Gramian matrix.

        A convenience method for calling meth:`compute_mean`. Equivalent to the call
        :code:`compute_mean(x, x, axis=0, block_size=block_size, unroll=unroll)`.

        :param x: Data matrix, :math:`n \times d`
        :param block_size: Size of matrix blocks to process; a value of :data:`None`
            sets :math:`B_x = n`, effectively disabling the block accumulation; an
            integer value ``B`` sets :math:`B_x = B`; to reduce overheads, it is often
            sensible to select the largest block size that does not exhaust the
            available memory resources
        :param unroll: Unrolling parameter for the outer and inner :func:`jax.lax.scan`
            calls, allows for trade-offs between compilation and runtime cost; consult
            the JAX docs for further information
        :return: Gramian 'row/column-mean', :math:`\frac{1}{n}\sum_{i=1}^{n} G_{ij}`.
        """
        return self.compute_mean(x, x, axis=0, block_size=block_size, unroll=unroll)

    def compute_mean(
        self,
        x: ArrayLike | Data,
        y: ArrayLike | Data,
        axis: int | None = None,
        *,
        block_size: int | None | tuple[int | None, int | None] = None,
        unroll: int | bool | tuple[int | bool, int | bool] = 1,
    ) -> Array:
        r"""
        Compute the (blocked) mean of the matrix :math:`K_{ij} = k(x_i, y_j)`.

        The :math:`n \times m` kernel matrix :math:`K_{ij} = k(x_i, y_j)`, where
        ``x`` and ``y`` are respectively :math:`n \times d` and :math:`m \times d`
        (weighted) data matrices, has the following (weighted) means:

        - mean (:code:`axis=None`) :math:`\frac{1}{n m}\sum_{i,j=1}^{n, m} K_{ij}`
        - row-mean (:code:`axis=0`) :math:`\frac{1}{n}\sum_{i=1}^{n} K_{ij}`
        - column-mean (:code:`axis=1`) :math:`\frac{1}{m}\sum_{j=1}^{m} K_{ij}`

        If ``x`` and ``y`` are of type :class:`~coreax.data.Data`, their weights are
        used to compute the weighted mean as defined in :func:`jax.numpy.average`.

        .. note::
            The conventional 'mean' is a scalar, the 'row-mean' is an :math:`m`-vector,
            while the 'column-mean' is an :math:`n`-vector.

        To avoid materializing the entire matrix (memory cost :math:`\mathcal{O}(n m)`),
        we accumulate the mean over blocks (memory cost :math:`\mathcal{O}(B_x B_y)`,
        where ``B_x`` and ``B_y`` are user-specified block-sizes for blocking the ``x``
        and ``y`` parameters respectively.

        .. note::
            The data ``x`` and/or ``y`` are padded with zero-valued and zero-weighted
            data points, when ``B_x`` and/or ``B_y`` are non-integer divisors of ``n``
            and/or ``m``. Padding does not alter the result, but does provide the block
            shape stability required by :func:`jax.lax.scan` (used for block iteration).

        :param x: Data matrix, :math:`n \times d`
        :param y: Data matrix, :math:`m \times d`
        :param axis: Which axis of the kernel matrix to compute the mean over; a value
            of `None` computes the mean over both axes
        :param block_size: Size of matrix blocks to process; a value of :data:`None`
            sets :math:`B_x = n` and :math:`B_y = m`, effectively disabling the block
            accumulation; an integer value ``B`` sets :math:`B_y = B_x = B`; a tuple
            allows different sizes to be specified for ``B_x`` and ``B_y``; to reduce
            overheads, it is often sensible to select the largest block size that does
            not exhaust the available memory resources
        :param unroll: Unrolling parameter for the outer and inner :func:`jax.lax.scan`
            calls, allows for trade-offs between compilation and runtime cost; consult
            the JAX docs for further information
        :return: The (weighted) mean of the kernel matrix :math:`K_{ij}`
        """
        operands = x, y
        inner_unroll, outer_unroll = tree_leaves_repeat(unroll, len(operands))
        _block_size = tree_leaves_repeat(block_size, len(operands))
        # Row-mean is the argument reversed column-mean due to symmetry k(x,y) = k(y,x)
        if axis == 0:
            operands = operands[::-1]
            _block_size = _block_size[::-1]
        (block_x, unpadded_len_x), (block_y, _) = jtu.tree_map(
            _block_data_convert, operands, tuple(_block_size), is_leaf=is_data
        )

        def block_sum(accumulated_sum: Array, x_block: Data) -> tuple[Array, Array]:
            """Block reduce/accumulate over ``x``."""

            def slice_sum(accumulated_sum: Array, y_block: Data) -> tuple[Array, Array]:
                """Block reduce/accumulate over ``y``."""
                x_, w_x = x_block.data, x_block.weights
                y_, w_y = y_block.data, y_block.weights
                column_sum_slice = jnp.dot(self.compute(x_, y_), w_y)
                accumulated_sum += jnp.dot(w_x, column_sum_slice)
                return accumulated_sum, column_sum_slice

            accumulated_sum, column_sum_slices = jax.lax.scan(
                slice_sum, accumulated_sum, block_y, unroll=inner_unroll
            )
            return accumulated_sum, jnp.sum(column_sum_slices, axis=0)

        accumulated_sum, column_sum_blocks = jax.lax.scan(
            block_sum, jnp.asarray(0.0), block_x, unroll=outer_unroll
        )
        if axis is None:
            return accumulated_sum
        num_rows_padded = block_x.data.shape[0] * block_x.data.shape[1]
        column_sum_padded = column_sum_blocks.reshape(num_rows_padded, -1).sum(axis=1)
        return column_sum_padded[:unpadded_len_x]


def _block_data_convert(
    x: ArrayLike | Data, block_size: int | None
) -> tuple[Array, int]:
    """Convert 'x' into padded and weight normalized blocks of size 'block_size'."""
    x = x if isinstance(x, Data) else Data(jnp.asarray(x))
    x = x.normalize()
    block_size = len(x) if block_size is None else min(max(int(block_size), 1), len(x))
    unpadded_length = len(x)

    def _pad_reshape(x: Array) -> Array:
        n, *remaining_shape = jnp.shape(x)
        padding = (0, ceil(n / block_size) * block_size - n)
        skip_padding = ((0, 0),) * (jnp.ndim(x) - 1)
        x_padded = jnp.pad(x, (padding, *skip_padding))
        try:
            return x_padded.reshape(-1, block_size, *remaining_shape)
        except ZeroDivisionError as err:
            if 0 in x.shape:
                raise ValueError("'x' must not be empty") from err
            raise

    return jtu.tree_map(_pad_reshape, x, is_leaf=eqx.is_array_like), unpadded_length


class AdditiveKernel(Kernel):
    r"""
    Define a kernel which is a summation of two kernels.

    Given kernel functions :math:`k:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}` and
    :math:`l:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}`, define the product kernel
    :math:`p:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}` where
    :math:`p(x,y) := k(x,y) + l(x,y)`

    :param first_kernel: Instance of class coreax.kernel.Kernel
    :param second_kernel: Instance of class coreax.kernel.Kernel
    """

    first_kernel: Kernel
    second_kernel: Kernel

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike):
        return self.first_kernel.compute_elementwise(
            x, y
        ) + self.second_kernel.compute_elementwise(x, y)

    @override
    def grad_x_elementwise(self, x: ArrayLike, y: ArrayLike):
        return self.first_kernel.grad_x_elementwise(
            x, y
        ) + self.second_kernel.grad_x_elementwise(x, y)

    @override
    def grad_y_elementwise(self, x: ArrayLike, y: ArrayLike):
        return self.first_kernel.grad_y_elementwise(
            x, y
        ) + self.second_kernel.grad_y_elementwise(x, y)

    @override
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike):
        return self.first_kernel.divergence_x_grad_y_elementwise(
            x, y
        ) + self.second_kernel.divergence_x_grad_y_elementwise(x, y)


class ProductKernel(Kernel):
    r"""
    Define a kernel which is a product of two kernels.

    Given kernel functions :math:`k:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}` and
    :math:`l:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}`, define the product kernel
    :math:`p:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}` where
    :math:`p(x,y) = k(x,y)l(x,y)`

    :param first_kernel: Instance of class coreax.kernel.Kernel
    :param second_kernel: Instance of class coreax.kernel.Kernel
    """

    first_kernel: Kernel
    second_kernel: Kernel

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike):
        if self.first_kernel == self.second_kernel:
            return self.first_kernel.compute_elementwise(x, y) ** 2
        return self.first_kernel.compute_elementwise(
            x, y
        ) * self.second_kernel.compute_elementwise(x, y)

    @override
    def grad_x_elementwise(self, x: ArrayLike, y: ArrayLike):
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
    def grad_y_elementwise(self, x: ArrayLike, y: ArrayLike):
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
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike):
        pseudo_hessian = jacrev(self.grad_y_elementwise, 0)(x, y)
        return pseudo_hessian.trace()


class LinearKernel(Kernel):
    r"""
    Define a linear kernel.

    Given :math:`\rho =`'output_scale' and :math:`c =`'constant',  the linear kernel is
    defined as :math:`k: \mathbb{R}^d\times \mathbb{R}^d \to \mathbb{R}`,
    :math:`k(x, y) = \rho x^Ty + c`.

    :param output_scale: Kernel normalisation constant, :math:`\rho`
    :param constant: Additive constant, :math:`c`
    """

    output_scale: float = 1.0
    constant: float = 0.0

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
        return self.output_scale * jnp.asarray(jnp.shape(x)[0])


class SquaredExponentialKernel(Kernel):
    r"""
    Define a squared exponential kernel.

    Given :math:`\lambda =`'length_scale' and :math:`\rho =`'output_scale', the squared
    exponential kernel is defined as
    :math:`k: \mathbb{R}^d\times \mathbb{R}^d \to \mathbb{R}`,
    :math:`k(x, y) = \rho * \exp(\frac{||x-y||^2}{2 \lambda^2})` where
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
        return (
            jnp.subtract(x, y) / self.length_scale**2 * self.compute_elementwise(x, y)
        )

    @override
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        k = self.compute_elementwise(x, y)
        scale = 1 / self.length_scale**2
        d = len(jnp.asarray(x))
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
