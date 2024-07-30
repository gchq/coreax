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
appropriate parameter values. For example, the
:class:`~coreax.kernels.SquaredExponentialKernel`,

.. math::

    k(x,y) = \text{output_scale} * \exp (-||x-y||^2/2 * \text{length_scale}^2),

which if parameterized with an ``output_scale`` of
:math:`\frac{1}{\sqrt{2\pi} \,*\, \text{length_scale}}`, yields the well known
Gaussian kernel.

A :class:`~coreax.kernels.ScalarValuedKernel` must implement a
:meth:`~coreax.kernels.ScalarValuedKernel.compute_elementwise` method, which returns the
floating point value after evaluating the kernel on two floats, ``x`` and ``y``.
Additional methods, such as :meth:`ScalarValuedKernel.grad_x_elementwise`, can
optionally be overridden to improve performance. The canonical example is when a
suitable closed-form representation of a higher-order gradient can be used to avoid the
expense of performing automatic differentiation.

Such an example can be seen in :class:`coreax.kernels.SteinKernel`, where the analytic
forms of :meth:`ScalarValuedKernel.divergence_x_grad_y` are significantly cheaper to
compute than the automatic differentiated default.
"""

# Support class typing annotations inside itself
from __future__ import annotations

from abc import abstractmethod
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array, grad, jacrev
from jax.typing import ArrayLike
from typing_extensions import override

from coreax.data import Data, is_data
from coreax.kernels.util import block_data_convert
from coreax.util import pairwise, tree_leaves_repeat


class ScalarValuedKernel(eqx.Module):
    """Abstract base class for scalar-valued kernels."""

    def __add__(self, addition: ScalarValuedKernel) -> AdditiveKernel:
        """Overload `+` operator."""
        if not isinstance(addition, ScalarValuedKernel):
            raise ValueError("'addition' must be an instance of a 'ScalarValuedKernel'")
        return AdditiveKernel(self, addition)

    def __mul__(self, product: ScalarValuedKernel) -> ProductKernel:
        """Overload `*` operator."""
        if not isinstance(product, ScalarValuedKernel):
            raise ValueError("'product' must be an instance of a 'ScalarValuedKernel'")
        return ProductKernel(self, product)

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
        :meth:`coreax.kernels.ScalarValuedKernel.grad_x` provides a vectorised version
        of this method for arrays.

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
        :meth:`coreax.kernels.ScalarValuedKernel.grad_y` provides a vectorised version
        of this method for arrays.

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
        computed in :meth:`~coreax.kernels.ScalarValuedKernel.divergence_x_grad_y`.

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
        x: Union[ArrayLike, Data],
        *,
        block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]] = None,
        unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
    ) -> Array:
        r"""
        Compute the (blocked) row-mean of the kernel's Gramian matrix.

        A convenience method for calling :meth:`compute_mean`. Equivalent to the call
        :code:`compute_mean(x, x, axis=0, block_size=block_size, unroll=unroll)`.

        :param x: Data matrix, :math:`n \times d`
        :param block_size: Block size parameter passed to :meth:`compute_mean`
        :param unroll: Unroll parameter passed to :meth:`compute_mean`
        :return: Gramian 'row/column-mean', :math:`\frac{1}{n}\sum_{i=1}^{n} G_{ij}`.
        """
        return self.compute_mean(x, x, axis=0, block_size=block_size, unroll=unroll)

    def compute_mean(
        self,
        x: Union[ArrayLike, Data],
        y: Union[ArrayLike, Data],
        axis: Union[int, None] = None,
        *,
        block_size: Union[int, None, tuple[Union[int, None], Union[int, None]]] = None,
        unroll: Union[int, bool, tuple[Union[int, bool], Union[int, bool]]] = 1,
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
            block_data_convert, operands, tuple(_block_size), is_leaf=is_data
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


class UniCompositeKernel(ScalarValuedKernel):
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


class DuoCompositeKernel(ScalarValuedKernel):
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


class AdditiveKernel(DuoCompositeKernel):
    r"""
    Define a kernel which is a summation of two kernels.

    Given kernel functions :math:`k:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}` and
    :math:`l:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}`, define the additive
    kernel :math:`p:\mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}` where
    :math:`p(x,y) := k(x,y) + l(x,y)`

    :param first_kernel: Instance of :class:`ScalarValuedKernel`
    :param second_kernel: Instance of :class:`ScalarValuedKernel`
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

    :param first_kernel: Instance of :class:`ScalarValuedKernel`
    :param second_kernel: Instance of :class:`ScalarValuedKernel`
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
