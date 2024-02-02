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

All kernels in this module implement the base class :class:`Kernel`. They therefore must
define some ``length_scale`` and ``output_scale``, with the former controlling the
amount of smoothing applied, and the latter acting as a normalisation constant. A common
kernel used across disciplines is the :class:`SquaredExponentialKernel`, defined as

.. math::

    k(x,y) = \text{output_scale} * \exp (-||x-y||^2/2 * \text{length_scale}^2).

One can see that, if ``output_scale`` takes the value
:math:`\frac{1}{\sqrt{2\pi} \,*\, \text{length_scale}}`, then the
:class:`SquaredExponentialKernel` becomes the well known Gaussian kernel.

There are only two mandatory methods to implement when defining a new kernel. The first
is :meth:`~Kernel._compute_elementwise`, which returns the floating point value after
evaluating the kernel on two floats, ``x`` and ``y``. Performance improvements can be
gained when kernels are used in other areas of the codebase by also implementing
:meth:`~Kernel._grad_x_elementwise` and :meth:`~Kernel._grad_y_elementwise` which are
simply the gradients of the kernel with respect to ``x`` and ``y`` respectively.
Finally, :meth:`~Kernel._divergence_x_grad_y_elementwise`, the divergence with respect
to ``x`` of the gradient of the kernel with respect to ``y`` can allow analytical
computation of the :class:`SteinKernel`, which itself requires a base kernel. However,
if this property is not known, one can turn to the approaches in
:class:`~coreax.score_matching.ScoreMatching` to side-step this requirement.

The other mandatory method to implement when defining a new kernel is
:meth:`~Kernel._tree_flatten`. To improve performance, kernel computation is JIT
compiled. As a result, definitions of dynamic and static values inside
:meth:`~Kernel._tree_flatten` ensure the kernel object can be mutated and the
corresponding JIT compilation does not yield unexpected results.
"""

# Support annotations with | in Python < 3.10
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array, grad, jacrev, jit, tree_util, vmap
from jax.typing import ArrayLike

import coreax.util
import coreax.validation

if TYPE_CHECKING:
    import coreax.approximation


@jit
def median_heuristic(x: ArrayLike) -> Array:
    """
    Compute the median heuristic for setting kernel bandwidth.

    Analysis of the performance of the median heuristic can be found in
    :cite:p:`garreau2018median`.

    :param x: Input array of vectors
    :return: Bandwidth parameter, computed from the median heuristic, as a
        zero-dimensional array
    """
    # Validate inputs
    x = coreax.validation.cast_as_type(x=x, object_name="x", type_caster=jnp.atleast_2d)
    # Calculate square distances as an upper triangular matrix
    square_distances = jnp.triu(coreax.util.squared_distance_pairwise(x, x), k=1)
    # Calculate the median of the square distances
    median_square_distance = jnp.median(
        square_distances[jnp.triu_indices_from(square_distances, k=1)]
    )

    return jnp.sqrt(median_square_distance / 2.0)


class Kernel(ABC):
    """
    Base class for kernels.

    :param length_scale: Kernel ``length_scale`` to use
    :param output_scale: Output scale to use
    """

    def __init__(self, length_scale: float = 1.0, output_scale: float = 1.0):
        """Define a kernel."""
        # Check that length_scale is above zero (the cast_as_type check here is to
        # ensure that we don't check a trace of an array when jit decorators interact
        # with code)

        # Validate inputs
        length_scale = coreax.validation.cast_as_type(
            x=length_scale, object_name="length_scale", type_caster=float
        )
        output_scale = coreax.validation.cast_as_type(
            x=output_scale, object_name="output_scale", type_caster=float
        )
        coreax.validation.validate_in_range(
            x=length_scale,
            object_name="length_scale",
            strict_inequalities=True,
            lower_bound=0,
        )
        coreax.validation.validate_in_range(
            x=output_scale,
            object_name="output_scale",
            strict_inequalities=True,
            lower_bound=0,
        )

        self.length_scale = length_scale
        self.output_scale = output_scale

    @abstractmethod
    def tree_flatten(self) -> tuple[tuple, dict]:
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable JIT decoration
        of methods inside this class.

        :return: Tuple containing two elements. The first is a tuple holding the arrays
            and dynamic values that are present in the class. The second is a dictionary
            holding the static auxiliary data for the class, with keys being the names
            of class attributes, and values being the values of the corresponding class
            attributes.
        """

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a pytree from the tree definition and the leaves.

        Arrays & dynamic values (children) and auxiliary data (static values) are
        reconstructed. A method to reconstruct the pytree needs to be specified to
        enable JIT decoration of methods inside this class.
        """
        return cls(*children, **aux_data)

    @jit
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
        # Validate inputs
        x = coreax.validation.cast_as_type(
            x=x, object_name="x", type_caster=jnp.atleast_2d
        )
        y = coreax.validation.cast_as_type(
            x=y, object_name="y", type_caster=jnp.atleast_2d
        )
        fn = vmap(
            vmap(self.compute_elementwise, in_axes=(0, None), out_axes=0),
            in_axes=(None, 0),
            out_axes=1,
        )
        return fn(x, y)

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

    @jit
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
        # Validate inputs
        x = coreax.validation.cast_as_type(
            x=x, object_name="x", type_caster=jnp.atleast_2d
        )
        y = coreax.validation.cast_as_type(
            x=y, object_name="y", type_caster=jnp.atleast_2d
        )

        fn = vmap(
            vmap(self.grad_x_elementwise, in_axes=(0, None), out_axes=0),
            in_axes=(None, 0),
            out_axes=1,
        )
        return fn(x, y)

    @jit
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
        # Validate inputs
        x = coreax.validation.cast_as_type(
            x=x, object_name="x", type_caster=jnp.atleast_2d
        )
        y = coreax.validation.cast_as_type(
            x=y, object_name="y", type_caster=jnp.atleast_2d
        )

        fn = vmap(
            vmap(self.grad_y_elementwise, in_axes=(0, None), out_axes=0),
            in_axes=(None, 0),
            out_axes=1,
        )
        return fn(x, y)

    def grad_x_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        # pylint: disable=line-too-long
        r"""
        Evaluate the element-wise gradient of the kernel function w.r.t. ``x``.

        The gradient (Jacobian) of the kernel function w.r.t. ``x`` is computed using
        `Autodiff <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html>`_.

        Only accepts single vectors ``x`` and ``y``, i.e. not arrays. :meth:`grad_x`
        provides a vectorised version of this method for arrays.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Jacobian
            :math:`\nabla_\mathbf{x} k(\mathbf{x}, \mathbf{y}) \in \mathbb{R}^d`
        """
        # pylint: enable=line-too-long
        return grad(self.compute_elementwise, 0)(x, y)

    def grad_y_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        # pylint: disable=line-too-long
        r"""
        Evaluate the element-wise gradient of the kernel function w.r.t. ``y``.

        The gradient (Jacobian) of the kernel function is computed using
        `Autodiff <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html>`_.

        Only accepts single vectors ``x`` and ``y``, i.e. not arrays. :meth:`grad_y`
        provides a vectorised version of this method for arrays.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`.
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`.
        :return: Jacobian
            :math:`\nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y}) \in \mathbb{R}^d`
        """
        # pylint: enable=line-too-long
        return grad(self.compute_elementwise, 1)(x, y)

    @jit
    def divergence_x_grad_y(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
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
        # Validate inputs
        x = coreax.validation.cast_as_type(
            x=x, object_name="x", type_caster=jnp.atleast_2d
        )
        y = coreax.validation.cast_as_type(
            x=y, object_name="y", type_caster=jnp.atleast_2d
        )

        fn = vmap(
            vmap(
                self.divergence_x_grad_y_elementwise,
                in_axes=(0, None),
                out_axes=0,
            ),
            in_axes=(None, 0),
            out_axes=1,
        )
        return fn(x, y)

    def divergence_x_grad_y_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        # pylint: disable=line-too-long
        r"""
        Evaluate the element-wise divergence w.r.t. ``x`` of Jacobian w.r.t. ``y``.

        The evaluation is done via
        `Autodiff <https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html>`_.

        :math:`\nabla_\mathbf{x} \cdot \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.
        Only accepts vectors ``x`` and ``y``. A vectorised version for arrays is
        computed in :meth:`Kernel.compute_divergence_x_grad_y`.

        This is the trace of the 'pseudo-Hessian', i.e. the trace of the Jacobian matrix
        :math:`\nabla_\mathbf{x} \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.

        :param x: First vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Second vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Trace of the Laplace-style operator; a real number
        """
        # pylint: enable=line-too-long
        pseudo_hessian = jacrev(self.grad_y_elementwise, 0)(x, y)
        return pseudo_hessian.trace()

    @staticmethod
    def update_kernel_matrix_row_sum(
        x: ArrayLike,
        kernel_row_sum: ArrayLike,
        i: int,
        j: int,
        kernel_pairwise: coreax.util.KernelComputeType,
        max_size: int = 10_000,
    ) -> Array:
        r"""
        Update the row sum of the kernel matrix with a single block of values.

        The row sum of the kernel matrix may involve a large number of pairwise
        computations, so this can be done in blocks to reduce memory requirements.

        The kernel matrix block ``i``:``i`` + ``max_size`` :math:`\times`
        ``j``:``j`` + ``max_size`` is used to update the row sum. Symmetry of the kernel
        matrix is exploited to reduced repeated calculation.

        :param x: Data matrix, :math:`n \times d`
        :param kernel_row_sum: Full data structure for Gram matrix row sum,
            :math:`1 \times n`
        :param i: Kernel matrix block start
        :param j: Kernel matrix block end
        :param kernel_pairwise: Pairwise kernel evaluation function
        :param max_size: Size of matrix block to process
        :return: Gram matrix row sum, with elements ``i``:``i`` + ``max_size`` and
            ``j``:``j`` + ``max_size`` populated
        """
        # Validate inputs
        i = coreax.validation.cast_as_type(x=i, object_name="i", type_caster=int)
        j = coreax.validation.cast_as_type(x=j, object_name="j", type_caster=int)
        coreax.validation.validate_in_range(
            x=i, object_name="i", strict_inequalities=False, lower_bound=0
        )
        coreax.validation.validate_in_range(
            x=j, object_name="i", strict_inequalities=False, lower_bound=0
        )
        x = coreax.validation.cast_as_type(
            x=x, object_name="x", type_caster=jnp.atleast_2d
        )
        kernel_row_sum = coreax.validation.cast_as_type(
            x=kernel_row_sum, object_name="kernel_row_sum", type_caster=jnp.asarray
        )
        max_size = coreax.validation.cast_as_type(
            x=max_size, object_name="max_size", type_caster=int
        )
        coreax.validation.validate_in_range(
            x=max_size, object_name="max_size", strict_inequalities=True, lower_bound=0
        )

        # Compute the kernel row sum for this particular chunk of data
        kernel_row_sum_part = kernel_pairwise(x[i : i + max_size], x[j : j + max_size])

        # Assign the kernel row sum to the relevant part of this full matrix
        kernel_row_sum = kernel_row_sum.at[i : i + max_size].set(
            kernel_row_sum[i : i + max_size] + kernel_row_sum_part.sum(axis=1)
        )

        if i != j:
            kernel_row_sum = kernel_row_sum.at[j : j + max_size].set(
                kernel_row_sum[j : j + max_size] + kernel_row_sum_part.sum(axis=0)
            )

        return kernel_row_sum

    def calculate_kernel_matrix_row_sum(
        self,
        x: ArrayLike,
        max_size: int = 10_000,
    ) -> Array:
        r"""
        Compute the row sum of the kernel matrix.

        The row sum of the kernel matrix is the sum of distances between a given point
        and all possible pairs of points that contain this given point. The row sum is
        calculated block-wise to limit memory overhead.

        :param x: Data matrix, :math:`n \times d`
        :param max_size: Size of matrix block to process
        :return: Kernel matrix row sum
        """
        # Validate inputs
        x = coreax.validation.cast_as_type(
            x=x, object_name="x", type_caster=jnp.atleast_2d
        )
        max_size = coreax.validation.cast_as_type(
            x=max_size, object_name="max_size", type_caster=int
        )
        coreax.validation.validate_in_range(
            x=max_size, object_name="max_size", strict_inequalities=True, lower_bound=0
        )

        # Define the function to call to evaluate the kernel for all pairwise sets of
        # points
        kernel_pairwise = jit(
            vmap(
                vmap(self.compute_elementwise, in_axes=(0, None), out_axes=0),
                in_axes=(None, 0),
                out_axes=1,
            )
        )

        # Ensure data format is as required
        num_data_points = len(x)
        kernel_row_sum = jnp.zeros(num_data_points)

        # Iterate over upper triangular blocks
        for i in range(0, num_data_points, max_size):
            for j in range(i, num_data_points, max_size):
                kernel_row_sum = self.update_kernel_matrix_row_sum(
                    x,
                    kernel_row_sum,
                    i,
                    j,
                    kernel_pairwise,
                    max_size,
                )
        return kernel_row_sum

    def calculate_kernel_matrix_row_sum_mean(
        self,
        x: ArrayLike,
        max_size: int = 10_000,
    ) -> Array:
        r"""
        Compute the mean of the row sum of the kernel matrix.

        The mean of the row sum of the kernel matrix is the mean of the sum of distances
        between a given point and all possible pairs of points that contain this given
        point.

        :param x: Data matrix, :math:`n \times d`
        :param max_size: Size of matrix block to process
        """
        # Validate inputs
        x = coreax.validation.cast_as_type(
            x=x, object_name="x", type_caster=jnp.atleast_2d
        )
        max_size = coreax.validation.cast_as_type(
            x=max_size, object_name="max_size", type_caster=int
        )
        coreax.validation.validate_in_range(
            x=max_size, object_name="max_size", strict_inequalities=True, lower_bound=0
        )

        return self.calculate_kernel_matrix_row_sum(x, max_size) / (1.0 * x.shape[0])

    @staticmethod
    def approximate_kernel_matrix_row_sum_mean(
        x: ArrayLike,
        approximator: coreax.approximation.KernelMeanApproximator,
    ) -> Array:
        r"""
        Approximate the mean of the row sum of the kernel matrix.

        The mean of the row sum of the kernel matrix is the mean of the sum of distances
        between a given point and all possible pairs of points that contain this given
        point. This can involve a large number of pairwise computations, so an
        approximation can be used in place of the true value.

        :param x: Data matrix, :math:`n \times d`
        :param approximator: Instantiated
            :class:`~coreax.approximation.KernelMeanApproximator` object that has been
            created using the same kernel one wishes to use
        :return: Approximation to the kernel matrix row sum
        """
        return approximator.approximate(x)


class SquaredExponentialKernel(Kernel):
    """
    Define a squared exponential kernel.

    :param length_scale: Kernel ``length_scale`` to use
    :param output_scale: Output scale to use
    """

    def tree_flatten(self) -> tuple[tuple, dict]:
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable JIT decoration
        of methods inside this class.

        :return: Tuple containing two elements. The first is a tuple holding the arrays
            and dynamic values that are present in the class. The second is a dictionary
            holding the static auxiliary data for the class, with keys being the names
            of class attributes, and values being the values of the corresponding class
            attributes.
        """
        children = ()
        aux_data = {
            "length_scale": self.length_scale,
            "output_scale": self.output_scale,
        }
        return children, aux_data

    def compute_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Evaluate the squared exponential kernel on input vectors ``x`` and ``y``.

        We assume ``x`` and ``y`` are two vectors of the same dimension.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Kernel evaluated at (``x``, ``y``)
        """
        return self.output_scale * jnp.exp(
            -coreax.util.squared_distance(x, y) / (2 * self.length_scale**2)
        )

    def grad_x_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Evaluate the element-wise grad of the squared exponential kernel w.r.t. ``x``.

        The gradient (Jacobian) is computed using the analytical form.

        Only accepts single vectors ``x`` and ``y``, i.e. not arrays. :meth:`grad_x`
        provides a vectorised version of this method for arrays.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Jacobian
            :math:`\nabla_\mathbf{x} k(\mathbf{x}, \mathbf{y}) \in \mathbb{R}^d`
        """
        return -self.grad_y_elementwise(x, y)

    def grad_y_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Evaluate the element-wise grad of the squared exponential kernel w.r.t. ``y``.

        The gradient (Jacobian) is computed using the analytical form.

        Only accepts single vectors ``x`` and ``y``, i.e. not arrays. :meth:`grad_y`
        provides a vectorised version of this method for arrays.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Jacobian
            :math:`\nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y}) \in \mathbb{R}^d`
        """
        return (x - y) / self.length_scale**2 * self.compute_elementwise(x, y)

    def divergence_x_grad_y_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Evaluate the element-wise divergence w.r.t. ``x`` of Jacobian w.r.t. ``y``.

        The computations are done using the analytical form of Jacobian and divergence
        of the squared exponential kernel.

        :math:`\nabla_\mathbf{x} \cdot \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.
        Only accepts vectors ``x`` and ``y``. A vectorised version for arrays is
        computed in :meth:`SquaredExponentialKernel.compute_divergence_x_grad_y`.

        This is the trace of the 'pseudo-Hessian', i.e. the trace of the Jacobian matrix
        :math:`\nabla_\mathbf{x} \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.

        :param x: First vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Second vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Trace of the Laplace-style operator; a real number
        """
        k = self.compute_elementwise(x, y)
        scale = 1 / self.length_scale**2
        d = len(x)
        return scale * k * (d - scale * coreax.util.squared_distance(x, y))


class LaplacianKernel(Kernel):
    """
    Define a Laplacian kernel.

    :param length_scale: Kernel ``length_scale`` to use
    :param output_scale: Output scale to use
    """

    def tree_flatten(self) -> tuple[tuple, dict]:
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable JIT decoration
        of methods inside this class.

        :return: Tuple containing two elements. The first is a tuple holding the arrays
            and dynamic values that are present in the class. The second is a dictionary
            holding the static auxiliary data for the class, with keys being the names
            of class attributes, and values being the values of the corresponding class
            attributes.
        """
        children = ()
        aux_data = {
            "length_scale": self.length_scale,
            "output_scale": self.output_scale,
        }
        return children, aux_data

    def compute_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Evaluate the Laplacian kernel on input vectors ``x`` and ``y``.

        We assume ``x`` and ``y`` are two vectors of the same dimension.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Kernel evaluated at (``x``, ``y``)
        """
        return self.output_scale * jnp.exp(
            -jnp.linalg.norm(x - y, ord=1) / (2 * self.length_scale**2)
        )

    def grad_x_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Evaluate the element-wise grad of the Laplacian kernel w.r.t. ``x``.

        The gradient (Jacobian) is computed using the analytical form.

        Only accepts single vectors ``x`` and ``y``, i.e. not arrays. :meth:`grad_x`
        provides a vectorised version of this method for arrays.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Jacobian
            :math:`\nabla_\mathbf{x} k(\mathbf{x}, \mathbf{y}) \in \mathbb{R}^d`
        """
        return -self.grad_y_elementwise(x, y)

    def grad_y_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Evaluate the element-wise grad of the Laplacian kernel w.r.t. ``y``.

        The gradient (Jacobian) is computed using the analytical form.

        Only accepts single vectors ``x`` and ``y``, i.e. not arrays. :meth:`grad_y`
        provides a vectorised version of this method for arrays.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Jacobian
            :math:`\nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y}) \in \mathbb{R}^d`
        """
        return (
            jnp.sign(x - y)
            / (2 * self.length_scale**2)
            * self.compute_elementwise(x, y)
        )

    def divergence_x_grad_y_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Evaluate the element-wise divergence w.r.t. ``x`` of Jacobian w.r.t. ``y``.

        The computations are done using the analytical form of Jacobian and divergence
        of the Laplacian kernel.

        :math:`\nabla_\mathbf{x} \cdot \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.
        Only accepts vectors ``x`` and ``y``. A vectorised version for arrays is
        computed in :meth:`LaplacianKernel.compute_divergence_x_grad_y`.

        This is the trace of the 'pseudo-Hessian', i.e. the trace of the Jacobian matrix
        :math:`\nabla_\mathbf{x} \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.

        :param x: First vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Second vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Trace of the Laplace-style operator; a real number
        """
        k = self.compute_elementwise(x, y)
        d = len(x)
        return -d * k / (4 * self.length_scale**4)


class PCIMQKernel(Kernel):
    """
    Define a pre-conditioned inverse multi-quadric (PCIMQ) kernel.

    :param length_scale: Kernel ``length_scale`` to use
    :param output_scale: Output scale to use
    """

    def tree_flatten(self) -> tuple[tuple, dict]:
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable JIT decoration
        of methods inside this class.

        :return: Tuple containing two elements. The first is a tuple holding the arrays
            and dynamic values that are present in the class. The second is a dictionary
            holding the static auxiliary data for the class, with keys being the names
            of class attributes, and values being the values of the corresponding class
            attributes.
        """
        children = ()
        aux_data = {
            "length_scale": self.length_scale,
            "output_scale": self.output_scale,
        }
        return children, aux_data

    def compute_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Evaluate the PCIMQ kernel on input vectors ``x`` and ``y``.

        We assume ``x`` and ``y`` are two vectors of the same dimension.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Kernel evaluated at (``x``, ``y``)
        """
        scaling = 2 * self.length_scale**2
        mq_array = coreax.util.squared_distance(x, y) / scaling
        return self.output_scale / jnp.sqrt(1 + mq_array)

    def grad_x_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Element-wise gradient (Jacobian) of the PCIMQ kernel function w.r.t. ``x``.

        Only accepts single vectors ``x`` and ``y``, i.e. not arrays. :meth:`grad_x`
        provides a vectorised version of this method for arrays.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Jacobian
            :math:`\nabla_\mathbf{x} k(\mathbf{x}, \mathbf{y}) \in \mathbb{R}^d`
        """
        return -self.grad_y_elementwise(x, y)

    def grad_y_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Element-wise gradient (Jacobian) of the PCIMQ kernel function w.r.t. ``y``.

        Only accepts single vectors ``x`` and ``y``, i.e. not arrays. :meth:`grad_y`
        provides a vectorised version of this method for arrays.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Jacobian
            :math:`\nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y}) \in \mathbb{R}^d`
        """
        return (
            self.output_scale
            * (x - y)
            / (2 * self.length_scale**2)
            * (self.compute_elementwise(x, y) / self.output_scale) ** 3
        )

    def divergence_x_grad_y_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Elementwise divergence w.r.t. ``x`` of Jacobian of PCIMQ w.r.t. ``y``.

        :math:`\nabla_\mathbf{x} \cdot \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.
        Only accepts vectors ``x`` and ``y``. A vectorised version for arrays is
        computed in :meth:`PCIMQKernel.compute_divergence_x_grad_y`.

        This is the trace of the 'pseudo-Hessian', i.e. the trace of the Jacobian matrix
        :math:`\nabla_\mathbf{x} \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.

        :param x: First vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Second vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Trace of the Laplace-style operator; a real number
        """
        k = self.compute_elementwise(x, y) / self.output_scale
        scale = 2 * self.length_scale**2
        d = len(x)
        return (
            self.output_scale
            / scale
            * (d * k**3 - 3 * k**5 * coreax.util.squared_distance(x, y) / scale)
        )


class SteinKernel(Kernel):
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

    :param base_kernel: Initialised kernel object with which to evaluate the Stein
        kernel, e.g. return from :func:`construct_kernel`
    :param score_function: A vector-valued callable defining a score function
        :math:`\mathbb{R}^d \to \mathbb{R}^d`
    :param output_scale: Output scale to use
    """

    def __init__(
        self,
        base_kernel: Kernel,
        score_function: Callable[[ArrayLike], Array],
        output_scale: float = 1.0,
    ):
        """Define the Stein kernel, i.e. the application of the Stein operator."""
        # Validate inputs
        coreax.validation.validate_is_instance(
            x=base_kernel, object_name="base_kernel", expected_type=Kernel
        )
        coreax.validation.validate_is_instance(
            x=score_function, object_name="score_function", expected_type=Callable
        )
        output_scale = coreax.validation.cast_as_type(
            x=output_scale, object_name="output_scale", type_caster=float
        )
        coreax.validation.validate_in_range(
            x=output_scale,
            object_name="output_scale",
            strict_inequalities=True,
            lower_bound=0,
        )

        self.base_kernel = base_kernel
        self.score_function = score_function
        self.output_scale = output_scale

        # Initialise parent
        super().__init__(output_scale=output_scale)

    def tree_flatten(self) -> tuple[tuple, dict]:
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable JIT decoration
        of methods inside this class.

        :return: Tuple containing two elements. The first is a tuple holding the arrays
            and dynamic values that are present in the class. The second is a dictionary
            holding the static auxiliary data for the class, with keys being the names
            of class attributes, and values being the values of the corresponding class
            attributes.
        """
        # The score function is assumed to not change here - but it might if the kernel
        # changes - but this does not work when kernel is specified in children
        children = (self.base_kernel,)
        aux_data = {
            "score_function": self.score_function,
            "output_scale": self.output_scale,
        }
        return children, aux_data

    def compute_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Evaluate the Stein kernel on input vectors ``x`` and ``y``.

        We assume ``x`` and ``y`` are two vectors of the same dimension.

        :param x: Vector :math:`\mathbf{x} \in \mathbb{R}^d`
        :param y: Vector :math:`\mathbf{y} \in \mathbb{R}^d`
        :return: Kernel evaluated at (``x``, ``y``)
        """
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


# Define the pytree node for the added class to ensure methods with JIT decorators
# are able to run. This tuple must be updated when a new class object is defined.
kernel_classes = (SquaredExponentialKernel, PCIMQKernel, SteinKernel, LaplacianKernel)
for current_class in kernel_classes:
    tree_util.register_pytree_node(
        current_class, current_class.tree_flatten, current_class.tree_unflatten
    )
