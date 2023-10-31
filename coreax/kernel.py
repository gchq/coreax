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

# Support annotations with | in Python < 3.10
# TODO: Remove once no longer supporting old code
from __future__ import annotations

import inspect
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable

import jax.numpy as jnp
from jax import Array, grad, jacrev, jit, random, tree_util, vmap
from jax.typing import ArrayLike

import coreax.approximation as ca
import coreax.util as cu


@jit
def median_heuristic(x: ArrayLike) -> Array:
    r"""
    Compute the median heuristic for setting kernel bandwidth.

    :param x: Input array of vectors
    :return: Bandwidth parameter, computed from the median heuristic, as a
        0-dimensional array
    """
    # calculate square distances as an upper triangular matrix
    square_distances = jnp.triu(cu.sq_dist_pairwise(x, x), k=1)
    # calculate the median
    median_square_distance = jnp.median(
        square_distances[jnp.triu_indices_from(square_distances, k=1)]
    )

    return jnp.sqrt(median_square_distance / 2.0)


class Kernel(ABC):
    """
    Base class for kernels.
    """

    def __init__(self, lengthscale: float = 1.0, scale: float = 1.0):
        """
        Define a kernel.

        :param lengthscale: Kernel lengthscale to use.
        :param scale: Output scale to use.
        """
        # TODO: generalise lengthscale to multiple dimensions.
        # Check that lengthscale is above zero (the isinstance check here is to ensure
        # that we don't check a trace of an array when jit decorators interact with
        # code)
        if isinstance(lengthscale, float) and lengthscale <= 0.0:
            raise ValueError(
                f"Lengthscale must be above zero. Current value {lengthscale}."
            )
        if isinstance(scale, float) and scale <= 0.0:
            raise ValueError(f"Output scale must be above zero. Current value {scale}.")
        self.lengthscale = lengthscale
        self.scale = scale

    @jit
    def compute(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Evaluate the kernel on input data x and y.

        The 'data' can be any of:
            * floating numbers (so a single data-point in 1-dimension)
            * zero-dimensional arrays (so a single data-point in 1-dimension)
            * a vector (a single-point in multiple dimensions)
            * array (multiple vectors).

        :param x: An :math:`n \times d` dataset (array) or a single value (point)
        :param y: An :math:`m \times d` dataset (array) or a single value (point)
        :return: Kernel evaluations between points in x and y. If x = y, then
            this is the Gram matrix corresponding to the RKHS' inner product.
        """
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        fn = vmap(
            vmap(self._compute_elementwise, in_axes=(0, None), out_axes=0),
            in_axes=(None, 0),
            out_axes=1,
        )
        return fn(x, y)

    @abstractmethod
    def _compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Evaluate the kernel on individual input vectors x and y, not-vectorised.

        Vectorisation only becomes relevant in terms of computational speed when we
        have multiple x or y

        :param x: vector :math:`\mathbf{x} \in \mathbb{R}^d`.
        :param y: vector :math:`\mathbf{y} \in \mathbb{R}^d`.
        :return: kernel evaluated at (x, y).
        """

    @jit
    def grad_x(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Gradient (Jacobian) of the kernel function w.r.t. x.

        The function is vectorised, so x or y can be any of:
            * floating numbers (so a single data-point in 1-dimension)
            * zero-dimensional arrays (so a single data-point in 1-dimension)
            * a vector (a single-point in multiple dimensions)
            * array (multiple vectors).

        :param x: An :math:`n \times d` dataset (array) or a single value (point)
        :param y: An :math:`m \times d` dataset (array) or a single value (point)
        :return: An :math:`n \times m \times d` array of pairwise Jacobians.
        """
        fn = vmap(
            vmap(self._grad_x_elementwise, in_axes=(0, None), out_axes=0),
            in_axes=(None, 0),
            out_axes=1,
        )
        return fn(x, y)

    @jit
    def grad_y(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Gradient (Jacobian) of the kernel function w.r.t. y.

        The function is vectorised, so x or y can be any of:
            * floating numbers (so a single data-point in 1-dimension)
            * zero-dimensional arrays (so a single data-point in 1-dimension)
            * a vector (a single-point in multiple dimensions)
            * array (multiple vectors).

        :param x: An :math:`n \times d` dataset (array) or a single value (point)
        :param y: An :math:`m \times d` dataset (array) or a single value (point)
        :return: An :math:`m \times n \times d` array of pairwise Jacobians.
        """
        fn = jit(
            vmap(
                vmap(self._grad_y_elementwise, in_axes=(0, None), out_axes=0),
                in_axes=(None, 0),
                out_axes=1,
            )
        )
        return fn(x, y)

    @jit
    def _grad_y_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Element-wise gradient (Jacobian) of the kernel function w.r.t. y via Autodiff.

        Only accepts single vectors x and y, i.e. not arrays. This function is
        vectorised for arrays in grad_y.

        :param x: vector :math:`\mathbf{x} \in \mathbb{R}^d`.
        :param y: vector :math:`\mathbf{y} \in \mathbb{R}^d`.
        :return: Jacobian :math:`\nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y}) \in
            \mathbb{R}^d`
        """
        return grad(self._compute_elementwise, 1)(x, y)

    @jit
    def _grad_x_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Element-wise gradient (Jacobian) of the kernel function w.r.t. x via Autodiff.

        Only accepts single vectors x and y, i.e. not arrays. This function is
        vectorised for arrays in grad_x.

        :param x: vector :math:`\mathbf{x} \in \mathbb{R}^d`.
        :param y: vector :math:`\mathbf{y} \in \mathbb{R}^d`.
        :return: Jacobian :math:`\nabla_\mathbf{x} k(\mathbf{x}, \mathbf{y}) \in
            \mathbb{R}^d`
        """
        return grad(self._compute_elementwise, 0)(x, y)

    @jit
    def _divergence_x_grad_y_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Elementwise divergence w.r.t. x of Jacobian w.r.t. y via Autodiff.

        :math:`\nabla_\mathbf{x} \cdot \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.
        Only accepts vectors x and y. A vectorised version for arrays is computed in
        compute_divergence_x_grad_y.

        This is the trace of the 'pseudo-Hessian', i.e. the trace of the Jacobian matrix
        :math:`\nabla_\mathbf{x} \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.

        :param x: First vector :math:`\mathbf{x} \in \mathbb{R}^d`.
        :param y: Second vector :math:`\mathbf{y} \in \mathbb{R}^d`.
        :return: Trace of the Laplace-style operator; a real number.
        """
        pseudo_hessian = jacrev(self._grad_y_elementwise, 0)(x, y)
        return pseudo_hessian.trace()

    @jit
    def divergence_x_grad_y(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Divergence operator w.r.t. x of Jacobian w.r.t. y.

        :math:`\nabla_\mathbf{x} \cdot \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.
        This function is vectorised, so it accepts vectors or arrays.

        This is the trace of the 'pseudo-Hessian', i.e. the trace of the Jacobian matrix
        :math:`\nabla_\mathbf{x} \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.

        :param x: First vector :math:`\mathbf{x} \in \mathbb{R}^d`.
        :param y: Second vector :math:`\mathbf{y} \in \mathbb{R}^d`.
        :return: Array of Laplace-style operator traces :math:`n \times m` array.
        """
        fn = vmap(
            vmap(
                self._divergence_x_grad_y_elementwise,
                in_axes=(0, None),
                out_axes=0,
            ),
            in_axes=(None, 0),
            out_axes=1,
        )
        return fn(x, y)

    @staticmethod
    def update_kernel_matrix_row_sum(
        x: ArrayLike,
        kernel_row_sum: ArrayLike,
        i: int,
        j: int,
        kernel_pairwise: cu.KernelFunction | cu.KernelFunctionWithGrads,
        grads: ArrayLike | None = None,
        lengthscale: float | None = None,
        max_size: int = 10_000,
    ) -> Array:
        r"""
        Update the row sum of the kernel matrix with a single block of values.

        The row sum of the kernel matrix may involve a large number of pairwise
        computations, so this can be done in blocks to reduce memory requirements.

        The kernel matrix block :math:`i:i+max_size \times j:j+max_size` is used to
        update the row sum. Symmetry of the kernel matrix is exploited to reduced
        repeated calculation.

        Note that `k_pairwise` should be of the form :math:`k(x,y)` if `grads` and
        `lengthscale` are `None`. Else, `k_pairwise` should be of the form :math:`k(x,y,
        grads, grads, n, lengthscale)`.

        :param x: Data matrix, :math:`n \times d`
        :param kernel_row_sum: Full data structure for Gram matrix row sum, :math:`1
            \times n`
        :param i: Kernel matrix block start
        :param j: Kernel matrix block end
        :param max_size: Size of matrix block to process
        :param kernel_pairwise: Pairwise kernel evaluation function
        :param grads: Array of gradients, if applicable, :math:`n \times d`; Optional,
            defaults to `None`
        :param lengthscale: Base kernel lengthscale. Optional, defaults to `None`
        :return: Gram matrix row sum, with elements :math:`i: i + max_size` and
            :math:`j: j + max_size` populated
        """
        # Ensure data format is as required
        x = jnp.asarray(x)
        kernel_row_sum = jnp.asarray(kernel_row_sum)
        num_datapoints = x.shape[0]

        # Compute the kernel row sum for this particular chunk of data
        if grads is None:
            kernel_row_sum_part = kernel_pairwise(
                x[i : i + max_size], x[j : j + max_size]
            )
        else:
            grads = jnp.asarray(grads)
            kernel_row_sum_part = kernel_pairwise(
                x[i : i + max_size],
                x[j : j + max_size],
                grads[i : i + max_size],
                grads[j : j + max_size],
                num_datapoints,
                lengthscale,
            )

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
        grads: ArrayLike | None = None,
        lengthscale: ArrayLike | None = None,
    ) -> Array:
        r"""
        Compute the row sum of the kernel matrix.

        The row sum of the kernel matrix is the sum of distances between a given point
        and all possible pairs of points that contain this given point. The row sum is
        calculated block-wise to limit memory overhead.

        Note that `k_pairwise` should be of the form :math:`k(x,y)` if `grads` and
        `lengthscale` are `None`. Else, `k_pairwise` should be of the form :math:`k(x,y,
        grads, grads, n, lengthscale)`.

        :param x: Data matrix, :math:`n \times d`
        :param max_size: Size of matrix block to process
        :param grads: Array of gradients, if applicable, :math:`n \times d` Optional,
                      defaults to `None`
        :param lengthscale: Base kernel lengthscale, if applicable, :math:`n \times d`
                   Optional, defaults to `None`
        :return: Kernel matrix row sum
        """
        # Define the function to call to evaluate the kernel for all pairwise sets of
        # points
        if grads is None:
            kernel_pairwise = jit(
                vmap(
                    vmap(self._compute_elementwise, in_axes=(0, None), out_axes=0),
                    in_axes=(None, 0),
                    out_axes=1,
                )
            )
        else:
            kernel_pairwise = jit(
                vmap(
                    vmap(self._compute_elementwise, (None, 0, None, 0, None, None), 0),
                    (0, None, 0, None, None, None),
                    0,
                )
            )

        # Ensure data format is as required
        x = jnp.asarray(x)
        num_datapoints = len(x)
        kernel_row_sum = jnp.zeros(num_datapoints)
        # Iterate over upper triangular blocks
        for i in range(0, num_datapoints, max_size):
            for j in range(i, num_datapoints, max_size):
                kernel_row_sum = self.update_kernel_matrix_row_sum(
                    x,
                    kernel_row_sum,
                    i,
                    j,
                    kernel_pairwise,
                    grads,
                    lengthscale,
                    max_size,
                )
        return kernel_row_sum

    def calculate_kernel_matrix_row_sum_mean(
        self,
        x: ArrayLike,
        grads: ArrayLike | None = None,
        lengthscale: ArrayLike | None = None,
        max_size: int = 10_000,
    ) -> Array:
        r"""
        Compute the mean of the row sum of the kernel matrix.

        The mean of the row sum of the kernel matrix is the mean of the sum of distances
        between a given point and all possible pairs of points that contain this given
        point.

        :param x: Data matrix, :math:`n \times d`
        :param max_size: Size of matrix block to process
        :param grads: Array of gradients, if applicable, :math:`n \times d`
                      Optional, defaults to `None`
        :param lengthscale: Base kernel lengthscale, if applicable, :math:`n \times d`
                   Optional, defaults to `None`
        """
        return self.calculate_kernel_matrix_row_sum(x, max_size, grads, lengthscale) / (
            1.0 * x.shape[0]
        )

    def approximate_kernel_matrix_row_sum_mean(
        self,
        x: ArrayLike,
        approximator: str | type[ca.KernelMeanApproximator],
        random_key: random.PRNGKeyArray = random.PRNGKey(0),
        num_kernel_points: int = 10_000,
        num_train_points: int = 10_000,
    ) -> Array:
        r"""
        Approximate the mean of the row sum of the kernel matrix.

        The mean of the row sum of the kernel matrix is the mean of the sum of distances
        between a given point and all possible pairs of points that contain this given
        point. This can involve a large number of pairwise computations, so an
        approximation can be used in place of the true value.

        :param x: Data matrix, :math:`n \times d`
        :param approximator: Name of the approximator to use, or an uninstatiated
            class object
        :param random_key: Key for random number generation
        :param num_kernel_points: Number of kernel evaluation points
        :param num_train_points: Number of training points used to fit kernel
            regression. This is ignored if not applicable to the approximator method.
        :return: Approximation to the kernel matrix row sum
        """
        # Create an approximator object
        approximator = self.create_approximator(
            approximator=approximator,
            random_key=random_key,
            num_kernel_points=num_kernel_points,
            num_train_points=num_train_points,
        )
        return approximator.approximate(x)

    def create_approximator(
        self,
        approximator: str | type[ca.KernelMeanApproximator],
        random_key: random.PRNGKeyArray = random.PRNGKey(0),
        num_kernel_points: int = 10_000,
        num_train_points: int = 10_000,
    ) -> ca.KernelMeanApproximator:
        r"""
        Create an approximator object for use with the kernel matrix row sum mean.

        :param approximator: The name of an approximator class to use, or the
            uninstantiated class directly as a dependency injection
        :param random_key: Key for random number generation
        :param num_kernel_points: Number of kernel evaluation points
        :param num_train_points: Number of training points used to fit kernel
            regression. This is ignored if not applicable to the approximator method.
        :return: Approximator object
        """
        approximator_obj = ca.approximator_factory.get(approximator)

        # Initialise, accounting for different classes having different numbers of
        # parameters
        return cu.call_with_excess_kwargs(
            approximator_obj,
            kernel_evaluation=self._compute_elementwise,
            random_key=random_key,
            num_kernel_points=num_kernel_points,
            num_train_points=num_train_points,
        )


class SquaredExponentialKernel(Kernel):
    """
    Define a squared exponential kernel.
    """

    def _tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable jit decoration
        of methods inside this class.
        """
        children = ()
        aux_data = {"lengthscale": self.lengthscale, "scale": self.scale}
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """
        Reconstructs a pytree from the tree definition and the leaves.

        Arrays & dynamic values (children) and auxiliary data (static values) are
        reconstructed. A method to reconstruct the pytree needs to be specified to
        enable jit decoration of methods inside this class.
        """
        return cls(*children, **aux_data)

    @jit
    def _compute_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Evaluate the squared exponential kernel on input vectors x and y.

        We assume x and y are two vectors of the same dimension.

        :param x: vector :math:`\mathbf{x} \in \mathbb{R}^d`.
        :param y: vector :math:`\mathbf{y} \in \mathbb{R}^d`.
        :return: kernel evaluated at (x, y).
        """
        return self.scale * jnp.exp(-cu.sq_dist(x, y) / (2 * self.lengthscale**2))

    @jit
    def _grad_x_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Element-wise gradient (Jacobian) of the squared exponential kernel w.r.t. x.

        Only accepts single vectors x and y, i.e. not arrays. This function is
        vectorised for arrays in grad_x.

        :param x: vector :math:`\mathbf{x} \in \mathbb{R}^d`.
        :param y: vector :math:`\mathbf{y} \in \mathbb{R}^d`.
        :return: Jacobian :math:`\nabla_\mathbf{x} k(\mathbf{x}, \mathbf{y}) \in
            \mathbb{R}^d`
        """
        return -self._grad_y_elementwise(x, y)

    @jit
    def _grad_y_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Element-wise gradient (Jacobian) of the squared exponential kernel w.r.t. y.

        Only accepts single vectors x and y, i.e. not arrays. This function is
        vectorised for arrays in grad_y.

        :param x: vector :math:`\mathbf{x} \in \mathbb{R}^d`.
        :param y: vector :math:`\mathbf{y} \in \mathbb{R}^d`.
        :return: Jacobian :math:`\nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y}) \in
            \mathbb{R}^d`
        """
        return (x - y) / self.lengthscale**2 * self._compute_elementwise(x, y)

    @jit
    def _divergence_x_grad_y_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Elementwise divergence w.r.t. x of Jacobian of squared exponential w.r.t. y.

        :math:`\nabla_\mathbf{x} \cdot \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.
        Only accepts vectors x and y. A vectorised version for arrays is computed in
        compute_divergence_x_grad_y.

        This is the trace of the 'pseudo-Hessian', i.e. the trace of the Jacobian matrix
        :math:`\nabla_\mathbf{x} \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.

        :param x: First vector :math:`\mathbf{x} \in \mathbb{R}^d`.
        :param y: Second vector :math:`\mathbf{y} \in \mathbb{R}^d`.
        :return: Trace of the Laplace-style operator; a real number.
        """
        k = self._compute_elementwise(x, y)
        scale = 1 / self.lengthscale**2
        d = len(x)
        return scale * k * (d - scale * cu.sq_dist(x, y))


class PCIMQKernel(Kernel):
    """
    Define a pre-conditioned inverse multi-quadric (PCIMQ) kernel.
    """

    def _tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable jit decoration
        of methods inside this class.
        """
        children = ()
        aux_data = {"lengthscale": self.lengthscale, "scale": self.scale}
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """
        Reconstructs a pytree from the tree definition and the leaves.

        Arrays & dynamic values (children) and auxiliary data (static values) are
        reconstructed. A method to reconstruct the pytree needs to be specified to
        enable jit decoration of methods inside this class.
        """
        return cls(*children, **aux_data)

    @jit
    def _compute_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Evaluate the PCIMQ kernel on input vectors x and y.

        We assume x and y are two vectors of the same dimension.

        :param x: vector :math:`\mathbf{x} \in \mathbb{R}^d`.
        :param y: vector :math:`\mathbf{y} \in \mathbb{R}^d`.
        :return: kernel evaluated at (x, y).
        """
        scaling = 2 * self.lengthscale**2
        mq_array = cu.sq_dist(x, y) / scaling
        return self.scale / jnp.sqrt(1 + mq_array)

    @jit
    def _grad_x_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Element-wise gradient (Jacobian) of the PCIMQ kernel function w.r.t. x.

        Only accepts single vectors x and y, i.e. not arrays. This function is
        vectorised for arrays in grad_x.

        :param x: vector :math:`\mathbf{x} \in \mathbb{R}^d`.
        :param y: vector :math:`\mathbf{y} \in \mathbb{R}^d`.
        :return: Jacobian :math:`\nabla_\mathbf{x} k(\mathbf{x}, \mathbf{y}) \in
            \mathbb{R}^d`
        """
        return -self._grad_y_elementwise(x, y)

    @jit
    def _grad_y_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Element-wise gradient (Jacobian) of the PCIMQ kernel function w.r.t. y.

        Only accepts single vectors x and y, i.e. not arrays. This function is
        vectorised for arrays in grad_y.

        :param x: vector :math:`\mathbf{x} \in \mathbb{R}^d`.
        :param y: vector :math:`\mathbf{y} \in \mathbb{R}^d`.
        :return: Jacobian :math:`\nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y}) \in
            \mathbb{R}^d`
        """
        return (
            self.scale
            * (x - y)
            / (2 * self.lengthscale**2)
            * (self._compute_elementwise(x, y) / self.scale) ** 3
        )

    @jit
    def _divergence_x_grad_y_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Elementwise divergence w.r.t. x of Jacobian of PCIMQ w.r.t. y.

        :math:`\nabla_\mathbf{x} \cdot \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.
        Only accepts vectors x and y. A vectorised version for arrays is computed in
        compute_divergence_x_grad_y.

        This is the trace of the 'pseudo-Hessian', i.e. the trace of the Jacobian matrix
        :math:`\nabla_\mathbf{x} \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y})`.

        :param x: First vector :math:`\mathbf{x} \in \mathbb{R}^d`.
        :param y: Second vector :math:`\mathbf{y} \in \mathbb{R}^d`.
        :return: Trace of the Laplace-style operator; a real number.
        """
        k = self._compute_elementwise(x, y) / self.scale
        scale = 2 * self.lengthscale**2
        d = len(x)
        return self.scale / scale * (d * k**3 - 3 * k**5 * cu.sq_dist(x, y) / scale)


class SteinKernel(Kernel):
    def __init__(
        self,
        base_kernel: Kernel,
        score_function: Callable[[ArrayLike], Array],
        scale: float = 1.0,
    ):
        r"""
        Define the Stein kernel, i.e. the application of the Stein operator.

        :math:`\mathcal{A}_\mathbb{P}(g(\mathbf{x})) := \nabla_\mathbf{x} g(\mathbf{x})
        + g(\mathbf{x}) \nabla_\mathbf{x} \log f_X(\mathbf{x})^\intercal,`

        w.r.t. probability measure :math:`\mathbb{P}` to the base kernel $k(\mathbf{x},
        \mathbf{y})$. Here, differentiable vector-valued :math:`g: \mathbb{R}^d \to
        \mathbb{R}^d`, and :math: `\nabla_\mathbf{x} \log f_X(\mathbf{x})` is the *score
        function* of measure :math:`\mathbb{P}`.

        :math:`\mathbb{P}` is assumed to admit a density function :math:`f_X` w.r.t.
        d-dimensional Lebesgue measure. The score function is assumed to be Lipschitz.

        The key property of a Stein operator is zero expectation under
        :math:`\mathbb{P}`, i.e. :math:`\mathbb{E}_\mathbb{P}[\mathcal{A}_\mathbb{P}
        f(\mathbf{x})]`, for positive differentiable :math:`f_X`.

        The Stein kernel for base kernel :math:`k(\mathbf{x}, \mathbf{y})` is defined as

        :math:`k_\mathbb{P}(\mathbf{x}, \mathbf{y}) = \nabla_\mathbf{x} \cdot
        \nabla_\mathbf{y}
        k(\mathbf{x}, \mathbf{y}) + \nabla_\mathbf{x} \log f_X(\mathbf{x})
        \cdot \nabla_\mathbf{y} k(\mathbf{x}, \mathbf{y}) + \nabla_\mathbf{y} \log
        f_X(\mathbf{y}) \cdot \nabla_\mathbf{x} k(\mathbf{x}, \mathbf{y}) +
        (\nabla_\mathbf{x} \log f_X(\mathbf{x}) \cdot \nabla_\mathbf{y} \log
        f_X(\mathbf{y})) k(\mathbf{x}, \mathbf{y})`.


        This kernel requires a 'base' kernel to evaluate. The base kernel can be any
        other implemented subclass of the Kernel abstract base class; even another Stein
        kernel.

        The score function :math:`\nabla_\mathbf{x} \log f_X: \mathbb{R}^d \to
        \mathbb{R}^d` can be any suitable Lipschitz score function, e.g. one that is
        learned from score matching (#TODO: link to score matching), computed explicitly
        from a density function, or known analytically.

        :param base_kernel: Initialised kernel object to evaluate the Stein kernel with.
        :param score_function: A vector-valued callable defining a score function.
            :math:`\mathbb{R}^d \to \mathbb{R}^d`.
        :param scale: Output scale to use.
        """
        self.base_kernel = base_kernel
        self.score_function = score_function
        self.scale = scale

        # Initialise parent
        super().__init__(scale=scale)

    def _tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable jit decoration
        of methods inside this class.
        """
        # TODO: score functon is assumed to not change here - but it might if the kernel
        #  changes - but does not work when specified in children
        children = (self.base_kernel,)
        aux_data = {"score_function": self.score_function, "scale": self.scale}
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """
        Reconstructs a pytree from the tree definition and the leaves.

        Arrays & dynamic values (children) and auxiliary data (static values) are
        reconstructed. A method to reconstruct the pytree needs to be specified to
        enable jit decoration of methods inside this class.
        """
        return cls(*children, **aux_data)

    @jit
    def _compute_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Evaluate the Stein kernel on input vectors x and y.

        We assume x and y are two vectors of the same dimension.

        :param x: vector :math:`\mathbf{x} \in \mathbb{R}^d`.
        :param y: vector :math:`\mathbf{y} \in \mathbb{R}^d`.
        :return: kernel evaluated at (x, y).
        """
        k = self.base_kernel._compute_elementwise(x, y)
        div = self.base_kernel._divergence_x_grad_y_elementwise(x, y)
        gkx = self.base_kernel._grad_x_elementwise(x, y)
        gky = self.base_kernel._grad_y_elementwise(x, y)
        score_x = self.score_function(x)
        score_y = self.score_function(y)
        return (
            div
            + jnp.dot(gkx, score_y)
            + jnp.dot(gky, score_x)
            + k * jnp.dot(score_x, score_y)
        )


# Define the pytree node for the added class to ensure methods with jit decorators
# are able to run. We rely on the naming convention that all child classes of Kernel
# include the sub-string Kernel inside of them.
for name, current_class in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    if "Kernel" in name and name != "Kernel":
        tree_util.register_pytree_node(
            current_class, current_class._tree_flatten, current_class._tree_unflatten
        )

# TODO: Laplace kernel
# TODO: Do we want weights to be used to align with MMD?
