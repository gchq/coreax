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
from jax import Array, jit, random, tree_util, vmap, grad
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

    @jit
    def compute(self, x: ArrayLike, y: ArrayLike) -> Array:
        """
        Evaluate the kernel on input data x and y.

        The 'data' can be any of:
            * floating numbers (so a single data-point in 1-dimension)
            * zero-dimensional arrays (so a single data-point in 1-dimension)
            * a vector (a single-point in multiple dimensions)
            * matrix (multiple points in multiple dimensions).

        :param x: An :math:`n \times d` dataset or a single value (point)
        :param y: An :math:`m \times d` dataset or a single value (point)
        :return: Distances (as determined by the kernel) between points in x and y
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
        Evaluate the kernel on input vectors x and y, not-vectorised.

        Vectorisation only becomes relevant in terms of computational speed when data
        is in 2 or more dimensions.

        :param x: First vector we consider
        :param y: Second vector we consider
        :return: Distance (as determined by the kernel) between point x and y
        """

    @jit
    def grad_x(self, x: ArrayLike, y: ArrayLike) -> Array:
        """
        Compute the gradient of the kernel with respect to x.
        """
        fn = vmap(
            vmap(grad(self._compute_elementwise, 0), in_axes=(0, None), out_axes=0),
            in_axes=(None, 0),
            out_axes=1,
        )
        return fn(x, y)

    @jit
    def grad_y(self, x: ArrayLike, y: ArrayLike) -> Array:
        """
        Compute the gradient of the kernel with respect to y.
        """
        fn = jit(
            vmap(
                vmap(grad(self._compute_elementwise, 1), in_axes=(0, None), out_axes=0),
                in_axes=(None, 0),
                out_axes=1,
            )
        )
        return fn(x, y)

    @abstractmethod
    def compute_divergence_x_grad_y(
        self,
        x: ArrayLike,
        y: ArrayLike,
        num_data_points: int | None = None,
        gram_matrix: ArrayLike | None = None,
    ) -> Array:
        r"""
        Apply divergence operator on gradient of kernel with respect to `y`.

        This avoids explicit computation of the Hessian. Note that the generating set is
        not necessarily the same as `x`.

        :param x: First set of vectors as a :math:`n \times d` array
        :param y: Second set of vectors as a :math:`m \times d` array
        :param num_data_points: Number of data points in the generating set. Optional,
            if :data:`None` or omitted, defaults to number of vectors in `x`
        :param gram_matrix: Gram matrix.
        :return: Divergence operator, an :math:`n \times m` matrix
        """

    @staticmethod
    def update_kernel_matrix_row_sum(
        x: ArrayLike,
        kernel_row_sum: ArrayLike,
        i: int,
        j: int,
        kernel_pairwise: cu.KernelFunction | cu.KernelFunctionWithGrads,
        grads: ArrayLike | None = None,
        nu: float | None = None,
        max_size: int = 10_000,
    ) -> Array:
        """
        Update the row sum of the kernel matrix with a single block of values.

        The row sum of the kernel matrix may involve a large number of pairwise
        computations, so this can be done in blocks to reduce memory requirements.

        The kernel matrix block :math:`i:i+max_size \times j:j+max_size` is used to
        update the row sum. Symmetry of the kernel matrix is exploited to reduced
        repeated calculation.

        Note that `k_pairwise` should be of the form :math:`k(x,y)` if `grads` and `nu`
        are `None`. Else, `k_pairwise` should be of the form
        :math:`k(x,y, grads, grads, n, nu)`.

        :param x: Data matrix, :math:`n \times d`
        :param kernel_row_sum: Full data structure for Gram matrix row sum,
            :math:`1 \times n`
        :param i: Kernel matrix block start
        :param j: Kernel matrix block end
        :param max_size: Size of matrix block to process
        :param kernel_pairwise: Pairwise kernel evaluation function
        :param grads: Array of gradients, if applicable, :math:`n \times d`;
            Optional, defaults to `None`
        :param nu: Base kernel bandwidth. Optional, defaults to `None`
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
                nu,
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
        nu: ArrayLike | None = None,
    ) -> Array:
        """
        Compute the row sum of the kernel matrix.

        The row sum of the kernel matrix is the sum of distances between a given point
        and all possible pairs of points that contain this given point. The row sum is
        calculated block-wise to limit memory overhead.

        Note that `k_pairwise` should be of the form :math:`k(x,y)` if `grads` and `nu`
        are `None`. Else, `k_pairwise` should be of the form
        :math:`k(x,y, grads, grads, n, nu)`.

        :param x: Data matrix, :math:`n \times d`
        :param max_size: Size of matrix block to process
        :param grads: Array of gradients, if applicable, :math:`n \times d`
                      Optional, defaults to `None`
        :param nu: Base kernel bandwidth, if applicable, :math:`n \times d`
                   Optional, defaults to `None`
        :return: Kernel matrix row sum
        """
        # Define the function to call to evaluate the kernel for all pairwise sets of
        # points
        if grads is None:
            # kernel_pairwise = self.compute_pairwise_no_grads
            kernel_pairwise = jit(
                vmap(
                    vmap(self._compute_elementwise, in_axes=(0, None), out_axes=0),
                    in_axes=(None, 0),
                    out_axes=1,
                )
            )
        else:
            # kernel_pairwise = self.compute_pairwise_with_grads
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
                    x, kernel_row_sum, i, j, kernel_pairwise, grads, nu, max_size
                )
        return kernel_row_sum

    def calculate_kernel_matrix_row_sum_mean(
        self,
        x: ArrayLike,
        grads: ArrayLike | None = None,
        lengthscale: ArrayLike | None = None,
        max_size: int = 10_000,
    ) -> Array:
        """
        Compute the mean of the row sum of the kernel matrix.

        The mean of the row sum of the kernel matrix is the mean of the sum of distances
        between a given point and all possible pairs of points that contain this given
        point.

        :param x: Data matrix, :math:`n \times d`
        :param max_size: Size of matrix block to process
        :param grads: Array of gradients, if applicable, :math:`n \times d`
                      Optional, defaults to `None`
        :param lengthscale: Base kernel bandwidth, if applicable, :math:`n \times d`
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
        """
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
        """
        Create an approximator object for use with the kernel matrix row sum mean.

        :param approximator: The name of an approximator class to use, or the
            uninstantiated class directly as a dependency injection
        :param kernel_evaluation: Kernel function
            :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
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
    Define a squared exponential (squared exponential) kernel.
    """

    def __init__(self, lengthscale: float = 1.0):
        """
        Define the squared exponential kernel to measure distances between points in some space.

        :param bandwidth: Kernel bandwidth to use (variance of the underlying Gaussian)
        """
        # Check that bandwidth is above zero (the isinstance check here is to ensure
        # that we don't check a trace of an array when jit decorators interact with
        # code)
        if isinstance(lengthscale, float) and lengthscale <= 0.0:
            raise ValueError(
                f"Bandwidth must be above zero. Current value {lengthscale}."
            )
        self.lengthscale = lengthscale

        # Initialise parent
        super().__init__()

    def _tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable jit decoration
        of methods inside this class.
        """
        # children = (self.lengthscale,)
        children = ()
        aux_data = {"lengthscale": self.lengthscale}
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
        Evaluate the kernel on input vectors x and y.

        We assume x and y are two vectors of the same length. We compute the distance
        between these two vectors, as determined by the selected kernel.

        :param x: First vector we consider
        :param y: Second vector we consider
        :return: Distance (as determined by the kernel) between point x and y
        """
        return jnp.exp(-cu.sq_dist(x, y) / (2 * self.lengthscale**2))

    @jit
    def _grad_y_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        return (x - y) / self.lengthscale**2 * self._compute_elementwise(x, y)

    @jit
    def grad_y(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Calculate the *element-wise* gradient of the squared exponential w.r.t. y.

        :param x: First set of vectors as a :math:`n \times d` array
        :param y: Second set of vectors as a :math:`m \times d` array
        :return: Gradients at each `x X y` point, an :math:`m \times n \times d` array
        """
        fn = vmap(
            vmap(self._grad_y_elementwise, in_axes=(0, None), out_axes=0),
            in_axes=(None, 0),
            out_axes=1,
        )
        return fn(x, y)

    def grad_x(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Calculate the *element-wise* gradient of the squared exponential w.r.t. x.

        :param x: First set of vectors as a :math:`n \times d` array
        :param y: Second set of vectors as a :math:`m \times d` array
        :return: Gradients at each `x X y` point, an :math:`m \times n \times d` array
        """
        return -self.grad_y(x, y)

    @jit
    def compute_divergence_x_grad_y(
        self,
        x: ArrayLike,
        y: ArrayLike,
        num_data_points: int | None = None,
        gram_matrix: ArrayLike | None = None,
    ) -> Array:
        r"""
        Apply divergence operator on gradient of squared exponential kernel with respect to `y`.

        This avoids explicit computation of the Hessian. Note that the generating set is
        not necessarily the same as `x_array`.

        :param x: First set of vectors as a :math:`n \times d` array
        :param y: Second set of vectors as a :math:`m \times d` array
        :param num_data_points: Number of data points in the generating set. Optional,
            if :data:`None` or omitted, defaults to number of vectors in `x`
        :param gram_matrix: Gram matrix. Optional, if :data:`None` or omitted, defaults
            to a normalised Gaussian kernel
        :return: Divergence operator, an :math:`n \times m` matrix
        """
        # TODO: Test this
        x = jnp.asarray(x)
        if gram_matrix is None:
            gram_matrix = self.compute(x, y)
        if num_data_points is None:
            num_data_points = x.shape[0]

        return (
            gram_matrix
            / self.lengthscale
            * (num_data_points - cu.sq_dist_pairwise(x, y) / self.lengthscale)
        )


class PCIMQKernel(Kernel):
    """
    Define a pre-conditioned inverse multi-quadric (pcimq) kernel.
    """

    def __init__(self, lengthscale: float = 1.0):
        """
        Define the pcimq kernel to measure distances between points in some space.

        :param bandwidth: Kernel bandwidth to use (variance of the underlying Gaussian)
        """
        # Check that bandwidth is above zero (the isinstance check here is to ensure
        # that we don't check a trace of an array when jit decorators interact with
        # code)
        if isinstance(lengthscale, float) and lengthscale <= 0.0:
            raise ValueError(
                f"Bandwidth must be above zero. Current value {lengthscale}."
            )
        self.lengthscale = lengthscale

        # Initialise parent
        super().__init__()

    def _tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable jit decoration
        of methods inside this class.
        """
        # children = (self.bandwidth,)
        children = ()
        aux_data = {"lengthscale": self.lengthscale}
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
        Evaluate the kernel on input vectors x and y.

        We assume x and y are two vectors of the same length. We compute the distance
        between these two vectors, as determined by the selected kernel.

        :param x: First vector we consider
        :param y: Second vector we consider
        :return: Distance (as determined by the kernel) between point x and y
        """
        scaling = 2 * self.lengthscale**2
        mq_array = cu.sq_dist(x, y) / scaling
        return 1 / jnp.sqrt(1 + mq_array)

    @jit
    def _grad_y_elementwise(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        return (
            (x - y) / (2 * self.lengthscale**2) * self._compute_elementwise(x, y) ** 3
        )

    @jit
    def grad_y(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Calculate the *element-wise* gradient of the PCIMQ w.r.t. y.

        :param x: First set of vectors as a :math:`n \times d` array
        :param y: Second set of vectors as a :math:`m \times d` array
        :return: Gradients at each `x X y` point, an :math:`m \times n \times d` array
        """
        fn = vmap(
            vmap(self._grad_y_elementwise, in_axes=(0, None), out_axes=0),
            in_axes=(None, 0),
            out_axes=1,
        )
        return fn(x, y)

    def grad_x(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Calculate the *element-wise* gradient of the PCIMQ w.r.t. x.

        :param x: First set of vectors as a :math:`n \times d` array
        :param y: Second set of vectors as a :math:`m \times d` array
        :return: Gradients at each `x X y` point, an :math:`m \times n \times d` array
        """
        return -self.grad_y(x, y)

    @jit
    def compute_divergence_x_grad_y(
        self,
        x: ArrayLike,
        y: ArrayLike,
        num_data_points: int | None = None,
        gram_matrix: ArrayLike | None = None,
    ) -> Array:
        r"""
        Apply divergence operator on gradient of pcimq kernel with respect to `y`.

        This avoids explicit computation of the Hessian. Note that the generating set is
        not necessarily the same as `x_array`.

        :param x: First set of vectors as a :math:`n \times d` array
        :param y: Second set of vectors as a :math:`m \times d` array
        :param num_data_points: Number of data points in the generating set. Optional,
            if :data:`None` or omitted, defaults to number of vectors in `x`
        :param gram_matrix: Gram matrix. Optional, if :data:`None` or omitted, defaults
            to a normalised Gaussian kernel
        :return: Divergence operator, an :math:`n \times m` matrix
        """
        # TODO: Test this
        # TODO: Should we call self.compute_normalised or self.compute for the
        #  gram_matrix?
        scaling = 2 * self.lengthscale**2
        x = jnp.asarray(x)
        if gram_matrix is None:
            gram_matrix = self.compute(x, y)
        if num_data_points is None:
            num_data_points = x.shape[0]
        return (
            num_data_points / scaling * gram_matrix**3
            - 3 * cu.sq_dist_pairwise(x, y) / scaling**2 * gram_matrix**5
        )


class SteinKernel:
    """
    Define a Stein kernel.

    TODO: We'll need kernel matrix row sum methods here, but not all of the other kernel
        methods. We could either make this a child class of Kernel and just raise a
        NotImplimentedError for the irrelevant parts, or we could copy/paste and adjust
        the kernel row sum parts as needed.

    TODO: Note we might need to edit how the pairwise vmapped functions work in this
        class when using the kernel matrix row sum parts.
    """

    def __init__(
        self,
        base_kernel: Kernel,
        score_function: Callable[[ArrayLike, ArrayLike], Array] | None,
    ):
        """
        Define the Stein kernel to measure distances between points in some space.

        This kernel requires a 'base' kernel to evaluate. The base kernel can be any
        other implemented subclass of the Kernel abstract base class.

        The score function is the gradient of the log-density function of a
        distribution. We can either approximate this via kernel density estimation (done
        if score_function is None) or model it directly (for example, through score
        matching techniques). In the latter case, a callable can be supplied as is
        treated as the score function.

        :param base_kernel: Initialised kernel object to evaluate the Stein kernel with
        :param score_function: If none, we determine the score function via kernel
            density estimation using base_kernel. Otherwise, a callable defining
            the score function (or an approximation to it).
        """
        self.base_kernel = base_kernel
        if score_function is None:
            # If no score function has been provided, then we use the base-kernels
            # grad_log_x method
            self.score_function = self.base_kernel.grad_log_x
        else:
            # If an alternative score function is passed (e.g. a pre-determined
            # analytical function, or a neural network callable) we assign this
            self.score_function = score_function

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
        aux_data = {"score_function": self.score_function}
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
    def compute(
        self,
        x: ArrayLike,
        y: ArrayLike,
    ) -> Array:
        r"""
        Compute a kernel from a base kernel with the canonical Stein operator.

        :param x: First set of vectors as a :math:`n \times d` array
        :param y: Second set of vectors as a :math:`n \times d` array
        :return: Gram matrix, an :math:`n \times m` array
        """
        # Format data
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        x_size = x.shape[0]
        y_size = y.shape[0]

        # n x m
        kernel_gram_matrix = self.base_kernel.compute(x, y)

        # n x m
        divergence = self.base_kernel.compute_divergence_x_grad_y(
            x, y, x_size, kernel_gram_matrix
        )

        # n x m x d
        # grad_k_x = self.base_kernel.grad_x(x, y, kernel_gram_matrix)
        grad_k_x = self.base_kernel.grad_x(x, y)

        # m x n x d
        # grad_k_y = jnp.transpose(
        #     self.base_kernel.grad_y(x, y, kernel_gram_matrix), (1, 0, 2)
        # )
        grad_k_y = jnp.transpose(self.base_kernel.grad_y(x, y), (1, 0, 2))

        # n x d
        grad_log_p_x = self.score_function(x, y)

        # m x d
        grad_log_p_y = self.score_function(y, x)

        # m x n x d
        tiled_grad_log_x = jnp.tile(grad_log_p_x, (y_size, 1, 1))
        # n x m x d
        tiled_grad_log_y = jnp.tile(grad_log_p_y, (x_size, 1, 1))
        # m x n
        x = jnp.einsum("ijk,ijk -> ij", tiled_grad_log_x, grad_k_y)
        # n x m
        y = jnp.einsum("ijk,ijk -> ij", tiled_grad_log_y, grad_k_x)
        # n x m
        z = jnp.dot(grad_log_p_x, grad_log_p_y.T) * kernel_gram_matrix
        return divergence + x.T + y + z

    @jit
    def compute_element(
        self,
        x: ArrayLike,
        y: ArrayLike,
        grad_log_p_x: ArrayLike,
        grad_log_p_y: ArrayLike,
        dimension: int,
    ) -> Array:
        r"""
        Evaluate the kernel element at `(x,y)`.

        This element is induced by the canonical Stein operator on a base kernel. The
        log-PDF can be arbitrary as only gradients are supplied.

        :param x: First vector as a :math:`1 \times d` array
        :param y: Second vector as a :math:`1 \times d` array
        :param grad_log_p_x: Gradient of log-PDF evaluated at `x`, a :math:`1 \times d`
            array
        :param grad_log_p_y: Gradient of log-PDF evaluated at `y`, a :math:`1 \times d`
            array
        :param dimension: Dimension of the input data.
        :return: Kernel evaluation at `(x,y)`, 0-dimensional array
        """
        # Format data
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)
        grad_log_p_x = jnp.atleast_2d(grad_log_p_x)
        grad_log_p_y = jnp.atleast_2d(grad_log_p_y)

        # n x m
        kernel_gram_matrix = self.base_kernel.compute(x, y)

        # n x m
        divergence = self.base_kernel.compute_divergence_x_grad_y(
            x, y, dimension, kernel_gram_matrix
        )

        # n x m x d
        # grad_p_x = jnp.squeeze(self.base_kernel.grad_x(x, y, kernel_gram_matrix))
        grad_p_x = jnp.squeeze(self.base_kernel.grad_x(x, y))

        # m x n x d
        # grad_p_y = jnp.squeeze(self.base_kernel.grad_y(x, y, kernel_gram_matrix))
        grad_p_y = jnp.squeeze(self.base_kernel.grad_y(x, y))

        x_ = jnp.dot(grad_log_p_x, grad_p_y)
        # n x m
        y_ = jnp.dot(grad_log_p_y, grad_p_x)
        # n x m
        z = jnp.dot(grad_log_p_x, grad_log_p_y.T) * kernel_gram_matrix

        kernel = divergence + x_.T + y_ + z
        return kernel[0, 0]


# Define the pytree node for the added class to ensure methods with jit decorators
# are able to run. We rely on the naming convention that all child classes of Kernel
# include the sub-string Kernel inside of them.
for name, current_class in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    if "Kernel" in name and name != "Kernel":
        tree_util.register_pytree_node(
            current_class, current_class._tree_flatten, current_class._tree_unflatten
        )

# TODO: Squared exponential kernel
# TODO: Gaussian density kernel
# TODO: Laplace kernel
# TODO: PCIMQ kernel
# TODO: Include divergence bits
# TODO: Do we want weights to be used to align with MMD?
