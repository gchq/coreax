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

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array, jit, random, tree_util, vmap
from jax.typing import ArrayLike

import coreax.approximation as ca
import coreax.kernel_functions as ckf
import coreax.utils as cu


class Kernel(ABC):
    """
    Base class for kernels.
    """

    def __init__(self):
        r"""
        Define a kernel to measure distances between points in some space.

        Kernels for the basic tool for measuring distances between points in a space,
        and through this constructing representations of the distribution a discrete set
        of samples follow.
        """

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
        # Convert data-types for clarity
        x = jnp.asarray(x)
        y = jnp.asarray(y)

        # Apply vectorised computations if dimensions are 2 or higher, otherwise use
        # elementwise
        if x.ndim < 2 and y.ndim < 2:
            return self._compute_elementwise(x, y)
        else:
            return self._compute_vectorised(x, y)

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

    def _compute_vectorised(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Evaluate the kernel on input data x and y, vectorised.

        Vectorisation only becomes relevant in terms of computational speed when data
        is in 2 or more dimensions.

        :param x: An :math:`n \times d` dataset
        :param y: An :math:`m \times d` dataset
        :return: Distances (as determined by the kernel) between points in x and y
        """
        return vmap(self._compute_elementwise, in_axes=(0, 0), out_axes=0)(x, y)

    def _define_pairwise_kernel_evaluation_no_grads(self) -> cu.KernelFunction:
        """
        Define a callable that returns an evaluation of all pairs of points in inputs.

        Note this is defined to return a callable so that the callable can be decorated
        with jit. We then use the resulting callable elsewhere in the code.

        :return: Callable that, given data inputs x and y, outputs kernel distances
            between all possible pairs of points within
        """
        return ckf.kernel_pairwise_evaluation_no_grads(self._compute_elementwise)

    def _define_pairwise_kernel_evaluation_with_grads(
        self,
    ) -> cu.KernelFunctionWithGrads:
        """
        Define a callable that returns an evaluation of all pairs of points in inputs.

        Note this is defined to return a callable so that the callable can be decorated
        with jit. We then use the resulting callable elsewhere in the code. We supply
        gradients with this callable.

        :return: Callable that, given data inputs x and y, gradients and a base kernel
            bandwidth nu, outputs kernel distances between all possible pairs of points
            within
        """
        return ckf.kernel_pairwise_evaluation_with_grads(self._compute_elementwise)

    # TODO: Weights need to be optional here to agree with metrics approach
    def update_kernel_matrix_row_sum(
        self,
        x: ArrayLike,
        kernel_row_sum: ArrayLike,
        i: int,
        j: int,
        max_size: int,
        kernel_pairwise: cu.KernelFunction | cu.KernelFunctionWithGrads,
        grads: ArrayLike | None = None,
        nu: float | None = None,
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

    # TODO: Weights need to be optional here to agree with metrics approach
    def calculate_kernel_matrix_row_sum(
        self,
        x: ArrayLike,
        max_size: int,
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
        :param k_pairwise: Pairwise kernel evaluation function
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
            kernel_pairwise = self._define_pairwise_kernel_evaluation_no_grads()
        else:
            kernel_pairwise = self._define_pairwise_kernel_evaluation_with_grads()

        # Ensure data format is as required
        x = jnp.asarray(x)
        num_datapoints = len(x)
        kernel_row_sum = jnp.zeros(num_datapoints)
        # Iterate over upper triangular blocks
        for i in range(0, num_datapoints, max_size):
            for j in range(i, num_datapoints, max_size):
                kernel_row_sum = self.update_kernel_matrix_row_sum(
                    x, kernel_row_sum, i, j, max_size, kernel_pairwise, grads, nu
                )
        return kernel_row_sum

    # TODO: Weights need to be optional here to agree with metrics approach
    def calculate_kernel_matrix_row_sum_mean(self, x, max_size, grads, nu) -> Array:
        """
        Compute the mean of the row sum of the kernel matrix.

        :param x: Data matrix, :math:`n \times d`
        :param max_size: Size of matrix block to process
        :param grads: Array of gradients, if applicable, :math:`n \times d`
                      Optional, defaults to `None`
        :param nu: Base kernel bandwidth, if applicable, :math:`n \times d`
                   Optional, defaults to `None`

        The mean of the row sum of the kernel matrix is the mean of the sum of distances
        between a given point and all possible pairs of points that contain this given
        point.
        """
        return self.calculate_kernel_matrix_row_sum(x, max_size, grads, nu) / (
            1.0 * x.shape[0]
        )

    # TODO: Weights need to be optional here to agree with metrics approach
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
        :return: Approximator object
        """
        # Create an approximator object (if needed)
        if not isinstance(approximator, ca.KernelMeanApproximator):
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

        :param approximator: The name of an approximator class to use, or the class
            directly as a dependency injection
        :param kernel_evaluation: Kernel function
            :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
        :param random_key: Key for random number generation
        :param num_kernel_points: Number of kernel evaluation points
        :param num_train_points: Number of training points used to fit kernel regression
        :return: Approximator object
        """
        # TODO: Move strings into constants module
        if isinstance(approximator, str):
            if approximator == "random":
                return ca.RandomApproximator(
                    kernel_evaluation=self._compute_elementwise,
                    random_key=random_key,
                    num_kernel_points=num_kernel_points,
                    num_train_points=num_train_points,
                )
            elif approximator == "annchor":
                return ca.ANNchorApproximator(
                    kernel_evaluation=self._compute_elementwise,
                    random_key=random_key,
                    num_kernel_points=num_kernel_points,
                    num_train_points=num_train_points,
                )
            elif approximator == "nystrom":
                return ca.NystromApproximator(
                    kernel_evaluation=self._compute_elementwise,
                    random_key=random_key,
                    num_kernel_points=num_kernel_points,
                )
            else:
                raise ValueError(f"Approximator choice {approximator} not known.")
        else:
            return approximator

    @abstractmethod
    def grad_x(
        self, x: ArrayLike, y: ArrayLike, gram_matrix: ArrayLike | None = None
    ) -> Array:
        """
        Compute the gradient of the kernel with respect to x.
        """

    @abstractmethod
    def grad_y(
        self, x: ArrayLike, y: ArrayLike, gram_matrix: ArrayLike | None = None
    ) -> Array:
        """
        Compute the gradient of the kernel with respect to y.
        """

    @abstractmethod
    def grad_log_x(
        self,
        x: ArrayLike,
        kde_data: ArrayLike,
        gram_matrix: ArrayLike | None = None,
        kernel_mean: ArrayLike | None = None,
    ) -> Array:
        """
        Compute the gradient of the log-PDF (score function) with respect to x.
        """


class RBFKernel(Kernel):
    """
    Define a radial basis function (RBF) kernel.
    """

    def __init__(self, bandwidth: float = 1.0):
        """
        Define the RBF kernel to measure distances between points in some space.

        :param bandwidth: Kernel bandwidth to use (variance of the underlying Gaussian)
        """
        # Check that bandwidth is above zero (the isinstance check here is to ensure
        # that we don't check a trace of an array when jit decorators interact with
        # code)
        if isinstance(bandwidth, float) and bandwidth <= 0.0:
            raise ValueError(
                f"Bandwidth must be above zero. Current value {bandwidth}."
            )
        self.bandwidth = bandwidth

        # Initialise parent
        super().__init__()

    def _tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable jit decoration
        of methods inside this class.
        """
        children = (self.bandwidth,)
        aux_data = {}
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
        return jnp.exp(-self.sq_dist(x, y) / (2 * self.bandwidth))

    @jit
    def sq_dist(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Calculate the squared distance between two vectors.

        :param x: First vector argument
        :param y: Second vector argument
        :return: Dot product of `x-y` and `x-y`, the square distance between `x` and `y`
        """
        return jnp.dot(x - y, x - y)

    def grad_x(
        self,
        x: ArrayLike,
        y: ArrayLike,
        gram_matrix: ArrayLike | None = None,
    ) -> Array:
        r"""
        Calculate the *element-wise* gradient of the radial basis function w.r.t. x.

        :param x: First set of vectors as a :math:`n \times d` array
        :param y: Second set of vectors as a :math:`m \times d` array
        :param gram_matrix: Gram matrix. Optional, if omitted, defaults to a normalised
            Gaussian kernel
        :return: Gradients at each `x X y` point, an :math:`m \times n \times d` array
        """
        return -self.grad_y(x, y, gram_matrix)

    def grad_y(
        self,
        x: ArrayLike,
        y: ArrayLike,
        gram_matrix: ArrayLike | None = None,
    ) -> Array:
        r"""
        Calculate the *element-wise* gradient of the radial basis function w.r.t. y.

        :param x: First set of vectors as a :math:`n \times d` array
        :param y: Second set of vectors as a :math:`m \times d` array
        :param gram_matrix: Gram matrix. Optional, if omitted, defaults to a normalised
            Gaussian kernel
        :return: Gradients at each `x X y` point, an :math:`m \times n \times d` array
        """
        return ckf.grad_rbf_y(x, y, self.bandwidth, gram_matrix)

    def grad_log_x(
        self,
        x: ArrayLike,
        kde_data: ArrayLike,
        gram_matrix: ArrayLike | None = None,
        kernel_mean: ArrayLike | None = None,
    ) -> Array:
        """
        Compute the gradient of the log-PDF (score function) with respect to x.

        The PDF is constructed from kernel density estimation.

        :param x: An :math:`n \times d` array of random variable values
        :param kde_data: The :math:`m \times d` kernel density estimation set
        :param gram_matrix: Gram matrix, an :math:`n \times m` array. Optional, if
            `gram_matrix` or `kernel_mean` are :data:`None` or omitted, defaults to a
            normalised Gaussian kernel
        :param kernel_mean: Kernel mean, an :math:`n \times 1` array. Optional, if
            `gram_matrix` or `kernel_mean` are :data:`None` or omitted, defaults to the
            mean of a Normalised Gaussian kernel
        :return: An :math:`n \times d` array of gradients evaluated at values of
            `random_var_values`
        """
        return ckf.rbf_grad_log_f_x(
            x, kde_data, self.bandwidth, gram_matrix, kernel_mean
        )


# Define the pytree node for the class RBFKernel to ensure methods with jit decorators
# are able to run
tree_util.register_pytree_node(
    RBFKernel, RBFKernel._tree_flatten, RBFKernel._tree_unflatten
)
