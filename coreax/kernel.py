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

import jax.numpy as jnp
from jax import Array, jit, random, tree_util, vmap
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

    def __init__(self):
        r"""
        Define a kernel to measure distances between points in some space.

        Kernels for the basic tool for measuring distances between points in a space,
        and through this constructing representations of the distribution a discrete set
        of samples follow.
        """
        # Define helper-functions for ease of computation - this assigns a callable
        # function that is jit compiled
        self.compute_pairwise_no_grads = jit(
            vmap(
                vmap(self._compute_elementwise, in_axes=(None, 0), out_axes=0),
                in_axes=(0, None),
                out_axes=0,
            )
        )
        # TODO: Test this method
        self.compute_pairwise_with_grads = jit(
            vmap(
                vmap(self._compute_elementwise, (None, 0, None, 0, None, None), 0),
                (0, None, 0, None, None, None),
                0,
            )
        )

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
    def compute_normalised(self, x: ArrayLike, y: ArrayLike) -> Array:
        """
        Compute the normalised kernel output
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

    # TODO: Weights need to be optional here to agree with metrics approach
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

    # TODO: Weights need to be optional here to agree with metrics approach
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
            kernel_pairwise = self.compute_pairwise_no_grads
        else:
            kernel_pairwise = self.compute_pairwise_with_grads

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

    # TODO: Weights need to be optional here to agree with metrics approach
    def calculate_kernel_matrix_row_sum_mean(
        self,
        x: ArrayLike,
        grads: ArrayLike | None = None,
        nu: ArrayLike | None = None,
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
        :param nu: Base kernel bandwidth, if applicable, :math:`n \times d`
                   Optional, defaults to `None`
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
        return jnp.exp(-cu.sq_dist(x, y) / (2 * self.bandwidth))

    @jit
    def compute_normalised(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Evaluate the normalised Gaussian kernel pairwise.

        :param x: First set of vectors as a :math:`n \times d` array
        :param y: Second set of vectors as a :math:`m \times d` array
        :return: Pairwise kernel evaluations for normalised RBF kernel
        """
        # TODO: Do we need to square root the bandwidth here so it's consistent with
        #  the interpretation that bandwidth = variance of the kernel?
        square_distances = cu.sq_dist_pairwise(x, y)
        kernel = jnp.exp(-0.5 * square_distances / self.bandwidth**2) / jnp.sqrt(
            2 * jnp.pi
        )
        return kernel / self.bandwidth

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

    @jit
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
        if gram_matrix is None:
            gram_matrix = self.compute_normalised(x, y)
        else:
            gram_matrix = jnp.asarray(gram_matrix)
        distances = cu.pdiff(x, y)
        return distances * gram_matrix[:, :, None] / self.bandwidth**2

    @jit
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
        x = jnp.atleast_2d(x)
        kde_data = jnp.atleast_2d(kde_data)
        if gram_matrix is None or kernel_mean is None:
            kernel_mean, gram_matrix = self.construct_pdf(x, kde_data)
        else:
            kernel_mean = jnp.asarray(kernel_mean)
        gradients = self.grad_x(x, kde_data, gram_matrix).mean(axis=1)

        return gradients / kernel_mean[:, None]

    @jit
    def construct_pdf(self, x: ArrayLike, kde_data: ArrayLike) -> tuple[Array, Array]:
        r"""
        Construct PDF of `x` by kernel density estimation for a radial basis function.

        :param x: An :math:`n \times d` array of random variable values
        :param kde_data: The :math:`m \times d` kernel density estimation set
        :return: Gram matrix mean over `random_var_values` as a :math:`n \times 1` array;
            Gram matrix as an :math:`n \times m` array
        """
        kernel = self.compute_normalised(x, kde_data)
        kernel_mean = kernel.mean(axis=1)
        return kernel_mean, kernel

    @jit
    def compute_divergence_x_grad_y(
        self,
        x: ArrayLike,
        y: ArrayLike,
        num_data_points: int | None = None,
        gram_matrix: ArrayLike | None = None,
    ) -> Array:
        r"""
        Apply divergence operator on gradient of RBF kernel with respect to `y`.

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
            gram_matrix = self.compute_normalised(x, y)
        if num_data_points is None:
            num_data_points = x.shape[0]

        return (
            gram_matrix
            / self.bandwidth
            * (num_data_points - cu.sq_dist_pairwise(x, y) / self.bandwidth)
        )


class PCIMQKernel(Kernel):
    """
    Define a pre-conditioned inverse multi-quadric (pcimq) kernel.
    """

    def __init__(self, bandwidth: float = 1.0):
        """
        Define the pcimq kernel to measure distances between points in some space.

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
        scaling = 2 * self.bandwidth**2
        mq_array = cu.sq_dist_pairwise(x, y) / scaling
        return 1 / jnp.sqrt(1 + mq_array)

    @jit
    def compute_normalised(self, x: ArrayLike, y: ArrayLike) -> Array:
        r"""
        Evaluate the normalised Gaussian kernel pairwise.

        :param x: First set of vectors as a :math:`n \times d` array
        :param y: Second set of vectors as a :math:`m \times d` array
        :return: Pairwise kernel evaluations for normalised RBF kernel
        """
        raise NotImplementedError

    def grad_x(
        self,
        x: ArrayLike,
        y: ArrayLike,
        gram_matrix: ArrayLike | None = None,
    ) -> Array:
        r"""
        Calculate the *element-wise* gradient of the PCIMQ function w.r.t. x.

        :param x: First set of vectors as a :math:`n \times d` array
        :param y: Second set of vectors as a :math:`m \times d` array
        :param gram_matrix: Gram matrix. Optional, if omitted, defaults to a normalised
            Gaussian kernel
        :return: Gradients at each `x X y` point, an :math:`m \times n \times d` array
        """
        return -self.grad_y(x, y, gram_matrix)

    @jit
    def grad_y(
        self,
        x: ArrayLike,
        y: ArrayLike,
        gram_matrix: ArrayLike | None = None,
    ) -> Array:
        r"""
        Calculate the *element-wise* gradient of the PCIMQ function w.r.t. y.

        :param x: First set of vectors as a :math:`n \times d` array
        :param y: Second set of vectors as a :math:`m \times d` array
        :param gram_matrix: Gram matrix. Optional, if omitted, defaults to a normalised
            Gaussian kernel
        :return: Gradients at each `x X y` point, an :math:`m \times n \times d` array
        """
        scaling = 2 * self.bandwidth**2
        if gram_matrix is None:
            gram_matrix = self.compute(x, y)
        else:
            gram_matrix = jnp.asarray(gram_matrix)
        mq_array = cu.pdiff(x, y)

        return gram_matrix[:, :, None] ** 3 * mq_array / scaling

    @jit
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
        # TODO: Implement & test
        raise NotImplementedError

    @jit
    def construct_pdf(self, x: ArrayLike, kde_data: ArrayLike) -> tuple[Array, Array]:
        r"""
        Construct PDF of `x` by kernel density estimation for a radial basis function.

        :param x: An :math:`n \times d` array of random variable values
        :param kde_data: The :math:`m \times d` kernel density estimation set
        :return: Gram matrix mean over `random_var_values` as a :math:`n \times 1` array;
            Gram matrix as an :math:`n \times m` array
        """
        # TODO: Implement & test
        raise NotImplementedError

    @jit
    def compute_divergence_x_grad_y(
        self,
        x: ArrayLike,
        y: ArrayLike,
        num_data_points: int | None = None,
        gram_matrix: ArrayLike | None = None,
    ) -> Array:
        r"""
        Apply divergence operator on gradient of RBF kernel with respect to `y`.

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
        # TODO: Implement & test
        raise NotImplementedError


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
# TODO: Stein kernels
# TODO: Include divergence bits
# TODO: Do we want weights to be used to align with MMD?
