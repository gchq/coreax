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

import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import jit, vmap, Array


@jit
def sq_dist(x: ArrayLike, y: ArrayLike) -> Array:
    """ Squared distance between two vectors

    Args:
        x: First vector
        y: Second vector

    Returns:
        ndarray: same format as numpy.dot
    """
    return jnp.dot(x - y, x - y)


@jit
def sq_dist_pairwise(x_array: ArrayLike, y_array: ArrayLike) -> Array:
    """ Efficient pairwise square distance

    Args:
        x_array: First set of vectors, n x d
        y_array: Second set of vectors, m x d

    Returns:
        Pairwise squared distances, n x m
    """
    # Use vmap to turn distance between individual vectors into a pairwise distance.
    d1 = vmap(sq_dist, in_axes=(None, 0), out_axes=0)
    d2 = vmap(d1, in_axes=(0, None), out_axes=0)

    return d2(x_array, y_array)


@jit
def rbf_kernel(x: ArrayLike, y: ArrayLike, variance: float = 1.) -> Array:
    """Squared exponential kernel for a pair of individual vectors

    Args:
        x: First vector.
        y: Second vector.
        variance: Variance parameter. Optional, defaults to 1.

    Returns:
        RBF kernel evaluated at x, y
    """
    return jnp.exp(-sq_dist(x, y) / (2 * variance))


@jit
def laplace_kernel(x: ArrayLike, y: ArrayLike, variance: float = 1.) -> Array:
    """Laplace kernel for a pair of individual vectors

    Args:
        x: First vector.
        y: Second vector.
        variance: Variance parameter. Optional, defaults to 1.

    Returns:
        Laplace kernel evaluated at x, y
    """
    return jnp.exp(-jnp.linalg.norm(x - y) / (2 * variance))


@jit
def diff(x: ArrayLike, y: ArrayLike) -> Array:
    """Vector difference for a pair of individual vectors

    Args:
        x: First vector.
        y: Second vector.

    Returns:
        Vector difference
    """
    return x - y


@jit
def pdiff(x_array: ArrayLike, y_array: ArrayLike) -> Array:
    """Efficient pairwise difference for two arrays of vectors

    Args:
        x_array: First set of vectors, n x d
        y_array: Second set of vectors, m x d

    Returns:
        Pairwise differences, n x m x d
    """
    d1 = vmap(diff, in_axes=(0, None), out_axes=0)
    d2 = vmap(d1, in_axes=(None, 0), out_axes=1)

    return d2(x_array, y_array)


@jit
def normalised_rbf(x_array: ArrayLike, y_array: ArrayLike, bandwidth: float = 1.) -> Array:
    """Normalised Gaussian kernel, pairwise.

    Args:
        x_array: First set of vectors, n x d.
        y_array: Second set of vectors, m x d.
        bandwidth: Kernel bandwidth (std dev). Defaults to 1.

    Returns:
        Pairwise kernel evaluations.
    """
    square_distances = sq_dist_pairwise(x_array, y_array)
    kernel = jnp.exp(-.5 * square_distances / bandwidth ** 2) / jnp.sqrt(2 * jnp.pi)

    return kernel / bandwidth


@jit
def pc_imq(x_array: ArrayLike, y_array: ArrayLike, bandwidth: float = 1.) -> Array:
    """Preconditioned inverse multi-quadric kernel

    Args:
        x_array: First set of vectors, n x d.
        y_array: Second set of vectors, m x d.
        bandwidth: Kernel bandwidth (std dev). Defaults to 1.

    Returns:
        Pairwise kernel evaluations.
    """
    scaling = 2 * bandwidth ** 2
    mq_array = sq_dist_pairwise(x_array, y_array) / scaling
    kernel = 1 / jnp.sqrt(1 + mq_array)

    return kernel


@jit
def grad_rbf_y(
        x_array: ArrayLike,
        y_array: ArrayLike,
        bandwidth: float = 1.,
        gram_matrix: ArrayLike | None = None,
) -> Array:
    """Gradient of normalised RBF wrt Y

    Args:
        x_array: First set of vectors, n x d.
        y_array: Second set of vectors, m x d.
        bandwidth: Kernel bandwidth (std dev). Defaults to 1.
        gram_matrix: Gram matrix, if available. Defaults to a Normalised Gaussian kernel.

    Returns:
        Gradients at each X x Y point, m x n x d
    """
    if gram_matrix is None:
        gram_matrix = normalised_rbf(y_array, x_array, bandwidth=bandwidth)
    else:
        gram_matrix = jnp.asarray(gram_matrix)

    distances = pdiff(y_array, x_array)

    return distances * gram_matrix[:, :, None] / bandwidth


@jit
def grad_rbf_x(
        x_array: ArrayLike,
        y_array: ArrayLike,
        bandwidth: float = 1.,
        gram_matrix: ArrayLike | None = None,
) -> Array:
    """Gradient of normalised RBF wrt X

    Args:
        x_array: First set of vectors, n x d.
        y_array: Second set of vectors, m x d.
        bandwidth: Kernel bandwidth (std dev). Defaults to 1.
        gram_matrix: Gram matrix, if available. Defaults to a Normalised Gaussian kernel.

    Returns:
        Gradients at each X x Y point, n x m x d
    """
    return -jnp.transpose(grad_rbf_y(x_array, y_array, bandwidth, gram_matrix), (1, 0, 2))


@jit
def grad_pc_imq_y(
        x_array: ArrayLike,
        y_array: ArrayLike,
        bandwidth: float = 1.,
        gram_matrix: ArrayLike | None = None,
) -> Array:
    """Gradient of pre-conditioned inverse multi-quadric wrt Y

    Args:
        x_array: First set of vectors, n x d.
        y_array: Second set of vectors, m x d.
        bandwidth: Kernel bandwidth (std dev). Defaults to 1.
        gram_matrix: Gram matrix, if available. Defaults to a Preconditioned inverse multi-quadric kernel.

    Returns:
        Gradients at each X x Y point, m x n x d
    """
    scaling = 2 * bandwidth**2
    if gram_matrix is None:
        gram_matrix = pc_imq(y_array, x_array, bandwidth)
    else:
        gram_matrix = jnp.asarray(gram_matrix)
    mq_array = pdiff(y_array, x_array)

    return gram_matrix[:, :, None]**3 * mq_array / scaling


@jit
def grad_pc_imq_x(
        x_array: ArrayLike,
        y_array: ArrayLike,
        bandwidth: float = 1.,
        gram_matrix: ArrayLike | None = None,
) -> Array:
    """Gradient of pre-conditioned inverse multi-quadric wrt X

    Args:
        x_array: First set of vectors, n x d.
        y_array: Second set of vectors, m x d.
        bandwidth: Kernel bandwidth (std dev). Defaults to 1.
        gram_matrix: Gram matrix, if available. Defaults to a Preconditioned inverse multi-quadric kernel.

    Returns:
        Gradients at each X x Y point, n x m x d
    """
    return -jnp.transpose(grad_pc_imq_y(x_array, y_array, bandwidth, gram_matrix), (1, 0, 2))


@jit
def rbf_div_x_grad_y(
        x_array: ArrayLike,
        y_array: ArrayLike,
        bandwidth: float = 1.,
        num_data_points: int | None = None,
        gram_matrix: ArrayLike | None = None,
) -> Array:
    """Divergence operator acting on gradient of RBF kernel wrt Y. Avoids explicit computation of the Hessian.

    Args:
        x_array: First set of vectors, n x d.
        y_array: Second set of vectors, m x d.
        bandwidth: Kernel bandwidth (std dev). Defaults to 1.
        num_data_points: The number of data points in the _generating_ set (not necessarily the same as X). Defaults to None.
        gram_matrix: Gram matrix, if available. Defaults to a Normalised Gaussian kernel.

    Returns:
        n x m matrix
    """
    x_array = jnp.asarray(x_array)
    if gram_matrix is None:
        gram_matrix = normalised_rbf(x_array, y_array, bandwidth=bandwidth)
    if num_data_points is None:
        num_data_points = x_array.shape[0]

    return gram_matrix / bandwidth * (num_data_points - sq_dist_pairwise(x_array, y_array) / bandwidth)


@jit
def pc_imq_div_x_grad_y(
        x_array: ArrayLike,
        y_array: ArrayLike,
        bandwidth: float = 1.,
        num_data_points: int = None,
        gram_matrix: ArrayLike | None = None,
) -> Array:
    """Divergence operator acting on gradient of PC-IMQ kernel wrt Y. Avoids explicit computation of the Hessian.

    Args:
        x_array: First set of vectors, n x d.
        y_array: Second set of vectors, m x d.
        bandwidth: Kernel bandwidth (std dev). Defaults to 1.
        num_data_points: The number of data points in the _generating_ set (not necessarily the same as X). Defaults to number of vectors in X.
        gram_matrix: Gram matrix, if available. Defaults to a Normalised Gaussian kernel.

    Returns:
       n x m matrix
    """
    scaling = 2 * bandwidth ** 2
    x_array = jnp.asarray(x_array)
    if gram_matrix is None:
        gram_matrix = pc_imq(x_array, y_array, bandwidth=bandwidth)
    if num_data_points is None:
        num_data_points = x_array.shape[0]
    return num_data_points / \
        scaling * gram_matrix ** 3 - 3 * sq_dist_pairwise(x_array, y_array) / scaling ** 2 * gram_matrix ** 5


@jit
def median_heuristic(x_array: ArrayLike) -> Array:
    """Compute the median heuristic for setting kernel bandwidth

    Args:
        x_array: Input array of vectors.

    Returns:
        Bandwidth parameter, computed from the median heuristic, as a zero-dimensional
        array
    """
    # calculate square distances as an upper triangular matrix
    square_distances = jnp.triu(sq_dist_pairwise(x_array, x_array), k=1)
    # calculate the median
    median_square_distance = jnp.median(square_distances[jnp.triu_indices_from(square_distances, k=1)])
    
    return jnp.sqrt(median_square_distance / 2.)


@jit
def rbf_f_x(random_var_values: ArrayLike, kde_data: ArrayLike, bandwidth: float) -> tuple[Array, Array]:
    """PDF of X, as constructed by an RBF KDE using data set D

    Args:
        random_var_values: Random variable values, n x d
        kde_data: KDE data set, m x d
        bandwidth: Kernel bandwidth (std dev).

    Returns:
        Gram matrix mean over Y, n x 1; Gram matrix, n x m
    """
    kernel = normalised_rbf(random_var_values, kde_data, bandwidth)
    kernel_mean = kernel.mean(axis=1)
    
    return kernel_mean, kernel


@jit
def rbf_grad_log_f_x(
        random_var_values: ArrayLike,
        kde_data: ArrayLike,
        bandwidth: float,
        gram_matrix: ArrayLike | None = None,
        kernel_mean: ArrayLike | None = None,
) -> Array:
    """Gradient of log PDF of X, where the PDF is a KDE induced by data set D.

    Args:
        random_var_values: Random variable values, n x d
        kde_data: KDE data set, m x d
        bandwidth: Kernel bandwidth (std dev).
        gram_matrix: Gram matrix, if available, n x m. Defaults to a Normalised Gaussian kernel.
        kernel_mean: Kernel mean, if available, n x 1. Defaults to a mean of a Normalised Gaussian kernel.

    Returns:
        Array of gradients evaluated at values of X, n x d.
    """
    random_var_values = jnp.atleast_2d(random_var_values)
    kde_data = jnp.atleast_2d(kde_data)
    if gram_matrix is None or kernel_mean is None:
        kernel_mean, gram_matrix = rbf_f_x(random_var_values, kde_data, bandwidth)
    else:
        kernel_mean = jnp.asarray(kernel_mean)
    num_kde_points = kde_data.shape[0]
    gradients = grad_rbf_x(random_var_values, kde_data, bandwidth, gram_matrix).mean(axis=1)
    
    return gradients / (num_kde_points * kernel_mean[:, None])


@jit
def grad_rbf_x(
        x_array: ArrayLike,
        y_array: ArrayLike,
        bandwidth: float,
        kernel: ArrayLike | None = None,
) -> Array:
    """Gradient of the RBF kernel, wrt X

    Args:
        x_array: First set of vectors, n x d.
        y_array: Second set of vectors, m x d.
        bandwidth: Kernel bandwidth (std dev).
        kernel: Gram matrix, if available, n x m. Defaults to None.

    Returns:
        Array of gradients evaluated at values of X, n x d.
    """
    if kernel is None:
        kernel = normalised_rbf(x_array, y_array, bandwidth)
    else:
        kernel = jnp.asarray(kernel)

    scaled_distances = -pdiff(x_array, y_array) / bandwidth
    return scaled_distances * kernel[:, :, None]


@jit
def grad_rbf_y(
        x_array: ArrayLike,
        y_array: ArrayLike,
        bandwidth: float,
        gram_matrix: ArrayLike | None = None,
) -> Array:
    """Gradient of the RBF kernel, wrt Y

    Args:
        x_array: First set of vectors, n x d.
        y_array: Second set of vectors, m x d.
        bandwidth: Kernel bandwidth (std dev).
        gram_matrix: Gram matrix, if available, n x m. Defaults to None.

    Returns:
        Array of gradients evaluated at values of Y, m x d.
    """
    return -jnp.transpose(grad_rbf_x(x_array, y_array, bandwidth, gram_matrix), (1, 0, 2))


@jit
def stein_kernel_rbf(x_array: ArrayLike, y_array: ArrayLike, bandwidth: float = 1.) -> Array:
    """Compute the kernel induced by the canonical Stein operator on an RBF base kernel.

    Args:
        x_array: First set of vectors, n x d.
        y_array: Second set of vectors, m x d.
        bandwidth: Base kernel bandwidth (std dev).

    Returns:
        Gram matrix, n x m
    """
    x_array = jnp.atleast_2d(x_array)
    y_array = jnp.atleast_2d(y_array)
    x_size = x_array.shape[0]
    y_size = y_array.shape[0]
    # n x m
    rbf_kernel = normalised_rbf(x_array, y_array, bandwidth)
    # n x m
    divergence = rbf_div_x_grad_y(x_array, y_array, bandwidth, x_size, rbf_kernel)
    # n x m x d
    grad_k_x = grad_rbf_x(x_array, y_array, bandwidth, rbf_kernel)
    # m x n x d
    grad_k_y = grad_rbf_y(x_array, y_array, bandwidth, rbf_kernel)
    # n x d
    grad_log_p_x = rbf_grad_log_f_x(x_array, y_array, bandwidth)
    # m x d
    grad_log_p_y = rbf_grad_log_f_x(y_array, x_array, bandwidth)
    # m x n x d
    tiled_grad_log_x = jnp.tile(grad_log_p_x, (y_size, 1, 1))
    # n x m x d
    tiled_grad_log_y = jnp.tile(grad_log_p_y, (x_size, 1, 1))
    # m x n
    x = jnp.einsum("ijk,ijk -> ij", tiled_grad_log_x, grad_k_y)
    # n x m
    y = jnp.einsum("ijk,ijk -> ij", tiled_grad_log_y, grad_k_x)
    # n x m
    z = jnp.dot(grad_log_p_x, grad_log_p_y.T) * rbf_kernel
    return divergence + x.T + y + z


@jit
def stein_kernel_pc_imq(x_array: ArrayLike, y_array: ArrayLike, bandwidth: float = 1.) -> Array:
    """Compute the kernel Gram matrix induced by the canonical Stein operator on a pre-conditioned inverse multi-quadric base kernel.

    The log PDF is assumed to be a KDE induced by the data in Y.

    Args:
        x_array: First set of vectors, n x d.
        y_array: Second set of vectors, m x d.
        bandwidth: Base kernel bandwidth (std dev). Defaults to 1

    Returns:
        Gram matrix, n x m
    """
    x_array = jnp.atleast_2d(x_array)
    y_array = jnp.atleast_2d(y_array)
    x_size = x_array.shape[0]
    y_size = y_array.shape[0]
    # n x m
    pc_imq_kernel = pc_imq(x_array, y_array, bandwidth)
    # n x m
    divergence = pc_imq_div_x_grad_y(x_array, y_array, bandwidth, x_size, pc_imq_kernel)
    # n x m x d
    grad_k_x = grad_pc_imq_x(x_array, y_array, bandwidth, pc_imq_kernel)
    # m x n x d
    grad_k_y = grad_pc_imq_y(x_array, y_array, bandwidth, pc_imq_kernel)
    # n x d
    grad_log_p_x = rbf_grad_log_f_x(x_array, y_array, bandwidth)
    # m x d
    grad_log_p_y = rbf_grad_log_f_x(y_array, x_array, bandwidth)
    # m x n x d
    tiled_grad_log_x = jnp.tile(grad_log_p_x, (y_size, 1, 1))
    # n x m x d
    tiled_grad_log_y = jnp.tile(grad_log_p_y, (x_size, 1, 1))
    # m x n
    x = jnp.einsum("ijk,ijk -> ij", tiled_grad_log_x, grad_k_y)
    # n x m
    y = jnp.einsum("ijk,ijk -> ij", tiled_grad_log_y, grad_k_x)
    # n x m
    z = jnp.dot(grad_log_p_x, grad_log_p_y.T) * pc_imq_kernel
    return divergence + x.T + y + z


@jit
def stein_kernel_pc_imq_element(
        x: ArrayLike,
        y: ArrayLike,
        grad_log_p_x: ArrayLike,
        grad_log_p_y: ArrayLike,
        num_data_points: int,
        bandwidth: float = 1.,
) -> Array:
    """Compute the kernel element at x, y induced by the canonical Stein operator on a pre-conditioned inverse multi-quadric base kernel.

    The log PDF can be arbitrary, as only the gradients are supplied.

    Args:
        x: First vector, 1 x d.
        y: Second vector, 1 x d.
        grad_log_p_x: Gradient of log PDF evaluated at x, 1 x d.
        grad_log_p_y: Gradient of log PDF evaluated at y, 1 x d.
        num_data_points: Number of data points in the _generating_ set (not necessarily the same as X).
        bandwidth: Base kernel bandwidth (std dev). Defaults to 1

    Returns:
        Kernel evaluation at x, y as zero-dimensional array
    """
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    grad_log_p_x = jnp.atleast_2d(grad_log_p_x)
    grad_log_p_y = jnp.atleast_2d(grad_log_p_y)
    # n x m
    pc_imq_kernel = pc_imq(x, y, bandwidth)
    # n x m
    divergence: Array = pc_imq_div_x_grad_y(x, y, bandwidth, num_data_points, pc_imq_kernel)
    # n x m x d
    grad_p_x = jnp.squeeze(grad_pc_imq_x(x, y, bandwidth, pc_imq_kernel))
    # m x n x d
    grad_p_y = jnp.squeeze(grad_pc_imq_y(x, y, bandwidth, pc_imq_kernel))
    x_ = jnp.dot(grad_log_p_x, grad_p_y)
    # n x m
    y_ = jnp.dot(grad_log_p_y, grad_p_x)
    # n x m
    z = jnp.dot(grad_log_p_x, grad_log_p_y.T) * pc_imq_kernel
    kernel = divergence + x_.T + y_ + z
    return kernel[0, 0]
