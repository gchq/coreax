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

"""TODO: Create top-level docstring."""

# Support annotations with | in Python < 3.10
# TODO: Remove once no longer supporting old code
from __future__ import annotations

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

from coreax.util import (
    KernelFunction,
    KernelFunctionWithGrads,
    pdiff,
    sq_dist,
    sq_dist_pairwise,
)


@jit
def rbf_kernel(x: ArrayLike, y: ArrayLike, variance: float = 1.0) -> Array:
    r"""
    Calculate the radial basis function (RBF) kernel for a pair of vectors.

    :param x: First vector
    :param y: Second vector
    :param variance: Variance; optional, defaults to 1
    :return: RBF kernel evaluated at ``(x,y)``
    """
    return jnp.exp(-sq_dist(x, y) / (2 * variance))


@jit
def laplace_kernel(x: ArrayLike, y: ArrayLike, variance: float = 1.0) -> Array:
    r"""
    Calculate the Laplace kernel for a pair of vectors.

    :param x: First vector
    :param y: Second vector
    :param variance: Variance; optional, defaults to 1
    :return: Laplace kernel evaluated at ``(x,y)``
    """
    return jnp.exp(-jnp.linalg.norm(x - y) / (2 * variance))


@jit
def normalised_rbf(
    x_array: ArrayLike, y_array: ArrayLike, bandwidth: float = 1.0
) -> Array:
    r"""
    Evaluate the normalised Gaussian kernel pairwise.

    :param x_array: First set of vectors as a :math:`n \times d` array
    :param y_array: Second set of vectors as a :math:`m \times d` array
    :param bandwidth: Kernel bandwidth (standard deviation); optional, defaults to 1
    :return: Pairwise kernel evaluations
    """
    square_distances = sq_dist_pairwise(x_array, y_array)
    kernel = jnp.exp(-0.5 * square_distances / bandwidth**2) / jnp.sqrt(2 * jnp.pi)

    return kernel / bandwidth


@jit
def pc_imq(x_array: ArrayLike, y_array: ArrayLike, bandwidth: float = 1.0) -> Array:
    r"""
    Evaluate the pre-conditioned inverse multi-quadric kernel pairwise.

    :param x_array: First set of vectors as a :math:`n \times d` array
    :param y_array: Second set of vectors as a :math:`m \times d` array
    :param bandwidth: Kernel bandwidth (standard deviation); optional, defaults to 1
    :return: Pairwise kernel evaluations
    """
    scaling = 2 * bandwidth**2
    mq_array = sq_dist_pairwise(x_array, y_array) / scaling
    kernel = 1 / jnp.sqrt(1 + mq_array)

    return kernel


@jit
def grad_rbf_y(
    x_array: ArrayLike,
    y_array: ArrayLike,
    bandwidth: float = 1.0,
    gram_matrix: ArrayLike | None = None,
) -> Array:
    r"""
    Calculate *element-wise* grad of the normalised RBF w.r.t. ``y_array``.

    :param x_array: First set of vectors as a :math:`n \times d` array
    :param y_array: Second set of vectors as a :math:`m \times d` array
    :param bandwidth: Kernel bandwidth (standard deviation); optional, defaults to 1
    :param gram_matrix: Gram matrix; optional, if :data:`None` or omitted, defaults to a
        normalised Gaussian kernel
    :return: Gradients at each ``x_array X y_array`` point, an
             :math:`m \times n \times d` array
    """
    if gram_matrix is None:
        gram_matrix = normalised_rbf(x_array, y_array, bandwidth=bandwidth)
    else:
        gram_matrix = jnp.asarray(gram_matrix)

    distances = pdiff(x_array, y_array)

    return distances * gram_matrix[:, :, None] / bandwidth**2


@jit
def grad_rbf_x(
    x_array: ArrayLike,
    y_array: ArrayLike,
    bandwidth: float = 1.0,
    gram_matrix: ArrayLike | None = None,
) -> Array:
    r"""
    Calculate *element-wise* grad of the normalised RBF w.r.t. ``x_array``.

    :param x_array: First set of vectors as a :math:`n \times d` array
    :param y_array: Second set of vectors as a :math:`m \times d` array
    :param bandwidth: Kernel bandwidth (standard deviation); optional, defaults to 1
    :param gram_matrix: Gram matrix; optional, if :data:`None` or omitted, defaults to a
        normalised Gaussian kernel
    :return: Gradients at each ``x_array X y_array`` point, an
             :math:`m \times n \times d` array
    """
    return -grad_rbf_y(x_array, y_array, bandwidth, gram_matrix)


@jit
def grad_pc_imq_y(
    x_array: ArrayLike,
    y_array: ArrayLike,
    bandwidth: float = 1.0,
    gram_matrix: ArrayLike | None = None,
) -> Array:
    r"""
    Calculate *element-wise* grad of the pcimq w.r.t. ``y_array``.

    :param x_array: First set of vectors as a :math:`n \times d` array
    :param y_array: Second set of vectors as a :math:`m \times d` array
    :param bandwidth: Kernel bandwidth (standard deviation); optional, defaults to 1
    :param gram_matrix: Gram matrix; optional, if :data:`None` or omitted, defaults to a
        preconditioned inverse multi-quadric kernel
    :return: Gradients at each ``x_array X y_array`` point, an
             :math:`m \times n \times d` array
    """
    scaling = 2 * bandwidth**2
    if gram_matrix is None:
        gram_matrix = pc_imq(x_array, y_array, bandwidth)
    else:
        gram_matrix = jnp.asarray(gram_matrix)
    mq_array = pdiff(x_array, y_array)

    return gram_matrix[:, :, None] ** 3 * mq_array / scaling


@jit
def grad_pc_imq_x(
    x_array: ArrayLike,
    y_array: ArrayLike,
    bandwidth: float = 1.0,
    gram_matrix: ArrayLike | None = None,
) -> Array:
    r"""
    Calculate *element-wise* grad of the pcimq w.r.t. ``x_array``.

    :param x_array: First set of vectors as a :math:`n \times d` array
    :param y_array: Second set of vectors as a :math:`m \times d` array
    :param bandwidth: Kernel bandwidth (standard deviation); optional, defaults to 1
    :param gram_matrix: Gram matrix; optional, if :data:`None` or omitted, defaults to a
        preconditioned inverse multi-quadric kernel
    :return: Gradients at each ``x_array X y_array`` point, an
             :math:`m \times n \times d` array
    """
    return -grad_pc_imq_y(x_array, y_array, bandwidth, gram_matrix)


@jit
def rbf_div_x_grad_y(
    x_array: ArrayLike,
    y_array: ArrayLike,
    bandwidth: float = 1.0,
    num_data_points: int | None = None,
    gram_matrix: ArrayLike | None = None,
) -> Array:
    r"""
    Apply divergence operator on gradient of RBF kernel with respect to ``y_array``.

    This avoids explicit computation of the Hessian. Note that the generating set is
    not necessarily the same as ``x_array``.

    :param x_array: First set of vectors as a :math:`n \times d` array
    :param y_array: Second set of vectors as a :math:`m \times d` array
    :param bandwidth: Kernel bandwidth (standard deviation); optional, defaults to 1
    :param num_data_points: Number of data points in the generating set; optional, if
        :data:`None` or omitted, defaults to number of vectors in ``x_array``
    :param gram_matrix: Gram matrix; optional, if :data:`None` or omitted, defaults to
        a normalised Gaussian kernel
    :return: Divergence operator, an :math:`n \times m` matrix
    """
    x_array = jnp.asarray(x_array)
    if gram_matrix is None:
        gram_matrix = normalised_rbf(x_array, y_array, bandwidth=bandwidth)
    if num_data_points is None:
        num_data_points = x_array.shape[0]

    return (
        gram_matrix
        / bandwidth
        * (num_data_points - sq_dist_pairwise(x_array, y_array) / bandwidth)
    )


@jit
def pc_imq_div_x_grad_y(
    x_array: ArrayLike,
    y_array: ArrayLike,
    bandwidth: float = 1.0,
    num_data_points: int | None = None,
    gram_matrix: ArrayLike | None = None,
) -> Array:
    r"""
    Apply divergence operator on gradient of PC-IMQ kernel with respect to ``y_array``.

    This avoids explicit computation of the Hessian. Note that the generating set is
    not necessarily the same as ``x_array``.

    :param x_array: First set of vectors as a :math:`n \times d` array
    :param y_array: Second set of vectors as a :math:`m \times d` array
    :param bandwidth: Kernel bandwidth (standard deviation); optional, defaults to 1
    :param num_data_points: Number of data points in the generating set; optional, if
        :data:`None` or omitted, defaults to number of vectors in ``x_array``
    :param gram_matrix: Gram matrix; optional, if :data:`None` or omitted, defaults to
        a normalised Gaussian kernel
    :return: Divergence operator, an :math:`n \times m` matrix
    """
    scaling = 2 * bandwidth**2
    x_array = jnp.asarray(x_array)
    if gram_matrix is None:
        gram_matrix = pc_imq(x_array, y_array, bandwidth=bandwidth)
    if num_data_points is None:
        num_data_points = x_array.shape[0]
    return (
        num_data_points / scaling * gram_matrix**3
        - 3 * sq_dist_pairwise(x_array, y_array) / scaling**2 * gram_matrix**5
    )


@jit
def median_heuristic(x_array: ArrayLike) -> Array:
    r"""
    Compute the median heuristic for setting kernel bandwidth.

    :param x_array: Input array of vectors
    :return: Bandwidth parameter, computed from the median heuristic, as a
        0-dimensional array
    """
    # calculate square distances as an upper triangular matrix
    square_distances = jnp.triu(sq_dist_pairwise(x_array, x_array), k=1)
    # calculate the median
    median_square_distance = jnp.median(
        square_distances[jnp.triu_indices_from(square_distances, k=1)]
    )

    return jnp.sqrt(median_square_distance / 2.0)


@jit
def rbf_f_x(
    random_var_values: ArrayLike, kde_data: ArrayLike, bandwidth: float
) -> tuple[Array, Array]:
    r"""
    Construct PDF of ``random_var_values`` by kernel density estimation.

    This is done for a radial basis function.

    :param random_var_values: An :math:`n \times d` array of random variable values
    :param kde_data: The :math:`m \times d` kernel density estimation set
    :param bandwidth: Kernel bandwidth (standard deviation)
    :return: Gram matrix mean over `random_var_values` as a :math:`n \times 1` array;
        Gram matrix as an :math:`n \times m` array
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
    r"""
    Compute gradient of log-PDF of ``random_var_values``.

    The PDF is constructed from kernel density estimation.

    :param random_var_values: An :math:`n \times d` array of random variable values
    :param kde_data: The :math:`m \times d` kernel density estimation set
    :param bandwidth: Kernel bandwidth (standard deviation)
    :param gram_matrix: Gram matrix, an :math:`n \times m` array; optional, if
        ``gram_matrix`` or ``kernel_mean`` are :data:`None` or omitted, defaults to a
        normalised Gaussian kernel
    :param kernel_mean: Kernel mean, an :math:`n \times 1` array; optional, if
        ``gram_matrix`` or ``kernel_mean`` are :data:`None` or omitted, defaults to the
        mean of a Normalised Gaussian kernel
    :return: An :math:`n \times d` array of gradients evaluated at values of
        ``random_var_values``
    """
    random_var_values = jnp.atleast_2d(random_var_values)
    kde_data = jnp.atleast_2d(kde_data)
    if gram_matrix is None or kernel_mean is None:
        kernel_mean, gram_matrix = rbf_f_x(random_var_values, kde_data, bandwidth)
    else:
        kernel_mean = jnp.asarray(kernel_mean)
    gradients = grad_rbf_x(random_var_values, kde_data, bandwidth, gram_matrix).mean(
        axis=1
    )

    return gradients / kernel_mean[:, None]


@jit
def stein_kernel_rbf(
    x_array: ArrayLike, y_array: ArrayLike, bandwidth: float = 1.0
) -> Array:
    r"""
    Compute a kernel from an RBF kernel with the canonical Stein operator.

    :param x_array: First set of vectors as a :math:`n \times d` array
    :param y_array: Second set of vectors as a :math:`n \times d` array
    :param bandwidth: Kernel bandwidth (standard deviation)
    :return: Gram matrix, an :math:`n \times m` array
    """
    x_array = jnp.atleast_2d(x_array)
    y_array = jnp.atleast_2d(y_array)
    x_size = x_array.shape[0]
    y_size = y_array.shape[0]
    # n x m
    rbf_kernel_ = normalised_rbf(x_array, y_array, bandwidth)
    # n x m
    divergence = rbf_div_x_grad_y(x_array, y_array, bandwidth, x_size, rbf_kernel_)
    # n x m x d
    grad_k_x = grad_rbf_x(x_array, y_array, bandwidth, rbf_kernel_)
    # m x n x d
    grad_k_y = jnp.transpose(
        grad_rbf_y(x_array, y_array, bandwidth, rbf_kernel_), (1, 0, 2)
    )
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
    z = jnp.dot(grad_log_p_x, grad_log_p_y.T) * rbf_kernel_
    return divergence + x.T + y + z


@jit
def stein_kernel_pc_imq(
    x_array: ArrayLike, y_array: ArrayLike, bandwidth: float = 1.0
) -> Array:
    r"""
    Compute a kernel from a PC-IMQ kernel with the canonical Stein operator.

    The log-PDF is assumed to be induced by kernel density estimation with the
    data in ``y_array``.

    :param x_array: First set of vectors as a :math:`n \times d` array
    :param y_array: Second set of vectors as a :math:`n \times d` array
    :param bandwidth: Kernel bandwidth (standard deviation)
    :return: Gram matrix, an :math:`n \times m` array
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
    grad_k_y = jnp.transpose(
        grad_pc_imq_y(x_array, y_array, bandwidth, pc_imq_kernel), (1, 0, 2)
    )
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
    dimension: int,
    bandwidth: float = 1.0,
) -> Array:
    r"""
    Evaluate the kernel element at ``(x,y)``.

    This element is induced by the canonical Stein operator on a PC-IMQ kernel. The
    log-PDF can be arbitrary as only gradients are supplied.

    :param x: First vector as a :math:`1 \times d` array
    :param y: Second vector as a :math:`1 \times d` array
    :param grad_log_p_x: Gradient of log-PDF evaluated at ``x``, a :math:`1 \times d`
        array
    :param grad_log_p_y: Gradient of log-PDF evaluated at ``y``, a :math:`1 \times d`
        array
    :param dimension: Dimension of the input data.
    :param bandwidth: Kernel bandwidth (standard deviation); optional, defaults to 1
    :return: Kernel evaluation at ``(x,y)``, 0-dimensional array
    """
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    grad_log_p_x = jnp.atleast_2d(grad_log_p_x)
    grad_log_p_y = jnp.atleast_2d(grad_log_p_y)
    # n x m
    pc_imq_kernel = pc_imq(x, y, bandwidth)
    # n x m
    divergence: Array = pc_imq_div_x_grad_y(x, y, bandwidth, dimension, pc_imq_kernel)
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


def update_K_sum(
    X: ArrayLike,
    K_sum: ArrayLike,
    i: int,
    j: int,
    max_size: int,
    k_pairwise: KernelFunction | KernelFunctionWithGrads,
    grads: ArrayLike | None = None,
    nu: float | None = None,
) -> Array:
    r"""
    Update row sum with a kernel matrix block.

    The kernel matrix block ``i:i+max_{size}`` :math:`\times` ``j:j+max_{size}`` is used
    to update the row sum. Symmetry of the kernel matrix is exploited to reduced
    repeated calculation.

    Note that `k_pairwise` should be of the form ``k(x, y)`` if ``grads`` and ``nu``
    are :data:`None`. Otherwise, ``k_pairwise`` should be of the form
    ``k(x, y, grads, grads, n, nu)``.

    :param X: Data matrix, :math:`n \times d`
    :param K_sum: Full data structure for Gram matrix row sum, :math:`1 \times n`
    :param i: Kernel matrix block start
    :param j: Kernel matrix block end
    :param max_size: Size of matrix block to process
    :param k_pairwise: Pairwise kernel evaluation function
    :param grads: Array of gradients, if applicable, :math:`n \times d`;
        optional, defaults to :data:`None`
    :param nu: Base kernel bandwidth; optional, defaults to :data:`None`
    :return: Gram matrix row sum, with elements ``i:i+max_{size}`` and
        ``j:j+max_{size}`` populated
    """
    X = jnp.asarray(X)
    K_sum = jnp.asarray(K_sum)
    n = X.shape[0]
    if grads is None:
        K_part = k_pairwise(X[i : i + max_size], X[j : j + max_size])
    else:
        grads = jnp.asarray(grads)
        K_part = k_pairwise(
            X[i : i + max_size],
            X[j : j + max_size],
            grads[i : i + max_size],
            grads[j : j + max_size],
            n,
            nu,
        )
    K_sum = K_sum.at[i : i + max_size].set(K_sum[i : i + max_size] + K_part.sum(axis=1))

    if i != j:
        K_sum = K_sum.at[j : j + max_size].set(
            K_sum[j : j + max_size] + K_part.sum(axis=0)
        )

    return K_sum


def calculate_K_sum(
    X: ArrayLike,
    k_pairwise: KernelFunction | KernelFunctionWithGrads,
    max_size: int,
    grads: ArrayLike | None = None,
    nu: ArrayLike | None = None,
) -> Array:
    r"""
    Calculate row sum of the kernel matrix.

    The row sum is calculated block-wise to limit memory overhead.

    Note that ``k_pairwise`` should be of the form ``k(x, y)`` if ``grads`` and ``nu``
    are :data:`None`. Otherwise, ``k_pairwise`` should be of the form
    ``k(x, y, grads, grads, n, nu)``.

    :param X: Data matrix, :math:`n \times d`
    :param k_pairwise: Pairwise kernel evaluation function
    :param max_size: Size of matrix block to process
    :param grads: Array of gradients, if applicable, :math:`n \times d`;
        optional, defaults to :data:`None`.
    :param nu: Base kernel bandwidth, if applicable, :math:`n \times d`;
        optional, defaults to :data:`None`.
    :return: Kernel matrix row sum
    """
    X = jnp.asarray(X)
    n = len(X)
    K_sum = jnp.zeros(n)
    # Iterate over upper triangular blocks
    for i in range(0, n, max_size):
        for j in range(i, n, max_size):
            K_sum = update_K_sum(X, K_sum, i, j, max_size, k_pairwise, grads, nu)

    return K_sum
