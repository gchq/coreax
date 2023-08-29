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
    r"""
    Calculate the squared distance between two vectors.

    :param x: First vector argument
    :param y: Second vector argument
    :return: Dot product of `x-y` and `x-y`, the square distance between `x` and `y`
    """
    return jnp.dot(x - y, x - y)


@jit
def sq_dist_pairwise(X: ArrayLike, Y: ArrayLike) -> Array:
    r"""
    Calculate efficient pairwise square distance between two arrays.

    :param X: First $n \times d$ array argument
    :param Y: Second $m \times d$ array argument
    :return: Pairwise squared distances between `X` and `Y` as an $n \times m$ array
    """
    # Use vmap to turn distance between individual vectors into a pairwise distance.
    d1 = vmap(sq_dist, in_axes=(None, 0), out_axes=0)
    d2 = vmap(d1, in_axes=(0, None), out_axes=0)

    return d2(X, Y)


@jit
def rbf_kernel(x: ArrayLike, y: ArrayLike, var: float = 1.) -> Array:
    r"""
    Calculate the radial basis function (RBF) kernel for a pair of vectors.

    :param x: First vector
    :param y: Second vector
    :param var: Variance. Optional, defaults to 1
    :return: RBF kernel evaluated at `(x,y)`
    """
    return jnp.exp(-sq_dist(x, y)/(2*var))


@jit
def laplace_kernel(x: ArrayLike, y: ArrayLike, var: float = 1.) -> Array:
    r"""
    Calculate the Laplace kernel for a pair of vectors.

    :param x: First vector
    :param y: Second vector
    :param var: Variance. Optional, defaults to 1
    :return: Laplace kernel evaluated at `(x,y)`
    """
    return jnp.exp(-jnp.linalg.norm(x - y)/(2*var))


@jit
def diff(x: ArrayLike, y: ArrayLike) -> Array:
    r"""
    Calculate vector difference for a pair of vectors.

    :param x: First vector
    :param y: Second vector
    :return: Vector difference `x-y`
    """
    return x - y


@jit
def pdiff(X: ArrayLike, Y: ArrayLike) -> Array:
    r"""
    Calculate efficient pairwise difference between two arrays of vectors.

    :param X: First $n \times d$ array argument
    :param Y: Second $m \times d$ array argument
    :return: Pairwise differences between `X` and `Y` as an $n \times m \times d$ array
    """
    d1 = vmap(diff, in_axes=(0, None), out_axes=0)
    d2 = vmap(d1, in_axes=(None, 0), out_axes=1)
    return d2(X, Y)


@jit
def normalised_rbf(X: ArrayLike, Y: ArrayLike, nu: float = 1.) -> Array:
    r"""
    Evaluate the normalised Gaussian kernel pairwise.

    :param X: First $n \times d$ array argument
    :param Y: Second $m \times d$ array argument
    :param nu: Kernel bandwidth (standard deviation). Optional, defaults to 1
    :return: Pairwise kernel evaluations
    """
    Z = sq_dist_pairwise(X, Y)
    k = jnp.exp(-.5*Z / nu**2) / jnp.sqrt(2 * jnp.pi)
    return k / nu


@jit
def pc_imq(X: ArrayLike, Y: ArrayLike, nu: float = 1.) -> Array:
    r"""
    Evaluate the pre-conditioned inverse multi-quadric kernel pairwise.

    :param X: First $n \times d$ array argument
    :param Y: Second $m \times d$ array argument
    :param nu: Kernel bandwidth (standard deviation). Optional, defaults to 1
    :return: Pairwise kernel evaluations
    """
    l = 2 * nu**2
    Z = sq_dist_pairwise(X, Y) / l
    k = 1 / jnp.sqrt(1 + Z)
    return k


@jit
def grad_rbf_y(
        X: ArrayLike,
        Y: ArrayLike,
        nu: float = 1.,
        K: ArrayLike | None = None,
) -> Array:
    r"""
    Calculate the gradient of the normalised radial basis function with respect to Y.

    :param X: First $n \times d$ array argument
    :param Y: Second $m \times d$ array argument
    :param nu: Kernel bandwidth (standard deviation). Optional, defaults to 1
    :param K: Gram matrix. Optional, defaults to `None`
    :return: Gradients at each `X x Y` point, an $m \times n \times d$ array
    """
    if K is None:
        K = normalised_rbf(Y, X, nu=nu)
    else:
        K = jnp.asarray(K)

    D = pdiff(Y, X)
    return D * K[:, :, None] / nu


@jit
def grad_rbf_x(
        X: ArrayLike,
        Y: ArrayLike,
        nu: float = 1.,
        K: ArrayLike | None = None,
) -> Array:
    r"""
    Calculate the gradient of the normalised radial basis function with respect to X.

    :param X: First $n \times d$ array argument
    :param Y: Second $m \times d$ array argument
    :param nu: Kernel bandwidth (standard deviation). Optional, defaults to 1
    :param K: Gram matrix. Optional, defaults to `None`
    :return: Gradients at each `X x Y` point, an $n \times m \times d$ array
    """
    return -jnp.transpose(grad_rbf_y(X, Y, nu, K), (1, 0, 2))


@jit
def grad_pc_imq_y(
        X: ArrayLike,
        Y: ArrayLike,
        nu: float = 1.,
        K: ArrayLike | None = None,
) -> Array:
    r"""
    Calculate gradient of the pre-conditioned inverse multi-quadric with respect to Y.

    :param X: First $n \times d$ array argument
    :param Y: Second $m \times d$ array argument
    :param nu: Kernel bandwidth (standard deviation). Optional, defaults to 1
    :param K: Gram matrix. Optional, defaults to `None`
    :return: Gradients at each `X x Y` point, an $m \times n \times d$ array
    """
    l = 2 * nu**2
    if K is None:
        K = pc_imq(Y, X, nu)
    else:
        K = jnp.asarray(K)
    D = pdiff(Y, X)
    return K[:, :, None]**3 * D / l


@jit
def grad_pc_imq_x(
        X: ArrayLike,
        Y: ArrayLike,
        nu: float = 1.,
        K: ArrayLike | None = None,
) -> Array:
    r"""
    Calculate gradient of the pre-conditioned inverse multi-quadric with respect to X.

    :param X: First $n \times d$ array argument
    :param Y: Second $m \times d$ array argument
    :param nu: Kernel bandwidth (standard deviation). Optional, defaults to 1
    :param K: Gram matrix. Optional, defaults to `None`
    :return: Gradients at each `X x Y` point, an $n \times m \times d$ array
    """
    return -jnp.transpose(grad_pc_imq_y(X, Y, nu, K), (1, 0, 2))


@jit
def rbf_div_x_grad_y(
        X: ArrayLike,
        Y: ArrayLike,
        nu: float = 1.,
        n: int | None = None,
        K: ArrayLike | None = None,
) -> Array:
    r"""
    Apply divergence operator on gradient of RBF kernel with respect to Y.

    This avoids explicit computation of the Hessian. Note that the generating set is
    not necessarily the same as `X`.

    :param X: First $n \times d$ array argument
    :param Y: Second $m \times d$ array argument
    :param nu: Kernel bandwidth (standard deviation). Optional, defaults to 1
    :param n: Number of data points in the generating set. Optional, defaults to `None`
    :param K: Gram matrix. Optional, defaults to `None`
    :return: Divergence operator, an $n \times m$ matrix
    """
    X = jnp.asarray(X)
    if K is None:
        K = normalised_rbf(X, Y, nu=nu)
    if n is None:
        n = X.shape[0]
    return K / nu * (n - sq_dist_pairwise(X, Y) / nu)


@jit
def pc_imq_div_x_grad_y(
        X: ArrayLike,
        Y: ArrayLike,
        nu: float = 1.,
        n: int = None,
        K: ArrayLike | None = None,
) -> Array:
    r"""
    Apply divergence operator on gradient of PC-IMQ kernel with respect to Y.

    This avoids explicit computation of the Hessian. Note that the generating set is
    not necessarily the same as `X`.

    :param X: First $n \times d$ array argument
    :param Y: Second $m \times d$ array argument
    :param nu: Kernel bandwidth (standard deviation). Optional, defaults to 1
    :param n: Number of data points in the generating set. Optional, defaults to `None`
    :param K: Gram matrix. Optional, defaults to `None`
    :return: Divergence operator, an $n \times m$ matrix
    """
    l = 2 * nu**2
    X = jnp.asarray(X)
    if K is None:
        K = pc_imq(X, Y, nu=nu)
    if n is None:
        n = X.shape[0]
    return n / l * K**3 - 3*sq_dist_pairwise(X, Y)/l**2 * K**5


@jit
def median_heuristic(X: ArrayLike) -> Array:
    r"""
    Compute the median heuristic for setting kernel bandwidth.

    :param X: Input array of vectors
    :return: Bandwidth parameter, computed from the median heuristic, as a
             0-dimensional array
    """
    D = jnp.triu(sq_dist_pairwise(X, X), k=1)
    h = jnp.median(D[jnp.triu_indices_from(D, k=1)])
    return jnp.sqrt(h / 2.)


@jit
def rbf_f_X(X: ArrayLike, D: ArrayLike, nu: float) -> tuple[Array, Array]:
    r"""
    Construct PDF of `X` by kernel density estimation for a radial basis function.

    :param X: An $n \times d$ array of random variable values
    :param D: The $m \times d$ kernel density estimation set
    :param nu: Kernel bandwidth (standard deviation)
    :return: Gram matrix mean over X as an $n \times 1$ array; Gram matrix as an
             $n \times m$ array
    """
    K = normalised_rbf(X, D, nu)
    k = K.mean(axis=1)
    return k, K


@jit
def rbf_grad_log_f_X(
        X: ArrayLike,
        D: ArrayLike,
        nu: float,
        K: ArrayLike | None = None,
        Kbar: ArrayLike | None = None,
) -> Array:
    r"""
    Compute gradient of log-PDF of `X`.

    The PDF is constructed from kernel density estimation.

    :param X: An $n \times d$ array of random variable values
    :param D: The $m \times d$ kernel density estimation set
    :param nu: Kernel bandwidth (standard deviation)
    :param K: Gram matrix, an $n \times m$ array. Optional, defaults to `None`
    :param Kbar: Kernel mean, an $n \times 1$ array. Optional, defaults to `None`
    :return: An $n \times d$ array of gradients evaluated at values of `X`
    """
    X = jnp.atleast_2d(X)
    D = jnp.atleast_2d(D)
    if K is None or Kbar is None:
        Kbar, K = rbf_f_X(X, D, nu)
    else:
        Kbar = jnp.asarray(Kbar)
    n = D.shape[0]
    J = grad_rbf_x(X, D, nu, K).mean(axis=1)
    return J / (n*Kbar[:, None])


@jit
def grad_rbf_x(
        X: ArrayLike,
        Y: ArrayLike,
        nu: float,
        K: ArrayLike | None = None,
) -> Array:
    r"""
    Compute gradient of the radial basis function kernel with respect to `X`.

    :param X: First $n \times d$ array argument
    :param Y: Second $m \times d$ array argument
    :param nu: Kernel bandwidth (standard deviation)
    :param K: Gram matrix, an $n \times m$ array. Optional, defaults to `None`
    :return: Gradient evaluated at values of `X`, as an $n \times d$ array
    """
    if K is None:
        K = normalised_rbf(X, Y, nu)
    else:
        K = jnp.asarray(K)

    Z = -pdiff(X, Y) / nu
    return Z * K[:, :, None]


@jit
def grad_rbf_y(
        X: ArrayLike,
        Y: ArrayLike,
        nu: float,
        K: ArrayLike | None = None,
) -> Array:
    r"""
    Compute gradient of the radial basis function kernel with respect to `Y`.

    :param X: First $n \times d$ array argument
    :param Y: Second $m \times d$ array argument
    :param nu: Kernel bandwidth (standard deviation)
    :param K: Gram matrix, an $n \times m$ array. Optional, defaults to `None`
    :return: Gradient evaluated at values of `Y`, as an $m \times d$ array
    """
    return -jnp.transpose(grad_rbf_x(X, Y, nu, K), (1, 0, 2))


@jit
def stein_kernel_rbf(X: ArrayLike, Y: ArrayLike, nu: float = 1.) -> Array:
    r"""
    Compute a kernel from a RBF kernel with the canonical Stein operator.

    :param X: First $n \times d$ array argument
    :param Y: Second $m \times d$ array argument
    :param nu: Kernel bandwidth (standard deviation). Optional, defaults to 1
    :return: Gram matrix, an $n \times m$ array
    """
    X = jnp.atleast_2d(X)
    Y = jnp.atleast_2d(Y)
    n = X.shape[0]
    m = Y.shape[0]
    # n x m
    K = normalised_rbf(X, Y, nu)
    # n x m
    div = rbf_div_x_grad_y(X, Y, nu, n, K)
    # n x m x d
    g_k_x = grad_rbf_x(X, Y, nu, K)
    # m x n x d
    g_k_y = grad_rbf_y(X, Y, nu, K)
    # n x d
    g_log_p_x = rbf_grad_log_f_X(X, Y, nu)
    # m x d
    g_log_p_y = rbf_grad_log_f_X(Y, X, nu)
    # m x n x d
    gxt = jnp.tile(g_log_p_x, (m, 1, 1))
    # n x m x d
    gyt = jnp.tile(g_log_p_y, (n, 1, 1))
    # m x n
    x = jnp.einsum("ijk,ijk -> ij", gxt, g_k_y)
    # n x m
    y = jnp.einsum("ijk,ijk -> ij", gyt, g_k_x)
    # n x m
    z = jnp.dot(g_log_p_x, g_log_p_y.T) * K
    return div + x.T + y + z


@jit
def stein_kernel_pc_imq(X: ArrayLike, Y: ArrayLike, nu: float = 1.) -> Array:
    r"""
    Compute a kernel from a PC-IMQ kernel with the canonical Stein operator.

    The log-PDF is assumed to be induced by kernel density estimation with the
    data in `Y`.

    :param X: First $n \times d$ array argument
    :param Y: Second $m \times d$ array argument
    :param nu: Kernel bandwidth (standard deviation). Optional, defaults to 1
    :return: Gram matrix, an $n \times m$ array
    """
    X = jnp.atleast_2d(X)
    Y = jnp.atleast_2d(Y)
    n = X.shape[0]
    m = Y.shape[0]
    # n x m
    K = pc_imq(X, Y, nu)
    # n x m
    div = pc_imq_div_x_grad_y(X, Y, nu, n, K)
    # n x m x d
    g_k_x = grad_pc_imq_x(X, Y, nu, K)
    # m x n x d
    g_k_y = grad_pc_imq_y(X, Y, nu, K)
    # n x d
    g_log_p_x = rbf_grad_log_f_X(X, Y, nu)
    # m x d
    g_log_p_y = rbf_grad_log_f_X(Y, X, nu)
    # m x n x d
    gxt = jnp.tile(g_log_p_x, (m, 1, 1))
    # n x m x d
    gyt = jnp.tile(g_log_p_y, (n, 1, 1))
    # m x n
    x = jnp.einsum("ijk,ijk -> ij", gxt, g_k_y)
    # n x m
    y = jnp.einsum("ijk,ijk -> ij", gyt, g_k_x)
    # n x m
    z = jnp.dot(g_log_p_x, g_log_p_y.T) * K
    return div + x.T + y + z


@jit
def stein_kernel_pc_imq_element(
        x: ArrayLike,
        y: ArrayLike,
        g_log_p_x: ArrayLike,
        g_log_p_y: ArrayLike,
        n: int,
        nu: float = 1.,
) -> Array:
    r"""
    Evaluate the kernel element at `(x,y)`.

    This element is induced by the canonical Stein operator on a PC-IMQ kernel. The
    log-PDF can be arbitrary as only gradients are supplied.

    :param x: First $1 \times d$ array argument
    :param y: Second $1 \times d$ array argument
    :param g_log_p_x: Gradient of log-PDF evaluated at `x`, a $1 \times d$ array
    :param g_log_p_y: Gradient of log-PDF evaluated at `y`, a $1 \times d$ array
    :param nu: Kernel bandwidth (standard deviation). Optional, defaults to 1
    :param n: Number of data points in the generating set. Optional, defaults to
              `None`.
    :return: Kernel evaluation at `(x,y)`, 0-dimensional array
    """
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    g_log_p_x = jnp.atleast_2d(g_log_p_x)
    g_log_p_y = jnp.atleast_2d(g_log_p_y)
    # n x m
    K = pc_imq(x, y, nu)
    # n x m
    div: Array = pc_imq_div_x_grad_y(x, y, nu, n, K)
    # n x m x d
    g_k_x = jnp.squeeze(grad_pc_imq_x(x, y, nu, K))
    # m x n x d
    g_k_y = jnp.squeeze(grad_pc_imq_y(x, y, nu, K))
    x_ = jnp.dot(g_log_p_x, g_k_y)
    # n x m
    y_ = jnp.dot(g_log_p_y, g_k_x)
    # n x m
    z = jnp.dot(g_log_p_x, g_log_p_y.T) * K
    fin = div + x_.T + y_ + z
    return fin[0, 0]
