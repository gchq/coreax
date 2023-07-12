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
import jax.numpy as jnp
from jax import jit, vmap


@jit
def sq_dist(x, y):
    """ Squared distance between two vectors

    Args:
        x (array_like): First argument
        y (array_like): Second argument

    Returns:
        ndarray: same format as numpy.dot
    """
    return jnp.dot(x - y, x - y)


@jit
def sq_dist_pairwise(X, Y):
    """ Efficient pairwise square distance

    Args:
        X (array_like): First argument, n x d
        Y (array_like): Second argument, m x d

    Returns:
        ndarray: Pairwise squared distances, n x m
    """
    # Use vmap to turn distance between individual vectors into a pairwise distance.
    d1 = vmap(sq_dist, in_axes=(None, 0), out_axes=0)
    d2 = vmap(d1, in_axes=(0, None), out_axes=0)

    return d2(X, Y)


@jit
def rbf_kernel(x, y, var=1.):
    """Squared exponential kernel for a pair of individual vectors

    Args:
        x (array_like): First argument.
        y (array_like): Second argument.
        var (float, optional): Variance parameter. Defaults to 1.

    Returns:
        ndarray: RBF kernel evaluated at x, y
    """
    return jnp.exp(-sq_dist(x, y)/(2*var))


@jit
def laplace_kernel(x, y, var=1.):
    """Laplace kernel for a pair of individual vectors

    Args:
        x (array_like): First argument.
        y (array_like): Second argument.
        var (float, optional): Variance parameter. Defaults to 1.

    Returns:
        ndarray: Laplace kernel evaluated at x, y
    """
    return jnp.exp(-jnp.linalg.norm(x - y)/(2*var))


@jit
def diff(x, y):
    """Vector difference for a pair of individual vectors

    Args:
        x (array_like): First argument.
        y (array_like): Second argument.

    Returns:
        ndarray: Vector difference
    """
    return x - y


@jit
def pdiff(X, Y):
    """Efficient pairwise difference for two arrays of vectors

    Args:
        X (array_like): First argument, n x d
        Y (array_like): Second argument, m x d

    Returns:
        ndarray: Pairwise differences, n x m
    """
    d1 = vmap(diff, in_axes=(0, None), out_axes=0)
    d2 = vmap(d1, in_axes=(None, 0), out_axes=1)
    return d2(X, Y)


@jit
def pdot(X, Y):
    """Efficient pairwise dot product for two arrays of vectors

    Args:
        X (array_like): First argument, n x d
        Y (array_like): Second argument, m x d

    Returns:
        ndarray: Pairwise dot products, n x m
    """
    d1 = vmap(jnp.dot, in_axes=(0, None), out_axes=0)
    d2 = vmap(d1, in_axes=(None, 0), out_axes=0)
    return d2(X, Y)


@jit
def idot(X, Y):
    """Dot product function for two tensors, over inner axes (preserving first axis)

    Args:
        X (array_like): First argument, d x n x m
        Y (array_like): Second argument, d x m x p

    Returns:
        TypeVar(F, bound=Callable): Function for inner axes dot products. f(X, Y) -> d x n x p
    """
    return vmap(jnp.dot, in_axes=(0, 0), out_axes=0)(X, Y)


@jit
def normalised_rbf(X, Y, nu=1.):
    """Normalised Gaussian kernel, pairwise.

    Args:
        X (array_like): First argument, n x d.
        Y (array_like): Second argument, m x d.
        nu (float, optional): Kernel bandwidth (std dev). Defaults to 1.

    Returns:
        ndarray: Pairwise kernel evaluations.
    """
    Z = sq_dist_pairwise(X, Y)
    k = jnp.exp(-.5*Z / nu**2) / jnp.sqrt(2 * jnp.pi)
    return k / nu


@jit
def pc_imq(X, Y, nu=1.):
    """Preconditioned inverse multi-quadric kernel

    Args:
        X (array_like): First argument, n x d.
        Y (array_like): Second argument, m x d.
        nu (float, optional): Kernel bandwidth (std dev). Defaults to 1.

    Returns:
        ndarray: Pairwise kernel evaluations.
    """
    l = 2 * nu**2
    Z = sq_dist_pairwise(X, Y) / l
    k = 1 / jnp.sqrt(1 + Z)
    return k


@jit
def grad_rbf_y(X, Y, nu=1., K=None):
    """Gradient of normalised RBF wrt Y

    Args:
        X (array_like): First argument, n x d.
        Y (array_like): Second argument, m x d.
        nu (float, optional): Kernel bandwidth (std dev). Defaults to 1.
        K (ndarray, optional): Gram matrix, if available. Defaults to None.

    Returns:
        ndarray: Gradients at each X x Y point, m x n x d
    """
    if K is None:
        K = normalised_rbf(Y, X, nu=nu)
    D = pdiff(Y, X)
    return D * K[:, :, None] / nu


@jit
def grad_rbf_x(X, Y, nu=1., K=None):
    """Gradient of normalised RBF wrt X

    Args:
        X (array_like): First argument, n x d.
        Y (array_like): Second argument, m x d.
        nu (float, optional): Kernel bandwidth (std dev). Defaults to 1.
        K (ndarray, optional): Gram matrix, if available. Defaults to None.

    Returns:
        ndarray: Gradients at each X x Y point, n x m x d
    """
    return -jnp.transpose(grad_rbf_y(X, Y, nu, K), (1, 0, 2))


@jit
def grad_pc_imq_y(X, Y, nu=1., K=None):
    """Gradient of pre-conditioned inverse multi-quadric wrt Y

    Args:
        X (array_like): First argument, n x d.
        Y (array_like): Second argument, m x d.
        nu (float, optional): Kernel bandwidth (std dev). Defaults to 1.
        K (ndarray, optional): Gram matrix, if available. Defaults to None.

    Returns:
        ndarray: Gradients at each X x Y point, m x n x d
    """
    l = 2 * nu**2
    if K is None:
        K = pc_imq(Y, X, nu)
    D = pdiff(Y, X)
    return K[:, :, None]**3 * D / l


@jit
def grad_pc_imq_x(X, Y, nu=1., K=None):
    """Gradient of pre-conditioned inverse multi-quadric wrt X

    Args:
        X (array_like): First argument, n x d.
        Y (array_like): Second argument, m x d.
        nu (float, optional): Kernel bandwidth (std dev). Defaults to 1.
        K (ndarray, optional): Gram matrix, if available. Defaults to None.

    Returns:
        ndarray: Gradients at each X x Y point, n x m x d
    """
    return -jnp.transpose(grad_pc_imq_y(X, Y, nu, K), (1, 0, 2))


@jit
def rbf_div_x_grad_y(X, Y, nu=1., n=None, K=None):
    """Divergence operator acting on gradient of RBF kernel wrt Y. Avoids explicit computation of the Hessian.

    Args:
        X (array_like): First argument, n x d.
        Y (array_like): Second argument, m x d.
        nu (float, optional): Kernel bandwidth (std dev). Defaults to 1.
        n (int, optional): The number of data points in the _generating_ set (not necessarily the same as X). Defaults to None.
        K (ndarray, optional): Gram matrix, if available. Defaults to None.

    Returns:
        ndarray: n x m matrix
    """
    if K is None:
        K = normalised_rbf(X, Y, nu=nu)
    if n is None:
        n = X.shape[0]
    return K / nu * (n - sq_dist_pairwise(X, Y) / nu)


@jit
def pc_imq_div_x_grad_y(X, Y, nu=1., n=None, K=None):
    """Divergence operator acting on gradient of PC-IMQ kernel wrt Y. Avoids explicit computation of the Hessian.

    Args:
        X (array_like): First argument, n x d.
        Y (array_like): Second argument, m x d.
        nu (float, optional): Kernel bandwidth (std dev). Defaults to 1.
        n (int, optional): The number of data points in the _generating_ set (not necessarily the same as X). Defaults to None.
        K (ndarray, optional): Gram matrix, if available. Defaults to None.

    Returns:
        ndarray: n x m matrix
    """
    l = 2 * nu**2
    if K is None:
        K = pc_imq(X, Y, nu=nu)
    if n is None:
        n = X.shape[0]
    return n / l * K**3 - 3*sq_dist_pairwise(X, Y)/l**2 * K**5


@jit
def median_heuristic(X):
    """Compute the median heuristic for setting kernel bandwidth

    Args:
        X (array_like): Input array of vectors.

    Returns:
        float: Bandwidth parameter, computed from the median heuristic.
    """
    D = jnp.triu(sq_dist_pairwise(X, X), k=1)
    h = jnp.median(D[jnp.triu_indices_from(D, k=1)])
    return jnp.sqrt(h / 2.)


# @jit
# def rbf_kde(X, Y, nu):
#     """Takes in data set and computes the RBF KDE value at each of the n points

#     Args:
#         X (array_like): First argument, n x d.
#         Y (array_like): Second argument, m x d.
#         nu (float, optional): Kernel bandwidth (std dev).

#     Returns:
#         ndarray: Gram matrix mean over Y, n x 1.
#         ndarray: Gram matrix, n x m
#     """
#     K = normalised_rbf(Y, X, nu=nu)
#     k = K.mean(axis=1)
#     return k, K


@jit
def rbf_f_X(X, D, nu):
    """PDF of X, as constructed by an RBF KDE using data set D

    Args:
        X (array_like): Random variable values, n x d
        D (array_like): KDE data set, m x d
        nu (float, optional): Kernel bandwidth (std dev).

    Returns:
        ndarray: Gram matrix mean over Y, n x 1.
        ndarray: Gram matrix, n x m
    """
    K = normalised_rbf(X, D, nu)
    k = K.mean(axis=1)
    return k, K


@jit
def rbf_grad_log_f_X(X, D, nu, K=None, Kbar=None):
    """Gradient of log PDF of X, where the PDF is a KDE induced by data set D.

    Args:
        X (array_like): Random variable values, n x d
        D (array_like): KDE data set, m x d
        nu (float, optional): Kernel bandwidth (std dev).
        K (ndarray, optional): Gram matrix, if available, n x m. Defaults to None.
        Kbar (ndarray, optional): Kernel mean, if available, n x 1. Defaults to None.

    Returns:
        ndarray: Array of gradients evaluated at values of X, n x d.
    """
    X = jnp.atleast_2d(X)
    D = jnp.atleast_2d(D)
    if K is None:
        Kbar, K = rbf_f_X(X, D, nu)
    n = D.shape[0]
    J = grad_rbf_x(X, D, nu, K).mean(axis=1)
    return J / (n*Kbar[:, None])


@jit
def grad_rbf_x(X, Y, nu, K=None):
    """Gradient of the RBF kernel, wrt X

    Args:
        X (array_like): First argument, n x d.
        Y (array_like): Second argument, m x d.
        nu (float, optional): Kernel bandwidth (std dev).
        K (ndarray, optional): Gram matrix, if available, n x m. Defaults to None.

    Returns:
        ndarray: Array of gradients evaluated at values of X, n x d.
    """
    if K is None:
        K = normalised_rbf(X, Y, nu)
    Z = -pdiff(X, Y) / nu
    return Z * K[:, :, None]


@jit
def grad_rbf_y(X, Y, nu, K=None):
    """Gradient of the RBF kernel, wrt Y

    Args:
        X (array_like): First argument, n x d.
        Y (array_like): Second argument, m x d.
        nu (float, optional): Kernel bandwidth (std dev).
        K (ndarray, optional): Gram matrix, if available, n x m. Defaults to None.

    Returns:
        ndarray: Array of gradients evaluated at values of Y, m x d.
    """
    return -jnp.transpose(grad_rbf_x(X, Y, nu, K), (1, 0, 2))


# @jit
# def grad_kde(X, Y, nu, K=None, Km=None):
#     """Gradient of an RBF KDE wrt X

#     Args:
#         X (_type_): _description_
#         Y (_type_): _description_
#         kernel (_type_): _description_
#         nu (_type_): _description_
#         K (_type_, optional): _description_. Defaults to None.
#         Km (_type_, optional): _description_. Defaults to None.

#     Returns:
#         _type_: _description_
#     """
#     if K is None:
#         Km, K = rbf_kde(X, Y, nu)
#     Km = Km.reshape(-1, 1)
#     n = X.shape[0]
#     Z = pdiff(Y, X)
#     J = idot(jnp.transpose(Z, (0, 2, 1)), K)
#     J /= -jnp.sqrt(2 * jnp.pi) * nu
#     return J / (n*Km)


@jit
def stein_kernel_rbf(X, Y, nu=1.):
    """Compute the kernel induced by the canonical Stein operator on an RBF base kernel.

    Args:
        X (array_like): First argument, n x d.
        Y (array_like): Second argument, m x d.
        nu (float, optional): Base kernel bandwidth (std dev).

    Returns:
        ndarray: Gram matrix, n x m
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
def stein_kernel_pc_imq(X, Y, nu=1.):
    """Compute the kernel Gram matrix induced by the canonical Stein operator on a pre-conditioned inverse multi-quadric base kernel.

    The log PDF is assumed to be a KDE induced by the data in Y.

    Args:
        X (array_like): First argument, n x d.
        Y (array_like): Second argument, m x d.
        nu (float, optional): Base kernel bandwidth (std dev).

    Returns:
        ndarray: Gram matrix, n x m
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
def stein_kernel_pc_imq_element(x, y, g_log_p_x, g_log_p_y, n, nu=1.):
    """Compute the kernel element at x, y induced by the canonical Stein operator on a pre-conditioned inverse multi-quadric base kernel.

    The log PDF can be arbitrary, as only the gradients are supplied.

    Args:
        x (array_like): First argument, 1 x d.
        y (array_like): Second argument, 1 x d.
        g_log_p_x (array_like): Gradient of log PDF evaluated at x, 1 x d.
        g_log_p_y (array_like): Gradient of log PDF evaluated at y, 1 x d.
        n (int): Number of data points in the 
        nu (float, optional): Base kernel bandwidth (std dev).

    Returns:
        float: kernel evaluation at x, y.
    """
    x = jnp.atleast_2d(x)
    y = jnp.atleast_2d(y)
    g_log_p_x = jnp.atleast_2d(g_log_p_x)
    g_log_p_y = jnp.atleast_2d(g_log_p_y)
    # n x m
    K = pc_imq(x, y, nu)
    # n x m
    div = pc_imq_div_x_grad_y(x, y, nu, n, K)
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
