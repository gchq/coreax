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

from collections.abc import Callable

import jax.numpy as jnp
import jax.lax as lax
from jax.typing import ArrayLike

from jax import vmap, jit, Array

from coreax.kernel import rbf_grad_log_f_X, stein_kernel_pc_imq_element
from coreax.utils import calculate_K_sum, KernelFunction, KernelFunctionWithGrads
from functools import partial

from sklearn.neighbors import KDTree
from multiprocessing.pool import ThreadPool


@partial(jit, static_argnames=["k_vec", "unique"])
def greedy_body(
        i: int,
        val: tuple[ArrayLike, ArrayLike, ArrayLike],
        X: ArrayLike,
        k_vec: KernelFunction,
        K_mean: ArrayLike,
        unique: bool,
) -> tuple[Array, Array, Array]:
    """Greedy kernel herding main loop.

    Args:
        i: Loop counter
        val: Loop updateables
        X: Original data set, n x d
        k_vec: Vectorised kernel function k(X, x) -> R^n, where X \in R^{n x d} and x \in R^d
        K_mean: Mean vector for the Gram matrix, i.e. mean over rows, 1 x n.
        unique: insist on unique elements

    Returns:
        Updated loop variables, (coreset, K_t objective).
    """
    X = jnp.asarray(X)
    S, K, K_t = val
    S = jnp.asarray(S)
    K = jnp.asarray(K)
    j = (K_mean - K_t/(i+1)).argmax()
    kv = k_vec(X, X[j])
    K_t = K_t + kv
    S = S.at[i].set(j)
    K = K.at[i].set(kv)
    if unique:
        K_t = K_t.at[j].set(jnp.inf)

    return S, K, K_t


@partial(jit, static_argnames=["k_vec", "unique"])
def stein_greedy_body(
        i: int,
        val: tuple[ArrayLike, ArrayLike, ArrayLike],
        X: ArrayLike,
        k_vec: KernelFunctionWithGrads,
        K_mean: ArrayLike,
        grads: ArrayLike,
        n: int,
        nu: float,
        unique: bool,
) -> tuple[Array, Array, Array]:
    """Greedy Stein herding main loop

    Args:
        i: Loop counter
        val: Loop updateables
        X: Original data set, n x d
        k_vec: Vectorised kernel function k(X, x, Y, y) -> R^n, where X, Y \in R^{n x d} and x, y \in R^d
        K_mean: Mean vector for the Gram matrix, i.e. mean over rows, 1 x n.
        grads: Gradients of log PDF evaluated at X, n x d.
        n: Number of data points that induced the original PDF.
        nu: Base kernel for Stein kernel, bandwidth parameter.
        unique: insist on unique elements

    Returns:
        Updated loop variables, (coreset, coreset Gram matrix, objective)
    """
    S, K, objective = val
    S = jnp.asarray(S)
    K = jnp.asarray(K)
    objective = jnp.asarray(objective)
    X = jnp.asarray(X)
    grads = jnp.asarray(grads)
    j = objective.argmax()
    S = S.at[i].set(j)
    K = K.at[i].set(k_vec(X, X[j], grads, grads[j], n, nu))
    objective = objective * (i + 1) / (i + 2) + (K_mean - K[i]) / (i + 2)
    if unique:
        objective = objective.at[j].set(-jnp.inf)
    return S, K, objective


def kernel_herding_block(
        X: ArrayLike,
        n_core: int,
        kernel: KernelFunction,
        max_size: int = 10000,
        K_mean: ArrayLike | None = None,
        unique: bool = True,
) -> tuple[Array, Array, Array]:
    """Implementation of kernel herding algorithm using Jax. 

    Args:
        X: n x d original data
        n_core: Number of coreset points to calculate
        kernel: Kernel function k: R^d x R^d \to R
        max_size: Size of matrix block to process
        K_mean: Row sum of kernel matrix divided by n
        unique: insist on unique elements

    Returns:
        coreset point indices, Kc, Kbar
    """

    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None, 0),
                               out_axes=0), in_axes=(0, None), out_axes=0))
    k_vec = jit(vmap(kernel, in_axes=(0, None)))

    X = jnp.asarray(X)
    n = len(X)
    if K_mean is None:
        K_mean = calculate_K_sum(X, k_pairwise, max_size) / n

    K_t = jnp.zeros(n)
    S = jnp.zeros(n_core, dtype=jnp.int32)
    K = jnp.zeros((n_core, n))

    # Greedly select coreset points
    body = partial(greedy_body, X=X, k_vec=k_vec, K_mean=K_mean, unique=unique)
    S, K, _ = lax.fori_loop(0, n_core, body, (S, K, K_t))
    Kbar = K.mean(axis=1)
    Kc = K[:, S]

    return S, Kc, Kbar


def stein_kernel_herding_block(
        X: ArrayLike,
        n_core: int,
        kernel: KernelFunction,
        grad_log_f_X: Callable[[ArrayLike, ArrayLike, float], Array],
        K_mean: ArrayLike | None = None,
        max_size: int = 10000,
        nu: float = 1.,
        unique: bool = True,
) -> tuple[Array, Array, Array]:
    """Stein herding

    Args:
        X: n x d original data
        n_core: Number of coreset points to calculate
        kernel: Kernel function k: R^d x R^d \to R
        grad_log_f_X: function to compute gradient of log PDF, g: X -> Y
        K_mean: Row sum of kernel matrix divided by n
        max_size: Size of matrix block to process
        nu: Base kernel for Stein kernel, bandwidth parameter.
        unique: insist on unique elements

    Returns:
        (coreset indices, coreset Gram matrix, coreset kernel mean)
    """
    X = jnp.asarray(X)
    g = vmap(grad_log_f_X, (0, None, None), 0)
    grads = g(X, X, nu).squeeze()
    k_pairwise = jit(vmap(vmap(kernel, (None, 0, None, 0, None,
                                        None), 0), (0, None, 0, None, None, None), 0))
    n = X.shape[0]
    k_vec = jit(vmap(kernel, in_axes=(0, None, 0, None, None, None)))

    if K_mean is None:
        K_mean = calculate_K_sum(X, k_pairwise, max_size, grads, nu) / n
    else:
        K_mean = jnp.asarray(K_mean)

    objective = K_mean.copy()
    S = jnp.zeros(n_core, dtype=jnp.int32)
    K = jnp.zeros((n_core, n))

    # Greedly select coreset points
    body = partial(stein_greedy_body, X=X, k_vec=k_vec,
                   K_mean=K_mean, grads=grads, n=n, nu=nu, unique=unique)
    S, K, _ = lax.fori_loop(0, n_core, body, (S, K, objective))
    Kbar = K.mean(axis=1)
    Kc = K[:, S]
    return S, Kc, Kbar


@jit
def fw_linesearch(arg_x_t: int, K: ArrayLike, Ek: ArrayLike) -> Array:
    """Frank-Wolfe line search.

    Args:
        arg_x_t: index of previous
        K: Gram matrix
        Ek: Gram matrix mean

    Returns:
        FW weight, as a zero-dimensional array
    """
    K = jnp.asarray(K)
    Ek = jnp.asarray(Ek)

    arg_x_p = jnp.argmin(K[arg_x_t] - Ek)
    rho_t_num = K[arg_x_t, arg_x_t] - \
        K[arg_x_t, arg_x_p] - Ek[arg_x_t] + Ek[arg_x_p]
    rho_t_den = K[arg_x_t, arg_x_t] + \
        K[arg_x_p, arg_x_p] - 2*K[arg_x_t, arg_x_p]
    rho_t = rho_t_num/rho_t_den
    return rho_t


@jit
def herding_body(
        i: int,
        val: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
) -> tuple[Array, Array, Array, Array]:
    """Basic herding body

    Args:
        i: Loop counter
        val: Loop updateables

    Returns:
        (coreset indices, objective, K mean, K)
    """
    S, objective, Kbar, K = val
    S = jnp.asarray(S)
    objective = jnp.asarray(objective)
    Kbar = jnp.asarray(Kbar)
    K = jnp.asarray(K)
    j = objective.argmax()
    S = S.at[i].set(j)
    objective = objective * (i + 1) / (i + 2) + (Kbar - K[S[i]]) / (i + 2)
    return S, objective, Kbar, K


@jit
def greedy_herding_body(
        i: int,
        val: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
) -> tuple[Array, Array, Array, Array]:
    """Stein thinning body

    Args:
        i: Loop counter
        val: Loop updateables

    Returns:
        coreset indices, objective, K mean, K)
    """
    S, objective, Kbar, K = val
    S = jnp.asarray(S)
    objective = jnp.asarray(objective)
    Kbar = jnp.asarray(Kbar)
    K = jnp.asarray(K)
    j = (objective + jnp.diag(K) / 2.).argmin()
    S = S.at[i].set(j)
    objective += K[S[i]]
    return S, objective, Kbar, K


@jit
def fw_herding_body(
        i: int,
        val: tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike],
) -> tuple[Array, Array, Array, Array]:
    """Frank-Wolfe herding body

    Args:
        i: Loop counter
        val: Loop updateables

    Returns:
        (coreset indices, objective, K mean, K)
    """
    S, objective, Kbar, K = val
    S = jnp.asarray(S)
    objective = jnp.asarray(objective)
    Kbar = jnp.asarray(Kbar)
    K = jnp.asarray(K)
    j = objective.argmax()
    S = S.at[i].set(j)
    rho = fw_linesearch(S[i], K, Kbar)
    objective = objective * (1 - rho) + (Kbar - K[S[i]])*rho
    return S, objective, Kbar, K


# def kernel_herding(
#         X: ArrayLike,
#         m: int,
#         method: str = "herding",
#         kernel: KernelFunction | None = None,
#         K: ArrayLike | None = None,
# ) -> Array:
#     if kernel is not None and K is None:
#         K = kernel(X, X)
#     Kbar = K.mean(axis=0)
#     n = X.shape[0]
#     S = jnp.zeros(m, dtype=jnp.int32)
#     # objective = jnp.zeros(n)
#     objective = Kbar.copy()
#     init_val = (S, objective, Kbar, K)
#     fn = herding_body
#     if method == "greedy":
#         fn = greedy_herding_body
#     elif method == "fw":
#         fn = fw_herding_body
#     val = lax.fori_loop(0, m, fn, init_val)
#     S = val[0]
#     return S

def scalable_stein_kernel_pc_imq_element(*args, **kwargs) -> Callable[..., Array]:
    """A wrapper for scalable (parallelised) herding with a decorated function

    Returns:
        Object: Same returns as stein_kernel_pc_imq_element
    """
    return stein_kernel_pc_imq_element(*args, **kwargs)


def scalable_rbf_grad_log_f_X(*args, **kwargs) -> Callable[..., Array]:
    """A wrapper for scalable (parallelised) herding with a decorated function

    Returns:
        Object: Same returns as rbf_grad_log_f_X
    """
    return rbf_grad_log_f_X(*args, **kwargs)


def scalable_herding(
        X: ArrayLike,
        indices: ArrayLike,
        n_core: int,
        function: Callable[..., Array],
        w_function: KernelFunction | None,
        size: int = 1000,
        parallel: bool = True,
        **kwargs,
) -> tuple[Array, Array]:
    """Scalable kernel herding. Uses a kd-tree to partition X space into patches, upon each a kernel herding problem is solved.

    The setup is a little intricate. First, n_core < size. If we have n points, unweighted herding is run recursively on each patch of k = ceil(n/size) points.

    If r is the recursion depth, then we recurse unweighted r = floor(log_{n_core/size} (n_core/n)) times, each recursion giving n_r = C*k_{r - 1} points.
    Unpacking the recursion, this gives n_r ~= n_0(n_core/n_size)^r.

    Once n_core < n_r <= size, we run a final weighted herding (if weighting is requested) to give n_core points.

    Args:
        X: n x d original data
        indices: Indices into original data set; useful for recursion.
        n_core: Number of coreset points to calculate
        function: Kernel function k: R^d x R^d \to R
        w_function: Weights' function. None if unweighted.
        size: Region size in number of points. Defaults to 1000.
        parallel: Use multiprocessing. Defaults to True.

    Returns:
        (coreset, weights). Weights will be empty if unweighted.
    """
    # check parameters to see if we need to invoke the kd-tree and recursion.
    if n_core >= size:
        raise OverflowError("Number of coreset points requested (%d) is larger than the region size (%d). Try increasing the size argument, or reducing the number of coreset points" % (n_core, size))
    X = jnp.asarray(X)
    indices = jnp.asarray(indices)
    n = X.shape[0]
    weights = None
    if n <= n_core:
        coreset = indices
        if w_function is not None:
            _, Kc, Kbar = function(X=X, n_core=n_core, **kwargs)
            weights = w_function(Kc, Kbar)
    elif n_core < n <= size: # tail case
        c, Kc, Kbar = function(X=X, n_core=n_core, **kwargs)
        coreset = indices[c]
        if w_function is not None:
            weights = w_function(Kc, Kbar)
    else:
        # build a kdtree
        kdtree = KDTree(X, leaf_size=size)
        _, nindices, nodes, _ = kdtree.get_arrays()
        new_indices = [jnp.array(nindices[nd[0]: nd[1]]) for nd in nodes if nd[2]]
        split_data = [X[n] for n in new_indices]
        # k = len(split_data)
        # print(n, k, n // k)
        # n_core_ = n_core // k

        # run k coreset problems
        coreset = []
        kwargs["n_core"] = n_core
        if parallel:
            with ThreadPool() as pool:
                res = pool.map_async(partial(function, **kwargs), split_data)
                res.wait()
                for herding_output, idx in zip(res.get(), new_indices):
                    # different herding algorithms return different things
                    if isinstance(herding_output, tuple):
                        c, _, _ = herding_output
                    else:
                        c = herding_output
                    coreset.append(idx[c])

        else:
            for X_, idx in zip(split_data, new_indices):
                c, _, _ = function(X_, **kwargs)
                coreset.append(idx[c])

        coreset = jnp.concatenate(coreset)
        Xc = X[coreset]
        indices_c = indices[coreset]
        # recurse; n_core is already in kwargs
        coreset, weights = scalable_herding(Xc, indices_c, function=function, w_function=w_function, size=size, parallel=parallel, **kwargs)

    return coreset, weights
