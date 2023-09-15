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

from collections.abc import Callable

import jax.numpy as jnp
import jax.lax as lax
from jax.typing import ArrayLike

from jax import vmap, jit, Array

from coreax.kernel import rbf_grad_log_f_x, stein_kernel_pc_imq_element
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
    r"""
    Execute main loop of greedy kernel herding.

    :param i: Loop counter
    :param val: Loop updatables
    :param X: Original :math:`n \times d` dataset
    :param k_vec: Vectorised kernel function on pairs `(X,x)`:
                  :math:`k: \mathbb{R}^{n \times d} \times \mathbb{R}^d \rightarrow \mathbb{R}^n`
    :param K_mean: Mean vector over rows for the Gram matrix, a :math:`1 \times n` array
    :param unique: Flag for enforcing unique elements
    :returns: Updated loop variables (`coreset`, `Gram matrix`, `objective`)
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
    r"""
    Execute the main loop of greedy Stein herding.

    :param i: Loop counter
    :param val: Loop updatables
    :param X: Original :math:`n \times d` dataset
    :param k_vec: Vectorised kernel function on pairs `(X,x,Y,y)`:
                  :math:`k: \mathbb{R}^{n \times d} \times \mathbb{R}^d \times`
                  :math:`\mathbb{R}^{n \times d} \times \mathbb{R}^d \rightarrow`
                  :math:`\mathbb{R}^n`
    :param K_mean: Mean vector over rows for the Gram matrix, a :math:`1 \times n` array
    :param grads: Gradients of log-PDF evaluated at `X`, an :math:`n \times d` array
    :param n: Number of data points inducing the original PDF
    :param nu: Bandwidth parameter for the base kernel of the Stein kernel
    :param unique: Flag for enforcing unique elements
    :returns: Updated loop variables (`coreset`, `coreset Gram matrix`, `K_t` objective)
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
        max_size: int = 10_000,
        K_mean: ArrayLike | None = None,
        unique: bool = True,
) -> tuple[Array, Array, Array]:
    r"""
    Execute kernel herding algorithm with Jax.

    :param X: Original :math:`n \times d` dataset
    :param n_core: Number of coreset points to calculate
    :param kernel: Kernel function
                   :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param max_size: Size of matrix blocks to process
    :param K_mean: Row sum of kernel matrix divided by `n`
    :param unique: Flag for enforcing unique elements
    :returns: Coreset point indices, coreset Gram matrix & corset Gram mean
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
        grad_log_f_X: Callable[[ArrayLike, ArrayLike, float], Array] |
                      Callable[[ArrayLike], Array],
        K_mean: ArrayLike | None = None,
        max_size: int = 10_000,
        nu: float = 1.,
        unique: bool = True,
        sm: bool = False
) -> tuple[Array, Array, Array]:
    r"""
    Execute Stein herding.

    :param X: Original :math:`n \times d` dataset
    :param n_core: Number of coreset points to calculate
    :param kernel: Kernel function
                   :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param grad_log_f_X: Function computing gradient of log-PDF
                         :math:`g: X \rightarrow Y`
    :param max_size: Size of matrix blocks to process
    :param K_mean: Row sum of kernel matrix divided by `n`
    :param unique: Flag for enforcing unique elements
    :param nu: Bandwidth parameter for the base kernel of the Stein kernel
    :returns: Coreset point indices, coreset Gram matrix & corset Gram mean
    """
    X = jnp.asarray(X)
    if sm:
        grads = grad_log_f_X(X)
    else:
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
    r"""
    Execute Frank-Wolfe line search.

    :param arg_x_t: Previous index
    :param K: Gram matrix
    :param Ek: Gram matrix mean
    :return: Frank-Wolfe weight as a 0-dimensional array
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
    r"""
    Execute body of default herding.

    :param i: Loop counter
    :param val: Loop updatables
    :return: Coreset indices, objective, Gram matrix mean and Gram matrix
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
    r"""
    Execute body of Stein thinning.

    :param i: Loop counter
    :param val: Loop updatables
    :return: Coreset indices, objective, Gram matrix mean and Gram matrix
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
    r"""
    Execute body of Frank-Wolfe herding.

    :param i: Loop counter
    :param val: Loop updatables
    :return: Coreset indices, objective, Gram matrix mean and Gram matrix
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
    r"""
    A wrapper for scalable (parallelised) herding with a decorated function.

    This function is deprecated, and scheduled for removal.

    :return: Kernel evaluation at `(x,y)`, 0-dimensional array
    """
    return stein_kernel_pc_imq_element(*args, **kwargs)


def scalable_rbf_grad_log_f_X(*args, **kwargs) -> Callable[..., Array]:
    r"""
    A wrapper for scalable (parallelised) herding with a decorated function.

    This function is deprecated, and scheduled for removal.

    :return: An :math:`n \times d` array of gradients evaluated at values of `X`
    """
    return rbf_grad_log_f_x(*args, **kwargs)


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
    r"""
    Execute scalable kernel herding.

    This uses a `kd-tree` to partition `X`-space into patches. Upon each of these a
    kernel herding problem is solved.

    There is some intricate setup:

        #.  Parameter `n_core` must be less than `size`.
        #.  If we have :math:`n` points, unweighted herding is executed recursively on
            each patch of :math:`\lceil \frac{n}{size} \rceil` points.
        #.  If :math:`r` is the recursion depth, then we recurse unweighted for
            :math:`r` iterations where

            .. math::

                     r = \lfloor \log_{frac{n_core}{size}}(\frac{n_core}{n})\rfloor

            Each recursion gives :math:`n_r = C \times k_{r-1}` points. Unpacking the
            recursion, this gives
            :math:`n_r \approx n_0 \left( \frac{n_core}{n_size}\right)^r`.
        #.  Once :math:`n_core < n_r \leq size`, we run a final weighted herding (if
            weighting is requested) to give :math:`n_core` points.


    :param X: Original :math:`n \times d` dataset
    :param indices: Indices into original dataset, used for recursion
    :param n_core: Number of coreset points to calculate
    :param function: The Kernel function,
                     :math:`k: \mathbb{R}^d \times \mathbb{R}^d`
                     :math:`\rightarrow \mathbb{R}
    :param w_function: Weights function. If unweighted, this is `None`
    :param size: Region size in number of points. Optional, defaults to `1000`
    :param parallel: Use multiprocessing. Optional, defaults to `True`
    :return: Coreset and weights, where weights is empty if unweighted
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
    elif n_core < n <= size:
        # Tail case
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
