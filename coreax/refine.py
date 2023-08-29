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

# Support annotations with tuple in Python < 3.9
# TODO: Remove once no longer supporting old code
from __future__ import annotations

import jax.numpy as jnp
import jax.lax as lax
from jax.typing import ArrayLike

from jax import jit, vmap, random, Array
from functools import partial

from coreax.utils import KernelFunction


#
# Refine Functions 
# 
# These functions take a coreset S as an input and refine it by replacing elements to improve the MMD. 

def refine(
        x: ArrayLike,
        S: ArrayLike,
        kernel: KernelFunction,
        K_mean: ArrayLike,
) -> Array:
    r"""
    Refine a coreset iteratively.

    The refinement procedure replaces elements with points most reducing maximum mean
    discrepancy (MMD).

    :param x: :math:`n \times d` original data
    :param S: Coreset point indices
    :param kernel: Kernel function
                   :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param K_mean: Kernel matrix row sum divided by n
    :return: Refined coreset point indices
    """
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    k_vec = jit(vmap(kernel, in_axes=(0,None)))
    
    K_diag = vmap(kernel)(x,x)

    S = jnp.asarray(S)
    m = len(S)
    body = partial(refine_body, x=x, K_mean=K_mean, K_diag=K_diag, k_pairwise=k_pairwise, k_vec=k_vec)
    S = lax.fori_loop(0, m, body, S)
    
    return S

@partial(jit, static_argnames=["k_pairwise", "k_vec"])
def refine_body(
        i: int,
        S: ArrayLike,
        x: ArrayLike,
        K_mean: ArrayLike,
        K_diag: ArrayLike,
        k_pairwise: KernelFunction,
        k_vec: KernelFunction,
) -> Array:
    S = jnp.asarray(S)
    S = S.at[i].set(comparison(S[i], S, x, K_mean, K_diag, k_pairwise, k_vec).argmax())

    return S

@partial(jit, static_argnames=["k_pairwise", "k_vec"])
def comparison(
        i: ArrayLike,
        S: ArrayLike,
        x: ArrayLike,
        K_mean: ArrayLike,
        K_diag: ArrayLike,
        k_pairwise: KernelFunction,
        k_vec: KernelFunction,
) -> Array:
    r"""
    Calculate the change in maximum mean discrepancy from point replacement.

    The change calculated is from replacing point `i` in `S` with any point in `x`.

    :param i: TODO
    :param S: TODO
    :param x: TODO
    :param K_mean: TODO
    :param K_diag: TODO
    :param k_pairwise: TODO
    :param k_vec: TODO
    :return: A vector of maximum mean discrepancy deltas.
    """
    S = jnp.asarray(S)
    m = len(S)
    x = jnp.asarray(x)
    K_mean = jnp.asarray(K_mean)
    return (
        (k_vec(x[S], x[i]).sum() - k_pairwise(x,x[S]).sum(axis=1) + k_vec(x,x[i]) - K_diag)/(m*m) - 
        (K_mean[i] - K_mean)/m
    )


def refine_rand(
        x: ArrayLike,
        S: ArrayLike,
        kernel: KernelFunction,
        K_mean: ArrayLike,
        p: float = 0.1,
) -> Array:
    r"""
    Refine a coreset iteratively.

    The refinement procedure replaces a random element with the best point among a set
    of candidate point. The candidate points are a random sample of :math:`n \times p`
    points from among the original data.

    :param x: :math:`n \times d` original data
    :param S: Coreset point indices
    :param kernel: Kernel function
                   :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param K_mean: Kernel matrix row sum divided by n
    :param p: Proportion of original data to use as candidates
    :return: Refined coreset point indices
    """
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    k_vec = jit(vmap(kernel, in_axes=(0,None)))
    
    K_diag = vmap(kernel)(x,x)

    S = jnp.asarray(S)
    x = jnp.asarray(x)
    m = len(S)
    n = len(x)
    n_cand = int(n*p)
    n_iter = m*(n//n_cand)

    key = random.PRNGKey(42)

    body = partial(refine_rand_body, x=x, n_cand=n_cand, K_mean=K_mean, K_diag=K_diag, k_pairwise=k_pairwise, k_vec=k_vec)
    key,S = lax.fori_loop(0, n_iter, body, (key,S))

    return S

def refine_rand_body(
        i: int,
        val: tuple[random.PRNGKeyArray, ArrayLike],
        x: ArrayLike,
        n_cand: int,
        K_mean: ArrayLike,
        K_diag: ArrayLike,
        k_pairwise: KernelFunction,
        k_vec: KernelFunction,
) -> tuple[random.PRNGKeyArray, Array]:

    key, S = val
    S = jnp.asarray(S)
    key, subkey = random.split(key)
    i = random.randint(subkey, (1,), 0, len(S))[0]
    key, subkey = random.split(key)
    cand = random.randint(subkey, (n_cand,), 0, len(x))
    #cand = random.choice(subkey, len(x), (n_cand,), replace=False)
    comps = comparison_cand(S[i], cand, S, x, K_mean, K_diag, k_pairwise, k_vec)
    S = lax.cond(jnp.any(comps > 0), change, nochange, i, S, cand, comps)

    return key,S

@partial(jit, static_argnames=["k_pairwise", "k_vec"])
def comparison_cand(
        i: ArrayLike,
        cand: ArrayLike,
        S: ArrayLike,
        x: ArrayLike,
        K_mean: ArrayLike,
        K_diag: ArrayLike,
        k_pairwise: KernelFunction,
        k_vec: KernelFunction,
) -> Array:
    r"""
    Calculate the change in maximum mean discrepancy (MMD).

    The change in MMD arises from replacing `i` in `S` with `x`.

    :param i: A coreset index
    :param cand: Indices for randomly sampled candidate points among the original data
    :param S: Coreset point indices
    :param x: :math:`n \times d` original data
    :param K_mean: Kernel matrix row sum divided by n
    :param K_diag: *TODO*
    :param k_pairwise: *TODO*
    :param k_vec: *TODO*
    :return: *TODO*
    """
    S = jnp.asarray(S)
    x = jnp.asarray(x)
    K_mean = jnp.asarray(K_mean)
    K_diag = jnp.asarray(K_diag)
    m = len(S)
    
    return (
        (k_vec(x[S], x[i]).sum() - k_pairwise(x[cand,:],x[S]).sum(axis=1) + k_vec(x[cand,:],x[i]) - K_diag[cand])/(m*m) - 
        (K_mean[i] - K_mean[cand])/m
    )

@jit
def change(i: int, S: ArrayLike, cand: ArrayLike, comps: ArrayLike) -> Array:
    S = jnp.asarray(S)
    cand = jnp.asarray(cand)
    return S.at[i].set(cand[comps.argmax()])
@jit
def nochange(i: int, S: ArrayLike, cand: ArrayLike, comps: ArrayLike) -> Array:
    return jnp.asarray(S)


def refine_rev(
        x: ArrayLike,
        S: ArrayLike,
        kernel: KernelFunction,
        K_mean: ArrayLike,
) -> Array:
    r"""
    Refine a coreset iteratively, replacing points which lead to the most improvement.

    The iteration is carred out over points in `x`.

    :param x: :math:`n \times d` original data
    :param S: Coreset point indices
    :param kernel: Kernel function
                   :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param K_mean: Kernel matrix row sum divided by n
    :return: Refined coreset point indices
    """
    x = jnp.asarray(x)
    S = jnp.asarray(S)

    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    k_vec = jit(vmap(kernel, in_axes=(0,None)))
    
    K_diag = vmap(kernel)(x,x)

    m = len(S)
    n = len(x)

    body = partial(refine_rev_body, x=x, K_mean=K_mean, K_diag=K_diag, k_pairwise=k_pairwise, k_vec=k_vec)
    S = lax.fori_loop(0, n, body, S)

    return S

def refine_rev_body(
        i: int,
        S: ArrayLike,
        x: ArrayLike,
        K_mean: ArrayLike,
        K_diag: ArrayLike,
        k_pairwise: KernelFunction,
        k_vec: KernelFunction,
) -> Array:

    comps = comparison_rev(i,S,x,K_mean,K_diag, k_pairwise, k_vec)
    S = lax.cond(jnp.any(comps > 0), change_rev, nochange_rev, i, S, comps)

    return S

@partial(jit, static_argnames=["k_pairwise", "k_vec"])
def comparison_rev(
        i: int,
        S: ArrayLike,
        x: ArrayLike,
        K_mean: ArrayLike,
        K_diag: ArrayLike,
        k_pairwise: KernelFunction,
        k_vec: KernelFunction,
) -> Array:
    r"""
    Calculate the change in maximum mean discrepancy (MMD).

    The change in MMD arises from replacing a point in `S` with `x[i]`.

    :param i: Index for original data
    :param S: Coreset point indices
    :param x: :math:`n \times d` original data
    :param K_mean: Kernel matrix row sum divided by n
    :param K_diag: *TODO*
    :param k_pairwise: *TODO*
    :param k_vec: *TODO*
    :return: *TODO*
    """
    S = jnp.asarray(S)
    x = jnp.asarray(x)
    K_mean = jnp.asarray(K_mean)
    K_diag = jnp.asarray(K_diag)
    m = len(S)

    return (
        (k_pairwise(x[S], x[S]).sum(axis=1) - k_vec(x[S],x[i]).sum() + k_vec(x[S],x[i]) - K_diag[S])/(m*m) - 
        (K_mean[S] - K_mean[i])/m
    )

@jit
def change_rev(i: int, S: ArrayLike, comps: ArrayLike) -> Array:
    S = jnp.asarray(S)
    comps = jnp.asarray(comps)
    j = comps.argmax()
    return S.at[j].set(i)

@jit
def nochange_rev(i: int, S: ArrayLike, comps: ArrayLike) -> Array:
    return jnp.asarray(S)
