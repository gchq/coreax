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
import jax.lax as lax

from jax import jit, vmap, random
from functools import partial

#
# Refine Functions 
# 
# These functions take a coreset S as an input and refine it by replacing elements to improve the MMD. 

def refine(x, S, kernel, K_mean):
    """
    Given a coreset S refine the coreset by iteratively replacing each element of S with the point in 
    x which gives greatest reduction in mmd.

    Args:
        x (array_like): n x d original data 
        S (array_like): coreset point indices
        kernel (callable): kernel function k: R^d x R^d \to R
        K_mean (array_like): n row sum of kernel matrix divided by n 
    
    Returns:
        array: coreset point indices 
    """
    
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    k_vec = jit(vmap(kernel, in_axes=(0,None)))
    
    K_diag = vmap(kernel)(x,x)
    
    m = len(S)
    body = partial(refine_body, x=x, K_mean=K_mean, K_diag=K_diag, k_pairwise=k_pairwise, k_vec=k_vec)
    S = lax.fori_loop(0, m, body, S)
    
    return S

@partial(jit, static_argnames=["k_pairwise", "k_vec"])
def refine_body(i, S, x, K_mean, K_diag, k_pairwise, k_vec):

    S = S.at[i].set(comparison(S[i], S, x, K_mean, K_diag, k_pairwise, k_vec).argmax())

    return S

@partial(jit, static_argnames=["k_pairwise", "k_vec"])
def comparison(i, S, x, K_mean, K_diag, k_pairwise, k_vec):
    """
    Calculate the change the mmd delta from replacing i in S with any point in x. 
    Returns a vector o f deltas.
    """
    m = len(S)
    
    return (
        (k_vec(x[S], x[i]).sum() - k_pairwise(x,x[S]).sum(axis=1) + k_vec(x,x[i]) - K_diag)/(m*m) - 
        (K_mean[i] - K_mean)/m
    )


def refine_rand(x, S, kernel, K_mean, p=0.1):
    """
    Given a coreset S refines the coreset by iteratively replacing a random element of S with the best 
    point in a random sample of n*p candidate points. 

    Args:
        x (array_like): n x d original data 
        S (array_like): coreset point indices
        kernel (callable): kernel function k: R^d x R^d \to R
        K_mean (array_like): n row sum of kernel matrix divided by n 
        p (float): proportion of original data to use as candidates.
    
    Returns:
        array: coreset point indices 
    """
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    k_vec = jit(vmap(kernel, in_axes=(0,None)))
    
    K_diag = vmap(kernel)(x,x)

    m = len(S)
    n = len(x)
    n_cand = int(n*p)
    n_iter = m*(n//n_cand)

    key = random.PRNGKey(42)

    body = partial(refine_rand_body, x=x, n_cand=n_cand, K_mean=K_mean, K_diag=K_diag, k_pairwise=k_pairwise, k_vec=k_vec)
    key,S = lax.fori_loop(0, n_iter, body, (key,S))

    return S

def refine_rand_body(i, val, x, n_cand, K_mean, K_diag, k_pairwise, k_vec):

    key, S = val 
    key, subkey = random.split(key)
    i = random.randint(subkey, (1,), 0, len(S))[0]
    key, subkey = random.split(key)
    cand = random.randint(subkey, (n_cand,), 0, len(x))
    #cand = random.choice(subkey, len(x), (n_cand,), replace=False)
    comps = comparison_cand(S[i], cand, S, x, K_mean, K_diag, k_pairwise, k_vec)
    S = lax.cond(jnp.any(comps > 0), change, nochange, i, S, cand, comps)

    return key,S

@partial(jit, static_argnames=["k_pairwise", "k_vec"])
def comparison_cand(i, cand, S, x, K_mean, K_diag, k_pairwise, k_vec):
    """
    Calculate the change the mmd delta from replacing i in S with any point in x. 
    Returns a vector o f deltas.
    """
    m = len(S)
    
    return (
        (k_vec(x[S], x[i]).sum() - k_pairwise(x[cand,:],x[S]).sum(axis=1) + k_vec(x[cand,:],x[i]) - K_diag[cand])/(m*m) - 
        (K_mean[i] - K_mean[cand])/m
    )

@jit
def change(i, S, cand, comps):

    return S.at[i].set(cand[comps.argmax()])
@jit
def nochange(i, S, cand, comps):

    return S


def refine_rev(x, S, kernel, K_mean):
    """
    Given a coreset S refines the coreset by iterativing over point in x and replacing point in S which gives 
    the most improvement. 

    Args:
        x (array_like): n x d original data 
        S (array_like): coreset point indices
        kernel (callable): kernel function k: R^d x R^d \to R
        K_mean (array_like): n row sum of kernel matrix divided by n 
    
    Returns:
        array: coreset point indices 
    """
    k_pairwise = jit(vmap(vmap(kernel, in_axes=(None,0), out_axes=0), in_axes =(0,None), out_axes=0 ))
    k_vec = jit(vmap(kernel, in_axes=(0,None)))
    
    K_diag = vmap(kernel)(x,x)

    m = len(S)
    n = len(x)

    body = partial(refine_rev_body, x=x, K_mean=K_mean, K_diag=K_diag, k_pairwise=k_pairwise, k_vec=k_vec)
    S = lax.fori_loop(0, n, body, S)

    return S

def refine_rev_body(i, S, x,  K_mean, K_diag, k_pairwise, k_vec):

    comps = comparison_rev(i,S,x,K_mean,K_diag, k_pairwise, k_vec)
    S = lax.cond(jnp.any(comps > 0), change_rev, nochange_rev, i, S, comps)

    return S

@partial(jit, static_argnames=["k_pairwise", "k_vec"])
def comparison_rev(i, S, x, K_mean, K_diag, k_pairwise, k_vec):
    """
    Calculate the change the mmd delta from replacing any point in S with x[i]. 
    Returns a vector o f deltas.
    """
    m = len(S)
    
    return (
        (k_pairwise(x[S], x[S]).sum(axis=1) - k_vec(x[S],x[i]).sum() + k_vec(x[S],x[i]) - K_diag[S])/(m*m) - 
        (K_mean[S] - K_mean[i])/m
    )

@jit
def change_rev(i, S, comps):

    j = comps.argmax()
    return S.at[j].set(i)

@jit
def nochange_rev(i, S, comps):

    return S