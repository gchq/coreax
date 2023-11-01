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

import inspect
import sys
from abc import ABC, abstractmethod
from functools import partial

import jax.lax as lax
import jax.numpy as jnp
from jax import Array, jit, random, tree_util, vmap
from jax.typing import ArrayLike

import coreax.kernel as ck
from coreax.util import ClassFactory

# Refine Functions
#
# These functions take a coreset and refine it by replacing elements to improve the MMD.


class Refine(ABC):
    """
    Base class for refinement functions.
    """

    def __init__(self, kernel: ck.Kernel):
        r"""
        Create a method to refine a coreset by optimising over indices in the dataset.

        The refinement process happens iteratively. Coreset elements are replaced by
        points most reducing the maximum mean discrepancy (MMD). The MMD is defined by
        :math:`\text{MMD}^2(X,X_c) = \mathbb{E}(k(X,X)) + \mathbb{E}(k(X_c,X_c)) -
        2\mathbb{E}(k(X,X_c))`.

        :param kernel: Kernel function
                    :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`

        self.k_pairwise: Vectorised kernel function on pairs `(x,x)`:
                      :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
        self.k_vec: Vectorised kernel function on pairs `(X,x)`:
                      :math:`k: \mathbb{R}^{n \times d} \times \mathbb{R}^d \rightarrow
                       \mathbb{R}^n`
        """

        self.kernel = kernel

        self.k_pairwise = jit(
            vmap(
                vmap(kernel._compute_elementwise, in_axes=(None, 0), out_axes=0),
                in_axes=(0, None),
                out_axes=0,
            )
        )
        self.k_vec = jit(vmap(kernel._compute_elementwise, in_axes=(0, None)))

    @abstractmethod
    def refine(self, x: ArrayLike, S: ArrayLike, K_mean: ArrayLike) -> Array:
        r"""
        Compute the refined coreset, of m points in d dimensions.

        The refinement procedure replaces elements with points most reducing maximum mean
        discrepancy (MMD). The iteration is carried out over points in `x`.

        :param x: :math:`n \times d` original data
        :param S: :math:`m` coreset point indices
        :param K_mean: :math:`1 \times n` Row mean of the :math:`n \times n` kernel matrix
        :return: :math:`m` Refined coreset point indices
        """
        raise NotImplementedError

    def _tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable jit decoration
        of methods inside this class.
        """
        children = ()  # dynamic values
        aux_data = {"kernel": self.kernel}  # static values
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


class RefineRegular(Refine):
    def __init__(self, kernel: ck.Kernel):
        super().__init__(kernel)

    def refine(self, x, S, K_mean) -> Array:
        r"""
        Refine a coreset iteratively. S -> x.

        The refinement procedure replaces elements with points most reducing maximum mean
        discrepancy (MMD). The iteration is carried out over points in `x`. This is a
        post-processing step in coreset generation, through a generic reduction algorithm.

        :param x: :math:`n \times d` original data
        :param S: :math:`m` Coreset point indices
        :param K_mean: :math:`1 \times n` Row mean of the :math:`n \times n` kernel matrix
        :return: :math:`m` Refined coreset point indices
        """

        K_diag = vmap(self.kernel.compute)(x, x)

        S = jnp.asarray(S)
        num_points_in_coreset = len(S)
        body = partial(
            self.refine_body,
            x=x,
            K_mean=K_mean,
            K_diag=K_diag,
        )
        S = lax.fori_loop(0, num_points_in_coreset, body, S)

        return S

    @jit
    def refine_body(
        self, i: int, S: ArrayLike, x: ArrayLike, K_mean: ArrayLike, K_diag: ArrayLike
    ) -> Array:
        r"""
        Execute main loop of the refine method, S -> x.

        :param i: Loop counter
        :param S: Loop updatables
        :param x: Original :math:`n \times d` dataset
        :param K_mean: Mean vector over rows for the Gram matrix, a :math:`1 \times n` array
        :param K_diag: Gram matrix diagonal, a :math:`1 \times n` array
        :returns: Updated loop variables `S`
        """
        S = jnp.asarray(S)
        S = S.at[i].set(
            self.comparison(
                S[i],
                S,
                x,
                K_mean,
                K_diag,
            ).argmax()
        )

        return S

    @jit
    def comparison(
        self,
        i: ArrayLike,
        S: ArrayLike,
        x: ArrayLike,
        K_mean: ArrayLike,
        K_diag: ArrayLike,
    ) -> Array:
        r"""
        Calculate the change in maximum mean discrepancy from point replacement. S -> x.

        The change calculated is from replacing point `i` in `S` with any point in `x`.

        :param S: Coreset point indices
        :param x: :math:`n \times d` original data
        :param K_mean: :math:`1 \times n` Row mean of the :math:`n \times n` kernel matrix
        :param K_diag: Gram matrix diagonal, a :math:`1 \times n` array
        :return: the MMD changes for each candidate point
        """
        S = jnp.asarray(S)
        num_points_in_coreset = len(S)
        x = jnp.asarray(x)
        K_mean = jnp.asarray(K_mean)
        return (
            self.k_vec(x[S], x[i]).sum()
            - self.k_pairwise(x, x[S]).sum(axis=1)
            + self.k_vec(x, x[i])
            - K_diag
        ) / (num_points_in_coreset**2) - (K_mean[i] - K_mean) / num_points_in_coreset


class RefineRandom(Refine):
    def __init__(self, kernel: ck.Kernel, p: float = 0.1):
        self.p = p

        super().__init__(kernel)

    def refine(self, x, S, K_mean):
        r"""
         Refine a coreset iteratively.

        The refinement procedure replaces a random element with the best point among a set
        of candidate point. The candidate points are a random sample of :math:`n \times p`
        points from among the original data.

        :param x: :math:`n \times d` original data
        :param S: Coreset point indices
        :param K_mean: :math:`1 \times n` Row mean of the :math:`n \times n` kernel matrix
        :return: Refined coreset point indices
        """

        K_diag = vmap(self.kernel.compute)(x, x)

        S = jnp.asarray(S)
        x = jnp.asarray(x)
        num_points_in_coreset = len(S)
        num_points_in_x = len(x)
        n_cand = int(num_points_in_x * self.p)
        n_iter = num_points_in_coreset * (num_points_in_x // n_cand)

        key = random.PRNGKey(42)

        body = partial(
            self.refine_rand_body,
            x=x,
            n_cand=n_cand,
            K_mean=K_mean,
            K_diag=K_diag,
        )
        key, S = lax.fori_loop(0, n_iter, body, (key, S))

        return S

    def refine_rand_body(
        self,
        i: int,
        val: tuple[random.PRNGKeyArray, ArrayLike],
        x: ArrayLike,
        n_cand: int,
        K_mean: ArrayLike,
        K_diag: ArrayLike,
    ) -> tuple[random.PRNGKeyArray, Array]:
        r"""
        Execute main loop of the random refine method

        :param i: Loop counter
        :param val: Loop updatables
        :param x: Original :math:`n \times d` dataset
        :param n_cand: Number of candidates for comparison
        :param K_mean: Mean vector over rows for the Gram matrix, a :math:`1 \times n` array
        :param K_diag: Gram matrix diagonal, a :math:`1 \times n` array
        :returns: Updated loop variables `S`
        """
        key, S = val
        S = jnp.asarray(S)
        key, subkey = random.split(key)
        i = random.randint(subkey, (1,), 0, len(S))[0]
        key, subkey = random.split(key)
        cand = random.randint(subkey, (n_cand,), 0, len(x))
        # cand = random.choice(subkey, len(x), (n_cand,), replace=False)
        comps = self.comparison_cand(
            S[i],
            cand,
            S,
            x,
            K_mean,
            K_diag,
        )
        S = lax.cond(jnp.any(comps > 0), self.change, self.nochange, i, S, cand, comps)

        return key, S

    @jit
    def comparison_cand(
        self,
        i: ArrayLike,
        cand: ArrayLike,
        S: ArrayLike,
        x: ArrayLike,
        K_mean: ArrayLike,
        K_diag: ArrayLike,
    ) -> Array:
        r"""
        Calculate the change in maximum mean discrepancy (MMD).

        The change in MMD arises from replacing `i` in `S` with `x`.

        :param i: A coreset index
        :param cand: Indices for randomly sampled candidate points among the original data
        :param S: Coreset point indices
        :param x: :math:`n \times d` original data
        :param K_mean: :math:`1 \times n` Row mean of the :math:`n \times n` kernel matrix
        :param K_diag: Gram matrix diagonal, a :math:`1 \times n` array
        :return: the MMD changes for each candidate point
        """
        S = jnp.asarray(S)
        x = jnp.asarray(x)
        K_mean = jnp.asarray(K_mean)
        K_diag = jnp.asarray(K_diag)
        num_points_in_coreset = len(S)

        return (
            self.k_vec(x[S], x[i]).sum()
            - self.k_pairwise(x[cand, :], x[S]).sum(axis=1)
            + self.k_vec(x[cand, :], x[i])
            - K_diag[cand]
        ) / (num_points_in_coreset**2) - (
            K_mean[i] - K_mean[cand]
        ) / num_points_in_coreset

    @jit
    def change(self, i: int, S: ArrayLike, cand: ArrayLike, comps: ArrayLike) -> Array:
        r"""
        Replace the i^th point in S with the candidate in cand with maximum value in comps.

        S -> x.

        :param i: Index in S to replace
        :param S: The dataset for replacement
        :param cand: A set of candidates for replacement
        :param comps: Comparison values for each candidate
        :return: Updated S, with i^th point replaced
        """
        S = jnp.asarray(S)
        cand = jnp.asarray(cand)
        return S.at[i].set(cand[comps.argmax()])

    @jit
    def nochange(
        self, i: int, S: ArrayLike, cand: ArrayLike, comps: ArrayLike
    ) -> Array:
        r"""
        Convenience function for leaving S unchanged (compare with refine.change). S -> x.

        :param i: Index in S to replace. Not used
        :param S: The dataset for replacement. Will remain unchanged
        :param cand: A set of candidates for replacement. Not used
        :param comps: Comparison values for each candidate. Not used
        :return: The original dataset S, unchanged
        """
        return jnp.asarray(S)


class RefineRev(Refine):
    def __init__(self, kernel: ck.Kernel):
        super().__init__(kernel)

    def refine(
        self,
        x: ArrayLike,
        S: ArrayLike,
        K_mean: ArrayLike,
    ) -> Array:
        r"""
        Refine a coreset iteratively, replacing points which lead to the most improvement.

        This greedy refine method, the iteration is carried out over points in `x`, with
        x -> S.

        :param x: :math:`n \times d` original data
        :param S: :math:`m` Coreset point indices
        :param K_mean: :math:`1 \times n` Row mean of the :math:`n \times n` kernel matrix
        :return: :math:`m` Refined coreset point indices
        """
        x = jnp.asarray(x)
        S = jnp.asarray(S)

        K_diag = vmap(self.kernel.compute)(x, x)

        num_points_in_x = len(x)

        body = partial(
            self.refine_rev_body,
            x=x,
            K_mean=K_mean,
            K_diag=K_diag,
        )
        S = lax.fori_loop(0, num_points_in_x, body, S)

        return S

    def refine_rev_body(
        self, i: int, S: ArrayLike, x: ArrayLike, K_mean: ArrayLike, K_diag: ArrayLike
    ) -> Array:
        r"""
        Execute main loop of the refine method, x -> S.

        :param i: Loop counter
        :param S: Loop updatables
        :param x: Original :math:`n \times d` dataset
        :param K_mean: Mean vector over rows for the Gram matrix, a :math:`1 \times n` array
        :param K_diag: Gram matrix diagonal, a :math:`1 \times n` array
        :returns: Updated loop variables `S`
        """
        comps = self.comparison_rev(
            i,
            S,
            x,
            K_mean,
            K_diag,
        )
        S = lax.cond(
            jnp.any(comps > 0), self.change_rev, self.nochange_rev, i, S, comps
        )

        return S

    @jit
    def comparison_rev(
        self,
        i: int,
        S: ArrayLike,
        x: ArrayLike,
        K_mean: ArrayLike,
        K_diag: ArrayLike,
    ) -> Array:
        r"""
        Calculate the change in maximum mean discrepancy (MMD). x -> S.

        The change in MMD arises from replacing a point in `S` with `x[i]`.

        :param i: Index for original data
        :param S: Coreset point indices
        :param x: :math:`n \times d` original data
        :param K_mean: :math:`1 \times n` Row mean of the :math:`n \times n` kernel matrix
        :param K_diag: Gram matrix diagonal, a :math:`1 \times n` array
        :return: the MMD changes for each point
        """
        S = jnp.asarray(S)
        x = jnp.asarray(x)
        K_mean = jnp.asarray(K_mean)
        K_diag = jnp.asarray(K_diag)
        num_points_in_coreset = len(S)

        return (
            self.k_pairwise(x[S], x[S]).sum(axis=1)
            - self.k_vec(x[S], x[i]).sum()
            + self.k_vec(x[S], x[i])
            - K_diag[S]
        ) / (num_points_in_coreset**2) - (
            K_mean[S] - K_mean[i]
        ) / num_points_in_coreset

    @jit
    def change_rev(self, i: int, S: ArrayLike, comps: ArrayLike) -> Array:
        r"""
        Replace the maximum comps value point in S with i. x -> S.

        :param i: Value to replace into S.
        :param S: The dataset for replacement
        :param comps: Comparison values for each candidate
        :return: Updated S, with maximum comps point replaced
        """
        S = jnp.asarray(S)
        comps = jnp.asarray(comps)
        j = comps.argmax()
        return S.at[j].set(i)

    @jit
    def nochange_rev(self, i: int, S: ArrayLike, comps: ArrayLike) -> Array:
        r"""
        Convenience function for leaving S unchanged (compare with refine.change_rev).

        x -> S.

        :param i: Value to replace into S. Not used
        :param S: The dataset for replacement. Will remain unchanged
        :param comps: Comparison values for each candidate. Not used
        :return: The original dataset S, unchanged
        """
        return jnp.asarray(S)


# Define the pytree node for the added class to ensure methods with jit decorators
# are able to run. We rely on the naming convention that all child classes of
# ScoreMatching include the sub-string ScoreMatching inside of them.
for name, current_class in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    if "Refine" in name and name != "Refine":
        tree_util.register_pytree_node(
            current_class, current_class._tree_flatten, current_class._tree_unflatten
        )

# Set up class factory
refine_factory = ClassFactory(Refine)
refine_factory.register("regular", RefineRegular)
refine_factory.register("random", RefineRandom)
refine_factory.register("rev", RefineRev)
