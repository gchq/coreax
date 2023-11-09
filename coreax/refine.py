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


class Refine(ABC):
    """
    Base class for refinement functions.

    # TODO: Do we want to be able to refine by additional quality measures, e.g.
        KL Divergence, ...?

    # TODO: Related to the above, we could see if using the metrics objects offer an
        easier way to incorporate a generic quality measure

    The refinement process happens iteratively. Coreset elements are replaced by
    points most reducing the maximum mean discrepancy (MMD). The MMD is defined by
    :math:`\text{MMD}^2(X,X_c) = \mathbb{E}(k(X,X)) + \mathbb{E}(k(X_c,X_c)) - 2\mathbb{E}(k(X,X_c))`.

    :param kernel: A :class:`coreax.kernel` object
    """

    def __init__(self, kernel: ck.Kernel):
        """
        Initilise a refinement object.
        """
        # Assign kernel object
        self.kernel = kernel

    @abstractmethod
    def refine(self, x: ArrayLike, coreset_indices: ArrayLike) -> Array:
        r"""
        Compute the refined coreset, of m points in d dimensions.

        The refinement procedure replaces elements with points most reducing maximum
        mean discrepancy (MMD). The iteration is carried out over points in ``x``.

        :param x: :math:`n \times d` original data
        :param coreset_indices: :math:`m` coreset point indices
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
    """
    Define the RefineRegular class.

    # TODO: Update docstrings to better detail the differences between refine methods

    The refinement process happens iteratively. Coreset elements are replaced by
    points most reducing the maximum mean discrepancy (MMD). The MMD is defined by
    :math:`\text{MMD}^2(X,X_c) = \mathbb{E}(k(X,X)) + \mathbb{E}(k(X_c,X_c)) - 2\mathbb{E}(k(X,X_c))`.

    :param kernel: A :class:`coreax.kernel` object
    """

    def __init__(self, kernel: ck.Kernel):
        """
        Initilise a RefineRegular object.
        """
        super().__init__(kernel)

    def refine(self, x: ArrayLike, coreset_indices: ArrayLike) -> Array:
        r"""
        Refine a coreset iteratively.

        The refinement procedure replaces elements with points most reducing maximum
        mean discrepancy (MMD). The iteration is carried out over points in ``x``. This
        is a post-processing step in coreset generation, through a generic reduction
        algorithm.

        :param x: :math:`n \times d` original data
        :param coreset_indices: ArrayLike: :math:`m` Coreset point indices
        :return: :math:`m` Refined coreset point indices
        """

        K_diag = vmap(self.kernel.compute)(x, x)
        K_mean = self.kernel.calculate_kernel_matrix_row_sum_mean(x)

        coreset_indices = jnp.asarray(coreset_indices)
        num_points_in_coreset = len(coreset_indices)
        body = partial(
            self.refine_body,
            x=x,
            K_mean=K_mean,
            K_diag=K_diag,
        )
        coreset_indices = lax.fori_loop(0, num_points_in_coreset, body, coreset_indices)

        return coreset_indices

    @jit
    def refine_body(
        self,
        i: int,
        coreset_indices: ArrayLike,
        x: ArrayLike,
        K_mean: ArrayLike,
        K_diag: ArrayLike,
    ) -> Array:
        r"""
        Execute main loop of the refine method, S -> x.

        :param i: Loop counter
        :param coreset_indices: Loop updatables
        :param x: Original :math:`n \times d` dataset
        :param K_mean: Mean vector over rows for the Gram matrix, a :math:`1 \times n` array
        :param K_diag: Gram matrix diagonal, a :math:`1 \times n` array
        :returns: Updated loop variables `S`
        """
        coreset_indices = jnp.asarray(coreset_indices)
        coreset_indices = coreset_indices.at[i].set(
            self.comparison(
                coreset_indices[i],
                coreset_indices,
                x,
                K_mean,
                K_diag,
            ).argmax()
        )

        return coreset_indices

    @jit
    def comparison(
        self,
        i: ArrayLike,
        coreset_indices: ArrayLike,
        x: ArrayLike,
        K_mean: ArrayLike,
        K_diag: ArrayLike,
    ) -> Array:
        r"""
        Calculate the change in maximum mean discrepancy from point replacement. S -> x.

        The change calculated is from replacing point `i` in `S` with any point in `x`.

        :param coreset_indices: Coreset point indices
        :param x: :math:`n \times d` original data
        :param K_mean: :math:`1 \times n` Row mean of the :math:`n \times n` kernel matrix
        :param K_diag: Gram matrix diagonal, a :math:`1 \times n` array
        :return: the MMD changes for each candidate point
        """
        coreset_indices = jnp.asarray(coreset_indices)
        num_points_in_coreset = len(coreset_indices)
        x = jnp.asarray(x)
        K_mean = jnp.asarray(K_mean)
        return (
            self.kernel.compute(x[coreset_indices], x[i]).sum()
            - self.kernel.compute(x, x[coreset_indices]).sum(axis=1)
            + self.kernel.compute(x, x[i])[:, 0]
            - K_diag
        ) / (num_points_in_coreset**2) - (K_mean[i] - K_mean) / num_points_in_coreset


class RefineRandom(Refine):
    def __init__(self, kernel: ck.Kernel, p: float = 0.1):
        self.p = p

        super().__init__(kernel)

    def refine(self, x, coreset_indices):
        r"""
         Refine a coreset iteratively.

        The refinement procedure replaces a random element with the best point among a set
        of candidate point. The candidate points are a random sample of :math:`n \times p`
        points from among the original data.

        :param x: :math:`n \times d` original data
        :param coreset_indices: Coreset point indices
        :return: Refined coreset point indices
        """

        K_diag = vmap(self.kernel.compute)(x, x)
        K_mean = self.kernel.calculate_kernel_matrix_row_sum_mean(x)

        coreset_indices = jnp.asarray(coreset_indices)
        x = jnp.asarray(x)
        num_points_in_coreset = len(coreset_indices)
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
        key, coreset_indices = lax.fori_loop(0, n_iter, body, (key, coreset_indices))

        return coreset_indices

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
        key, coreset_indices = val
        coreset_indices = jnp.asarray(coreset_indices)
        key, subkey = random.split(key)
        i = random.randint(subkey, (1,), 0, len(coreset_indices))[0]
        key, subkey = random.split(key)
        cand = random.randint(subkey, (n_cand,), 0, len(x))
        # cand = random.choice(subkey, len(x), (n_cand,), replace=False)
        comps = self.comparison_cand(
            coreset_indices[i],
            cand,
            coreset_indices,
            x,
            K_mean,
            K_diag,
        )
        coreset_indices = lax.cond(
            jnp.any(comps > 0),
            self.change,
            self.nochange,
            i,
            coreset_indices,
            cand,
            comps,
        )

        return key, coreset_indices

    @jit
    def comparison_cand(
        self,
        i: ArrayLike,
        cand: ArrayLike,
        coreset_indices: ArrayLike,
        x: ArrayLike,
        K_mean: ArrayLike,
        K_diag: ArrayLike,
    ) -> Array:
        r"""
        Calculate the change in maximum mean discrepancy (MMD).

        The change in MMD arises from replacing `i` in `S` with `x`.

        :param i: A coreset index
        :param cand: Indices for randomly sampled candidate points among the original data
        :param coreset_indices: Coreset point indices
        :param x: :math:`n \times d` original data
        :param K_mean: :math:`1 \times n` Row mean of the :math:`n \times n` kernel matrix
        :param K_diag: Gram matrix diagonal, a :math:`1 \times n` array
        :return: the MMD changes for each candidate point
        """
        coreset_indices = jnp.asarray(coreset_indices)
        x = jnp.asarray(x)
        K_mean = jnp.asarray(K_mean)
        K_diag = jnp.asarray(K_diag)
        num_points_in_coreset = len(coreset_indices)

        return (
            self.kernel.compute(x[coreset_indices], x[i]).sum()
            - self.kernel.compute(x[cand, :], x[coreset_indices]).sum(axis=1)
            + self.kernel.compute(x[cand, :], x[i])[:, 0]
            - K_diag[cand]
        ) / (num_points_in_coreset**2) - (
            K_mean[i] - K_mean[cand]
        ) / num_points_in_coreset

    @jit
    def change(
        self, i: int, coreset_indices: ArrayLike, cand: ArrayLike, comps: ArrayLike
    ) -> Array:
        r"""
        Replace the i^th point in S with the candidate in cand with maximum value in comps.

        coreset_indices -> x.

        :param i: Index in S to replace
        :param coreset_indices: The dataset for replacement
        :param cand: A set of candidates for replacement
        :param comps: Comparison values for each candidate
        :return: Updated S, with i^th point replaced
        """
        coreset_indices = jnp.asarray(coreset_indices)
        cand = jnp.asarray(cand)
        return coreset_indices.at[i].set(cand[comps.argmax()])

    @jit
    def nochange(
        self, i: int, coreset_indices: ArrayLike, cand: ArrayLike, comps: ArrayLike
    ) -> Array:
        r"""
        Convenience function for leaving S unchanged (compare with refine.change). coreset_indices -> x.

        :param i: Index in coreset_indices to replace. Not used
        :param coreset_indices: The dataset for replacement. Will remain unchanged
        :param cand: A set of candidates for replacement. Not used
        :param comps: Comparison values for each candidate. Not used
        :return: The original dataset S, unchanged
        """
        return jnp.asarray(coreset_indices)


class RefineRev(Refine):
    def __init__(self, kernel: ck.Kernel):
        super().__init__(kernel)

    def refine(self, x: ArrayLike, coreset_indices: ArrayLike) -> Array:
        r"""
        Refine a coreset iteratively, replacing points which lead to the most improvement.

        This greedy refine method, the iteration is carried out over points in `x`, with
        x -> coreset_indices.

        :param x: :math:`n \times d` original data
        :param coreset_indices: :math:`m` Coreset point indices
        :return: :math:`m` Refined coreset point indices
        """
        x = jnp.asarray(x)
        coreset_indices = jnp.asarray(coreset_indices)

        K_diag = vmap(self.kernel.compute)(x, x)
        K_mean = self.kernel.calculate_kernel_matrix_row_sum_mean(x)

        num_points_in_x = len(x)

        body = partial(
            self.refine_rev_body,
            x=x,
            K_mean=K_mean,
            K_diag=K_diag,
        )
        coreset_indices = lax.fori_loop(0, num_points_in_x, body, coreset_indices)

        return coreset_indices

    def refine_rev_body(
        self,
        i: int,
        coreset_indices: ArrayLike,
        x: ArrayLike,
        K_mean: ArrayLike,
        K_diag: ArrayLike,
    ) -> Array:
        r"""
        Execute main loop of the refine method, x -> S.

        :param i: Loop counter
        :param coreset_indices: Loop updatables
        :param x: Original :math:`n \times d` dataset
        :param K_mean: Mean vector over rows for the Gram matrix, a :math:`1 \times n` array
        :param K_diag: Gram matrix diagonal, a :math:`1 \times n` array
        :returns: Updated loop variables `S`
        """
        comps = self.comparison_rev(
            i,
            coreset_indices,
            x,
            K_mean,
            K_diag,
        )
        coreset_indices = lax.cond(
            jnp.any(comps > 0),
            self.change_rev,
            self.nochange_rev,
            i,
            coreset_indices,
            comps,
        )

        return coreset_indices

    @jit
    def comparison_rev(
        self,
        i: int,
        coreset_indices: ArrayLike,
        x: ArrayLike,
        K_mean: ArrayLike,
        K_diag: ArrayLike,
    ) -> Array:
        r"""
        Calculate the change in maximum mean discrepancy (MMD). x -> coreset_indices.

        The change in MMD arises from replacing a point in `coreset_indices` with `x[i]`.

        :param i: Index for original data
        :param coreset_indices: Coreset point indices
        :param x: :math:`n \times d` original data
        :param K_mean: :math:`1 \times n` Row mean of the :math:`n \times n` kernel matrix
        :param K_diag: Gram matrix diagonal, a :math:`1 \times n` array
        :return: the MMD changes for each point
        """
        coreset_indices = jnp.asarray(coreset_indices)
        x = jnp.asarray(x)
        K_mean = jnp.asarray(K_mean)
        K_diag = jnp.asarray(K_diag)
        num_points_in_coreset = len(coreset_indices)

        return (
            self.kernel.compute(x[coreset_indices], x[coreset_indices]).sum(axis=1)
            - self.kernel.compute(x[coreset_indices], x[i]).sum()
            + self.kernel.compute(x[coreset_indices], x[i])[:, 0]
            - K_diag[coreset_indices]
        ) / (num_points_in_coreset**2) - (
            K_mean[coreset_indices] - K_mean[i]
        ) / num_points_in_coreset

    @jit
    def change_rev(self, i: int, coreset_indices: ArrayLike, comps: ArrayLike) -> Array:
        r"""
        Replace the maximum comps value point in S with i. x -> S.

        :param i: Value to replace into S.
        :param coreset_indices: The dataset for replacement
        :param comps: Comparison values for each candidate
        :return: Updated S, with maximum comps point replaced
        """
        coreset_indices = jnp.asarray(coreset_indices)
        comps = jnp.asarray(comps)
        j = comps.argmax()
        return coreset_indices.at[j].set(i)

    @jit
    def nochange_rev(
        self, i: int, coreset_indices: ArrayLike, comps: ArrayLike
    ) -> Array:
        r"""
        Convenience function for leaving S unchanged (compare with refine.change_rev).

        x -> S.

        :param i: Value to replace into S. Not used
        :param coreset_indices: The dataset for replacement. Will remain unchanged
        :param comps: Comparison values for each candidate. Not used
        :return: The original dataset S, unchanged
        """
        return jnp.asarray(coreset_indices)


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
