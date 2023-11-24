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

r"""
Classes and associated functionality to perform refinement of coresets.

Several greedy algorithms are implemented within this codebase that generate a
compressed representation (coreset) of an original :math:`n \times d` dataset. As these
methods are greedy, it can be beneficial to apply a refinement step after generation,
which is yet another greedy strategy to improve the coreset generated.

Generally, refinement strategies loop through the elements of a corset and consider if
some metric assessing coreset quality can be improved by replacing this element with
another from the original dataset.

All refinement approaches implement :class:`Refine`, in-particular with a method
:meth:`refine` that manipulates a :class:`~coreax.reduction.DataReduction` object.

The other mandatory method to implement is :meth:`_tree_flatten`. To improve
performance, refine computation is jit compiled. As a result, definitions of dynamic
and static values inside :meth:`_tree_flatten` ensure the refine object can be mutated
and the corresponding jit compilation does not yield unexpected results.
"""

# Support annotations with | in Python < 3.10
# TODO: Remove once no longer supporting old code
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial

import jax.lax as lax
import jax.numpy as jnp
from jax import Array, jit, random, tree_util, vmap
from jax.typing import ArrayLike

import coreax.kernel as ck
from coreax.approximation import KernelMeanApproximator
from coreax.reduction import DataReduction
from coreax.util import ClassFactory


class Refine(ABC):
    r"""
    Base class for refinement functions.

    # TODO: Do we want to be able to refine by additional quality measures, e.g.
        KL Divergence, ...?

    # TODO: Related to the above, we could see if using the metrics objects offer an
        easier way to incorporate a generic quality measure

    The refinement process happens iteratively. Coreset elements are replaced by
    points most reducing the maximum mean discrepancy (MMD). The MMD is defined by
    :math:`\text{MMD}^2(X,X_c) = \mathbb{E}(k(X,X)) + \mathbb{E}(k(X_c,X_c)) - 2\mathbb{E}(k(X,X_c))`
    for a dataset ``X`` and corresponding coreset ``X_c``.

    The default calculates the kernel mean row sum in full. To reduce computational
    load, the kernel mean row sum can be approximated by setting the variable
    ``approximate_kernel_row_sum`` = :data:`True` when initializing the Refine object.

    :param approximate_kernel_row_sum: Boolean determining how the kernel mean row
        sum is calculated. If :data:`True`, the sum is approximate.
    :param approximator: :class:`~coreax.approximation.KernelMeanApproximator` object
        for the kernel mean approximation method
    """

    def __init__(
        self,
        approximate_kernel_row_sum: bool = False,
        approximator: type[KernelMeanApproximator] | None = None,
    ):
        """Initialise a refinement object."""
        self.approximate_kernel_row_sum = approximate_kernel_row_sum
        self.approximator = approximator

    @abstractmethod
    def refine(self, data_reduction: DataReduction) -> None:
        r"""
        Compute the refined coreset, of ``m`` points in ``d`` dimensions.

        The :class:`~coreax.reduction.DataReduction` object is updated in-place. The
        refinement procedure replaces elements with points most reducing maximum mean
        discrepancy (MMD).

        :param data_reduction: :class:`~coreax.reduction.DataReduction` object with
            :math:`n \times d` original data, :math:`m` coreset point indices, coreset
            and kernel object
        :return: Nothing
        """

    def _tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable jit decoration
        of methods inside this class.
        """
        # dynamic values:
        children = ()
        # static values:
        aux_data = {}
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
    r"""
    Define the RefineRegular class.

    The refinement process happens iteratively. The iteration is carried out over
    points in ``X``. Coreset elements are replaced by points most reducing the maximum
    mean discrepancy (MMD). The MMD is defined by:
    :math:`\text{MMD}^2(X,X_c) = \mathbb{E}(k(X,X)) + \mathbb{E}(k(X_c,X_c)) - 2\mathbb{E}(k(X,X_c))`
    for a dataset ``X`` and corresponding coreset ``X_c``.
    """

    def refine(self, data_reduction: DataReduction) -> None:
        r"""
        Compute the refined coreset, of ``m`` points in ``d`` dimensions.

        The DataReduction object is updated in-place. The refinement procedure replaces
        elements with points most reducing maximum mean discrepancy (MMD).

        :param data_reduction: :class:`~coreax.reduction.DataReduction` object with
            :math:`n \times d` original data, :math:`m` coreset point indices, coreset
            and kernel object
        :return: Nothing
        """
        x = data_reduction.original_data
        coreset_indices = data_reduction.reduction_indices

        kernel_gram_matrix_diagonal = vmap(data_reduction.kernel.compute)(x, x)

        if self.approximate_kernel_row_sum:
            kernel_mean_row_sum = (
                data_reduction.kernel.approximate_kernel_matrix_row_sum_mean(
                    x, self.approximator
                )
            )
        else:
            kernel_mean_row_sum = (
                data_reduction.kernel.calculate_kernel_matrix_row_sum_mean(x)
            )

        coreset_indices = jnp.asarray(coreset_indices)
        num_points_in_coreset = len(coreset_indices)
        body = partial(
            self._refine_body,
            x=x,
            kernel=data_reduction.kernel,
            kernel_mean_row_sum=kernel_mean_row_sum,
            kernel_gram_matrix_diagonal=kernel_gram_matrix_diagonal,
        )
        coreset_indices = lax.fori_loop(0, num_points_in_coreset, body, coreset_indices)

        data_reduction.reduction_indices = coreset_indices
        data_reduction.reduced_data = data_reduction.original_data[coreset_indices, :]

    @jit
    def _refine_body(
        self,
        i: int,
        coreset_indices: ArrayLike,
        x: ArrayLike,
        kernel: ck.Kernel,
        kernel_mean_row_sum: ArrayLike,
        kernel_gram_matrix_diagonal: ArrayLike,
    ) -> Array:
        r"""
        Execute main loop of the refine method, ``coreset_indices`` -> ``x``.

        :param i: Loop counter
        :param coreset_indices: Loop updatable-variables
        :param x: Original :math:`n \times d` dataset
        :param kernel_mean_row_sum: Mean vector over rows for the Gram matrix,
            a :math:`1 \times n` array
        :param kernel_gram_matrix_diagonal: Gram matrix diagonal, a :math:`1 \times n`
            array
        :return: Updated loop variables `coreset_indices`
        """
        coreset_indices = jnp.asarray(coreset_indices)
        coreset_indices = coreset_indices.at[i].set(
            self._comparison(
                i=coreset_indices[i],
                coreset_indices=coreset_indices,
                x=x,
                kernel=kernel,
                kernel_mean_row_sum=kernel_mean_row_sum,
                kernel_gram_matrix_diagonal=kernel_gram_matrix_diagonal,
            ).argmax()
        )

        return coreset_indices

    @jit
    def _comparison(
        self,
        i: ArrayLike,
        coreset_indices: ArrayLike,
        x: ArrayLike,
        kernel: ck.Kernel,
        kernel_mean_row_sum: ArrayLike,
        kernel_gram_matrix_diagonal: ArrayLike,
    ) -> Array:
        r"""
        Calculate the change in maximum mean discrepancy from point replacement.

        ``coreset_indices`` -> ``x``.

        The change calculated is from replacing point ``i`` in ``coreset_indices`` with
        any point in ``x``.

        :param coreset_indices: Coreset point indices
        :param x: :math:`n \times d` original data
        :param kernel_mean_row_sum: :math:`1 \times n` row mean of the
            :math:`n \times n` kernel matrix
        :param kernel_gram_matrix_diagonal: Gram matrix diagonal,
            a :math:`1 \times n` array
        :return: The MMD changes for each candidate point
        """
        coreset_indices = jnp.asarray(coreset_indices)
        num_points_in_coreset = len(coreset_indices)
        x = jnp.asarray(x)
        kernel_mean_row_sum = jnp.asarray(kernel_mean_row_sum)
        return (
            kernel.compute(x[coreset_indices], x[i]).sum()
            - kernel.compute(x, x[coreset_indices]).sum(axis=1)
            + kernel.compute(x, x[i])[:, 0]
            - kernel_gram_matrix_diagonal
        ) / (num_points_in_coreset**2) - (
            kernel_mean_row_sum[i] - kernel_mean_row_sum
        ) / num_points_in_coreset


class RefineRandom(Refine):
    r"""
    Define the RefineRandom class.

    The refinement procedure replaces a random element with the best point among a set
    of candidate points. The candidate points are a random sample of :math:`n \times p`
    points from among the original data.

    :param approximate_kernel_row_sum: Boolean determining how the kernel mean row
        sum is calculated. If ``True``, the sum is approximate.
    :param approximator: :class:'~coreax.approximation.KernelMeanApproximator` object
        for the kernel mean approximation method
    :param p: Proportion of original dataset to randomly sample for candidate points
        to replace those in the coreset
    :param random_key: Pseudo-random number generator key
    """

    def __init__(
        self,
        approximate_kernel_row_sum: bool = False,
        approximator: KernelMeanApproximator = None,
        p: float = 0.1,
        random_key: int = 0,
    ):
        """Initialise a random refinement object."""
        self.random_key = random_key
        self.p = p
        super().__init__(
            approximate_kernel_row_sum=approximate_kernel_row_sum,
            approximator=approximator,
        )

    def refine(self, data_reduction: DataReduction) -> None:
        r"""
        Refine a coreset iteratively.

        The DataReduction object is updated in-place. The refinement procedure replaces
        a random element with the best point among a set of candidate points. The
        candidate points are a random sample of :math:`n \times p` points from among
        the original data.

        :param data_reduction: :class:`~coreax.reduction.DataReduction` object with
            :math:`n \times d` original data, :math:`m` coreset point indices, coreset
            and kernel object
        :return: Nothing
        """
        x = data_reduction.original_data
        coreset_indices = data_reduction.reduction_indices

        kernel_gram_matrix_diagonal = vmap(data_reduction.kernel.compute)(x, x)

        if self.approximate_kernel_row_sum:
            kernel_mean_row_sum = (
                data_reduction.kernel.approximate_kernel_matrix_row_sum_mean(
                    x, self.approximator
                )
            )
        else:
            kernel_mean_row_sum = (
                data_reduction.kernel.calculate_kernel_matrix_row_sum_mean(x)
            )

        coreset_indices = jnp.asarray(coreset_indices)
        x = jnp.asarray(x)
        num_points_in_coreset = len(coreset_indices)
        num_points_in_x = len(x)
        n_cand = int(num_points_in_x * self.p)
        n_iter = num_points_in_coreset * (num_points_in_x // n_cand)

        key = random.PRNGKey(self.random_key)

        body = partial(
            self._refine_rand_body,
            x=x,
            n_cand=n_cand,
            kernel=data_reduction.kernel,
            kernel_mean_row_sum=kernel_mean_row_sum,
            kernel_gram_matrix_diagonal=kernel_gram_matrix_diagonal,
        )
        key, coreset_indices = lax.fori_loop(0, n_iter, body, (key, coreset_indices))

        data_reduction.reduction_indices = coreset_indices
        data_reduction.reduced_data = data_reduction.original_data[coreset_indices, :]

    def _refine_rand_body(
        self,
        i: int,
        val: tuple[random.PRNGKeyArray, ArrayLike],
        x: ArrayLike,
        n_cand: int,
        kernel: ck.Kernel,
        kernel_mean_row_sum: ArrayLike,
        kernel_gram_matrix_diagonal: ArrayLike,
    ) -> tuple[random.PRNGKeyArray, Array]:
        r"""
        Execute main loop of the random refine method.

        :param i: Loop counter
        :param val: Loop updatable-variables
        :param x: Original :math:`n \times d` dataset
        :param n_cand: Number of candidates for comparison
        :param kernel_mean_row_sum: Mean vector over rows for the Gram matrix,
            a :math:`1 \times n` array
        :param kernel_gram_matrix_diagonal: Gram matrix diagonal,
            a :math:`1 \times n` array
        :return: Updated loop variables ``coreset_indices``
        """
        key, coreset_indices = val
        coreset_indices = jnp.asarray(coreset_indices)
        key, subkey = random.split(key)
        i = random.randint(subkey, (1,), 0, len(coreset_indices))[0]
        key, subkey = random.split(key)
        candidate_indices = random.randint(subkey, (n_cand,), 0, len(x))
        comparisons = self._comparison_cand(
            coreset_indices[i],
            candidate_indices,
            coreset_indices,
            x=x,
            kernel=kernel,
            kernel_mean_row_sum=kernel_mean_row_sum,
            kernel_gram_matrix_diagonal=kernel_gram_matrix_diagonal,
        )
        coreset_indices = lax.cond(
            jnp.any(comparisons > 0),
            self._change,
            self._nochange,
            i,
            coreset_indices,
            candidate_indices,
            comparisons,
        )

        return key, coreset_indices

    @jit
    def _comparison_cand(
        self,
        i: ArrayLike,
        candidate_indices: ArrayLike,
        coreset_indices: ArrayLike,
        x: ArrayLike,
        kernel: ck.Kernel,
        kernel_mean_row_sum: ArrayLike,
        kernel_gram_matrix_diagonal: ArrayLike,
    ) -> Array:
        r"""
        Calculate the change in maximum mean discrepancy (MMD).

        The change in MMD arises from replacing ``i`` in ``coreset_indices`` with ``x``.

        :param i: A coreset index
        :param candidate_indices: Indices for randomly sampled candidate points among
            the original data
        :param coreset_indices: Coreset point indices
        :param x: :math:`n \times d` original data
        :param kernel_mean_row_sum: :math:`1 \times n` row mean of the
            :math:`n \times n` kernel matrix
        :param kernel_gram_matrix_diagonal: Gram matrix diagonal,
            a :math:`1 \times n` array
        :return: The MMD changes for each candidate point
        """
        coreset_indices = jnp.asarray(coreset_indices)
        x = jnp.asarray(x)
        kernel_mean_row_sum = jnp.asarray(kernel_mean_row_sum)
        kernel_gram_matrix_diagonal = jnp.asarray(kernel_gram_matrix_diagonal)
        num_points_in_coreset = len(coreset_indices)

        return (
            kernel.compute(x[coreset_indices], x[i]).sum()
            - kernel.compute(x[candidate_indices, :], x[coreset_indices]).sum(axis=1)
            + kernel.compute(x[candidate_indices, :], x[i])[:, 0]
            - kernel_gram_matrix_diagonal[candidate_indices]
        ) / (num_points_in_coreset**2) - (
            kernel_mean_row_sum[i] - kernel_mean_row_sum[candidate_indices]
        ) / num_points_in_coreset

    @jit
    def _change(
        self,
        i: int,
        coreset_indices: ArrayLike,
        candidate_indices: ArrayLike,
        comparisons: ArrayLike,
    ) -> Array:
        r"""
        Replace the ``i``th point in ``coreset_indices``.

        The point is replaced with the candidate in ``candidate_indices`` with maximum
        value in ``comparisons``. ``coreset_indices`` -> ``x``.

        :param i: Index in ``coreset_indices`` to replace
        :param coreset_indices: Indices in the original dataset for replacement
        :param candidate_indices: A set of candidates for replacement
        :param comparisons: Comparison values for each candidate
        :return: Updated ``coreset_indices``, with ``i``th point replaced
        """
        coreset_indices = jnp.asarray(coreset_indices)
        candidate_indices = jnp.asarray(candidate_indices)
        return coreset_indices.at[i].set(candidate_indices[comparisons.argmax()])

    @jit
    def _nochange(
        self,
        i: int,
        coreset_indices: ArrayLike,
        candidate_indices: ArrayLike,
        comparisons: ArrayLike,
    ) -> Array:
        r"""
        Leave coreset indices unchanged (compare with :meth:`_change`).

        ``coreset_indices`` -> ``x``.

        :param i: Index in coreset_indices to replace. Not used.
        :param coreset_indices: The dataset for replacement. Will remain unchanged.
        :param candidate_indices: A set of candidates for replacement. Not used.
        :param comparisons: Comparison values for each candidate. Not used.
        :return: The original ``coreset_indices``, unchanged
        """
        return jnp.asarray(coreset_indices)


class RefineReverse(Refine):
    """
    Define the RefineRev (refine reverse) object.

    This performs the same style of refinement as :class:'~coreax.refine.RefineRegular'
    but reverses the order.
    """

    def refine(self, data_reduction: DataReduction) -> None:
        r"""
        Refine a coreset iteratively, replacing points which yield the most improvement.

        The DataReduction object is updated in-place. In this greedy refine method, the
        iteration is carried out over points in ``x``. ``x`` -> ``coreset_indices``.

        :param data_reduction: :class:`~coreax.reduction.DataReduction` object with
            :math:`n \times d` original data, :math:`m` coreset point indices, coreset
            and kernel object
        :return: Nothing
        """
        x = jnp.asarray(data_reduction.original_data)
        coreset_indices = jnp.asarray(data_reduction.reduction_indices)

        kernel_gram_matrix_diagonal = vmap(data_reduction.kernel.compute)(x, x)

        if self.approximate_kernel_row_sum:
            kernel_mean_row_sum = (
                data_reduction.kernel.approximate_kernel_matrix_row_sum_mean(
                    x, self.approximator
                )
            )
        else:
            kernel_mean_row_sum = (
                data_reduction.kernel.calculate_kernel_matrix_row_sum_mean(x)
            )

        num_points_in_x = len(x)

        body = partial(
            self._refine_rev_body,
            x=x,
            kernel=data_reduction.kernel,
            kernel_mean_row_sum=kernel_mean_row_sum,
            kernel_gram_matrix_diagonal=kernel_gram_matrix_diagonal,
        )
        coreset_indices = lax.fori_loop(0, num_points_in_x, body, coreset_indices)

        data_reduction.reduction_indices = coreset_indices
        data_reduction.reduced_data = data_reduction.original_data[coreset_indices, :]

    def _refine_rev_body(
        self,
        i: int,
        coreset_indices: ArrayLike,
        x: ArrayLike,
        kernel: ck.Kernel,
        kernel_mean_row_sum: ArrayLike,
        kernel_gram_matrix_diagonal: ArrayLike,
    ) -> Array:
        r"""
        Execute main loop of the refine method, ``x`` -> ``coreset_indices``.

        :param i: Loop counter
        :param coreset_indices: Loop updatable-variables
        :param x: Original :math:`n \times d` dataset
        :param kernel_mean_row_sum: Mean vector over rows for the Gram matrix,
            a :math:`1 \times n` array
        :param kernel_gram_matrix_diagonal: Gram matrix diagonal,
            a :math:`1 \times n` array
        :return: Updated loop variables `coreset_indices`
        """
        comps = self._comparison_rev(
            i,
            coreset_indices,
            x,
            kernel,
            kernel_mean_row_sum,
            kernel_gram_matrix_diagonal,
        )
        coreset_indices = lax.cond(
            jnp.any(comps > 0),
            self._change_rev,
            self._nochange_rev,
            i,
            coreset_indices,
            comps,
        )

        return coreset_indices

    @jit
    def _comparison_rev(
        self,
        i: int,
        coreset_indices: ArrayLike,
        x: ArrayLike,
        kernel: ck.Kernel,
        kernel_mean_row_sum: ArrayLike,
        kernel_gram_matrix_diagonal: ArrayLike,
    ) -> Array:
        r"""
        Calculate the change in maximum mean discrepancy (MMD).

        ``x`` -> ``coreset_indices``. The change in MMD occurs by replacing a point in
        ``coreset_indices`` with ``x[i]``.

        :param i: Index for original data
        :param coreset_indices: Coreset point indices
        :param x: :math:`n \times d` original data
        :param kernel_mean_row_sum: :math:`1 \times n` row mean of the
            :math:`n \times n` kernel matrix
        :param kernel_gram_matrix_diagonal: Gram matrix diagonal,
            a :math:`1 \times n` array
        :return: The MMD changes for each point
        """
        coreset_indices = jnp.asarray(coreset_indices)
        x = jnp.asarray(x)
        kernel_mean_row_sum = jnp.asarray(kernel_mean_row_sum)
        kernel_gram_matrix_diagonal = jnp.asarray(kernel_gram_matrix_diagonal)
        num_points_in_coreset = len(coreset_indices)

        return (
            kernel.compute(x[coreset_indices], x[coreset_indices]).sum(axis=1)
            - kernel.compute(x[coreset_indices], x[i]).sum()
            + kernel.compute(x[coreset_indices], x[i])[:, 0]
            - kernel_gram_matrix_diagonal[coreset_indices]
        ) / (num_points_in_coreset**2) - (
            kernel_mean_row_sum[coreset_indices] - kernel_mean_row_sum[i]
        ) / num_points_in_coreset

    @jit
    def _change_rev(
        self, i: int, coreset_indices: ArrayLike, comparisons: ArrayLike
    ) -> Array:
        r"""
        Replace the maximum comparison value point in ``coreset_indices`` with ``i``.

        ``x`` -> ``coreset_indices``.

        :param i: Value to replace into ``coreset_indices``
        :param coreset_indices: The dataset for replacement
        :param comparisons: Comparison values for each candidate
        :return: Updated ``coreset_indices``, with maximum ``comparisons`` point
            replaced
        """
        coreset_indices = jnp.asarray(coreset_indices)
        comparisons = jnp.asarray(comparisons)
        j = comparisons.argmax()
        return coreset_indices.at[j].set(i)

    @jit
    def _nochange_rev(
        self, i: int, coreset_indices: ArrayLike, comparisons: ArrayLike
    ) -> Array:
        r"""
        Leave coreset indices unchanged (compare with ``refine.change_rev``).

        ``x`` -> ``coreset_indices``.

        :param i: Value to replace into ``coreset_indices``. Not used.
        :param coreset_indices: The dataset for replacement. Will remain unchanged.
        :param comparisons: Comparison values for each candidate. Not used.
        :return: The original ``coreset_indices``, unchanged
        """
        return jnp.asarray(coreset_indices)


# Define the pytree node for the added class to ensure methods with jit decorators
# are able to run. We rely on the naming convention that all child classes of
# ScoreMatching include the sub-string ScoreMatching inside of them.
refine_classes = (RefineRegular, RefineRandom, RefineReverse)
for current_class in refine_classes:
    tree_util.register_pytree_node(
        current_class, current_class._tree_flatten, current_class._tree_unflatten
    )

# Set up class factory
refine_factory = ClassFactory(Refine)
refine_factory.register("regular", RefineRegular)
refine_factory.register("random", RefineRandom)
refine_factory.register("reverse", RefineReverse)
