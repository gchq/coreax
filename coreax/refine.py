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
:meth:`~Refine.refine` that manipulates a :class:`~coreax.reduction.Coreset`
object.

The other mandatory method to implement is :meth:`~Refine._tree_flatten`. To improve
performance, refine computation is JIT compiled. As a result, definitions of dynamic
and static values inside :meth:`~Refine._tree_flatten` ensure the refine object can be
mutated and the corresponding JIT compilation does not yield unexpected results.
"""

# Support annotations with | in Python < 3.10
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array, jit, lax, random, tree_util, vmap
from jax.typing import ArrayLike

import coreax.approximation
import coreax.kernel
import coreax.util
import coreax.validation

if TYPE_CHECKING:
    import coreax.reduction


class Refine(ABC):
    r"""
    Base class for refinement functions.

    The refinement process happens iteratively. Coreset elements are replaced by
    points most reducing the maximum mean discrepancy (MMD). The MMD is defined by

    .. math::
        \text{MMD}^2(X,X_c) =
        \mathbb{E}(k(X,X)) + \mathbb{E}(k(X_c,X_c)) - 2\mathbb{E}(k(X,X_c))

    for a dataset ``X`` and corresponding coreset ``X_c``.

    :param approximator: :class:`~coreax.approximation.KernelMeanApproximator` object
        for the kernel mean approximation method or :data:`None` (default) if
        calculations should be exact
    """

    def __init__(
        self,
        approximator: coreax.approximation.KernelMeanApproximator | None = None,
    ):
        """Initialise a refinement object."""
        # Validate inputs
        self.approximator = approximator

    @abstractmethod
    def refine(
        self,
        coreset: coreax.reduction.Coreset,
    ) -> None:
        r"""
        Compute the refined coreset, of :math:`m` points in :math:`d` dimensions.

        The :class:`~coreax.reduction.Coreset` object is updated in-place. The
        refinement procedure replaces elements with points most reducing maximum mean
        discrepancy (MMD).

        :param coreset: :class:`~coreax.reduction.Coreset` object with
            :math:`n \times d` original data, :math:`m` coreset point indices, coreset
            and kernel object
        :return: Nothing
        """

    @staticmethod
    def _validate_coreset(coreset: coreax.reduction.Coreset) -> None:
        """
        Validate that refinement can be performed on this coreset.

        :param coreset: :class:`~coreax.reduction.Coreset` object to validate
        :raises TypeError: When called on a class that does not generate coresubsets
        :return: Nothing
        """
        # validate_fitted checks original_data
        coreset.validate_fitted("refine")
        if coreset.coreset_indices is None:
            raise TypeError("Cannot refine when not finding a coresubset")

    def tree_flatten(self) -> tuple[tuple, dict]:
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable JIT decoration
        of methods inside this class.

        :return: Tuple containing two elements. The first is a tuple holding the arrays
            and dynamic values that are present in the class. The second is a dictionary
            holding the static auxiliary data for the class, with keys being the names
            of class attributes, and values being the values of the corresponding class
            attributes.
        """
        # dynamic values:
        children = ()
        # static values:
        aux_data = {"approximator": self.approximator}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstructs a pytree from the tree definition and the leaves.

        Arrays & dynamic values (children) and auxiliary data (static values) are
        reconstructed. A method to reconstruct the pytree needs to be specified to
        enable JIT decoration of methods inside this class.
        """
        return cls(*children, **aux_data)


class RefineRegular(Refine):
    r"""
    Define the RefineRegular class.

    The refinement process happens iteratively. The iteration is carried out over
    points in ``X``. Coreset elements are replaced by points most reducing the maximum
    mean discrepancy (MMD). The MMD is defined by:

    .. math::
        \text{MMD}^2(X,X_c) =
        \mathbb{E}(k(X,X)) + \mathbb{E}(k(X_c,X_c)) - 2\mathbb{E}(k(X,X_c))

    for a dataset ``X`` and corresponding coreset ``X_c``.

    :param approximator: :class:`~coreax.approximation.KernelMeanApproximator` object
        for the kernel mean approximation method or :data:`None` (default) if
        calculations should be exact
    """

    def refine(
        self,
        coreset: coreax.reduction.Coreset,
    ) -> None:
        r"""
        Compute the refined coreset, of ``m`` points in ``d`` dimensions.

        The Coreset object is updated in-place. The refinement procedure replaces
        elements with points most reducing maximum mean discrepancy (MMD).

        :param coreset: :class:`~coreax.reduction.Coreset` object with
            :math:`n \times d` original data, :math:`m` coreset point indices, coreset
            and kernel object
        :return: Nothing
        """
        self._validate_coreset(coreset)
        original_array = coreset.original_data.pre_coreset_array
        coreset_indices = coreset.coreset_indices

        kernel_gram_matrix_diagonal = vmap(coreset.kernel.compute)(
            original_array, original_array
        )

        # If not already done on Coreset, calculate kernel_matrix_row_sum_mean
        kernel_matrix_row_sum_mean = coreset.kernel_matrix_row_sum_mean
        if kernel_matrix_row_sum_mean is None:
            if self.approximator is not None:
                kernel_matrix_row_sum_mean = (
                    coreset.kernel.approximate_kernel_matrix_row_sum_mean(
                        original_array, self.approximator
                    )
                )
            else:
                kernel_matrix_row_sum_mean = (
                    coreset.kernel.calculate_kernel_matrix_row_sum_mean(original_array)
                )

        coreset_indices = jnp.asarray(coreset_indices)
        num_points_in_coreset = len(coreset_indices)
        body = partial(
            self._refine_body,
            x=original_array,
            kernel=coreset.kernel,
            kernel_matrix_row_sum_mean=kernel_matrix_row_sum_mean,
            kernel_gram_matrix_diagonal=kernel_gram_matrix_diagonal,
        )
        coreset_indices = lax.fori_loop(0, num_points_in_coreset, body, coreset_indices)

        coreset.coreset_indices = coreset_indices
        coreset.coreset = original_array[coreset_indices, :]

    @jit
    def _refine_body(
        self,
        i: int,
        coreset_indices: ArrayLike,
        x: ArrayLike,
        kernel: coreax.kernel.Kernel,
        kernel_matrix_row_sum_mean: ArrayLike,
        kernel_gram_matrix_diagonal: ArrayLike,
    ) -> Array:
        r"""
        Execute main loop of the refine method, ``coreset_indices`` -> ``x``.

        :param i: Loop counter
        :param coreset_indices: Loop updatable-variables
        :param x: Original :math:`n \times d` dataset
        :param kernel: Kernel used for calculating the maximum
            mean discrepancy, for comparing candidate coresets during refinement
        :param kernel_matrix_row_sum_mean: Mean vector over rows for the Gram matrix,
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
                kernel_matrix_row_sum_mean=kernel_matrix_row_sum_mean,
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
        kernel: coreax.kernel.Kernel,
        kernel_matrix_row_sum_mean: ArrayLike,
        kernel_gram_matrix_diagonal: ArrayLike,
    ) -> Array:
        r"""
        Calculate the change in maximum mean discrepancy from point replacement.

        ``coreset_indices`` -> ``x``.

        The change calculated is from replacing point ``i`` in ``coreset_indices`` with
        any point in ``x``.

        :param i: Loop counter
        :param coreset_indices: Coreset point indices
        :param x: :math:`n \times d` original data
        :param kernel: Kernel used for calculating the maximum
            mean discrepancy, for comparing candidate coresets during refinement
        :param kernel_matrix_row_sum_mean: :math:`1 \times n` row mean of the
            :math:`n \times n` kernel matrix
        :param kernel_gram_matrix_diagonal: Gram matrix diagonal,
            a :math:`1 \times n` array
        :return: The MMD changes for each candidate point
        """
        coreset_indices = jnp.asarray(coreset_indices)
        num_points_in_coreset = len(coreset_indices)
        x = jnp.asarray(x)
        kernel_matrix_row_sum_mean = jnp.asarray(kernel_matrix_row_sum_mean)
        return (
            kernel.compute(x[coreset_indices], x[i]).sum()
            - kernel.compute(x, x[coreset_indices]).sum(axis=1)
            + kernel.compute(x, x[i])[:, 0]
            - kernel_gram_matrix_diagonal
        ) / (num_points_in_coreset**2) - (
            kernel_matrix_row_sum_mean[i] - kernel_matrix_row_sum_mean
        ) / num_points_in_coreset


class RefineRandom(Refine):
    r"""
    Define the RefineRandom class.

    The refinement procedure replaces a random element with the best point among a set
    of candidate points. The candidate points are a random sample of :math:`n \times p`
    points from among the original data.

    :param random_key: Pseudo-random number generator key
    :param approximator: :class:`~coreax.approximation.KernelMeanApproximator` object
        for the kernel mean approximation method or :data:`None` (default) if
        calculations should be exact
    :param p: Proportion of original dataset to randomly sample for candidate points
        to replace those in the coreset
    """

    def __init__(
        self,
        random_key: coreax.validation.KeyArrayLike,
        approximator: coreax.approximation.KernelMeanApproximator = None,
        p: float = 0.1,
    ):
        """Initialise a random refinement object."""
        # Perform input validation
        p = coreax.validation.cast_as_type(x=p, object_name="p", type_caster=float)
        coreax.validation.validate_in_range(
            x=p, object_name="p", strict_inequalities=True, lower_bound=0.0
        )
        coreax.validation.validate_in_range(
            x=p, object_name="p", strict_inequalities=False, upper_bound=1.0
        )
        coreax.validation.validate_key_array(x=random_key, object_name="random_key")

        # Assign attributes
        self.p = p
        self.random_key = random_key
        super().__init__(
            approximator=approximator,
        )

    def tree_flatten(self) -> tuple[tuple, dict]:
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable JIT decoration
        of methods inside this class.

        :return: Tuple containing two elements. The first is a tuple holding the arrays
            and dynamic values that are present in the class. The second is a dictionary
            holding the static auxiliary data for the class, with keys being the names
            of class attributes, and values being the values of the corresponding class
            attributes.
        """
        children = (self.random_key,)
        auxiliary_data = {"approximator": self.approximator, "p": self.p}
        return children, auxiliary_data

    def refine(
        self,
        coreset: coreax.reduction.Coreset,
    ) -> None:
        r"""
        Refine a coreset iteratively.

        The :class:`~coreax.reduction.Coreset` instance is updated in-place. The
        refinement procedure replaces a random element with the best point among a set
        of candidate points. The candidate points are a random sample of
        :math:`n \times p` points from among the original data.

        :param coreset: :class:`~coreax.reduction.Coreset` object with
            :math:`n \times d` original data, :math:`m` coreset point indices, coreset
            and kernel object
        :return: Nothing
        """
        self._validate_coreset(coreset)
        original_array = coreset.original_data.pre_coreset_array
        coreset_indices = coreset.coreset_indices

        kernel_gram_matrix_diagonal = vmap(coreset.kernel.compute)(
            original_array, original_array
        )

        # If not already done on Coreset, calculate kernel_matrix_row_sum_mean
        kernel_matrix_row_sum_mean = coreset.kernel_matrix_row_sum_mean
        if kernel_matrix_row_sum_mean is None:
            if self.approximator is not None:
                kernel_matrix_row_sum_mean = (
                    coreset.kernel.approximate_kernel_matrix_row_sum_mean(
                        original_array, self.approximator
                    )
                )
            else:
                kernel_matrix_row_sum_mean = (
                    coreset.kernel.calculate_kernel_matrix_row_sum_mean(original_array)
                )

        coreset_indices = jnp.asarray(coreset_indices)
        num_points_in_coreset = len(coreset_indices)
        num_points_in_x = len(original_array)
        n_cand = int(num_points_in_x * self.p)
        n_iter = num_points_in_coreset * (num_points_in_x // n_cand)

        body = partial(
            self._refine_rand_body,
            x=original_array,
            n_cand=n_cand,
            kernel=coreset.kernel,
            kernel_matrix_row_sum_mean=kernel_matrix_row_sum_mean,
            kernel_gram_matrix_diagonal=kernel_gram_matrix_diagonal,
        )
        _, coreset_indices = lax.fori_loop(
            0, n_iter, body, (self.random_key, coreset_indices)
        )

        coreset.coreset_indices = coreset_indices
        coreset.coreset = original_array[coreset_indices, :]

    def _refine_rand_body(
        self,
        _i: int,
        val: tuple[coreax.validation.KeyArrayLike, ArrayLike],
        x: ArrayLike,
        n_cand: int,
        kernel: coreax.kernel.Kernel,
        kernel_matrix_row_sum_mean: ArrayLike,
        kernel_gram_matrix_diagonal: ArrayLike,
    ) -> tuple[coreax.validation.KeyArray, Array]:
        r"""
        Execute main loop of the random refine method.

        :param _i: Loop counter. This parameter is unused. It is only required by
            :func:`~jax.lax.fori_loop` for executing the refinement ``n_iter`` times.
        :param val: Loop updatable-variables
        :param x: Original :math:`n \times d` dataset
        :param n_cand: Number of candidates for comparison
        :param kernel: Kernel used for calculating the maximum
            mean discrepancy, for comparing candidate coresets during refinement
        :param kernel_matrix_row_sum_mean: Mean vector over rows for the Gram matrix,
            a :math:`1 \times n` array
        :param kernel_gram_matrix_diagonal: Gram matrix diagonal,
            a :math:`1 \times n` array
        :return: Updated loop variables ``coreset_indices``
        """
        key, coreset_indices = val
        coreset_indices = jnp.asarray(coreset_indices)
        key, subkey = random.split(key)
        index_to_compare = random.randint(subkey, (1,), 0, len(coreset_indices))[0]
        key, subkey = random.split(key)
        candidate_indices = random.randint(subkey, (n_cand,), 0, len(x))
        comparisons = self._comparison_cand(
            coreset_indices[index_to_compare],
            candidate_indices,
            coreset_indices,
            x=x,
            kernel=kernel,
            kernel_matrix_row_sum_mean=kernel_matrix_row_sum_mean,
            kernel_gram_matrix_diagonal=kernel_gram_matrix_diagonal,
        )
        coreset_indices = lax.cond(
            jnp.any(comparisons > 0),
            self._change,
            self._no_change,
            index_to_compare,
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
        kernel: coreax.kernel.Kernel,
        kernel_matrix_row_sum_mean: ArrayLike,
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
        :param kernel: Kernel used for calculating the maximum
            mean discrepancy, for comparing candidate coresets during refinement
        :param kernel_matrix_row_sum_mean: :math:`1 \times n` row mean of the
            :math:`n \times n` kernel matrix
        :param kernel_gram_matrix_diagonal: Gram matrix diagonal,
            a :math:`1 \times n` array
        :return: The MMD changes for each candidate point
        """
        coreset_indices = jnp.asarray(coreset_indices)
        x = jnp.asarray(x)
        kernel_matrix_row_sum_mean = jnp.asarray(kernel_matrix_row_sum_mean)
        kernel_gram_matrix_diagonal = jnp.asarray(kernel_gram_matrix_diagonal)
        num_points_in_coreset = len(coreset_indices)

        return (
            kernel.compute(x[coreset_indices], x[i]).sum()
            - kernel.compute(x[candidate_indices, :], x[coreset_indices]).sum(axis=1)
            + kernel.compute(x[candidate_indices, :], x[i])[:, 0]
            - kernel_gram_matrix_diagonal[candidate_indices]
        ) / (num_points_in_coreset**2) - (
            kernel_matrix_row_sum_mean[i]
            - kernel_matrix_row_sum_mean[candidate_indices]
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
        Replace the ``i``\th point in ``coreset_indices``.

        The point is replaced with the candidate in ``candidate_indices`` with maximum
        value in ``comparisons``. ``coreset_indices`` -> ``x``.

        :param i: Index in ``coreset_indices`` to replace
        :param coreset_indices: Indices in the original dataset for replacement
        :param candidate_indices: A set of candidates for replacement
        :param comparisons: Comparison values for each candidate
        :return: Updated ``coreset_indices``, with ``i``\th point replaced
        """
        coreset_indices = jnp.asarray(coreset_indices)
        candidate_indices = jnp.asarray(candidate_indices)
        return coreset_indices.at[i].set(candidate_indices[comparisons.argmax()])

    @jit
    def _no_change(
        self,
        _i: int,
        coreset_indices: ArrayLike,
        _candidate_indices: ArrayLike,
        _comparisons: ArrayLike,
    ) -> Array:
        r"""
        Leave coreset indices unchanged (compare with :meth:`_change`).

        ``coreset_indices`` -> ``x``.

        .. note:: The signature of this method must match :meth:`_change` for use with
            :func:`jax.lax.cond`. Since no indices are swapped in this method, only
            ``coreset_indices`` is used.

        :param _i: Index in coreset_indices to replace. Not used.
        :param coreset_indices: The dataset for replacement. Will remain unchanged.
        :param _candidate_indices: A set of candidates for replacement. Not used.
        :param _comparisons: Comparison values for each candidate. Not used.
        :return: The original ``coreset_indices``, unchanged
        """
        return jnp.asarray(coreset_indices)


class RefineReverse(Refine):
    """
    Define the RefineReverse (refine reverse) object.

    This performs the same style of refinement as :class:`~coreax.refine.RefineRegular`
    but reverses the order.

    :param approximator: :class:`~coreax.approximation.KernelMeanApproximator` object
        for the kernel mean approximation method or :data:`None` (default) if
        calculations should be exact
    """

    def refine(
        self,
        coreset: coreax.reduction.Coreset,
    ) -> None:
        r"""
        Refine a coreset iteratively, replacing points which yield the most improvement.

        The :class:`~coreax.reduction.Coreset` instance is updated in-place. In this
        greedy refine method, the iteration is carried out over points in ``x``.
        ``x`` -> ``coreset_indices``.

        :param coreset: :class:`~coreax.reduction.Coreset` object with
            :math:`n \times d` original data, :math:`m` coreset point indices, coreset
            and kernel object
        :return: Nothing
        """
        self._validate_coreset(coreset)
        original_array = coreset.original_data.pre_coreset_array
        coreset_indices = coreset.coreset_indices

        kernel_gram_matrix_diagonal = vmap(coreset.kernel.compute)(
            original_array, original_array
        )

        # If not already done on Coreset, calculate kernel_matrix_row_sum_mean
        kernel_matrix_row_sum_mean = coreset.kernel_matrix_row_sum_mean
        if kernel_matrix_row_sum_mean is None:
            if self.approximator is not None:
                kernel_matrix_row_sum_mean = (
                    coreset.kernel.approximate_kernel_matrix_row_sum_mean(
                        original_array, self.approximator
                    )
                )
            else:
                kernel_matrix_row_sum_mean = (
                    coreset.kernel.calculate_kernel_matrix_row_sum_mean(original_array)
                )

        num_points_in_x = len(original_array)

        body = partial(
            self._refine_rev_body,
            x=original_array,
            kernel=coreset.kernel,
            kernel_matrix_row_sum_mean=kernel_matrix_row_sum_mean,
            kernel_gram_matrix_diagonal=kernel_gram_matrix_diagonal,
        )
        coreset_indices = lax.fori_loop(0, num_points_in_x, body, coreset_indices)

        coreset.coreset_indices = coreset_indices
        coreset.coreset = original_array[coreset_indices, :]

    def _refine_rev_body(
        self,
        i: int,
        coreset_indices: ArrayLike,
        x: ArrayLike,
        kernel: coreax.kernel.Kernel,
        kernel_matrix_row_sum_mean: ArrayLike,
        kernel_gram_matrix_diagonal: ArrayLike,
    ) -> Array:
        r"""
        Execute main loop of the refine method, ``x`` -> ``coreset_indices``.

        :param i: Loop counter
        :param coreset_indices: Loop updatable-variables
        :param x: Original :math:`n \times d` dataset
        :param kernel: Kernel used for calculating the maximum
            mean discrepancy, for comparing candidate coresets during refinement
        :param kernel_matrix_row_sum_mean: Mean vector over rows for the Gram matrix,
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
            kernel_matrix_row_sum_mean,
            kernel_gram_matrix_diagonal,
        )
        coreset_indices = lax.cond(
            jnp.any(comps > 0),
            self._change_rev,
            self._no_change_rev,
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
        kernel: coreax.kernel.Kernel,
        kernel_matrix_row_sum_mean: ArrayLike,
        kernel_gram_matrix_diagonal: ArrayLike,
    ) -> Array:
        r"""
        Calculate the change in maximum mean discrepancy (MMD).

        ``x`` -> ``coreset_indices``. The change in MMD occurs by replacing a point in
        ``coreset_indices`` with ``x[i]``.

        :param i: Index for original data
        :param coreset_indices: Coreset point indices
        :param x: :math:`n \times d` original data
        :param kernel: Kernel used for calculating the maximum
            mean discrepancy, for comparing candidate coresets during refinement
        :param kernel_matrix_row_sum_mean: :math:`1 \times n` row mean of the
            :math:`n \times n` kernel matrix
        :param kernel_gram_matrix_diagonal: Gram matrix diagonal,
            a :math:`1 \times n` array
        :return: The MMD changes for each point
        """
        coreset_indices = jnp.asarray(coreset_indices)
        x = jnp.asarray(x)
        kernel_matrix_row_sum_mean = jnp.asarray(kernel_matrix_row_sum_mean)
        kernel_gram_matrix_diagonal = jnp.asarray(kernel_gram_matrix_diagonal)
        num_points_in_coreset = len(coreset_indices)

        return (
            kernel.compute(x[coreset_indices], x[coreset_indices]).sum(axis=1)
            - kernel.compute(x[coreset_indices], x[i]).sum()
            + kernel.compute(x[coreset_indices], x[i])[:, 0]
            - kernel_gram_matrix_diagonal[coreset_indices]
        ) / (num_points_in_coreset**2) - (
            kernel_matrix_row_sum_mean[coreset_indices] - kernel_matrix_row_sum_mean[i]
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
    def _no_change_rev(
        self, _i: int, coreset_indices: ArrayLike, _comparisons: ArrayLike
    ) -> Array:
        r"""
        Leave coreset indices unchanged (compare with :meth:`_change_rev`).

        ``x`` -> ``coreset_indices``.

        .. note:: The signature of this method must match :meth:`_change_rev` for use
            with :func:`jax.lax.cond`. Since no indices are swapped in this method, only
            ``coreset_indices`` is used.

        :param _i: Value to replace into ``coreset_indices``. Not used.
        :param coreset_indices: The dataset for replacement. Will remain unchanged.
        :param _comparisons: Comparison values for each candidate. Not used.
        :return: The original ``coreset_indices``, unchanged
        """
        return jnp.asarray(coreset_indices)


# Define the pytree node for the added class to ensure methods with JIT decorators
# are able to run. This tuple must be updated when a new class object is defined.
refine_classes = (RefineRegular, RefineRandom, RefineReverse)
for current_class in refine_classes:
    tree_util.register_pytree_node(
        current_class, current_class.tree_flatten, current_class.tree_unflatten
    )
