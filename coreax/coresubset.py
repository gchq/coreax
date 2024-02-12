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
Classes and associated functionality to construct coresubsets.

Given a :math:`n \times d` dataset, one may wish to construct a compressed
:math:`m \times d` representation of this dataset, where :math:`m << n`. This
compressed representation is often referred to as a coreset. When the elements of a
coreset are required to be elements of the original dataset, we denote this a
coresubset. This module contains implementations of approaches to construct coresubsets.
Coresets and coresubset are a type of data reduction, and these inherit from
:class:`~coreax.reduction.Coreset`. The aim is to select a small set of indices
that represent the key features of a larger dataset.

The abstract base class is :class:`~coreax.reduction.Coreset`.
"""

# Support annotations with | in Python < 3.10
from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax, random
from jax.typing import ArrayLike

import coreax.approximation
import coreax.kernel
import coreax.reduction
import coreax.refine
import coreax.util
import coreax.validation
import coreax.weights


class KernelHerding(coreax.reduction.Coreset):
    r"""
    Apply kernel herding to a dataset.

    Kernel herding is a deterministic, iterative and greedy approach to determine this
    compressed representation.

    Given one has selected :math:`T` data points for their compressed representation of
    the original dataset, kernel herding selects the next point as:

    .. math::

        x_{T+1} = \arg\max_{x} \left( \mathbb{E}[k(x, x')] -
            \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) \right)

    where :math:`k` is the kernel used, the expectation :math:`\mathbb{E}` is taken over
    the entire dataset, and the search is over the entire dataset. This can informally
    be seen as a balance between using points at which the underlying density is high
    (the first term) and exploration of distinct regions of the space (the second term).

    This class works with all children of :class:`~coreax.kernel.Kernel`, including
    Stein kernels.

    :param random_key: Key for random number generation
    :param kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param weights_optimiser: :class:`~coreax.weights.WeightsOptimiser` object to
        determine weights for coreset points to optimise some quality metric, or
        :data:`None` (default) if unweighted
    :param block_size: Size of matrix blocks to process when computing the kernel
        matrix row sum mean. Larger blocks will require more memory in the system.
    :param unique: Boolean that enforces the resulting coreset will only contain
        unique elements
    :param refine_method: :class:`~coreax.refine.Refine` object to use, or :data:`None`
        (default) if no refinement is required
    :param approximator: :class:`~coreax.approximation.KernelMeanApproximator` object
        that has been created using the same kernel one wishes to use for herding. If
        :data:`None` (default) then calculation is exact, but can be computationally
        intensive.
    """

    def __init__(
        self,
        random_key: coreax.validation.KeyArrayLike,
        *,
        kernel: coreax.kernel.Kernel,
        weights_optimiser: coreax.weights.WeightsOptimiser | None = None,
        block_size: int = 10_000,
        unique: bool = True,
        refine_method: coreax.refine.Refine | None = None,
        approximator: coreax.approximation.KernelMeanApproximator | None = None,
    ):
        """Initialise a KernelHerding class."""
        # Validate inputs. Note that inputs passed to the parent are validated within
        # that class, however a valid kernel object must be passed for kernel
        # herding, so this is verified here.
        coreax.validation.validate_is_instance(
            x=kernel, object_name="kernel", expected_type=coreax.kernel.Kernel
        )
        block_size = coreax.validation.cast_as_type(
            x=block_size, object_name="block_size", type_caster=int
        )
        coreax.validation.validate_in_range(
            x=block_size,
            object_name="block_size",
            strict_inequalities=True,
            lower_bound=0,
        )
        unique = coreax.validation.cast_as_type(
            x=unique, object_name="unique", type_caster=bool
        )
        coreax.validation.validate_is_instance(
            x=approximator,
            object_name="approximator",
            expected_type=(coreax.approximation.KernelMeanApproximator, type(None)),
        )
        coreax.validation.validate_key_array(x=random_key, object_name="random_key")

        # Assign herding-specific attributes
        self.block_size = block_size
        self.unique = unique
        self.approximator = approximator
        self.random_key = random_key

        # Initialise parent
        super().__init__(
            weights_optimiser=weights_optimiser,
            kernel=kernel,
            refine_method=refine_method,
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
        children = (
            self.random_key,
            self.kernel,
            self.kernel_matrix_row_sum_mean,
            self.coreset_indices,
            self.coreset,
        )
        aux_data = {
            "block_size": self.block_size,
            "unique": self.unique,
            "refine_method": self.refine_method,
            "weights_optimiser": self.weights_optimiser,
            "approximator": self.approximator,
        }
        return children, aux_data

    def fit_to_size(self, coreset_size: int) -> None:
        r"""
        Execute kernel herding algorithm with Jax.

        We first compute the kernel matrix row sum mean (either exactly, or
        approximately) if it is not given, and then iterative add points to the coreset
        balancing  selecting points in high density regions with selecting points far
        from those already in the coreset.

        :param coreset_size: The size of the of coreset to generate
        :return: Nothing
        """
        # Validate inputs
        coreset_size = coreax.validation.cast_as_type(
            x=coreset_size, object_name="coreset_size", type_caster=int
        )
        coreax.validation.validate_in_range(
            x=coreset_size,
            object_name="coreset_size",
            strict_inequalities=True,
            lower_bound=0,
        )

        # Record the size of the original dataset
        num_data_points = len(self.original_data.pre_coreset_array)

        # If needed, compute the kernel matrix row sum mean - with or without an
        # approximator as specified by the inputs to this method
        if self.kernel_matrix_row_sum_mean is None:
            if self.approximator is not None:
                self.kernel_matrix_row_sum_mean = (
                    self.kernel.approximate_kernel_matrix_row_sum_mean(
                        x=self.original_data.pre_coreset_array,
                        approximator=self.approximator,
                    )
                )
            else:
                self.kernel_matrix_row_sum_mean = (
                    self.kernel.calculate_kernel_matrix_row_sum_mean(
                        x=self.original_data.pre_coreset_array, max_size=self.block_size
                    )
                )

        # Initialise variables that will be updated throughout the loop. These are
        # initially local variables, with the coreset indices being assigned to self
        # when the entire set is created
        kernel_similarity_penalty = jnp.zeros(num_data_points)
        coreset_indices = jnp.zeros(coreset_size, dtype=jnp.int32)

        # Greedily select coreset points
        body = partial(
            self._greedy_body,
            x=self.original_data.pre_coreset_array,
            kernel_vectorised=self.kernel.compute,
            kernel_matrix_row_sum_mean=self.kernel_matrix_row_sum_mean,
            unique=self.unique,
        )
        coreset_indices, kernel_similarity_penalty = lax.fori_loop(
            lower=0,
            upper=coreset_size,
            body_fun=body,
            init_val=(coreset_indices, kernel_similarity_penalty),
        )

        # Assign coreset indices & coreset to original data object
        self.coreset_indices = coreset_indices
        self.coreset = self.original_data.pre_coreset_array[self.coreset_indices, :]

    @staticmethod
    @partial(jit, static_argnames=["kernel_vectorised", "unique"])
    def _greedy_body(
        i: int,
        val: tuple[ArrayLike, ArrayLike],
        x: ArrayLike,
        kernel_vectorised: coreax.util.KernelComputeType,
        kernel_matrix_row_sum_mean: ArrayLike,
        unique: bool,
    ) -> tuple[Array, Array]:
        r"""
        Execute main loop of greedy kernel herding.

        This function carries out one iteration of kernel herding. Recall that kernel
        herding is defined as

        .. math::

            x_{T+1} = \arg\max_{x} \left( \mathbb{E}[k(x, x')] -
                \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) \right)

        where :math:`k` is the kernel used, the expectation :math:`\mathbb{E}` is taken
        over the entire dataset, and the search is over the entire dataset.

        The kernel matrix row sum mean, :math:`\mathbb{E}[k(x, x')]` does not change
        across iterations. Each iteration, the set of coreset points updates with the
        newly selected point :math:`x_{T+1}`. Additionally, the penalties one applies,
        :math:`k(x, x_t)` update as these coreset points are updated.

        :param i: Loop counter, counting how many points are in the coreset on call of
            this method
        :param val: Tuple containing a :math:`1 \times m` array with the current coreset
            indices in and a :math:`1 \times n` array holding current kernel similarity
            penalties. Note that the array holding current coreset indices should always
            be the same length (however many coreset points are desired). The ``i``-th
            element of this gets updated to the index of the selected coreset point in
            iteration ``i``.
        :param x: :math:`n \times d` data matrix
        :param kernel_vectorised: Vectorised kernel computation function. This should be
            the :meth:`compute` method of a :class:`~coreax.kernel.Kernel` object,
        :param kernel_matrix_row_sum_mean: A :math:`1 \times n` array holding the mean
            over rows for the kernel Gram matrix
        :param unique: Flag for enforcing unique elements
        :returns: Updated loop variables ``current_coreset_indices`` and
            ``current_kernel_similarity_penalty``
        """
        # Unpack the components of the loop variables
        current_coreset_indices, current_kernel_similarity_penalty = val

        # Format inputs - note that the calls in jax for loops already validate the
        # ``i`` variable before calling.
        current_coreset_indices = coreax.validation.cast_as_type(
            x=current_coreset_indices,
            object_name="current_coreset_indices",
            type_caster=jnp.asarray,
        )
        current_kernel_similarity_penalty = coreax.validation.cast_as_type(
            x=current_kernel_similarity_penalty,
            object_name="current_kernel_similarity_penalty",
            type_caster=jnp.asarray,
        )
        x = coreax.validation.cast_as_type(
            x=x, object_name="x", type_caster=jnp.atleast_2d
        )
        coreax.validation.validate_is_instance(
            x=kernel_vectorised, object_name="kernel_vectorised", expected_type=Callable
        )
        kernel_matrix_row_sum_mean = coreax.validation.cast_as_type(
            x=kernel_matrix_row_sum_mean,
            object_name="kernel_matrix_row_sum_mean",
            type_caster=jnp.asarray,
        )
        unique = coreax.validation.cast_as_type(
            x=unique, object_name="unique", type_caster=bool
        )

        # Evaluate the kernel herding formula at this iteration - that is, select which
        # point in the data-set, when added to the coreset, will minimise maximum mean
        # discrepancy. This is essentially a balance between a point lying in a high
        # density region (the corresponding entry in kernel_matrix_row_sum_mean is
        # large) whilst being far away from points already in the coreset
        # (current_kernel_similarity_penalty being small).
        index_to_include_in_coreset = (
            kernel_matrix_row_sum_mean - current_kernel_similarity_penalty / (i + 1)
        ).argmax()

        # Update all the penalties we apply, because we now have additional points
        # in the coreset
        penalty_update = kernel_vectorised(
            x, jnp.atleast_2d(x[index_to_include_in_coreset])
        )[:, 0]
        current_kernel_similarity_penalty += penalty_update

        # Update the coreset indices to include the selected point
        current_coreset_indices = current_coreset_indices.at[i].set(
            index_to_include_in_coreset
        )

        # If we wish to force all points in a coreset to be unique, set the penalty term
        # to infinite at the newly selected point, so it can't be selected again
        if unique:
            current_kernel_similarity_penalty = current_kernel_similarity_penalty.at[
                index_to_include_in_coreset
            ].set(jnp.inf)

        return current_coreset_indices, current_kernel_similarity_penalty


class RandomSample(coreax.reduction.Coreset):
    r"""
    Reduce a dataset by uniformly randomly sampling a fixed number of points.

    :param random_key: Pseudo-random number generator key for sampling
    :param kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`, or
        :data:`None` if not applicable. Note that if this is supplied, it is only used
        for refinement, not during creation of the initial coreset.
    :param weights_optimiser: :class:`~coreax.weights.WeightsOptimiser` object to
        determine weights for coreset points to optimise some quality metric, or
        :data:`None` (default) if unweighted
    :param unique: If :data:`True`, this flag enforces unique elements, i.e. sampling
        without replacement
    :param refine_method: :class:`~coreax.refine.Refine` object to use, or :data:`None`
        (default) if no refinement is required
    """

    def __init__(
        self,
        random_key: coreax.validation.KeyArrayLike,
        *,
        kernel: coreax.kernel.Kernel | None = None,
        weights_optimiser: coreax.weights.WeightsOptimiser | None = None,
        unique: bool = True,
        refine_method: coreax.refine.Refine | None = None,
    ):
        """Initialise a random sampling object."""
        # Validate inputs
        coreax.validation.validate_key_array(x=random_key, object_name="random_key")
        unique = coreax.validation.cast_as_type(
            x=unique, object_name="unique", type_caster=bool
        )

        # Assign random sample specific attributes
        self.random_key = random_key
        self.unique = unique

        # Initialise Coreset parent
        super().__init__(
            weights_optimiser=weights_optimiser,
            kernel=kernel,
            refine_method=refine_method,
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
        children = (
            self.random_key,
            self.kernel,
            self.kernel_matrix_row_sum_mean,
            self.coreset_indices,
            self.coreset,
        )
        aux_data = {
            "weights_optimiser": self.weights_optimiser,
            "refine_method": self.refine_method,
            "unique": self.unique,
        }
        return children, aux_data

    def fit_to_size(self, coreset_size: int) -> None:
        """
        Reduce a dataset by uniformly randomly sampling a fixed number of points.

        This class is updated in-place. The randomly sampled points are stored in the
        ``reduction_indices`` attribute.

        :param coreset_size: The size of the of coreset to generate
        """
        # Validate inputs
        coreset_size = coreax.validation.cast_as_type(
            x=coreset_size, object_name="coreset_size", type_caster=int
        )
        coreax.validation.validate_in_range(
            x=coreset_size,
            object_name="coreset_size",
            strict_inequalities=True,
            lower_bound=0,
        )

        # Setup for sampling
        num_data_points = len(self.original_data.pre_coreset_array)

        # Randomly sample the desired number of points to form a coreset
        random_indices = random.choice(
            self.random_key,
            a=jnp.arange(0, num_data_points),
            shape=(coreset_size,),
            replace=not self.unique,
        )

        # Assign coreset indices and coreset to the object
        self.coreset_indices = random_indices
        self.coreset = self.original_data.pre_coreset_array[random_indices]
