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

from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax, random, vmap
from jax.typing import ArrayLike

import coreax.approximation
import coreax.kernel
import coreax.reduction
import coreax.refine
import coreax.util
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
        random_key: coreax.util.KeyArrayLike,
        *,
        kernel: coreax.kernel.Kernel,
        weights_optimiser: coreax.weights.WeightsOptimiser | None = None,
        block_size: int = 10_000,
        unique: bool = True,
        refine_method: coreax.refine.Refine | None = None,
        approximator: coreax.approximation.KernelMeanApproximator | None = None,
    ):
        """Initialise a KernelHerding class."""
        # Assign herding-specific attributes
        self.block_size = block_size
        self.unique = unique
        self.approximator = approximator
        self.random_key = random_key
        self.kernel_matrix_row_sum_mean = None

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
        approximately) if it is not given, and then iteratively add points to the
        coreset balancing selecting points in high density regions with selecting points
        far from those already in the coreset.

        :param coreset_size: The size of the of coreset to generate
        """
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
        try:
            # Note that a TypeError is raised if the size input to jnp.zeros is negative
            coreset_indices = jnp.zeros(coreset_size, dtype=jnp.int32)
        except TypeError as exception:
            if coreset_size < 0:
                raise ValueError("coreset_size must not be negative") from exception
            if isinstance(coreset_size, float):
                raise ValueError(
                    "coreset_size must be a positive integer"
                ) from exception
            raise

        # Greedily select coreset points
        body = partial(
            self._greedy_body,
            x=self.original_data.pre_coreset_array,
            kernel_vectorised=self.kernel.compute,
            kernel_matrix_row_sum_mean=self.kernel_matrix_row_sum_mean,
            unique=self.unique,
        )
        try:
            coreset_indices, kernel_similarity_penalty = lax.fori_loop(
                lower=0,
                upper=coreset_size,
                body_fun=body,
                init_val=(coreset_indices, kernel_similarity_penalty),
            )
        except IndexError as exception:
            if coreset_size == 0:
                raise ValueError("coreset_size must be non-zero") from exception
            raise

        # Assign coreset indices & coreset to original data object
        # pylint: disable=attribute-defined-outside-init
        self.coreset_indices = coreset_indices
        self.coreset = self.original_data.pre_coreset_array[self.coreset_indices, :]
        # pylint: enable=attribute-defined-outside-init

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
            the :meth:`~coreax.kernel.Kernel.compute` method of a
            :class:`~coreax.kernel.Kernel` object
        :param kernel_matrix_row_sum_mean: A :math:`1 \times n` array holding the mean
            over rows for the kernel Gram matrix
        :param unique: Flag for enforcing unique elements
        :returns: Updated loop variables ``current_coreset_indices`` and
            ``current_kernel_similarity_penalty``
        """
        # Unpack the components of the loop variables
        current_coreset_indices, current_kernel_similarity_penalty = val
        x = jnp.atleast_2d(x)
        current_coreset_indices = jnp.asarray(current_coreset_indices)
        current_kernel_similarity_penalty = jnp.asarray(
            current_kernel_similarity_penalty
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

    .. note::
        Any value other than :data:`True` will lead to random sampling with replacement
        of points from the original data to construct the coreset.

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
        random_key: coreax.util.KeyArrayLike,
        *,
        kernel: coreax.kernel.Kernel | None = None,
        weights_optimiser: coreax.weights.WeightsOptimiser | None = None,
        unique: bool = True,
        refine_method: coreax.refine.Refine | None = None,
    ):
        """Initialise a random sampling object."""
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

        Define arrays and dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable JIT decoration of
        methods inside this class.

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
        # Setup for sampling
        num_data_points = len(self.original_data.pre_coreset_array)

        # Randomly sample the desired number of points to form a coreset
        try:
            # Note that a TypeError is raised if the size input to random.choice is
            # negative, and an AttributeError is raised if the shape is a float
            random_indices = random.choice(
                self.random_key,
                a=jnp.arange(0, num_data_points),
                shape=(coreset_size,),
                replace=not self.unique,
            )
        except AttributeError as exception:
            if not isinstance(coreset_size, int):
                raise ValueError(
                    "coreset_size must be a positive integer"
                ) from exception
            raise
        except TypeError as exception:
            if coreset_size < 0:
                raise ValueError(
                    "coreset_size must be a positive integer"
                ) from exception
            raise

        # Assign coreset indices and coreset to the object
        # pylint: disable=attribute-defined-outside-init
        self.coreset_indices = random_indices
        self.coreset = self.original_data.pre_coreset_array[random_indices, :]
        # pylint: enable=attribute-defined-outside-init


class RPCholesky(coreax.reduction.Coreset):
    r"""
    Apply Randomly Pivoted Cholesky (RPCholesky) to a dataset.

    RPCholesky is a stochastic, iterative and greedy approach to determine this
    compressed representation.

    Given a dataset :math:`X` with :math:`N` data-points, and a desired coreset size of
    :math:`M`, RPCholesky determines which points to select to constitute this coreset.

    This is done by first computing the kernel Gram matrix of the original data, and
    isolating the diagonal of this. A 'pivot point' is then sampled, where sampling
    probabilities correspond to the size of the elements on this diagonal. The
    data-point corresponding to this pivot point is added to the coreset, and the
    diagonal of the Gram matrix is updated to add a repulsion term of sorts -
    encouraging the coreset to select a range of distinct points in the original data.
    The pivot sampling and diagonal updating steps are repeated until :math:`M` points
    have been selected.

    :param random_key: Key for random number generation
    :param kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param weights_optimiser: :class:`~coreax.weights.WeightsOptimiser` object to
        determine weights for coreset points to optimise some quality metric, or
        :data:`None` (default) if unweighted
    :param block_size: Size of matrix blocks to process when computing the kernel matrix
        row sum mean. Larger blocks will require more memory in the system.
    :param unique: Boolean that enforces the resulting coreset will only contain unique
        elements
    :param refine_method: :class:`~coreax.refine.Refine` object to use, or :data:`None`
        (default) if no refinement is required
    """

    def __init__(
        self,
        random_key: coreax.util.KeyArrayLike,
        *,
        kernel: coreax.kernel.Kernel,
        weights_optimiser: coreax.weights.WeightsOptimiser | None = None,
        block_size: int = 10_000,
        unique: bool = True,
        refine_method: coreax.refine.Refine | None = None,
    ):
        """Initialise a RPCholesky class."""
        # Assign specific attributes
        self.block_size = block_size
        self.unique = unique
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

        Define arrays and dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable JIT decoration of
        methods inside this class.

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
        }
        return children, aux_data

    def fit_to_size(self, coreset_size: int) -> None:
        r"""
        Execute RPCholesky algorithm with Jax.

        Computes a low-rank approximation of the Gram matrix.

        :param coreset_size: The size of the of coreset to generate
        """
        try:
            # Note that a TypeError is raised if the size input to jnp.zeros is negative
            coreset_indices = jnp.zeros(coreset_size, dtype=jnp.int32)
        except TypeError as exception:
            if coreset_size < 0:
                raise ValueError("coreset_size must not be negative") from exception
            if isinstance(coreset_size, float):
                raise ValueError(
                    "coreset_size must be a positive integer"
                ) from exception
            raise

        body = partial(
            self._loop_body,
            x=self.original_data.pre_coreset_array,
            kernel_vectorised=self.kernel.compute,
            unique=self.unique,
        )

        try:
            num_data_points = len(self.original_data.pre_coreset_array)
            approximation_matrix = jnp.zeros((num_data_points, coreset_size))
            _, key = random.split(self.random_key)

            # Evaluate the diagonal of the Gram matrix
            residual_diagonal = vmap(
                self.kernel.compute_elementwise, in_axes=(0, 0), out_axes=0
            )(
                self.original_data.pre_coreset_array,
                self.original_data.pre_coreset_array,
            )

            residual_diagonal, approximation_matrix, coreset_indices, key = (
                lax.fori_loop(
                    0,
                    coreset_size,
                    body,
                    (residual_diagonal, approximation_matrix, coreset_indices, key),
                )
            )

        except IndexError as exception:
            if coreset_size == 0:
                raise ValueError("coreset_size must be non-zero") from exception
            raise

        # pylint: disable=attribute-defined-outside-init
        self.coreset_indices = coreset_indices
        self.coreset = self.original_data.pre_coreset_array[self.coreset_indices, :]
        # pylint: enable=attribute-defined-outside-init

    @staticmethod
    @partial(jit, static_argnames=["kernel_vectorised", "unique"])
    def _loop_body(
        i: int,
        val: tuple[ArrayLike, ArrayLike, ArrayLike, coreax.util.KeyArrayLike],
        x: ArrayLike,
        kernel_vectorised: coreax.util.KernelComputeType,
        unique: bool,
    ) -> tuple[Array, Array, Array, Array]:
        r"""
        Execute main loop of RPCholesky.

        This function carries out one iteration of RPCholesky, defined in Algorithm 1 of
        :cite:`chen2023randomly`.

        :param i: Loop counter, counting how many points are in the coreset on call of
            this method
        :param val: Tuple containing a :math:`m \times 1` array of the residual
            diagonal, :math:`n \times m` Cholesky matrix F, :math:`1 \times m` array of
            current coreset indices and a PRNGKey for sampling. Note that the array
            holding current coreset indices should always be the same length (however
            many coreset points are desired). The ``i``-th element of this gets updated
            to the index of the selected coreset point in iteration ``i``.
        :param x: :math:`n \times d` data matrix
        :param kernel_vectorised: Vectorised kernel computation function. This should be
            the :meth:`~coreax.kernel.Kernel.compute` method of a
            :class:`~coreax.kernel.Kernel` object
        :param unique: Flag for enforcing unique elements
        :returns: Updated loop variables ``residual_diagonal``, ``F``,
            ``current_coreset_indices`` and ``key``
        """
        # Unpack the components of the loop variables
        residual_diagonal, approximation_matrix, current_coreset_indices, key = val
        key, subkey = random.split(key)
        num_data_points = len(x)

        # Sample a new index with probability proportional to residual diagonal
        selected_pivot_point = random.choice(
            subkey, num_data_points, (1,), p=residual_diagonal, replace=False
        )[0]
        current_coreset_indices = current_coreset_indices.at[i].set(
            selected_pivot_point
        )

        # Remove overlap with previously chosen columns
        g = (
            kernel_vectorised(x, x[selected_pivot_point])
            - jnp.dot(approximation_matrix, approximation_matrix[selected_pivot_point])[
                :, None
            ]
        )

        # Update approximation
        approximation_matrix = approximation_matrix.at[:, i].set(
            (g / jnp.sqrt(g[selected_pivot_point])).flatten()
        )

        # Track diagonal of residual matrix
        residual_diagonal = residual_diagonal - jnp.square(approximation_matrix[:, i])

        # Ensure diagonal remains nonnegative
        residual_diagonal = residual_diagonal.clip(min=0)
        if unique:
            # ensures that index selected_pivot_point can't be drawn again in future
            residual_diagonal = residual_diagonal.at[selected_pivot_point].set(0.0)
        return residual_diagonal, approximation_matrix, current_coreset_indices, key


class GreedyCMMD(coreax.reduction.Coreset):
    r"""
    Apply GreedyCMMD to a supervised dataset.

    GreedyCMMD is a deterministic, iterative and greedy approach to determine this
    compressed representation.

    Given one has an original dataset :math:`\mathcal{D}^{(1)} = \{(x_i, y_i)\}_{i=1}^n`
    of ``n`` pairs with :math:`x\in\mathbb{R}^d` and :math:`y\in\mathbb{R}^p`, and one
    has selected :math:`m` data pairs :math:`\mathcal{D}^{(2)} = \{(\tilde{x}_i,
    \tilde{y}_i)\}_{i=1}^m` already for their compressed representation of the original
    dataset, GreedyCMMD selects the next point to minimise the conditional maximum mean
    discrepancy (CMMD):

    .. math::

        \text{CMMD}^2(\mathcal{D}^{(1)}, \mathcal{D}^{(2)}) = ||\hat{\mu}^{(1)} -
        \hat{\mu}^{(2)}||^2_{\mathcal{H}_k \otimes \mathcal{H}_l}

    where :math:`\hat{\mu}^{(1)},\hat{\mu}^{(2)}` are the conditional mean embeddings
    estimated with :math:`\mathcal{D}^{(1)}` and :math:`\mathcal{D}^{(2)}` respectively,
    and :math:`\mathcal{H}_k,\mathcal{H}_l` are the RKHSs corresponding to the kernel
    functions :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` and
    :math:`l: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}` respectively.
    The search is performed over the entire dataset, or optionally over random batches
    at each iteration.

    This class works with all children of :class:`~coreax.kernel.Kernel`, including
    Stein kernels.

    :param random_key: Key for random number generation
    :param feature_kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
        on the feature space
    :param response_kernel: :class:`~coreax.kernel.Kernel` instance implementing a
        kernel function :math:`k: \mathbb{R}^p \times \mathbb{R}^p \rightarrow
        \mathbb{R}` on the response space
    :param num_feature_dimensions: An integer representing the dimensionality of the
        features :math:`x`
    :param regularisation_parameter: Regularisation parameter for stable inversion of
        feature gram matrix
    :param unique: Boolean that enforces the resulting coreset will only contain
        unique elements
    :param batch_size: An integer representing the size of the batches of data pairs
        sampled at each iteration for consideration for adding to the coreset
    :param refine_method: :class:`~coreax.refine.RefineCMMD` object, or :data:`None`
        (default) if no refinement is required
    """

    def __init__(
        self,
        random_key: coreax.util.KeyArrayLike,
        *,
        feature_kernel: coreax.kernel.Kernel,
        response_kernel: coreax.kernel.Kernel,
        num_feature_dimensions: int,
        regularisation_parameter: float = 1e-6,
        unique: bool = True,
        batch_size: int | None = None,
        refine_method: coreax.refine.RefineCMMD | None = None,
    ):
        # Assign GreedyCMMD-specific attributes
        self.random_key = random_key
        self.feature_kernel = feature_kernel
        self.response_kernel = response_kernel
        self.num_feature_dimensions = num_feature_dimensions
        self.regularisation_parameter = regularisation_parameter
        self.unique = unique
        self.batch_size = batch_size
        self.refine_method = refine_method
        self._feature_gramian = None
        self._response_gramian = None
        self._training_cme = None

        # CMMD coreset points cannot be weighted and the parent kernel is unused
        # as we need separate kernels for features and responses.
        super().__init__(
            weights_optimiser=None,
            kernel=None,
            refine_method=self.refine_method,
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
            self.feature_kernel,
            self.response_kernel,
            self.coreset_indices,
            self.coreset,
            self._feature_gramian,
            self._response_gramian,
            self._training_cme,
        )
        aux_data = {
            "num_feature_dimensions": self.num_feature_dimensions,
            "regularisation_parameter": self.regularisation_parameter,
            "unique": self.unique,
            "batch_size": self.batch_size,
            "refine_method": self.refine_method,
            "unused_kernel": self.kernel,
            "unused_weights_optimiser": self.weights_optimiser,
        }
        return children, aux_data

    def fit_to_size(self, coreset_size: int) -> None:
        r"""
        Execute greedy CMMD algorithm with Jax.

        We first compute the kernel matrices, and inverse feature kernel matrix
        and then iteratively add those points to the coreset that minimise the CMMD.

        :param coreset_size: The size of the of coreset to generate
        """
        # Compute original feature and response kernel gramians.
        x = self.original_data.pre_coreset_array[:, : self.num_feature_dimensions]
        y = self.original_data.pre_coreset_array[:, self.num_feature_dimensions :]
        num_data_pairs = self.original_data.pre_coreset_array.shape[0]

        feature_gramian = self.feature_kernel.compute(x, x)
        response_gramian = self.response_kernel.compute(y, y)

        # Invert the feature gramian
        inverse_feature_gramian = coreax.util.invert_regularised_array(
            array=feature_gramian,
            regularisation_parameter=self.regularisation_parameter,
            identity=jnp.eye(num_data_pairs),
        )

        # Evaluate conditional mean embedding (CME) at all possible pairs of the
        # available training data.
        training_cme = feature_gramian @ inverse_feature_gramian @ response_gramian

        # Pad the gramians and training CME evaluations with zeros in an additional
        # column and row to allow us to extract sub-arrays and fill in elements with
        # zeros simultaneously.
        feature_gramian = jnp.pad(feature_gramian, [(0, 1)], mode="constant")
        response_gramian = jnp.pad(response_gramian, [(0, 1)], mode="constant")
        training_cme = jnp.pad(training_cme, [(0, 1)], mode="constant")

        # Sample the indices to be considered at each iteration ahead of time.
        # If we are batching, each column will consist of a subset of indices up to the
        # dataset size with no repeats. If we are not batching then each column will
        # consist of a random permutation of indices up to the dataset size.
        if (self.batch_size is not None) and self.batch_size < num_data_pairs:
            batch_size = self.batch_size
        else:
            batch_size = num_data_pairs

        batch_indices = coreax.util.sample_batch_indices(
            random_key=self.random_key,
            data_size=num_data_pairs,
            batch_size=batch_size,
            num_batches=coreset_size,
        )

        # Initialise a rectangular array of size batch_size x coreset_size where each
        # entry consists of -1 apart from the first column which consists
        # of all the possible first coreset indices. This will allow us to extract
        # every possible "next" coreset kernel array ready for operations.
        all_possible_next_coreset_indices = jnp.hstack(
            (
                batch_indices[:, [0]],
                jnp.tile(-1, (batch_indices.shape[0], coreset_size - 1)),
            )
        )

        # Initialise coreset index variable that will be updated throughout the loop.
        # This is initially a local variable with the coreset indices being assigned to
        # self when the entire set is created.
        coreset_indices = -1 * jnp.ones(coreset_size, dtype=jnp.int32)

        # Initialise a zeros matrix that will eventually become a coreset_size x
        # coreset_size identity matrix as we iterate to the full coreset size.
        coreset_identity = jnp.zeros((coreset_size, coreset_size))

        # Greedily select coreset points
        body = partial(
            self._greedy_body,
            feature_gramian=feature_gramian,
            response_gramian=response_gramian,
            training_CME=training_cme,
            batch_indices=batch_indices,
            regularisation_parameter=self.regularisation_parameter,
            unique=self.unique,
        )
        (coreset_indices, _, _) = lax.fori_loop(
            lower=0,
            upper=coreset_size,
            body_fun=body,
            init_val=(
                coreset_indices,
                coreset_identity,
                all_possible_next_coreset_indices,
            ),
        )

        # If we are refining, update necessary class attributes
        if self.refine_method is not None:
            self._feature_gramian = feature_gramian
            self._response_gramian = response_gramian
            self._training_cme = training_cme

        # Assign coreset indices & coreset to original data object
        # pylint: disable=attribute-defined-outside-init
        self.coreset_indices = coreset_indices
        self.coreset = self.original_data.pre_coreset_array[self.coreset_indices, :]
        # pylint: enable=attribute-defined-outside-init

    @staticmethod
    @partial(jit, static_argnames=["regularisation_parameter", "unique"])
    def _greedy_body(
        i: int,
        val: tuple[ArrayLike, ArrayLike, ArrayLike],
        feature_gramian: ArrayLike,
        response_gramian: ArrayLike,
        training_cme: ArrayLike,
        batch_indices: ArrayLike,
        regularisation_parameter: float,
        unique: bool,
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        r"""
        Execute main loop of GreedyCMMD.

        :param i: Loop counter
        :param val: Loop updatable-variables
        :param feature_gramian: Gram matrix of training features
        :param response_gramian: Gram matrix of training responses
        :param training_CME: Evaluation of CME on the training data
        :param batch_indices: Array of sampled batch indices
        :param regularisation_parameter: Regularisation parameter for stable inversion
            of feature gram matrix
        :param unique: Boolean that enforces the resulting coreset will only contain
            unique elements
        :return: Updated loop variables
        """
        # Unpack the components of the loop variables
        (
            current_coreset_indices,
            current_identity,
            all_possible_next_coreset_indices,
        ) = val

        # Update the "identity" matrix
        current_identity = current_identity.at[i, i].set(1)

        # Extract all the possible "next" coreset arrays
        extract_indices = (
            all_possible_next_coreset_indices[:, :, None],
            all_possible_next_coreset_indices[:, None, :],
        )
        coreset_feature_gramians = feature_gramian[extract_indices]
        coreset_response_gramians = response_gramian[extract_indices]
        coreset_cmes = training_cme[extract_indices]

        # Compute and store inverses for each coreset feature kernel matrix
        inverse_coreset_feature_gramians = (
            coreax.util.invert_stacked_regularised_arrays(
                stacked_arrays=coreset_feature_gramians,
                regularisation_parameter=regularisation_parameter,
                identity=current_identity,
            )
        )

        # Compute each term of CMMD for each possible new coreset index
        term_2s = jnp.trace(
            inverse_coreset_feature_gramians
            @ coreset_response_gramians
            @ inverse_coreset_feature_gramians
            @ coreset_feature_gramians,
            axis1=1,
            axis2=2,
        )
        term_3s = jnp.trace(
            coreset_cmes @ inverse_coreset_feature_gramians, axis1=1, axis2=2
        )
        loss = term_2s - 2 * term_3s

        # Find the optimal next coreset index, ensuring we don't pick an already chosen
        # point if we want the indices to be unique.
        if unique:
            already_chosen_indices_mask = jnp.isin(
                all_possible_next_coreset_indices[:, i], current_coreset_indices
            )
            loss = loss + jnp.where(already_chosen_indices_mask, jnp.inf, 0)
        index_to_include_in_coreset = all_possible_next_coreset_indices[
            loss.argmin(), i
        ]

        # Repeat the chosen coreset index into the ith column of the array of possible
        # next coreset indices and replace the (i+1)th column with the next batch of
        # possible coreset indices.
        all_possible_next_coreset_indices = all_possible_next_coreset_indices.at[
            :, [i, i + 1]
        ].set(
            jnp.hstack(
                (
                    jnp.tile(index_to_include_in_coreset, (batch_indices.shape[0], 1)),
                    batch_indices[:, [i + 1]],
                )
            )
        )

        # Add the chosen coreset index to the current coreset indices
        current_coreset_indices = current_coreset_indices.at[i].set(
            index_to_include_in_coreset
        )
        return (
            current_coreset_indices,
            current_identity,
            all_possible_next_coreset_indices,
        )
