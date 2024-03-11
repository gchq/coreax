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
        approximately) if it is not given, and then iteratively add points to the coreset
        balancing selecting points in high density regions with selecting points far
        from those already in the coreset.

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

class GreedyCMMD(coreax.reduction.Coreset):
    r"""
    Apply GreedyCMMD to a supervised dataset.

    GreedyCMMD is a deterministic, iterative and greedy approach to determine this
    compressed representation.

    Given one has an original dataset :math:`\mathcal{D}^{(1)} = \{(x_i, y_i)\}_{i=1}^n` of ``n`` 
    pairs with :math:`x\in\mathbb{R}^d` and :math:`y\in\mathbb{R}^p`, and one has selected 
    :math:`T` data pairs :math:`\mathcal{D}^{(2)} = \{(\tilde{x}_i, \tilde{y}_i)\}_{i=1}^T`
    already for their compressed representation of the original dataset, GreedyCMMD selects 
    the next point to minimise the conditional maximum mean discrepancy (CMMD):
    
    .. math::

        \text{CMMD}^2(\mathcal{D}^{(1)}, \mathcal{D}^{(2)}) = ||\hat{\mu}^{(1)} - \hat{\mu}^{(2)}||^2_{\mathcal{H}_k \otimes \mathcal{H}_l}

    where :math:`\hat{\mu}^{(1)},\hat{\mu}^{(2)}` are the conditional mean embeddings estimated 
    with :math:`\mathcal{D}^{(1)}` and :math:`\mathcal{D}^{(2)}` respectively, and 
    :math:`\mathcal{H}_k,\mathcal{H}_l` are the RKHSs corresponding to the kernel functions
    :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` and
    :math:`l: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}` respectively. The search
    is performed over the entire dataset. 

    This class works with all children of :class:`~coreax.kernel.Kernel`, including
    Stein kernels.

    :param random_key: Key for random number generation
    :param feature_kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param response_kernel: :class:`~coreax.kernel.Kernel` instance implementing a kernel
        function :math:`k: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}`
    :param num_feature_dimensions: An integer representing the dimensionality of the features 
        :math:`x`
    :param lambdas: A  :math:`1 \times 2` array of reguralisation parameters corresponding to 
        the original dataset :math:`\mathcal{D}^{(1)}` and the coreset :math:`\mathcal{D}^{(2)}`
    :param unique: Boolean that enforces the resulting coreset will only contain
        unique elements
    :param batch_size: An integer representing the size of the batches of data pairs sampled at
        each iteration for consideration for adding to the coreset
    """
    def __init__(
        self,
        random_key: coreax.validation.KeyArrayLike,
        *,
        feature_kernel: coreax.kernel.Kernel,
        response_kernel: coreax.kernel.Kernel,
        num_feature_dimensions: int,
        lambdas: ArrayLike = jnp.array([1e-6, 1e-6]),
        unique: bool = True,
        batch_size: int | None = None
    ):
        # Validate inputs
        coreax.validation.validate_key_array(x=random_key, object_name="random_key")
        self.random_key = random_key
        
        coreax.validation.validate_is_instance(
            x=feature_kernel, object_name="feature_kernel", expected_type=coreax.kernel.Kernel
        )
        self.feature_kernel = feature_kernel

        coreax.validation.validate_is_instance(
            x=response_kernel, object_name="response_kernel", expected_type=coreax.kernel.Kernel
        )
        self.response_kernel = response_kernel
        
        num_feature_dimensions = coreax.validation.cast_as_type(
            x=num_feature_dimensions, object_name="num_feature_dimensions", type_caster=int
        )
        self.num_feature_dimensions = num_feature_dimensions

        lambdas = coreax.validation.cast_as_type(
            x=lambdas, object_name="lambdas", type_caster=jnp.atleast_1d
        )
        coreax.validation.validate_in_range(
            x=lambdas[0],
            object_name="lambdas[0]",
            strict_inequalities=True,
            lower_bound=0,
        )
        coreax.validation.validate_in_range(
            x=lambdas[1],
            object_name="lambdas[1]",
            strict_inequalities=True,
            lower_bound=0,
        )
        self.lambdas = lambdas
        
        unique = coreax.validation.cast_as_type(
            x=unique, object_name="unique", type_caster=bool
        )
        self.unique = unique

        if batch_size is not None:
            batch_size = coreax.validation.cast_as_type(
                x=batch_size, object_name="batch_size", type_caster=int
            )
            coreax.validation.validate_in_range(
                x=batch_size,
                object_name="batch_size",
                strict_inequalities=True,
                lower_bound=0,
            )
        self.batch_size = batch_size
        
        # Initialise parent with an arbitrary unused kernel as this method requires a separate
        # kernel for both the features and the response.
        super().__init__(
            weights_optimiser=None,
            kernel=SquaredExponentialKernel(),
            refine_method=None,
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
            self.coreset
        )
        aux_data = {
            "arbitrary_unused_kernel": self.kernel,
            "num_feature_dimensions": self.num_feature_dimensions,
            "lambdas": self.lambdas,
            "unique": self.unique,
            "batch_size": self.batch_size,
            "refine_method": self.refine_method,
            "weights_optimiser": self.weights_optimiser,
        }
        return children, aux_data

    def fit_to_size(self, coreset_size: int) -> None:
        r"""
        Execute greedy CMMD algorithm with Jax.

        We first compute the kernel matrices, and inverse feature kernel matrix
        and then iteratively add those points to the coreset that minimise the CMMD.

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
            lower_bound=0
        )

        # Compute and store original feature and response kernel matrices, and invert feature kernel matrix
        self.K1 = self.feature_kernel.compute(
            self.original_data.pre_coreset_array[:, :self.num_feature_dimensions],
            self.original_data.pre_coreset_array[:, :self.num_feature_dimensions]
        )
        self.L1 = self.response_kernel.compute(
            self.original_data.pre_coreset_array[:, self.num_feature_dimensions:],
            self.original_data.pre_coreset_array[:, self.num_feature_dimensions:]
        )
        identity = jnp.eye(self.K1.shape[0])
        self.W1 = jnp.linalg.lstsq(self.K1 + self.lambdas[0]*identity, identity)[0]

        # Compute KWL matrix product
        self.KWL1 = self.K1.dot(self.W1).dot(self.L1)

        # Record the size of the original dataset
        num_data_points = len(self.original_data.pre_coreset_array)
        
        # Initialise variable that will be updated throughout the loop. This is
        # initially a local variables with the coreset indices being assigned to self
        # when the entire set is created.
        coreset_indices = jnp.zeros(coreset_size, dtype=jnp.int32)

        # Greedily select coreset points
        body = partial(
            self._greedy_body,
            batch_size=self.batch_size,
            unique=self.unique,
        )
        for i in range(0, coreset_size):
            coreset_indices = body(i, coreset_indices)
            
        # Assign coreset indices & coreset to original data object
        self.coreset_indices = coreset_indices
        self.coreset = self.original_data.pre_coreset_array[self.coreset_indices, :]

    def _greedy_body(
        self,
        i: int,
        val: ArrayLike,
        unique: bool,
        batch_size: int | None
    ) -> Array:
        r"""
        Execute main loop of GreedyCMMD.

        This function carries out one iteration of GreedyCMMD.

        :param i: Loop counter, counting how many points are in the coreset on call of
            this method
        :param val: A a :math:`1 \times m` array with the current coreset
            indices in. Note that the array holding current coreset indices should always
            be the same length (however many coreset points are desired). The ``i``-th
            element of this gets updated to the index of the selected coreset point in
            iteration ``i``.
        :param unique: Flag for enforcing unique elements
        :param batch_size: An integer representing the size of the batches of data pairs sampled at
        each iteration for consideration for adding to the coreset
        :returns: Updated loop variable ``current_coreset_indices``.
        """
        # Unpack the components of the loop variables
        current_coreset_indices = val[:i]

        # Format inputs - note that the calls in jax for loops already validate the
        # ``i`` variable before calling.
        current_coreset_indices = coreax.validation.cast_as_type(
            x=current_coreset_indices,
            object_name="current_coreset_indices",
            type_caster=jnp.asarray,
        )
        
        # Get current coreset, produce an array of idxs that represent each possible next coreset
        num_data_points = self.original_data.pre_coreset_array.shape[0]
        if unique:
            candidate_indices = jnp.delete(jnp.arange(num_data_points), current_coreset_indices).reshape(-1, 1)
        else:
            candidate_indices = jnp.arange(num_data_points).reshape(-1, 1)
    
        if batch_size is not None:
            _, self.random_key = random.split(self.random_key)
            batch_idx = random.choice(
                self.random_key,
                a=candidate_indices,
                shape=(batch_size,),
                replace=False,
            )
            candidate_indices = candidate_indices[batch_idx, :].reshape(-1, 1)
        all_possible_next_coreset_indices = jnp.hstack(( jnp.tile( current_coreset_indices, (candidate_indices.shape[0], 1) ), candidate_indices ))
        
        # Extract all the coreset feature and response kernel matrices, and the coreset KWL product
        K2s = self.K1[ all_possible_next_coreset_indices[:, :, None], all_possible_next_coreset_indices[:, None, :] ]
        L2s = self.L1[ all_possible_next_coreset_indices[:, :, None], all_possible_next_coreset_indices[:, None, :] ]
        KWL2s = self.KWL1[ all_possible_next_coreset_indices[:, :, None], all_possible_next_coreset_indices[:, None, :] ]
    
        # Compute and store inverses for each coreset feature kernel matrix
        identity = jnp.eye(current_coreset_indices.shape[0] + 1)
        reg = lambdas[1] * identity
        W2s = jnp.array([ jnp.linalg.lstsq( K2s[i, :] + reg, identity, rcond = None )[0] for i in range(candidate_indices.shape[0]) ])
    
        # Compute each term of CMMD for each possible new coreset index
        term_2s = jnp.trace(jnp.matmul(jnp.matmul(jnp.matmul(W2s, L2s), W2s), K2s), axis1=1, axis2=2)
        term_3s = jnp.trace(jnp.matmul(KWL2s, W2s), axis1=1, axis2=2)
    
        # Choose the next coreset index
        index_to_include_in_coreset = candidate_indices[(term_2s - 2*term_3s).argmin()].item()
        return val.at[i].set(index_to_include_in_coreset)
