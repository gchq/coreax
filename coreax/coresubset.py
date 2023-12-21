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
Classes and associated functionality to construct coresets.

Given a :math:`n \times d` dataset, one may wish to construct a compressed
:math:`m \times d` dataset representation of this dataset, where :math:`m << n`. This
module contains implementations of approaches to do such a construction using coresets.
Coresets are a type of data reduction, and these inherit from
:class:`~coreax.reduction.Coreset`. The aim is to select a small set of indices
that represent the key features of a larger dataset.

The abstract base class is :class:`CoreSubset`. Concrete implementations are:

*   :class:`KernelHerding` defines the kernel herding method for both standard and Stein
    kernels.
*   :class:`RandomSample` selects points for the coreset using random sampling. It is
    typically only used for benchmarking against other coreset methods.

**:class:`KernelHerding`**
Kernel herding is a deterministic, iterative and greedy approach to determine this
compressed representation.

Given one has selected ``T`` data points for their compressed representation of the
original dataset, kernel herding selects the next point as

.. math::

    x_{T+1} = \argmax_{x} \left( \mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) \right)

where ``k`` is the kernel used, the expectation :math:`\mathbb{E}` is taken over the
entire dataset, and the search is over the entire dataset. This can informally be seen
as a balance between using points at which the underlying density is high (the first
term) and exploration of distinct regions of the space (the second term).
"""

# Support annotations with | in Python < 3.10
# TODO: Remove once no longer supporting old code
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import jax.lax as lax
import jax.numpy as jnp
from jax import Array, jit, random
from jax.typing import ArrayLike

from coreax.approximation import KernelMeanApproximator, approximator_factory
from coreax.kernel import Kernel
from coreax.reduction import Coreset, coreset_factory
from coreax.refine import Refine, refine_factory
from coreax.util import KernelFunction, create_instance_from_factory
from coreax.validation import cast_as_type, validate_in_range, validate_is_instance
from coreax.weights import WeightsOptimiser

if TYPE_CHECKING:
    from coreax.data import DataReader


class CoreSubset(Coreset):
    """
    Abstract base class for a method to construct a CoreSubset.

    A CoreSubset is considered to be a compressed representation of the original data,
    where elements in the CoreSubset must be contained within the original dataset.

    :param original_data: A :class:`~coreax.data.DataReader` object holding the data
    :param weights_optimiser:  TODO: Is this needed?
    :param kernel: A :class:`~coreax.kernel.Kernel` object
    :param coreset_size: The size of the of coreset to generate
    """

    def __init__(
        self,
        original_data: DataReader,
        weights_optimiser: str | WeightsOptimiser | None,
        kernel: Kernel | None,
        coreset_size: int,
    ):
        """Initialise a CoreSubset class."""
        # Validate inputs
        # TODO: Reviewer - To add when inputs are aligned with wider code.
        coreset_size = cast_as_type(
            x=coreset_size, object_name="coreset_size", type_caster=int
        )
        validate_in_range(
            x=coreset_size,
            object_name="coreset_size",
            strict_inequalities=True,
            lower_bound=0,
        )

        # Assign inputs
        self.coreset_size = coreset_size
        super().__init__(original_data, weights_optimiser, kernel)

        # Predefine to store coreset indices
        self.coreset_indices = jnp.zeros(
            [
                self.coreset_size,
            ]
        )

    @abstractmethod
    def fit(
        self,
        x: ArrayLike,
        kernel: Kernel | None,
    ) -> None:
        r"""
        Fit...

        TODO Update when fit and fit_to_size determined and code glued together - what
            is the expected input to fit?

        :param x: The original :math:`n \times d` data to generate a coreset from
        :param kernel: A :class:`~coreax.kernel.Kernel` object. The only instance where
            this can be set to :data:`None` is when one is using
            :class:`~coreax.coresubset.RandomSample`.
        """


class KernelHerding(CoreSubset):
    r"""
    Apply kernel herding to a dataset.

    Kernel herding is a deterministic, iterative and greedy approach to determine this
    compressed representation.

    Given one has selected ``T`` data points for their compressed representation of the
    original dataset, kernel herding selects the next point as:

    .. math::

        x_{T+1} = \argmax_{x} \left( \mathbb{E}[k(x, x')] -
            \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) \right)

    where ``k`` is the kernel used, the expectation :math:`\mathbb{E}` is taken over
    the entire dataset, and the search is over the entire dataset. This can informally
    be seen as a balance between using points at which the underlying density is high
    (the first term) and exploration of distinct regions of the space (the second term).

    This class works with all children of `~coreax.kernel.Kernel`, including Stein
    kernels.
    """

    def __init__(
        self,
        original_data: DataReader,
        weights_optimiser: str | WeightsOptimiser | None,
        kernel: Kernel,
        coreset_size: int,
    ):
        """
        Initialise a KernelHerding class.

        :param original_data:  TODO
        :param weights_optimiser:  TODO
        :param kernel: A :class:`~coreax.kernel.Kernel` object
        :param coreset_size: The size of the of coreset to generate
        """
        super().__init__(original_data, weights_optimiser, kernel, coreset_size)

    def fit(
        self,
        block_size: int = 10000,
        kernel_matrix_row_sum_mean: Array | None = None,
        unique: bool = True,
        refine: str | Refine | None = None,
        approximator: str | KernelMeanApproximator | None = None,
        random_key: random.PRNGKeyArray = random.PRNGKey(0),
        num_kernel_points: int = 10_000,
        num_train_points: int = 10_000,
    ):
        r"""
        Execute kernel herding algorithm with Jax.

        We first compute the kernel matrix row sum mean (either exactly, or
        approximately) if it is not given, and then iterative add points to the coreset
        balancing  selecting points in high density regions with selecting points far
        from those already in the coreset.

        :param block_size: Size of matrix blocks to process when computing the kernel
            matrix row sum mean. Larger blocks will require more memory in the system.
        :param kernel_matrix_row_sum_mean: Row sum of kernel matrix divided by the
            number of points. If given, re-computation will be avoided and performance
            gains are expected.
        :param unique: Boolean, that enforces the resulting coreset will only contain
            unique elements
        :param refine: Refine method to use or None (default) if no refinement required.
            Refinement is performed after the herding procedure is complete.
        :param approximator: The name of an approximator class to use, or the
            uninstantiated class directly as a dependency injection. If None (default)
            then calculation is exact, but can be computational intensive.
        :param random_key: Key for random number generation
        :param num_kernel_points: Number of kernel evaluation points for approximation
            of the kernel matrix row sum mean. Only used if ``approximator`` is not
            :data:`None`.
        :param num_train_points: Number of training points for ``approximator``. Only
            used if approximation method specified trains a model to approximate the
            kernel matrix row sum mean.
        """
        # Validate inputs
        # TODO: Reviewer - To add when inputs are aligned with wider code.

        # Record the size of the original dataset
        num_data_points = len(self.original_data.pre_coreset_array)

        # If needed, set up an approximator. This can be used for both kernel matrix row
        # sum mean computation inside of this method, as-well as optional refinement
        # later in the method
        if approximator is not None:
            approximator_instance = create_instance_from_factory(
                approximator_factory,
                approximator,
                random_key=random_key,
                num_kernel_points=num_kernel_points,
            )
        else:
            approximator_instance = None

        # If needed, compute the kernel matrix row sum mean - with or without an
        # approximator as specified by the inputs to this method
        if kernel_matrix_row_sum_mean is None:
            if approximator_instance is not None:
                kernel_matrix_row_sum_mean = (
                    self.kernel.approximate_kernel_matrix_row_sum_mean(
                        x=self.original_data.pre_coreset_array,
                        approximator=approximator_instance,
                        random_key=random_key,
                        num_kernel_points=num_kernel_points,
                        num_train_points=num_train_points,
                    )
                )
            else:
                kernel_matrix_row_sum_mean = (
                    self.kernel.calculate_kernel_matrix_row_sum_mean(
                        x=self.original_data.pre_coreset_array, max_size=block_size
                    )
                )

        # Initialise loop updateables - as local variables and assign the coreset
        # indices to the object when the entire set is created
        kernel_similarity_penalty = jnp.zeros(num_data_points)
        coreset_indices = jnp.zeros(self.coreset_size, dtype=jnp.int32)

        # Greedly select coreset points
        body = partial(
            self._greedy_body,
            x=self.original_data.pre_coreset_array,
            kernel_vectorised=self.kernel.compute,
            kernel_matrix_row_sum_mean=kernel_matrix_row_sum_mean,
            unique=unique,
        )
        coreset_indices, kernel_similarity_penalty = lax.fori_loop(
            lower=0,
            upper=self.coreset_size,
            body_fun=body,
            init_val=(coreset_indices, kernel_similarity_penalty),
        )

        # Assign coreset indices & coreset to original data object, so that they can be
        # refined if needed in the next step
        self.coreset_indices = coreset_indices
        self.coreset = self.original_data.pre_coreset_array[self.coreset_indices, :]

        # TODO: Reviewer - This may need altering to align with wider OOP design
        # Apply refinement to the coreset if needed
        if refine is not None:
            # Create a Refine object
            refine_instance = create_instance_from_factory(
                refine_factory,
                refine,
                approximate_kernel_row_sum=False if approximator is None else True,
                approximator=approximator_instance,
            )

            # Refine the coreset - which constitutes a greedy search for points that
            # can be replaced to improve some metric assessing quality. This will alter
            # the coreset indices attribute on the data object.
            refine_instance.refine(data_reduction=self)
            # TODO: Maybe fix refine terminology etc to use coreset

        # TODO: Reviewer - This has been commented out as I believe weight computation
        #  now happens outside of fit. Delete if so, or uncomment if not.
        # # If we wish to determine optimal weights for the points, do so
        # if self.weights_optimiser is not None:
        #     # Create the weights optimiser
        #     weights_optimiser = create_instance_from_factory(
        #         weights_factory,
        #         self.weights_optimiser,
        #         kernel=self.kernel
        #     )
        #
        #     # Determine the optimal weights
        #     self.weights = weights_optimiser.solve(
        #         x=self.original_data.pre_coreset_array,
        #         y=self.original_data.pre_coreset_array[self.coreset_indices, :]
        #     )

    @staticmethod
    @partial(jit, static_argnames=["kernel_vectorised", "unique"])
    def _greedy_body(
        i: int,
        val: tuple[ArrayLike, ArrayLike],
        x: ArrayLike,
        kernel_vectorised: KernelFunction,
        kernel_matrix_row_sum_mean: ArrayLike,
        unique: bool,
    ) -> tuple[Array, Array]:
        r"""
        Execute main loop of greedy kernel herding.

        This function carries out one iteration of kernel herding. Recall that kernel
        herding is defined as

        .. math::

            x_{T+1} = \argmax_{x} \left( \mathbb{E}[k(x, x')] -
                \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) \right)

        where ``k`` is the kernel used, the expectation :math:`\mathbb{E}` is taken over
        the entire dataset, and the search is over the entire dataset.

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
        :param kernel_vectorised: Vectorised kernel computation function. This should be
            the :meth:`compute` method of a :class:`~coreax.kernel.Kernel` object,
        :param kernel_matrix_row_sum_mean: A :math:`1 \times n` array holding the mean
            over rows for the kernel Gram matrix
        :param unique: Flag for enforcing unique elements
        :returns: Updated loop variables ``current_coreset_indices`` and
            ``current_kernel_similarity_penalty``
        """
        # Unpack the components of the updatable
        (current_coreset_indices, current_kernel_similarity_penalty) = val

        # Format inputs - note that the calls in jax for loops already validate the
        # ``i`` variable before calling.
        current_coreset_indices = cast_as_type(
            x=current_coreset_indices,
            object_name="current_coreset_indices",
            type_caster=jnp.asarray,
        )
        current_kernel_similarity_penalty = cast_as_type(
            x=current_kernel_similarity_penalty,
            object_name="current_kernel_similarity_penalty",
            type_caster=jnp.asarray,
        )
        x = cast_as_type(x=x, object_name="x", type_caster=jnp.atleast_2d)
        validate_is_instance(
            x=kernel_vectorised, object_name="kernel_vectorised", expected_type=Callable
        )
        kernel_matrix_row_sum_mean = cast_as_type(
            x=kernel_matrix_row_sum_mean,
            object_name="kernel_matrix_row_sum_mean",
            type_caster=jnp.asarray,
        )
        unique = cast_as_type(x=unique, object_name="unique", type_caster=bool)

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
        current_kernel_similarity_penalty = (
            current_kernel_similarity_penalty + penalty_update
        )

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


class RandomSample(CoreSubset):
    r"""
    Reduce a dataset by uniformly randomly sampling a fixed number of points.

    :param original_data: The :math:`n \times d` dataset to be reduced
    :param coreset_size: The size of the of coreset to generate
    :param random_key: Pseudo-random number generator key for sampling
    :param unique: If :data:`True`, this flag enforces unique elements, i.e. sampling
        without replacement
    """

    def __init__(
        self,
        original_data: DataReader,
        coreset_size: int,
        random_key: ArrayLike = 0,
        unique: bool = True,
    ):
        """Initialise a random sampling object."""
        self.random_key = random_key
        self.unique_flag = unique

        # Initialise Coreset parent
        super().__init__(
            original_data=original_data,
            weights_optimiser=None,
            kernel=None,
            coreset_size=coreset_size,
        )

    def fit(self) -> None:
        r"""
        Reduce a dataset by uniformly randomly sampling a fixed number of points.

        TODO Update when fit and fit_to_size determined and code glued together - what
            is the expected input to fit?

        This class is updated in-place. The randomly sampled points are stored in the
        ``reduction_indices`` attribute.

        :return: Nothing, the coreset is assigned to :attr:`coreset`
        """
        key = random.PRNGKey(self.random_key)
        num_data_points = len(self.original_data.pre_coreset_array)

        # Randomly sample the desired number of points to form a coreset
        random_indices = random.choice(
            key,
            a=jnp.arange(0, num_data_points),
            shape=(self.coreset_size,),
            replace=not self.unique_flag,
        )

        # Assign coreset indices and coreset to the object
        self.coreset_indices = random_indices
        self.coreset = self.original_data.pre_coreset_array[random_indices]

        # TODO: Does this need refine and weights? To update when all merged.


coreset_factory.register("kernel_herding", KernelHerding)
coreset_factory.register("random_sample", RandomSample)
