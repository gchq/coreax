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
Classes and associated functionality to approximate kernel matrix row sum mean.

When a dataset is very large, computing the mean distance between a given point
and all other points can be time-consuming. For many approaches in this library, such an
operation would need to be done for all data points in the dataset. To reduce the
computational demand of this, the quantity can instead be approximated by various
methods.

For a concrete example of the (true) kernel matrix row sum mean, consider the data:

.. math::

    x = [ [0.0, 0.0], [0.5, 0.5], [1.0, 0.0], [-1.0, 0.0] ]

and a :class:`~coreax.kernel.SquaredExponentialKernel` which is defined as

.. math::

    k(x,y) = \text{output_scale} * \exp(-||x-y||^2/2 * \text{length_scale}^2)

For simplicity, we set ``length_scale`` to :math:`\frac{1}{\sqrt{2}}`
and ``output_scale`` to 1.

For a single row (data point), the kernel matrix row sum mean is computed by
applying the kernel to this data record and all other data records. We then sum
the results and divide by the number of data points. The first
data point ``[0, 0]`` in the data considered here therefore gives a result of:

.. math::

      (1/4) * (
      exp(-((0.0 - 0.0)^2 + (0.0 - 0.0)^2)) +
      exp(-((0.0 - 0.5)^2 + (0.0 - 0.5)^2)) +
      exp(-((0.0 - 1.0)^2 + (0.0 - 0.0)^2)) +
      exp(-((0.0 - -1.0)^2 + (0.0 - 0.0)^2))
      )

which evaluates to 0.5855723855138795.

We can repeat the above but considering each data-point in ``x`` in turn and
attain an array with one element for each data point in the original dataset. For this
example data, the final result would be:

.. math::

    [
        0.5855723855138795,
        0.5737865795122914,
        0.4981814349432025,
        0.3670700196710188
    ]

The approximations within this module attempt to accurately estimate the result of the
above calculations, without having to perform as many evaluations of the kernel. All
approximators in this module implement the base class :class:`KernelMeanApproximator`.
"""

# Support annotations with | in Python < 3.10
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax, random
from jax.typing import ArrayLike

import coreax.kernel
import coreax.util


class KernelMeanApproximator(ABC):
    """
    Base class for approximation methods to kernel row sum means.

    When a dataset is very large, computing the mean distance between a given point
    and all other points can be time-consuming. Instead, this property can be
    approximated by various methods. :class:`KernelMeanApproximator` is the base class
    for implementing these approximation methods.

    .. note::

        The parameter `num_kernel_points` can take any non-negative value, however,
        setting this to 0 will simply produce a kernel mean approximation of zero at all
        points.

    :param random_key: Key for random number generation
    :param kernel: A :class:`~coreax.kernel.Kernel` object
    :param num_kernel_points: Number of kernel evaluation points
    """

    def __init__(
        self,
        random_key: coreax.util.KeyArrayLike,
        kernel: coreax.kernel.Kernel,
        num_kernel_points: int = 10_000,
    ):
        """Define approximator to the mean of the row sum of kernel distance matrix."""
        # Assign inputs
        self.kernel = kernel
        self.random_key = random_key
        self.num_kernel_points = num_kernel_points

    @abstractmethod
    def approximate(self, data: ArrayLike) -> Array:
        r"""
        Approximate kernel matrix row sum mean.

        :param data: Original :math:`n \times d` data
        :return: Approximation of the kernel matrix row sum divided by the number of
            data points in the dataset
        """


class RandomApproximator(KernelMeanApproximator):
    """
    Approximate the kernel matrix row mean using regression on randomly sampled points.

    When a dataset is very large, computing the mean distance between a given point
    and all other points can be time-consuming. Instead, this property can be
    approximated by various methods. :class:`RandomApproximator` is a class that does
    such an approximation using kernel regression on a subset of randomly selected
    points from the dataset.

    :param random_key: Key for random number generation
    :param kernel: A :class:`~coreax.kernel.Kernel` object
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    """

    def __init__(
        self,
        random_key: coreax.util.KeyArrayLike,
        kernel: coreax.kernel.Kernel,
        num_kernel_points: int = 10_000,
        num_train_points: int = 10_000,
    ):
        """Approximate kernel row mean by regression on points selected randomly."""
        self.num_train_points = num_train_points

        # Initialise parent
        super().__init__(
            random_key=random_key,
            kernel=kernel,
            num_kernel_points=num_kernel_points,
        )

    def approximate(
        self,
        data: ArrayLike,
    ) -> Array:
        r"""
        Compute approximate kernel row mean by regression on randomly selected points.

        :param data: Original :math:`n \times d` data
        :return: Approximation of the kernel matrix row sum divided by the number of
            data points in the dataset
        """
        # Format input
        data = jnp.atleast_2d(data)

        # Record dataset size
        num_data_points = len(data)

        # Randomly select points for kernel regression
        key, subkey = random.split(self.random_key)
        try:
            features_idx = random.choice(
                subkey, num_data_points, (self.num_kernel_points,), replace=False
            )
        except TypeError as exception:
            if self.num_kernel_points < 0:
                raise ValueError("num_kernel_points must be positive") from exception
            raise
        except ValueError as exception:
            if self.num_kernel_points > num_data_points:
                raise ValueError(
                    "num_kernel_points must be no larger than the number of points in "
                    "the provided data"
                ) from exception
            raise

        # Compute feature matrix
        features = self.kernel.compute(data, data[features_idx])

        try:
            train_idx = random.choice(
                key, num_data_points, (self.num_train_points,), replace=False
            )
        except TypeError as exception:
            if self.num_train_points < 0:
                raise ValueError("num_train_points must be positive") from exception
            raise
        except ValueError as exception:
            if self.num_train_points > num_data_points:
                raise ValueError(
                    "num_train_points must be no larger than the number of points in "
                    "the provided data"
                ) from exception
            raise

        # Isolate targets for regression problem
        target = (
            self.kernel.compute(data[train_idx], data).sum(axis=1) / num_data_points
        )

        # Solve regression problem
        params, _, _, _ = jnp.linalg.lstsq(features[train_idx], target)

        return features @ params


class ANNchorApproximator(KernelMeanApproximator):
    r"""
    Approximation method to kernel mean through regression on ANNchor selected points.

    When a dataset is very large, computing the mean distance between a given point
    and all other points can be time-consuming. Instead, this property can be
    approximated by various methods. :class:`ANNchorApproximator` is a class that does
    such an approximation using kernel regression on a subset of points selected via the
    ANNchor approach from the dataset. The ANNchor implementation used can be found
    `here <https://github.com/gchq/annchor>`_.

    :param random_key: Key for random number generation
    :param kernel: A :class:`~coreax.kernel.Kernel` object
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    """

    def __init__(
        self,
        random_key: coreax.util.KeyArrayLike,
        kernel: coreax.kernel.Kernel,
        num_kernel_points: int = 10_000,
        num_train_points: int = 10_000,
    ):
        """Approximate kernel row mean by regression on ANNchor selected points."""
        self.num_train_points = num_train_points

        # Initialise parent
        super().__init__(
            random_key=random_key,
            kernel=kernel,
            num_kernel_points=num_kernel_points,
        )

    def approximate(
        self,
        data: ArrayLike,
    ) -> Array:
        r"""
        Compute approximate kernel row mean by regression on ANNchor selected points.

        :param data: Original :math:`n \times d` data
        :return: Approximation of the kernel matrix row sum divided by the number of
            data points in the dataset
        """
        # Format input
        data = jnp.atleast_2d(data)

        # Record dataset size
        num_data_points = len(data)

        # Select point for kernel regression using ANNchor construction
        try:
            features = jnp.zeros((num_data_points, self.num_kernel_points))
        except TypeError as exception:
            if self.num_kernel_points <= 0:
                raise ValueError("num_kernel_points must be positive") from exception
            raise

        # Compute feature matrix
        try:
            features = features.at[:, 0].set(self.kernel.compute(data, data[0])[:, 0])
        except IndexError as exception:
            if self.num_kernel_points <= 0:
                raise ValueError(
                    "num_kernel_points must be positive and non-zero"
                ) from exception
            raise
        body = partial(_anchor_body, data=data, kernel_function=self.kernel.compute)
        features = lax.fori_loop(1, self.num_kernel_points, body, features)

        # Randomly select training points
        try:
            train_idx = random.choice(
                self.random_key,
                num_data_points,
                (self.num_train_points,),
                replace=False,
            )
        except TypeError as exception:
            if self.num_train_points < 0:
                raise ValueError("num_train_points must be positive") from exception
            raise
        except ValueError as exception:
            if self.num_train_points > num_data_points:
                raise ValueError(
                    "num_train_points must be no larger than the number of points in "
                    "the provided data"
                ) from exception
            raise

        # Isolate targets for regression problem
        target = (
            self.kernel.compute(data[train_idx], data).sum(axis=1) / num_data_points
        )

        # solve regression problem
        params, _, _, _ = jnp.linalg.lstsq(features[train_idx], target)

        return features @ params


class NystromApproximator(KernelMeanApproximator):
    """
    Approximate kernel matrix row sum mean using as Nystrom approximation.

    When a dataset is very large, computing the mean distance between a given point
    and all other points can be time-consuming. Instead, this property can be
    approximated by various methods. NystromApproximator is a class that does such an
    approximation using a Nystrom approximation on a subset of points selected at
    random from the data. Further details for Nystrom kernel mean embeddings can be
    found in :cite:`chatalic2022nystrom`.

    :param random_key: Key for random number generation
    :param kernel: A :class:`~coreax.kernel.Kernel` object
    :param num_kernel_points: Number of kernel evaluation points
    """

    def __init__(
        self,
        random_key: coreax.util.KeyArrayLike,
        kernel: coreax.kernel.Kernel,
        num_kernel_points: int = 10_000,
    ):
        """Approximate kernel row mean by using Nystrom approximation."""
        # Initialise parent
        super().__init__(
            random_key=random_key,
            kernel=kernel,
            num_kernel_points=num_kernel_points,
        )

    def approximate(
        self,
        data: ArrayLike,
    ) -> Array:
        r"""
        Compute approximate kernel row sum mean using a Nystrom approximation.

        We consider a :math:`n \times d` dataset, and wish to use an :math:`m \times d`
        subset of this to approximate the kernel matrix row sum mean. The ``m`` points
        are selected uniformly at random, and the Nystrom estimator, as defined in
        :cite:`chatalic2022nystrom` is computed using this subset.

        :param data: Original :math:`n \times d` data
        :return: Approximation of the kernel matrix row sum divided by the number of
            data points in the dataset
        """
        # Format input
        data = jnp.atleast_2d(data)

        # Record dataset size
        num_data_points = len(data)

        # Randomly select points for kernel regression
        try:
            sample_points = random.choice(
                self.random_key,
                num_data_points,
                (self.num_kernel_points,),
                replace=False,
            )
        except TypeError as exception:
            if self.num_kernel_points <= 0:
                raise ValueError("num_kernel_points must be positive") from exception
            raise
        except ValueError as exception:
            if self.num_kernel_points > num_data_points:
                raise ValueError(
                    "num_kernel_points must be no larger than the number of points in "
                    "the provided data"
                ) from exception
            raise

        # Solve for kernel distances
        kernel_mn = self.kernel.compute(data[sample_points], data)
        kernel_mm = self.kernel.compute(data[sample_points], data[sample_points])
        alpha = (jnp.linalg.pinv(kernel_mm) @ kernel_mn).sum(axis=1) / num_data_points

        return kernel_mn.T @ alpha


@partial(jit, static_argnames=["kernel_function"])
def _anchor_body(
    idx: int,
    features: ArrayLike,
    data: ArrayLike,
    kernel_function: coreax.util.KernelComputeType,
) -> Array:
    r"""
    Execute main loop of the ANNchor construction.

    :param idx: Loop counter
    :param features: Loop variables to be updated
    :param data: Original :math:`n \times d` dataset
    :param kernel_function: Vectorised kernel function on pairs ``(X,x)``:
        :math:`k: \mathbb{R}^{n \times d} \times \mathbb{R}^d \rightarrow \mathbb{R}^n`
    :return: Updated loop variables ``features``
    """
    # Format inputs
    features = jnp.atleast_2d(features)
    data = jnp.atleast_2d(data)

    max_entry = features.max(axis=1).argmin()
    features = features.at[:, idx].set(kernel_function(data, data[max_entry])[:, 0])

    return features


class KernelInverseApproximator(ABC):
    """
    Base class for approximation methods to invert regularised kernel matrices.

    When a dataset is very large, computing the inverse of the kernel gram matrix
    can be very time-consuming. Instead, this property can be approximated by various
    methods. :class:`KernelInverseApproximator` is the base class
    for implementing these approximation methods.

    :param random_key: Key for random number generation
    """

    def __init__(self, random_key: coreax.util.KeyArrayLike):
        """Define approximator to the kernel matrix inverse."""
        # Assign inputs
        self.random_key = random_key

    @abstractmethod
    def approximate(
        self,
        kernel_gramian: Array,
        regularisation_parameter: float,
        identity: ArrayLike,
    ) -> Array:
        r"""
        Approximate kernel matrix inverse.

        :param kernel_gramian: Original :math:`n \times n` kernel gram matrix
        :param regularisation_parameter: Regularisation parameter for stable inversion
            of array, negative values will be converted to positive
        :param identity: Block identity matrix
        :return: Approximation of the kernel matrix inverse
        """


class RandomisedEigendecompositionApproximator(KernelInverseApproximator):
    """
     Approximate regularised kernel matrix inverse using randomised eigendecomposition.

     When a dataset is very large, computing the inverse of the kernel gram matrix
     can be very time-consuming. Instead, this property can be approximated by various
     methods. :class:`RandomisedEigendecompositionApproximator` is a class that does
     such an approximation using a randomised eigendecomposition. Further details can be
    found in (:cite:`halko2011randomness` Algorithm 4.4. and 5.3).

     :param random_key: Key for random number generation
     :param rcond: Cut-off ratio for small singular values of a. For the purposes of
        rank determination, singular values are treated as zero if they are smaller than
        rcond times the largest singular value of a. The default value of None will use
        the machine precision multiplied by the largest dimension of the array.
        An alternate value of -1 will use machine precision.
    :param oversampling_parameter: Number of random columns to sample; the larger the
        oversampling_parameter, the more accurate, but slower the method will be
    :param power_iterations: Number of power iterations to do; the larger the
        power_iterations, the more accurate, but slower the method will be
    """

    def __init__(
        self,
        random_key: coreax.util.KeyArrayLike,
        oversampling_parameter: int = 25,
        power_iterations: int = 1,
        rcond: float | None = None,
    ):
        """Randomised eigendecomposition approximator to the kernel matrix inverse."""
        # Initialise parent
        super().__init__(random_key=random_key)
        self.oversampling_parameter = oversampling_parameter
        self.power_iterations = power_iterations
        self.rcond = rcond

        # Check attributes are valid
        if self.rcond is not None:
            if self.rcond < 0 and self.rcond != -1:
                raise ValueError("rcond must be non-negative, except for value of -1")

    def approximate(
        self,
        kernel_gramian: ArrayLike,
        regularisation_parameter: float,
        identity: ArrayLike,
    ) -> Array:
        r"""
        Compute approximate kernel matrix inverse using randomised eigendecomposition.

        We consider a :math:`n \times n` kernel matrix, and use
        (:cite:`halko2011randomness` Algorithm 4.4. and 5.3) to approximate the
        eigendecomposition, and using this, the inverse.

        .. note::
            The function is designed to invert "kernel matrices" where only the top-left
            block contains non-zero elements. We return a block array, the same size as
            the input array, where each block has only zero elements except for the
            top-left block, which is the inverse of the non-zero input block. The most
            efficient way to compute this in JAX requires the 'identity' array to be a
            matrix of zeros except for ones on the diagonal up to the size of the
            non-zero block.

        :param kernel_gramian: Original :math:`n \times n` kernel gram matrix
        :param regularisation_parameter: Regularisation parameter for stable inversion
            of array, negative values will be converted to positive
        :param identity: Identity matrix
        :return: Approximation of the kernel matrix inverse
        """
        # Set rcond parameter if not given
        n = kernel_gramian.shape[0]
        machine_precision = jnp.finfo(kernel_gramian.dtype).eps
        if self.rcond is None:
            rcond = machine_precision * n
        elif self.rcond == -1:
            rcond = machine_precision
        else:
            rcond = self.rcond

        # Get randomised eigendecomposition of regularised kernel matrix
        approximate_eigenvalues, approximate_eigenvectors = (
            coreax.util.randomised_eigendecomposition(
                random_key=self.random_key,
                array=kernel_gramian + abs(regularisation_parameter) * identity,
                oversampling_parameter=self.oversampling_parameter,
                power_iterations=self.power_iterations,
            )
        )

        # Mask the eigenvalues that are zero or almost zero according to value of rcond
        # for safe inversion.
        mask = approximate_eigenvalues >= jnp.array(rcond) * approximate_eigenvalues[-1]
        safe_approximate_eigenvalues = jnp.where(mask, approximate_eigenvalues, 1)

        # Invert the eigenvalues and extend array ready for broadcasting
        approximate_inverse_eigenvalues = jnp.where(
            mask, 1 / safe_approximate_eigenvalues, 0
        )[:, jnp.newaxis]

        # Solve Ax = I, x = A^-1 = UL^-1U^T
        return approximate_eigenvectors.dot(
            (approximate_inverse_eigenvalues * approximate_eigenvectors.T).dot(identity)
        )
