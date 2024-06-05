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
Classes and associated functionality to approximate kernels.

When a dataset is very large, methods which have to evaluate all pairwise combinations
of the data, such as :meth:`~coreax.kernel.Kernel.gramian_row_mean`, can become
prohibitively expensive. To reduce this computational cost, such methods can instead be
approximated (providing suitable approximation error can be achieved).

The :class:`ApproximateKernel`\ s in this module provide the functionality required to
override specific methods of a ``base_kernel`` with their approximate counterparts.
Because :class:`ApproximateKernel`\ s inherit from :class:`~coreax.kernel.Kernel`, with
all functionality provided through composition with a ``base_kernel``, they can be
freely used in any place where a standard :class:`~coreax.kernel.Kernel` is expected.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.typing import ArrayLike
from typing_extensions import TYPE_CHECKING, Literal, override

from coreax.data import Data
from coreax.kernel import CompositeKernel
from coreax.util import KeyArrayLike

if TYPE_CHECKING:
    from coreax.kernel import Kernel  # noqa: F401


def _random_indices(
    key: KeyArrayLike,
    num_data_points: int,
    num_select: int,
    mode: Literal["kernel", "train"] = "kernel",
):
    """
    Select a random subset of indices.

    :param key: RNG key for seeding the random selection
    :param num_data_points: The total number of indexable data points
    :param num_select: The number of indices to select
    :param mode: The selection mode, used for error message formatting
    :return: A randomly selected subset of indices, of size ``num_samples``, for a
        dataset with ``num_data_points`` indexable entries.
    """
    try:
        selected_indices = jr.choice(key, num_data_points, (num_select,), replace=False)
    except ValueError as exception:
        if num_select > num_data_points:
            raise ValueError(
                f"'num_{mode}_points' must be no larger than the number of points in "
                "the provided data"
            ) from exception
        raise
    return selected_indices


def _random_least_squares(
    key: KeyArrayLike,
    data: Array,
    features: Array,
    num_indices: int,
    target_map: Callable[[Array], Array] = lambda x: x,
) -> Array:
    r"""
    Solve the least-square problem on a random subset of the system.

    A linear system :math:`Ax = b`, solved via least-squares as :math:`x = A^+ b`, can
    be approximated by random least-square as `x \approx \hat{x} = \hat{A}^+ \hat{b}`,
    where :math:`\hat{A} = A_i\ \text{and}\ \hat{b} = b_i\, \forall i \in I]`. `I` is a
    random subset of indices for the original system of equations.

    :param key: RNG key for seeding the random selection
    :param data: The data :math:`z`; yields :math:`b` when pushed through the target map
    :param features: The feature matrix :math:`A`
    :param num_indices: The size of the random subset of indices :math:`I`
    :param target_map: The target map :math:`\phi` which defines :math:`b := \phi(z)`,
        where :math:`z` is the input ``data``
    :return: The push-forward of the approximate solution :math:`A\hat{x}`
    """
    num_data_points = len(data)
    train_idx = _random_indices(key, num_data_points, num_indices, mode="train")
    target = target_map(data[train_idx])
    approximate_solution, _, _, _ = jnp.linalg.lstsq(features[train_idx], target)
    return features @ approximate_solution


class ApproximateKernel(CompositeKernel):
    """
    Base class for approximated kernels.

    Provides approximations of the methods in the ``base_kernel``.

    The :meth:`~coreax.kernel.Kernel.gramian_row_mean` method is particularly amenable
    to approximation, with significant performance improvements possible depending on
    the acceptable levels of error.

    :param base_kernel: a :class:`~coreax.kernel.Kernel` whose attributes/methods are
        to be approximated
    """

    @override
    def compute_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return self.base_kernel.compute_elementwise(x, y)

    @override
    def grad_x_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return self.base_kernel.grad_x_elementwise(x, y)

    @override
    def grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return self.base_kernel.grad_y_elementwise(x, y)

    @override
    def divergence_x_grad_y_elementwise(self, x: ArrayLike, y: ArrayLike) -> Array:
        return self.base_kernel.divergence_x_grad_y_elementwise(x, y)


class RandomRegressionKernel(ApproximateKernel):
    """
    An approximate kernel that requires the attributes for random regression.

    :param base_kernel: a :class:`~coreax.kernel.Kernel` whose attributes/methods are
        to be approximated
    :param random_key: Key for random number generation
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    """

    random_key: KeyArrayLike
    num_kernel_points: int = 10_000
    num_train_points: int = 10_000

    def __check_init__(self):
        """Check that 'num_kernel_points' and 'num_train_points' are feasible."""
        if self.num_kernel_points <= 0:
            raise ValueError("'num_kernel_points' must be a positive integer")
        if self.num_train_points <= 0:
            raise ValueError("'num_train_points' must be a positive integer")


class MonteCarloApproximateKernel(RandomRegressionKernel):
    """
    Approximate a base kernel via random subset selection.

    Only the Gramian row-mean is approximated here, all other methods are inherited
    directly from the ``base_kernel``.

    :param base_kernel: a :class:`~coreax.kernel.Kernel` whose attributes/methods are
        to be approximated
    :param random_key: Key for random number generation
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    """

    def gramian_row_mean(self, x: Union[ArrayLike, Data], **kwargs) -> Array:
        r"""
        Approximate the Gramian row-mean by Monte-Carlo sampling.

        A uniform random subset of ``x`` is used to approximate the base kernel's
        Gramian row-mean.

        :param x: Data matrix, :math:`n \times d`
        :return: Approximation of the base kernel's Gramian row-mean
        """
        del kwargs
        data = jnp.atleast_2d(jnp.asarray(x))
        num_data_points = len(data)
        key = self.random_key
        features_idx = _random_indices(key, num_data_points, self.num_kernel_points - 1)
        features = self.base_kernel.compute(data, data[features_idx])
        return _random_least_squares(
            key,
            data,
            features,
            self.num_train_points,
            partial(self.base_kernel.compute_mean, data, axis=0),
        )


class ANNchorApproximateKernel(RandomRegressionKernel):
    r"""
    Approximate a base kernel via random kernel regression on ANNchor selected points.

    Only the base kernel's Gramian row-mean is approximated here, all other methods are
    inherited directly from the ``base_kernel``.

    :param base_kernel: a :class:`~coreax.kernel.Kernel` whose attributes/methods are
        to be approximated
    :param random_key: Key for random number generation
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    """

    def gramian_row_mean(self, x: Union[ArrayLike, Data], **kwargs) -> Array:
        r"""
        Approximate the Gramian row-mean by random regression on ANNchor points.

        A subset of ``x`` is selected via the ANNchor approach and random kernel
        regression used to approximate the base kernel's Gramian row-mean. The ANNchor
        implementation used can be found `here <https://github.com/gchq/annchor>`_.

        :param x: Data matrix, :math:`n \times d`
        :return: Approximation of the base kernel's Gramian row-mean
        """
        del kwargs
        data = jnp.atleast_2d(jnp.asarray(x))
        num_data_points = len(data)
        features = jnp.zeros((num_data_points, self.num_kernel_points))
        features = features.at[:, 0].set(self.base_kernel.compute(data, data[0])[:, 0])

        def _annchor_body(idx: int, _features: Array) -> Array:
            r"""
            Execute main loop of the ANNchor construction.

            :param idx: Loop counter
            :param _features: Loop variables to be updated
            :return: Updated loop variables ``features``
            """
            max_entry = _features.max(axis=1).argmin()
            _features = _features.at[:, idx].set(
                self.base_kernel.compute(data, data[max_entry])[:, 0]
            )
            return _features

        features = jax.lax.fori_loop(1, self.num_kernel_points, _annchor_body, features)
        return _random_least_squares(
            self.random_key,
            data,
            features,
            self.num_train_points,
            partial(self.base_kernel.compute_mean, data, axis=0),
        )


class NystromApproximateKernel(RandomRegressionKernel):
    """
    Approximate a base kernel via Nystrom approximation.

    Only the base kernel's Gramian row-mean is approximated here, all other methods
    are inherited directly from the ``base_kernel``.

    :param base_kernel: a :class:`~coreax.kernel.Kernel` whose attributes/methods are
        to be approximated
    :param random_key: Key for random number generation
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    """

    def gramian_row_mean(self, x: Union[ArrayLike, Data], **kwargs) -> Array:
        r"""
        Approximate the Gramian row-mean by Nystrom approximation.

        We consider a :math:`n \times d` dataset, and wish to use an :math:`m \times d`
        subset of this to approximate the base kernel's Gramian row-mean. The ``m``
        points are selected uniformly at random, and the Nystrom estimator, as defined
        in :cite:`chatalic2022nystrom` is computed using this subset.

        :param x: Data matrix, :math:`n \times d`
        :return: Approximation of the base kernel's Gramian row-mean
        """
        del kwargs
        data = jnp.atleast_2d(jnp.asarray(x))
        num_data_points = len(data)
        feature_idx = _random_indices(
            self.random_key, num_data_points, self.num_kernel_points
        )
        features = self.base_kernel.compute(data, data[feature_idx])
        return _random_least_squares(
            self.random_key,  # intentional key reuse to ensure train_idx = feature_idx
            data,
            features,
            self.num_train_points,
            self.base_kernel.gramian_row_mean,
        )


class RegularisedInverseApproximator(ABC):
    """
    Base class for approximation methods to invert regularised kernel gram matrices.

    When a dataset is very large, computing the regularised inverse of the kernel gram
    matrix can be very time-consuming. Instead, this property can be approximated by
    various methods. :class:`RegularisedInverseApproximator` is the base class
    for implementing these approximation methods.

    :param random_key: Key for random number generation
    """

    def __init__(self, random_key: KeyArrayLike):
        """Define approximator to the kernel matrix inverse."""
        # Assign inputs
        self.random_key = random_key

    @abstractmethod
    def approximate(
        self,
        kernel_gramian: Array,
        regularisation_parameter: float,
        identity: Array,
    ) -> Array:
        r"""
        Approximate regularised kernel matrix inverse.

        .. note::
            The function is designed to invert blocked "kernel matrices" where only the
            top-left block contains non-zero elements. We return a block array, the same
            size as the input array, where each block has only zero elements except
            for the top-left block, which is the inverse of the non-zero input block.
            The most efficient way to compute this in JAX requires the 'identity' array
            to be a matrix of zeros except for ones on the diagonal up to the dimension
            of the non-zero block.

        :param kernel_gramian: Original :math:`n \times n` kernel gram matrix
        :param regularisation_parameter: Regularisation parameter for stable inversion
            of array, negative values will be converted to positive
        :param identity: Block identity matrix
        :return: Approximation of the kernel matrix inverse
        """

    def _map_approximate(self) -> Callable[[Array, float, Array], Array]:
        """Define helper function to map approximate over horizontal array stack."""
        return jax.vmap(self.approximate, in_axes=(0, None, None))

    def approximate_stack(
        self, kernel_gramians: Array, regularisation_parameter: float, identity: Array
    ) -> Array:
        r"""
        Approximate the regularised inverses of a horizontal stack of kernel matrices.

        :param kernel_gramian: Horizontal Stack of :math:`n \times n` kernel gram
            matrices
        :param regularisation_parameter: Regularisation parameter for stable inversion
            of array, negative values will be converted to positive
        :param identity: Block identity matrix
        :return: Approximation of the kernel matrix inverses
        """
        return self._map_approximate()(
            kernel_gramians, regularisation_parameter, identity
        )


def randomised_eigendecomposition(
    random_key: KeyArrayLike,
    array: Array,
    oversampling_parameter: int = 10,
    power_iterations: int = 1,
) -> tuple[Array, Array]:
    r"""
    Approximate the eigendecomposition of square matrices.

    Using (:cite:`halko2009randomness` Algorithm 4.4. and 5.3) we approximate the
    eigendecomposition of a matrix. The parameters 'oversampling_parameter'
    and 'power_iterations' present a trade-off between speed and approximation quality.

    Given the gram matrix :math:`K \in \mathbb{R}^{n\times n} and
    :math:`r=`oversampling_parameter we return a diagonal array of eigenvalues
    :math:`\Lambda \in \mathbb{R}^{r \times r}` and a rectangular array of eigenvectors
    :math:`U\in\mathbb{R}^{n\times r}` such that we have :math:`K \approx U\Lambda U^T`.

    :param random_key: Key for random number generation
    :param array: Array to be decomposed
    :param oversampling_parameter: Number of random columns to sample; the larger the
        oversampling_parameter, the more accurate, but slower the method will be
    :param power_iterations: Number of power iterations to do; the larger the
        power_iterations, the more accurate, but slower the method will be
    :return: Eigenvalues and eigenvectors that approximately decompose the target array
    """
    # Input handling
    supported_array_shape = 2
    if len(array.shape) != supported_array_shape:
        raise ValueError("'array' must be two-dimensional")
    if array.shape[0] != array.shape[1]:
        raise ValueError("'array' must be square")
    if (oversampling_parameter <= 0.0) or not isinstance(oversampling_parameter, int):
        raise ValueError("'oversampling_parameter' must be a positive integer")
    if (power_iterations <= 0.0) or not isinstance(power_iterations, int):
        raise ValueError("'power_iterations' must be a positive integer")

    standard_gaussian_draws = jr.normal(
        random_key, shape=(array.shape[0], oversampling_parameter)
    )

    # QR decomposition to find orthonormal array with range approximating range of array
    approximate_range = array @ standard_gaussian_draws
    q, _ = jnp.linalg.qr(approximate_range)

    # Power iterations for improved accuracy
    for _ in range(power_iterations):
        approximate_range_ = array.T @ q
        q_, _ = jnp.linalg.qr(approximate_range_)
        approximate_range = array @ q_
        q, _ = jnp.linalg.qr(approximate_range)

    # Form the low rank array, compute its exact eigendecomposition and ortho-normalise
    # the eigenvectors.
    array_approximation = q.T @ array @ q
    approximate_eigenvalues, eigenvectors = jnp.linalg.eigh(array_approximation)
    approximate_eigenvectors = q @ eigenvectors

    return approximate_eigenvalues, approximate_eigenvectors


class RandomisedEigendecompositionApproximator(RegularisedInverseApproximator):
    """
    Approximate inverse of regularised gramian using its randomised eigendecomposition.

    When a dataset is very large, computing the regularised inverse of the kernel gram
    matrix can be very time-consuming. Instead, this property can be approximated by
    various methods. :class:`RandomisedEigendecompositionApproximator` is a class that
    does such an approximation using a randomised eigendecomposition. Further details
    can be found in (:cite:`halko2009randomness` Algorithm 4.4. and 5.3).

    :param random_key: Key for random number generation
    :param oversampling_parameter: Number of random columns to sample; the larger the
        oversampling_parameter, the more accurate, but slower the method will be
    :param power_iterations: Number of power iterations to do; the larger the
        power_iterations, the more accurate, but slower the method will be
    :param rcond: Cut-off ratio for small singular values of the kernel gramian. For the
        purposes of rank determination, singular values are treated as zero if they are
        smaller than rcond times the largest singular value of a. The default value of
        None will use the machine precision multiplied by the largest dimension of
        the array. An alternate value of -1 will use machine precision.

    """

    def __init__(
        self,
        random_key: KeyArrayLike,
        oversampling_parameter: int = 10,
        power_iterations: int = 1,
        rcond: Union[float, None] = None,
    ):
        """Initialise RandomisedEigendecompositionApproximator and validate input."""
        # Initialise parent
        super().__init__(random_key=random_key)
        self.oversampling_parameter = oversampling_parameter
        self.power_iterations = power_iterations
        self.rcond = rcond

        # Check attributes are valid
        if self.rcond is not None:
            if self.rcond < 0 and self.rcond != -1:
                raise ValueError("'rcond' must be non-negative, except for value of -1")

    def approximate(
        self,
        kernel_gramian: Array,
        regularisation_parameter: float,
        identity: Array,
    ) -> Array:
        r"""
        Compute approximate kernel matrix inverse using randomised eigendecomposition.

        We consider a :math:`n \times n` regularised kernel matrix, and use
        (:cite:`halko2009randomness` Algorithm 4.4. and 5.3) to approximate its
        eigendecomposition, and using this, its inverse.

        :param kernel_gramian: Original :math:`n \times n` kernel gram matrix
        :param regularisation_parameter: Regularisation parameter for stable inversion
            of array, negative values will be converted to positive
        :param identity: Identity matrix
        :return: Approximation of the kernel matrix inverse
        """
        # Validate inputs
        if kernel_gramian.shape != identity.shape:
            raise ValueError("Leading dimensions of 'array' and 'identity' must match")

        # Set rcond parameter if not given
        num_rows = kernel_gramian.shape[0]
        machine_precision = jnp.finfo(kernel_gramian.dtype).eps
        if self.rcond is None:
            rcond = machine_precision * num_rows
        elif self.rcond == -1:
            rcond = machine_precision
        else:
            rcond = self.rcond

        # Get randomised eigendecomposition of regularised kernel matrix
        approximate_eigenvalues, approximate_eigenvectors = (
            randomised_eigendecomposition(
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
