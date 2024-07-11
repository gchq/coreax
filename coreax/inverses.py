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
Classes and associated functionality to approximate regularised inverses of matrices.

Given a matrix :math:`A\in\mathbb{R}^{n\times n}` and some regularisation parameter
:math:`\lambda\in\mathbb{R}_{\ge 0}`, the regularised inverse of :math:`A` is
:math:`(A + \lambda I_n)^{-1}` where :math:`I_n` is the :math:`n\times n` identity
matrix.

Coreset algorithms which use the regularised inverse of large kernel matrices can become
prohibitively expensive as the data size increases. To reduce this computational cost,
this regularised inverse can be approximated by various methods, depending on the
acceptable levels of error.

The :class:`RegularisedInverseApproximator` in this module provides the functionality
required to approximate the regularised inverse of matrices. Furthermore, this class
allows for "block-inversion" where given an invertible matrix :math:`B`, the block
array

.. math::
    A = \begin{bmatrix}B & 0 & \dots & 0 \\ 0 & 0 & \dots & 0 \\
         \vdots & \ddots & \dots & \vdots \\ 0 & 0 & \dots & 0\end{bmatrix},

where only the top-left block contains non-zero elements can be "inverted" to give

.. math::
    A^{-1} := \begin{bmatrix}B^{-1} & 0 & \dots & 0 \\ 0 & 0 & \dots & 0 \\
         \vdots & \ddots & \dots & \vdots \\ 0 & 0 & \dots & 0\end{bmatrix}.

This functionality allows iterative coreset algorithms which require inverting growing
arrays to have fully static array shapes and thus be JIT-compatible.

To compute these "inverses" in JAX, we require the `identity` array
passed in :meth:`~coreax.inverses.RegularisedInverseApproximator.approximate` to be a
matrix of zeros except for ones on the diagonal up to the dimension of the non-zero
block.
"""

from abc import abstractmethod
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Shaped
from typing_extensions import override

from coreax.util import KeyArrayLike


class RegularisedInverseApproximator(eqx.Module):
    """
    Base class for methods which approximate the regularised inverse of an array.

    Computing the regularised inverse of large arrays can become prohibitively expensive
    as size increases. To reduce this computational cost, this quantity can be
    approximated, depending on the acceptable levels of error.

    :param random_key: Key for random number generation
    """

    random_key: KeyArrayLike

    @abstractmethod
    def approximate(
        self,
        array: Shaped[Array, " n n"],
        regularisation_parameter: float,
        identity: Shaped[Array, " n n"],
    ) -> Shaped[Array, " n n"]:
        r"""
        Approximate the regularised inverse of an array.

        .. note::
            The function is designed to invert blocked arrays where only the
            top-left block contains non-zero elements. We return a block array, the same
            size as the input array, where each block has only zero elements except
            for the top-left block, which is the inverse of the non-zero input block.
            To compute these "inverses" in JAX, we require the `identity` array
            to be a matrix of zeros except for ones on the diagonal up to the dimension
            of the non-zero block.

        :param array: :math:`n \times n` array
        :param regularisation_parameter: Regularisation parameter for stable inversion
            of array, negative values will be converted to positive
        :param identity: Block :math:`n \times n` identity matrix
        :return: Approximation of the kernel matrix inverse
        """

    def approximate_stack(
        self,
        arrays: Shaped[Array, "m n n"],
        regularisation_parameter: float,
        identity: Shaped[Array, " n n"],
    ) -> Shaped[Array, "m n n"]:
        r"""
        Approximate the regularised inverses of a horizontal stack of kernel matrices.

        :param array: Horizontal stack of :math:`m` :math:`n \times n` arrays
        :param regularisation_parameter: Regularisation parameter for stable inversion
            of array, negative values will be converted to positive
        :param identity: Block identity matrix
        :return: Approximation of the kernel matrix inverses
        """
        _map_approximate = jax.vmap(self.approximate, in_axes=(0, None, None))
        return _map_approximate(arrays, regularisation_parameter, identity)


class LeastSquareApproximator(RegularisedInverseApproximator):
    """
    Approximate the regularised inverse of an array by solving a least-squares problem.

    Note that this approximator does not give time savings and instead acts as a
    default option useful for comparing other approximators to.

    :param random_key: Key for random number generation
    :param rcond: Cut-off ratio for small singular values of 'array'. For the purposes
        of rank determination, singular values are treated as zero if they are smaller
        than rcond times the largest singular value of 'array'. The default value of
        None will use the machine precision multiplied by the largest dimension of the
        array. An alternate value of -1 will use machine precision.
    """

    rcond: Optional[float] = None

    @override
    def approximate(
        self,
        array: Shaped[Array, " n n"],
        regularisation_parameter: float,
        identity: Shaped[Array, " n n"],
    ) -> Shaped[Array, " n n"]:
        try:
            return jnp.linalg.lstsq(
                array + abs(regularisation_parameter) * identity,
                identity,
                rcond=self.rcond,
            )[0]
        except Exception as err:
            if array.shape != identity.shape:
                raise ValueError(
                    "Leading dimensions of 'array' and 'identity' must match"
                ) from err
            raise


def randomised_eigendecomposition(
    random_key: KeyArrayLike,
    array: Shaped[Array, " n n"],
    oversampling_parameter: int = 25,
    power_iterations: int = 1,
) -> tuple[Shaped[Array, " r r"], Shaped[Array, " n r"]]:
    r"""
    Approximate the eigendecomposition of Hermitian matrices.

    Using Algorithm 4.4. and 5.3 from :cite:`halko2009randomness` we approximate the
    eigendecomposition of a matrix. The parameters `oversampling_parameter`
    and `power_iterations` present a trade-off between speed and approximation quality.
    See :cite:`halko2009randomness` for discussion on choosing sensible parameters, the
    defaults chosen here are cautious.

    Given the matrix :math:`A \in \mathbb{R}^{n\times n}` and
    :math:`r=`oversampling_parameter` we return a diagonal array of eigenvalues
    :math:`\Lambda \in \mathbb{R}^{r \times r}` and a rectangular array of eigenvectors
    :math:`U\in\mathbb{R}^{n\times r}` such that we have :math:`A \approx U\Lambda U^T`.

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
    Approximate regularised inverse of a Hermitian array via random eigendecomposition.

    Computing the regularised inverse of large arrays can become prohibitively expensive
    as size increases. To reduce this computational cost, this quantity can be
    approximated by various methods. :class:`RandomisedEigendecompositionApproximator`
    is a class that does such an approximation using the randomised eigendecomposition
    of the input array.

    Using Algorithm 4.4. and 5.3 from :cite:`halko2009randomness` we approximate the
    eigendecomposition of a matrix. The parameters `oversampling_parameter`
    and `power_iterations` present a trade-off between speed and approximation quality.
    See :cite:`halko2009randomness` for discussion on choosing sensible parameters, the
    defaults chosen here are cautious.

    :param random_key: Key for random number generation
    :param oversampling_parameter: Number of random columns to sample; the larger the
        oversampling_parameter, the more accurate, but slower the method will be
    :param power_iterations: Number of power iterations to do; the larger the
        power_iterations, the more accurate, but slower the method will be
    :param rcond: Cut-off ratio for small singular values of the `array`. For the
        purposes of rank determination, singular values are treated as zero if they are
        smaller than rcond times the largest singular value of a. The default value of
        None will use the machine precision multiplied by the largest dimension of
        the array. An alternate value of -1 will use machine precision.

    """

    oversampling_parameter: int = 25
    power_iterations: int = 1
    rcond: Optional[float] = None

    def __check_init__(self):
        """Validate rcond input."""
        if self.rcond is not None:
            if self.rcond < 0 and self.rcond != -1:
                raise ValueError("'rcond' must be non-negative, except for value of -1")

    @override
    def approximate(
        self,
        array: Shaped[Array, " n n"],
        regularisation_parameter: float,
        identity: Shaped[Array, " n n"],
    ) -> Shaped[Array, " n n"]:
        # Validate inputs
        if array.shape != identity.shape:
            raise ValueError("Leading dimensions of 'array' and 'identity' must match")

        # Set rcond parameter if not given
        num_rows = array.shape[0]
        machine_precision = jnp.finfo(array.dtype).eps
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
                array=array + abs(regularisation_parameter) * identity,
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
