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
Classes to approximate least-squares solution to regularised linear matrix equations.

Primarily used in Coreax to approximate regularised matrix inverses.

Given a matrix :math:`A\in\mathbb{R}^{n\times n}` and some regularisation parameter
:math:`\lambda\in\mathbb{R}_{\ge 0}`, the regularised inverse of :math:`A` is
:math:`(A + \lambda I_n)^{-1}` where :math:`I_n` is the :math:`n\times n` identity
matrix.

Coreset algorithms which use the regularised inverse of large kernel matrices can become
prohibitively expensive as the data size increases. To reduce this computational cost,
this regularised inverse can be approximated by various methods, depending on the
acceptable levels of error.

The :class:`RegularisedLeastSquaresSolver` in this module provides the functionality
required to approximate the regularised inverse of matrices. Furthermore, this class
allows for "block-inversion" where given an invertible matrix :math:`B`, the block
array

.. math::
    A = \begin{bmatrix}B & 0 & \dots & 0 \\ 0 & 0 & \dots & 0 \\
         \vdots & \ddots & \dots & \vdots \\ 0 & 0 & \dots & 0\end{bmatrix},

where only the top-left block contains non-zero elements, can be "inverted" to give

.. math::
    A^{-1} := \begin{bmatrix}B^{-1} & 0 & \dots & 0 \\ 0 & 0 & \dots & 0 \\
         \vdots & \ddots & \dots & \vdots \\ 0 & 0 & \dots & 0\end{bmatrix}.

This functionality allows for iterative coreset algorithms which require inverting
growing arrays to have fully static array shapes and thus be JIT-compatible.

To compute these "inverses" in JAX, we require the `target` and `identity` array
passed in :meth:`~coreax.least_squares.RegularisedLeastSquaresSolver.solve` to be a
matrix of zeros except for ones on the diagonal up to the dimension of the non-zero
block.
"""

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Optional, Union, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Shaped
from typing_extensions import override

from coreax.util import KeyArrayLike


class RegularisedLeastSquaresSolver(eqx.Module):
    r"""
    Base class for solving regularised linear matrix equations via least-squares.

    Given an array :math:`A \in \mathbb{R}^{n \times n}`, a regularisation parameter
    :math:`\lambda \in \mathbb{R}_{\ge 0}`, and an array of targets
    :math:`B \in \mathbb{R}^{n \times m}` the least-squares solution to the regularised
    linear equation :math:`(A + \lambda I_n)B = X` has solution
    :math:`X = (A + \lambda I_n)^{-1}B` where
    :math:`I_n \in \mathbb{R}^{n\times n}` is the identity matrix.
    """

    @abstractmethod
    def solve(
        self,
        array: Shaped[Array, " n n"],
        regularisation_parameter: float,
        target: Shaped[Array, " n m"],
        identity: Shaped[Array, " n n"],
    ) -> Shaped[Array, " n n"]:
        r"""
        Compute the least-squares solution to a regularised linear matrix equation.

        :param array: :math:`n \times n` array, corresponds to :math:`A`
        :param regularisation_parameter: Regularisation parameter for stable inversion
            of array; negative values will be converted to positive
        :param target: :math:`n \times m` array of targets, corresponds to :math:`B`
        :param identity: Identity matrix
        :return: Approximation of the regularised least-squares solution :math:`X`
        """

    def solve_stack(
        self,
        arrays: Shaped[Array, " l n n"],
        regularisation_parameter: float,
        targets: Shaped[Array, " l n m"],
        identity: Shaped[Array, " n n"],
        in_axes: Union[int, None, Sequence[Any]] = (0, None, 0, None),
    ) -> Shaped[Array, " l n n"]:
        r"""
        Compute least-squares solutions to stack of regularised linear matrix equations.

        :param arrays: Horizontal stack of arrays with shape :math:`l \times n \times n`
        :param regularisation_parameter: Regularisation parameter for stable inversion
            of ``arrays``; negative values will be converted to positive
        :param targets: Horizontal stack of targets with shape
            :math:`l \times n \times m`
        :param identity: Identity matrix
        :param in_axes: An integer, :data:`None`, or sequence of values specifying which
            array axes of parameters to :meth:`solve` to map over. See :func:`~jax.vmap`
            documentation for further information.
        :return: Approximation of the regularised least-squares solutions
        """
        _map_solve = jax.vmap(self.solve, in_axes=in_axes)
        return _map_solve(arrays, abs(regularisation_parameter), targets, identity)


class MinimalEuclideanNormSolver(RegularisedLeastSquaresSolver):
    """
    Find minimal-norm least-squares solution to the regularised linear matrix equation.

    Computes the solution that approximately solves the regularised linear matrix
    equation. The equation may be under-, well-, or over-determined. If ``array`` is
    full rank, then the solution is *exact*, up to floating-point errors. Else, the
    solution minimises the Euclidean 2-norm. If there are multiple minimising solutions,
    the one with the smallest 2-norm is returned.

    .. note::
        This solver does not give time savings and instead acts as a robust default
        option useful for comparing other solvers to.

    :param rcond: Cut-off ratio for small singular values of ``array``. For the purposes
        of rank determination, singular values are treated as zero if they are smaller
        than rcond times the largest singular value of ``array``. The default value of
        :data:`None` will use the machine precision multiplied by the largest dimension
        of the ``array``. An alternate value of -1 will use machine precision.
    """

    rcond: Optional[float] = None

    @override
    def solve(
        self,
        array: Shaped[Array, " n n"],
        regularisation_parameter: float,
        target: Shaped[Array, " n m"],
        identity: Shaped[Array, " n n"],
    ) -> Shaped[Array, " n n"]:
        return jnp.linalg.lstsq(
            array + abs(regularisation_parameter) * identity,
            target,
            rcond=self.rcond,
        )[0]


def _gaussian_range_finder(
    random_key: KeyArrayLike,
    array: Shaped[Array, " n n"],
    oversampling_parameter: int = 25,
    power_iterations: int = 1,
) -> Shaped[Array, " n oversampling_parameter"]:
    r"""
    Produce an orthonormal matrix whose range captures the action of an input ``array``.

    :param random_key: Key for random number generation
    :param array: Array :math:`A \in \mathbb{R}^{n \times n}` to be decomposed
    :param oversampling_parameter: Number of random columns to sample; the larger the
        oversampling_parameter, the more accurate, but slower the method will be
    :param power_iterations: Number of power iterations to do; the larger the
        ``power_iterations``, the more accurate, but slower the method will be
    :return: Orthonormal array capturing action of input ``array``
    """
    # Input handling
    supported_array_shape = 2
    if len(array.shape) != supported_array_shape:
        raise ValueError("'array' must be two-dimensional")
    if array.shape[0] != array.shape[1]:
        raise ValueError("'array' must be square")

    standard_gaussian_draws = jr.normal(
        random_key, shape=(array.shape[0], oversampling_parameter)
    )

    # QR decomposition to find orthonormal array with range approximating range of
    # array
    approximate_range = array @ standard_gaussian_draws
    q, _ = jnp.linalg.qr(approximate_range)

    # Power iterations for improved accuracy
    for _ in range(power_iterations):
        approximate_range_ = array.T @ q
        q_, _ = jnp.linalg.qr(approximate_range_)
        approximate_range = array @ q_
        q, _ = jnp.linalg.qr(approximate_range)
    return q


def _eigendecomposition_invert(
    eigenvalues: Shaped[Array, " n r"],
    eigenvectors: Shaped[Array, " r"],
    rcond: float,
) -> Shaped[Array, " n n"]:
    r"""
    Given an array's rank-:math:`r` eigendecomposition, return the inverse of the array.

    .. warning::
        We assume the order of the ``eigenvalues`` and ``eigenvectors`` correspond, i.e.
        the first element of ``eigenvalues`` is paired with the first column of
        ``eigenvectors``.

    :param eigenvalues: Vector of :math:`r` eigenvalues
    :param eigenvectors: :math:`n \times r` array of eigenvectors
    :param rcond: Cut-off ratio for small eigenvalues
    :return: Approximate inverse of array using its eigendecomposition
    """
    # Mask the eigenvalues that are zero or almost zero according to value of rcond
    # for safe inversion.
    mask = eigenvalues >= jnp.array(rcond) * jnp.max(eigenvalues)
    safe_eigenvalues = jnp.where(mask, eigenvalues, 1)

    # Invert the eigenvalues safely and extend array for broadcasting
    inverse_eigenvalues = jnp.where(mask, 1 / safe_eigenvalues, 0)[:, jnp.newaxis]

    # Solve Ax = I, x = A^-1 = UL^-1U^T
    return eigenvectors.dot(inverse_eigenvalues * eigenvectors.T)


class RandomisedEigendecompositionSolver(RegularisedLeastSquaresSolver):
    """
    Approximate solution to regularised linear equations via random eigendecomposition.

    Solving regularised linear equations involving large arrays can become prohibitively
    expensive as size increases. To reduce this computational cost, the solution can be
    approximated by various methods. :class:`RandomisedEigendecompositionSolver`
    is a class that does such an approximation using the randomised eigendecomposition
    of the input array.

    .. warning::
        Input arrays must be Hermitian for this method to have predictable
        behaviour. We do not check this.

    Using Algorithm 4.4. and 5.3 from :cite:`halko2009randomness` we approximate the
    eigendecomposition of a Hermitian matrix. The parameters ``oversampling_parameter``
    and ``power_iterations`` present a trade-off between speed and approximation
    quality. See :cite:`halko2009randomness` for discussion on choosing sensible
    parameters; the defaults chosen here are cautious.

    :param random_key: Key for random number generation
    :param oversampling_parameter: Number of random columns to sample; the larger the
        oversampling_parameter, the more accurate, but slower the method will be
    :param power_iterations: Number of power iterations to do; the larger the
        power_iterations, the more accurate, but slower the method will be
    :param rcond: Cut-off ratio for small singular values of the ``array``. For the
        purposes of rank determination, singular values are treated as zero if they are
        smaller than ``rcond`` times the largest singular value of a. The default value
        of :data:`None` will use the machine precision multiplied by the largest
        dimension of the ``array``. An alternate value of -1 will use machine precision.
    """

    random_key: KeyArrayLike
    oversampling_parameter: int = 25
    power_iterations: int = 1
    rcond: Optional[float] = None

    def __check_init__(self):
        """Validate inputs."""
        if (self.oversampling_parameter <= 0.0) or not isinstance(
            self.oversampling_parameter, int
        ):
            raise ValueError("'oversampling_parameter' must be a positive integer")
        if (self.power_iterations < 0.0) or not isinstance(self.power_iterations, int):
            raise ValueError("'power_iterations' must be a non-negative integer")
        if self.rcond is not None:
            if self.rcond < 0 and self.rcond != -1:
                raise ValueError("'rcond' must be non-negative or -1")

    def randomised_eigendecomposition(
        self, array: Shaped[Array, " n n"]
    ) -> tuple[
        Shaped[Array, " oversampling_parameter"],
        Shaped[Array, " n oversampling_parameter"],
    ]:
        r"""
        Approximate the eigendecomposition of Hermitian matrices.

        Using Algorithm 4.4. and 5.3 from :cite:`halko2009randomness` we approximate the
        eigendecomposition of a matrix. The parameters `oversampling_parameter` and
        `power_iterations` present a trade-off between speed and approximation quality.
        See :cite:`halko2009randomness` for discussion on choosing sensible parameters,
        the defaults chosen here are cautious.

        Given the matrix :math:`A \in \mathbb{R}^{n\times n}` and
        :math:`r=` ``oversampling_parameter``, we return a diagonal array of eigenvalues
        :math:`\Lambda \in \mathbb{R}^{r \times r}` and a rectangular array of
        eigenvectors :math:`U\in\mathbb{R}^{n\times r}` such that we have
        :math:`A \approx U\Lambda U^T`.

        :param array: Array to be decomposed
        :return: Eigenvalues and eigenvectors that approximately decompose the ``array``
        """
        # Find orthonormal array with range approximating range of array
        q = _gaussian_range_finder(
            random_key=self.random_key,
            array=array,
            oversampling_parameter=self.oversampling_parameter,
            power_iterations=self.power_iterations,
        )

        # Form the low rank array, compute its exact eigendecomposition and
        # ortho-normalise the eigenvectors.
        array_approximation = q.T @ array @ q
        approximate_eigenvalues, eigenvectors = jnp.linalg.eigh(array_approximation)
        approximate_eigenvectors = q @ eigenvectors

        return approximate_eigenvalues, approximate_eigenvectors

    @override
    def solve(
        self,
        array: Shaped[Array, " n n"],
        regularisation_parameter: float,
        target: Shaped[Array, " n m"],
        identity: Shaped[Array, " n n"],
    ) -> Shaped[Array, " n n"]:
        # Set rcond parameter if not given using array dimension
        num_rows = array.shape[0]
        machine_precision = cast(float, jnp.finfo(array.dtype).eps)
        if self.rcond is None:
            rcond = machine_precision * num_rows
        elif self.rcond == -1:
            rcond = machine_precision
        else:
            rcond = self.rcond

        # Get randomised eigendecomposition of regularised kernel matrix
        approximate_eigenvalues, approximate_eigenvectors = (
            self.randomised_eigendecomposition(
                array=array + abs(regularisation_parameter) * identity,
            )
        )

        # Solve AX = B, X = A^-1B = UL^-1U^TB
        return _eigendecomposition_invert(
            approximate_eigenvalues, approximate_eigenvectors, rcond
        ).dot(target)
