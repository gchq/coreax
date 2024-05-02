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
of the data, such as :meth:`~coreax.kernel.Kernel.calculate_kernel_matrix_row_sum_mean`,
can become prohibitively expensive. To reduce this computational cost, such methods can
instead be approximated (providing suitable approximation error can be achieved).

The :class:`ApproximateKernel`\ s in this module provide the functionality required to
override specific methods of a ``base_kernel`` with their approximate counterparts.
Because :class:`ApproximateKernel`\ s inherit from :class:`~coreax.kernel.Kernel`, with
all functionality provided through composition with a ``base_kernel``, they can be
freely used in any place where a standard :class:`~coreax.kernel.Kernel` is expected.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.typing import ArrayLike
from typing_extensions import TYPE_CHECKING, Literal, override

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

    The :meth:`~coreax.kernel.Kernel.calculate_kernel_matrix_row_sum` method is
    particularly amenable to approximation, with significant performance improvements
    possible depending on the acceptable levels of error.

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

    Only the kernel matrix row sum mean is approximated here, all other methods
    are inherited directly from the ``base_kernel``.

    :param base_kernel: a :class:`~coreax.kernel.Kernel` whose attributes/methods are
        to be approximated
    :param random_key: Key for random number generation
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    """

    def calculate_kernel_matrix_row_sum_mean(
        self, x: ArrayLike, max_size: int = 10_000
    ) -> Array:
        r"""
        Calculate the kernel matrix row sum mean by Monte Carlo approximation.

        A uniform random subset of ``x`` is used to approximate the kernel matrix row
        sum mean.

        :param x: Data matrix, :math:`n \times d`
        :param max_size: Size of matrix block to process
        :return: Approximation of the kernel matrix row sum
        """
        data = jnp.atleast_2d(x)
        num_data_points = len(data)
        key = self.random_key
        features_idx = _random_indices(key, num_data_points, self.num_kernel_points - 1)
        features = self.base_kernel.compute(data, data[features_idx])
        return _random_least_squares(
            key,
            data,
            features,
            self.num_train_points,
            lambda x: self.base_kernel.compute(x, data).sum(axis=1) / num_data_points,
        )


class ANNchorApproximateKernel(RandomRegressionKernel):
    r"""
    Approximate a base kernel via random kernel regression on ANNchor selected points.

    Only the kernel matrix row sum mean is approximated here, all other methods
    are inherited directly from the ``base_kernel``.

    :param base_kernel: a :class:`~coreax.kernel.Kernel` whose attributes/methods are
        to be approximated
    :param random_key: Key for random number generation
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    """

    def calculate_kernel_matrix_row_sum_mean(
        self, x: ArrayLike, max_size: int = 10_000
    ) -> Array:
        r"""
        Calculate the kernel matrix row sum mean by random regression on ANNchor points.

        A subset of ``x`` is selected via the ANNchor approach and random kernel
        regression used to approximate the kernel matrix row sum mean. The ANNchor
        implementation used can be found `here <https://github.com/gchq/annchor>`_.

        :param x: Data matrix, :math:`n \times d`
        :param max_size: Size of matrix block to process
        :return: Approximation of the kernel matrix row sum
        """
        data = jnp.atleast_2d(x)
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
            lambda x: self.base_kernel.compute(x, data).mean(axis=1),
        )


class NystromApproximateKernel(RandomRegressionKernel):
    """
    Approximate a base kernel via Nystrom approximation.

    Only the kernel matrix row sum mean is approximated here, all other methods
    are inherited directly from the ``base_kernel``.

    :param base_kernel: a :class:`~coreax.kernel.Kernel` whose attributes/methods are
        to be approximated
    :param random_key: Key for random number generation
    :param kernel: A :class:`~coreax.kernel.Kernel` object
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    """

    def calculate_kernel_matrix_row_sum_mean(
        self, x: ArrayLike, max_size: int = 10_000
    ) -> Array:
        r"""
        Calculate the kernel matrix row sum mean by Nystrom approximation.

        We consider a :math:`n \times d` dataset, and wish to use an :math:`m \times d`
        subset of this to approximate the kernel matrix row sum mean. The ``m`` points
        are selected uniformly at random, and the Nystrom estimator, as defined in
        :cite:`chatalic2022nystrom` is computed using this subset.

        :param x: Data matrix, :math:`n \times d`
        :param max_size: Size of matrix block to process
        :return: Approximation of the kernel matrix row sum
        """
        data = jnp.atleast_2d(x)
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
            lambda x: self.base_kernel.compute(x, x).sum(axis=1) / num_data_points,
        )
