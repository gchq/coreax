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
of the data, such as :meth:`~coreax.kernels.ScalarValuedKernel.gramian_row_mean`, can
become prohibitively expensive. To reduce this computational cost, such methods can
instead be approximated (providing suitable approximation error can be achieved).

The :class:`ApproximateKernel`\ s in this module provide the functionality required to
override specific methods of a ``base_kernel`` with their approximate counterparts.
Because :class:`ApproximateKernel`\ s inherit from
:class:`~coreax.kernels.ScalarValuedKernel`, with all functionality provided through
composition with a ``base_kernel``, they can be freely used in any place where a
standard :class:`~coreax.kernels.ScalarValuedKernel` is expected.
"""

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, Union

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Shaped
from typing_extensions import Literal, override

from coreax.data import Data, _atleast_2d_consistent
from coreax.kernels import UniCompositeKernel
from coreax.util import KeyArrayLike

if TYPE_CHECKING:
    from coreax.kernels import ScalarValuedKernel  # noqa: F401


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
    data: Shaped[Array, " n p"],
    features: Shaped[Array, " n n"],
    num_indices: int,
    target_map: Callable[[Shaped[Array, " n p"]], Shaped[Array, " n p"]] = lambda x: x,
) -> Shaped[Array, " n p"]:
    r"""
    Solve the least-square problem on a random subset of the system.

    A linear system :math:`AX = B`, solved via least-squares as :math:`X = A^+ B`, can
    be approximated by random least-square as `X \approx \hat{X} = \hat{A}^+ \hat{B}`,
    where
    :math:`\hat{A} = A_{i\cdot}\ \text{and}\ \hat{B} = B_{i\cdot}\, \forall i \in I]`.
    :math:`I` is a random subset of indices for the original system of equations.

    :param key: RNG key for seeding the random selection
    :param data: The data :math:`Z \in \mathbb{R}^{n \times p}`; yields
        :math:`B \in \mathbb{R}^{n \times p}` when pushed through the target map
    :param features: The feature matrix :math:`A \in \mathbb{R}^{n \times n}`
    :param num_indices: The size of the random subset of indices :math:`I`
    :param target_map: The target map :math:`\phi` which defines :math:`b := \phi(z)`,
        where :math:`z` is the input ``data``
    :return: The push-forward of the approximate solution :math:`A\hat{X}`
    """
    num_data_points = len(data)
    train_idx = _random_indices(key, num_data_points, num_indices, mode="train")
    target = target_map(data[train_idx])
    approximate_solution, _, _, _ = jnp.linalg.lstsq(features[train_idx], target)
    return features @ approximate_solution


class ApproximateKernel(UniCompositeKernel):
    """
    Base class for approximated kernels.

    Provides approximations of the methods in the ``base_kernel``.

    The :meth:`~coreax.kernels.ScalarValuedKernel.gramian_row_mean` method is
    particularly amenable to approximation, with significant performance improvements
    possible depending on the acceptable levels of error.

    :param base_kernel: a :class:`~coreax.kernels.ScalarValuedKernel` whose
        attributes/methods are to be approximated
    """

    @override
    def compute_elementwise(self, x, y):
        return self.base_kernel.compute_elementwise(x, y)

    @override
    def grad_x_elementwise(self, x, y):
        return self.base_kernel.grad_x_elementwise(x, y)

    @override
    def grad_y_elementwise(self, x, y):
        return self.base_kernel.grad_y_elementwise(x, y)

    @override
    def divergence_x_grad_y_elementwise(self, x, y):
        return self.base_kernel.divergence_x_grad_y_elementwise(x, y)


class RandomRegressionKernel(ApproximateKernel):
    """
    An approximate kernel that requires the attributes for random regression.

    :param base_kernel: a :class:`~coreax.kernels.ScalarValuedKernel` whose
        attributes/methods are to be approximated
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

    :param base_kernel: a :class:`~coreax.kernels.ScalarValuedKernel` whose
        attributes/methods are to be approximated
    :param random_key: Key for random number generation
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    """

    def gramian_row_mean(
        self,
        x: Union[
            Shaped[Array, " n d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            Data,
        ],
        **kwargs: Any,
    ) -> Shaped[Array, " n"]:
        r"""
        Approximate the Gramian row-mean by Monte-Carlo sampling.

        A uniform random subset of ``x`` is used to approximate the base kernel's
        Gramian row-mean.

        :param x: Data matrix, :math:`n \times d`
        :return: Approximation of the base kernel's Gramian row-mean
        """
        del kwargs
        # This method does not support weighted computation of the mean, therefore
        # we need to handle the case where `x` is passed as a `Data` instance
        if isinstance(x, Data):
            x = x.data
        x = _atleast_2d_consistent(x)

        num_data_points = len(x)
        key = self.random_key
        features_idx = _random_indices(key, num_data_points, self.num_kernel_points - 1)
        features = self.base_kernel.compute(x, x[features_idx])
        return _random_least_squares(
            key,
            x,
            features,
            self.num_train_points,
            partial(self.base_kernel.compute_mean, x, axis=0),
        )


class ANNchorApproximateKernel(RandomRegressionKernel):
    r"""
    Approximate a base kernel via random kernel regression on ANNchor selected points.

    Only the base kernel's Gramian row-mean is approximated here, all other methods are
    inherited directly from the ``base_kernel``.

    :param base_kernel: a :class:`~coreax.kernels.ScalarValuedKernel` whose
        attributes/methods are to be approximated
    :param random_key: Key for random number generation
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    """

    def gramian_row_mean(
        self,
        x: Union[
            Shaped[Array, " n d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            Data,
        ],
        **kwargs: Any,
    ) -> Shaped[Array, " n"]:
        r"""
        Approximate the Gramian row-mean by random regression on ANNchor points.

        A subset of ``x`` is selected via the ANNchor approach and random kernel
        regression used to approximate the base kernel's Gramian row-mean. The ANNchor
        implementation used can be found `here <https://github.com/gchq/annchor>`_.

        :param x: Data matrix, :math:`n \times d`
        :return: Approximation of the base kernel's Gramian row-mean
        """
        del kwargs
        # This method does not support weighted computation of the mean, therefore
        # we need to handle the case where `x` is passed as a `Data` instance
        if isinstance(x, Data):
            x = x.data
        x = _atleast_2d_consistent(x)

        num_data_points = len(x)
        features = jnp.zeros((num_data_points, self.num_kernel_points))
        features = features.at[:, 0].set(self.base_kernel.compute(x, x[0])[:, 0])

        def _annchor_body(
            idx: int, _features: Shaped[Array, " n num_kernel_points"]
        ) -> Shaped[Array, " n num_kernel_points"]:
            r"""
            Execute main loop of the ANNchor construction.

            :param idx: Loop counter
            :param _features: Loop variables to be updated
            :return: Updated loop variables ``features``
            """
            max_entry = _features.max(axis=1).argmin()
            _features = _features.at[:, idx].set(
                self.base_kernel.compute(x, x[max_entry])[:, 0]
            )
            return _features

        features = jax.lax.fori_loop(1, self.num_kernel_points, _annchor_body, features)
        return _random_least_squares(
            self.random_key,
            x,
            features,
            self.num_train_points,
            partial(self.base_kernel.compute_mean, x, axis=0),
        )


class NystromApproximateKernel(RandomRegressionKernel):
    """
    Approximate a base kernel via Nystrom approximation.

    Only the base kernel's Gramian row-mean is approximated here, all other methods
    are inherited directly from the ``base_kernel``.

    :param base_kernel: a :class:`~coreax.kernels.ScalarValuedKernel` whose
        attributes/methods are to be approximated
    :param random_key: Key for random number generation
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    """

    def gramian_row_mean(
        self,
        x: Union[
            Shaped[Array, " n d"],
            Shaped[Array, " d"],
            Shaped[Array, ""],
            float,
            int,
            Data,
        ],
        **kwargs: Any,
    ) -> Shaped[Array, " n"]:
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
        # This method does not support weighted computation of the mean, therefore
        # we need to handle the case where `x` is passed as a `Data` instance
        if isinstance(x, Data):
            x = x.data
        x = _atleast_2d_consistent(x)

        num_data_points = len(x)
        feature_idx = _random_indices(
            self.random_key, num_data_points, self.num_kernel_points
        )
        features = self.base_kernel.compute(x, x[feature_idx])
        return _random_least_squares(
            self.random_key,  # intentional key reuse to ensure train_idx = feature_idx
            x,
            features,
            self.num_train_points,
            self.base_kernel.gramian_row_mean,
        )
