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
:math:`k(x,y) = \text{output_scale}\exp(-||x-y||^2/2 * \text{length_scale}^2)`.
For simplicity, we set ``length_scale`` to :math:`1.0/np.sqrt(2)`
and ``output_scale`` to 1.0.

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

from abc import ABC, abstractmethod
from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax, random
from jax.typing import ArrayLike

import coreax.kernel as ck
from coreax.util import ClassFactory, KernelFunction
from coreax.validation import cast_as_type, validate_in_range, validate_is_instance


class KernelMeanApproximator(ABC):
    """
    Base class for approximation methods to kernel row sum means.

    When a dataset is very large, computing the mean distance between a given point
    and all other points can be time-consuming. Instead, this property can be
    approximated by various methods. :class:`KernelMeanApproximator` is the base class
    for implementing these approximation methods.

    :param kernel: A :class:`~coreax.kernel.Kernel` object
    :param random_key: Key for random number generation
    :param num_kernel_points: Number of kernel evaluation points
    """

    def __init__(
        self,
        kernel: "ck.Kernel",
        random_key: random.PRNGKeyArray = random.PRNGKey(0),
        num_kernel_points: int = 10_000,
    ):
        """
        Define an approximator to the mean of the row sum of a kernel distance matrix.
        """
        # Validate inputs of coreax defined classes
        validate_is_instance(kernel, "kernel", ck.Kernel)

        # Validate inputs of non-coreax defined classes
        random_key = cast_as_type(
            x=random_key, object_name="random_key", type_caster=jnp.asarray
        )
        num_kernel_points = cast_as_type(
            x=num_kernel_points, object_name="num_kernel_points", type_caster=int
        )

        # Validate inputs lie within accepted ranges
        validate_in_range(
            x=num_kernel_points,
            object_name="num_kernel_points",
            strict_inequalities=True,
            lower_bound=0,
        )

        # Assign inputs
        self.kernel = kernel
        self.random_key = random_key
        self.num_kernel_points = num_kernel_points

    @abstractmethod
    def approximate(self, data: ArrayLike) -> Array:
        r"""
        Approximate kernel matrix row sum mean.

        :param data: The original :math:`n \times d` data
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

    :param kernel: A :class:`~coreax.kernel.Kernel` object
    :param random_key: Key for random number generation
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    """

    def __init__(
        self,
        kernel: "ck.Kernel",
        random_key: random.PRNGKeyArray = random.PRNGKey(0),
        num_kernel_points: int = 10_000,
        num_train_points: int = 10_000,
    ):
        """
        Approximate kernel row mean by regression on points selected randomly.
        """
        # Validate inputs of non-coreax defined classes
        num_train_points = cast_as_type(
            x=num_train_points, object_name="num_train_points", type_caster=int
        )

        # Validate inputs lie within accepted ranges
        validate_in_range(
            x=num_train_points,
            object_name="num_train_points",
            strict_inequalities=True,
            lower_bound=0,
        )

        # Assign inputs
        self.num_train_points = num_train_points

        # Initialise parent
        super().__init__(
            kernel=kernel,
            random_key=random_key,
            num_kernel_points=num_kernel_points,
        )

    def approximate(
        self,
        data: ArrayLike,
    ) -> Array:
        r"""
        Compute approximate kernel row mean by regression on randomly selected points.

        :param data: The original :math:`n \times d` data
        :return: Approximation of the kernel matrix row sum divided by the number of
            data points in the dataset
        """
        # Validate inputs
        data = cast_as_type(x=data, object_name="data", type_caster=jnp.atleast_2d)

        # Record dataset size
        num_data_points = len(data)

        # Randomly select points for kernel regression
        key, subkey = random.split(self.random_key)
        features_idx = random.choice(
            subkey, num_data_points, (self.num_kernel_points,), replace=False
        )
        features = self.kernel.compute(data, data[features_idx])

        # Select training points
        train_idx = random.choice(
            key, num_data_points, (self.num_train_points,), replace=False
        )
        target = (
            self.kernel.compute(data[train_idx], data).sum(axis=1) / num_data_points
        )

        # Solve regression problem.
        params, _, _, _ = jnp.linalg.lstsq(features[train_idx], target)

        return features @ params


class ANNchorApproximator(KernelMeanApproximator):
    """
    Approximation method to kernel mean through regression on ANNchor selected points.

    When a dataset is very large, computing the mean distance between a given point
    and all other points can be time-consuming. Instead, this property can be
    approximated by various methods. :class:`ANNchorApproximator` is a class that does
    such an approximation using kernel regression on a subset of points selected via the
    ANNchor approach from the dataset. The ANNchor implementation used can be found
    `here<https://github.com/gchq/annchor>`_.

    :param kernel: A :class:`~coreax.kernel.Kernel` object
    :param random_key: Key for random number generation
    :param num_kernel_points: Number of kernel evaluation points
    :param num_train_points: Number of training points used to fit kernel regression
    """

    def __init__(
        self,
        kernel: "ck.Kernel",
        random_key: random.PRNGKeyArray = random.PRNGKey(0),
        num_kernel_points: int = 10_000,
        num_train_points: int = 10_000,
    ):
        """
        Approximate kernel row mean by regression on ANNchor selected points.
        """
        # Validate inputs of non-coreax defined classes
        num_train_points = cast_as_type(
            x=num_train_points, object_name="num_train_points", type_caster=int
        )

        # Validate inputs lie within accepted ranges
        validate_in_range(
            x=num_train_points,
            object_name="num_train_points",
            strict_inequalities=True,
            lower_bound=0,
        )

        # Assign inputs
        self.num_train_points = num_train_points

        # Initialise parent
        super().__init__(
            kernel=kernel,
            random_key=random_key,
            num_kernel_points=num_kernel_points,
        )

    def approximate(
        self,
        data: ArrayLike,
    ) -> Array:
        r"""
        Compute approximate kernel row mean by regression on ANNchor selected points.

        :param data: The original :math:`n \times d` data
        :return: Approximation of the kernel matrix row sum divided by the number of
            data points in the dataset
        """
        # Validate inputs
        data = cast_as_type(x=data, object_name="data", type_caster=jnp.atleast_2d)

        # Record dataset size
        num_data_points = len(data)

        # Select point for kernel regression using ANNchor construction
        features = jnp.zeros((num_data_points, self.num_kernel_points))

        features = features.at[:, 0].set(self.kernel.compute(data, data[0])[:, 0])
        body = partial(_anchor_body, data=data, kernel_function=self.kernel.compute)
        features = lax.fori_loop(1, self.num_kernel_points, body, features)

        train_idx = random.choice(
            self.random_key, num_data_points, (self.num_train_points,), replace=False
        )
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
    found here [chatalic2022nystrom]_.

    :param kernel: A :class:`~coreax.kernel.Kernel` object
    :param random_key: Key for random number generation
    :param num_kernel_points: Number of kernel evaluation points
    """

    def __init__(
        self,
        kernel: "ck.Kernel",
        random_key: random.PRNGKeyArray = random.PRNGKey(0),
        num_kernel_points: int = 10_000,
    ):
        r"""
        Approximate kernel row mean by using Nystrom approximation.
        """
        # Initialise parent
        super().__init__(
            kernel=kernel,
            random_key=random_key,
            num_kernel_points=num_kernel_points,
        )

    def approximate(
        self,
        data: ArrayLike,
    ) -> Array:
        r"""
        Compute approximate kernel row mean by regression on ANNchor selected points.

        :param data: The original :math:`n \times d` data
        :return: Approximation of the kernel matrix row sum divided by the number of
            data points in the dataset
        """
        # Validate inputs
        data = cast_as_type(x=data, object_name="data", type_caster=jnp.atleast_2d)

        # Record dataset size
        num_data_points = len(data)

        # Randomly select points for kernel regression
        sample_points = random.choice(
            self.random_key, num_data_points, (self.num_kernel_points,)
        )

        # Solve for kernel distances
        k_mn = self.kernel.compute(data[sample_points], data)
        k_mm = self.kernel.compute(data[sample_points], data[sample_points])
        alpha = (jnp.linalg.pinv(k_mm) @ k_mn).sum(axis=1) / num_data_points

        return k_mn.T @ alpha


@partial(jit, static_argnames=["kernel_function"])
def _anchor_body(
    idx: int,
    features: ArrayLike,
    data: ArrayLike,
    kernel_function: KernelFunction,
) -> Array:
    r"""
    Execute main loop of the ANNchor construction.

    :param idx: Loop counter
    :param features: Loop updateables
    :param data: Original :math:`n \times d` dataset
    :param kernel_function: Vectorised kernel function on pairs ``(X,x)``:
        :math:`k: \mathbb{R}^{n \times d} \times \mathbb{R}^d \rightarrow \mathbb{R}^n`
    :return: Updated loop variables `features`
    """
    # Validate inputs
    features = cast_as_type(
        x=features, object_name="features", type_caster=jnp.atleast_2d
    )
    data = cast_as_type(x=data, object_name="data", type_caster=jnp.atleast_2d)

    max_entry = features.max(axis=1).argmin()
    features = features.at[:, idx].set(kernel_function(data, data[max_entry])[:, 0])

    return features


# Set up class factory
approximator_factory = ClassFactory(KernelMeanApproximator)
approximator_factory.register("random", RandomApproximator)
approximator_factory.register("annchor", ANNchorApproximator)
approximator_factory.register("nystrom", NystromApproximator)
