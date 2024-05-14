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

"""
Tests for kernel implementations.

The tests within this file verify that the implementations of kernels used throughout
the codebase produce the expected results on simple examples.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Literal, NamedTuple, TypeVar

import numpy as np
import pytest
from jax import Array
from jax import numpy as jnp
from jax.random import key, normal
from jax.typing import ArrayLike
from scipy.stats import norm as scipy_norm
from typing_extensions import override

from coreax.data import Data
from coreax.kernel import (
    Kernel,
    LaplacianKernel,
    LinearKernel,
    PCIMQKernel,
    SquaredExponentialKernel,
    SteinKernel,
)

_Kernel = TypeVar("_Kernel", bound=Kernel)
_Kernel_co = TypeVar("_Kernel_co", bound=Kernel, covariant=True)


class _KernelFactory(Protocol, Generic[_Kernel_co]):
    def __call__(self, length_scale: float, output_scale: float) -> _Kernel_co: ...


# Once we support only python 3.11+ this should be generic on _Kernel
class _Problem(NamedTuple):
    x: ArrayLike
    y: ArrayLike
    expected_output: ArrayLike
    kernel: Kernel


class BaseKernelTest(ABC, Generic[_Kernel]):
    """Test the ``compute`` methods of a ``coreax.kernel.Kernel``."""

    @abstractmethod
    def kernel(self):
        """Abstract pytest fixture which initialises a kernel with parameters fixed."""

    @abstractmethod
    def problem(self, request) -> _Problem:
        """Abstract pytest fixture which returns a problem for ``Kernel.compute``."""

    def test_compute(self, problem: _Problem):
        """Test ``compute`` method of ``coreax.kernel.Kernel``."""
        x, y, expected_output, kernel = problem
        output = kernel.compute(x, y)
        np.testing.assert_array_almost_equal(output, expected_output)


class KernelMeanTest(Generic[_Kernel]):
    """Test the ``compute_mean`` method of a ``coreax.kernel.Kernel``."""

    @pytest.mark.parametrize(
        "block_size",
        [None, 0, -1, 1.2, 2, 3, 9, (None, 2), (2, None)],
        ids=[
            "none",
            "zero",
            "negative",
            "floating",
            "integer_multiple",
            "fractional_multiple",
            "oversized",
            "tuple[none, integer_multiple]",
            "tuple[integer_multiple, none]",
        ],
    )
    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_compute_mean(
        self,
        kernel: _Kernel,
        block_size: int | float | None,
        axis: int | None,
    ):
        """Test the ``gramian_row_mean`` for all expected classes of 'block_size'."""
        kernel = kernel_factory(1 / np.sqrt(2), 1.0)
        x = np.array(
            [
                [0.0, 1.0],
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
                [5.0, 6.0],
                [6.0, 7.0],
                [7.0, 8.0],
            ]
        )
        expected_output = kernel.compute(x, x).mean(axis=0)
        mean_output = kernel.gramian_row_mean(x, block_size=block_size)
        np.testing.assert_array_almost_equal(mean_output, expected_output)

    def tests_empty_gramian_row_mean(self, kernel_factory: _KernelFactory[_Kernel]):
        """Test the ``gramian_row_mean`` for an empty dataset."""
        kernel = kernel_factory(1 / np.sqrt(2), 1.0)
        x = jnp.array([])
        with pytest.raises(ValueError, match="'x' must not be empty"):
            kernel.gramian_row_mean(x)


class KernelGradientTest(ABC, Generic[_Kernel]):
    """Test the gradient and divergence methods of a ``coreax.kernel.Kernel``."""

    @pytest.fixture
    def gradient_problem(self):
        """Return a problem for testing kernel gradients and divergence."""
        num_points = 10
        dimension = 2
        random_data_generation_key = 1_989
        generator = np.random.default_rng(random_data_generation_key)
        x = generator.random((num_points, dimension))
        y = generator.random((num_points, dimension))
        return x, y

    @pytest.mark.parametrize("mode", ["grad_x", "grad_y", "divergence_x_grad_y"])
    @pytest.mark.parametrize("elementwise", [False, True])
    def test_gradients(
        self,
        gradient_problem: tuple[Array, Array],
        kernel: _Kernel,
        mode: Literal["grad_x", "grad_y", "divergence_x_grad_y"],
        elementwise: bool,
    ):
        """Test computation of the kernel gradients."""
        x, y = gradient_problem
        test_mode = mode
        reference_mode = "expected_" + mode
        if elementwise:
            test_mode += "_elementwise"
            x, y = x[:, 0], y[:, 0]
        expected_output = getattr(self, reference_mode)(x, y, kernel)
        if elementwise:
            expected_output = expected_output.squeeze()
        output = getattr(kernel, test_mode)(x, y)
        np.testing.assert_array_almost_equal(output, expected_output, decimal=3)

    @abstractmethod
    def expected_grad_x(
        self, x: ArrayLike, y: ArrayLike, kernel: _Kernel
    ) -> Array | np.ndarray:
        """Compute expected gradient of the kernel w.r.t ``x``."""

    @abstractmethod
    def expected_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: _Kernel
    ) -> Array | np.ndarray:
        """Compute expected gradient of the kernel w.r.t ``y``."""

    @abstractmethod
    def expected_divergence_x_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: _Kernel
    ) -> Array | np.ndarray:
        """Compute expected divergence of the kernel w.r.t ``x`` gradient ``y``."""


class TestLinearKernel(
    BaseKernelTest[LinearKernel],
    KernelMeanTest[LinearKernel],
    KernelGradientTest[LinearKernel],
):
    """Test ``coreax.kernel.LinearKernel``."""

    @pytest.fixture
    def parameter_factory(self):
        def parameter_generator():
            random_seed = 2_024
            generator = np.random.default_rng(random_seed)
            parameters = jnp.abs(generator.random(2))
            return parameters[0], parameters[1]

        return parameter_generator

    @pytest.fixture
    def kernel_factory(self) -> _KernelFactory[LinearKernel]:
        def kernel(length_scale, output_scale):
            del length_scale, output_scale
            return LinearKernel()

        return kernel

    @override
    @pytest.fixture(params=["floats", "vectors", "arrays"])
    def problem(self, request) -> _Problem:
        r"""
        Test problems for the Linear kernel.

        The kernel is defined as
        :math:`k(x,y) = \text{output_scale} * x^Ty` + \text{constant}.

        We consider the following cases:
        - `floats`: where x and y are floats
        - `vectors`: where x and y are vectors of the same size
        - `arrays`: where x and y are arrays of the same shape
        """
        mode = request.param
        x = 0.5
        y = 2.0
        if mode == "floats":
            expected_distances = 1.0
        elif mode == "vectors":
            x = 1.0 * np.arange(4)
            y = x + 1.0
            expected_distances = 20.0
        elif mode == "arrays":
            x = np.array(([0, 1, 2, 3], [5, 6, 7, 8]))
            y = np.array(([1, 2, 3, 4], [5, 6, 7, 8]))
            expected_distances = np.array([[20, 44], [70, 174]])
        else:
            raise ValueError("Invalid problem mode")
        kernel = LinearKernel(1.0, 0.0)
        return _Problem(x, y, expected_distances, kernel)

    def expected_grad_x(
        self, x: ArrayLike, y: ArrayLike, kernel: LinearKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        expected_gradients = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_gradients[x_idx, y_idx] = kernel.output_scale * y[y_idx]
        return expected_gradients

    def expected_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: LinearKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        expected_gradients = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_gradients[x_idx, y_idx] = kernel.output_scale * x[x_idx]
        return expected_gradients

    def expected_divergence_x_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: LinearKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        expected_divergence = np.zeros((num_points, num_points))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_divergence[x_idx, y_idx] = kernel.output_scale * dimension
        return expected_divergence


class TestSquaredExponentialKernel(
    BaseKernelTest[SquaredExponentialKernel],
    KernelMeanTest[SquaredExponentialKernel],
    KernelGradientTest[SquaredExponentialKernel],
):
    """Test ``coreax.kernel.SquaredExponentialKernel``."""

    @pytest.fixture
    def parameter_factory(self):
        def parameter_generator():
            random_seed = 2_024
            generator = np.random.default_rng(random_seed)
            parameters = jnp.abs(generator.random(2))
            return parameters[0], parameters[1]

        return parameter_generator

    @pytest.fixture
    def kernel_factory(self) -> _KernelFactory[SquaredExponentialKernel]:
        def kernel(length_scale=1.0, output_scale=1.0):
            return SquaredExponentialKernel(length_scale, output_scale)

        return kernel

    @override
    @pytest.fixture(
        params=[
            "floats",
            "vectors",
            "arrays",
            "normalized",
            "negative_length_scale",
            "large_negative_length_scale",
            "near_zero_length_scale",
            "negative_output_scale",
        ]
    )
    def problem(  # noqa: C901
        self, request
    ) -> _Problem:
        r"""
        Test problems for the SquaredExponential kernel.

        The kernel is defined as
        :math:`k(x,y) = \exp (-||x-y||^2/(2 * \text{length_scale}^2))`.

        We consider the following cases:
        1. length scale is :math:`\sqrt{\pi} / 2`:
        - `floats`: where x and y are floats
        - `vectors`: where x and y are vectors of the same size
        - `arrays`: where x and y are arrays of the same shape

        2. length scale is :math:`\exp(1)` and output scale is
        :math:`\frac{1}{\sqrt{2*\pi} * \exp(1)}`:
        - `normalized`: where x and y are vectors of the same size (this is the
        special case where the squared exponential kernel is the Gaussian kernel)

        3. length scale or output scale is degenerate:
        - `negative_length_scale`: should give same result as positive equivalent
        - `large_negative_length_scale`: should approximately equal one
        - `near_zero_length_scale`: should approximately equal zero
        - `negative_output_scale`: should negate the positive equivalent.
        """
        length_scale = np.sqrt(np.float32(np.pi) / 2.0)
        output_scale = 1.0
        mode = request.param
        x = 0.5
        y = 2.0
        if mode == "floats":
            expected_distances = 0.48860678
        elif mode == "vectors":
            x = 1.0 * np.arange(4)
            y = x + 1.0
            expected_distances = 0.279923327
        elif mode == "arrays":
            x = np.array(([0, 1, 2, 3], [5, 6, 7, 8]))
            y = np.array(([1, 2, 3, 4], [5, 6, 7, 8]))
            expected_distances = np.array(
                [[0.279923327, 1.4996075e-14], [1.4211038e-09, 1.0]]
            )
        elif mode == "normalized":
            length_scale = np.e
            output_scale = 1 / (np.sqrt(2 * np.pi) * length_scale)
            num_points = 10
            x = np.arange(num_points)
            y = x + 1.0

            # Compute expected output using standard implementation of the Gaussian PDF
            expected_distances = np.zeros((num_points, num_points))
            for x_idx, x_ in enumerate(x):
                for y_idx, y_ in enumerate(y):
                    expected_distances[x_idx, y_idx] = scipy_norm(y_, length_scale).pdf(
                        x_
                    )
            x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        elif mode == "negative_length_scale":
            length_scale = -1
            expected_distances = 0.324652467
        elif mode == "large_negative_length_scale":
            length_scale = -10000.0
            expected_distances = 1.0
        elif mode == "near_zero_length_scale":
            length_scale = -0.0000001
            expected_distances = 0.0
        elif mode == "negative_output_scale":
            length_scale = 1
            output_scale = -1
            expected_distances = -0.324652467
        else:
            raise ValueError("Invalid problem mode")
        kernel = SquaredExponentialKernel(length_scale, output_scale)
        return _Problem(x, y, expected_distances, kernel)

    def expected_grad_x(
        self, x: ArrayLike, y: ArrayLike, kernel: SquaredExponentialKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        expected_gradients = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_gradients[x_idx, y_idx] = (
                    -kernel.output_scale
                    * (x[x_idx, :] - y[y_idx, :])
                    / kernel.length_scale**2
                    * np.exp(
                        -(np.linalg.norm(x[x_idx, :] - y[y_idx, :]) ** 2)
                        / (2 * kernel.length_scale**2)
                    )
                )
        return expected_gradients

    def expected_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: SquaredExponentialKernel
    ) -> np.ndarray:
        return -self.expected_grad_x(x, y, kernel)

    def expected_divergence_x_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: SquaredExponentialKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        expected_divergence = np.zeros((num_points, num_points))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                dot_product = np.dot(
                    x[x_idx, :] - y[y_idx, :], x[x_idx, :] - y[y_idx, :]
                )
                kernel_scaled = kernel.output_scale * np.exp(
                    -dot_product / (2.0 * kernel.length_scale**2)
                )
                expected_divergence[x_idx, y_idx] = (
                    kernel_scaled
                    / kernel.length_scale**2
                    * (dimension - dot_product / kernel.length_scale**2)
                )
        return expected_divergence


class TestLaplacianKernel(
    BaseKernelTest[LaplacianKernel],
    KernelMeanTest[LaplacianKernel],
    KernelGradientTest[LaplacianKernel],
):
    """Test ``coreax.kernel.LaplacianKernel``."""

    @pytest.fixture
    def parameter_factory(self):
        def parameter_generator():
            random_seed = 2_024
            generator = np.random.default_rng(random_seed)
            parameters = jnp.abs(generator.random(2))
            return parameters[0], parameters[1]

        return parameter_generator

    @pytest.fixture
    def kernel_factory(self):
        def kernel(length_scale=1.0, output_scale=1.0):
            return LaplacianKernel(length_scale, output_scale)

        return kernel

    @override
    @pytest.fixture(
        params=[
            "floats",
            "vectors",
            "arrays",
            "negative_length_scale",
            "large_negative_length_scale",
            "near_zero_length_scale",
            "negative_output_scale",
        ]
    )
    def problem(self, request) -> _Problem:
        r"""
        Test problems for the Laplacian kernel.

        The kernel is defined as
        :math:`k(x,y) = \exp (-||x-y||_1/(2 * \text{length_scale}^2))`.

        We consider the following cases:
        1. length scale is :math:`\sqrt{\pi} / 2`:
        - `floats`: where x and y are floats
        - `vectors`: where x and y are vectors of the same size
        - `arrays`: where x and y are arrays of the same shape

        2. length scale or output scale is degenerate:
        - `negative_length_scale`: should give same result as positive equivalent
        - `large_negative_length_scale`: should approximately equal one
        - `near_zero_length_scale`: should approximately equal zero
        - `negative_output_scale`: should negate the positive equivalent.
        """
        length_scale = np.sqrt(np.float32(np.pi) / 2.0)
        output_scale = 1.0
        mode = request.param
        x = 0.5
        y = 2.0
        if mode == "floats":
            expected_distances = 0.62035410351
        elif mode == "vectors":
            x = 1.0 * np.arange(4)
            y = x + 1.0
            expected_distances = 0.279923327
        elif mode == "arrays":
            x = np.array(([0, 1, 2, 3], [5, 6, 7, 8]))
            y = np.array(([1, 2, 3, 4], [5, 6, 7, 8]))
            expected_distances = np.array(
                [[0.279923327, 0.00171868172], [0.00613983027, 1.0]]
            )
        elif mode == "negative_length_scale":
            length_scale = -1
            expected_distances = 0.472366553
        elif mode == "large_negative_length_scale":
            length_scale = -10000.0
            expected_distances = 1.0
        elif mode == "near_zero_length_scale":
            length_scale = -0.0000001
            expected_distances = 0.0
        elif mode == "negative_output_scale":
            length_scale = 1
            output_scale = -1
            expected_distances = -0.472366553
        else:
            raise ValueError("Invalid problem mode")
        kernel = LaplacianKernel(length_scale, output_scale)
        return _Problem(x, y, expected_distances, kernel)

    def expected_grad_x(
        self, x: ArrayLike, y: ArrayLike, kernel: LaplacianKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        expected_gradients = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_gradients[x_idx, y_idx] = (
                    -kernel.output_scale
                    * np.sign(x[x_idx, :] - y[y_idx, :])
                    / (2 * kernel.length_scale**2)
                    * np.exp(
                        -np.linalg.norm(x[x_idx, :] - y[y_idx, :], ord=1)
                        / (2 * kernel.length_scale**2)
                    )
                )
        return expected_gradients

    def expected_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: LaplacianKernel
    ) -> np.ndarray:
        return -self.expected_grad_x(x, y, kernel)

    def expected_divergence_x_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: LaplacianKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        expected_divergence = np.zeros((num_points, num_points))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                expected_divergence[x_idx, y_idx] = (
                    -kernel.output_scale
                    * dimension
                    / (4 * kernel.length_scale**4)
                    * np.exp(
                        -jnp.linalg.norm(x[x_idx, :] - y[y_idx, :], ord=1)
                        / (2 * kernel.length_scale**2)
                    )
                )
        return expected_divergence


class TestPCIMQKernel(
    BaseKernelTest[PCIMQKernel],
    KernelMeanTest[PCIMQKernel],
    KernelGradientTest[PCIMQKernel],
):
    """Test ``coreax.kernel.PCIMQKernel``."""

    @pytest.fixture
    def parameter_factory(self):
        def parameter_generator():
            random_seed = 2_024
            generator = np.random.default_rng(random_seed)
            parameters = jnp.abs(generator.random(2))
            return parameters[0], parameters[1]

        return parameter_generator

    @pytest.fixture
    def kernel_factory(self):
        def kernel(length_scale=1.0, output_scale=1.0):
            return PCIMQKernel(length_scale, output_scale)

        return kernel

    @override
    @pytest.fixture(
        params=[
            "floats",
            "vectors",
            "arrays",
            "negative_length_scale",
            "large_negative_length_scale",
            "near_zero_length_scale",
            "negative_output_scale",
        ]
    )
    def problem(self, request) -> _Problem:
        r"""
        Test problems for the PCIMQ kernel.

        The kernel is defined as
        :math:`k(x,y) = \frac{1}{\sqrt(1 + ((x - y) / \text{length_scale}) ** 2 / 2)}`.

        We consider the following cases:
        1. length scale is :math:`\sqrt{\pi} / 2`:
        - `floats`: where x and y are floats
        - `vectors`: where x and y are vectors of the same size
        - `arrays`: where x and y are arrays of the same shape

        2. length scale or output scale is degenerate:
        - `negative_length_scale`: should give same result as positive equivalent
        - `large_negative_length_scale`: should approximately equal one
        - `near_zero_length_scale`: should approximately equal zero
        - `negative_output_scale`: should negate the positive equivalent.
        """
        length_scale = np.sqrt(np.float32(np.pi) / 2.0)
        output_scale = 1.0
        mode = request.param
        x = 0.5
        y = 2.0
        if mode == "floats":
            expected_distances = 0.76333715144
        elif mode == "vectors":
            x = 1.0 * np.arange(4)
            y = x + 1.0
            expected_distances = 0.66325021409
        elif mode == "arrays":
            x = np.array(([0, 1, 2, 3], [5, 6, 7, 8]))
            y = np.array(([1, 2, 3, 4], [5, 6, 7, 8]))
            expected_distances = np.array(
                [[0.66325021409, 0.17452514991], [0.21631125495, 1.0]]
            )
        elif mode == "negative_length_scale":
            length_scale = -1
            expected_distances = 0.685994341
        elif mode == "large_negative_length_scale":
            length_scale = -10000.0
            expected_distances = 1.0
        elif mode == "near_zero_length_scale":
            length_scale = -0.0000001
            expected_distances = 0.0
        elif mode == "negative_output_scale":
            length_scale = 1
            output_scale = -1
            expected_distances = -0.685994341
        else:
            raise ValueError("Invalid problem mode")
        kernel = PCIMQKernel(length_scale, output_scale)
        return _Problem(x, y, expected_distances, kernel)

    def expected_grad_x(
        self, x: ArrayLike, y: ArrayLike, kernel: PCIMQKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        expected_gradients = np.zeros((num_points, num_points, dimension))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                scaling = 2 * kernel.length_scale**2
                mq_array = np.linalg.norm(x[x_idx, :] - y[y_idx, :]) ** 2 / scaling
                primal = kernel.output_scale / np.sqrt(1 + mq_array)
                expected_gradients[x_idx, y_idx] = (
                    -kernel.output_scale
                    / scaling
                    * (x[x_idx, :] - y[y_idx, :])
                    * (primal / kernel.output_scale) ** 3
                )
        return expected_gradients

    def expected_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: PCIMQKernel
    ) -> np.ndarray:
        return -self.expected_grad_x(x, y, kernel)

    def expected_divergence_x_grad_y(
        self, x: ArrayLike, y: ArrayLike, kernel: PCIMQKernel
    ) -> np.ndarray:
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        num_points, dimension = x.shape
        length_scale = kernel.length_scale
        output_scale = kernel.output_scale
        expected_divergence = np.zeros((num_points, num_points))
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                dot_product = np.dot(
                    x[x_idx, :] - y[y_idx, :], x[x_idx, :] - y[y_idx, :]
                )
                kernel_scaled = output_scale / (
                    (1 + dot_product / (2 * length_scale**2)) ** (1 / 2)
                )
                expected_divergence[x_idx, y_idx] = (
                    output_scale
                    / (2 * length_scale**2)
                    * (
                        dimension * (kernel_scaled / output_scale) ** 3
                        - 3
                        * dot_product
                        * (kernel_scaled / output_scale) ** 5
                        / (2 * length_scale**2)
                    )
                )

        return expected_divergence


class TestSteinKernel(BaseKernelTest[SteinKernel]):
    """Test ``coreax.kernel.SteinKernel``."""

    @pytest.fixture
    def parameter_factory(self):
        def parameter_generator():
            random_seed = 2_024
            generator = np.random.default_rng(random_seed)
            parameters = jnp.abs(generator.random(2))
            return parameters[0], parameters[1]

        return parameter_generator

    @pytest.fixture
    def kernel_factory(self) -> _KernelFactory[SteinKernel]:
        def kernel(length_scale=1.0, output_scale=1.0):
            base_kernel = PCIMQKernel(length_scale, output_scale)
            return SteinKernel(base_kernel=base_kernel, score_function=jnp.negative)

    @pytest.fixture
    def problem(self, request) -> _Problem:
        """Test problem for the Stein kernel."""
        length_scale = 1 / np.sqrt(2)
        kernel = SteinKernel(
            base_kernel=PCIMQKernel(length_scale, 1), score_function=jnp.negative
        )
        score_function = kernel.score_function
        beta = 0.5
        num_points_x = 10
        num_points_y = 5
        dimension = 2
        generator = np.random.default_rng(1_989)
        x = generator.random((num_points_x, dimension))
        y = generator.random((num_points_y, dimension))

        def k_x_y(x_input, y_input):
            r"""
            Compute Stein kernel.

            Throughout this docstring, x_input and y_input are simply referred to as x
            and y.

            The base kernel is :math:`(1 + \lvert \mathbf{x} - \mathbf{y}
            \rvert^2)^{-1/2}`. :math:`\mathbb{P}` is :math:`\mathcal{N}(0, \mathbf{I})`
            with :math:`\nabla \log f_X(\mathbf{x}) = -\mathbf{x}`.

            In the code: n, m and r refer to shared denominators in the Stein kernel
            equation (rather than divergence, x_, y_ and z in the main code function).

            :math:`k_\mathbb{P}(\mathbf{x}, \mathbf{y}) = n + m + r`.

            :math:`n := -\frac{3 \lvert \mathbf{x} - \mathbf{y} \rvert^2}{(1 + \lvert
            \mathbf{x} - \mathbf{y} \rvert^2)^{5/2}}`.

            :math:`m := 2\beta\left[ \frac{d + [\mathbf{y} -
            \mathbf{x}]^\intercal[\mathbf{x} - \mathbf{y}]}{(1 + \lvert \mathbf{x} -
            \mathbf{y} \rvert^2)^{3/2}} \right]`.

            :math:`r := \frac{\mathbf{x}^\intercal \mathbf{y}}{(1 + \lvert \mathbf{x} -
            \mathbf{y} \rvert^2)^{1/2}}`.

            :param x_input: a d-dimensional vector
            :param y_input: a d-dimensional vector
            :return: kernel evaluated at x, y
            """
            norm_sq = np.linalg.norm(x_input - y_input) ** 2
            n = -3 * norm_sq / (1 + norm_sq) ** 2.5
            dot_prod = np.dot(
                score_function(x_input) - score_function(y_input),
                x_input - y_input,
            )
            m = 2 * beta * (dimension + dot_prod) / (1 + norm_sq) ** 1.5
            r = (
                np.dot(score_function(x_input), score_function(y_input))
                / (1 + norm_sq) ** 0.5
            )
            return n + m + r

        expected_output = np.zeros([x.shape[0], y.shape[0]])
        for x_idx in range(x.shape[0]):
            for y_idx in range(y.shape[0]):
                # Compute via our hand-coded kernel evaluation
                expected_output[x_idx, y_idx] = k_x_y(x[x_idx, :], y[y_idx, :])

        return _Problem(x, y, expected_output, kernel)
