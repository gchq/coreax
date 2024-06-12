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

"""Tests for tensor-product kernel implementation."""

from typing import Callable, Literal, Union
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest

from coreax.data import SupervisedData
from coreax.kernel import Kernel, LinearKernel, SquaredExponentialKernel
from coreax.tensor_kernel import TensorProductKernel


class TestTensorProductKernel:
    """Test the methods of ``coreax.tensor_kernel.TensorProductKernel``."""

    @pytest.fixture(scope="class")
    def kernel(self) -> TensorProductKernel:
        """Return a mocked paired kernel function."""
        return TensorProductKernel(
            feature_kernel=SquaredExponentialKernel(), response_kernel=LinearKernel()
        )

    @pytest.mark.parametrize("mode", ["floats", "vectors", "arrays"])
    def test_compute(
        self,
        kernel: TensorProductKernel,
        jit_variant: Callable[[Callable], Callable],
        mode: Literal["floats", "vectors", "arrays"],
    ):
        """Test `compute` method across various input data type."""
        if mode == "floats":
            x1, y1 = jnp.array([1.0]), jnp.array([2.0])
            x2, y2 = jnp.array([3.0]), jnp.array([4.0])
            expected_output = jnp.array([[1.0826823]])
        elif mode == "vectors":
            x1, y1 = jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0])
            x2, y2 = jnp.array([3.0, 4.0]), jnp.array([4.0, 5.0])
            expected_output = jnp.array([[0.4212597]])
        elif mode == "arrays":
            x1, y1 = (
                jnp.array(([1, 2, 3], [5, 6, 7])),
                jnp.array(([1, 2, 3], [5, 6, 7])),
            )
            x2, y2 = (
                jnp.array(([1, 2, 3], [5, 6, 7])),
                jnp.array(([1, 2, 3], [5, 6, 7])),
            )
            expected_output = jnp.array([[14.0, 1.434551e-09], [1.434551e-09, 110.0]])

        output = jit_variant(kernel.compute)(
            SupervisedData(x1, y1), SupervisedData(x2, y2)
        )
        np.testing.assert_array_almost_equal(output, expected_output)

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
        kernel: TensorProductKernel,
        jit_variant: Callable[[Callable], Callable],
        block_size: Union[int, float, None],
        axis: Union[int, None],
    ) -> None:
        """
        Test the `compute_mean` methods.

        Considers all classes of 'block_size' and 'axis', along with implicitly and
        explicitly weighted data.
        """
        a = jnp.array(
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
        b = a[:-1] + 1
        kernel_matrix = kernel.compute(SupervisedData(a, a), SupervisedData(b, b))
        a_weights, b_weights = jnp.arange(a.shape[0]), jnp.arange(b.shape[0])
        a_data, b_data = (
            SupervisedData(a, a, a_weights),
            SupervisedData(b, b, b_weights),
        )
        if axis == 0:
            weights = a_weights
        elif axis == 1:
            weights = b_weights
        else:
            weights = a_weights[..., None] * b_weights[None, ...]
        expected = jnp.average(kernel_matrix, axis, weights)
        mean_output = jit_variant(kernel.compute_mean)(
            a_data, b_data, axis, block_size=block_size
        )
        np.testing.assert_array_almost_equal(mean_output, expected, decimal=5)

    def test_gramian_row_mean(
        self,
        kernel: TensorProductKernel,
        jit_variant: Callable[[Callable], Callable],
    ) -> None:
        """Test that `gramian_row_mean` method is working as expected."""
        data = jnp.ones((10, 10))
        a = SupervisedData(data, data)
        expected = kernel.compute(a, a).mean(axis=0)
        mean_output = jit_variant(kernel.gramian_row_mean)(a)
        np.testing.assert_array_almost_equal(mean_output, expected, decimal=5)

    def test_compute_elementwise_calls_sub_kernels_correctly(self) -> None:
        """Check that `compute_elementwise` calls sub-kernels as expected."""
        k1, k2 = MagicMock(spec=Kernel), MagicMock(spec=Kernel)
        k1.compute_elementwise.return_value = jnp.array(1.0)
        k2.compute_elementwise.return_value = jnp.array(2.0)
        k = TensorProductKernel(feature_kernel=k1, response_kernel=k2)

        x1, y1, x2, y2 = 1, 2, 3, 4
        k.compute_elementwise((x1, y1), (x2, y2))
        k.feature_kernel.compute_elementwise.assert_called_once()  # pyright:ignore
        k.response_kernel.compute_elementwise.assert_called_once()  # pyright:ignore
