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

"""Analytic and integration checks for per-feature kernel scales."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, ArrayLike

from coreax.data import Data
from coreax.kernels import (
    AnisotropicKernel,
    LaplacianKernel,
    LinearKernel,
    PCIMQKernel,
    ScalarValuedKernel,
    SquaredExponentialKernel,
    SteinKernel,
)
from coreax.metrics import MMD


@pytest.mark.parametrize("scale", [2.0, [0.5, 2.0]])
def test_gaussian_values(
    jit_variant: Callable[[Callable], Callable], scale: ArrayLike
) -> None:
    """Match the anisotropic Gaussian formula for an entire kernel matrix."""
    first = np.array([[0.0, 1.0], [2.0, -1.0]])
    second = np.array([[1.0, 3.0], [-1.0, 2.0], [0.5, 0.5]])
    kernel = AnisotropicKernel(SquaredExponentialKernel(output_scale=2), scale)
    delta = (first[:, None, :] - second[None, :, :]) / np.asarray(scale)
    expected = 2 * np.exp(-np.sum(delta**2, axis=-1) / 2)
    actual = jit_variant(kernel.compute)(jnp.asarray(first), jnp.asarray(second))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize(
    "base",
    [
        LinearKernel(),
        LaplacianKernel(),
        PCIMQKernel(),
        SquaredExponentialKernel() + LinearKernel(),
    ],
)
def test_existing_kernels(
    jit_variant: Callable[[Callable], Callable], base: ScalarValuedKernel
) -> None:
    """Scale base kernels and compositions without changing their parameters."""
    first = jnp.array([[0.0, 1.0], [2.0, -1.0]])
    second = jnp.array([[1.0, 3.0], [-1.0, 2.0]])
    scale = jnp.array([0.5, 2.0])
    kernel = AnisotropicKernel(base, scale)
    actual = jit_variant(kernel.compute)(first, second)
    expected = base.compute(first / scale, second / scale)
    np.testing.assert_allclose(actual, expected, rtol=2e-6)


@pytest.mark.parametrize("coincident", [False, True])
def test_gaussian_derivatives(
    jit_variant: Callable[[Callable], Callable], coincident: bool
) -> None:
    """Apply the coordinate chain rule, including the mixed derivative trace."""
    first = jnp.array([0.3, -0.7])
    second = first if coincident else jnp.array([1.2, 0.4])
    scale = np.array([0.5, 2.0])
    delta = np.asarray(first - second)
    value = 2 * np.exp(-np.sum((delta / scale) ** 2) / 2)
    gradient = -delta / scale**2 * value
    divergence = np.sum(1 / scale**2 - delta**2 / scale**4) * value
    kernel = AnisotropicKernel(SquaredExponentialKernel(output_scale=2), scale)
    np.testing.assert_allclose(
        jit_variant(kernel.grad_x_elementwise)(first, second), gradient, rtol=2e-6
    )
    np.testing.assert_allclose(
        jit_variant(kernel.grad_y_elementwise)(first, second), -gradient, rtol=2e-6
    )
    np.testing.assert_allclose(
        jit_variant(kernel.divergence_x_grad_y_elementwise)(first, second),
        divergence,
        rtol=2e-6,
    )


def test_differentiate_length_scales(
    jit_variant: Callable[[Callable], Callable],
) -> None:
    """Keep length scales differentiable when construction occurs inside JIT."""
    first, second = jnp.array([0.2, 1.0]), jnp.array([-0.4, 0.5])
    scale = jnp.array([1.5, 0.7])

    def evaluate(length_scale: Array) -> Array:
        return AnisotropicKernel(
            SquaredExponentialKernel(), length_scale
        ).compute_elementwise(first, second)

    actual = jit_variant(jax.grad(evaluate))(scale)
    delta = first - second
    expected = jnp.exp(-jnp.sum((delta / scale) ** 2) / 2) * delta**2 / scale**3
    np.testing.assert_allclose(actual, expected, rtol=2e-6)


def test_tree_update(jit_variant: Callable[[Callable], Callable]) -> None:
    """Retain the scale as a replaceable PyTree leaf."""
    kernel = AnisotropicKernel(SquaredExponentialKernel(), jnp.ones(2))
    updated = eqx.tree_at(lambda item: item.length_scale, kernel, jnp.array([0.5, 2.0]))
    first, second = jnp.zeros(2), jnp.ones(2)
    expected = jnp.exp(-jnp.sum((first - second) ** 2 / updated.length_scale**2) / 2)
    np.testing.assert_allclose(
        jit_variant(updated.compute_elementwise)(first, second), expected
    )
    np.testing.assert_array_equal(kernel.length_scale, jnp.ones(2))


def test_weighted_metric(jit_variant: Callable[[Callable], Callable]) -> None:
    """Preserve weighted metric behaviour under an equivalent input transformation."""
    first = Data(jnp.array([[0.0, 1.0], [2.0, -1.0]]), jnp.array([1.0, 3.0]))
    second = Data(jnp.array([[1.0, 3.0], [-1.0, 2.0]]), jnp.array([2.0, 1.0]))
    scale = jnp.array([0.5, 2.0])
    base = SquaredExponentialKernel()
    actual = jit_variant(MMD(AnisotropicKernel(base, scale)).compute)(first, second)
    expected = MMD(base).compute(
        Data(first.data / scale, first.weights),
        Data(second.data / scale, second.weights),
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6)


def test_stein_composition(jit_variant: Callable[[Callable], Callable]) -> None:
    """Use anisotropic base derivatives without rescaling the target score."""
    scale = jnp.array([0.5, 2.0])
    first, second = jnp.array([0.3, -0.7]), jnp.array([1.2, 0.4])
    delta = first - second
    value = jnp.exp(-jnp.sum((delta / scale) ** 2) / 2)
    gradient = -delta / scale**2 * value
    expected = (
        jnp.sum(1 / scale**2 - delta**2 / scale**4) * value
        + gradient @ (-second)
        - gradient @ (-first)
        + value * (first @ second)
    )
    kernel = SteinKernel(
        AnisotropicKernel(SquaredExponentialKernel(), scale), jnp.negative
    )
    np.testing.assert_allclose(
        jit_variant(kernel.compute_elementwise)(first, second), expected, rtol=2e-6
    )


@pytest.mark.parametrize("scale", [2.0, [2.0]])
def test_scalar_inputs(
    jit_variant: Callable[[Callable], Callable], scale: ArrayLike
) -> None:
    """Support one-dimensional scalar inputs through the vectorised interface."""
    kernel = AnisotropicKernel(SquaredExponentialKernel(), scale)
    np.testing.assert_allclose(
        jit_variant(kernel.compute)(1.0, 2.0), [[np.exp(-0.125)]]
    )
    for method, expected_gradient in (
        (kernel.grad_x_elementwise, np.exp(-0.125) / 4),
        (kernel.grad_y_elementwise, -np.exp(-0.125) / 4),
    ):
        gradient = jit_variant(method)(1.0, 2.0)
        assert gradient.shape == ()
        np.testing.assert_allclose(gradient, expected_gradient)
    expected_trace = np.exp(-0.125) * (1 / 4 - 1 / 16)
    np.testing.assert_allclose(
        jit_variant(kernel.divergence_x_grad_y_elementwise)(1.0, 2.0), expected_trace
    )
    np.testing.assert_allclose(
        jit_variant(kernel.divergence_x_grad_y)(1.0, 2.0), [[expected_trace]]
    )


@pytest.mark.parametrize(
    "scale", [0.0, -1.0, [1.0, 0.0], [1.0, -1.0], [1.0, np.nan], [1.0, np.inf]]
)
def test_invalid_scale_values(
    jit_variant: Callable[[Callable], Callable], scale: ArrayLike
) -> None:
    """Reject non-positive or non-finite scales in eager and compiled calls."""

    def evaluate(length_scale: Array) -> Array:
        kernel = AnisotropicKernel(SquaredExponentialKernel(), length_scale)
        return kernel.compute_elementwise(jnp.zeros(2), jnp.ones(2))

    with pytest.raises((ValueError, RuntimeError), match="finite, positive"):
        jit_variant(evaluate)(jnp.asarray(scale)).block_until_ready()


@pytest.mark.parametrize("scale", [[], [[1.0, 2.0]], [1.0, 1j]])
def test_invalid_scale_shapes(scale: ArrayLike) -> None:
    """Reject empty, complex and higher-dimensional scales."""
    with pytest.raises(ValueError, match="real scalar or non-empty vector"):
        AnisotropicKernel(SquaredExponentialKernel(), scale)


@pytest.mark.parametrize("scale", [[1.0], [1.0, 2.0, 3.0]])
def test_feature_mismatch(
    jit_variant: Callable[[Callable], Callable], scale: ArrayLike
) -> None:
    """Prevent broadcasting a vector scale onto the wrong feature count."""
    kernel = AnisotropicKernel(SquaredExponentialKernel(), scale)
    with pytest.raises(ValueError, match="one entry per input feature"):
        jit_variant(kernel.compute)(jnp.zeros((3, 2)), jnp.ones((4, 2)))
