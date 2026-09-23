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

"""Tests for constructing Stein kernels from log-density functions."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from coreax.kernels import SquaredExponentialKernel, SteinKernel


class GaussianLogDensity(eqx.Module):
    """Gaussian log density with explicit PyTree parameters."""

    mean: Array
    precision: Array

    def __call__(self, point: Array) -> Array:
        """Evaluate the log density up to an additive constant."""
        return -jnp.sum(self.precision * (point - self.mean) ** 2) / 2


@pytest.mark.parametrize("dimension", [1, 3])
def test_matches_analytic_score(
    jit_variant: Callable[[Callable], Callable], dimension: int
) -> None:
    """Match analytic Gaussian scores and Stein kernel matrices."""
    density = GaussianLogDensity(
        jnp.arange(dimension, dtype=float), jnp.arange(dimension) + 1.0
    )
    points = jnp.arange(4 * dimension, dtype=float).reshape(4, dimension) / 3
    base = SquaredExponentialKernel()
    kernel = SteinKernel.from_log_density(base, density)
    reference = SteinKernel(
        base, lambda point: -density.precision * (point - density.mean)
    )

    assert kernel.log_density is density
    np.testing.assert_allclose(
        jit_variant(kernel.score_function)(points[0]),
        reference.score_function(points[0]),
    )
    np.testing.assert_allclose(
        jit_variant(kernel.compute)(points, points),
        reference.compute(points, points),
        rtol=1e-5,
    )


def test_density_without_normalisation(
    jit_variant: Callable[[Callable], Callable],
) -> None:
    """An additive log-normalisation constant must not change the kernel."""

    def log_density(point: Array) -> Array:
        return -jnp.sum(point**2) / 2 + 7.0

    kernel = SteinKernel.from_log_density(SquaredExponentialKernel(), log_density)
    reference = SteinKernel(SquaredExponentialKernel(), jnp.negative)
    points = jnp.array([[0.0, 1.0], [2.0, -1.0]])
    np.testing.assert_allclose(
        jit_variant(kernel.compute)(points, points), reference.compute(points, points)
    )


def test_kernel_derivatives(jit_variant: Callable[[Callable], Callable]) -> None:
    """Differentiate through the derived score when differentiating the kernel."""
    density = GaussianLogDensity(jnp.array([1.0, -1.0]), jnp.array([2.0, 0.5]))
    base = SquaredExponentialKernel()
    kernel = SteinKernel.from_log_density(base, density)
    reference = SteinKernel(
        base, lambda point: -density.precision * (point - density.mean)
    )
    first, second = jnp.array([0.5, 1.0]), jnp.array([1.5, -0.5])
    for name in [
        "grad_x_elementwise",
        "grad_y_elementwise",
        "divergence_x_grad_y_elementwise",
    ]:
        np.testing.assert_allclose(
            jit_variant(getattr(kernel, name))(first, second),
            getattr(reference, name)(first, second),
            rtol=1e-5,
            atol=1e-6,
        )


def test_parameter_updates_are_not_captured() -> None:
    """Updating density parameters must also update the score in compiled kernels."""
    density = GaussianLogDensity(jnp.array([0.0]), jnp.array([2.0]))
    kernel = SteinKernel.from_log_density(SquaredExponentialKernel(), density)
    updated = eqx.tree_at(_density_mean, kernel, jnp.array([3.0]))
    points = jnp.array([[0.0], [1.0]])
    compute = eqx.filter_jit(lambda item: item.compute(points, points))
    before = compute(kernel)
    after = compute(updated)
    reference = SteinKernel(
        SquaredExponentialKernel(), lambda point: -2 * (jnp.asarray(point) - 3)
    )

    assert not np.allclose(before, after)
    np.testing.assert_allclose(after, reference.compute(points, points))
    np.testing.assert_allclose(_density_mean(kernel), jnp.array([0.0]))


def test_differentiates_density_parameters() -> None:
    """Density parameters remain visible to Equinox differentiation."""
    density = GaussianLogDensity(jnp.array([0.5]), jnp.array([2.0]))
    kernel = SteinKernel.from_log_density(SquaredExponentialKernel(), density)
    points = jnp.array([[0.0], [1.0], [2.0]])
    gradient = eqx.filter_grad(lambda item: jnp.sum(item.compute(points, points)))(
        kernel
    )

    def reference_loss(mean: Array) -> Array:
        reference = SteinKernel(
            SquaredExponentialKernel(), lambda point: -2 * (point - mean)
        )
        return jnp.sum(reference.compute(points, points))

    expected = jax.grad(reference_loss)(density.mean)
    np.testing.assert_allclose(_density_mean(gradient), expected, rtol=1e-5)
    assert np.any(np.asarray(expected) != 0)


def test_score_only_constructor_is_unchanged() -> None:
    """The existing positional score-function constructor needs no log density."""
    kernel = SteinKernel(SquaredExponentialKernel(), jnp.negative)
    assert kernel.log_density is None
    np.testing.assert_array_equal(
        kernel.score_function(jnp.array([1.0])), jnp.array([-1.0])
    )


def _density_mean(kernel: SteinKernel) -> Array:
    """Read the mean from a kernel built with a Gaussian log density."""
    density = kernel.log_density
    assert isinstance(density, GaussianLogDensity)
    return density.mean


@pytest.mark.parametrize("point", [2, 2.0, jnp.array([2, 3])])
def test_score_accepts_numeric_inputs(
    jit_variant: Callable[[Callable], Callable], point: Array | float | int
) -> None:
    """Scalar and integer inputs are converted before automatic differentiation."""
    kernel = SteinKernel.from_log_density(
        SquaredExponentialKernel(), lambda value: -jnp.sum(value**2) / 2
    )
    np.testing.assert_allclose(
        jit_variant(kernel.score_function)(point), -np.asarray(point)
    )
