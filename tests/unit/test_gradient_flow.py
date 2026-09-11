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

"""Analytic and behavioural tests for MMD particle gradient flow."""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from coreax.coreset import Coresubset, PseudoCoreset
from coreax.data import Data, SupervisedData
from coreax.kernels import LinearKernel, SquaredExponentialKernel
from coreax.metrics import MMD
from coreax.solvers import GradientFlow, GradientFlowState


@pytest.fixture(name="initial_coreset")
def make_initial_coreset() -> PseudoCoreset[Data]:
    """Create a weighted target and particles with distinct, non-uniform weights."""
    target = Data(
        jnp.array([[0.0, 1.0], [4.0, 3.0], [10.0, -2.0]]), jnp.array([1.0, 3.0, 0.0])
    )
    points = Data(jnp.array([[1.0, 0.0], [2.0, 2.0]]), jnp.array([3.0, 1.0]))
    return PseudoCoreset(points, target)


def test_linear_step(
    jit_variant: Callable[[Callable], Callable], initial_coreset: PseudoCoreset[Data]
) -> None:
    """Linear-kernel velocity is the difference between the two weighted means."""
    solver = GradientFlow(
        2, jr.key(1), LinearKernel(), optimiser=optax.sgd(0.2), num_iterations=1
    )
    result, _ = jit_variant(solver.refine)(initial_coreset)
    expected = np.array([[1.0, 0.0], [2.0, 2.0]]) - 0.2 * (
        np.array([1.25, 0.5]) - np.array([3.0, 2.5])
    )
    np.testing.assert_allclose(result.points.data, expected, atol=2e-7)
    np.testing.assert_array_equal(result.points.weights, initial_coreset.points.weights)
    np.testing.assert_array_equal(
        result.pre_coreset_data.data, initial_coreset.pre_coreset_data.data
    )
    np.testing.assert_array_equal(
        result.pre_coreset_data.weights, initial_coreset.pre_coreset_data.weights
    )


def _gaussian_gradient(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Independent derivative of the test Gaussian kernel, with scales 1.5 and 2."""
    difference = first[:, None, :] - second[None, :, :]
    kernel = 2 * np.exp(-np.sum(difference**2, axis=-1) / (2 * 1.5**2))
    return -difference * kernel[:, :, None] / 1.5**2


@pytest.mark.parametrize("noise_scale", [0.0, 0.3])
def test_gaussian_step(
    jit_variant: Callable[[Callable], Callable],
    initial_coreset: PseudoCoreset[Data],
    noise_scale: float,
) -> None:
    """Noise shifts only the witness evaluation point, with no extra particle factor."""
    key = jr.key(42)
    solver = GradientFlow(
        2,
        key,
        SquaredExponentialKernel(1.5, 2.0),
        optimiser=optax.sgd(0.2),
        num_iterations=1,
        noise_scale=noise_scale,
    )
    result, state = jit_variant(solver.refine)(initial_coreset, GradientFlowState(key))
    next_key, noise_key = jr.split(key)
    points = np.asarray(initial_coreset.points.data)
    evaluation = points + noise_scale * np.asarray(jr.normal(noise_key, points.shape))
    repulsion = np.einsum(
        "ijd,j->id", _gaussian_gradient(evaluation, points), [0.75, 0.25]
    )
    attraction = np.einsum(
        "ijd,j->id",
        _gaussian_gradient(
            evaluation, np.asarray(initial_coreset.pre_coreset_data.data)
        ),
        [0.25, 0.75, 0.0],
    )
    np.testing.assert_allclose(
        result.points.data,
        points - 0.2 * (repulsion - attraction),
        atol=3e-7,
        rtol=2e-6,
    )
    np.testing.assert_array_equal(jr.key_data(state.random_key), jr.key_data(next_key))


@pytest.mark.parametrize(
    "noise_scale",
    [0.0, 0.25, optax.linear_schedule(0.05, 0.3, 4)],
    ids=["zero", "constant", "schedule"],
)
def test_refinement_continues_noise_sequence(
    jit_variant: Callable[[Callable], Callable],
    initial_coreset: PseudoCoreset[Data],
    noise_scale: float | optax.Schedule,
) -> None:
    """Two refinements with state match an uninterrupted run of the same length."""
    solver = GradientFlow(
        2,
        jr.key(13),
        SquaredExponentialKernel(),
        num_iterations=4,
        noise_scale=noise_scale,
    )
    total_iterations = 8
    first, state = jit_variant(solver.refine)(initial_coreset)
    second, state = jit_variant(solver.refine)(first, state)
    long_solver = eqx.tree_at(
        lambda item: item.num_iterations, solver, total_iterations
    )
    expected, expected_state = jit_variant(long_solver.refine)(initial_coreset)
    np.testing.assert_allclose(second.points.data, expected.points.data, atol=1e-6)
    np.testing.assert_array_equal(
        jr.key_data(state.random_key), jr.key_data(expected_state.random_key)
    )
    assert int(state.iteration) == total_iterations


def test_refinement_continues_adaptive_optimiser_state(
    jit_variant: Callable[[Callable], Callable],
    initial_coreset: PseudoCoreset[Data],
) -> None:
    """Adaptive optimiser state is preserved across refinement calls."""
    solver = GradientFlow(
        2,
        jr.key(13),
        SquaredExponentialKernel(),
        optimiser=optax.adam(0.05),
        num_iterations=4,
    )
    total_iterations = 8
    first, state = jit_variant(solver.refine)(initial_coreset)
    second, state = jit_variant(solver.refine)(first, state)
    long_solver = eqx.tree_at(
        lambda item: item.num_iterations, solver, total_iterations
    )
    expected, expected_state = jit_variant(long_solver.refine)(initial_coreset)
    np.testing.assert_allclose(second.points.data, expected.points.data, atol=1e-6)
    assert state.optimiser_state is not None
    assert int(state.iteration) == int(expected_state.iteration) == total_iterations


@pytest.mark.parametrize(("learning_rate", "num_iterations"), [(0.0, 5), (0.1, 0)])
def test_zero_update_preserves_points(
    jit_variant: Callable[[Callable], Callable],
    initial_coreset: PseudoCoreset[Data],
    learning_rate: float,
    num_iterations: int,
) -> None:
    """Check that zero learning rate or iteration count preserves particle values."""
    solver = GradientFlow(
        2,
        jr.key(0),
        SquaredExponentialKernel(),
        optimiser=optax.sgd(learning_rate),
        num_iterations=num_iterations,
        noise_scale=1.0,
    )
    result, _ = jit_variant(solver.refine)(initial_coreset)
    np.testing.assert_array_equal(result.points.data, initial_coreset.points.data)


def test_sampling_uses_weights_and_is_reproducible(
    jit_variant: Callable[[Callable], Callable],
) -> None:
    """Check that sampling excludes zero-mass points and can repeat active points."""
    target = Data(jnp.array([[0], [5], [10]]), jnp.array([0, 1, 0]))
    solver = GradientFlow(2, jr.key(3), LinearKernel(), num_iterations=0)
    first, state = jit_variant(solver.reduce)(target)
    second, second_state = jit_variant(solver.reduce)(target)
    np.testing.assert_array_equal(first.points.data, jnp.array([[5.0], [5.0]]))
    np.testing.assert_array_equal(second.points.data, first.points.data)
    np.testing.assert_array_equal(
        jr.key_data(state.random_key), jr.key_data(second_state.random_key)
    )
    assert jnp.issubdtype(first.points.data.dtype, jnp.floating)
    np.testing.assert_array_equal(first.points.weights, jnp.ones(2))


def test_weight_rescaling_and_zero_mass_do_not_change_flow(
    jit_variant: Callable[[Callable], Callable], initial_coreset: PseudoCoreset[Data]
) -> None:
    """Only normalised probability masses affect the update."""
    target = initial_coreset.pre_coreset_data
    trimmed = PseudoCoreset(
        Data(initial_coreset.points.data, initial_coreset.points.weights * 7),
        Data(target.data[:2], jnp.asarray(target.weights)[:2] * 4),
    )
    solver = GradientFlow(2, jr.key(5), SquaredExponentialKernel(), num_iterations=4)
    result, _ = jit_variant(solver.refine)(trimmed)
    expected, _ = jit_variant(solver.refine)(initial_coreset)
    np.testing.assert_allclose(result.points.data, expected.points.data, atol=1e-6)


def test_refines_existing_coresubset(
    jit_variant: Callable[[Callable], Callable],
) -> None:
    """Accept a coresubset as initial particles without changing base interfaces."""
    target = Data(jnp.array([[0.0], [1.0], [2.0]]))
    initial = Coresubset(Data(jnp.array([0, 1])), target)
    solver = GradientFlow(2, jr.key(5), LinearKernel(), num_iterations=1)
    result, _ = jit_variant(solver.refine)(initial)
    assert isinstance(result, PseudoCoreset)
    np.testing.assert_allclose(result.points.data, [[0.05], [1.05]], atol=1e-7)


def test_small_steps_reduce_mmd(jit_variant: Callable[[Callable], Callable]) -> None:
    """A well-scaled Gaussian example improves through actual particle movement."""
    target = Data(jnp.linspace(-1, 1, 20))
    initial = PseudoCoreset(Data(jnp.array([[2.0], [3.0]])), target)
    kernel = SquaredExponentialKernel()
    solver = GradientFlow(
        2, jr.key(5), kernel, optimiser=optax.sgd(0.1), num_iterations=1
    )
    refine = jit_variant(solver.refine)
    result = initial
    losses = [float(MMD(kernel).compute(target, initial.points))]
    for _ in range(20):
        result, _ = refine(result)
        losses.append(float(MMD(kernel).compute(target, result.points)))
    rounding_tolerance = 1e-6
    assert np.all(np.diff(losses) <= rounding_tolerance)
    assert losses[-1] < losses[0]
    np.testing.assert_array_equal(initial.points.data, [[2.0], [3.0]])


@pytest.mark.parametrize(
    "weights", [[0.0, 0.0], [-1.0, 2.0], [jnp.nan, 1.0], [jnp.inf, 1.0]]
)
def test_rejects_invalid_weights(
    jit_variant: Callable[[Callable], Callable], weights: list[float]
) -> None:
    """Invalid probability measures fail clearly, also inside compiled calls."""
    solver = GradientFlow(1, jr.key(0), LinearKernel(), num_iterations=1)
    target = Data(jnp.array([[0.0], [1.0]]), jnp.array(weights))
    with pytest.raises((RuntimeError, ValueError), match="non-negative weights"):
        result, _ = jit_variant(solver.reduce)(target)
        result.points.data.block_until_ready()


@pytest.mark.parametrize("value", [jnp.nan, jnp.inf])
def test_rejects_non_finite_coordinates(value: float) -> None:
    """Non-finite coordinates are rejected before they reach the kernel."""
    solver = GradientFlow(1, jr.key(0), LinearKernel())
    with pytest.raises((RuntimeError, ValueError), match="Data must be finite"):
        solver.reduce(Data(jnp.array([[value]])))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("noise_scale", -1.0),
        ("noise_scale", float("inf")),
        ("num_iterations", -1),
        ("num_iterations", 1.5),
        ("coreset_size", 0),
    ],
)
def test_invalid_parameters(name: str, value: float) -> None:
    """Invalid solver configuration is rejected at construction."""
    kwargs: dict[str, Any] = {
        "coreset_size": 1,
        "random_key": jr.key(0),
        "kernel": LinearKernel(),
        name: value,
    }
    with pytest.raises(ValueError, match=name):
        GradientFlow(**kwargs)


def test_rejects_invalid_optimiser() -> None:
    """The optimiser must be an instantiated Optax gradient transformation."""
    kwargs: dict[str, Any] = {
        "coreset_size": 1,
        "random_key": jr.key(0),
        "kernel": LinearKernel(),
        "optimiser": None,
    }
    with pytest.raises(TypeError, match="optimiser"):
        GradientFlow(**kwargs)


@pytest.mark.parametrize("scheduled_scale", [-0.1, float("inf")])
def test_rejects_invalid_noise_schedule_values(
    jit_variant: Callable[[Callable], Callable], scheduled_scale: float
) -> None:
    """Noise schedules are validated at the optimisation step where they are used."""
    solver = GradientFlow(
        1,
        jr.key(0),
        LinearKernel(),
        num_iterations=1,
        noise_scale=optax.constant_schedule(scheduled_scale),
    )
    with pytest.raises(RuntimeError, match="Noise scale schedule"):
        result, _ = jit_variant(solver.reduce)(Data(jnp.array([[0.0], [1.0]])))
        result.points.data.block_until_ready()


def test_invalid_shapes_and_supervision() -> None:
    """Unsupported shapes and supervised inputs are not silently reinterpreted."""
    solver = GradientFlow(2, jr.key(0), LinearKernel())
    with pytest.raises(ValueError, match="non-empty two-dimensional"):
        solver.reduce(Data(jnp.empty((0, 1))))
    with pytest.raises(ValueError, match="non-empty two-dimensional"):
        solver.reduce(Data(jnp.ones((3, 2, 2))))
    with pytest.raises(ValueError, match="cannot exceed"):
        solver.reduce(Data(jnp.ones((1, 1))))
    with pytest.raises(TypeError, match="unsupervised"):
        solver.reduce(SupervisedData(jnp.ones((3, 1)), jnp.ones((3, 1))))
    with pytest.raises(ValueError, match="real coordinates"):
        solver.reduce(Data(jnp.array([[1j], [2j]])))
    with pytest.raises(ValueError, match="size must match"):
        solver.refine(PseudoCoreset(Data(jnp.ones((1, 1))), Data(jnp.ones((3, 1)))))
    with pytest.raises(ValueError, match="dimensions must match"):
        solver.refine(PseudoCoreset(Data(jnp.ones((2, 2))), Data(jnp.ones((3, 1)))))


def test_mixed_coordinate_and_weight_precision(
    jit_variant: Callable[[Callable], Callable],
) -> None:
    """Promote coordinates when higher-precision weights affect the update."""
    previous_setting = jax.config.read("jax_enable_x64")
    try:
        jax.config.update("jax_enable_x64", True)
        target = Data(
            jnp.array([[0.0], [2.0], [4.0]], dtype=jnp.float32),
            jnp.array([1.0, 2.0, 1.0], dtype=jnp.float64),
        )
        particles = PseudoCoreset(
            Data(jnp.array([[0.0], [1.0]], dtype=jnp.float32)), target
        )
        solver = GradientFlow(2, jr.key(0), LinearKernel(), num_iterations=1)
        result, _ = jit_variant(solver.refine)(particles)
        np.testing.assert_allclose(result.points.data, [[0.15], [1.15]])
        assert result.points.data.dtype == jnp.float64
        np.testing.assert_array_equal(result.points.weights, particles.points.weights)
    finally:
        jax.config.update("jax_enable_x64", previous_setting)
