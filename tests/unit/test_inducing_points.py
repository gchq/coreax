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

"""Tests for joint kernel inducing points."""

# pylint: disable=protected-access

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

from coreax.data import Data, SupervisedData
from coreax.kernels import LinearKernel, SquaredExponentialKernel
from coreax.metrics import JMMD
from coreax.solvers import JointKernelInducingPoints
from coreax.solvers.inducing_points import (
    _joint_inducing_objective,  # noqa: PLC2701
)

CORESET_SIZE = 3


def _dataset() -> SupervisedData:
    return SupervisedData(
        jnp.array([[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]]),
        jnp.array([[4.0], [1.0], [0.0], [1.0], [4.0], [9.0]]),
        jnp.array([1.0, 2.0, 1.0, 2.0, 1.0, 1.0]),
    )


def _solver(**kwargs) -> JointKernelInducingPoints:
    parameters = {
        "coreset_size": CORESET_SIZE,
        "random_key": jr.key(123),
        "feature_kernel": SquaredExponentialKernel(length_scale=1.5),
        "response_kernel": SquaredExponentialKernel(length_scale=2.0),
        "optimiser": optax.adam(0.03),
        "num_iterations": 40,
        "num_initialisations": 4,
    }
    parameters.update(kwargs)
    return JointKernelInducingPoints(**parameters)


def test_objective_is_jmmd_without_target_constant() -> None:
    """Match the paper objective to squared JMMD up to its fixed target term."""
    target = _dataset()
    features = jnp.array([[-1.5], [0.5], [2.5]])
    responses = jnp.array([[2.0], [0.5], [6.0]])
    feature_kernel = SquaredExponentialKernel(length_scale=1.5)
    response_kernel = SquaredExponentialKernel(length_scale=2.0)
    candidate = SupervisedData(features, responses)
    target_normalised = target.normalize()
    target_constant = (
        target_normalised.weights
        @ (
            feature_kernel.compute(target.data, target.data)
            * response_kernel.compute(target.supervision, target.supervision)
        )
        @ target_normalised.weights
    )

    objective = _joint_inducing_objective(
        features, responses, target, feature_kernel, response_kernel
    )
    squared_jmmd = JMMD(feature_kernel, response_kernel).compute(target, candidate) ** 2

    assert objective == pytest.approx(squared_jmmd - target_constant, abs=1e-6)


def test_initialisation_chooses_best_random_candidate() -> None:
    """Choose the lowest-objective candidate before gradient refinement."""
    target = _dataset()
    solver = _solver(num_iterations=0)
    indices = solver._sample_candidate_indices(len(target))  # noqa: SLF001
    scores = jnp.asarray(
        [
            _joint_inducing_objective(
                target.data[index],
                target.supervision[index],
                target,
                solver.feature_kernel,
                solver.response_kernel,
            )
            for index in indices
        ]
    )

    result, state = solver.reduce(target)
    best = indices[jnp.argmin(scores)]

    assert state is None
    assert isinstance(result.points, SupervisedData)
    assert result.points.data == pytest.approx(target.data[best])
    assert result.points.supervision == pytest.approx(target.supervision[best])


def test_candidate_sets_have_unique_rows() -> None:
    """Subsample without replacement independently within every candidate set."""
    indices = _solver()._sample_candidate_indices(6)  # noqa: SLF001

    assert indices.shape == (4, CORESET_SIZE)
    for row in indices:
        assert len(jnp.unique(row)) == CORESET_SIZE


def test_reduction_improves_joint_objective(
    jit_variant: Callable[[Callable], Callable],
) -> None:
    """Jointly move inducing features and responses to reduce empirical JMMD."""
    target = _dataset()
    initial_solver = _solver(num_iterations=0)
    solver = _solver()
    initial, _ = initial_solver.reduce(target)
    initial_points = initial.points
    assert isinstance(initial_points, SupervisedData)
    initial_objective = _joint_inducing_objective(
        initial_points.data,
        initial_points.supervision,
        target,
        solver.feature_kernel,
        solver.response_kernel,
    )

    result, state = jit_variant(solver.reduce)(target)
    points = result.points
    assert isinstance(points, SupervisedData)
    final_objective = _joint_inducing_objective(
        points.data,
        points.supervision,
        target,
        solver.feature_kernel,
        solver.response_kernel,
    )

    assert state is None
    assert len(result) == CORESET_SIZE
    assert eqx.tree_equal(result.pre_coreset_data, target)
    assert final_objective < initial_objective
    assert jnp.all(jnp.isfinite(points.data))
    assert jnp.all(jnp.isfinite(points.supervision))


def test_reduction_is_reproducible() -> None:
    """Use the supplied random key deterministically."""
    target = _dataset()
    solver = _solver()

    first, _ = solver.reduce(target)
    second, _ = solver.reduce(target)

    assert first.points.data == pytest.approx(second.points.data)
    assert isinstance(first.points, SupervisedData)
    assert isinstance(second.points, SupervisedData)
    assert first.points.supervision == pytest.approx(second.points.supervision)


def test_weighted_target_changes_objective() -> None:
    """Use normalised target weights in the empirical expectation."""
    target = _dataset()
    uniform_target = SupervisedData(target.data, target.supervision)
    features = target.data[:3]
    responses = target.supervision[:3]
    kernel = LinearKernel()

    weighted = _joint_inducing_objective(features, responses, target, kernel, kernel)
    uniform = _joint_inducing_objective(
        features, responses, uniform_target, kernel, kernel
    )

    assert weighted != pytest.approx(uniform)


def test_rejects_oversized_coreset() -> None:
    """Require each random candidate to be a true subsample."""
    solver = JointKernelInducingPoints(
        7,
        jr.key(0),
        SquaredExponentialKernel(),
        SquaredExponentialKernel(),
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        solver.reduce(_dataset())


def test_rejects_unsupervised_data() -> None:
    """Require paired feature and response observations."""
    solver = _solver()
    with pytest.raises(TypeError, match="SupervisedData"):
        solver.reduce(Data(jnp.arange(6.0)))  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_iterations": -1},
        {"num_initialisations": 0},
    ],
)
def test_rejects_invalid_configuration(kwargs: dict) -> None:
    """Validate iteration and initialisation counts."""
    with pytest.raises(ValueError):
        _solver(**kwargs)
