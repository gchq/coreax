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

"""Particle approximations to maximum mean discrepancy gradient flow."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array
from typing_extensions import override

from coreax.coreset import AbstractCoreset, PseudoCoreset
from coreax.data import Data, SupervisedData
from coreax.kernels import ScalarValuedKernel
from coreax.solvers.base import ExplicitSizeSolver
from coreax.util import KeyArrayLike


class GradientFlowState(eqx.Module):
    """
    Random state for continuing a noisy gradient flow without repeating its noise.

    :param random_key: Next unused random key
    """

    random_key: KeyArrayLike


class GradientFlow(ExplicitSizeSolver[PseudoCoreset[Data], Data, GradientFlowState]):
    r"""
    Move coreset points with maximum mean discrepancy gradient flow.

    Implements the particle update in equation (21) of :cite:`arbel2019maximum`.
    For current particles :math:`x_i`, normalised particle weights :math:`a_j`,
    target points :math:`y_j` and normalised target weights :math:`b_j`, a step is

    .. math::

        x_i' = x_i - \gamma \left[
            \sum_j a_j \nabla_1 k(x_i + \beta u_i, x_j)
            - \sum_j b_j \nabla_1 k(x_i + \beta u_i, y_j)\right].

    Here :math:`u_i` is independent standard Gaussian noise. Setting ``noise_scale``
    to zero gives the deterministic flow. Noise perturbs the gradient evaluation
    location, not the current particle measure or the resulting points directly.
    Point weights are kept fixed; their normalised values determine the flow.

    The kernel must be symmetric and differentiable. A suitable step size depends
    on the kernel and data; arbitrary step sizes need not decrease the MMD.
    Each iteration computes all particle-particle and particle-target gradients.

    :param coreset_size: Number of particles, no larger than the target dataset
    :param random_key: Key for initial sampling and optional gradient noise
    :param kernel: Differentiable scalar-valued kernel
    :param step_size: Non-negative finite Euler step size
    :param num_iterations: Non-negative number of Euler steps per call
    :param noise_scale: Non-negative finite gradient noise scale
    """

    random_key: KeyArrayLike
    kernel: ScalarValuedKernel
    step_size: float = 0.1
    num_iterations: int = 100
    noise_scale: float = 0.0

    def __check_init__(self) -> None:
        """Validate the flow parameters."""
        if not isinstance(self.kernel, ScalarValuedKernel):
            raise TypeError("'kernel' must be a ScalarValuedKernel")
        if not isinstance(self.num_iterations, int) or self.num_iterations < 0:
            raise ValueError("'num_iterations' must be a non-negative integer")
        for name, value in (
            ("step_size", self.step_size),
            ("noise_scale", self.noise_scale),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"'{name}' must be finite and non-negative")

    @override
    def reduce(
        self, dataset: Data, solver_state: GradientFlowState | None = None
    ) -> tuple[PseudoCoreset[Data], GradientFlowState]:
        """
        Sample initial particles from the target, then apply gradient flow.

        Initial points are sampled with replacement using normalised dataset
        weights and receive uniform weights. Use :meth:`refine` to choose a
        different initial distribution. Only finite, real, unsupervised data and
        non-negative weights with a positive finite total are supported.

        :param dataset: Target data to approximate
        :param solver_state: Optional random state to use for sampling and noise
        :return: Reduced data and next random state
        """
        target = _normalised_data(dataset)
        if self.coreset_size > len(dataset):
            raise ValueError("'coreset_size' cannot exceed the dataset size")
        key = self.random_key if solver_state is None else solver_state.random_key
        key, sample_key = jr.split(key)
        indices = jr.choice(
            sample_key,
            len(target),
            shape=(self.coreset_size,),
            replace=True,
            p=target.weights,
        )
        initial = PseudoCoreset(Data(target.data[indices]), dataset)
        return self.refine(initial, GradientFlowState(key))

    def refine(
        self,
        coreset: AbstractCoreset[Data, Data],
        solver_state: GradientFlowState | None = None,
    ) -> tuple[PseudoCoreset[Data], GradientFlowState]:
        """
        Move an existing coreset's points while preserving its weights and target.

        The coreset must have exactly ``coreset_size`` points. Passing the returned
        state to another call continues the noise sequence. Without a state, each
        call starts from ``random_key``.

        :param coreset: Initial coreset, whose original data defines the target
        :param solver_state: Optional random state from an earlier call
        :return: Refined pseudo-coreset and next random state
        """
        if len(coreset) != self.coreset_size:
            raise ValueError("Initial coreset size must match 'coreset_size'")
        target = _normalised_data(coreset.pre_coreset_data)
        initial = _normalised_data(coreset.points)
        if initial.data.shape[1] != target.data.shape[1]:
            raise ValueError("Particle and target feature dimensions must match")
        dtype = jnp.result_type(
            initial.data, target.data, initial.weights, target.weights
        )
        points = initial.data.astype(dtype)
        target_points = target.data.astype(dtype)
        key = self.random_key if solver_state is None else solver_state.random_key

        def step(_: int, carry: tuple[Array, KeyArrayLike]):
            current_points, current_key = carry
            current_key, noise_key = jr.split(current_key)
            evaluation_points = current_points
            if self.noise_scale != 0:
                evaluation_points = current_points + self.noise_scale * jr.normal(
                    noise_key, current_points.shape, dtype=dtype
                )
            repulsion = jnp.einsum(
                "ijd,j->id",
                self.kernel.grad_x(evaluation_points, current_points),
                initial.weights,
            )
            attraction = jnp.einsum(
                "ijd,j->id",
                self.kernel.grad_x(evaluation_points, target_points),
                target.weights,
            )
            return (
                current_points - self.step_size * (repulsion - attraction),
                current_key,
            )

        points, key = jax.lax.fori_loop(0, self.num_iterations, step, (points, key))
        result = PseudoCoreset(
            Data(points, coreset.points.weights), coreset.pre_coreset_data
        )
        return result, GradientFlowState(key)


def _normalised_data(data: Data) -> Data:
    """Validate a probability measure and promote integer coordinates to floats."""
    if isinstance(data, SupervisedData):
        raise TypeError("GradientFlow requires unsupervised Data")
    required_dimensions = 2
    if data.data.ndim != required_dimensions or 0 in data.data.shape:
        raise ValueError("GradientFlow requires a non-empty two-dimensional dataset")
    if jnp.iscomplexobj(data.data) or jnp.iscomplexobj(data.weights):
        raise ValueError("GradientFlow requires real coordinates and weights")
    points = data.data.astype(jnp.result_type(data.data, 0.0))
    weights = jnp.asarray(data.weights, dtype=jnp.result_type(data.weights, 0.0))
    total = jnp.sum(weights)
    checked = eqx.error_if(
        Data(points, weights),
        jnp.any(~jnp.isfinite(points))
        | jnp.any(~jnp.isfinite(weights))
        | jnp.any(weights < 0)
        | ~jnp.isfinite(total)
        | (total <= 0),
        "Data must be finite with non-negative weights and a positive finite total",
    )
    return checked.normalize()
