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

"""Joint kernel inducing points for supervised distribution compression."""

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Shaped
from typing_extensions import override

from coreax.coreset import PseudoCoreset
from coreax.data import SupervisedData
from coreax.kernels import ScalarValuedKernel
from coreax.solvers.base import ExplicitSizeSolver
from coreax.util import KeyArrayLike


def _joint_inducing_objective(
    feature_points: Shaped[Array, "m d"],
    response_points: Shaped[Array, "m p"],
    target: SupervisedData,
    feature_kernel: ScalarValuedKernel,
    response_kernel: ScalarValuedKernel,
) -> Shaped[Array, ""]:
    r"""Return the variable part of empirical squared JMMD from equation (4)."""
    target = target.normalize()
    coreset_size = feature_points.shape[0]
    coreset_weights = jnp.ones(coreset_size, dtype=feature_points.dtype) / coreset_size

    feature_cc = feature_kernel.compute(feature_points, feature_points)
    response_cc = response_kernel.compute(response_points, response_points)
    feature_cd = feature_kernel.compute(feature_points, target.data)
    response_cd = response_kernel.compute(response_points, target.supervision)

    within = coreset_weights @ (feature_cc * response_cc) @ coreset_weights
    cross = coreset_weights @ (feature_cd * response_cd) @ target.weights
    return within - 2 * cross


class JointKernelInducingPoints(
    ExplicitSizeSolver[PseudoCoreset[SupervisedData], SupervisedData, None]
):
    r"""
    Joint Kernel Inducing Points (JKIP) for labelled data.

    Implements Algorithm 3 of :cite:`broadbent2026conditional`. The solver draws
    ``num_initialisations`` uniformly sampled candidate compressed sets, chooses the
    candidate with the smallest empirical joint-kernel objective, then jointly optimises
    all feature and response inducing points with an Optax gradient optimiser.

    For target data :math:`(X,Y)` and inducing points
    :math:`(\widetilde X,\widetilde Y)`, the optimised part of empirical squared JMMD is

    .. math::

        \frac{1}{m^2}\operatorname{Tr}(K_{\widetilde X\widetilde X}
        L_{\widetilde Y\widetilde Y})
        - \frac{2}{mn}\operatorname{Tr}(K_{\widetilde X X}L_{Y\widetilde Y}).

    Dataset weights are normalised and used in the target expectation, extending the
    paper's uniform empirical distribution. The returned pseudo-coreset has uniform
    weights. Feature and response kernels must be differentiable with respect to their
    inputs. For discrete response spaces, use a differentiable relaxation or a
    gradient-free method instead.

    :param coreset_size: Number of inducing pairs to optimise
    :param random_key: Key used to sample initial candidate sets
    :param feature_kernel: Kernel on the feature space
    :param response_kernel: Kernel on the response space
    :param optimiser: Optax optimiser used for joint refinement
    :param num_iterations: Number of joint optimisation steps
    :param num_initialisations: Number of random candidate sets considered before
        optimisation
    """

    random_key: KeyArrayLike
    feature_kernel: ScalarValuedKernel
    response_kernel: ScalarValuedKernel
    optimiser: optax.GradientTransformation = optax.adam(1e-2)
    num_iterations: int = 100
    num_initialisations: int = 4

    def __check_init__(self) -> None:
        """Validate JKIP configuration."""
        super().__check_init__()
        if not isinstance(self.feature_kernel, ScalarValuedKernel):
            raise TypeError("'feature_kernel' must be a ScalarValuedKernel")
        if not isinstance(self.response_kernel, ScalarValuedKernel):
            raise TypeError("'response_kernel' must be a ScalarValuedKernel")
        if not isinstance(self.optimiser, optax.GradientTransformation):
            raise TypeError("'optimiser' must be an optax.GradientTransformation")
        if not isinstance(self.num_iterations, int) or self.num_iterations < 0:
            raise ValueError("'num_iterations' must be a non-negative integer")
        if (
            not isinstance(self.num_initialisations, int)
            or self.num_initialisations < 1
        ):
            raise ValueError("'num_initialisations' must be a positive integer")

    @override
    def reduce(
        self, dataset: SupervisedData, solver_state: None = None
    ) -> tuple[PseudoCoreset[SupervisedData], None]:
        """
        Optimise a supervised pseudo-coreset against empirical JMMD.

        Candidate initialisations are sampled without replacement within each set. The
        returned feature and response points are floating point arrays because gradient
        refinement may move them away from observations in the original dataset.

        :param dataset: Supervised target data
        :param solver_state: Unused; JKIP is fully determined by ``random_key``
        :return: Optimised pseudo-coreset and ``None``
        """
        if not isinstance(dataset, SupervisedData):
            raise TypeError("'dataset' must be SupervisedData")
        if self.coreset_size > len(dataset):
            raise ValueError("'coreset_size' cannot exceed the dataset size")

        dtype = jnp.result_type(dataset.data, dataset.supervision, 0.0)
        target = SupervisedData(
            dataset.data.astype(dtype),
            dataset.supervision.astype(dtype),
            jnp.asarray(dataset.weights, dtype=dtype),
        )
        checked_weights = eqx.error_if(
            target.weights,
            (~jnp.all(jnp.isfinite(target.weights)))
            | (~jnp.all(target.weights >= 0))
            | (jnp.sum(target.weights) <= 0),
            "Dataset weights must be finite and non-negative with positive total mass",
        )
        target = eqx.tree_at(lambda value: value.weights, target, checked_weights)

        candidate_indices = self._sample_candidate_indices(len(target))
        candidate_features = target.data[candidate_indices]
        candidate_responses = target.supervision[candidate_indices]
        candidate_scores = jax.vmap(
            lambda features, responses: _joint_inducing_objective(
                features,
                responses,
                target,
                self.feature_kernel,
                self.response_kernel,
            )
        )(candidate_features, candidate_responses)
        best = jnp.argmin(candidate_scores)
        params = (candidate_features[best], candidate_responses[best])
        optimiser_state = self.optimiser.init(params)

        def step(
            _: int,
            carry: tuple[tuple[Array, Array], optax.OptState],
        ) -> tuple[tuple[Array, Array], optax.OptState]:
            current, current_state = carry

            def loss(value: tuple[Array, Array]) -> Array:
                return _joint_inducing_objective(
                    value[0],
                    value[1],
                    target,
                    self.feature_kernel,
                    self.response_kernel,
                )

            gradients = jax.grad(loss)(current)
            updates, next_state = self.optimiser.update(
                gradients, current_state, current
            )
            updated = cast(tuple[Array, Array], optax.apply_updates(current, updates))
            return updated, next_state

        params, _ = jax.lax.fori_loop(
            0, self.num_iterations, step, (params, optimiser_state)
        )
        points = SupervisedData(params[0], params[1])
        return PseudoCoreset(points, dataset), solver_state

    def _sample_candidate_indices(self, dataset_size: int) -> Shaped[Array, "c m"]:
        """Sample independent candidate sets uniformly without replacement."""
        keys = jr.split(self.random_key, self.num_initialisations)
        return jax.vmap(
            lambda key: jr.choice(
                key,
                dataset_size,
                shape=(self.coreset_size,),
                replace=False,
            )
        )(keys)
