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

"""Solvers for constructing pseudocoresets."""

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import grad, lax, vmap
from jaxtyping import Array, Shaped
from optax import OptState
from typing_extensions import override

from coreax.coreset import PseudoCoreset
from coreax.data import Data, SupervisedData
from coreax.kernels import ScalarValuedKernel
from coreax.solvers.base import ExplicitSizeSolver, PseudoRefinementSolver
from coreax.util import KeyArrayLike


class UnsupervisedGradientState(eqx.Module):
    """
    Optimisation information for :class:`_UnsupervisedGradientSolver`.

    :param losses: Array of loss values at each iteration. Note that due to early
        stopping, not all entries may be filled. To remove :data:`nan` values, use
        ``losses[~jnp.isnan(losses)]``.
    :param gradient_norms: Array of gradient norms at each iteration. Note that due to
        early  stopping, not all entries may be filled. To remove :data:`nan` values,
        use ``gradient_norms[~jnp.isnan(gradient_norms)]``.
    :param opt_state: Optimiser state
    """

    losses: Array
    gradient_norms: Array
    opt_state: OptState


class SupervisedGradientState(eqx.Module):
    """
    Optimisation information for :class:`_SupervisedGradientSolver`.

    :param losses: Array of loss values at each iteration. Note that due to early
        stopping, not all entries may be filled. To remove :data:`nan` values, use
        ``losses[~jnp.isnan(losses)]``.
    :param gradient_norms: Array of gradient norms at each iteration. Note that due to
        early  stopping, not all entries may be filled. To remove :data:`nan` values,
        use ``gradient_norms[~jnp.isnan(gradient_norms)]``.
    :param feature_opt_state: Feature optimiser state
    :param response_opt_state: Response optimiser state
    """

    losses: Array
    gradient_norms: Array
    feature_opt_state: OptState
    response_opt_state: OptState


class _UnsupervisedGradientSolver(
    PseudoRefinementSolver[Data, UnsupervisedGradientState],
    ExplicitSizeSolver,
):
    r"""
    Generic class for solving unlabelled coreset problems via gradient descent.

    .. warning::
        This class is only suitable for use with unlabelled data.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the Adam optimiser with a constant step schedule of 1e-2.
    :param max_iterations: An integer representing the maximum permitted number of
        gradient steps. Defaults to :math:`100`.
    :param num_seeds: Number of initial seeds to check for optimisation. Defaults to
        :data:`None`, indicating  a single random sample is used.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param track_info: Whether or not to print store optimisation information.
        Defaults to :data:`False`.
    """

    coreset_size: int = eqx.field(converter=int)
    random_key: KeyArrayLike
    kernel: ScalarValuedKernel
    optimiser: optax.GradientTransformation = optax.adam(optax.constant_schedule(1e-2))
    convergence_parameter: float = 1e-3
    max_iterations: int = 100
    num_seeds: int | None = None
    track_info: bool = False

    @abstractmethod
    def _loss_function(
        self, target: Shaped[Array, "n d"], coreset: Shaped[Array, "m d"]
    ) -> Shaped[Array, ""]:
        """
        Loss function that the solver targets.

        :param target: A two-dimensional array containing the target dataset.
        :param coreset: A two-dimensional array containing the current coreset.
        :return: Estimated value of loss as a two-dimensional array.
        """

    @eqx.filter_jit
    def _step(
        self,
        target: Shaped[Array, "n d"],
        coreset: Shaped[Array, "m d"],
        opt_state: OptState,
    ) -> tuple[
        Shaped[Array, "M d"],
        Shaped[Array, "M d"],
        OptState,
    ]:
        """
        Do a gradient step.

        :param target: A two-dimensional array containing the target dataset.
        :param coreset: A two-dimensional array containing the current coreset.
        :opt_state: Current state of the :class:`~optax.GradientTransformation`
            optimiser.
        :return: Tuple containing updated coreset, gradient of loss wrt coreset points
            and updated optimiser state.
        """
        coreset_grad = grad(self._loss_function, argnums=1)(target, coreset)
        update, opt_state = self.optimiser.update(
            updates=coreset_grad, state=opt_state, params=coreset
        )
        coreset_ = jnp.array(optax.apply_updates(coreset, update))

        return coreset_, coreset_grad, opt_state

    def _initialise(self, dataset: Shaped[Array, "n d"]) -> Shaped[Array, "m d"]:
        """
        Initialise the coreset from the dataset.

        :param dataset: The data to initialise the coreset from.
        :return: A two-dimensional array containing the initial coreset
        """
        dataset_size = dataset.shape[0]

        if self.num_seeds is None:
            # Initialise the coreset with a random subset
            initialisation_indices = jr.choice(
                self.random_key, dataset_size, shape=(self.coreset_size,), replace=False
            )
        else:
            # Get keys to choose seeds for optimisation
            seed_keys = jr.split(self.random_key, num=(self.num_seeds,))

            # Sample sets of indices to check
            seed_indices = vmap(
                lambda key: jr.choice(
                    key, dataset_size, shape=(self.coreset_size,), replace=False
                ),
                in_axes=0,
            )(seed_keys)

            # Extract all possible initial coresets as a 3d array
            seed_coresets = dataset[seed_indices]

            # Compute the loss for each initial coreset
            seed_losses = vmap(
                self._loss_function,
                in_axes=(None, 0),
            )(dataset, seed_coresets)

            # Choose the best coreset and store the loss
            initialisation_indices = seed_indices[jnp.argmin(seed_losses)]
        return dataset[initialisation_indices]

    @override
    def reduce(
        self, dataset: Data, solver_state: UnsupervisedGradientState | None = None
    ) -> tuple[PseudoCoreset[Data], UnsupervisedGradientState]:
        initial_coreset = self._initialise(dataset=dataset.data)
        return self.refine(PseudoCoreset.build(initial_coreset, dataset), solver_state)

    def refine(
        self,
        coreset: PseudoCoreset[Data],
        solver_state: UnsupervisedGradientState | None = None,
    ) -> tuple[PseudoCoreset[Data], UnsupervisedGradientState]:
        r"""
        Refine a coreset via gradient descent.

        .. warning::

            If the input ``coreset`` is smaller than the requested ``coreset_size``,
            it will be padded with extra points. If the input ``coreset`` is larger than
            the requested ``coreset_size``, the extra points will not be optimised and
            will be clipped from the return ``coreset``.

        :param coreset: Coreset to refine.
        :param solver_state: Solution state information, including optimiser state.
        :return: A refined coreset; Relevant solver state information.
        """
        # Unpack current coreset and target dataset
        coreset_points, data = coreset.points.data, coreset.pre_coreset_data.data

        # Pad or clip the coreset if size is incorrect
        if coreset_points.shape[0] > self.coreset_size:
            coreset_points = coreset_points[: self.coreset_size - 1]
        elif coreset_points.shape[0] < self.coreset_size:
            initial_coreset = self._initialise(dataset=data)
            coreset_points = jnp.vstack((coreset_points, initial_coreset[-1]))

        # Initialise storage for optimisation information
        losses = jnp.full((self.max_iterations), jnp.nan)
        gradient_norms = jnp.full((self.max_iterations), jnp.nan)

        # Initialise optimiser state and loss
        if solver_state is None:
            opt_state = self.optimiser.init(coreset_points)
            initial_loss = self._loss_function(target=data, coreset=coreset_points)
        else:
            opt_state = solver_state.opt_state
            initial_loss = jnp.array([])

        def cond_fun(carry):
            """Check when to stop optimisation."""
            i, _, _, _, _, grad_norm = carry
            not_done = i < self.max_iterations
            not_converged = grad_norm >= self.convergence_parameter
            return jnp.logical_and(not_done, not_converged)

        def body_fun(carry):
            """Do optimisation step."""
            i, coreset, opt_state, losses, gradient_norms, _ = carry

            coreset, grads, opt_state = self._step(
                target=data, coreset=coreset, opt_state=opt_state
            )
            grad_norm = jnp.linalg.norm(grads)

            if self.track_info:
                losses = losses.at[i].set(
                    self._loss_function(target=data, coreset=coreset)
                )
                gradient_norms = gradient_norms.at[i].set(grad_norm)

            return i + 1, coreset, opt_state, losses, gradient_norms, grad_norm

        init_carry = (0, coreset_points, opt_state, losses, gradient_norms, jnp.inf)
        _, coreset_points, opt_state, losses, gradient_norms, _ = lax.while_loop(
            cond_fun, body_fun, init_carry
        )

        if solver_state is None:
            state = UnsupervisedGradientState(
                jnp.hstack((initial_loss, losses)), gradient_norms, opt_state
            )
        else:
            state = UnsupervisedGradientState(
                jnp.hstack((solver_state.losses, losses)),
                jnp.hstack((solver_state.gradient_norms, gradient_norms)),
                opt_state,
            )

        return PseudoCoreset.build(coreset_points, data), state


class GradientFlow(_UnsupervisedGradientSolver):
    r"""
    Gradient Flow - a gradient descent coreset solver.

    Gradient Flow (:cite:`arbel2019flow`) is a gradient descent algorithm which learns a
    coreset by targeting the Maximum Mean Discrepancy (MMD) between the true
    distribution, and the distribution of the coreset.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the Adam optimiser with a constant step schedule of 1e-2. Input of
        :data:`None` corresponds to no optimisation.
    :param max_iterations: An integer representing the maximum permitted number of
        gradient steps. Defaults to :math:`100`.
    :param num_seeds: Number of initial seeds to check for optimisation. Defaults to
        :data:`None`, indicating  a single random sample is used.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param track_info: Whether or not to print store optimisation information.
        Defaults to :data:`False`.
    """

    def _loss_function(
        self, target: Shaped[Array, "n d"], coreset: Shaped[Array, "m d"]
    ) -> Shaped[Array, ""]:
        """
        Compute the Gradient Flow loss function.

        :param target: A two-dimensional array containing the target dataset.
        :param coreset: A two-dimensional array containing the current coreset.
        :return: Estimated value of loss as a two-dimensional array.
        """
        term_1 = self.kernel.compute(coreset, coreset).mean()
        term_2 = self.kernel.compute(coreset, target).mean()

        return term_1 - 2 * term_2


class _SupervisedGradientSolver(
    PseudoRefinementSolver[SupervisedData, SupervisedGradientState],
    ExplicitSizeSolver,
):
    r"""
    Generic class for solving unlabelled coreset problems via gradient descent.

    .. warning::
        This class is only suitable for use with unlabelled data.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param feature_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param response_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`r: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}`
    :param feature_optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the Adam optimiser with a constant step schedule of 1e-2.
    :param response_optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the Adam optimiser with a constant step schedule of 1e-2.
    :param max_iterations: An integer representing the maximum permitted number of
        gradient steps. Defaults to :math:`100`.
    :param num_seeds: Number of initial seeds to check for optimisation. Defaults to
        :data:`None`, indicating  a single random sample is used.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param track_info: Whether or not to print store optimisation information.
        Defaults to :data:`False`.
    """

    coreset_size: int = eqx.field(converter=int)
    random_key: KeyArrayLike
    feature_kernel: ScalarValuedKernel
    response_kernel: ScalarValuedKernel
    feature_optimiser: optax.GradientTransformation = optax.adam(
        optax.constant_schedule(1e-2)
    )
    response_optimiser: optax.GradientTransformation = optax.adam(
        optax.constant_schedule(1e-2)
    )
    convergence_parameter: float = 1e-3
    max_iterations: int = 100
    num_seeds: int | None = None
    track_info: bool = False

    @abstractmethod
    def _loss_function(
        self,
        target_features: Shaped[Array, "n d"],
        target_responses: Shaped[Array, "n p"],
        coreset_features: Shaped[Array, "m d"],
        coreset_responses: Shaped[Array, "m p"],
    ) -> Shaped[Array, ""]:
        """
        Loss function that the solver targets.

        :param target_features: A two-dimensional array containing the target features.
        :param target_responses: A two-dimensional array containing the target
            responses.
        :param coreset_features: A two-dimensional array containing the current coreset
        features.
        :param coreset_responses: A two-dimensional array containing the current coreset
        responses.
        :return: Estimated value of loss as a two-dimensional array.
        """

    @eqx.filter_jit
    def _step(
        self,
        target_features: Shaped[Array, "n d"],
        target_responses: Shaped[Array, "n p"],
        coreset_features: Shaped[Array, "m d"],
        coreset_responses: Shaped[Array, "m p"],
        feature_opt_state: OptState,
        response_opt_state: OptState,
    ) -> tuple[
        Shaped[Array, "M d"],
        Shaped[Array, "M p"],
        Shaped[Array, "M d + p"],
        OptState,
        OptState,
    ]:
        """
        Do a gradient step.

        :param target_features: A two-dimensional array containing the target features.
        :param target_responses: A two-dimensional array containing the target
            responses.
        :param coreset_features: A two-dimensional array containing the current coreset
        features.
        :param coreset_responses: A two-dimensional array containing the current coreset
        responses.
        :feature_opt_state: Current state of the :class:`~optax.GradientTransformation`
            optimiser.
        :response_opt_state: Current state of the :class:`~optax.GradientTransformation`
            optimiser.
        :return: Tuple containing updated coreset features and responses, gradient of
            loss wrt coreset features and responses, and updated optimiser states.
        """
        # Do a step on the coreset features
        coreset_feature_grad = grad(self._loss_function, argnums=2)(
            target_features, target_responses, coreset_features, coreset_responses
        )
        feature_update, feature_opt_state = self.feature_optimiser.update(
            updates=coreset_feature_grad,
            state=feature_opt_state,
            params=coreset_features,
        )
        coreset_features_ = jnp.array(
            optax.apply_updates(coreset_features, feature_update)
        )

        # Do a step on the coreset responses
        coreset_response_grad = grad(self._loss_function, argnums=3)(
            target_features, target_responses, coreset_features, coreset_responses
        )
        response_update, response_opt_state = self.response_optimiser.update(
            updates=coreset_response_grad,
            state=response_opt_state,
            params=coreset_responses,
        )
        coreset_responses_ = jnp.array(
            optax.apply_updates(coreset_responses, response_update)
        )

        return (
            coreset_features_,
            coreset_responses_,
            jnp.hstack((coreset_feature_grad, coreset_response_grad)),
            feature_opt_state,
            response_opt_state,
        )

    def _initialise(
        self,
        dataset_features: Shaped[Array, "n d"],
        dataset_responses: Shaped[Array, "n p"],
    ) -> tuple[Shaped[Array, "m d"], Shaped[Array, "m p"]]:
        """
        Initialise the coreset from the dataset.

        :param dataset_features: The data features to initialise the coreset from.
        :param dataset_responses: The data responses to initialise the coreset from.
        :return: A tuple of two-dimensional array containing the initial coreset
            features and responses.
        """
        dataset_size = dataset_features.shape[0]

        if self.num_seeds is None:
            # Initialise the coreset with a random subset
            initialisation_indices = jr.choice(
                self.random_key, dataset_size, shape=(self.coreset_size,), replace=False
            )
        else:
            # Get keys to choose seeds for optimisation
            seed_keys = jr.split(self.random_key, num=(self.num_seeds,))

            # Sample sets of indices to check
            seed_indices = vmap(
                lambda key: jr.choice(
                    key, dataset_size, shape=(self.coreset_size,), replace=False
                ),
                in_axes=0,
            )(seed_keys)

            # Extract all possible initial coresets as a 3d array
            seed_coreset_features = dataset_features[seed_indices]
            seed_coreset_responses = dataset_responses[seed_indices]

            # Compute the loss for each initial coreset
            seed_losses = vmap(
                self._loss_function,
                in_axes=(None, None, 0, 0),
            )(
                dataset_features,
                dataset_responses,
                seed_coreset_features,
                seed_coreset_responses,
            )

            # Choose the best coreset and store the loss
            initialisation_indices = seed_indices[jnp.argmin(seed_losses)]
        return (
            dataset_features[initialisation_indices],
            dataset_responses[initialisation_indices],
        )

    @override
    def reduce(
        self,
        dataset: SupervisedData,
        solver_state: SupervisedGradientState | None = None,
    ) -> tuple[PseudoCoreset[SupervisedData], SupervisedGradientState]:
        initial_coreset_features, initial_coreset_responses = self._initialise(
            dataset_features=dataset.data, dataset_responses=dataset.supervision
        )
        return self.refine(
            PseudoCoreset.build(
                (initial_coreset_features, initial_coreset_responses),
                dataset,
            ),
            solver_state,
        )

    def refine(
        self,
        coreset: PseudoCoreset[SupervisedData],
        solver_state: SupervisedGradientState | None = None,
    ) -> tuple[PseudoCoreset[SupervisedData], SupervisedGradientState]:
        r"""
        Refine a coreset via gradient descent.

        .. warning::

            If the input ``coreset`` is smaller than the requested ``coreset_size``,
            it will be padded with extra points. If the input ``coreset`` is larger than
            the requested ``coreset_size``, the extra points will not be optimised and
            will be clipped from the return ``coreset``.

        :param coreset: Coreset to refine.
        :param solver_state: Solution state information, including optimiser state.
        :return: A refined coreset; Relevant solver state information.
        """
        # Unpack current coreset and target dataset
        coreset_features, coreset_responses, dataset_features, dataset_responses = (
            coreset.points.data,
            coreset.points.supervision,
            coreset.pre_coreset_data.data,
            coreset.pre_coreset_data.supervision,
        )

        # Pad or clip the coreset if size is incorrect
        if coreset_features.shape[0] > self.coreset_size:
            coreset_features = coreset_features[: self.coreset_size - 1]
            coreset_responses = coreset_responses[: self.coreset_size - 1]
        elif coreset_features.shape[0] < self.coreset_size:
            initial_coreset_features, initial_coreset_responses = self._initialise(
                dataset_features=dataset_features, dataset_responses=dataset_responses
            )
            coreset_features = jnp.vstack(
                (coreset_features, initial_coreset_features[-1])
            )
            initial_coreset_responses = jnp.vstack(
                (coreset_features, initial_coreset_responses[-1])
            )

        # Initialise storage for optimisation information
        losses = jnp.full((self.max_iterations), jnp.nan)
        gradient_norms = jnp.full((self.max_iterations), jnp.nan)

        # Initialise optimiser state and loss
        if solver_state is None:
            feature_opt_state = self.feature_optimiser.init(coreset_features)
            response_opt_state = self.response_optimiser.init(coreset_responses)
            initial_loss = self._loss_function(
                target_features=dataset_features,
                target_responses=dataset_responses,
                coreset_features=coreset_features,
                coreset_responses=coreset_responses,
            )
        else:
            feature_opt_state = solver_state.feature_opt_state
            response_opt_state = solver_state.response_opt_state
            initial_loss = jnp.array([])

        def cond_fun(carry):
            """Check when to stop optimisation."""
            i, _, _, _, _, _, _, grad_norm = carry
            not_done = i < self.max_iterations
            not_converged = grad_norm >= self.convergence_parameter
            return jnp.logical_and(not_done, not_converged)

        def body_fun(carry):
            """Do optimisation step."""
            (
                i,
                coreset_features,
                coreset_responses,
                feature_opt_state,
                response_opt_state,
                losses,
                gradient_norms,
                _,
            ) = carry

            (
                coreset_features,
                coreset_responses,
                grads,
                feature_opt_state,
                response_opt_state,
            ) = self._step(
                dataset_features,
                dataset_responses,
                coreset_features,
                coreset_responses,
                feature_opt_state,
                response_opt_state,
            )
            grad_norm = jnp.linalg.norm(grads)

            if self.track_info:
                losses = losses.at[i].set(
                    self._loss_function(
                        target_features=dataset_features,
                        target_responses=dataset_responses,
                        coreset_features=coreset_features,
                        coreset_responses=coreset_responses,
                    )
                )
                gradient_norms = gradient_norms.at[i].set(grad_norm)

            return (
                i + 1,
                coreset_features,
                coreset_responses,
                feature_opt_state,
                response_opt_state,
                losses,
                gradient_norms,
                grad_norm,
            )

        init_carry = (
            0,
            coreset_features,
            coreset_responses,
            feature_opt_state,
            response_opt_state,
            losses,
            gradient_norms,
            jnp.inf,
        )
        (
            _,
            coreset_features,
            coreset_responses,
            feature_opt_state,
            response_opt_state,
            losses,
            gradient_norms,
            _,
        ) = lax.while_loop(cond_fun, body_fun, init_carry)

        if solver_state is None:
            state = SupervisedGradientState(
                jnp.hstack((initial_loss, losses)),
                gradient_norms,
                feature_opt_state,
                response_opt_state,
            )
        else:
            state = SupervisedGradientState(
                jnp.hstack((solver_state.losses, losses)),
                jnp.hstack((solver_state.gradient_norms, gradient_norms)),
                feature_opt_state,
                response_opt_state,
            )

        return PseudoCoreset.build(
            (coreset_features, coreset_responses), (dataset_features, dataset_responses)
        ), state


class JointKernelInducingPoints(_SupervisedGradientSolver):
    r"""
    Joint Kernel Inducing Points - a gradient descent coreset solver.

    Joint Kernel Inducing Points(:cite:`broadbent2026conditional`) is a gradient descent
    algorithm which learns a coreset by targeting the Joint Maximum Mean Discrepancy
    (JMMD) between the true joint distribution, and the joint distribution of the
    coreset.

    :param coreset_size: The desired size of the solved coreset
    :param random_key: Key for random number generation
    :param feature_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param response_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`r: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}`
    :param feature_optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the Adam optimiser with a constant step schedule of 1e-2.
    :param response_optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the Adam optimiser with a constant step schedule of 1e-2.
    :param max_iterations: An integer representing the maximum permitted number of
        gradient steps. Defaults to :math:`100`.
    :param num_seeds: Number of initial seeds to check for optimisation. Defaults to
        :data:`None`, indicating  a single random sample is used.
    :param convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param track_info: Whether or not to print store optimisation information.
        Defaults to :data:`False`.
    """

    def _loss_function(
        self,
        target_features: Shaped[Array, "n d"],
        target_responses: Shaped[Array, "n p"],
        coreset_features: Shaped[Array, "m d"],
        coreset_responses: Shaped[Array, "m p"],
    ) -> Shaped[Array, ""]:
        """
        Compute the Joint Kernel Inducing Points loss function.

        :param target_features: A two-dimensional array containing the target features.
        :param target_responses: A two-dimensional array containing the target
            responses.
        :param coreset_features: A two-dimensional array containing the current coreset
        features.
        :param coreset_responses: A two-dimensional array containing the current coreset
        responses.
        :return: Estimated value of loss as a two-dimensional array.
        """
        term_1 = (
            self.feature_kernel.compute(coreset_features, coreset_features)
            * self.response_kernel.compute(coreset_responses, coreset_responses)
        ).mean()
        term_2 = (
            self.feature_kernel.compute(target_features, coreset_features)
            * self.response_kernel.compute(target_responses, coreset_responses)
        ).mean()
        return term_1 - 2 * term_2
