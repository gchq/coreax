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

from coreax.coreset import PseudoCoreset
from coreax.data import Data
from coreax.kernels import ScalarValuedKernel
from coreax.solvers.base import ExplicitSizeSolver, PseudoRefinementSolver
from coreax.util import KeyArrayLike


class UnsupervisedState(eqx.Module):
    """
    Optimisation information for :class:`_UnsupervisedSolver`.

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


class _UnsupervisedSolver(
    PseudoRefinementSolver[Data, UnsupervisedState], ExplicitSizeSolver
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

    def reduce(
        self, dataset: Data, solver_state: UnsupervisedState | None = None
    ) -> tuple[PseudoCoreset[Data], UnsupervisedState]:
        initial_coreset = self._initialise(dataset=dataset.data)
        return self.refine(PseudoCoreset.build(initial_coreset, dataset), solver_state)

    def refine(
        self,
        coreset: PseudoCoreset[Data],
        solver_state: UnsupervisedState | None = None,
    ) -> tuple[PseudoCoreset[Data], UnsupervisedState]:
        r"""
        Reduce 'dataset' to a coreset - solve the coreset problem.

        :param dataset: The data to generate the coreset from.
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step information.
        :return: a tuple of the solved coreset and intermediate solver state information
        """
        # Unpack current coreset and target dataset
        coreset_, data = coreset.points.data, coreset.pre_coreset_data.data

        # Initialise storage for optimisation information
        losses = jnp.full((self.max_iterations), jnp.nan)
        gradient_norms = jnp.full((self.max_iterations), jnp.nan)

        # Initialise optimiser state
        if solver_state is None:
            opt_state = self.optimiser.init(coreset_)
            initial_loss = self._loss_function(target=data, coreset=coreset_)
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

        init_carry = (0, coreset_, opt_state, losses, gradient_norms, jnp.inf)
        _, coreset_, opt_state, losses, gradient_norms, _ = lax.while_loop(
            cond_fun, body_fun, init_carry
        )

        if solver_state is None:
            state = UnsupervisedState(
                jnp.hstack((initial_loss, losses)), gradient_norms, opt_state
            )
        else:
            state = UnsupervisedState(
                jnp.hstack((solver_state.losses, losses)),
                jnp.hstack((solver_state.gradient_norms, gradient_norms)),
                opt_state,
            )

        return PseudoCoreset.build(coreset_, data), state


class GradientFlow(_UnsupervisedSolver):
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
