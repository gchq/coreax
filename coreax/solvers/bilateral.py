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

"""Solvers for constructing bilateral coresets."""

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
from coreax.solvers.base import ExplicitSizeSolver
from coreax.solvers.gradient import GradientFlow, UnsupervisedGradientState
from coreax.util import KeyArrayLike


class LinearProjectionState(eqx.Module):
    """
    Optimisation information for :class:`UnsupervisedBilateralLinear`.

    :param projection: Array containing the linear projection
    :param losses: Array of loss values at each iteration. Note that due to
        early stopping, not all entries may be filled. To remove :data:`nan` values, use
        ``losses[~jnp.isnan(losses)]``.
    :param gradient_norms: Array of gradient norms at each iteration. Note
        that due to early  stopping, not all entries may be filled. To remove
        :data:`nan` values, use ``gradient_norms[~jnp.isnan(gradient_norms)]``.
    :param opt_state: Optimiser state
    """

    projection: Array
    losses: Array
    gradient_norms: Array
    opt_state: OptState


class UnsupervisedBilateralLinearState(eqx.Module):
    """
    Optimisation information for :class:`UnsupervisedBilateralLinear`.

    :param projection_state: Instance of :class:`LinearProjectionState`
    :param coreset_state: Instance of :class:`UnsupervisedGradientState`
    """

    projection_state: LinearProjectionState
    coreset_state: UnsupervisedGradientState


class UnsupervisedBilateralLinear(
    ExplicitSizeSolver[PseudoCoreset, Data, UnsupervisedBilateralLinearState]
):
    r"""
    Bilateral Distribution Compression with a Linear projection.

    .. warning::
        This class is only suitable for use with unlabelled data.

    :param random_key: Key for random number generation

    :param intrinsic_dimension: The desired size of the intrinsic dimension.
    :param projection_kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`
    :param projection_optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the Adam optimiser with a constant step schedule of 1e-2.
    :param orthonormal: A boolean representing whether to keep the projection on the
        Stiefel manifold or not. Defaults to :data:`True`.
    :param max_projection_iterations: An integer representing the maximum permitted
        number of gradient steps. Defaults to :math:`100`.
    :param num_projection_seeds: Number of initial seeds to check for optimisation.
        Defaults to :data:`None`, indicating  PCA is used.
    :param projection_convergence_parameter: Parameter to decide when gradient descent
        has converged. Defaults to :math:`1e-3`.
    :param projection_batch_size: Number of data points used to estimate RMMD, defaults
        to 128.
    :param coreset_size: The desired size of the solved coreset
    :param coreset_kernel: :class:`~coreax.kernels.ScalarValuedKernel` class
        implementing a kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}` which is
        reliant on a `length_scale` parameter.
    :param coreset_optimiser: A :class:`~optax.GradientTransformation` optimiser.
        Defaults to the Adam optimiser with a constant step schedule of 1e-2.
    :param max_coreset_iterations: An integer representing the maximum permitted number
        of gradient steps. Defaults to :math:`100`.
    :param num_coreset_seeds: Number of initial seeds to check for optimisation.
        Defaults to :data:`None`, indicating  a single random sample is used.
    :param coreset_convergence_parameter: Parameter to decide when gradient descent has
        converged. Defaults to :math:`1e-3`.
    :param track_info: Whether or not to print store optimisation information.
        Defaults to :data:`False`.
    """

    random_key: KeyArrayLike
    intrinsic_dimension: int = eqx.field(converter=int)
    coreset_size: int = eqx.field(converter=int)
    projection_kernel: ScalarValuedKernel
    coreset_kernel: ScalarValuedKernel
    projection_optimiser: optax.GradientTransformation = optax.adam(
        optax.constant_schedule(1e-2)
    )
    orthonormal: bool = True
    projection_convergence_parameter: float = 1e-3
    max_projection_iterations: int = 100
    num_projection_seeds: int | None = None
    projection_batch_size: int = 128
    coreset_optimiser: optax.GradientTransformation = optax.adam(
        optax.constant_schedule(1e-2)
    )
    coreset_convergence_parameter: float = 1e-3
    max_coreset_iterations: int = 100
    num_coreset_seeds: int | None = None
    track_info: bool = False

    @staticmethod
    def _project_to_tangent_space(
        projection: Shaped[Array, " d p"],
        array: Shaped[Array, " d p"],
    ) -> Shaped[Array, " d p"]:
        """Project `array` onto the tangent of the Stiefel manifold at `projection`."""
        product = projection.T @ array
        return array - 1 / 2 * projection @ (product + product.T)

    @staticmethod
    def _retract_to_manifold(
        projection: Shaped[Array, " d p"],
    ) -> Shaped[Array, " d p"]:
        """Retract `projection` to Stiefel manifold using Q from QR decomposition."""
        return jnp.linalg.qr(projection)[0]

    @staticmethod
    def _reconstruction_maximum_mean_discrepancy(
        batch: Shaped[Array, " B d"],
        projection: Shaped[Array, " d p"],
        reconstruction_kernel: ScalarValuedKernel,
    ) -> Shaped[Array, ""]:
        r"""
        Compute the MMD between the original data set and the reconstructed dataset.

        .. math::

            \Vert \mu_{\mathbb{P}_{X} - \mu_{\mathbb{P}_{XVV^T}} \Vert_{\mathcal{H}_k}
        """
        # Rename for better formatting
        b, v = batch, projection

        # Project and reconstruct the batch
        b_reconstructed = b @ v @ v.T

        # Estimate reconstruction MMD
        term_1 = reconstruction_kernel.compute(b, b).mean()
        term_2 = reconstruction_kernel.compute(b_reconstructed, b).mean()
        term_3 = reconstruction_kernel.compute(b_reconstructed, b_reconstructed).mean()

        return term_1 - 2 * term_2 + term_3

    def _initialise_projection_with_gaussian_sketch(
        self,
        random_key: KeyArrayLike,
        num_seeds: int,
        ambient_dimension: int,
        batch: Shaped[Array, " B d_x"],
    ) -> Shaped[Array, " d p"]:
        """Initialise the projection with Gaussian sketch."""
        # Sample Gaussian sketches according to size of ambient dimension
        if num_seeds == 1:
            return jr.normal(
                random_key,
                shape=(ambient_dimension, self.intrinsic_dimension),
            ) / jnp.sqrt(ambient_dimension)

        initial_vs = jr.normal(
            random_key,
            shape=(num_seeds, ambient_dimension, self.intrinsic_dimension),
        ) / jnp.sqrt(ambient_dimension)

        # If orthonormal, retract to Stiefel manifold
        if self.orthonormal:
            initial_vs = vmap(self._retract_to_manifold)(initial_vs)

        # Find the optimal initialisation according to reconstruction MMD
        reconstruction_errors = vmap(
            self._reconstruction_maximum_mean_discrepancy, in_axes=(None, 0, None)
        )(batch, initial_vs, self.projection_kernel)
        return initial_vs[jnp.argmin(reconstruction_errors)]

    def _initialise_projection_with_pca(
        self, data: Shaped[Array, " n d"]
    ) -> Shaped[Array, " d p"]:
        """Initialise the projection with PCA."""
        # Ensure zero mean
        data -= data.mean(axis=0)

        # Compute the empirical covariance matrix
        sigma = 1 / (data.shape[0] - 1) * data.T @ data

        # Do SVD and return projection matrix
        sig_svd = jnp.linalg.svd(sigma, compute_uv=True)
        return sig_svd.U[:, : self.intrinsic_dimension]

    def _projection_step(
        self,
        batch: Shaped[Array, "n d"],
        projection: Shaped[Array, "m d"],
        projection_opt_state: OptState,
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
        # Rename for better formatting
        b, v = batch, projection

        # Evaluate gradient wrt V
        v_gradient = grad(self._reconstruction_maximum_mean_discrepancy, argnums=1)(
            b, v, self.projection_kernel
        )

        # Do gradient step on v
        v_update, projection_opt_state = self.projection_optimiser.update(
            updates=v_gradient, state=projection_opt_state, params=v
        )
        v_ = jnp.asarray(optax.apply_updates(v, v_update))

        return v_, v_gradient, projection_opt_state

    def _stiefel_projection_step(
        self,
        batch: Shaped[Array, " B d"],
        projection: Shaped[Array, " d p"],
        projection_opt_state: OptState,
    ) -> tuple[Shaped[Array, " d p"], Shaped[Array, " d p"], OptState]:
        """Do one gradient step on the projection, restricted to Stiefel manifold."""
        # Rename for better formatting
        b, v = batch, projection

        # Evaluate gradient wrt V
        v_gradient = grad(self._reconstruction_maximum_mean_discrepancy, argnums=1)(
            b, v, self.projection_kernel
        )

        # Project the gradient to the tangent space defined by the current iterate v
        projected_v_gradient = self._project_to_tangent_space(v, v_gradient)

        # Do gradient step on v
        v_update, projection_opt_state = self.projection_optimiser.update(
            updates=projected_v_gradient, state=projection_opt_state, params=v
        )
        v_ = jnp.asarray(optax.apply_updates(v, v_update))

        # Retract V_ to Stiefel manifold
        v_ = self._retract_to_manifold(v_)

        return v_, v_gradient, projection_opt_state

    def reduce(
        self,
        dataset: Data,
        solver_state: UnsupervisedBilateralLinearState | None = None,
    ) -> tuple[PseudoCoreset[Data], UnsupervisedBilateralLinearState]:
        r"""
        Reduce 'dataset' in both dimension and number of observations.

        :param dataset: The data to generate the coreset from.
        :param solver_state: Solution state information, including optimiser state.
        :return: a tuple of the solved coreset and solver state information
        """
        del solver_state
        x, n = dataset.data, len(dataset.data)

        # Initialise the projection
        if self.num_projection_seeds is None:
            v = self._initialise_projection_with_pca(x)
        else:
            # Get a batch to compute an initial projection
            batch_key, seed_key = jr.split(self.random_key)
            batch_indices = jr.choice(
                batch_key, n, shape=(self.projection_batch_size,), replace=False
            )
            batch = x[batch_indices]

            v = self._initialise_projection_with_gaussian_sketch(
                seed_key, self.num_projection_seeds, x.shape[1], batch
            )

        # Initialise storage and optimisation state
        projection_losses = jnp.full((self.max_projection_iterations), jnp.nan)
        projection_gradient_norms = jnp.full((self.max_projection_iterations), jnp.nan)
        projection_opt_state = self.projection_optimiser.init(v)

        def cond_fun(carry):
            """Check when to stop optimisation."""
            i, _, _, _, _, grad_norm, _ = carry
            not_done = i < self.max_projection_iterations
            not_converged = grad_norm >= self.projection_convergence_parameter
            return jnp.logical_and(not_done, not_converged)

        def body_fun(carry):
            """Do optimisation step."""
            (
                i,
                projection,
                projection_opt_state,
                projection_losses,
                projection_gradient_norms,
                _,
                batch_key,
            ) = carry

            # Sample a batch
            batch = x[
                jr.choice(
                    batch_key, n, shape=(self.projection_batch_size,), replace=False
                )
            ]
            # Do a step on v
            projection_step = self._projection_step
            if self.orthonormal:
                projection_step = self._stiefel_projection_step
            projection, projection_grad, projection_opt_state = projection_step(
                batch=batch,
                projection=projection,
                projection_opt_state=projection_opt_state,
            )

            # Store optimisation information
            projection_grad_norm = jnp.linalg.norm(projection_grad)
            if self.track_info:
                projection_losses = projection_losses.at[i].set(
                    self._reconstruction_maximum_mean_discrepancy(
                        batch=batch,
                        projection=projection,
                        reconstruction_kernel=self.projection_kernel,
                    )
                )
                projection_gradient_norms = projection_gradient_norms.at[i].set(
                    projection_grad_norm
                )

            return (
                i + 1,
                projection,
                projection_opt_state,
                projection_losses,
                projection_gradient_norms,
                projection_grad_norm,
                jr.split(batch_key)[0],
            )

        init_carry = (
            0,
            v,
            projection_opt_state,
            projection_losses,
            projection_gradient_norms,
            jnp.inf,
            jr.split(self.random_key)[0],
        )
        (
            _,
            v,
            projection_opt_state,
            projection_losses,
            projection_gradient_norms,
            _,
            _,
        ) = lax.while_loop(cond_fun, body_fun, init_carry)

        projection_state = LinearProjectionState(
            v,
            projection_losses,
            projection_gradient_norms,
            projection_opt_state,
        )

        solver = GradientFlow(
            random_key=self.random_key,
            coreset_size=self.coreset_size,
            kernel=self.coreset_kernel,
            optimiser=self.coreset_optimiser,
            convergence_parameter=self.coreset_convergence_parameter,
            max_iterations=self.max_coreset_iterations,
            num_seeds=self.num_coreset_seeds,
            track_info=self.track_info,
        )
        coreset, coreset_state = solver.reduce(Data(x @ v))

        return coreset, UnsupervisedBilateralLinearState(
            projection_state, coreset_state
        )
