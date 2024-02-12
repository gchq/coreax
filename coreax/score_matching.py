# © Crown Copyright GCHQ
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""
Classes and associated functionality to perform score matching.

The score function of some data is the derivative of the log-PDF. Score matching
aims to determine a model by matching the score function of the model to that
of the data. Exactly how the score function is modelled is specific to each
child class of the abstract base class :class:`ScoreMatching`.

An example use of score matching arises when trying to work with a
:class:`~coreax.kernel.SteinKernel`, which requires as an input a score function. If
this is known analytically, one can provide an exact score function. In other cases,
approximations to the score function are required, which can be determined using
:class:`ScoreMatching`.

When using :class:`SlicedScoreMatching`, the score function is approximated using a
neural network, whereas in :class:`KernelDensityMatching`, it is approximated by fitting
and then differentiating a kernel density estimate to the data.
"""

# Support annotations with | in Python < 3.10
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial

import jax
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import Array, jit, jvp
from jax import numpy as jnp
from jax import random, tree_util, vmap
from jax.lax import cond, fori_loop
from jax.typing import ArrayLike
from tqdm import tqdm

import coreax.kernel
import coreax.networks
import coreax.validation


class ScoreMatching(ABC):
    """
    Base class for score matching algorithms.

    The score function of some data is the derivative of the log-PDF. Score matching
    aims to determine a model by 'matching' the score function of the model to that
    of the data. Exactly how the score function is modelled is specific to each
    child class of this base class.
    """

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a pytree from the tree definition and the leaves.

        Arrays & dynamic values (children) and auxiliary data (static values) are
        reconstructed. A method to reconstruct the pytree needs to be specified to
        enable JIT decoration of methods inside this class.
        """
        return cls(*children, **aux_data)

    @abstractmethod
    def match(self, x):
        r"""
        Match some model score function to that of a dataset ``x``.

        :param x: The :math:`n \times d` data vectors
        """


class SlicedScoreMatching(ScoreMatching):
    r"""
    Implementation of slice score matching, defined in :cite:p:`ssm`.

    The score function of some data is the derivative of the log-PDF. Score matching
    aims to determine a model by 'matching' the score function of the model to that
    of the data. Exactly how the score function is modelled is specific to each
    child class of :class:`ScoreMatching`.

    With sliced score matching, we train a neural network to directly approximate
    the score function of the data. The approach is outlined in detail in
    :cite:p:`ssm`.

    :param random_key: Key for random number generation
    :param random_generator: Distribution sampler (``key``, ``shape``, ``dtype``)
        :math:`\rightarrow` :class:`~jax.Array`, e.g. distributions in
        :class:`~jax.random`
    :param noise_conditioning: Use the noise conditioning version of score matching.
        Defaults to :data:`True`.
    :param use_analytic: Use the analytic (reduced variance) objective or not.
        Defaults to :data:`False`.
    :param num_random_vectors: The number of random vectors to use per data vector.
        Defaults to 1.
    :param learning_rate: Optimiser learning rate. Defaults to 1e-3.
    :param num_epochs: Number of epochs for training. Defaults to 10.
    :param batch_size: Size of mini-batch. Defaults to 64.
    :param hidden_dim: The ScoreNetwork hidden dimension. Defaults to 128.
    :param optimiser: The optax optimiser to use. Defaults to optax.adam.
    :param num_noise_models: Number of noise models to use in noise
        conditional score matching. Defaults to 100.
    :param sigma: Initial noise standard deviation for noise geometric progression
        in noise conditional score matching. Defaults to 1.
    :param gamma: Geometric progression ratio. Defaults to 0.95.
    """

    def _tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable jit decoration
        of methods inside this class.
        """
        children = (self.random_key,)
        aux_data = {
            "random_generator": self.random_generator,
            "noise_conditioning": self.noise_conditioning,
            "use_analytic": self.use_analytic,
            "num_random_vectors": self.num_random_vectors,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "optimiser": self.optimiser,
            "num_noise_models": self.num_noise_models,
            "sigma": self.sigma,
            "gamma": self.gamma,
        }

        return children, aux_data

    def __init__(
        self,
        random_key: coreax.validation.KeyArrayLike,
        random_generator: Callable,
        noise_conditioning: bool = True,
        use_analytic: bool = False,
        num_random_vectors: int = 1,
        learning_rate: float = 1e-3,
        num_epochs: int = 10,
        batch_size: int = 64,
        hidden_dim: int = 128,
        optimiser: Callable = optax.adamw,
        num_noise_models: int = 100,
        sigma: float = 1.0,
        gamma: float = 0.95,
    ):
        """Define a sliced score matching class."""
        # Validate all inputs
        coreax.validation.validate_is_instance(
            x=random_generator, object_name="random_generator", expected_type=Callable
        )
        coreax.validation.validate_key_array(x=random_key, object_name="random_key")
        noise_conditioning = coreax.validation.cast_as_type(
            x=noise_conditioning, object_name="noise_conditioning", type_caster=bool
        )
        use_analytic = coreax.validation.cast_as_type(
            x=use_analytic, object_name="use_analytic", type_caster=bool
        )
        num_random_vectors = coreax.validation.cast_as_type(
            x=num_random_vectors, object_name="num_random_vectors", type_caster=int
        )
        coreax.validation.validate_in_range(
            x=num_random_vectors,
            object_name="num_random_vectors",
            strict_inequalities=True,
            lower_bound=0,
        )
        learning_rate = coreax.validation.cast_as_type(
            x=learning_rate, object_name="learning_rate", type_caster=float
        )
        coreax.validation.validate_in_range(
            x=learning_rate,
            object_name="learning_rate",
            strict_inequalities=True,
            lower_bound=0,
        )
        num_epochs = coreax.validation.cast_as_type(
            x=num_epochs, object_name="num_epochs", type_caster=int
        )
        coreax.validation.validate_in_range(
            x=num_epochs,
            object_name="num_epochs",
            strict_inequalities=True,
            lower_bound=0,
        )
        batch_size = coreax.validation.cast_as_type(
            x=batch_size, object_name="batch_size", type_caster=int
        )
        coreax.validation.validate_in_range(
            x=batch_size,
            object_name="batch_size",
            strict_inequalities=True,
            lower_bound=0,
        )
        hidden_dim = coreax.validation.cast_as_type(
            x=hidden_dim, object_name="hidden_dim", type_caster=int
        )
        coreax.validation.validate_in_range(
            x=hidden_dim,
            object_name="hidden_dim",
            strict_inequalities=True,
            lower_bound=0,
        )
        coreax.validation.validate_is_instance(
            x=optimiser, object_name="optimiser", expected_type=Callable
        )
        num_noise_models = coreax.validation.cast_as_type(
            x=num_noise_models, object_name="num_noise_models", type_caster=int
        )
        coreax.validation.validate_in_range(
            x=num_noise_models,
            object_name="num_noise_models",
            strict_inequalities=True,
            lower_bound=0,
        )
        sigma = coreax.validation.cast_as_type(
            x=sigma, object_name="sigma", type_caster=float
        )
        coreax.validation.validate_in_range(
            x=sigma, object_name="sigma", strict_inequalities=True, lower_bound=0
        )
        gamma = coreax.validation.cast_as_type(
            x=gamma, object_name="gamma", type_caster=float
        )
        coreax.validation.validate_in_range(
            x=gamma, object_name="gamma", strict_inequalities=True, lower_bound=0
        )

        # Assign all inputs
        self.random_key = random_key
        self.random_generator = random_generator
        self.noise_conditioning = noise_conditioning
        self.use_analytic = use_analytic
        self.num_random_vectors = num_random_vectors
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.optimiser = optimiser
        self.num_noise_models = num_noise_models
        self.sigma = sigma
        self.gamma = gamma

        # Initialise parent
        super().__init__()

    def tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable JIT decoration
        of methods inside this class.
        """
        children = (self.random_key,)
        aux_data = {
            "random_generator": self.random_generator,
            "noise_conditioning": self.noise_conditioning,
            "use_analytic": self.use_analytic,
            "num_random_vectors": self.num_random_vectors,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "optimiser": self.optimiser,
            "num_noise_models": self.num_noise_models,
            "sigma": self.sigma,
            "gamma": self.gamma,
        }

        return children, aux_data

    def _objective_function(
        self,
        random_direction_vector: ArrayLike,
        grad_score_times_random_direction_matrix: ArrayLike,
        score_matrix: ArrayLike,
    ):
        """
        Compute the score matching loss function.

        Two objectives are proposed in :cite:p:`ssm`, a general objective, and a
        simplification with reduced variance that holds for particular assumptions. The
        choice between the two is determined by the boolean ``use_analytic`` defined
        when the class is initiated.

        :param random_direction_vector: :math:`d`-dimensional random vector
        :param grad_score_times_random_direction_matrix: Product of the gradient of
            score_matrix (w.r.t. ``x``) and the random_direction_vector
        :param score_matrix: Gradients of log-density
        :return: Evaluation of score matching objective, see equations 7 and 8 in
            :cite:p:`ssm`
        """
        return cond(
            self.use_analytic,
            self._analytic_objective,
            self._general_objective,
            random_direction_vector,
            grad_score_times_random_direction_matrix,
            score_matrix,
        )

    @staticmethod
    def _analytic_objective(
        random_direction_vector: ArrayLike,
        grad_score_times_random_direction_matrix: ArrayLike,
        score_matrix: ArrayLike,
    ) -> ArrayLike:
        """
        Compute reduced variance score matching loss function.

        This is for use with certain random measures, e.g. normal and Rademacher. If
        this assumption is not true, then
        :meth:`SlicedScoreMatching._general_objective` should be used instead.

        :param random_direction_vector: :math:`d`-dimensional random vector
        :param grad_score_times_random_direction_matrix: Product of the gradient of
            score_matrix (w.r.t. ``x``) and the random_direction_vector
        :param score_matrix: Gradients of log-density
        :return: Evaluation of score matching objective, see equation 8 in :cite:p:`ssm`
        """
        result = (
            random_direction_vector @ grad_score_times_random_direction_matrix
            + 0.5 * score_matrix @ score_matrix
        )
        return result

    @staticmethod
    def _general_objective(
        random_direction_vector: ArrayLike,
        grad_score_times_random_direction_matrix: ArrayLike,
        score_matrix: ArrayLike,
    ) -> ArrayLike:
        """
        Compute general score matching loss function.

        This is to be used when one cannot assume normal or Rademacher random measures
        when using score matching, but has higher variance than
        :meth:`SlicedScoreMatching._analytic_objective` if these assumptions hold.

        :param random_direction_vector: :math:`d`-dimensional random vector
        :param grad_score_times_random_direction_matrix: Product of the gradient of
            score_matrix (w.r.t. ``x``) and the random_direction_vector
        :param score_matrix: Gradients of log-density
        :return: Evaluation of score matching objective, see equation 7 in :cite:p:`ssm`
        """
        result = (
            random_direction_vector @ grad_score_times_random_direction_matrix
            + 0.5 * (random_direction_vector @ score_matrix) ** 2
        )
        return result

    def _loss_element(
        self, x: ArrayLike, v: ArrayLike, score_network: Callable
    ) -> float:
        """
        Compute element-wise loss function.

        Computes the loss function from Section 3.2 of Song el al.'s paper on sliced
        score matching :cite:p:`ssm`.

        :param x: :math:`d`-dimensional data vector
        :param v: :math:`d`-dimensional random vector
        :param score_network: Function that calls the neural network on ``x``
        :return: Objective function output for single ``x`` and ``v`` inputs
        """
        s, u = jvp(score_network, (x,), (v,))
        return self._objective_function(v, u, s)

    def _loss(self, score_network: Callable) -> Callable:
        """
        Compute vector mapped loss function for arbitrary many ``X`` and ``V`` vectors.

        In the context of score matching, we expect to call the objective function on
        the data vector ``x``, random vectors ``v`` and using the score neural
        network.

        :param score_network: Function that calls the neural network on ``x``
        :return: Callable vectorised sliced score matching loss function
        """
        inner = vmap(
            lambda x, v: self._loss_element(x, v, score_network),
            (None, 0),
            0,
        )
        return vmap(inner, (0, 0), 0)

    @jit
    def _train_step(
        self, state: TrainState, x: ArrayLike, random_vectors: ArrayLike
    ) -> tuple[TrainState, float]:
        r"""
        Apply a single training step that updates model parameters using loss gradient.

        :param state: The :class:`~flax.training.train_state.TrainState` object
        :param x: The :math:`n \times d` data vectors
        :param random_vectors: The :math:`n \times m \times d` random vectors
        :return: The updated :class:`~flax.training.train_state.TrainState` object
        """

        def loss(params):
            return self._loss(lambda x_: state.apply_fn({"params": params}, x_))(
                x, random_vectors
            ).mean()

        val, grads = jax.value_and_grad(loss)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, val

    def _noise_conditional_loop_body(
        self,
        i: int,
        obj: float,
        state: TrainState,
        params: dict,
        x: ArrayLike,
        random_vectors: ArrayLike,
        sigmas: ArrayLike,
    ) -> float:
        r"""
        Sum objective function with noise perturbations.

        Inputs are perturbed by Gaussian random noise to improve performance of score
        matching. See :cite:p:`improved_sgm` for details.

        :param i: Loop index
        :param obj: Running objective, i.e. the current partial sum
        :param state: The :class:`~flax.training.train_state.TrainState` object
        :param params: The current iterate parameter settings
        :param x: The :math:`n \times d` data vectors
        :param random_vectors: The :math:`n \times m \times d` random vectors
        :param sigmas: The geometric progression of noise standard deviations
        :return: The updated objective, i.e. partial sum
        """
        # This will generate the same set of random numbers on each function call. We
        #  might want to replace this with random.key(i) to get a unique set each
        #  time.
        # Perturb the inputs with Gaussian noise
        x_perturbed = x + sigmas[i] * random.normal(random.key(0), x.shape)
        obj = (
            obj
            + sigmas[i] ** 2
            * self._loss(lambda x_: state.apply_fn({"params": params}, x_))(
                x_perturbed, random_vectors
            ).mean()
        )
        return obj

    @jit
    def _noise_conditional_train_step(
        self,
        state: TrainState,
        x: ArrayLike,
        random_vectors: ArrayLike,
        sigmas: ArrayLike,
    ) -> tuple[TrainState, float]:
        r"""
        Apply a single training step that updates model parameters using loss gradient.

        :param state: The :class:`~flax.training.train_state.TrainState` object
        :param x: The :math:`n \times d` data vectors
        :param random_vectors: The :math:`n \times m \times d` random vectors
        :param sigmas: Array of noise standard deviations to use in objective function
        :return: The updated :class:`~flax.training.train_state.TrainState` object
        """

        def loss(params):
            body = partial(
                self._noise_conditional_loop_body,
                state=state,
                params=params,
                x=x,
                random_vectors=random_vectors,
                sigmas=sigmas,
            )
            return fori_loop(0, self.num_noise_models, body, 0.0)

        val, grads = jax.value_and_grad(loss)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, val

    # pylint: disable=too-many-locals
    def match(self, x: ArrayLike) -> Callable:
        r"""
        Learn a sliced score matching function from Song et al.'s paper :cite:p:`ssm`.

        We currently use the :class:`~coreax.networks.ScoreNetwork` neural network to
        approximate the score function. Alternative network architectures can be
        considered.

        :param x: The :math:`n \times d` data vectors
        :return: A function that applies the learned score function to input ``x``
        """
        # Validate inputs
        x = coreax.validation.cast_as_type(
            x=x, object_name="x", type_caster=jnp.atleast_2d
        )

        # Setup neural network that will approximate the score function
        num_points, data_dimension = x.shape
        score_network = coreax.networks.ScoreNetwork(self.hidden_dim, data_dimension)

        # Define what a training step consists of - dependent on if we want to include
        #  noise perturbations
        if self.noise_conditioning:
            gammas = self.gamma ** jnp.arange(self.num_noise_models)
            sigmas = self.sigma * gammas
            train_step = partial(self._noise_conditional_train_step, sigmas=sigmas)

        else:
            train_step = self._train_step

        # Define random projection vectors
        generator_key, state_key, batch_key = random.split(self.random_key, 3)
        random_vectors = self.random_generator(
            generator_key,
            (num_points, self.num_random_vectors, data_dimension),
            dtype=float,
        )

        # Define a training state
        state = coreax.networks.create_train_state(
            state_key, score_network, self.learning_rate, data_dimension, self.optimiser
        )

        loop_keys = random.split(batch_key, self.num_epochs)

        # Carry out main training loop to fit the neural network
        for i in tqdm(range(self.num_epochs)):
            # Sample some data-points to pass for this step
            idx = random.randint(loop_keys[i], (self.batch_size,), 0, num_points)

            # Apply training step
            state, val = train_step(state, x[idx, :], random_vectors[idx, :])

            # Print progress (limited to avoid excessive output)
            if i % 10 == 0:
                tqdm.write(f"{i:>6}/{self.num_epochs}: loss {val:<.5f}")

        # Return the learned score function, which is a callable
        return lambda x_: state.apply_fn({"params": state.params}, x_)

    # pylint: enable=too-many-locals


class KernelDensityMatching(ScoreMatching):
    r"""
    Implementation of a kernel density estimate to determine a score function.

    The score function of some data is the derivative of the log-PDF. Score matching
    aims to determine a model by 'matching' the score function of the model to that
    of the data. Exactly how the score function is modelled is specific to each
    child class of this base class.

    With kernel density matching, we approximate the underlying distribution function
    from a dataset using kernel density estimation, and then differentiate this to
    compute an estimate of the score function. A Gaussian kernel is used to construct
    the kernel density estimate.

    :param length_scale: Kernel ``length_scale`` to use when fitting the kernel density
        estimate
    :param kde_data: Set of :math:`n \times d` samples from the underlying distribution
        that are used to build the kernel density estimate
    """

    def __init__(self, length_scale: float, kde_data: ArrayLike):
        """Define the kernel density matching class."""
        # Validate inputs (note that the kernel length_scale is validated upon kernel
        # creation)
        kde_data = coreax.validation.cast_as_type(
            x=kde_data, object_name="kde_data", type_caster=jnp.atleast_2d
        )

        # Define a normalised Gaussian kernel (which is a special cases of the squared
        # exponential kernel) to construct the kernel density estimate
        self.kernel = coreax.kernel.SquaredExponentialKernel(
            length_scale=length_scale,
            output_scale=1.0 / (np.sqrt(2 * np.pi) * length_scale),
        )

        # Hold the data the kernel density estimate will be built from, which will be
        # needed for any call of the score function.
        self.kde_data = kde_data

        # Initialise parent
        super().__init__()

    def tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable JIT decoration
        of methods inside this class.
        """
        children = (self.kde_data,)
        aux_data = {"kernel": self.kernel}

        return children, aux_data

    def match(self, x: ArrayLike | None = None) -> Callable:
        r"""
        Learn a score function using kernel density estimation to model a distribution.

        For the kernel density matching approach, the score function is determined by
        fitting a kernel density estimate to samples from the underlying distribution
        and then differentiating this. Therefore, learning in this context refers to
        simply defining the score function and kernel density estimate given some
        samples we wish to evaluate the score function at, and the data used to build
        the kernel density estimate.

        :param x: The :math:`n \times d` data vectors. Unused in this implementation.
        :return: A function that applies the learned score function to input ``x``
        """
        # Validate inputs
        coreax.validation.validate_is_instance(
            x=x, object_name="x", expected_type=(Array, type(None))
        )

        def score_function(x_):
            r"""
            Compute the score function using a kernel density estimation.

            The score function is determined by fitting a kernel density estimate to
            samples from the underlying distribution and then differentiating this. The
            kernel density estimate is create using a Gaussian kernel.

            :param x_: The :math:`n \times d` data vectors we wish to evaluate the score
                function at
            """
            # Check format
            original_number_of_dimensions = x_.ndim
            x_ = coreax.validation.cast_as_type(
                x=x_, object_name="x_", type_caster=jnp.atleast_2d
            )

            # Get the gram matrix row means
            gram_matrix_row_means = self.kernel.compute(x_, self.kde_data).mean(axis=1)

            # Compute gradients with respect to x
            gradients = self.kernel.grad_x(x_, self.kde_data).mean(axis=1)

            # Compute final evaluation of the score function
            score_result = gradients / gram_matrix_row_means[:, None]

            # Ensure output format accounts for 1-dimensional inputs as-well as
            # multi-dimensional ones
            if original_number_of_dimensions == 1:
                score_result = score_result[0, :]

            return score_result

        return score_function


# Define the pytree node for the added class to ensure methods with JIT decorators
# are able to run. This tuple must be updated when a new class object is defined.
score_matching_classes = (SlicedScoreMatching, KernelDensityMatching)
for current_class in score_matching_classes:
    tree_util.register_pytree_node(
        current_class, current_class.tree_flatten, current_class.tree_unflatten
    )
