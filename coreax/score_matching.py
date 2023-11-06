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

import inspect
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial

import jax
import optax
from flax.training.train_state import TrainState
from jax import jit, jvp
from jax import numpy as jnp
from jax import random, tree_util, vmap
from jax.lax import cond, fori_loop
from jax.typing import ArrayLike
from tqdm import tqdm

from coreax.networks import ScoreNetwork, create_train_state


class ScoreMatching(ABC):
    """
    Base class for score matching algorithms.
    """

    def __init__(self):
        r"""
        Define a score matching algorithm.

        The score function of some data is the derivative of the log-PDF. Score matching
        aims to determine a model by 'matching' the score function of the model to that
        of the data. Exactly how the score function is modelled is specific to each
        child class of this base class.
        """

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """
        Reconstruct a pytree from the tree definition and the leaves.

        Arrays & dynamic values (children) and auxiliary data (static values) are
        reconstructed. A method to reconstruct the pytree needs to be specified to
        enable jit decoration of methods inside this class.
        """
        return cls(*children, **aux_data)

    @abstractmethod
    def match(self, x):
        r"""
        Match some model score function to that of a dataset ``x``.

        :param x: The :math:`n \times d` data vectors
        """


class SlicedScoreMatching(ScoreMatching):
    """
    Implementation of slice score matching, defined in [ssm]_.
    """

    def __init__(
        self,
        random_generator: Callable,
        random_key: random.PRNGKeyArray = random.PRNGKey(0),
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
        r"""
        Define a sliced score matching class.

        The score function of some data is the derivative of the log-PDF. Score matching
        aims to determine a model by 'matching' the score function of the model to that
        of the data. Exactly how the score function is modelled is specific to each
        child class of this base class.

        With sliced score matching, we train a neural network to directly approximate
        the score function of the data. The approach is outlined in detail in [ssm]_.

        TODO: Allow user to pass hidden_dim as a list and build network with
            # layers = len(hidden_dim), with each layer size assigned as appropriate.

        :param random_generator: Distribution sampler (``key``, ``shape``, ``dtype``)
            :math:`\rightarrow` :class:`~jax.Array`, e.g. distributions in
            :class:`~jax.random`
        :param random_key: Key for random number generation
        :param noise_conditioning: Use the noise conditioning version of score matching.
            Defaults to :data:`True`.
        :param use_analytic: Use the analytic (reduced variance) objective or not.
            Defaults to :data:`False`.
        :param num_random_vectors: The number of random vectors to use per data vector.
            Defaults to 1.
        :param learning_rate: Optimiser learning rate. Defaults to 1e-3.
        :param num_epochs: Number of epochs for training. Defaults to 10.
        :param batch_size: Size of minibatch. Defaults to 64.
        :param hidden_dim: The ScoreNetwork hidden dimension. Defaults to 128.
        :param optimiser: The optax optimiser to use. Defaults to optax.adam.
        :param num_noise_models: Number of noise models to use in noise
            conditional score matching. Defaults to 100.
        :param sigma: Initial noise standard deviation for noise geometric progression
            in noise conditional score matching. Defaults to 1.
        :param gamma: Geometric progression ratio. Defaults to 0.95.
        """
        # Assign all inputs
        self.random_generator = random_generator
        self.random_key = random_key
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

    def _tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable jit decoration
        of methods inside this class.
        """
        children = ()
        aux_data = {
            "random_generator": self.random_generator,
            "random_key": self.random_key,
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

        Two objectives are proposed in [ssm]_, a general objective, and a simplification
        with reduced variance that holds for particular assumptions. The choice between
        the two is determined by the boolean ``use_analytic`` defined when the class is
        initiated.

        :param random_direction_vector: :math:`d`-dimensional random vector
        :param grad_score_times_random_direction_matrix: Product of the gradient of
            score_matrix (w.r.t. ``x``) and the random_direction_vector
        :param score_matrix: Gradients of log-density
        :return: Evaluation of score matching objective, see equation 8 in [ssm]_
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
        this assumption is not true, then :meth:`general_obj` should be used instead.

        :param random_direction_vector: :math:`d`-dimensional random vector
        :param grad_score_times_random_direction_matrix: Product of the gradient of
            score_matrix (w.r.t. ``x``) and the random_direction_vector
        :param score_matrix: Gradients of log-density
        :return: Evaluation of score matching objective, see equation 8 in [ssm]_
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
        when using score matching, but has higher variance than :meth:`analytic_obj` if
        these assumptions hold.

        :param random_direction_vector: `:math:`d`-dimensional random vector
        :param grad_score_times_random_direction_matrix: Product of the gradient of
            score_matrix (w.r.t. ``x``) and the random_direction_vector
        :param score_matrix: Gradients of log-density
        :return: Evaluation of score matching objective, see equation 7 in [ssm]_
        """
        result = (
            random_direction_vector @ grad_score_times_random_direction_matrix
            + 0.5 * (random_direction_vector @ score_matrix) ** 2
        )
        return result

    def _loss_element(
        self, x: ArrayLike, v: ArrayLike, score_network: Callable
    ) -> float:
        r"""
        Compute element-wise loss function.

        Computes the loss function from Section 3.2 of Song el al.'s paper on sliced
        score matching [ssm]_.

        :param x: :math:`d`-dimensional data vector
        :param v: :math:`d`-dimensional random vector
        :param score_network: Function that calls the neural network on ``x``
        :return: Objective function output for single ``x`` and ``v`` inputs
        """
        s, u = jvp(score_network, (x,), (v,))
        return self._objective_function(v, u, s)

    def _loss(self, score_network: Callable) -> Callable:
        r"""
        Compute vector mapped loss function for arbitrary many ``X`` and ``V`` vectors.

        In the context of score matching, we expect to call the objective function on
        the data vector (``x``), random vectors (``v``) and using the score neural
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

        :param state: The :class:`~flax.training.train_state.TrainState` object.
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
        matching. See [improvedsgm]_ for details.

        :param i: Loop index
        :param obj: Running objective, i.e. the current partial sum
        :param state: The :class:`~flax.training.train_state.TrainState` object
        :param params: The current iterate parameter settings
        :param x: The :math:`n \times d` data vectors
        :param random_vectors: The :math:`n \times m \times d` random vectors
        :param sigmas: The geometric progression of noise standard deviations
        :return: The updated objective, i.e. partial sum
        """
        # TODO: This will generate the same set of random numbers on each function call.
        #  We might want to replace this with random.PRNGKey(i) to get a unique set each
        #  time.
        # Perturb the inputs with Gaussian noise
        x_perturbed = x + sigmas[i] * random.normal(random.PRNGKey(0), x.shape)
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

    def match(self, x: ArrayLike) -> Callable:
        r"""
        Learn a sliced score matching function from Song et al.'s paper [ssm]_.

        We currently use the :class:`coreax.networks.ScoreNetwork` neural network to
        approximate the score function. Alternative network architectures can be
        considered.

        :param x: The :math:`n \times d` data vectors
        :return: A function that applies the learned score function to input ``x``
        """
        # Setup neural network that will approximate the score function
        num_points, data_dimension = x.shape
        score_network = ScoreNetwork(self.hidden_dim, data_dimension)

        # Define what a training step consists of - dependent on if we want to include
        # noise perturbations
        if self.noise_conditioning:
            gammas = self.gamma ** jnp.arange(self.num_noise_models)
            sigmas = self.sigma * gammas
            train_step = partial(self._noise_conditional_train_step, sigmas=sigmas)

        else:
            train_step = self._train_step

        # Define random projection vectors
        random_key_1, random_key_2 = random.split(self.random_key)
        random_vectors = self.random_generator(
            random_key_1,
            (num_points, self.num_random_vectors, data_dimension),
            dtype=float,
        )

        # Define a training state
        state = create_train_state(
            score_network,
            random_key_2,
            self.learning_rate,
            data_dimension,
            self.optimiser,
        )
        _, random_key_4 = random.split(random_key_2)
        batch_key = random.PRNGKey(random_key_4[-1])

        # Carry out main training loop to fit the neural network
        for i in tqdm(range(self.num_epochs)):
            # TODO: In the existing code, idx gives the same output each time. We might
            #  want to change this to split the random key and use the result from the
            #  split each time.
            # Sample some data-points to pass for this step
            idx = random.randint(batch_key, (self.batch_size,), 0, num_points)

            # Apply training step
            state, val = train_step(state, x[idx, :], random_vectors[idx, :])

            # Print progress (limited to avoid excessive output)
            if i % 10 == 0:
                tqdm.write(f"{i:>6}/{self.num_epochs}: loss {val:<.5f}")

        # Return the learned score function, which is a callable
        return lambda x_: state.apply_fn({"params": state.params}, x_)


# Define the pytree node for the added class to ensure methods with jit decorators
# are able to run. This tuple must be updated when a new class object is defined.
kernel_classes = (SlicedScoreMatching,)
for current_class in kernel_classes:
    tree_util.register_pytree_node(
        current_class, current_class._tree_flatten, current_class._tree_unflatten
    )
