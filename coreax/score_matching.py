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
from functools import partial
from typing import Callable, Tuple

import jax
import optax
from flax.training.train_state import TrainState
from jax import jit, jvp
from jax import numpy as jnp
from jax import random, tree_util, vmap
from jax.lax import fori_loop
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

    @abstractmethod
    def match(self, x):
        """
        Match some model score function to that of a dataset x.

        :param x: :math:`d`-dimensional data vector
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

        :param random_generator: Distribution sampler (key, shape, dtype)
            :math:`\rightarrow` :class:`~jax.Array`, e.g. distributions in
            :class:`~jax.random`
        :param random_key: Key for random number generation
        :param noise_conditioning: Use the noise conditioning version of score matching.
            Defaults to True.
        :param use_analytic: Use the analytic (reduced variance) objective or not.
            Defaults to False.
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
        # TODO: Allow user to pass hidden_dim as a list and build network
        #  with # layers = len(hidden_dim), with each layer size assigned as
        #  appropriate.
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

        # Objective function to use is determined given the set of inputs, so assign
        # it. Type check here to avoid failures on JAX jit compiles.
        if isinstance(use_analytic, bool):
            if use_analytic:
                self.objective_function = self.analytic_objective
            else:
                self.objective_function = self.general_objective

        # Initialise parent
        super().__init__()

    def _tree_flatten(self):
        """
        Flatten a pytree.

        Define arrays & dynamic values (children) and auxiliary data (static values).
        A method to flatten the pytree needs to be specified to enable jit decoration
        of methods inside this class.
        """
        children = (
            self.random_key,
            self.noise_conditioning,
            self.use_analytic,
            self.num_random_vectors,
            self.learning_rate,
            self.num_epochs,
            self.batch_size,
            self.hidden_dim,
            self.num_noise_models,
            self.sigma,
            self.gamma,
        )
        aux_data = {}
        return children, aux_data

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """
        Reconstructs a pytree from the tree definition and the leaves.

        Arrays & dynamic values (children) and auxiliary data (static values) are
        reconstructed. A method to reconstruct the pytree needs to be specified to
        enable jit decoration of methods inside this class.
        """
        return cls(*children, **aux_data)

    def match(self, x: ArrayLike) -> Callable:
        r"""
        Learn a sliced score matching function from Song et al.'s paper [ssm]_.

        We currently use the ScoreNetwork neural network in coreax.networks to
        approximate the score function. Alternative network architectures can be
        considered.

        :param x: The :math:`n \times d` data vectors
        :return: A function that applies the learned score function to input x
        """
        # Define the neural network
        num_points, dimension = x.shape
        score_network = ScoreNetwork(self.hidden_dim, dimension)

        # Define what a training step is for the network
        if self.noise_conditioning:
            gammas = self.gamma ** jnp.arange(self.num_noise_models)
            sigmas = self.sigma * gammas
            train_step = partial(
                self.noise_conditional_train_step,
                sigmas=sigmas,
                objective_function=self.objective_function,
                num_noise_models=self.num_noise_models,
            )
        else:
            train_step = partial(
                self.train_step, objective_function=self.objective_function
            )

        # Define random projection vectors
        random_key_1, random_key_2 = random.split(self.random_key)
        random_vectors = self.random_generator(
            random_key_1, (num_points, self.num_random_vectors, dimension), dtype=float
        )

        # Setup training of hte neural network
        state = create_train_state(
            score_network, random_key_2, self.learning_rate, dimension, self.optimiser
        )
        random_key_3, random_key_4 = random.split(random_key_2)
        batch_key = random.PRNGKey(random_key_4[-1])

        # Perform the main training loop
        for i in tqdm(range(self.num_epochs)):
            idx = random.randint(batch_key, (self.batch_size,), 0, num_points)
            state, val = train_step(state, x[idx, :], random_vectors[idx, :])
            if i % 10 == 0:
                tqdm.write(f"{i:>6}/{self.num_epochs}: loss {val:<.5f}")
        return lambda y: state.apply_fn({"params": state.params}, y)

    @jit
    def analytic_objective(
        self,
        random_direction_vector: ArrayLike,
        grad_score_times_random_direction_matrix: ArrayLike,
        score_matrix: ArrayLike,
    ) -> ArrayLike:
        """
        Compute reduced variance score matching loss function.

        This is for use with certain random measures, e.g. normal and Rademacher. If
        this assumption is not true, then general_obj should be used instead.

        :param random_direction_vector: d-dimensional random vector
        :param grad_score_times_random_direction_matrix: Product of the gradient of
            score_matrix (w.r.t. x) and the random_direction_vector
        :param score_matrix: Gradients of log-density
        :return: Evaluation of score matching objective, see equation 8 in [ssm]_
        """
        result = (
            random_direction_vector @ grad_score_times_random_direction_matrix
            + 0.5 * score_matrix @ score_matrix
        )
        return result

    @jit
    def general_objective(
        self,
        random_direction_vector: ArrayLike,
        grad_score_times_random_direction_matrix: ArrayLike,
        score_matrix: ArrayLike,
    ) -> ArrayLike:
        """
        Compute general score matching loss function.

        This is to be used when one cannot assume normal or Rademacher random measures
        when using score matching, but has higher variance than analytic_obj if these
        assumptions hold.

        :param random_direction_vector: d-dimensional random vector
        :param grad_score_times_random_direction_matrix: Product of the gradient of
            score_matrix (w.r.t. x) and the random_direction_vector
        :param score_matrix: Gradients of log-density
        :return: Evaluation of score matching objective, see equation 7 in [ssm]_
        """
        result = (
            random_direction_vector @ grad_score_times_random_direction_matrix
            + 0.5 * (random_direction_vector @ score_matrix) ** 2
        )
        return result

    @partial(jit, static_argnames=["score_network", "objective_function"])
    def loss_element(
        self,
        x: ArrayLike,
        random_direction_vector: ArrayLike,
        score_network: Callable,
        objective_function: Callable,
    ) -> float:
        r"""
        Compute element-wise loss function.

        Computes the loss function from Section 3.2 of Song el al.'s paper on sliced
        score matching [ssm]_.

        :param x: :math:`d`-dimensional data vector
        :param random_direction_vector: :math:`d`-dimensional random vector
        :param score_network: Function that calls the neural network on x
        :param objective_function: Function that computes objective value
        :return: Objective function output for single x and v inputs
        """
        s, u = jvp(score_network, (x,), (random_direction_vector,))
        return objective_function(random_direction_vector, u, s)

    def loss(self, score_network: Callable, objective_function: Callable) -> Callable:
        r"""
        Compute vector mapped loss function for arbitrary numbers of X and V vectors.

        In the context of score matching, we expect to call the objective function on
        the data vector (x), random vectors (v) and using the score neural network.

        :param score_network: Function that calls the neural network on x
        :param objective_function: Element-wise function (vector, vector, score_network)
            :math:`\rightarrow \mathbb{R}`
        :return: Callable vectorised sliced score matching loss function
        """
        inner = vmap(
            lambda x, v: self.loss_element(x, v, score_network, objective_function),
            (None, 0),
            0,
        )
        return vmap(inner, (0, 0), 0)

    @partial(jit, static_argnames=["objective_function"])
    def train_step(
        self,
        state: TrainState,
        x: ArrayLike,
        random_vectors: ArrayLike,
        objective_function: Callable,
    ) -> Tuple[TrainState, float]:
        r"""
        Apply a single training step that updates model parameters using loss gradient.

        :param state: The :class:`~flax.training.train_state.TrainState` object.
        :param x: The :math:`n \times d` data vectors
        :param random_vectors: The :math:`n \times m \times d` random vectors
        :param objective_function: Objective function (vector, vector, vector)
            :math:`\rightarrow \mathbb{R}`
        :return: The updated :class:`~flax.training.train_state.TrainState` object
        """

        def loss(params):
            return self.loss(
                lambda y: state.apply_fn({"params": params}, y), objective_function
            )(x, random_vectors).mean()

        val, grads = jax.value_and_grad(loss)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, val

    @partial(jit, static_argnames=["objective_function"])
    def noise_conditional_loop_body(
        self,
        i: int,
        objective_value: float,
        state: TrainState,
        params: dict,
        x: ArrayLike,
        random_vectors: ArrayLike,
        sigmas: ArrayLike,
        objective_function: Callable,
    ) -> float:
        r"""
        Sum objective function with noise perturbations.

        Inputs are perturbed by Gaussian random noise to improve performance of score
        matching. See [improvedsgm]_ for details.

        :param i: Loop index
        :param objective_value: Running objective, i.e. the current partial sum
        :param state: The :class:`~flax.training.train_state.TrainState` object
        :param params: The current iterate parameter settings
        :param x: The :math:`n \times d` data vectors
        :param random_vectors: The :math:`n \times m \times d` random vectors
        :param sigmas: The geometric progression of noise standard deviations
        :param objective_function: Element objective function (vector, vector, vector)
            :math:`\rightarrow real`
        :return: The updated objective, i.e. partial sum
        """
        # Perturb inputs with Gaussian noise
        x_ = x + sigmas[i] * random.normal(random.PRNGKey(0), x.shape)
        objective_value = (
            objective_value
            + sigmas[i] ** 2
            * self.loss(
                lambda y: state.apply_fn({"params": params}, y), objective_function
            )(x_, random_vectors).mean()
        )
        return objective_value

    @partial(jit, static_argnames=["objective_function", "num_noise_models"])
    def noise_conditional_train_step(
        self,
        state: TrainState,
        x: ArrayLike,
        random_vectors: ArrayLike,
        sigmas: ArrayLike,
        objective_function: Callable,
        num_noise_models: int,
    ) -> Tuple[TrainState, float]:
        r"""
        Apply a single training step that updates model parameters using loss gradient.

        :param state: The :class:`~flax.training.train_state.TrainState` object
        :param x: The :math:`n \times d` data vectors
        :param random_vectors: The :math:`n \times m \times d` random vectors
        :param sigmas: Length num_noise_models array of noise standard deviations to use
            in objective function
        :param objective_function: Objective function (vector, vector, vector)
            :math:`\rightarrow real`
        :param num_noise_models: The static number of terms in the geometric
            progression. (Required for reverse mode autodiff)
        :return: The updated :class:`~flax.training.train_state.TrainState` object
        """

        def loss(params):
            body = partial(
                self.noise_conditional_loop_body,
                state=state,
                params=params,
                x=x,
                random_vectors=random_vectors,
                sigmas=sigmas,
                objective_function=objective_function,
            )
            return fori_loop(0, num_noise_models, body, 0.0)

        val, grads = jax.value_and_grad(loss)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, val


# Define the pytree node for the added class to ensure methods with jit decorators
# are able to run. We rely on the naming convention that all child classes of
# ScoreMatching include the sub-string ScoreMatching inside of them.
for name, current_class in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    if "ScoreMatching" in name and name != "ScoreMatching":
        tree_util.register_pytree_node(
            current_class, current_class._tree_flatten, current_class._tree_unflatten
        )
