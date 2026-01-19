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

"""
Classes and associated functionality to perform score matching.

The score function of some data is the derivative of the log-PDF. Score matching
aims to determine a model by matching the score function of the model to that
of the data. Exactly how the score function is modelled is specific to each
child class of the abstract base class :class:`ScoreMatching`.

An example use of score matching arises when trying to work with a
:class:`~coreax.kernels.SteinKernel`, which requires as an input a score function. If
this is known analytically, one can provide an exact score function. In other cases,
approximations to the score function are required, which can be determined using
:class:`ScoreMatching`.

When using :class:`SlicedScoreMatching`, the score function is approximated using a
neural network, whereas in :class:`KernelDensityMatching`, it is approximated by fitting
and then differentiating a kernel density estimate to the data.
"""

from abc import abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
from typing import overload

import equinox as eqx
import numpy as np
import optax
from flax.training import train_state
from jax import (
    Array,
    jvp,
    numpy as jnp,
    random,
    value_and_grad,
    vmap,
)
from jax.lax import cond, fori_loop
from jax.typing import DTypeLike
from jaxtyping import Shaped
from tqdm import tqdm
from typing_extensions import override

from coreax.kernels import ScalarValuedKernel, SquaredExponentialKernel, SteinKernel
from coreax.networks import ScoreNetwork, _LearningRateOptimiser, create_train_state
from coreax.util import KeyArrayLike

_RandomGenerator = Callable[[KeyArrayLike, Sequence[int], DTypeLike], Array]


class ScoreMatching(eqx.Module):
    """
    Abstract base class for score matching algorithms.

    The score function of some data is the derivative of the log-PDF. Score matching
    aims to determine a model by 'matching' the score function of the model to that
    of the data. Exactly how the score function is modelled is specific to each
    child class of this base class.

    This class should only contain abstract methods. Subclasses must implement all
    abstract methods to create concrete score matching algorithms.
    """

    @abstractmethod
    @overload
    def match(
        self, x: Shaped[Array, " 1 1"] | Shaped[Array, ""] | float | int
    ) -> Callable[
        [Shaped[Array, " 1 1"] | Shaped[Array, ""] | float | int],
        Shaped[Array, " 1 1"],
    ]: ...

    @abstractmethod
    @overload
    def match(  # pyright: ignore[reportOverlappingOverload]
        self, x: Shaped[Array, " n d"]
    ) -> Callable[[Shaped[Array, " n d"]], Shaped[Array, " n d"]]: ...

    @abstractmethod
    def match(
        self, x: Shaped[Array, " n d"] | Shaped[Array, ""] | float | int
    ) -> (
        Callable[[Shaped[Array, " n d"]], Shaped[Array, " n d"]]
        | Callable[
            [Shaped[Array, " 1 1"] | Shaped[Array, ""] | float | int],
            Shaped[Array, " 1 1"],
        ]
    ):
        r"""
        Match some model score function to dataset :math:`X\in\mathbb{R}^{n \times d}`.

        :param x: The :math:`n \times d` data vectors
        """


# pylint: disable=too-many-instance-attributes
class SlicedScoreMatching(ScoreMatching):
    r"""
    Implementation of slice score matching, defined in :cite:`song2020ssm`.

    The score function of some data is the derivative of the log-PDF. Score matching
    aims to determine a model by 'matching' the score function of the model to that
    of the data. Exactly how the score function is modelled is specific to each
    child class of :class:`ScoreMatching`.

    With sliced score matching, we train a neural network to directly approximate
    the score function of the data. The approach is outlined in detail in
    :cite:`song2020ssm`.

    .. note::
        The inputs `num_random_vectors` and `num_noise_models` are set to 1 if they are
        given any smaller than this.

    :param random_key: Key for random number generation
    :param random_generator: Distribution sampler (``key``, ``shape``, ``dtype``)
        :math:`\rightarrow` :class:`~jax.Array`, e.g. distributions in
        :mod:`~jax.random`
    :param noise_conditioning: Use the noise conditioning version of score matching.
        Defaults to :data:`True`.
    :param use_analytic: Use the analytic (reduced variance) objective or not.
        Defaults to :data:`False`.
    :param num_random_vectors: The number of random vectors to use per data vector.
        Defaults to 1.
    :param learning_rate: Optimiser learning rate. Defaults to 1e-3.
    :param num_epochs: Number of epochs for training. Defaults to 10.
    :param batch_size: Size of mini-batch. Defaults to 64.
    :param hidden_dims: Sequence of ScoreNetwork hidden layer sizes. Defaults to
        [128, 128, 128] denoting 3 hidden layers each composed of 128 nodes.
    :param optimiser: The optax optimiser to use. Defaults to optax.adam.
    :param num_noise_models: Number of noise models to use in noise
        conditional score matching. Defaults to 100.
    :param sigma: Initial noise standard deviation for noise geometric progression
        in noise conditional score matching. Defaults to 1.
    :param gamma: Geometric progression ratio. Defaults to 0.95.
    :param progress_bar: Boolean indicating whether or not to write a progress bar
        tracking the training of the neural network. Defaults to :data:`False`.
    """

    random_key: KeyArrayLike
    random_generator: _RandomGenerator
    noise_conditioning: bool
    use_analytic: bool
    num_random_vectors: int
    learning_rate: float
    num_epochs: int
    batch_size: int
    hidden_dims: Sequence[int]
    optimiser: _LearningRateOptimiser
    num_noise_models: int
    sigma: float
    gamma: float
    progress_bar: bool

    # TODO: refactor this to require use of keyword arguments
    # https://github.com/gchq/coreax/issues/782
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        random_key: KeyArrayLike,
        random_generator: _RandomGenerator,
        noise_conditioning: bool = True,
        use_analytic: bool = False,
        num_random_vectors: int = 1,
        learning_rate: float = 1e-3,
        num_epochs: int = 10,
        batch_size: int = 64,
        hidden_dims: Sequence[int] = (128, 128, 128),
        optimiser: _LearningRateOptimiser = optax.adamw,
        num_noise_models: int = 100,
        sigma: float = 1.0,
        gamma: float = 0.95,
        progress_bar: bool = False,
    ):
        """Define a sliced score matching class and update invalid inputs."""
        # JAX will not error if we have num_random_vectors set to 0, but this approach
        # is fundamentally about projecting along random vectors, so we cap the lower
        # value for this at 1. Similarly, there must be at-least one noise model for
        # the code to do the projections.
        num_random_vectors = max(num_random_vectors, 1)
        num_noise_models = max(num_noise_models, 1)

        # Assign all inputs
        self.random_key = random_key
        self.random_generator = random_generator
        self.noise_conditioning = noise_conditioning
        self.use_analytic = use_analytic
        self.num_random_vectors = num_random_vectors
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.optimiser = optimiser
        self.num_noise_models = num_noise_models
        self.sigma = sigma
        self.gamma = gamma
        self.progress_bar = progress_bar

    # pylint: enable=too-many-arguments

    def _objective_function(
        self,
        random_direction_vector: Shaped[Array, " d"],
        grad_score_times_random_direction_matrix: Shaped[Array, " d"],
        score_matrix: Shaped[Array, " d"],
    ) -> float:
        """
        Compute the score matching loss function.

        Two objectives are proposed in :cite:`song2020ssm`, a general objective, and a
        simplification with reduced variance that holds for particular assumptions. The
        choice between the two is determined by the boolean ``use_analytic`` defined
        when the class is initiated.

        :param random_direction_vector: :math:`d`-dimensional random vector
        :param grad_score_times_random_direction_matrix: Product of the gradient of
            score_matrix (w.r.t. ``x``) and the random_direction_vector
        :param score_matrix: Gradients of log-density
        :return: Evaluation of score matching objective, see equations 7 and 8 in
            :cite:`song2020ssm`
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
        random_direction_vector: Shaped[Array, " d"],
        grad_score_times_random_direction_matrix: Shaped[Array, " d"],
        score_matrix: Shaped[Array, " d"],
    ) -> Shaped[Array, ""]:
        """
        Compute reduced variance score matching loss function.

        This is for use with certain random measures, e.g. normal and Rademacher. If
        this assumption is not true, then
        :meth:`SlicedScoreMatching._general_objective` should be used instead.

        :param random_direction_vector: :math:`d`-dimensional random vector
        :param grad_score_times_random_direction_matrix: Product of the gradient of
            score_matrix (w.r.t. ``x``) and the random_direction_vector
        :param score_matrix: Gradients of log-density
        :return: Evaluation of score matching objective, see equation 8 in
            :cite:`song2020ssm`
        """
        result = (
            random_direction_vector @ grad_score_times_random_direction_matrix
            + 0.5 * score_matrix @ score_matrix
        )
        return result

    @staticmethod
    def _general_objective(
        random_direction_vector: Shaped[Array, " d"],
        grad_score_times_random_direction_matrix: Shaped[Array, " d"],
        score_matrix: Shaped[Array, " d"],
    ) -> Shaped[Array, ""]:
        """
        Compute general score matching loss function.

        This is to be used when one cannot assume normal or Rademacher random measures
        when using score matching, but has higher variance than
        :meth:`SlicedScoreMatching._analytic_objective` if these assumptions hold.

        :param random_direction_vector: :math:`d`-dimensional random vector
        :param grad_score_times_random_direction_matrix: Product of the gradient of
            score_matrix (w.r.t. ``x``) and the random_direction_vector
        :param score_matrix: Gradients of log-density
        :return: Evaluation of score matching objective, see equation 7 in
            :cite:`song2020ssm`
        """
        result = (
            random_direction_vector @ grad_score_times_random_direction_matrix
            + 0.5 * (random_direction_vector @ score_matrix) ** 2
        )
        return result

    def _loss_element(
        self, x: Shaped[Array, " d"], v: Shaped[Array, " d"], score_network: Callable
    ) -> float:
        """
        Compute element-wise loss function.

        Computes the loss function from Section 3.2 of Song el al.'s paper on sliced
        score matching :cite:`song2020ssm`.

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

    @eqx.filter_jit
    def _train_step(
        self,
        state: train_state.TrainState,
        x: Shaped[Array, " n d"],
        random_vectors: Shaped[Array, " n m d"],
    ) -> tuple[train_state.TrainState, float]:
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

        val, grads = value_and_grad(loss)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, val

    def _noise_conditional_loop_body(
        self,
        i: int,
        obj: float,
        state: train_state.TrainState,
        params: dict,
        x: Shaped[Array, " n d"],
        random_vectors: Shaped[Array, " n m d"],
        sigmas: Shaped[Array, " num_noise_models"],
    ) -> float:
        r"""
        Sum objective function with noise perturbations.

        Inputs are perturbed by Gaussian random noise to improve performance of score
        matching. See :cite:`song2020improved_sgm` for details.

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
        obj += (
            sigmas[i] ** 2
            * self._loss(lambda x_: state.apply_fn({"params": params}, x_))(
                x_perturbed, random_vectors
            ).mean()
        )
        return obj

    @eqx.filter_jit
    def _noise_conditional_train_step(
        self,
        state: train_state.TrainState,
        x: Shaped[Array, " n d"],
        random_vectors: Shaped[Array, " n m d"],
        sigmas: Shaped[Array, " num_noise_models"],
    ) -> tuple[train_state.TrainState, float]:
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

        val, grads = value_and_grad(loss)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, val

    @override
    def match(self, x):  # noqa: C901, PLR0912
        r"""
        Learn a sliced score matching function via :cite:`song2020ssm`.

        We currently use the :class:`~coreax.networks.ScoreNetwork` neural network to
        approximate the score function. Alternative network architectures can be
        considered.

        :param x: The :math:`n \times d` data vectors
        :return: A function that applies the learned score function to input ``x``
        """
        # Check format of input array. We use atleast_2d from JAX to perform
        # conversions here which provides the desired handling of 1 dimensional arrays,
        # whereas this handling differs if we instead used the custom function
        # _atleast_2d_consistent in coreax.data.
        x = jnp.atleast_2d(x)

        # Setup neural network that will approximate the score function
        num_points, data_dimension = x.shape
        score_network = ScoreNetwork(self.hidden_dims, data_dimension)

        # Define what a training step consists of - dependent on if we want to include
        # noise perturbations
        if self.noise_conditioning:
            gammas = self.gamma ** jnp.arange(self.num_noise_models)
            sigmas = self.sigma * gammas
            train_step = partial(self._noise_conditional_train_step, sigmas=sigmas)

        else:
            train_step = self._train_step

        # Define random projection vectors
        generator_key, state_key, batch_key = random.split(self.random_key, 3)
        try:
            random_vectors = self.random_generator(
                generator_key,
                (num_points, self.num_random_vectors, data_dimension),
                float,
            )
        except TypeError as exception:
            if isinstance(self.num_random_vectors, float):
                raise ValueError("num_random_vectors must be an integer") from exception
            raise

        # Define a training state
        state = create_train_state(
            state_key, score_network, self.learning_rate, data_dimension, self.optimiser
        )

        try:
            loop_keys = random.split(batch_key, self.num_epochs)
        except TypeError as exception:
            if self.num_epochs < 0:
                raise ValueError("num_epochs must be a positive integer") from exception
            if isinstance(self.num_epochs, float):
                raise TypeError("num_epochs must be a positive integer") from exception
            raise

        # Carry out main training loop to fit the neural network
        tqdm_progress_bar = tqdm(range(self.num_epochs), disable=not self.progress_bar)
        for i in tqdm_progress_bar:
            # Sample some data-points to pass for this step
            try:
                idx = random.randint(loop_keys[i], (self.batch_size,), 0, num_points)
            except TypeError as exception:
                if self.batch_size < 0:
                    raise ValueError(
                        "batch_size must be a positive integer"
                    ) from exception
                if isinstance(self.batch_size, float):
                    raise TypeError(
                        "batch_size must be a positive integer"
                    ) from exception
                raise
            # Apply training step
            state, val = train_step(state, x[idx, :], random_vectors[idx, :])

            # Print progress (limited to avoid excessive output)
            if i % 10 == 0 and self.progress_bar:
                tqdm_progress_bar.write(f"{i:>6}/{self.num_epochs}: loss {val:<.5f}")

        # Return the learned score function, which is a callable
        return lambda x_: state.apply_fn({"params": state.params}, x_)


# pylint: enable=too-many-instance-attributes


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
    """

    kernel: ScalarValuedKernel

    def __init__(self, length_scale: float):
        """Define the kernel density matching class."""
        # Define a normalised Gaussian kernel (which is a special cases of the squared
        # exponential kernel) to construct the kernel density estimate
        self.kernel = SquaredExponentialKernel(
            length_scale=length_scale,
            output_scale=1.0 / (np.sqrt(2 * np.pi) * length_scale),
        )
        super().__init__()

    @override
    def match(self, x):
        r"""
        Learn a score function using kernel density estimation to model a distribution.

        For the kernel density matching approach, the score function is determined by
        fitting a kernel density estimate to samples from the underlying distribution
        and then differentiating this. Therefore, learning in this context refers to
        simply defining the score function and kernel density estimate given some
        samples we wish to evaluate the score function at, and the data used to build
        the kernel density estimate.

        :param x: Set of :math:`n \times d` samples from the underlying distribution
            that are used to build the kernel density estimate
        :return: A function that applies the learned score function to input ``x``
        """
        kde_data = x

        @overload
        def score_function(
            x_: Shaped[Array, " 1 1"] | Shaped[Array, ""] | float | int,
        ) -> Shaped[Array, " 1 1"]: ...

        @overload
        def score_function(  # pyright: ignore[reportOverlappingOverload]
            x_: Shaped[Array, " n d"],
        ) -> Shaped[Array, " n d"]: ...

        def score_function(
            x_: Shaped[Array, " n d"] | Shaped[Array, ""] | float | int,
        ) -> Shaped[Array, " n d"] | Shaped[Array, " 1 1"]:
            r"""
            Compute the score function using a kernel density estimation.

            The score function is determined by fitting a kernel density estimate to
            samples from the underlying distribution and then differentiating this. The
            kernel density estimate is create using a Gaussian kernel.

            :param x_: The :math:`n \times d` data vectors we wish to evaluate the score
                function at
            """
            # Check format of input array. We use atleast_2d from JAX to perform
            # conversions here. If we instead used the custom function
            # _atleast_2d_consistent in coreax.data, we would require more
            # processing when calling the methods on the kernel and the output values
            # from these methods can differ from the expected outputs.
            original_number_of_dimensions = jnp.asarray(x_).ndim
            x_ = jnp.atleast_2d(x_)

            # Get the gram matrix row-mean
            gram_matrix_row_means = self.kernel.compute_mean(x_, kde_data, axis=1)

            # Compute gradients with respect to x
            gradients = self.kernel.grad_x(x_, kde_data).mean(axis=1)

            # Compute final evaluation of the score function
            score_result = gradients / gram_matrix_row_means[:, None]

            # Ensure output format accounts for 1-dimensional inputs as-well as
            # multi-dimensional ones
            if original_number_of_dimensions == 1:
                score_result = score_result[0, :]

            return score_result

        return score_function


def convert_stein_kernel(
    x: Shaped[Array, " n d"],
    kernel: ScalarValuedKernel,
    score_matching: ScoreMatching | None,
) -> SteinKernel:
    r"""
    Convert the kernel to a :class:`~coreax.kernels.SteinKernel`.

    :param x: The data used to call `score_matching.match(x)`
    :param kernel: :class:`~coreax.kernels.ScalarValuedKernel` instance implementing a
        kernel function
        :math:`k: \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}`; if 'kernel'
        is a :class:`~coreax.kernels.SteinKernel` and :code:`score_matching is not
        data:`None`, a new instance of the kernel will be generated where the score
        function is given by :code:`score_matching.match(x)`
    :param score_matching: Specifies/overwrite the score function of the implied/passed
       :class:`~coreax.kernels.SteinKernel`; if :data:`None`, default to
       :class:`~coreax.score_matching.KernelDensityMatching` unless 'kernel' is a
       :class:`~coreax.kernels.SteinKernel`, in which case the kernel's existing score
       function is used.
    :return: The (potentially) converted/updated :class:`~coreax.kernels.SteinKernel`.
    """
    if isinstance(kernel, SteinKernel):
        if score_matching is not None:
            _kernel = eqx.tree_at(
                lambda x: x.score_function, kernel, score_matching.match(x)
            )
        else:
            _kernel = kernel
    else:
        if score_matching is None:
            length_scale = getattr(kernel, "length_scale", 1.0)
            score_matching = KernelDensityMatching(length_scale)
        _kernel = SteinKernel(kernel, score_function=score_matching.match(x))
    return _kernel
