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
from typing import overload

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array, DTypeLike, Float, Shaped
from tqdm import tqdm
from typing_extensions import override

from coreax.kernels import (
    ScalarValuedKernel,
    SquaredExponentialKernel,
    SteinKernel,
)
from coreax.util import KeyArrayLike

_RandomGenerator = Callable[[KeyArrayLike, Sequence[int], DTypeLike], Array]
_LearningRateOptimiser = Callable[[float], optax.GradientTransformation]


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
    Implementation of sliced score matching, defined in :cite:`song2020ssm`.

    The score function of some data is the derivative of the log-PDF. Score matching
    aims to determine a model by 'matching' the score function of the model to that
    of the data. Exactly how the score function is modelled is specific to each
    child class of :class:`ScoreMatching`.

    With sliced score matching, we train a neural network to directly approximate
    the score function of the data. The approach is outlined in detail in
    :cite:`song2020ssm`.

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
    :param hidden_dims: Sequence of score network hidden layer sizes. Defaults to
        [128, 128, 128] denoting 3 hidden layers each composed of 128 nodes.
    :param optimiser: An instance of an :class:`optax.GradientTransformation`. Defaults
        to 'optax.adamw(1e-3)'.
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
    optimiser: optax.GradientTransformation
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
        learning_rate: float | None = None,
        num_epochs: int = 10,
        batch_size: int = 64,
        hidden_dims: Sequence[int] = (128, 128, 128),
        optimiser: (
            _LearningRateOptimiser | optax.GradientTransformation | None
        ) = None,
        num_noise_models: int = 100,
        sigma: float = 1.0,
        gamma: float = 0.95,
        progress_bar: bool = False,
    ):
        """Define a sliced score matching class and update invalid inputs."""
        if learning_rate is not None:
            raise DeprecationWarning(
                "'learning_rate' is deprecated and will be removed in v1.2.0."
            )
        optimiser = optimiser or optax.adamw(1e-3)
        if not isinstance(optimiser, optax.GradientTransformation):
            raise DeprecationWarning(
                "Passing an 'optimiser' as a '_LearningRateOptimiser' is deprecated and"
                " will be removed in v1.2.0. Pass an optax.GradientTransformation"
                " instead. E.G. `optimiser = optax.adamw(learning_rate=1e-3)."
            )
        self.random_key = random_key
        self.random_generator = random_generator
        self.noise_conditioning = noise_conditioning
        self.use_analytic = use_analytic
        self.num_random_vectors = num_random_vectors
        self.learning_rate = learning_rate or 1e-3
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.optimiser = optimiser
        self.num_noise_models = num_noise_models
        self.sigma = sigma
        self.gamma = gamma
        self.progress_bar = progress_bar

    # pylint: enable=too-many-arguments

    def __check_init__(self):
        """Check attributes are positive integers."""
        non_negative_integer_attrs = ("num_epochs", "batch_size")
        for attr in non_negative_integer_attrs:
            val = getattr(self, attr)
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"'{attr}' must be a non-negative integer")
        positive_integer_attrs = ("num_random_vectors", "num_noise_models")
        for attr in positive_integer_attrs:
            val = getattr(self, attr)
            if not isinstance(val, int) or val < 1:
                raise ValueError(f"'{attr}' must be a positive integer")

    def _loss(self, score_network: Callable) -> Callable:
        """
        Compute vector mapped loss function for arbitrary many ``X`` and ``V`` vectors.

        In the context of score matching, we expect to call the objective function on
        the data vector ``x``, random vectors ``v`` and using the score neural
        network.

        :param score_network: Function that calls the neural network on ``x``
        :return: Callable vectorised sliced score matching loss function
        """

        def _loss_element(x: Shaped[Array, " d"], v: Shaped[Array, " d"]):
            """
            Compute element-wise loss function.

            Computes the loss function from Section 3.2 of Song el al.'s paper on sliced
            score matching :cite:`song2020ssm`.

            :param x: :math:`d`-dimensional data vector
            :param v: :math:`d`-dimensional random vector
            :return: Objective function output for single ``x`` and ``v`` inputs
            """
            s, u = jax.jvp(score_network, (x,), (v,))
            if self.use_analytic:
                return v @ u + 0.5 * s @ s  # Equation 8 of song2020ssm
            return v @ u + 0.5 * (v @ s) ** 2  # Equation 7 of song2020ssm

        return jax.vmap(jax.vmap(_loss_element, in_axes=(None, 0)), in_axes=(0, 0))

    @eqx.filter_jit
    def _train_step(
        self,
        score_network: eqx.nn.Sequential,
        optimiser: optax.GradientTransformation,
        opt_state: optax.OptState,
        x: Shaped[Array, " n d"],
        random_vectors: Shaped[Array, " n m d"],
    ) -> tuple[eqx.nn.Sequential, optax.OptState, Float[Array, ""]]:
        r"""
        Apply a single training step on the score network weights via gradient descent.

        :param score_network: The current model.
        :param optimiser: The optimiser.
        :param opt_state: The current state of the optimiser.
        :param x: The :math:`n \times d` data vectors.
        :param random_vectors: The :math:`n \times m \times d` random vectors.
        :return: The updated model, optimiser state, and loss value
        """

        def standard_loss(score_network: eqx.nn.Sequential, _x: Float[Array, "n d"]):
            """Evaluate loss and compute mean."""
            return self._loss(score_network)(_x, random_vectors).mean()

        def loss(score_network: eqx.nn.Sequential, _x: Float[Array, "n d"]):
            """Handle case where we condition the loss."""
            if self.noise_conditioning:

                def noise_conditioned_loss(i, loss):
                    sigma = self.sigma * self.gamma**i
                    x_perturbed = x + sigma * jr.normal(jr.key(i), x.shape)
                    return loss + sigma**2 * standard_loss(score_network, x_perturbed)

                return jax.lax.fori_loop(
                    0, self.num_noise_models, noise_conditioned_loss, 0.0
                )
            return standard_loss(score_network, _x)

        val, grads = eqx.filter_value_and_grad(loss)(score_network, x)
        updates, opt_state = optimiser.update(
            grads, opt_state, eqx.filter(score_network, eqx.is_array)
        )
        score_network = eqx.apply_updates(score_network, updates)
        return score_network, opt_state, val

    @override
    def match(self, x):
        r"""
        Learn a sliced score matching function via :cite:`song2020ssm`.

        We currently use a neural network built with :class:`equinox.nn.Sequential`
        to approximate the score function. Alternative network architectures can be
        considered.

        :param x: The :math:`n \times d` data vectors
        :return: A function that applies the learned score function to input ``x``
        """
        # Ensure n x d is 1 x n if d = 1
        x = jnp.atleast_2d(x)
        generator_key, net_key, batch_key = jr.split(self.random_key, 3)

        # Setup neural network that will approximate the score function
        dims = [x.shape[1], *self.hidden_dims, x.shape[1]]
        keys = jr.split(net_key, len(dims) - 1)
        layers = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            layers.append(eqx.nn.Linear(d_in, d_out, key=keys[i]))
            if i < len(dims) - 2:
                layers.append(eqx.nn.Lambda(jax.nn.softplus))
        score_network = eqx.nn.Sequential(layers)

        # Define random projection vectors
        random_vectors = self.random_generator(
            generator_key,
            (x.shape[0], self.num_random_vectors, x.shape[1]),
            float,
        )

        # Define the optimiser state
        opt_state = self.optimiser.init(eqx.filter(score_network, eqx.is_array))
        loop_keys = jr.split(batch_key, self.num_epochs)

        # Carry out main training loop to fit the neural network
        tqdm_progress_bar = tqdm(range(self.num_epochs), disable=not self.progress_bar)
        for i in tqdm_progress_bar:
            # Sample some data-points to pass for this step
            idx = jr.randint(loop_keys[i], (self.batch_size,), 0, x.shape[0])
            score_network, opt_state, val = self._train_step(
                score_network,
                self.optimiser,
                opt_state,
                x[idx, :],
                random_vectors[idx, :],
            )

            # Print progress (limited to avoid excessive output)
            if i % 10 == 0 and self.progress_bar:
                tqdm_progress_bar.write(f"{i:>6}/{self.num_epochs}: loss {val:<.5f}")

        # Return the learned score function, which is a callable
        batched_network = jax.vmap(score_network)

        def score_function(
            x_: Shaped[Array, " n d"] | Shaped[Array, ""] | float | int,
        ) -> Shaped[Array, " n d"] | Shaped[Array, " 1 1"]:
            original_ndim = jnp.asarray(x_).ndim
            result = batched_network(jnp.atleast_2d(x_))
            if original_ndim < 2:  # noqa: PLR2004
                return result[0]
            return result

        return score_function


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
