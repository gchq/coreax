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

import jax
from jax import random, vmap, numpy as jnp, jvp, jit
from jax.lax import fori_loop
from jax.typing import ArrayLike
from typing import Callable, Optional, Tuple
from flax.training.train_state import TrainState
from functools import partial
from coreax.networks import ScoreNetwork, create_train_state
import optax
from tqdm import tqdm


@jit
def analytic_obj(
        random_direction_vector: ArrayLike,
        grad_score_times_random_direction_matrix: ArrayLike,
        score_matrix: ArrayLike
) -> ArrayLike:
    """
    Compute reduced variance score matching loss function.

    This is for use with certain random measures, e.g. normal and Rademacher. If this
    assumption is not true, then general_obj should be used instead.

    :param random_direction_vector: d-dimensional random vector
    :param grad_score_times_random_direction_matrix: Product of the gradient of
        score_matrix (w.r.t. x) and the random_direction_vector
    :param score_matrix: Gradients of log-density
    :return: Evaluation of score matching objective, see equation 8 in [ssm]_
    """
    result = (random_direction_vector @ grad_score_times_random_direction_matrix +
              0.5 * score_matrix @ score_matrix)
    return result


@jit
def general_obj(
    random_direction_vector, grad_score_times_random_direction_matrix, score_matrix
) -> ArrayLike:
    """
    Compute general score matching loss function.

    This is to be used when one cannot assume normal or Rademacher random measures when
    using score matching, but has higher variance than analytic_obj if these
    assumptions hold.

    :param random_direction_vector: d-dimensional random vector
    :param grad_score_times_random_direction_matrix: Product of the gradient of
        score_matrix (w.r.t. x) and the random_direction_vector
    :param score_matrix: Gradients of log-density
    :return: Evaluation of score matching objective, see equation 7 in [ssm]_
    """
    result = (random_direction_vector @ grad_score_times_random_direction_matrix + 0.5 *
              (random_direction_vector @ score_matrix) ** 2)
    return result


@partial(jit, static_argnames=["score_network", "obj_fn"])
def sliced_score_matching_loss_element(
    x: ArrayLike, v: ArrayLike, score_network: Callable, obj_fn: Callable
) -> float:
    r"""
    Compute element-wise loss function.

    Computes the loss function from Section 3.2 of Song el al.'s paper on sliced score
    matching [ssm]_.

    :param x: :math:`d`-dimensional data vector
    :param v: :math:`d`-dimensional random vector
    :param score_network: Function that calls the neural network on x
    :param obj_fn: Objective function with arguments :math:`(v, u, s)
                    \rightarrow \mathbb{R}`
    :return: Objective function output for single x and v inputs
    """
    s, u = jvp(score_network, (x,), (v,))
    return obj_fn(v, u, s)


def sliced_score_matching_loss(score_network: Callable, obj_fn: Callable) -> Callable:
    r"""
    Compute vector mapped loss function for arbitrary numbers of X and V vectors.

    In the context of score matching, we expect to call the objective function on the
    data vector (x), random vectors (v) and using the score neural network.

    :param score_network: Function that calls the neural network on x
    :param obj_fn: Element-wise function (vector, vector, score_network)
                    :math:`\rightarrow \mathbb{R}`
    :return: Callable vectorised sliced score matching loss function
    """
    inner = vmap(
        lambda x, v: sliced_score_matching_loss_element(x, v, score_network, obj_fn),
        (None, 0),
        0,
    )
    return vmap(inner, (0, 0), 0)


@partial(jit, static_argnames=["obj_fn"])
def sliced_score_matching_train_step(
    state: TrainState, X: ArrayLike, V: ArrayLike, obj_fn: Callable
) -> Tuple[TrainState, float]:
    r"""
    Apply a single training step that updates model parameters using loss gradient.

    :param state: The :class:`~flax.training.train_state.TrainState` object.
    :param X: the :math:`n \times d` data vectors
    :param V: the :math:`n \times m \times d` random vectors
    :param obj_fn: Objective function (vector, vector, vector) :math:`\rightarrow
        \mathbb{R}`
    :return: The updated :class:`~flax.training.train_state.TrainState` object
    """

    def loss(params):
        return sliced_score_matching_loss(
            lambda x: state.apply_fn({"params": params}, x), obj_fn
        )(X, V).mean()

    val, grads = jax.value_and_grad(loss)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, val


@partial(jit, static_argnames=["obj_fn"])
def noise_conditional_loop_body(
    i: int,
    obj: float,
    state: TrainState,
    params: dict,
    X: ArrayLike,
    V: ArrayLike,
    sigmas: ArrayLike,
    obj_fn: Callable,
) -> float:
    r"""
    Sum objective function with noise perturbations.

    Inputs are perturbed by Gaussian random noise to improve performance of score
    matching. See [improvedsgm]_ for details.

    :param i: Loop index
    :param obj: Running objective, i.e. the current partial sum
    :param state: The :class:`~flax.training.train_state.TrainState` object
    :param params: The current iterate parameter settings
    :param X: the :math:`n \times d` data vectors
    :param V: the :math:`n \times m \times d` random vectors
    :param sigmas: The geometric progression of noise standard deviations
    :param obj_fn: Element objective function (vector, vector, vector)
                    :math:`\rightarrow real`
    :return: The updated objective, i.e. partial sum
    """
    # perturb X
    X_ = X + sigmas[i] * random.normal(random.PRNGKey(0), X.shape)
    obj = (
        obj
        + sigmas[i] ** 2
        * sliced_score_matching_loss(
            lambda x: state.apply_fn({"params": params}, x), obj_fn
        )(X_, V).mean()
    )
    return obj


@partial(jit, static_argnames=["obj_fn", "L"])
def noise_conditional_train_step(
    state: TrainState,
    X: ArrayLike,
    V: ArrayLike,
    obj_fn: Callable,
    sigmas: ArrayLike,
    L: int,
) -> Tuple[TrainState, float]:
    r"""
    Apply a single training step that updates model parameters using loss gradient.

    :param state: The :class:`~flax.training.train_state.TrainState` object
    :param X: the :math:`n \times d` data vectors
    :param V: the :math:`n \times m \times d` random vectors
    :param obj_fn: Objective function (vector, vector, vector) :math:`\rightarrow real`
    :param sigmas: Length L array of noise standard deviations to use in objective
        function
    :param L: The static number of terms in the geometric progression. (Required for
        reverse mode autodiff)
    :return: The updated :class:`~flax.training.train_state.TrainState` object
    """

    def loss(params):
        body = partial(
            noise_conditional_loop_body,
            state=state,
            params=params,
            X=X,
            V=V,
            sigmas=sigmas,
            obj_fn=obj_fn,
        )
        return fori_loop(0, L, body, 0.0)

    val, grads = jax.value_and_grad(loss)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, val


r"""
    Learn a sliced score matching function from Song et al.'s paper [ssm]_.

    We currently use the ScoreNetwork neural network in coreax.networks to approximate
    the score function. Alternative network architectures can be considered.

    :param X: The :math:`n \times d` data vectors
    :param rgenerator: Distribution sampler (key, shape, dtype) :math:`\rightarrow`
        :class:`~jax._src.typing.Array`, e.g. distributions in :class:`~jax.random`
    :param noise_conditioning: Use the noise conditioning version of score matching.
        Defaults to True.
    :param use_analytic: Use the analytic (reduced variance) objective or not. Defaults
        to False.
    :param M: The number of random vectors to use per data vector. Defaults to 1.
    :param lr: Optimiser learning rate. Defaults to 1e-3.
    :param epochs: Epochs for training. Defaults to 10.
    :param batch_size: Size of minibatch. Defaults to 64.
    :param hidden_dim: The ScoreNetwork hidden dimension. Defaults to 128.
    :param optimiser: The optax optimiser to use. Defaults to optax.adam.
    :param L: Number of noise models to use in noise conditional score matching.
        Defaults to 100.
    :param sigma: Initial noise standard deviation for noise geometric progression in
        noise conditional score matching. Defaults to 1.
    :param gamma: Geometric progression ratio. Defaults to 0.95.
    :return: A function that applies the learned score function to input x
    """
def sliced_score_matching(
    X: ArrayLike,
    rgenerator: Callable,
    noise_conditioning: bool = True,
    use_analytic: bool = False,
    M: int = 1,
    lr: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 64,
    hidden_dim: int = 128,
    optimiser: Callable = optax.adamw,
    L: int = 100,
    sigma: float = 1.0,
    gamma: float = 0.95,
) -> Callable:
    # main objects
    n, d = X.shape
    sn = ScoreNetwork(hidden_dim, d)
    obj_fn = analytic_obj if use_analytic else general_obj
    if noise_conditioning:
        gammas = gamma ** jnp.arange(L)
        sigmas = sigma * gammas
        train_step = partial(
            noise_conditional_train_step, obj_fn=obj_fn, sigmas=sigmas, L=L
        )
    else:
        train_step = partial(sliced_score_matching_train_step, obj_fn=obj_fn)

    # random vector setup
    k1, k2 = random.split(random.PRNGKey(0))
    V = rgenerator(k1, (n, M, d), dtype=float)

    # training setup
    state = create_train_state(sn, k2, lr, d, optimiser)
    batch_key = random.PRNGKey(1)

    # main training loop
    for i in tqdm(range(epochs)):
        idx = random.randint(batch_key, (batch_size,), 0, n)
        state, val = train_step(state, X[idx, :], V[idx, :])
        if i % 10 == 0:
            tqdm.write(f"{i:>6}/{epochs}: loss {val:<.5f}")
    return lambda x: state.apply_fn({"params": state.params}, x)
