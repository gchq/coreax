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
from jax.random import PRNGKey
from jax.lax import fori_loop
from jax.typing import ArrayLike
from typing import Callable, Optional, Tuple
from flax.linen import Module
from flax.training.train_state import TrainState
from functools import partial
from coreax.networks import ScoreNetwork
import optax
from tqdm import tqdm

# analytic (reduced variance) loss, for use with certain random measures, e.g. normal
# and Rademacher.
analytic_obj = jit(lambda v, u, s: v @ u + 0.5 * s @ s)

# general loss, for use with general random measures.
general_obj = jit(lambda v, u, s: v @ u + 0.5 * (v @ s) ** 2)


@partial(jit, static_argnames=["score_network", "obj_fn"])
def sliced_score_matching_loss_element(
    x: ArrayLike, v: ArrayLike, score_network: Callable, obj_fn: Callable
) -> float:
    """Element-wise loss function computation.

    Computes the loss function from Section 3.2 of Song el al.'s paper on sliced score
    matching [ssm]_

    :param x: d-dimensional data vector
    :param v: d-dimensional random vector
    :param score_network: function that calls the neural network on x
    :param obj_fn: objective function with arguments (v, u, s) -> real
    :return: objective function output for single x and v inputs
    """
    s, u = jvp(score_network, (x,), (v,))
    return obj_fn(v, u, s)


def sliced_score_matching_loss(score_network: Callable, obj_fn: Callable) -> Callable:
    """Vector mapped loss function for application to arbitrary numbers of X and V
    vectors.

    :param score_network: function that calls the neural network on x
    :param obj_fn: element-wise function (vector, vector, score_network) -> real
    :return: callable vectorised sliced score matching loss function
    """
    inner = vmap(
        lambda x, v: sliced_score_matching_loss_element(x, v, score_network, obj_fn),
        (None, 0),
        0,
    )
    return vmap(inner, (0, 0), 0)


def create_train_state(
    module: Module,
    rng: PRNGKey,
    learning_rate: float,
    dimension: int,
    optimiser: Callable,
) -> TrainState:
    """Creates a flax TrainState for learning with

    :param module: flax network class that inherits flax.nn.Module
    :param rng: random number generator
    :param learning_rate: optimiser learning rate
    :param dimension: data dimension
    :param optimiser: optax optimiser, e.g. optax.adam
    :return: TrainState object
    """
    params = module.init(rng, jnp.ones((1, dimension)))["params"]
    tx = optimiser(learning_rate)
    return TrainState.create(apply_fn=module.apply, params=params, tx=tx)


@partial(jit, static_argnames=["obj_fn"])
def sliced_score_matching_train_step(
    state: TrainState, X: ArrayLike, V: ArrayLike, obj_fn: Callable
) -> Tuple[TrainState, float]:
    """A single training step that updates model parameters using loss gradient.

    :param state: the TrainState object.
    :param X: the n x d data vectors
    :param V: the n x m x d random vectors
    :param obj_fn: objective function (vector, vector, vector) -> real
    :return: the updated TrainState object
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
    """Noise conditioning objective function summand.

    See [improvedsgm]_ for details.

    :param i: loop index
    :param obj: running objective, i.e. the current partial sum
    :param state: the TrainState object
    :param params: the current iterate parameter settings
    :param X: the n x d data vectors
    :param V: the n x m x d random vectors
    :param sigmas: the geometric progression of noise standard deviations
    :param obj_fn: element objective function (vector, vector, vector) -> real
    :return: the updated objective, i.e. partial sum
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
    """A single training step that updates model parameters using loss gradient.

    :param state: the TrainState object.
    :param X: the n x d data vectors
    :param V: the n x m x d random vectors
    :param obj_fn: objective function (vector, vector, vector) -> real
    :param sigmas: length L array of noise standard deviations to use in objective
        function.
    :param L: the static number of terms in the geometric progression. (Required for
        reverse mode autodiff.)
    :return: the updated TrainState object
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


def sliced_score_matching(
    X: ArrayLike,
    rgenerator: Callable,
    noise_conditioning: Optional[bool] = True,
    use_analytic: Optional[bool] = False,
    M: Optional[int] = 1,
    lr: Optional[float] = 1e-3,
    epochs: Optional[int] = 10,
    batch_size: Optional[int] = 64,
    hidden_dim: Optional[int] = 128,
    optimiser: Optional[Callable] = optax.adamw,
    L: Optional[int] = 100,
    sigma: Optional[float] = 1.0,
    gamma: Optional[float] = 0.95,
) -> Callable:
    """Learn a sliced score matching function from Song et al.'s paper [ssm]_

    Currently uses the ScoreNetwork network in coreax.networks.

    :param X: the n x d data vectors
    :param rgenerator: distribution sampler (key, shape, dtype) ->
        jax._src.typing.Array, e.g. distributions in jax.random
    :param noise_conditioning: use the noise conditioning version of score matching.
        Defaults to True.
    :param use_analytic: use the analytic (reduced variance) objective or not. Defaults
        to False.
    :param M: the number of random vectors to use per data vector. Defaults to 1.
    :param lr: optimiser learning rate. Defaults to 1e-3.
    :param epochs: epochs for training. Defaults to 10.
    :param batch_size: size of minibatch. Defaults to 64.
    :param hidden_dim: the ScoreNetwork hidden dimension. Defaults to 128.
    :param optimiser: the optax optimiser to use. Defaults to optax.adam.
    :param L: number of noise models to use in noise conditional score matching.
        Defaults to 100.
    :param sigma: initial noise standard deviation for noise geometric progression in
        noise conditional score matching. Defaults to 1.
    :param gamma: geometric progression ratio. Defaults to 0.95.
    :return: a function that applies the learned score function to input x
    """
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
