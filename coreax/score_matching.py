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

import jax
from jax import random, vmap, numpy as jnp, jvp, jit
from flax.training import train_state
from functools import partial
from coreax.networks import ScoreNetwork
import optax
from tqdm import tqdm


@partial(jit, static_argnames=["score_network"])
def sliced_score_matching_loss_element(x, v, score_network):
    """Element-wise loss function computation. Computes the loss function from Section 3.2 of 
    Song el al.'s paper on sliced score matching: https://arxiv.org/pdf/1905.07088.pdf.

    Note, this function assumes Gaussian or Rademacher v variables, so it uses the L2 norm squared.

    Args:
        x (arraylike): d-dimensional data vector
        v (arraylike): d-dimensional random vector
        score_network (callable): function that calls the neural network on x

    Returns:
        float: objective function for single x and v.
    """
    s, u = jvp(score_network, (x,), (v,))
    # assumes Gaussian or Rademacher random variables for the norm-squared term
    obj = v @ u + .5 * s @ s
    return obj


def sliced_score_matching_loss(score_network):
    """Vector mapped loss function for application to arbitrary numbers of X and V vectors.

    Args:
        score_network (callable): function that calls the neural network on x

    Returns:
        callable: vectorised sliced score matching loss function
    """
    inner = vmap(lambda x, v: sliced_score_matching_loss_element(
        x, v, score_network), (None, 0), 0)
    return vmap(inner, (0, 0), 0)


def create_train_state(module, rng, learning_rate, dimension, optimiser):
    """Creates a flax TrainState for learning with 

    Args:
        module (flax.nn.Module): flax network class that inherits flax.nn.Module
        rng (jax.random.PRNGKey): random number generator
        learning_rate (float): optimiser learning rate
        dimension (int): data dimension 
        optimiser (callable): optax optimiser, e.g. optax.adam

    Returns:
        flax.train_state.TrainState: TrainState object
    """
    params = module.init(rng, jnp.ones((1, dimension)))['params']
    tx = optimiser(learning_rate)
    return train_state.TrainState.create(apply_fn=module.apply, params=params, tx=tx)


@jit
def train_step(state, X, V):
    """A single training step that updates model parameters using loss gradient.

    Args:
        state (flax.train_state.TrainState): the TrainState object.
        X (arraylike): the n x d data vectors
        V (arraylike): the n x m x d random vectors

    Returns:
        flax.train_state.TrainState: the updated TrainState object 
    """
    def loss(params): return sliced_score_matching_loss(
        lambda x: state.apply_fn({'params': params}, x))(X, V).mean()
    grads = jax.grad(loss)(state.params)
    state = state.apply_gradients(grads=grads)
    return state


def sliced_score_matching(X, rtype="normal", M=1, lr=1e-3, epochs=10, batch_size=64, hidden_dim=128, optimiser=optax.adam):
    """Learn a sliced score matching function from Song et al.'s paper: https://arxiv.org/pdf/1905.07088.pdf.

    Currently uses the ScoreNetwork network in coreax.networks.

    Args:
        X (arraylike): the n x d data vectors
        rtype (str, optional): random vector type. One of "normal" or "rademacher". Defaults to "normal".
        M (int, optional): the number of random vectors to use per data vector. Defaults to 1.
        lr (float, optional): optimiser learning rate. Defaults to 1e-3.
        epochs (int, optional): epochs for training. Defaults to 10.
        batch_size (int, optional): size of minibatch. Defaults to 64.
        hidden_dim (int, optional): the ScoreNetwork hidden dimension. Defaults to 128.
        optimiser (callable, optional): the optax optimiser to use. Defaults to optax.adam.

    Returns:
        callable: a function that applies the learned score function to input x
    """
    n, d = X.shape
    k1, k2 = random.split(random.PRNGKey(0))
    sn = ScoreNetwork(hidden_dim, d)
    if rtype == "normal":
        V = random.normal(k1, (n, M, d))
    elif rtype == "rademacher":
        V = random.rademacher(k1, (n, M, d), dtype=float)
    else:
        raise ValueError(f"{rtype} is not one of 'normal' or 'rademacher'")
    state = create_train_state(sn, k2, lr, d, optimiser)
    batch_key = random.PRNGKey(1)
    for _ in tqdm(range(epochs)):
        idx = random.randint(batch_key, (batch_size,), 0, n)
        state = train_step(state, X[idx, :], V[idx, :])
    return lambda x: state.apply_fn({'params': state.params}, x)
