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

import jax, flax
from jax import random, vmap, numpy as jnp, jvp, vjp, jit
from flax.training import train_state
from functools import partial
from coreax.networks import ScoreNetwork
import optax
from tqdm import tqdm

@partial(jit, static_argnames=["score_network"])
def sliced_score_matching_loss_element(x, v, score_network):
    # s = score_network(x)
    s, u = jvp(score_network, (x,), (v,))
    # assumes Gaussian and Rademacher random variables for the norm-squared term
    obj = v @ u + .5 * s @ s
    return obj

def sliced_score_matching_loss(score_network):
    inner = vmap(lambda x, v: sliced_score_matching_loss_element(x, v, score_network), (None, 0), 0)
    # return vmap(inner, (0, 0), 0)(X, V).mean()
    return vmap(inner, (0, 0), 0)

def create_train_state(module, rng, learning_rate, dimension):
    params = module.init(rng, jnp.ones((1, dimension)))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=module.apply, params=params, tx=tx)

@jit
def train_step(state, X, V):
    loss = lambda params: sliced_score_matching_loss(lambda x: state.apply_fn({'params': params}, x))(X, V).mean()
    grads = jax.grad(loss)(state.params)
    state = state.apply_gradients(grads=grads)
    return state

def sliced_score_matching(X, rtype="normal", M=1, lr=1e-3, epochs=10, batch_size=64):
    H = 128
    n, d = X.shape
    k1, k2 = random.split(random.PRNGKey(0))
    sn = ScoreNetwork(H, d)
    V = random.normal(k1, (n, M, d))
    # params = sn.init(k2, X)
    # obj = sliced_score_matching_loss(lambda x: sn.apply(params, x))
    # print(obj(X, V))
    # print(sn.tabulate(random.PRNGKey(0), jnp.ones((1, 3))))
    state = create_train_state(sn, k2, lr, d)
    batch_key = random.PRNGKey(1)
    for _ in tqdm(range(epochs)):
        idx = random.randint(batch_key, (batch_size,), 0, n)
        state = train_step(state, X[idx, :], V[idx, :])
    return state

import matplotlib.pyplot as plt
key = random.PRNGKey(0)
# X = random.normal(key, (1000, 1))
X1 = random.normal(key, (500, 1)) + 5.
X2 = random.normal(key, (500, 1)) - 2.
X = jnp.vstack((X1, X2))
state = sliced_score_matching(X, M=1)
x = jnp.linspace(-5, 5, num=100).reshape(-1, 1)
score = state.apply_fn({'params': state.params}, x)
plt.plot(x, score)
plt.savefig("./score.png")
plt.close()
