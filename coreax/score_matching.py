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
from functools import partial
from coreax.networks import ScoreNetwork

@partial(jit, static_argnames=["score_network"])
def sliced_score_matching_loss_element(x, v, score_network):
    # s = score_network(x)
    s, u = jvp(score_network, (x,), (v,))
    # assumes Gaussian and Rademacher random variables for the norm-squared term
    obj = v @ u + .5 * s @ s
    return obj

def sliced_score_matching_loss(X, V, score_network):
    inner = vmap(lambda x, v: sliced_score_matching_loss_element(x, v, score_network), (None, 0), 0)
    return vmap(inner, (0, 0), 0)(X, V).mean()

def sliced_score_matching(X, rtype="normal", M=1, lr=1e-3, batch_size=64):
    H = 128
    d = 1
    k1, k2 = random.split(random.PRNGKey(0))
    sn = ScoreNetwork(H, d)
    X = random.normal(k1, (10, d))
    V = random.normal(k2, (10, 5, d))
    params = sn.init(k2, X)
    obj = sliced_score_matching_loss(X, V, lambda x: sn.apply(params, x))
    print(obj)
    # print(sn.tabulate(random.PRNGKey(0), jnp.ones((1, 3))))

sliced_score_matching(None)