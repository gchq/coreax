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
import matplotlib.pyplot as plt
import jax.numpy as jnp
from sklearn.datasets import make_blobs
import sys

sys.path.append('../coreax')
from coreax.weights import qp
from coreax.kernel import rbf_kernel, median_heuristic, stein_kernel_pc_imq_element, rbf_grad_log_f_X
from coreax.kernel_herding import stein_kernel_herding_block, scalable_herding
from coreax.metrics import mmd_block, mmd_weight_block

from jax.config import config
import numpy as np
config.update("jax_enable_x64", True)

X, _ = make_blobs(10000, n_features=2, centers=6, random_state=32)
C = 100

N = min(X.shape[0], 1000)
idx = np.random.choice(X.shape[0], N, replace=False)
nu = median_heuristic(X[idx])

k = lambda x, y : rbf_kernel(x, y, jnp.float32(nu)**2)/(nu * jnp.sqrt(2. * jnp.pi))
weighted = True

coreset, Kc, Kbar = stein_kernel_herding_block(X, C, stein_kernel_pc_imq_element, rbf_grad_log_f_X, nu=nu, max_size=1000)

weights = qp(Kc + 1e-10, Kbar)


rsample = np.random.choice(X.shape[0], size=C, replace=False)

if weighted:
    m = mmd_weight_block(X, X[coreset], jnp.ones(X.shape[0]), weights, k, max_size=1000)
else:
    m = mmd_block(X, X[coreset], k, max_size=1000)

rm = mmd_block(X, X[rsample], k, max_size=1000)

plt.scatter(X[:, 0], X[:, 1], s=2., alpha=.1)
plt.scatter(X[coreset, 0], X[coreset, 1], s=weights*1000, color="red")
plt.axis('off')
plt.title('Stein kernel herding, m=%d, MMD=%.6f' % (C, m))
plt.show()

plt.scatter(X[:, 0], X[:, 1], s=2., alpha=.1)
plt.scatter(X[rsample, 0], X[rsample, 1], s=10, color="red")
plt.title('Random, m=%d, MMD=%.6f' % (C, rm))
plt.axis('off')
plt.show()

print("Random")
print(rm)
print("Coreset")
print(m)