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

import numpy as np
from sklearn.datasets import make_blobs

from coreax.data import Data
from coreax.kernels import SquaredExponentialKernel, median_heuristic
from coreax.solvers import KernelHerding

# Generate some data
num_data_points = 10_000
num_features = 2
num_cluster_centers = 6
random_seed = 1989
x, *_ = make_blobs(
    num_data_points,
    n_features=num_features,
    centers=num_cluster_centers,
    random_state=random_seed,
)

# Request 100 coreset points
coreset_size = 100

# Setup the original data object
data = Data(x)

# Set the bandwidth parameter of the kernel using a median heuristic derived from
# at most 1000 random samples in the data.
num_samples_length_scale = min(num_data_points, 1_000)
generator = np.random.default_rng(random_seed)
idx = generator.choice(num_data_points, num_samples_length_scale, replace=False)
length_scale = median_heuristic(x[idx])

# Compute a coresubset using kernel herding with a squared exponential kernel.
herding_solver = KernelHerding(
    coreset_size, kernel=SquaredExponentialKernel(length_scale=length_scale)
)
herding_coreset, _ = herding_solver.reduce(data)

# We can now print the selected coresubset indices and the materialized coresubset
print(herding_coreset.unweighted_indices)
print(herding_coreset.points)
