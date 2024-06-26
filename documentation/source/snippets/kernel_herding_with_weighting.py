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

from coreax import SquaredExponentialKernel
from coreax.solvers import KernelHerding
from coreax.weights import MMDWeightsOptimiser

# Define a kernel
kernel = SquaredExponentialKernel(length_scale=length_scale)

# Define a weights optimiser to learn optimal weights for the coreset after creation
weights_optimiser = MMDWeightsOptimiser(kernel=kernel)

# Compute a coreset using kernel herding with a squared exponential kernel.
herding_solver = KernelHerding(coreset_size, kernel=kernel)
herding_coreset, _ = herding_solver.reduce(data)

# Determine optimal weights for the coreset
re_weighted_herding_coreset = herding_coreset.solve_weights(weights_optimiser)
