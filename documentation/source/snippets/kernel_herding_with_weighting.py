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

from coreax import KernelHerding, SizeReduce, SquaredExponentialKernel
from coreax.weights import MMD as MMDWeightsOptimiser

# Define a kernel
kernel = SquaredExponentialKernel(length_scale=length_scale)

# Define a weights optimiser to learn optimal weights for the coreset after creation
weights_optimiser = MMDWeightsOptimiser(kernel=kernel)

# Compute a coreset using kernel herding with a squared exponential kernel.
herding_object = KernelHerding(
    herding_key, kernel=kernel, weights_optimiser=weights_optimiser
)
herding_object.fit(original_data=data, strategy=SizeReduce(coreset_size=coreset_size))

# Determine optimal weights for the coreset
herding_weights = herding_object.solve_weights()
