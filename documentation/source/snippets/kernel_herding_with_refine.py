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
from coreax.refine import RefineRegular

# Define a refinement object
refiner = RefineRegular()

# Compute a coreset using kernel herding with a squared exponential kernel.
herding_object = KernelHerding(
    herding_key,
    kernel=SquaredExponentialKernel(length_scale=length_scale),
    refine_method=refiner,
)
herding_object.fit(original_data=data, strategy=SizeReduce(coreset_size=coreset_size))

# Refine the coreset to improve quality
herding_object.refine()

# The herding object now has the refined coreset, and the indices of the original
# data that makeup the refined coreset as populated attributes
print(herding_object.coreset)
print(herding_object.coreset_indices)
