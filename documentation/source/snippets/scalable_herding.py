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

from coreax.coresubset import KernelHerding
from coreax.kernel import SquaredExponentialKernel
from coreax.reduction import MapReduce

# Compute a coreset using kernel herding with a squared exponential kernel.
herding_object = KernelHerding(
    herding_key,
    kernel=SquaredExponentialKernel(length_scale=length_scale),
)
herding_object.fit(
    original_data=data, strategy=MapReduce(coreset_size=coreset_size, leaf_size=200)
)
