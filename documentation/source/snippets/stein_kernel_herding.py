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

from coreax import (
    KernelDensityMatching,
    KernelHerding,
    SizeReduce,
    SquaredExponentialKernel,
    SteinKernel,
)

# Select a subset of data from which to learn score function
generator = np.random.default_rng(random_seed)
idx = generator.choice(len(data), subset_size, replace=False)
data_subset = data[idx, :]

# Learn a score function from the subset of the data, through a kernel density
# estimation applied to a subset of the data.
kernel_density_score_matcher = KernelDensityMatching(length_scale=length_scale)
score_function = kernel_density_score_matcher.match(data_subset)

# Define a kernel to use for herding
herding_kernel = SteinKernel(
    SquaredExponentialKernel(length_scale=length_scale),
    score_function=score_function,
)

# Compute a coreset using kernel herding with a Stein kernel
herding_object = KernelHerding(herding_key, kernel=herding_kernel)
herding_object.fit(original_data=data, strategy=SizeReduce(coreset_size=coreset_size))
