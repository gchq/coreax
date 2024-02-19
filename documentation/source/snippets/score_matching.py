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
import numpy as np

from coreax import SlicedScoreMatching, SteinKernel
from coreax.kernel import PCIMQKernel

# Select indices to form a subset of data for learning score function
generator = np.random.default_rng(random_seed)
idx = generator.choice(len(data), subset_size, replace=False)
data_subset = data[idx, :]

# Learn a score function from a subset of the data, through approximation using a neural
# network applied to a subset of the data
score_key = jax.random.key(random_seed)
sliced_score_matcher = SlicedScoreMatching(
    score_key,
    random_generator=jax.random.rademacher,
    use_analytic=True,
    num_epochs=10,
    num_random_vectors=1,
    sigma=1.0,
    gamma=0.95,
)
score_function = sliced_score_matcher.match(data_subset)

# Define a kernel to use for herding
herding_kernel = SteinKernel(
    PCIMQKernel(length_scale=length_scale),
    score_function=score_function,
)
