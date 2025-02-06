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

"""A simple test that serves to normalise other performance test results."""

from jax import (
    Array,
    numpy as jnp,
    random as jr,
)

from coreax.util import JITCompilableFunction


def _setup_data() -> tuple[Array, Array]:
    """Set up a large random matrix to be operated on."""
    key = jr.key(1_234)
    # big matrix to ensure high execution time
    matrix = jr.normal(key, (4000, 4000))
    small_matrix = jr.normal(key, (50, 50))
    return matrix, small_matrix


def _operate(matrix: Array, small_matrix: Array) -> Array:
    """Perform some arbitrary operations on a matrix."""
    squared = jnp.square(matrix)
    halved_transpose = matrix.T / 2
    for _ in range(1_000):
        # for loop is here to ensure high compilation time
        small_matrix @= small_matrix
    result = squared @ halved_transpose
    return result.trace() + small_matrix.trace()


def setup_normaliser() -> JITCompilableFunction:
    """Set up a test against which other performance tests will be normalised."""
    return JITCompilableFunction(
        fn=_operate,
        fn_args=_setup_data(),
        fn_kwargs=None,
        jit_kwargs=None,
        name="NORMALISER",
    )
