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

"""Fixtures used across multiple unit tests modules."""

from collections.abc import Callable

import equinox as eqx
import jax
import pytest


@pytest.fixture(params=["with_jit", "without_jit"], scope="class")
def jit_variant(request: pytest.FixtureRequest) -> Callable[[Callable], Callable]:
    """Return a callable that (may) JIT compile a passed callable."""
    if request.param == "without_jit":
        return jax.tree_util.Partial
    if request.param == "with_jit":
        jax.clear_caches()
        return eqx.filter_jit
    raise ValueError("Invalid fixture parametrization.")
