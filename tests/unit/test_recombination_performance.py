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

"""Tests for recombination performance case definitions."""

import jax.numpy as jnp
import pytest

from tests.performance.cases.recombination import _setup_problem, setup_recombination


@pytest.mark.parametrize(
    ("degree", "num_nodes", "num_test_functions"),
    [(3, 27, 19), (4, 64, 34)],
)
def test_problem_size_tracks_polynomial_count(
    degree: int, num_nodes: int, num_test_functions: int
) -> None:
    """Vary degree so the benchmark exercises a larger test-function system."""
    data, powers = _setup_problem(3, degree)

    assert len(data) == num_nodes
    assert powers.shape == (num_test_functions, 3)


def test_cases_cover_recombination_scaling_axes() -> None:
    """Include both algorithms, polynomial sizes and two tree reduction factors."""
    names = [setup.name for setup in setup_recombination()]

    assert names == [
        "recombination_caratheodory_d3_degree3",
        "recombination_caratheodory_d3_degree4",
        "recombination_tree_d3_degree3_factor2",
        "recombination_tree_d3_degree4_factor2",
        "recombination_tree_d3_degree4_factor4",
    ]


@pytest.mark.parametrize("setup", setup_recombination(), ids=lambda setup: setup.name)
def test_cases_run_and_preserve_probability_mass(setup) -> None:
    """Keep every benchmark case executable and its resulting weights normalised."""
    coresubset, state = setup.fn(*setup.fn_args, **(setup.fn_kwargs or {}))

    assert state is None
    assert jnp.sum(coresubset.points.weights) == pytest.approx(1.0, abs=1e-6)
    assert jnp.all(coresubset.points.weights >= 0)
