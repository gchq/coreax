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

"""Keep solver helper options explicit without changing public solver constructors."""

# These tests check the private helper calling contracts.
# ruff: noqa: PLC2701

import inspect
from collections.abc import Callable

import pytest

from coreax.solvers.coresubset import (
    _greedy_kernel_points_loss,
    _greedy_kernel_selection,
    _setup_batch_solver,
    _update_candidate_coresets_and_coreset_indices,
)


@pytest.mark.parametrize(
    ("helper", "options"),
    [
        (_greedy_kernel_selection, ["unique", "block_size", "unroll"]),
        (
            _setup_batch_solver,
            [
                "candidate_batch_size",
                "loss_batch_size",
                "random_key",
                "setup_identity",
                "setup_loss_batch_indices",
            ],
        ),
        (
            _update_candidate_coresets_and_coreset_indices,
            [
                "unique",
                "candidate_coresets",
                "coreset_indices",
                "loss",
                "candidate_batch_indices",
            ],
        ),
        (
            _greedy_kernel_points_loss,
            [
                "regularisation_parameter",
                "identity",
                "least_squares_solver",
                "loss_batch",
            ],
        ),
    ],
)
def test_options_are_keyword_only(helper: Callable, options: list[str]) -> None:
    """Positional arguments must not silently swap solver settings or state arrays."""
    parameters = inspect.signature(helper).parameters
    for option in options:
        assert parameters[option].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    ("helper", "argument_count"),
    [
        (_greedy_kernel_selection, 7),
        (_setup_batch_solver, 8),
        (_update_candidate_coresets_and_coreset_indices, 6),
        (_greedy_kernel_points_loss, 7),
    ],
)
def test_positional_options_are_rejected(helper: Callable, argument_count: int) -> None:
    """Reject the old positional calling form before attempting any array operations."""
    with pytest.raises(TypeError, match="positional"):
        helper(*([None] * argument_count))
