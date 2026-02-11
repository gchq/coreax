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

"""Test all pseudocoreset solvers in :module:`coreax.solvers.pseudocoreset`."""

from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar

import equinox as eqx
import jax.random as jr
import jax.tree_util as jtu
import pytest

from coreax.coreset import AbstractCoreset, PseudoCoreset
from coreax.data import Data, SupervisedData
from coreax.solvers.base import PseudoRefinementSolver, PseudoSolver
from coreax.util import KeyArrayLike

_Data = TypeVar("_Data", Data, SupervisedData)
_PseudoSolver = TypeVar("_PseudoSolver", bound=PseudoSolver)
_PseudoRefinementSolver = TypeVar(
    "_PseudoRefinementSolver", bound=PseudoRefinementSolver
)

if TYPE_CHECKING:
    # In Python 3.10, this raises
    # `TypeError: Multiple inheritance with NamedTuple is not supported`.
    # Thus, we have to do the actual full typing here, and a non-generic one
    # below to be used at runtime.
    class _PseudoReduceProblem(NamedTuple, Generic[_Data, _PseudoSolver]):
        dataset: _Data
        solver: _PseudoSolver
        expected_coreset: PseudoCoreset | None = None

    class _PseudoRefineProblem(NamedTuple, Generic[_PseudoRefinementSolver]):
        initial_coreset: PseudoCoreset
        solver: _PseudoRefinementSolver
        expected_coreset: PseudoCoreset | None = None
else:
    # This is the implementation that's used at runtime.
    class _PseudoReduceProblem(NamedTuple):
        dataset: _Data
        solver: _PseudoSolver
        expected_coreset: AbstractCoreset | None = None

    class _PseudoRefineProblem(NamedTuple):
        initial_coreset: PseudoCoreset
        solver: _PseudoRefinementSolver
        expected_coreset: PseudoCoreset | None = None


class PseudoSolverTest:
    """Base tests for all children of :class:`coreax.solvers.PseudoSolver`."""

    random_key: KeyArrayLike = jr.key(2024)
    shape: tuple[int, int] = (64, 10)

    @pytest.fixture
    @abstractmethod
    def solver_factory(self, request: pytest.FixtureRequest) -> jtu.Partial:
        """
        Pytest fixture that returns a partially applied solver initialiser.

        Partial application allows us to modify the init kwargs/args inside the tests.
        """

    @pytest.fixture(scope="class")
    def reduce_problem(
        self,
        request: pytest.FixtureRequest,
        solver_factory: jtu.Partial,
    ) -> _PseudoReduceProblem:
        """
        Pytest fixture that returns a problem dataset and the expected coreset.

        An expected coreset of 'None' implies the expected coreset for this solver and
        dataset combination is unknown.
        """
        del request
        dataset = jr.uniform(self.random_key, self.shape)
        solver = solver_factory()
        expected_coreset = None
        return _PseudoReduceProblem(Data(dataset), solver, expected_coreset)

    def check_solution_invariants(
        self,
        coreset: AbstractCoreset,
        problem: _PseudoRefineProblem | _PseudoReduceProblem,
    ) -> None:
        """
        Check that a coreset obeys certain expected invariant properties.

        1. Check 'coreset.pre_coreset_data' is equal to 'dataset'
        2. Check 'coreset' is equal to 'expected_coreset' (if expected is not 'None')
        """
        dataset, _, expected_coreset = problem
        if isinstance(problem, _PseudoRefineProblem):
            dataset = problem.initial_coreset.pre_coreset_data
        assert isinstance(dataset, Data)
        assert eqx.tree_equal(coreset.pre_coreset_data, dataset)
        if expected_coreset is not None:
            assert isinstance(coreset, type(expected_coreset))
            assert eqx.tree_equal(coreset, expected_coreset)

    @pytest.mark.parametrize(
        "use_cached_state", (False, True), ids=["not_cached", "cached"]
    )
    def test_reduce(
        self,
        jit_variant: Callable[[Callable], Callable],
        reduce_problem: _PseudoReduceProblem,
        use_cached_state: bool,
        **kwargs: Any,
    ) -> None:
        """
        Check 'reduce' raises no errors and is resultant 'solver_state' invariant.

        By resultant 'solver_state' invariant we mean the following procedure succeeds:
        1. Call 'reduce' with the default 'solver_state' to get the resultant state
        2. Call 'reduce' again, this time passing the 'solver_state' from the previous
            run, and keeping all other arguments the same.
        3. Check the two calls to 'refine' yield that same result.
        """
        del kwargs
        dataset, solver, _ = reduce_problem
        _reduce = jit_variant(solver.reduce)
        coreset, state = _reduce(dataset)
        if use_cached_state:
            coreset_with_state, recycled_state = _reduce(dataset, state)
            assert eqx.tree_equal(recycled_state, state)
            assert eqx.tree_equal(coreset_with_state, coreset)
        self.check_solution_invariants(coreset, reduce_problem)
