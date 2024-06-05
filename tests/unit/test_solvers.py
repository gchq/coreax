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

"""Test all solvers in :module:`coreax.solvers`."""

import re
from abc import abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import NamedTuple, Optional, Union, cast
from unittest.mock import MagicMock

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest
from typing_extensions import override

from coreax.coreset import Coreset, Coresubset
from coreax.data import Data
from coreax.kernel import Kernel, LinearKernel, PCIMQKernel, SteinKernel
from coreax.score_matching import KernelDensityMatching, ScoreMatching
from coreax.solvers import (
    KernelHerding,
    MapReduce,
    RandomSample,
    RPCholesky,
    Solver,
    SteinThinning,
)
from coreax.solvers.base import (
    ExplicitSizeSolver,
    PaddingInvariantSolver,
    RefinementSolver,
)
from coreax.solvers.coresubset import (
    HerdingState,
    RPCholeskyState,
    _convert_stein_kernel,  # noqa: PLC2701 - deliberate import/test of private method
)
from coreax.util import KeyArrayLike, tree_zero_pad_leading_axis


class _ReduceProblem(NamedTuple):
    dataset: Data
    solver: Solver
    expected_coreset: Optional[Coreset] = None


class _RefineProblem(NamedTuple):
    initial_coresubset: Coresubset
    solver: RefinementSolver
    expected_coresubset: Optional[Coresubset] = None


class SolverTest:
    """Base tests for all children of :class:`coreax.solvers.Solver`."""

    random_key: KeyArrayLike = jr.key(2024)
    shape: tuple[int, int] = (128, 10)

    @abstractmethod
    def solver_factory(self) -> Union[type[Solver], jtu.Partial]:
        """
        Pytest fixture that returns a partially applied solver initialiser.

        Partial application allows us to modify the init kwargs/args inside the tests.
        """

    @pytest.fixture(scope="class")
    def reduce_problem(
        self,
        request: pytest.FixtureRequest,
        solver_factory: Union[type[Solver], jtu.Partial],
    ) -> _ReduceProblem:
        """
        Pytest fixture that returns a problem dataset and the expected coreset.

        An expected coreset of 'None' implies the expected coreset for this solver and
        dataset combination is unknown.
        """
        del request
        dataset = jr.uniform(self.random_key, self.shape)
        solver = solver_factory()
        expected_coreset = None
        return _ReduceProblem(Data(dataset), solver, expected_coreset)

    def check_solution_invariants(
        self, coreset: Coreset, problem: Union[_RefineProblem, _ReduceProblem]
    ) -> None:
        """
        Check that a coreset obeys certain expected invariant properties.

        1. Check 'coreset.pre_coreset_data' is equal to 'dataset'
        2. Check 'coreset' is equal to 'expected_coreset' (if expected is not 'None')
        3. If 'isinstance(coreset, Coresubset)', check coreset is a subset of 'dataset'
        4. If 'not hasattr(solver, random_key))', check that the
            addition of zero weighted data-points to the leading axis of the input
            'dataset' does not modify the resulting coreset when the solver is a
            'PaddingInvariantSolver'.
        """
        dataset, _, expected_coreset = problem
        if isinstance(problem, _RefineProblem):
            dataset = problem.initial_coresubset.pre_coreset_data
        assert eqx.tree_equal(coreset.pre_coreset_data, dataset)
        if expected_coreset is not None:
            assert isinstance(coreset, type(expected_coreset))
            assert eqx.tree_equal(coreset, expected_coreset)
        if isinstance(coreset, Coresubset):
            membership = jtu.tree_map(jnp.isin, coreset.coreset, dataset)
            all_membership = jtu.tree_map(jnp.all, membership)
            assert jtu.tree_all(all_membership)
        if isinstance(problem.solver, PaddingInvariantSolver):
            padded_dataset = tree_zero_pad_leading_axis(dataset, len(dataset))
            if isinstance(problem, _RefineProblem):
                padded_initial_coreset = eqx.tree_at(
                    lambda x: x.pre_coreset_data,
                    problem.initial_coresubset,
                    padded_dataset,
                )
                coreset_from_padded, _ = problem.solver.refine(padded_initial_coreset)
            else:
                coreset_from_padded, _ = problem.solver.reduce(padded_dataset)
            assert eqx.tree_equal(coreset_from_padded.coreset, coreset.coreset)

    @pytest.mark.parametrize("use_cached_state", (False, True))
    def test_reduce(
        self,
        jit_variant: Callable[[Callable], Callable],
        reduce_problem: _ReduceProblem,
        use_cached_state: bool,
    ) -> None:
        """
        Check 'reduce' raises no errors and is resultant 'solver_state' invariant.

        By resultant 'solver_state' invariant we mean the following procedure succeeds:
        1. Call 'reduce' with the default 'solver_state' to get the resultant state
        2. Call 'reduce' again, this time passing the 'solver_state' from the previous
            run, and keeping all other arguments the same.
        3. Check the two calls to 'refine' yield that same result.
        """
        dataset, solver, _ = reduce_problem
        coreset, state = jit_variant(solver.reduce)(dataset)
        if use_cached_state:
            coreset_with_state, recycled_state = solver.reduce(dataset, state)
            assert eqx.tree_equal(recycled_state, state)
            assert eqx.tree_equal(coreset_with_state, coreset)
        self.check_solution_invariants(coreset, reduce_problem)


class RefinementSolverTest(SolverTest):
    """Test cases for coresubset solvers that provide a 'refine' method."""

    @pytest.fixture(
        params=[
            "well-sized",
            "under-sized",
            "over-sized",
            "random",
            "random-zero-weights",
        ],
        scope="class",
    )
    def refine_problem(
        self, request: pytest.FixtureRequest, reduce_problem: _ReduceProblem
    ) -> _RefineProblem:
        """
        Pytest fixture that returns a problem dataset and the expected coreset.

        An expected coreset of 'None' implies the expected coreset for this solver and
        dataset combination is unknown.

        We expect the '{well,under,over}-sized' and the 'random-zero-weights' cases to
        return the same result as a call to 'reduce'. The 'random' case we only expect
        to pass without raising an error.
        """
        dataset, solver, expected_coreset = reduce_problem
        indices_key, weights_key = jr.split(self.random_key)
        solver = cast(KernelHerding, solver)
        coreset_size = min(len(dataset), solver.coreset_size)
        # We expect 'refine' to produce the same result as 'reduce' when the initial
        # coresubset has all its indices equal to zero.
        expected_coresubset = None
        if expected_coreset is None:
            expected_coresubset, _ = solver.reduce(dataset)
        elif isinstance(expected_coreset, Coresubset):
            expected_coresubset = expected_coreset
        if request.param == "well-sized":
            indices = Data(jnp.zeros(coreset_size, jnp.int32), 0)
        elif request.param == "under-sized":
            indices = Data(jnp.zeros(coreset_size - 1, jnp.int32), 0)
        elif request.param == "over-sized":
            indices = Data(jnp.zeros(coreset_size + 1, jnp.int32), 0)
        elif request.param == "random":
            random_indices = jr.choice(indices_key, len(dataset), (coreset_size,))
            random_weights = jr.uniform(weights_key, (coreset_size,))
            indices = Data(random_indices, random_weights)
            expected_coresubset = None
        elif request.param == "random-zero-weights":
            random_indices = jr.choice(indices_key, len(dataset), (coreset_size,))
            indices = Data(random_indices, 0)
        else:
            raise ValueError("Invalid fixture parametrization")
        initial_coresubset = Coresubset(indices, dataset)
        return _RefineProblem(initial_coresubset, solver, expected_coresubset)

    @pytest.mark.parametrize("use_cached_state", (False, True))
    def test_refine(
        self,
        jit_variant: Callable[[Callable], Callable],
        refine_problem: _RefineProblem,
        use_cached_state: bool,
    ) -> None:
        """
        Check 'refine' raises no errors and is resultant 'solver_state' invariant.

        By resultant 'solver_state' invariant we mean the following procedure succeeds:
        1. Call 'reduce' with the default 'solver_state' to get the initial coresubset
        2. Call 'refine' with the default 'solver_state' to get the resultant state
        3. Call 'refine' again, this time passing the 'solver_state' from the previous
            run, and keeping all other arguments the same.
        4. Check the two calls to 'refine' yield that same result.
        """
        initial_coresubset, solver, _ = refine_problem
        coresubset, state = jit_variant(solver.refine)(initial_coresubset)
        if use_cached_state:
            coresubset_cached_state, recycled_state = solver.refine(
                initial_coresubset, state
            )
            assert eqx.tree_equal(recycled_state, state)
            assert eqx.tree_equal(coresubset_cached_state, coresubset)
        self.check_solution_invariants(coresubset, refine_problem)


class ExplicitSizeSolverTest(SolverTest):
    """Test cases for solvers that have an explicitly specified 'coreset_size'."""

    SIZE_MSG = "'coreset_size' must be a positive integer"
    OVERSIZE_MSG = "'coreset_size' must be less than 'len(dataset)'"

    @override
    def check_solution_invariants(
        self, coreset: Coreset, problem: Union[_RefineProblem, _ReduceProblem]
    ) -> None:
        super().check_solution_invariants(coreset, problem)
        solver = problem.solver
        if isinstance(solver, ExplicitSizeSolver):
            assert len(coreset) == solver.coreset_size

    @pytest.mark.parametrize(
        "coreset_size, context",
        [
            (1, does_not_raise()),
            (1.2, does_not_raise()),
            (0, pytest.raises(ValueError, match=SIZE_MSG)),
            (-1, pytest.raises(ValueError, match=SIZE_MSG)),
            ("not_cast-able_to_int", pytest.raises(ValueError)),
        ],
    )
    def test_check_init(
        self,
        solver_factory: jtu.Partial,
        coreset_size: Union[int, float, str],
        context: AbstractContextManager,
    ) -> None:
        """
        Ensure '__check_init__' prevents initialisations with infeasible 'coreset_size'.

        A 'coreset_size' is infeasible if it can't be cast to a positive integer.
        """
        solver_factory.keywords["coreset_size"] = coreset_size
        with context:
            solver = solver_factory()
            assert solver.coreset_size == int(coreset_size)

    def test_reduce_oversized_coreset_size(
        self, reduce_problem: _ReduceProblem
    ) -> None:
        """
        Check error is raised if 'coreset_size' is too large.

        This can not be handled in '__check_init__' as the 'coreset_size' is being
        compared against the 'dataset', which is only passed in the 'reduce' call.
        """
        dataset, solver, _ = reduce_problem
        modified_solver = eqx.tree_at(
            lambda x: x.coreset_size, solver, len(dataset) + 1
        )
        with pytest.raises(ValueError, match=re.escape(self.OVERSIZE_MSG)):
            modified_solver.reduce(dataset)


class TestKernelHerding(RefinementSolverTest, ExplicitSizeSolverTest):
    """Test cases for :class:`coreax.solvers.coresubset.KernelHerding`."""

    @override
    @pytest.fixture(scope="class")
    def solver_factory(self) -> jtu.Partial:
        kernel = PCIMQKernel()
        coreset_size = self.shape[0] // 10
        return jtu.Partial(KernelHerding, coreset_size=coreset_size, kernel=kernel)

    @override
    @pytest.fixture(params=["random", "analytic"], scope="class")
    def reduce_problem(
        self,
        request: pytest.FixtureRequest,
        solver_factory: Union[type[Solver], jtu.Partial],
    ) -> _ReduceProblem:
        if request.param == "random":
            dataset = jr.uniform(self.random_key, self.shape)
            solver = solver_factory()
            expected_coreset = None
        elif request.param == "analytic":
            dataset = jnp.array([[0, 0], [1, 1], [2, 2]])
            # Set the kernel such that a simple analytic solution exists.
            kernel_matrix = jnp.asarray([[1, 1, 1], [1, 1, 1], [0.5, 0.5, 0.5]])
            kernel = MagicMock(Kernel)
            kernel.compute = lambda x, y: jnp.hstack(
                [kernel_matrix[:, y[0]], jnp.zeros((len(x) - 3,))]
            )
            kernel.compute_mean = lambda x, y, **kwargs: jnp.zeros(len(x))
            kernel.gramian_row_mean = lambda x, **kwargs: jnp.hstack(
                [jnp.asarray([0.6, 0.75, 0.55]), jnp.zeros((len(x) - 3,))]
            )
            solver = KernelHerding(coreset_size=2, kernel=kernel, unique=True)
            # The selection criterion for kernel herding is the argmax of the difference
            # between the `gramian_row_mean` generated by the original data and the
            # kernel similarity penalty (computed from the iteratively updated coreset).
            # Due to the simplicity of the problem setup here, we can reason about the
            # `expected_coreset`/solution as follows: Because Index 1 has the highest
            # `gramian_row_mean`, and the kernel similarity penalty is initialised to
            # zero it will be the first selected coreset point. Acknowledging that
            # `unique=True`, the second selection must be one of either '0' or '2'.
            # While '0' has a larger `gramian_row_mean`, after application of the kernel
            # similarity penalty (updated with the result of the first index), the
            # highest scoring index is Index 2, completing our expected coreset.
            expected_coreset = Coresubset(jnp.array([1, 2]), Data(dataset))
        else:
            raise ValueError("Invalid fixture parametrization")
        return _ReduceProblem(Data(dataset), solver, expected_coreset)

    def test_herding_state(self, reduce_problem: _ReduceProblem) -> None:
        """Check that the cached herding state is as expected."""
        dataset, solver, _ = reduce_problem
        solver = cast(KernelHerding, solver)
        _, state = solver.reduce(dataset)
        expected_state = HerdingState(solver.kernel.gramian_row_mean(dataset))
        assert eqx.tree_equal(state, expected_state)


class TestRandomSample(ExplicitSizeSolverTest):
    """Test cases for :class:`coreax.solvers.coresubset.RandomSample`."""

    @override
    def check_solution_invariants(
        self, coreset: Coreset, problem: Union[_RefineProblem, _ReduceProblem]
    ) -> None:
        super().check_solution_invariants(coreset, problem)
        solver = cast(RandomSample, problem.solver)
        if solver.unique:
            _, counts = jnp.unique(coreset.nodes.data, return_counts=True)
            assert max(counts) <= 1

    @override
    @pytest.fixture(scope="class")
    def solver_factory(self) -> jtu.Partial:
        coreset_size = self.shape[0] // 10
        key = jr.fold_in(self.random_key, self.shape[0])
        return jtu.Partial(RandomSample, coreset_size=coreset_size, random_key=key)


class TestRPCholesky(ExplicitSizeSolverTest):
    """Test cases for :class:`coreax.solvers.coresubset.RPCholesky`."""

    @override
    def check_solution_invariants(
        self, coreset: Coreset, problem: Union[_RefineProblem, _ReduceProblem]
    ) -> None:
        """Check functionality of 'unique' in addition to the default checks."""
        super().check_solution_invariants(coreset, problem)
        solver = cast(RPCholesky, problem.solver)
        if solver.unique:
            _, counts = jnp.unique(coreset.nodes.data, return_counts=True)
            assert max(counts) <= 1

    @override
    @pytest.fixture(scope="class")
    def solver_factory(self) -> Union[type[Solver], jtu.Partial]:
        kernel = PCIMQKernel()
        coreset_size = self.shape[0] // 10
        return jtu.Partial(
            RPCholesky,
            coreset_size=coreset_size,
            random_key=self.random_key,
            kernel=kernel,
        )

    def test_rpcholesky_state(self, reduce_problem: _ReduceProblem) -> None:
        """Check that the cached RPCholesky state is as expected."""
        dataset, solver, _ = reduce_problem
        solver = cast(RPCholesky, solver)
        _, state = solver.reduce(dataset)
        x = dataset.data
        gramian_diagonal = jax.vmap(solver.kernel.compute_elementwise)(x, x)
        expected_state = RPCholeskyState(gramian_diagonal)
        assert eqx.tree_equal(state, expected_state)


class TestSteinThinning(RefinementSolverTest, ExplicitSizeSolverTest):
    """Test cases for :class:`coreax.solvers.coresubset.SteinThinning`."""

    @override
    @pytest.fixture(scope="class")
    def solver_factory(self) -> jtu.Partial:
        kernel = PCIMQKernel()
        coreset_size = self.shape[0] // 10
        return jtu.Partial(SteinThinning, coreset_size=coreset_size, kernel=kernel)

    @pytest.mark.parametrize(
        "kernel", (PCIMQKernel(), SteinKernel(LinearKernel(), MagicMock()))
    )
    @pytest.mark.parametrize("score_matching", (None, MagicMock()))
    def test_convert_stein_kernel(
        self, kernel: Kernel, score_matching: Union[ScoreMatching, None]
    ) -> None:
        """Check handling of Stein kernels and standard kernels is consistent."""
        dataset = jr.uniform(self.random_key, self.shape)
        converted_kernel = _convert_stein_kernel(dataset, kernel, score_matching)

        if isinstance(kernel, SteinKernel):
            if score_matching is not None:
                expected_kernel = eqx.tree_at(
                    lambda x: x.score_function, kernel, score_matching.match(dataset)
                )
            expected_kernel = kernel
        else:
            if score_matching is None:
                length_scale = getattr(kernel, "length_scale", 1.0)
                score_matching = KernelDensityMatching(length_scale)
            expected_kernel = SteinKernel(
                kernel, score_function=score_matching.match(dataset)
            )
        assert eqx.tree_equal(converted_kernel.base_kernel, expected_kernel.base_kernel)
        # Score function hashes won't match; resort to checking identical evaluation.
        assert eqx.tree_equal(
            converted_kernel.score_function(dataset),
            expected_kernel.score_function(dataset),
        )


class _ExplicitPaddingInvariantSolver(ExplicitSizeSolver, PaddingInvariantSolver):
    """Required for mocking with multiple inheritance."""


class TestMapReduce(SolverTest):
    """Test cases for :class:`coreax.solvers.composite.MapReduce`."""

    leaf_size: int = 32
    coreset_size: int = 16

    @override
    @pytest.fixture(scope="class")
    def solver_factory(self) -> Union[type[Solver], jtu.Partial]:
        base_solver = MagicMock(_ExplicitPaddingInvariantSolver)
        base_solver.coreset_size = self.coreset_size

        def mock_reduce(
            dataset: Data, solver_state: None = None
        ) -> tuple[Coreset[Data], None]:
            indices = jnp.arange(base_solver.coreset_size)
            return Coreset(dataset[indices], dataset), solver_state

        base_solver.reduce = mock_reduce

        class _MockTree:
            def __init__(self, _data: np.ndarray, **kwargs):
                del kwargs
                self.data = _data

            def get_arrays(self) -> tuple[Union[np.ndarray | None], ...]:
                """Mock sklearn.neighbours.BinaryTree.get_arrays method."""
                return None, np.arange(len(self.data)), None, None

        return jtu.Partial(
            MapReduce,
            base_solver=base_solver,
            leaf_size=self.leaf_size,
            tree_type=_MockTree,
        )

    @override
    @pytest.fixture(scope="class")
    def reduce_problem(
        self,
        request: pytest.FixtureRequest,
        solver_factory: Union[type[Solver], jtu.Partial],
    ) -> _ReduceProblem:
        del request
        dataset = jnp.broadcast_to(jnp.arange(self.shape[0])[..., None], self.shape)
        solver = solver_factory()
        # Expected procedure:
        # len(dataset) = 128; leaf_size=32
        # (1): 128 -> Partition -> 4x32 -> Reduce -> 4x16 -> Reshape -> 64
        # (2): 64  -> Partition -> 2x32 -> Reduce -> 2x16 -> Reshape -> 32
        # (3): 32  -> Partition -> 1x32 -> Reduce -> 1x16 -> Reshape -> 16
        # Expected sub-coreset values at each step.
        # (1): [:16], [32:48], [64:80], [96:112]
        # (2): [:16], [64:80]
        # (3): [:16] <- 'coreset_size'
        expected_coreset = Coreset(dataset[: self.coreset_size], Data(dataset))
        return _ReduceProblem(Data(dataset), solver, expected_coreset)

    @pytest.mark.parametrize(
        "leaf_size, context",
        [
            (-1, pytest.raises(ValueError, match="must be larger")),
            (0, pytest.raises(ValueError, match="must be larger")),
            (coreset_size, pytest.raises(ValueError, match="must be larger")),
            (coreset_size + 1, does_not_raise()),
            ("str", pytest.raises(ValueError)),
        ],
    )
    def test_leaf_sizes(
        self,
        solver_factory: jtu.Partial,
        leaf_size: int,
        context: AbstractContextManager,
    ) -> None:
        """Check that invalid 'leaf_size' raises a suitable error."""
        with context:
            solver_factory.keywords["leaf_size"] = leaf_size
            solver_factory()

    @pytest.mark.parametrize(
        "base_solver, context",
        (
            (MagicMock(_ExplicitPaddingInvariantSolver), does_not_raise()),
            (
                MagicMock(ExplicitSizeSolver),
                pytest.warns(UserWarning, match="PaddingInvariantSolver"),
            ),
            (MagicMock(Solver), pytest.raises(ValueError, match="ExplicitSizeSolver")),
        ),
    )
    @pytest.mark.filterwarnings("error")
    def test_base_solver(
        self,
        solver_factory: jtu.Partial,
        base_solver: Solver,
        context: AbstractContextManager,
    ):
        """Check that invalid 'base_solver' raises an error or warns."""
        with context:
            if isinstance(base_solver, ExplicitSizeSolver):
                base_solver.coreset_size = self.leaf_size - 1
            solver_factory.keywords["leaf_size"] = self.leaf_size
            solver_factory.keywords["base_solver"] = base_solver
            solver_factory()
