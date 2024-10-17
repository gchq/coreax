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
from contextlib import (
    AbstractContextManager,
    nullcontext as does_not_raise,
)
from typing import Literal, NamedTuple, Optional, Union, cast
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
from coreax.data import Data, SupervisedData
from coreax.kernels import PCIMQKernel, ScalarValuedKernel, SquaredExponentialKernel
from coreax.least_squares import RandomisedEigendecompositionSolver
from coreax.solvers import (
    CaratheodoryRecombination,
    GreedyKernelPoints,
    GreedyKernelPointsState,
    HerdingState,
    KernelHerding,
    MapReduce,
    RandomSample,
    RPCholesky,
    RPCholeskyState,
    Solver,
    SteinThinning,
    TreeRecombination,
)
from coreax.solvers.base import (
    ExplicitSizeSolver,
    PaddingInvariantSolver,
    RefinementSolver,
)
from coreax.util import KeyArrayLike, tree_zero_pad_leading_axis


class _ReduceProblem(NamedTuple):
    dataset: Union[Data, SupervisedData]
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
        3. If 'isinstance(coreset, Coresubset)', check coreset is a subset of 'dataset';
            note: 'coreset.weights' doesn't need to be a subset of dataset.weights
        4. If 'isinstance(problem.solver, PaddingInvariantSolver)', check that the
            addition of zero weighted data-points to the leading axis of the input
            'dataset' does not modify the resulting coreset when the solver is a
            'PaddingInvariantSolver'.
        """
        dataset, solver, expected_coreset = problem
        if isinstance(problem, _RefineProblem):
            dataset = problem.initial_coresubset.pre_coreset_data
        assert eqx.tree_equal(coreset.pre_coreset_data, dataset)
        if expected_coreset is not None:
            assert isinstance(coreset, type(expected_coreset))
            assert eqx.tree_equal(coreset, expected_coreset)
        if isinstance(coreset, Coresubset):
            membership = jtu.tree_map(jnp.isin, coreset.coreset.data, dataset.data)
            all_membership = jtu.tree_map(jnp.all, membership)
            assert jtu.tree_all(all_membership)
        if isinstance(solver, PaddingInvariantSolver):
            padded_dataset = tree_zero_pad_leading_axis(dataset, len(dataset))
            if isinstance(problem, _RefineProblem):
                padded_initial_coreset = eqx.tree_at(
                    lambda x: x.pre_coreset_data,
                    problem.initial_coresubset,
                    padded_dataset,
                )
                coreset_from_padded, _ = solver.refine(padded_initial_coreset)
            else:
                coreset_from_padded, _ = solver.reduce(padded_dataset)
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


class RecombinationSolverTest(SolverTest):
    """Test cases for coresubset solvers that perform recombination."""

    @override
    @pytest.fixture(
        params=["random", "partial-null", "null", "full_rank", "rank_deficient"],
        scope="class",
    )
    def reduce_problem(
        self, request: pytest.FixtureRequest, solver_factory: Union[Solver, jtu.Partial]
    ) -> _ReduceProblem:
        node_key, weight_key, rng_key = jr.split(self.random_key, num=3)
        nodes = jr.uniform(node_key, self.shape)
        weights = jr.uniform(weight_key, (self.shape[0],))
        expected_coreset = None
        if request.param == "random":
            test_functions = None
        elif request.param == "partial-null":
            zero_weights = jr.choice(rng_key, self.shape[0], (self.shape[0] // 2,))
            weights = weights.at[zero_weights].set(0)
            test_functions = None
        elif request.param == "null":

            def test_functions(x):
                return jnp.zeros(x.shape)
        elif request.param == "full_rank":

            def test_functions(x):
                norm_x = jnp.linalg.norm(x)
                return jnp.array([norm_x, norm_x**2, norm_x**3])
        elif request.param == "rank_deficient":

            def test_functions(x):
                norm_x = jnp.linalg.norm(x)
                return jnp.array([norm_x, 2 * norm_x, 2 + norm_x])
        else:
            raise ValueError("Invalid fixture parametrization")
        solver_factory.keywords["test_functions"] = test_functions
        solver = solver_factory()
        return _ReduceProblem(Data(nodes, weights), solver, expected_coreset)

    @override
    def check_solution_invariants(
        self, coreset: Coreset, problem: Union[_RefineProblem, _ReduceProblem]
    ) -> None:
        r"""
        Check that a coreset obeys certain expected invariant properties.

        In addition to the standard checks in the parent class we also check:
        1. Check 'sum(coreset.weights)' is one.
        1. Check 'len(coreset)' is less than or equal to the upper bound `m`.
        2. Check 'len(coreset[idx]) where idx = jnp.nonzero(coreset.weights)' is less
            than or equal to the rank, :math:`m^\prime`, of the pushed forward nodes.
        3. Check the push-forward of the coreset preserves the "centre-of-mass" (CoM) of
            the pushed-forward dataset (with implicit and explicit zero weight removal).
        4. Check the default value of 'test_functions' is the identity map.
        """
        super().check_solution_invariants(coreset, problem)
        dataset, solver, _ = problem
        coreset_nodes, coreset_weights = coreset.coreset.data, coreset.coreset.weights
        assert eqx.tree_equal(jnp.sum(coreset_weights), jnp.asarray(1.0), rtol=5e-5)
        if solver.test_functions is None:
            solver = eqx.tree_at(
                lambda x: x.test_functions,
                solver,
                lambda x: x,
                is_leaf=lambda x: x is None,
            )
            expected_default_coreset, _ = solver.reduce(dataset)
            assert eqx.tree_equal(coreset, expected_default_coreset)

        vmap_test_functions = jax.vmap(solver.test_functions)
        pushed_forward_nodes = vmap_test_functions(dataset.data)
        augmented_pushed_forward_nodes = jnp.c_[
            jnp.ones_like(dataset.weights), pushed_forward_nodes
        ]
        rank = jnp.linalg.matrix_rank(augmented_pushed_forward_nodes)
        max_rank = augmented_pushed_forward_nodes.shape[-1]
        assert rank <= max_rank
        non_zero = jnp.flatnonzero(coreset_weights)
        if solver.mode == "implicit-explicit":
            assert len(coreset) <= max_rank
            assert len(non_zero) <= len(coreset) - (max_rank - rank)
        if solver.mode == "implicit":
            assert len(coreset) == len(augmented_pushed_forward_nodes)
            assert len(non_zero) <= len(coreset) - (max_rank - rank)
        if solver.mode == "explicit":
            assert len(non_zero) == len(coreset)
            assert len(coreset) <= rank
        pushed_forward_com = jnp.average(
            pushed_forward_nodes, 0, weights=dataset.weights
        )
        pushed_forward_coreset_nodes = vmap_test_functions(
            jnp.atleast_2d(coreset_nodes)
        )
        coreset_pushed_forward_com = jnp.average(
            pushed_forward_coreset_nodes, 0, weights=coreset_weights
        )
        assert eqx.tree_equal(pushed_forward_com, coreset_pushed_forward_com, rtol=1e-5)
        explicit_coreset_pushed_forward_com = jnp.average(
            pushed_forward_coreset_nodes[non_zero], 0, weights=coreset_weights[non_zero]
        )
        assert eqx.tree_equal(
            coreset_pushed_forward_com, explicit_coreset_pushed_forward_com, rtol=1e-5
        )

    @override
    @pytest.mark.parametrize("use_cached_state", (False, True))
    @pytest.mark.parametrize(
        "recombination_mode, context",
        (
            ("implicit-explicit", does_not_raise()),
            ("implicit", does_not_raise()),
            (
                "explicit",
                pytest.raises(ValueError, match="'explicit' mode is incompatible"),
            ),
            (None, pytest.raises(ValueError, match="Invalid mode")),
        ),
    )
    # We don't care too much that arguments differ as this is required to override the
    # parametrization. Nevertheless, this should probably be revisited in the future.
    def test_reduce(  # pylint: disable=arguments-differ
        self,
        jit_variant: Callable[[Callable], Callable],
        reduce_problem: _ReduceProblem,
        use_cached_state: bool,
        recombination_mode: Literal["implicit-explicit", "implicit", "explicit"],
        context: AbstractContextManager,
    ) -> None:
        """
        Check 'reduce' raises no errors and is resultant 'solver_state' invariant.

        Overrides the default implementation to provide handling of different modes of
        recombination.

        By resultant 'solver_state' invariant we mean the following procedure succeeds:
        1. Call 'reduce' with the default 'solver_state' to get the resultant state
        2. Call 'reduce' again, this time passing the 'solver_state' from the previous
            run, and keeping all other arguments the same.
        3. Check the two calls to 'refine' yield that same result.
        """
        dataset, base_solver, expected_coreset = reduce_problem
        solver = eqx.tree_at(lambda x: x.mode, base_solver, recombination_mode)
        updated_problem = _ReduceProblem(dataset, solver, expected_coreset)
        # Explicit should only raise if jit_variant is eqx.filter_jit (or jax.jit).
        if jit_variant is not eqx.filter_jit and recombination_mode == "explicit":
            context = does_not_raise()
        with context:
            coreset, state = jit_variant(solver.reduce)(dataset)
            if use_cached_state:
                coreset_with_state, recycled_state = solver.reduce(dataset, state)
                assert eqx.tree_equal(recycled_state, state)
                assert eqx.tree_equal(coreset_with_state, coreset)
            self.check_solution_invariants(coreset, updated_problem)


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
            kernel = MagicMock(ScalarValuedKernel)
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


class TestGreedyKernelPoints(RefinementSolverTest, ExplicitSizeSolverTest):
    """Test cases for :class:`coreax.solvers.coresubset.GreedyKernelPoints`."""

    @override
    @pytest.fixture(scope="class")
    def solver_factory(self) -> jtu.Partial:
        feature_kernel = SquaredExponentialKernel()
        coreset_size = self.shape[0] // 10
        return jtu.Partial(
            GreedyKernelPoints,
            random_key=self.random_key,
            coreset_size=coreset_size,
            feature_kernel=feature_kernel,
        )

    @override
    @pytest.fixture(params=["random"], scope="class")
    def reduce_problem(
        self,
        request: pytest.FixtureRequest,
        solver_factory: Union[type[Solver], jtu.Partial],
    ) -> _ReduceProblem:
        if request.param == "random":
            data_key, supervision_key = jr.split(self.random_key)
            data = jr.uniform(data_key, self.shape)
            supervision = jr.uniform(supervision_key, (self.shape[0], 1))
            solver = solver_factory()
            expected_coreset = None
        else:
            raise ValueError("Invalid fixture parametrization")
        return _ReduceProblem(
            SupervisedData(data=data, supervision=supervision), solver, expected_coreset
        )

    @override
    def check_solution_invariants(
        self, coreset: Coreset, problem: Union[_RefineProblem, _ReduceProblem]
    ) -> None:
        """Check functionality of 'unique' in addition to the default checks."""
        super().check_solution_invariants(coreset, problem)
        solver = cast(GreedyKernelPoints, problem.solver)
        if solver.unique:
            _, counts = jnp.unique(coreset.nodes.data, return_counts=True)
            assert max(counts) <= 1

    @pytest.fixture(
        params=[
            "well-sized",
            "under-sized",
            "over-sized",
            "random-unweighted",
            "random-weighted",
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

        We expect the '{well,under,over}-sized' cases to return the same result as a
        call to 'reduce'. The 'random-unweighted' and 'random-weighted' case we only
        expect to pass without raising an error.
        """
        dataset, solver, expected_coreset = reduce_problem
        indices_key, weights_key = jr.split(self.random_key)
        solver = cast(GreedyKernelPoints, solver)
        coreset_size = min(len(dataset), solver.coreset_size)
        # We expect 'refine' to produce the same result as 'reduce' when the initial
        # coresubset has all its indices equal to negative one.
        expected_coresubset = None
        if expected_coreset is None:
            expected_coresubset, _ = solver.reduce(dataset)
        elif isinstance(expected_coreset, Coresubset):
            expected_coresubset = expected_coreset
        if request.param == "well-sized":
            indices = Data(-1 * jnp.ones(coreset_size, jnp.int32))
        elif request.param == "under-sized":
            indices = Data(-1 * jnp.ones(coreset_size - 1, jnp.int32))
        elif request.param == "over-sized":
            indices = Data(-1 * jnp.ones(coreset_size + 1, jnp.int32))
        elif request.param == "random-unweighted":
            random_indices = jr.choice(indices_key, len(dataset), (coreset_size,))
            indices = Data(random_indices)
            expected_coresubset = None
        elif request.param == "random-weighted":
            random_indices = jr.choice(indices_key, len(dataset), (coreset_size,))
            random_weights = jr.uniform(weights_key, (coreset_size,))
            indices = Data(random_indices, random_weights)
            expected_coresubset = None
        else:
            raise ValueError("Invalid fixture parametrization")
        initial_coresubset = Coresubset(indices, dataset)
        return _RefineProblem(initial_coresubset, solver, expected_coresubset)

    def test_greedy_kernel_inducing_point_state(
        self, reduce_problem: _ReduceProblem
    ) -> None:
        """Check that the cached herding state is as expected."""
        dataset, solver, _ = reduce_problem
        solver = cast(GreedyKernelPoints, solver)
        _, state = solver.reduce(dataset)

        x = dataset.data
        expected_state = GreedyKernelPointsState(
            jnp.pad(solver.feature_kernel.compute(x, x), [(0, 1)], mode="constant")
        )
        assert eqx.tree_equal(state, expected_state)

    def test_non_uniqueness(self, reduce_problem: _ReduceProblem) -> None:
        """Test that setting `unique` to be false produces no errors."""
        dataset, _, _ = reduce_problem
        solver = GreedyKernelPoints(
            random_key=self.random_key,
            coreset_size=10,
            feature_kernel=SquaredExponentialKernel(),
            unique=False,
        )
        solver.reduce(dataset)

    def test_approximate_inverse(self, reduce_problem: _ReduceProblem) -> None:
        """Test that using an approximate least squares solver produces no errors."""
        dataset, _, _ = reduce_problem
        solver = GreedyKernelPoints(
            random_key=self.random_key,
            coreset_size=10,
            feature_kernel=SquaredExponentialKernel(),
            least_squares_solver=RandomisedEigendecompositionSolver(self.random_key),
        )
        solver.reduce(dataset)

    def test_batch_size_not_none(self, reduce_problem: _ReduceProblem) -> None:
        """Test that setting a not `None` `batch_size` produces no errors."""
        dataset, _, _ = reduce_problem
        solver = GreedyKernelPoints(
            random_key=self.random_key,
            coreset_size=10,
            feature_kernel=SquaredExponentialKernel(),
            batch_size=10,
        )
        solver.reduce(dataset)


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

            def get_arrays(self) -> tuple[Union[np.ndarray, None], ...]:
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

    def test_map_reduce_diverse_selection(self):
        """Check if MapReduce returns indices from multiple partitions."""
        dataset_size = 40
        data_dim = 5
        coreset_size = 6
        leaf_size = 12

        key = jr.PRNGKey(0)
        dataset = jr.normal(key, shape=(dataset_size, data_dim))

        kernel = SquaredExponentialKernel()
        base_solver = KernelHerding(coreset_size=coreset_size, kernel=kernel)

        solver = MapReduce(base_solver=base_solver, leaf_size=leaf_size)
        coreset, _ = solver.reduce(Data(dataset))
        selected_indices = coreset.nodes.data

        assert jnp.any(
            selected_indices >= coreset_size
        ), "MapReduce should select points beyond the first few"

        # Check if there are indices from different partitions
        partitions_represented = jnp.unique(selected_indices // leaf_size)
        assert (
            len(partitions_represented) > 1
        ), "MapReduce should select points from multiple partitions"

    def test_map_reduce_analytic(self):
        r"""
        Test ``MapReduce`` on an analytical example, enforcing a unique coreset.

        In this example, we start with the original dataset
        :math:`[10, 20, 30, 210, 40, 60, 180, 90, 150, 70, 120,
                    200, 50, 140, 80, 170, 100, 190, 110, 160, 130]`.

        Suppose we want a subset size of 3, and we want maximum leaf size of 6.

        We can see that we have a dataset of size 21. The partitioning scheme
        only allows for :math:`n` partitions where :math:`n` is a power of 2.
        Therefore, we can partition into:

        1. 1 partition of size 21
        2. 2 partitions of size :math:`\lceil 10.5 \rceil = 11` each (with one padded 0)
        3. 4 partitions of size :math:`\lceil 5.25 \rceil = 6` each (with 3 padded 0's)
        4. 8 partitions of size :math:`\lceil 2.625 \rceil = 3` each (with 3 padded 0's)

        Since we set the maximum leaf size :math:`m = 6`, we choose the largest
        partition size that is less than or equal to 6. Thus, we have 4 partitions
        each of size 6.

        This results in the following 4 partitions (see how
         data is in ascending order):

        1. :math:`[0, 0, 0, 10, 20, 30]`
        2. :math:`[40, 50, 60, 70, 80, 90]`
        3. :math:`[100, 110, 120, 130, 140, 150]`
        4. :math:`[160, 170, 180, 190, 200, 210]`

        Now we want to reduce each partition with our ``interleaved_base_solver``
        which is designed to choose first, last, second, second-last, third,
        third-last elements etc. until the coreset of correct size is formed.
        Hence, we obtain:

        1. :math:`[0, 30, 0]`
        2. :math:`[40, 90, 50]`
        3. :math:`[100, 150, 110]`
        4. :math:`[160, 210, 170]`

        Concatenating we obtain
        :math:`[0, 30, 0, 40, 90, 50, 100, 150, 110, 160, 210, 170]`.
        We repeat the process, checking how many partitions we want to divide this
        intermediate dataset (of size 12) into. Recall, this number of partitions must
        be a power of 2. Our options are:

        1. 1 partition of size 12
        2. 2 partitions of size 6
        3. 4 partitions of size 3
        4. 8 partitions of size 1.5 (rounded up to 2)

        Given our maximum leaf size :math:`m = 6`, we choose the largest partition size
        that is less than or equal to 6. Therefore, we select 2 partitions of size 6.
        This time no padding is necessary. The two partitions resulting from this step
        are (note that it is again in ascending order):

        1. :math:`[0, 0, 30, 40, 50, 90]`
        2. :math:`[100, 110, 150, 160, 170, 210]`

        Applying our ``interleaved_base_solver`` with `coreset_size` 3 on
        each partition, we obtain:

        1. :math:`[0, 90, 0]`
        2. :math:`[100, 210, 110]`

        Now, we concatenate the two subsets and repeat the process to
        obtain only one partition:

        1. Concatenated subset: :math:`[0, 90, 0, 100, 210, 110]`

        Note that the size of the dataset is 6,
        therefore, no more partitioning is necessary.

        Applying ``interleaved_base_solver`` one last time we obtain the final coreset:
            :math:`[0, 110, 90]`.
        """
        interleaved_base_solver = MagicMock(_ExplicitPaddingInvariantSolver)
        interleaved_base_solver.coreset_size = 3

        def interleaved_mock_reduce(
            dataset: Data, solver_state: None = None
        ) -> tuple[Coreset[Data], None]:
            half_size = interleaved_base_solver.coreset_size // 2
            indices = jnp.arange(interleaved_base_solver.coreset_size)
            forward_indices = indices[:half_size]
            backward_indices = -(indices[:half_size] + 1)
            interleaved_indices = jnp.stack(
                [forward_indices, backward_indices], axis=1
            ).ravel()

            if interleaved_base_solver.coreset_size % 2 != 0:
                interleaved_indices = jnp.append(interleaved_indices, half_size)
            return Coreset(dataset[interleaved_indices], dataset), solver_state

        interleaved_base_solver.reduce = interleaved_mock_reduce

        original_data = Data(jnp.array([
            10, 20, 30, 210, 40, 60, 180, 90, 150, 70, 120,
            200, 50, 140, 80, 170, 100, 190, 110, 160, 130
        ]))
        expected_coreset_data = Data(jnp.array([0, 110, 90]))

        coreset, _ = MapReduce(base_solver=interleaved_base_solver, leaf_size=6).reduce(
            original_data
        )
        assert eqx.tree_equal(coreset.coreset.data == expected_coreset_data.data)


class TestCaratheodoryRecombination(RecombinationSolverTest):
    """Tests for :class:`coreax.solvers.recombination.CaratheodoryRecombination`."""

    @override
    @pytest.fixture(scope="class")
    def solver_factory(self) -> Union[Solver, jtu.Partial]:
        return jtu.Partial(CaratheodoryRecombination, test_functions=None, rcond=None)


class TestTreeRecombination(RecombinationSolverTest):
    """Tests for :class:`coreax.solvers.recombination.TreeRecombination`."""

    @override
    @pytest.fixture(scope="class")
    def solver_factory(self) -> Union[Solver, jtu.Partial]:
        return jtu.Partial(
            TreeRecombination, test_functions=None, rcond=None, tree_reduction_factor=3
        )
