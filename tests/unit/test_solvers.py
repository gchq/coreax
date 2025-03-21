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

import copy
import re
from abc import abstractmethod
from collections.abc import Callable
from contextlib import (
    AbstractContextManager,
    nullcontext as does_not_raise,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
    cast,
)
from unittest.mock import MagicMock, patch

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest
from typing_extensions import override

from coreax.coreset import AbstractCoreset, Coresubset, PseudoCoreset
from coreax.data import Data, SupervisedData
from coreax.kernels import (
    PCIMQKernel,
    ScalarValuedKernel,
    SquaredExponentialKernel,
    SteinKernel,
)
from coreax.least_squares import (
    MinimalEuclideanNormSolver,
    RandomisedEigendecompositionSolver,
)
from coreax.solvers import (
    CaratheodoryRecombination,
    CompressPlusPlus,
    GreedyKernelPoints,
    GreedyKernelPointsState,
    HerdingState,
    KernelHerding,
    KernelThinning,
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
from coreax.solvers.coresubset import (
    _greedy_kernel_points_loss,  # noqa: PLC2701
)
from coreax.util import KeyArrayLike, tree_zero_pad_leading_axis

_Data = TypeVar("_Data", Data, SupervisedData)
_Solver = TypeVar("_Solver", bound=Solver)
_RefinementSolver = TypeVar("_RefinementSolver", bound=RefinementSolver)

if TYPE_CHECKING:
    # In Python 3.9-3.10, this raises
    # `TypeError: Multiple inheritance with NamedTuple is not supported`.
    # Thus, we have to do the actual full typing here, and a non-generic one
    # below to be used at runtime.
    class _ReduceProblem(NamedTuple, Generic[_Data, _Solver]):
        dataset: _Data
        solver: _Solver
        expected_coreset: Optional[AbstractCoreset] = None

    class _RefineProblem(NamedTuple, Generic[_RefinementSolver]):
        initial_coresubset: Coresubset
        solver: _RefinementSolver
        expected_coresubset: Optional[Coresubset] = None
else:
    # This is the implementation that's used at runtime.
    class _ReduceProblem(NamedTuple):
        dataset: _Data
        solver: _Solver
        expected_coreset: Optional[AbstractCoreset] = None

    class _RefineProblem(NamedTuple):
        initial_coresubset: Coresubset
        solver: _RefinementSolver
        expected_coresubset: Optional[Coresubset] = None


class SolverTest:
    """Base tests for all children of :class:`coreax.solvers.Solver`."""

    random_key: KeyArrayLike = jr.key(2024)
    shape: tuple[int, int] = (128, 10)

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
        self, coreset: AbstractCoreset, problem: Union[_RefineProblem, _ReduceProblem]
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
        assert isinstance(dataset, Data)
        assert eqx.tree_equal(coreset.pre_coreset_data, dataset)
        if expected_coreset is not None:
            assert isinstance(coreset, type(expected_coreset))
            assert eqx.tree_equal(coreset, expected_coreset)
        if isinstance(coreset, Coresubset):
            membership = jtu.tree_map(jnp.isin, coreset.points.data, dataset.data)
            all_membership = jtu.tree_map(jnp.all, membership)
            assert jtu.tree_all(all_membership)
        if isinstance(solver, PaddingInvariantSolver):
            padded_dataset = tree_zero_pad_leading_axis(dataset, len(dataset))
            if isinstance(problem, _RefineProblem):
                assert isinstance(solver, RefinementSolver)
                padded_initial_coreset = eqx.tree_at(
                    lambda x: x.pre_coreset_data,
                    problem.initial_coresubset,
                    padded_dataset,
                )
                coreset_from_padded, _ = solver.refine(padded_initial_coreset)
            else:
                coreset_from_padded, _ = solver.reduce(padded_dataset)
            assert eqx.tree_equal(coreset_from_padded.points, coreset.points)

    @pytest.mark.parametrize(
        "use_cached_state", (False, True), ids=["not_cached", "cached"]
    )
    def test_reduce(
        self,
        jit_variant: Callable[[Callable], Callable],
        reduce_problem: _ReduceProblem,
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
        params=[
            "random",
            "partial-null",
            "null",
            "full_rank",
            "rank_deficient",
            "excessive_test_functions",
        ],
        scope="class",
    )
    def reduce_problem(  # noqa: C901 complex-structure
        self, request: pytest.FixtureRequest, solver_factory: jtu.Partial
    ) -> _ReduceProblem:
        node_key, weight_key, rng_key = jr.split(self.random_key, num=3)
        nodes = jr.uniform(node_key, self.shape)
        weights = jr.uniform(weight_key, (self.shape[0],))
        expected_coreset = None
        if request.param == "random":
            # Random dataset with default test-functions.
            test_functions = None
        elif request.param == "partial-null":
            # Same as 'random' but with some dataset entries given zero weight.
            zero_weights = jr.choice(rng_key, self.shape[0], (self.shape[0] // 2,))
            weights = weights.at[zero_weights].set(0)
            test_functions = None
        elif request.param == "null":
            # Same as 'random' but with test-functions mapping to the zero vector.
            def test_functions_impl(x):
                return jnp.zeros(x.shape)

            test_functions = test_functions_impl
        elif request.param == "full_rank":
            # Same as 'random' but with all test-functions linearly-independent.
            def test_functions_impl(x):
                norm_x = jnp.linalg.norm(x)
                return jnp.array([norm_x, norm_x**2, norm_x**3])

            test_functions = test_functions_impl
        elif request.param == "rank_deficient":
            # Same as 'full_rank' but with some test-functions linearly-dependent.
            def test_functions_impl(x):
                norm_x = jnp.linalg.norm(x)
                return jnp.array([norm_x, 2 * norm_x, 2 + norm_x])

            test_functions = test_functions_impl
        elif request.param == "excessive_test_functions":
            # Same as 'random' but with more test-functions than dataset entries.
            def test_functions_impl(x):
                del x
                return jnp.zeros((len(nodes) + 1,))

            test_functions = test_functions_impl
        else:
            raise ValueError("Invalid fixture parametrization")
        solver_factory.keywords["test_functions"] = test_functions
        solver = solver_factory()
        return _ReduceProblem(Data(nodes, weights), solver, expected_coreset)

    @override
    def check_solution_invariants(
        self, coreset: AbstractCoreset, problem: Union[_RefineProblem, _ReduceProblem]
    ) -> None:
        r"""
        Check that a coreset obeys certain expected invariant properties.

        In addition to the standard checks in the parent class we also check:
        1. Check 'sum(coreset.weights)' is one.
        2. Check 'len(coreset)' is less than or equal to the upper bound `m`.
        3. Check 'len(coreset[idx]) where idx = jnp.nonzero(coreset.weights)' is less
            than or equal to the rank, :math:`m^\prime`, of the pushed forward nodes.
        4. Check the push-forward of the coreset preserves the "centre-of-mass" (CoM) of
            the pushed-forward dataset (with implicit and explicit zero weight removal).
        5. Check the default value of 'test_functions' is the identity map.
        """
        super().check_solution_invariants(coreset, problem)
        dataset, solver, _ = problem
        assert isinstance(dataset, Data)
        coreset_nodes, coreset_weights = coreset.points.data, coreset.points.weights
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
        max_rank = min(len(dataset.data), augmented_pushed_forward_nodes.shape[-1])
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
    # Ignore pylint here; this is a perfectly valid override (because of **kwargs).
    # pylint: disable-next=arguments-differ
    def test_reduce(
        self,
        jit_variant: Callable[[Callable], Callable],
        reduce_problem: _ReduceProblem,
        use_cached_state: bool,
        *,
        recombination_mode: Literal["implicit-explicit", "implicit", "explicit"],
        context: AbstractContextManager,
        **kwargs: Any,
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
        del kwargs
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
        initial_coresubset = Coresubset.build(indices, dataset)
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
        self, coreset: AbstractCoreset, problem: Union[_RefineProblem, _ReduceProblem]
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
        solver_factory = copy.deepcopy(solver_factory)
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
    def solver_factory(self, request: pytest.FixtureRequest) -> jtu.Partial:
        del request
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
            expected_coreset = Coresubset.build(jnp.array([1, 2]), Data(dataset))
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

    def test_kernel_herding_analytic_unique(self) -> None:
        # pylint: disable=line-too-long
        r"""
        Test kernel herding on an analytical example, enforcing a unique coreset.

        In this example, we have data of:

        .. math::
            x = \begin{pmatrix}
                0.3 & 0.25 \\
                0.4 & 0.2 \\
                0.5 & 0.125
            \end{pmatrix}

        and choose a ``length_scale`` of :math:`\frac{1}{\sqrt{2}}` to simplify
        computations with the ``SquaredExponentialKernel``, in particular it becomes:

        .. math::
            k(x, y) = e^{-||x - y||^2}.

        Kernel herding should do as follows:
            - Compute the Gramian row mean, that is for each data-point :math:`x` and
              all other data-points :math:`x'`, :math:`\frac{1}{N} \sum_{x'} k(x, x')`
              where we have :math:`N` data-points in total.
            - Select the first coreset point :math:`x_{1}` as the data-point where the
              Gramian row mean is highest.
            - Compute all future coreset points as
              :math:`x_{T+1} = \arg\max_{x} \left( \mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) \right)`
              where we currently have :math:`T` points in the coreset.

        We ask for a coreset of size 2 in this example. With an empty coreset, we first
        compute :math:`\mathbb{E}[k(x, x')]` as:

        .. math::
            \mathbb{E}[k(x, x')] = \frac{1}{3} \cdot \begin{pmatrix}
                k([0.3, 0.25]', [0.3, 0.25]') + k([0.3, 0.25]', [0.4, 0.2]') + k([0.3, 0.25]', [0.5, 0.125]') \\
                k([0.4, 0.2]', [0.3, 0.25]') + k([0.4, 0.2]', [0.4, 0.2]') + k([0.4, 0.2]', [0.5, 0.125]') \\
                k([0.5, 0.125]', [0.3, 0.25]') + k([0.5, 0.125]', [0.4, 0.2]') + k([0.5, 0.125]', [0.5, 0.125]')
            \end{pmatrix}

        resulting in:

        .. math::
            \mathbb{E}[k(x, x')] = \begin{pmatrix}
                0.9778238600172561 \\
                0.9906914124997632 \\
                0.9767967388544317
            \end{pmatrix}

        The largest value in this array is 0.9906914124997632, so we expect the first
        coreset point to be [0.4, 0.2], that is the data-point at index 1 in the
        dataset. At this point we have ``coreset_indices`` as [1, ?].

        We then compute the penalty update term
        :math:`\frac{1}{T+1}\sum_{t=1}^T k(x, x_t)` with :math:`T = 1`:

        .. math::
            \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \frac{1}{2} \cdot \begin{pmatrix}
                k([0.3, 0.25]', [0.4, 0.2]') \\
                k([0.4, 0.2]', [0.4, 0.2]') \\
                k([0.5, 0.125]', [0.4, 0.2]')
            \end{pmatrix}

        which evaluates to:

        .. math::
            \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \begin{pmatrix}
                0.4937889002469407 \\
                0.5 \\
                0.4922482185027042
            \end{pmatrix}

        We now select the data-point that maximises
        :math:`\mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T k(x, x_t)`, which
        evaluates to:

        .. math::
            \mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \begin{pmatrix}
                0.9778238600172561 - 0.4937889002469407 \\
                0.9906914124997632 - 0.5 \\
                0.9767967388544317 - 0.4922482185027042
            \end{pmatrix}

        giving a final result of:

        .. math::
            \mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \begin{pmatrix}
                0.4840349597703154 \\
                0.4906914124997632 \\
                0.4845485203517275
            \end{pmatrix}

        The largest value in this array is at index 1, which would be to again choose
        the point [0.4, 0.2] for the coreset. However, in this example we enforce the
        coreset to be unique, that is not to select the same data-point twice, which
        means we should take the next highest value in the above result to include in
        our coreset. This happens to be 0.4845485203517275, the data-point at index 2.
        This means our final ``coreset_indices`` should be [1, 2].

        Finally, the solver state tracks variables we need not compute repeatedly. In
        the case of kernel herding, we don't need to recompute
        :math:`\mathbb{E}[k(x, x')]` at every single step - so the solver state from the
        coreset reduce method should be set to:

        .. math::
            \mathbb{E}[k(x, x')] = \begin{pmatrix}
                0.9778238600172561 \\
                0.9906914124997632 \\
                0.9767967388544317
            \end{pmatrix}
        """  # noqa: E501
        # pylint: enable=line-too-long
        # Setup example data - note we have specifically selected points that are very
        # close to manipulate the penalty applied for nearby points, and hence enable
        # a check of unique points using the same data.
        coreset_size = 2
        length_scale = 1.0 / jnp.sqrt(2)
        x = jnp.array(
            [
                [0.3, 0.25],
                [0.4, 0.2],
                [0.5, 0.125],
            ]
        )

        # Define a kernel
        kernel = SquaredExponentialKernel(length_scale=length_scale)

        # Generate the coreset
        data = Data(x)
        solver = KernelHerding(coreset_size=coreset_size, kernel=kernel, unique=True)
        coreset, solver_state = solver.reduce(data)

        # Define the expected outputs, following the arguments in the docstring
        expected_coreset_indices = jnp.array([1, 2])
        expected_gramian_row_mean = jnp.array(
            [
                0.9778238600172561,
                0.9906914124997632,
                0.9767967388544317,
            ]
        )

        # Check output matches expected
        np.testing.assert_array_equal(
            coreset.unweighted_indices, expected_coreset_indices
        )
        np.testing.assert_array_equal(
            coreset.points.data, data.data[expected_coreset_indices]
        )
        np.testing.assert_array_almost_equal(
            solver_state.gramian_row_mean, expected_gramian_row_mean
        )

    def test_kernel_herding_analytic_not_unique(self) -> None:
        # pylint: disable=line-too-long
        r"""
        Test kernel herding on an analytical example, enforcing a non-unique coreset.

        In this example, we have data of:

        .. math::
            x = \begin{pmatrix}
                0.3 & 0.25 \\
                0.4 & 0.2 \\
                0.5 & 0.125
            \end{pmatrix}

        and choose a ``length_scale`` of :math:`\frac{1}{\sqrt{2}}` to simplify
        computations with the ``SquaredExponentialKernel``, in particular it becomes:

        .. math::
            k(x, y) = e^{-||x - y||^2}.

        Kernel herding should do as follows:
            - Compute the Gramian row mean, that is for each data-point :math:`x` and
              all other data-points :math:`x'`, :math:`\frac{1}{N} \sum_{x'} k(x, x')`
              where we have :math:`N` data-points in total.
            - Select the first coreset point :math:`x_{1}` as the data-point where the
              Gramian row mean is highest.
            - Compute all future coreset points as
              :math:`x_{T+1} = \arg\max_{x} \left( \mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) \right)`
              where we currently have :math:`T` points in the coreset.

        We ask for a coreset of size 2 in this example. With an empty coreset, we first
        compute :math:`\mathbb{E}[k(x, x')]` as:

        .. math::
            \mathbb{E}[k(x, x')] = \frac{1}{3} \cdot \begin{pmatrix}
                k([0.3, 0.25]', [0.3, 0.25]') + k([0.3, 0.25]', [0.4, 0.2]') + k([0.3, 0.25]', [0.5, 0.125]') \\
                k([0.4, 0.2]', [0.3, 0.25]') + k([0.4, 0.2]', [0.4, 0.2]') + k([0.4, 0.2]', [0.5, 0.125]') \\
                k([0.5, 0.125]', [0.3, 0.25]') + k([0.5, 0.125]', [0.4, 0.2]') + k([0.5, 0.125]', [0.5, 0.125]')
            \end{pmatrix}

        resulting in:

        .. math::
            \mathbb{E}[k(x, x')] = \begin{pmatrix}
                0.9778238600172561 \\
                0.9906914124997632 \\
                0.9767967388544317
            \end{pmatrix}

        The largest value in this array is 0.9906914124997632, so we expect the first
        coreset point to be [0.4, 0.2], that is the data-point at index 1 in the
        dataset. At this point we have ``coreset_indices`` as [1, ?].

        We then compute the penalty update term
        :math:`\frac{1}{T+1}\sum_{t=1}^T k(x, x_t)` with :math:`T = 1`:

        .. math::
            \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \frac{1}{2} \cdot \begin{pmatrix}
                k([0.3, 0.25]', [0.4, 0.2]') \\
                k([0.4, 0.2]', [0.4, 0.2]') \\
                k([0.5, 0.125]', [0.4, 0.2]')
            \end{pmatrix}

        which evaluates to:

        .. math::
            \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \begin{pmatrix}
                0.4937889002469407 \\
                0.5 \\
                0.4922482185027042
            \end{pmatrix}

        We now select the data-point that maximises
        :math:`\mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T k(x, x_t)`, which
        evaluates to:

        .. math::
            \mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \begin{pmatrix}
                0.9778238600172561 - 0.4937889002469407 \\
                0.9906914124997632 - 0.5 \\
                0.9767967388544317 - 0.4922482185027042
            \end{pmatrix}

        giving a final result of:

        .. math::
            \mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \begin{pmatrix}
                0.4840349597703154 \\
                0.4906914124997632 \\
                0.4845485203517275
            \end{pmatrix}

        The largest value in this array is at index 1, which would be to again choose
        the point [0.4, 0.2] for the coreset. Since we don't enforce the coreset points
        selected to be unique here, our final ``coreset_indices`` should be [1, 1].

        Finally, the solver state tracks variables we need not compute repeatedly. In
        the case of kernel herding, we don't need to recompute
        :math:`\mathbb{E}[k(x, x')]` at every single step - so the solver state from the
        coreset reduce method should be set to:

        .. math::
            \mathbb{E}[k(x, x')] = \begin{pmatrix}
                0.9778238600172561 \\
                0.9906914124997632 \\
                0.9767967388544317
            \end{pmatrix}
        """  # noqa: E501
        # pylint: enable=line-too-long
        # Setup example data - note we have specifically selected points that are very
        # close to manipulate the penalty applied for nearby points, and hence enable
        # a check of unique points using the same data.
        coreset_size = 2
        length_scale = 1.0 / jnp.sqrt(2)
        x = jnp.array(
            [
                [0.3, 0.25],
                [0.4, 0.2],
                [0.5, 0.125],
            ]
        )

        # Define a kernel
        kernel = SquaredExponentialKernel(length_scale=length_scale)

        # Generate the coreset
        data = Data(x)
        solver = KernelHerding(coreset_size=coreset_size, kernel=kernel, unique=False)
        coreset, solver_state = solver.reduce(data)

        # Define the expected outputs, following the arguments in the docstring
        expected_coreset_indices = jnp.array([1, 1])
        expected_gramian_row_mean = jnp.array(
            [
                0.9778238600172561,
                0.9906914124997632,
                0.9767967388544317,
            ]
        )

        # Check output matches expected
        np.testing.assert_array_equal(
            coreset.unweighted_indices, expected_coreset_indices
        )
        np.testing.assert_array_equal(
            coreset.points.data, data.data[expected_coreset_indices]
        )
        np.testing.assert_array_almost_equal(
            solver_state.gramian_row_mean, expected_gramian_row_mean
        )

    def test_kernel_herding_analytic_unique_weighted_data(self) -> None:
        # pylint: disable=line-too-long
        r"""
        Test kernel herding on a weighted analytical example, with a unique coreset.

        In this example, we have data of:

        .. math::
            x = \begin{pmatrix}
                0.3 & 0.25 \\
                0.4 & 0.2 \\
                0.5 & 0.125
            \end{pmatrix}

        with weights:

        .. math::
            w = \begin{pmatrix}
                0.8 \\
                0.1 \\
                0.1
            \end{pmatrix}

        and choose a ``length_scale`` of :math:`\frac{1}{\sqrt{2}}` to simplify
        computations with the ``SquaredExponentialKernel``, in particular it becomes:

        .. math::
            k(x, y) = e^{-||x - y||^2}.

        Kernel herding should do as follows:
            - Compute the Gramian row mean, that is for each data-point :math:`x` and
              all other data-points :math:`x'`, :math:`\sum_{x'} w_{x'} \cdot k(x, x')`
              where we sum over all :math:`N` data-points.
            - Select the first coreset point :math:`x_{1}` as the data-point where the
              Gramian row mean is highest.
            - Compute all future coreset points as
              :math:`x_{T+1} = \arg\max_{x} \left( \mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T w_{x_t} \cdot k(x, x_t) \right)`
              where we currently have :math:`T` points in the coreset.

        We ask for a coreset of size 2 in this example. With an empty coreset, we first
        compute :math:`\mathbb{E}[k(x, x')]` as:

        .. math::
            \mathbb{E}[k(x, x')] = \begin{pmatrix}
                0.8 \cdot k([0.3, 0.25]', [0.3, 0.25]') + 0.1 \cdot k([0.3, 0.25]', [0.4, 0.2]') + 0.1 \cdot k([0.3, 0.25]', [0.5, 0.125]') \\
                0.8 \cdot  k([0.4, 0.2]', [0.3, 0.25]') + 0.1 \cdot k([0.4, 0.2]', [0.4, 0.2]') + 0.1 \cdot k([0.4, 0.2]', [0.5, 0.125]') \\
                0.8 \cdot  k([0.5, 0.125]', [0.3, 0.25]') + 0.1 \cdot k([0.5, 0.125]', [0.4, 0.2]') + 0.1 \cdot k([0.5, 0.125]', [0.5, 0.125]')
            \end{pmatrix}

        resulting in:

        .. math::
            \mathbb{E}[k(x, x')] = \begin{pmatrix}
                0.9933471580051769 \\
                0.988511884095646 \\
                0.9551646673468503
            \end{pmatrix}

        The largest value in this array is 0.9933471580051769, so we expect the first
        coreset point to be [0.3  0.25], that is the data-point at index 0 in the
        dataset. At this point we have ``coreset_indices`` as [0, ?].

        We then compute the penalty update term
        :math:`\frac{1}{T+1}\sum_{t=1}^T k(x, x_t)` with :math:`T = 1`:

        .. math::
            \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \frac{1}{2} \cdot \begin{pmatrix}
                k([0.3, 0.25]', [0.3, 0.25]') \\
                k([0.4, 0.2]', [0.3, 0.25]') \\
                k([0.5, 0.125]', [0.3, 0.25]')
            \end{pmatrix}

        which evaluates to:

        .. math::
            \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \begin{pmatrix}
                0.5 \\
                0.4937889002469407 \\
                0.4729468897789434
            \end{pmatrix}

        We now select the data-point that maximises
        :math:`\mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T k(x, x_t)`,
        which evaluates to:

        .. math::
            \mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \begin{pmatrix}
                0.9933471580051769 - 0.5 \\
                0.988511884095646 - 0.4937889002469407 \\
                0.9551646673468503 - 0.4729468897789434
            \end{pmatrix}

        giving a final result of:

        .. math::
            \mathbb{E}[k(x, x')] - \frac{1}{T+1}\sum_{t=1}^T k(x, x_t) = \begin{pmatrix}
                0.4933471580051769 \\
                0.49472298384870533 \\
                0.48221777756790696
            \end{pmatrix}

        The largest value in this array is at index 1, which means we choose
        the point [0.4, 0.2] for the coreset. This means our final ``coreset_indices``
        should be [0, 1].

        Finally, the solver state tracks variables we need not compute repeatedly. In
        the case of kernel herding, we don't need to recompute
        :math:`\mathbb{E}[k(x, x')]` at every single step - so the solver state from the
        coreset reduce method should be set to:

        .. math::
            \mathbb{E}[k(x, x')] = \begin{pmatrix}
                0.9933471580051769 \\
                0.988511884095646 \\
                0.9551646673468503
            \end{pmatrix}
        """  # noqa: E501
        # pylint: enable=line-too-long
        # Setup example data - note we have specifically selected points that are very
        # close to manipulate the penalty applied for nearby points, and hence enable
        # a check of unique points using the same data.
        coreset_size = 2
        length_scale = 1.0 / jnp.sqrt(2)
        x = jnp.array(
            [
                [0.3, 0.25],
                [0.4, 0.2],
                [0.5, 0.125],
            ]
        )
        weights = jnp.array([0.8, 0.1, 0.1])

        # Define a kernel
        kernel = SquaredExponentialKernel(length_scale=length_scale)

        # Generate the coreset
        data = Data(data=x, weights=weights)
        solver = KernelHerding(coreset_size=coreset_size, kernel=kernel, unique=True)
        coreset, solver_state = solver.reduce(data)

        # Define the expected outputs, following the arguments in the docstring
        expected_coreset_indices = jnp.array([0, 1])
        expected_gramian_row_mean = jnp.array(
            [0.9933471580051769, 0.988511884095646, 0.9551646673468503]
        )

        # Check output matches expected
        np.testing.assert_array_equal(
            coreset.unweighted_indices, expected_coreset_indices
        )
        np.testing.assert_array_equal(
            coreset.points.data, data.data[expected_coreset_indices]
        )
        np.testing.assert_array_almost_equal(
            solver_state.gramian_row_mean, expected_gramian_row_mean
        )

    def test_kernel_herding_refine_analytic(self):
        """
        Test whether `KernelHerding.refine` works correctly on an existing coreset.

        The test case has been verified independently to minimise MMD at every
        iteration.
        """
        # Small testing dataset with a fixed seed
        generator = np.random.default_rng(97)
        x = jnp.asarray(generator.uniform(size=(100, 2)))
        data = Data(x)

        # Initialise the solver using a simple kernel
        kernel = PCIMQKernel()
        herding_solver = KernelHerding(coreset_size=10, kernel=kernel, unique=True)

        # Run reduce and then refine the output
        herding_coreset, herding_state = herding_solver.reduce(data)
        herding_coreset_ref, _ = herding_solver.refine(herding_coreset, herding_state)

        # Check output matches expected
        expected_reduce_indices = jnp.array([94, 62, 54, 15, 85, 72, 1, 31, 32, 86])
        np.testing.assert_array_equal(
            herding_coreset.unweighted_indices, expected_reduce_indices
        )
        expected_refine_indices = jnp.array([97, 73, 10, 15, 85, 40, 1, 70, 32, 86])
        np.testing.assert_array_equal(
            herding_coreset_ref.unweighted_indices, expected_refine_indices
        )

    @pytest.mark.parametrize("reduce_problem", ["random"], indirect=True)
    def test_kernel_herding_probabilistic(self, reduce_problem: _ReduceProblem):
        """
        Test the probabilistic version of Kernel Herding.
        """
        # Set up solver and problem
        dataset, solver_base, _ = reduce_problem
        solver_prob = KernelHerding(
            coreset_size=solver_base.coreset_size,
            kernel=solver_base.kernel,
            probabilistic=True,
            temperature=0.1,
            random_key=jr.key(0),
        )

        # Run the standard and probabilistic KH solvers
        coreset_base, state_base = solver_base.reduce(dataset)
        coreset_prob, state_prob = solver_prob.reduce(dataset)

        # Test whether probabilistic KH outputs the same type of coreset
        assert isinstance(coreset_prob, type(coreset_base))
        # Test whether the state is the same in both versions
        np.testing.assert_array_equal(
            state_base.gramian_row_mean, state_prob.gramian_row_mean
        )

    @pytest.mark.parametrize("reduce_problem", ["random"], indirect=True)
    def test_reduce_iterative(self, reduce_problem: _ReduceProblem):
        """
        Test the iterative version of Kernel Herding.
        """
        dataset, solver_base, _ = reduce_problem
        num_iter = 4

        # Check that reduce_iterative() outputs the same coreset as when applying
        # refine num_iter times
        coreset_det, state = solver_base.reduce(dataset)
        for _ in range(num_iter - 1):
            coreset_det, _ = solver_base.refine(coreset_det, state)
        coreset_det_iter, _ = solver_base.reduce_iterative(
            dataset, state, num_iterations=num_iter
        )
        np.testing.assert_array_equal(
            coreset_det.unweighted_indices, coreset_det_iter.unweighted_indices
        )

        # Initialise the probabilistic solver
        solver_prob = KernelHerding(
            coreset_size=solver_base.coreset_size,
            kernel=solver_base.kernel,
            probabilistic=True,
            temperature=1.0,
            random_key=jr.key(0),
        )

        def deterministic_choice(*_, p, **__):
            """
            Return the index of largest element of p.

            If there is a tie, return the largest index.
            This is used to mimic random sampling, where we have a deterministic
            sampling approach.
            """
            # Find indices where the value equals the maximum
            is_max = p == p.max()
            # Convert boolean mask to integers and multiply by index
            # This way, we'll get the highest index where True appears
            indices = jnp.arange(p.shape[0])
            return jnp.where(is_max, indices, -1).max()

        # Mock the random choice function
        with patch("jax.random.choice", deterministic_choice):
            coreset_prob, _ = solver_prob.reduce(dataset)
            for i in range(num_iter - 1):
                new_solver = eqx.tree_at(
                    lambda x: x.random_key,
                    solver_prob,
                    jr.fold_in(solver_prob.random_key, i),
                )
                coreset_prob, _ = new_solver.refine(coreset_prob, state)

            coreset_prob_iter, _ = solver_base.reduce_iterative(
                dataset,
                num_iterations=num_iter,
                t_schedule=jnp.ones(num_iter) * solver_base.temperature,
            )

        # Check that reduce_iterative() outputs the same coreset as when applying
        # refine num_iter times
        np.testing.assert_array_equal(
            coreset_prob.unweighted_indices, coreset_prob_iter.unweighted_indices
        )


class TestRandomSample(ExplicitSizeSolverTest):
    """Test cases for :class:`coreax.solvers.coresubset.RandomSample`."""

    @override
    def check_solution_invariants(
        self, coreset: AbstractCoreset, problem: Union[_RefineProblem, _ReduceProblem]
    ) -> None:
        super().check_solution_invariants(coreset, problem)
        solver = cast(RandomSample, problem.solver)
        assert isinstance(coreset, Coresubset)
        if solver.unique:
            _, counts = jnp.unique(coreset.indices.data, return_counts=True)
            assert max(counts) <= 1

    @override
    @pytest.fixture(scope="class")
    def solver_factory(self, request: pytest.FixtureRequest) -> jtu.Partial:
        del request
        coreset_size = self.shape[0] // 10
        key = jr.fold_in(self.random_key, self.shape[0])
        return jtu.Partial(RandomSample, coreset_size=coreset_size, random_key=key)


class TestRPCholesky(ExplicitSizeSolverTest):
    """Test cases for :class:`coreax.solvers.coresubset.RPCholesky`."""

    @override
    def check_solution_invariants(
        self, coreset: AbstractCoreset, problem: Union[_RefineProblem, _ReduceProblem]
    ) -> None:
        """Check functionality of 'unique' in addition to the default checks."""
        super().check_solution_invariants(coreset, problem)
        solver = cast(RPCholesky, problem.solver)
        assert isinstance(coreset, Coresubset)
        if solver.unique:
            _, counts = jnp.unique(coreset.indices.data, return_counts=True)
            assert max(counts) <= 1

    @override
    @pytest.fixture(scope="class")
    def solver_factory(self, request) -> jtu.Partial:
        del request
        kernel = PCIMQKernel()
        coreset_size = self.shape[0] // 10
        return jtu.Partial(
            RPCholesky,
            coreset_size=coreset_size,
            random_key=self.random_key,
            kernel=kernel,
        )

    @override
    @pytest.mark.parametrize("use_cached_state", (False,))
    def test_reduce(
        self,
        jit_variant: Callable[[Callable], Callable],
        reduce_problem: _ReduceProblem,
        use_cached_state: bool,
        **kwargs: Any,
    ) -> None:
        """
        Check `coreax.solvers.RPCholesky.reduce` raises no errors.

        Note:
            This overrides `SolverTest.test_reduce` since that assumes that `reduce`
            is `solver_state` invariant, which is not true for RPCholesky.

        """
        super().test_reduce(jit_variant, reduce_problem, use_cached_state, **kwargs)

    def test_rpcholesky_state(self, reduce_problem: _ReduceProblem) -> None:
        """
        Check that the cached RPCholesky state is as expected.

        Here we assume that the RPCholesky state (the gramian diagonal) should change
        after running `reduce`.
        """
        dataset, solver, _ = reduce_problem
        solver = cast(RPCholesky, solver)
        _, state = solver.reduce(dataset)

        # Compute the diagonal of the initial Gram matrix
        x = dataset.data
        gramian_diagonal = jax.vmap(solver.kernel.compute_elementwise)(x, x)
        expected_state = RPCholeskyState(gramian_diagonal)

        assert not eqx.tree_equal(state, expected_state)

    def test_rpcholesky_analytic_unique(self):
        # pylint: disable=line-too-long
        r"""
        Analytical example with RPCholesky.

        Step-by-step usage of the RPCholesky algorithm (Algorithm 1 in
        :cite:`chen2023randomly`) on a small example with 3 data points in 2 dimensions and a
        coreset of size 2, i.e., :math:`N=3, m=2`.

        In this example, we have the following data:

        .. math::
            X = \begin{pmatrix}
                0.5 & 0.2 \\
                0.4 & 0.6 \\
                0.8 & 0.3
            \end{pmatrix}

        We choose a ``SquaredExponentialKernel`` with ``length_scale`` of
        :math:`\frac{1}{\sqrt{2}}`: for two points :math:`x, y \in X`, :math:`k(x, y) =
        e^{-||x - y||^2}`. We now compute the Gram matrix, :math:`A`, of the dataset
        :math:`X` with respect to the kernel :math:`k` as :math:`A_{ij} = k(X_i, X_j)`:

        .. math::
            A = \begin{pmatrix}
                1.0 & 0.84366477 & 0.90483737 \\
                0.84366477 & 1.0 & 0.7788007 \\
                0.90483737 & 0.7788007 & 1.0
            \end{pmatrix}

        Note that, in practice, we do not need to precompute the full Gram matrix, the algorithm
        only needs to evaluate the pivot column at each iteration.

        To apply the RPCholesky algorithm, we first initialise the *residual diagonal*
        :math:`d = \text{diag}(A)` and the *approximation matrix* :math:`F = \mathbf{0}_{N
        \times m}`, where :math:`N = 3, m = 2` in our case.

        We now build a coreset iteratively by applying the following steps at each iteration i:
            - Sample a datapoint index (called a pivot) proportional to :math:`d`
            - Compute/extract column :math:`g` corresponding to the pivot index from :math:`A`
            - Remove the overlap with previously selected columns from :math:`g`
            - Normalize the column and add it to the approximation matrix :math:`F`
            - Update the residual diagonal: :math:`d = d - |F[:,i]|^2`

        For the first iteration (i=0):

        1. We sample a pivot point proportional to their value on the diagonal. Since
        :math:`d` is initialised as :math:`(1, 1, 1)` in our case, all choices are equally
        likely, so let us suppose we choose the pivot with index = 2.

        2. We now compute :math:`g`, the column at index 2, as:

        .. math::
            g = \begin{pmatrix}
            0.90483737 \\
            0.7788007 \\
            1.0
            \end{pmatrix}

        3. Remove overlap with previously chosen columns (not needed on the first iteration).

        4. Update the approximation matrix:

        .. math::
            F[:, 0] = g / \sqrt{(g[2])} = \begin{pmatrix}
            0.90483737 \\
            0.7788007 \\
            1.0
            \end{pmatrix}

        5. Update the residual diagonal:

        .. math::
            d = d - |F[:,0]|^2 = \begin{pmatrix}
            0.18126933 \\
            0.39346947 \\
            0
            \end{pmatrix}

        For the second iteration (i=1):

        1. We again sample a pivot point proportional to their value on the updated residual
        diagonal, :math:`d`. Let's suppose we choose the most likely pivot here (index=1).

        2. We now compute g, the column at index 1, as:

        .. math::
            g = \begin{pmatrix}
            0.84366477 \\
            1.0 \\
            0.7788007
            \end{pmatrix}

        3. Remove overlap with previously chosen columns:

        .. math::
            g = g - F[:, 0] F[1, 0]^T = \begin{pmatrix}
            0.13897679 \\
            0.39346947 \\
            0
            \end{pmatrix}

        4. Update the approximation matrix:

        .. math::
            F[:, 1] = g / \sqrt{(g[1])} = \begin{pmatrix}
            0.22155766 \\
            0.62727145 \\
            0
            \end{pmatrix}

        5. Update the residual diagonal:

        .. math::
            d = d - |F[:,0]|^2 = \begin{pmatrix}
              0.13218154 \\
              0 \\
              0
            \end{pmatrix}

        After this iteration, the final state is:

        .. math::
            F = \begin{pmatrix}
            0.90483737 & 0.22155766 \\
            0.7788007 & 0.62727145 \\
            1.0 & 0
            \end{pmatrix}, \quad
            d = \begin{pmatrix}
            0.13218154 \\
            0 \\
            0
            \end{pmatrix}, \quad
            S = \{2, 1\} \, .

        This completes the coreset of size :math:`m = 2`. We can also use the :math:`F` to
        compute an approximation to the original Gram matrix:

        .. math::

            F \cdot F^T = \begin{pmatrix}
            0.86781846 & 0.84366477 & 0.90483737 \\
            0.84366477 & 1.0 & 0.7788007 \\
            0.90483737 & 0.7788007 & 1.0
            \end{pmatrix}

        Note that we have recovered the original matrix except for :math:`A_{00}`, which was not
        covered by any of the chosen pivots.
        """  # noqa: E501
        # pylint: enable=line-too-long

        # Setup example data
        coreset_size = 2
        x = jnp.array(
            [
                [0.5, 0.2],
                [0.4, 0.6],
                [0.8, 0.3],
            ]
        )

        # Define a kernel
        length_scale = 1.0 / jnp.sqrt(2)
        kernel = SquaredExponentialKernel(length_scale=length_scale)

        # Create a mock for the random choice function
        def deterministic_choice(*_, p, **__):
            """
            Return the index of largest element of p.

            If there is a tie, return the largest index.
            This is used to mimic random sampling, where we have a deterministic
            sampling approach.
            """
            # Find indices where the value equals the maximum
            is_max = p == p.max()
            # Convert boolean mask to integers and multiply by index
            # This way, we'll get the highest index where True appears
            indices = jnp.arange(p.shape[0])
            return jnp.where(is_max, indices, -1).max()

        # Generate the coreset
        data = Data(x)
        solver = RPCholesky(
            coreset_size=coreset_size,
            random_key=jax.random.PRNGKey(0),  # Fixed seed for reproducibility
            kernel=kernel,
            unique=True,
        )
        # Mock the random choice function
        with patch("jax.random.choice", deterministic_choice):
            coreset, solver_state = solver.reduce(data)

        # Independently computed gramian diagonal
        expected_gramian_diagonal = jnp.array([0.13218154, 0.0, 0.0])

        # Coreset indices forced by our mock choice function
        expected_coreset_indices = jnp.array([2, 1])

        # Check output matches expected
        np.testing.assert_array_equal(
            coreset.unweighted_indices, expected_coreset_indices
        )
        np.testing.assert_array_equal(
            coreset.points.data, data.data[expected_coreset_indices]
        )
        np.testing.assert_array_almost_equal(
            solver_state.gramian_diagonal, expected_gramian_diagonal
        )

    def test_rpcholesky_unique_points(self):
        """
        Test whether a coreset contains no duplicates when running RPCholesky.

        We use a relatively large dataset (N=1_000) and set `coreset_size = N` to
        make sure RPCholesky adds no duplicates to the coreset even after convergence.
        """
        shape = (1_000, 2)
        data = Data(jr.uniform(self.random_key, shape))
        kernel = PCIMQKernel()
        solver = RPCholesky(
            coreset_size=shape[0],
            random_key=jax.random.PRNGKey(0),
            kernel=kernel,
            unique=True,
        )

        rpc_coreset, _ = solver.reduce(data)

        coreset_indices = rpc_coreset.unweighted_indices
        assert len(coreset_indices) == len(np.unique(coreset_indices))


class TestSteinThinning(RefinementSolverTest, ExplicitSizeSolverTest):
    """Test cases for :class:`coreax.solvers.coresubset.SteinThinning`."""

    @override
    @pytest.fixture(scope="class")
    def solver_factory(self, request: pytest.FixtureRequest) -> jtu.Partial:
        del request
        kernel = PCIMQKernel()
        coreset_size = self.shape[0] // 10
        return jtu.Partial(SteinThinning, coreset_size=coreset_size, kernel=kernel)

    @pytest.mark.parametrize(
        "test_lambda",
        [
            None,
            0.1,
            10.0,
            2,
        ],
    )
    def test_regulariser_lambda(
        self, test_lambda: Optional[Union[float, int]], reduce_problem: _ReduceProblem
    ) -> None:
        """Basic checks for the regularisation parameter, lambda."""
        dataset, base_solver, _ = reduce_problem
        solver = SteinThinning(
            coreset_size=base_solver.coreset_size,
            kernel=base_solver.kernel,
            regulariser_lambda=test_lambda,
        )
        coreset, _ = solver.reduce(dataset)

        # None should be equivalent to coreset_size
        if test_lambda is None:
            equiv_value = 1.0 / solver.coreset_size
            solver_equiv = SteinThinning(
                coreset_size=base_solver.coreset_size,
                kernel=base_solver.kernel,
                regulariser_lambda=equiv_value,
            )
            coreset_equiv, _ = solver_equiv.reduce(dataset)

            np.testing.assert_array_equal(
                coreset.unweighted_indices, coreset_equiv.unweighted_indices
            )
            np.testing.assert_array_equal(
                coreset.points.data, coreset_equiv.points.data
            )
        # Check that int is cast to float
        elif isinstance(test_lambda, int):
            solver_float = SteinThinning(
                coreset_size=base_solver.coreset_size,
                kernel=base_solver.kernel,
                regulariser_lambda=float(test_lambda),
            )
            coreset_float, _ = solver_float.reduce(dataset)

            np.testing.assert_array_equal(
                coreset.unweighted_indices, coreset_float.unweighted_indices
            )
            np.testing.assert_array_equal(
                coreset.points.data, coreset_float.points.data
            )

    def test_stein_thinning_analytic_unique(self):
        # pylint: disable=line-too-long
        r"""
        Analytical example with SteinThinning.

        .. note::
            We only compute the first 3 iterations of the algorithm below for
            illustration purposes. The test uses `coreset_size` of 10 to make sure
            all points are picked in the correct order.

        Step-by-step usage of the Stein Thinning algorithm (:cite:`liu2016kernelized`,
        :cite:`benard2023kernel`) on a small example with 10 data points in 2 dimensions
        and selecting a 2 point coreset, i.e., :math:`N=10, m=2`.

        For Stein Thinning, we need to provide a kernel and a score function, :math:`\nabla
        \log p(x)`, where :math:`p` is the underlying distribution of the data. In practice,
        we can estimate this using methods such as kernel density estimation and sliced score
        matching :cite:`song2020ssm`.

        In this example we will use ``PCIMQ`` kernel with ``length_scale`` of
        :math:`\frac{1}{\sqrt{2}}`: for two points :math:`x, y \in X`, :math:`k(x, y) =
        \frac{1}{\sqrt{1+\| x - y \|^2}}`. We use the standard normal density :math:`p(x)
        \propto e^{-\|x\|^2/2}`, so the score function is :math:`s_p(x) = -x`.

        The first step of the algorithm is to convert the given kernel into a Stein kernel
        using the given score function. The Stein kernel between points :math:`x` and
        :math:`y` is given as:

        .. math::
            k_p(x,y) &= \langle \nabla_\mathbf{x}, \nabla_{\mathbf{y}} k(\mathbf{x},
            \mathbf{y}) \rangle + \langle s_p(\mathbf{x}), \nabla_{\mathbf{y}} k(\mathbf{x}, \mathbf{y}) \rangle + \langle s_p(\mathbf{y}), \nabla_\mathbf{x} k(\mathbf{x}, \mathbf{y}) \rangle + \langle s_p(\mathbf{x}), s_p(\mathbf{y}) \rangle k(\mathbf{x}, \mathbf{y}) \\
            & = -3\|x-y\|^2(1 + \|x-y\|^2)^{-5/2} + (d - \|x-y\|^2)(1 + \|x-y\|^2)
            ^{-3/2} + (x\cdot y)(1 + \|x-y\|^2)^{-1/2}

        Now the algorithm proceeds iteratively, selecting the next point greedily by
        minimising the regularised KSD metric:

        .. math::
            x_{t} = \arg\min_{x} \left( k_p(x, x)  + \Delta^+ \log p(x) -
                \lambda t \log p(x) + 2\sum_{j=1}^{t-1} k_p(x, x_j) \right)

        Note that the Laplacian regularisation term (:math:`\Delta^+ \log p(x)`) vanishes for
        the given score function. Hence, using :math:`k_p(x,y)` and :math:`p(x)` given above,
        we have:

        .. math::
            k_p(x, x) &= d + \|x\|^2 \\
            \Delta^+ \log p(x) &= 0 \\
            - \lambda t \log p(x) &= \frac{\lambda t}{2} \|x\|^2 \\

        We can now simplify the metric at iteration :math:`t` in this example:

        .. math::
            d + \|x\|^2(1 + \lambda t/2) + 2\sum_{j=1}^{t-1} k_p(x_{j}, x)

        For now let's suppose no regularisation (:math:`\lambda = 0`) and selecting a unique
        point at each iteration. We now select points iteratively by minimizing the metric
        at each step.

        Let's suppose we have the following data:

        .. math::
           X = \begin{pmatrix}
               -0.1 & -0.1 \\
               -0.3 & -0.2 \\
               -0.2 & 0.6 \\
               0.8 & 0.2 \\
               -0.0 & 0.3 \\
               0.9 & -0.7 \\
               0.2 & -0.1 \\
               0.7 & -1.0 \\
               -0.4 & -0.4 \\
               0.0 & -0.3
           \end{pmatrix}

        First iteration (t=1):

        In the first iteration there are no previously selected points, hence we simply
        compute :math:`k_p(x, x) = d + \| x \|^2` for each point:

        .. math::
           \text{KSD}(X) = \begin{pmatrix}
               2.020 \\
               2.130 \\
               2.400 \\
               2.680 \\
               2.090 \\
               3.300 \\
               2.050 \\
               3.490 \\
               2.320 \\
               2.090
           \end{pmatrix}

        We select point at index 0 (assuming 0-indexing), :math:`(-0.1, -0.1)`, as it has the
        minimum score of 2.020.

        Second iteration (t=2):

        We now have to additionally compute :math:`k_p(x, X[0])` for each point since
        :math:`X[0]` was selected in the first iteration:

        .. math::
           \text{KSD}(X) = \begin{pmatrix}
               6.060 \\
               5.587 \\
               2.879 \\
               2.290 \\
               4.238 \\
               2.673 \\
               4.952 \\
               2.889 \\
               4.593 \\
               5.508
           \end{pmatrix}

        We now select the point at index 3, :math:`(0.8, 0.2)`, with the corresponding score of
        2.290.

        Third iteration (t=3):

        We now compute :math:`k_p(x, X[3])` for each point and update the scores:

        .. math::
           \text{KSD}(X) = \begin{pmatrix}
               5.670 \\
               4.618 \\
               2.339 \\
               7.650 \\
               4.490 \\
               3.393 \\
               5.894 \\
               2.710 \\
               3.377 \\
               5.187
           \end{pmatrix}

        We select the point at index 2, :math:`(-0.2, 0.6)`, with the corresponding score of
        2.339.

        Note that selecting a particular point changes the metric significantly at each
        iteration, emphasising that the algorithm attempts to move away from the already
        selected points and explore the rest of the space.

        The final selected points are :math:`\{0, 3, 2\}` with corresponding data points:

        .. math::
           X_{\text{coreset}} = \begin{pmatrix}
               -0.1 & -0.1 \\
               0.8 & 0.2 \\
               -0.2 & 0.6
           \end{pmatrix}
        """  # noqa: E501
        # pylint: enable=line-too-long

        # Setup example data
        coreset_size = 10
        x = jnp.array(
            [
                [-0.1, -0.1],
                [-0.3, -0.2],
                [-0.2, 0.6],
                [0.8, 0.2],
                [-0.0, 0.3],
                [0.9, -0.7],
                [0.2, -0.1],
                [0.7, -1.0],
                [-0.4, -0.4],
                [0.0, -0.3],
            ]
        )
        data = Data(x)

        # Initialise and run the SteinThinning solver
        stein_kernel = SteinKernel(
            base_kernel=PCIMQKernel(length_scale=1 / np.sqrt(2)),
            score_function=jnp.negative,
        )
        solver = SteinThinning(
            coreset_size=coreset_size,
            kernel=stein_kernel,
            unique=True,
            regularise=False,
        )
        coreset, _ = solver.reduce(data)

        # Expected selections based on our analytical calculations
        expected_coreset_indices = jnp.array([0, 3, 2, 7, 8, 5, 4, 1, 6, 9])

        # Check output matches expected
        np.testing.assert_array_equal(
            coreset.unweighted_indices, expected_coreset_indices
        )
        np.testing.assert_array_equal(
            coreset.points.data, data.data[expected_coreset_indices]
        )

    def test_stein_thinning_analytic_non_unique(self):
        """
        Analytical example for SteinThinning with repeating points.

        The example data is the same as in the unique case. If the solver is set to
        `unique=False`, some points will be repeated multiple times.
        """
        # Setup example data
        coreset_size = 10
        x = jnp.array(
            [
                [-0.1, -0.1],
                [-0.3, -0.2],
                [-0.2, 0.6],
                [0.8, 0.2],
                [-0.0, 0.3],
                [0.9, -0.7],
                [0.2, -0.1],
                [0.7, -1.0],
                [-0.4, -0.4],
                [0.0, -0.3],
            ]
        )
        data = Data(x)

        # Initialise and run the SteinThinning solver
        stein_kernel = SteinKernel(
            base_kernel=PCIMQKernel(length_scale=1 / np.sqrt(2)),
            score_function=jnp.negative,
        )
        solver = SteinThinning(
            coreset_size=coreset_size,
            kernel=stein_kernel,
            unique=False,
            regularise=False,
        )
        coreset, _ = solver.reduce(data)

        # Expected selections based on our analytical calculations
        expected_coreset_indices = jnp.array([0, 3, 2, 7, 8, 2, 5, 8, 3, 2])

        # Check output matches expected
        np.testing.assert_array_equal(
            coreset.unweighted_indices, expected_coreset_indices
        )
        np.testing.assert_array_equal(
            coreset.points.data, data.data[expected_coreset_indices]
        )

    def test_stein_thinning_analytic_reg(self):
        r"""
        Analytical example for SteinThinning with regularisation.

        The example data is the same as for the other analytic tests. When
        `regularise=True`, regularisation terms are added when computing the metric
        for each point. In particular, in our example, the additional term is
        :math:`-\lambda t \log p(x)` where p is the density and lambda is the
        regularisation parameter.

        Note that the `SteinThinning` solver uses a Gaussian KDE estimate of p,
        which we also use in our calculations to stay consistent.
        """
        # Setup example data
        coreset_size = 10
        x = jnp.array(
            [
                [-0.1, -0.1],
                [-0.3, -0.2],
                [-0.2, 0.6],
                [0.8, 0.2],
                [-0.0, 0.3],
                [0.9, -0.7],
                [0.2, -0.1],
                [0.7, -1.0],
                [-0.4, -0.4],
                [0.0, -0.3],
            ]
        )
        data = Data(x)

        # Initialise and run the SteinThinning solver
        stein_kernel = SteinKernel(
            base_kernel=PCIMQKernel(length_scale=1 / np.sqrt(2)),
            score_function=jnp.negative,
        )
        solver = SteinThinning(
            coreset_size=coreset_size,
            kernel=stein_kernel,
            unique=True,
            regularise=True,
            regulariser_lambda=1,
        )
        coreset, _ = solver.reduce(data)

        # Expected selections based on our analytical calculations
        expected_coreset_indices = jnp.array([0, 2, 5, 8, 6, 3, 1, 4, 7, 9])

        # Check output matches expected
        np.testing.assert_array_equal(
            coreset.unweighted_indices, expected_coreset_indices
        )
        np.testing.assert_array_equal(
            coreset.points.data, data.data[expected_coreset_indices]
        )


class TestGreedyKernelPoints(RefinementSolverTest, ExplicitSizeSolverTest):
    """Test cases for :class:`coreax.solvers.coresubset.GreedyKernelPoints`."""

    @override
    @pytest.fixture(scope="class")
    def solver_factory(self, request) -> jtu.Partial:
        del request
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
        self, coreset: AbstractCoreset, problem: Union[_RefineProblem, _ReduceProblem]
    ) -> None:
        """Check functionality of 'unique' in addition to the default checks."""
        super().check_solution_invariants(coreset, problem)
        solver = cast(GreedyKernelPoints, problem.solver)
        assert isinstance(coreset, Coresubset)
        if solver.unique:
            _, counts = jnp.unique(coreset.indices.data, return_counts=True)
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
        """Test that setting not `None` batch sizes produces no errors."""
        dataset, _, _ = reduce_problem
        solver = GreedyKernelPoints(
            random_key=self.random_key,
            coreset_size=10,
            feature_kernel=SquaredExponentialKernel(),
            candidate_batch_size=10,
            loss_batch_size=5,
        )
        solver.reduce(dataset)

    @pytest.mark.parametrize(
        "data_size, coreset_size, candidate_batch_size",
        [(100, 50, 10), (100, 50, 13), (100, 10, 50), (50, 10, 100), (50, 50, 10)],
        ids=[
            "multiple_candidate_batch_size_less_than_coreset_size_less_than_data_size",
            "non_multi_candidate_batch_size_less_than_coreset_size_less_than_data_size",
            "coreset_size_less_than_candidate_batch_size_less_than_data_size",
            "coreset_size_less_than_data_size_less_than_candidate_batch_size",
            "coreset_size_equal_to_data_size",
        ],
    )
    def test_candidate_batch_size(
        self,
        reduce_problem: _ReduceProblem,
        data_size: int,
        coreset_size: int,
        candidate_batch_size: int,
    ) -> None:
        """Test that various choices of `candidate_batch_size` throw no errors."""
        dataset, _, _ = reduce_problem
        dataset = dataset[:data_size]
        solver = GreedyKernelPoints(
            random_key=self.random_key,
            coreset_size=coreset_size,
            feature_kernel=SquaredExponentialKernel(),
            candidate_batch_size=candidate_batch_size,
        )
        solver.reduce(dataset)

    @pytest.mark.parametrize(
        "data_size, coreset_size, loss_batch_size",
        [(100, 50, 10), (100, 50, 13), (100, 10, 50), (50, 10, 100), (50, 50, 10)],
        ids=[
            "multiple_loss_batch_size_less_than_coreset_size_less_than_data_size",
            "non_multiple_loss_batch_size_less_than_coreset_size_less_than_data_size",
            "coreset_size_less_than_loss_batch_size_less_than_data_size",
            "coreset_size_less_than_data_size_less_than_loss_batch_size",
            "coreset_size_equal_to_data_size",
        ],
    )
    def test_loss_batch_size(
        self,
        reduce_problem: _ReduceProblem,
        data_size: int,
        coreset_size: int,
        loss_batch_size: int,
    ) -> None:
        """Test that various choices of `loss_batch_size` throw no errors."""
        dataset, _, _ = reduce_problem
        dataset = dataset[:data_size]
        solver = GreedyKernelPoints(
            random_key=self.random_key,
            coreset_size=coreset_size,
            feature_kernel=SquaredExponentialKernel(),
            loss_batch_size=loss_batch_size,
        )
        solver.reduce(dataset)

    @pytest.mark.parametrize(
        ("candidate_coresets", "feature_gramian", "responses", "identity", "expect"),
        (
            (
                jnp.array([[1, 0], [1, 2]]),
                jnp.array([[1, 1 / 2, 1 / 2], [1 / 2, 1, 1 / 5], [1 / 2, 1 / 5, 1]]),
                jnp.array([[0], [1], [2]]),
                jnp.identity(2),
                jnp.array([-164 / 225, -55 / 16]),
            ),
            (
                jnp.array([[0], [1], [2]]),
                jnp.array([[1, 0, 2], [0, 1, 1], [2, 1, 5]]),
                jnp.array([[0], [1], [5]]),
                jnp.identity(1),
                jnp.array([0, -10, -22]),
            ),
            (
                jnp.array([[2, 0], [2, 1]]),
                jnp.array([[1, 0, 2], [0, 1, 1], [2, 1, 5]]),
                jnp.array([[0], [1], [5]]),
                jnp.identity(2),
                jnp.array([-10, -22]),
            ),
            (
                jnp.array([[0, -1], [1, -1], [2, -1]]),
                jnp.array([[1, 0, 2, 0], [0, 1, 1, 0], [2, 1, 5, 0], [0, 0, 0, 0]]),
                jnp.array([[0], [1], [5]]),
                jnp.array([[1, 0], [0, 0]]),
                jnp.array([0, -10, -22]),
            ),
        ),
        ids=("standalone", "integration[0]", "integration[1]", "padding"),
    )
    def test_analytic_greedy_kernel_points_loss(
        self,
        candidate_coresets: jax.Array,
        feature_gramian: jax.Array,
        responses: jax.Array,
        identity: jax.Array,
        expect: jax.Array,
    ) -> None:
        r"""
        Test _greedy_kernel_points_loss() analytically with zero regularisation.

        In the first test case, we consider a dataset of size three, and we test the
        second iteration, where we have already chosen the first index to be element
        one. The feature Gramian is

        .. math::

            K^{(11)} = \begin{pmatrix}
                1 & \frac{1}{2} & \frac{1}{2} \\
                \frac{1}{2} & 1 & \frac{1}{5} \\
                \frac{1}{2} & \frac{1}{5} & 1
                \end{pmatrix}

        with response vector

        .. math::

            y^{(1)} = \begin{pmatrix} 0 \\ 1 \\ 2 \end{pmatrix} .

        If element zero joins the coreset,

        .. math::

            K^{(12)} &= \begin{pmatrix} \frac{1}{2} & 1 \\ 1 & \frac{1}{2} \\
                \frac{1}{5} & \frac{1}{2} \end{pmatrix} ; \\
            K^{(22)} &= \begin{pmatrix} 1 & \frac{1}{2} \\
                \frac{1}{2} & 1 \end{pmatrix} ; \\
            y^{(2)} &= \begin{pmatrix} 1 \\ 0 \end{pmatrix} .

        Then, the inverse of the kernel matrix is

        .. math::

            {K^{(22)}}^{-1} = \frac{4}{3}
                \begin{pmatrix} 1 & -\frac{1}{2} \\ -\frac{1}{2} & 1 \end{pmatrix} ,

        and the prediction is

        .. math::

            z &= K^{(12)} {K^{(22)}}^{-1} y^{(2)} \\
            &= \frac{4}{3} \begin{pmatrix} \frac{1}{2} & 1 \\ 1 & \frac{1}{2} \\
                \frac{1}{5} & \frac{1}{2} \end{pmatrix}
                \begin{pmatrix} 1 & -\frac{1}{2} \\ -\frac{1}{2} & 1 \end{pmatrix}
                \begin{pmatrix} 1 \\ 0 \end{pmatrix} \\
            &= \begin{pmatrix} \frac{1}{2} & 1 \\ 1 & \frac{1}{2} \\
                \frac{1}{5} & \frac{1}{2} \end{pmatrix}
                \begin{pmatrix} -\frac{4}{3} \\ -\frac{2}{3} \end{pmatrix} \\
            &= \begin{pmatrix} 0 \\ 1 \\ -\frac{1}{15} \end{pmatrix} ,

        so the loss is

        .. math::

            L &= 0^2 + 0^2 + \left( \frac{31}{15} \right)^2 \\
            &= \frac{961}{225} .

        If element two joins the coreset,

        .. math::

            K^{(12)} &= \begin{pmatrix} \frac{1}{2} & \frac{1}{2} \\ 1 & \frac{1}{5} \\
                \frac{1}{5} & 1 \end{pmatrix} ; \\
            K^{(22)} &= \begin{pmatrix} 1 & \frac{1}{5} \\
                \frac{1}{5} & 1 \end{pmatrix} ; \\
            y^{(2)} &= \begin{pmatrix} 1 \\ 2 \end{pmatrix} .

        Then, the inverse of the kernel matrix is

        .. math::

            {K^{(22)}}^{-1} = \frac{25}{24}
                \begin{pmatrix} 1 & -\frac{1}{5} \\ -\frac{1}{5} & 1 \end{pmatrix} ,

        and the prediction is

        .. math::

            z &= K^{(12)} {K^{(22)}}^{-1} y^{(2)} \\
            &= \frac{25}{24} \begin{pmatrix} \frac{1}{2} & \frac{1}{2} \\
                1 & \frac{1}{5} \\frac{1}{5} & 1 \end{pmatrix}
                \begin{pmatrix} 1 & -\frac{1}{5} \\ -\frac{1}{5} & 1 \end{pmatrix}
                \begin{pmatrix} 1 \\ 2 \end{pmatrix} \\
            &= \begin{pmatrix} \frac{1}{2} & \frac{1}{2} \\ 1 & \frac{1}{5} \\
                \frac{1}{5} & 1 \end{pmatrix}
                \begin{pmatrix} \frac{5}{8} \\ \frac{15}{8} \end{pmatrix} \\
            &= \begin{pmatrix} \frac{5}{4} \\ 1 \\ 2 \end{pmatrix} ,

        so the loss is

        .. math::

            L &= \left( \frac{5}{4} \right)^2 + 0^2 + 0^2\\
            &= \frac{25}{16} .

        In the implementation of greedy_kernel_points_loss(), the constant
        :math:`\left\| y^{(1)} \right\|^2 = 5` is excluded, so the expected losses are
        five less than the full analytic loss,

        .. math::

            \begin{pmatrix} -\frac{164}{225} \\ -\frac{55}{16} .

        The remaining test cases are from :mod:`examples.greedy_kernel_points_analytic`.
        Now, :math:`\left\| y^{(1)} \right\|^2 = 26`, so 26 needs to be subtracted from
        the calculated losses to match the implementation in
        _greedy_kernel_points_loss().

        The final test includes padding introduced to take advantage of fixed array
        sizes to avoid JAX recompilation.

        :param candidate_coresets: Array of all candidate coresets.
        :param feature_gramian: Gramian of input data.
        :param responses: Responses of input data.
        :param identity: Identity matrix of size matching the coreset. If padding is
            used in the candidate coresets, this must be padded with zeroes to match.
        :param expect: Loss for each candidate coreset with constant term excluded.
        """
        actual = _greedy_kernel_points_loss(
            candidate_coresets=candidate_coresets,
            responses=responses,
            feature_gramian=feature_gramian,
            regularisation_parameter=0,
            identity=identity,
            least_squares_solver=MinimalEuclideanNormSolver(),
            loss_batch=jnp.arange(3),
        )
        assert actual == pytest.approx(expect)


class _ExplicitPaddingInvariantSolver(ExplicitSizeSolver, PaddingInvariantSolver):
    """Required for mocking with multiple inheritance."""


class TestMapReduce(SolverTest):
    """Test cases for :class:`coreax.solvers.composite.MapReduce`."""

    leaf_size: int = 32
    coreset_size: int = 16

    @override
    @pytest.fixture(scope="class")
    def solver_factory(self, request) -> jtu.Partial:
        del request

        class _MockTree:
            def __init__(self, _data: np.ndarray, **kwargs):
                del kwargs
                self.data = _data

            def get_arrays(self) -> tuple[Union[np.ndarray, None], ...]:
                """Mock sklearn.neighbours.BinaryTree.get_arrays method."""
                return None, np.arange(len(self.data)), None, None

        def get_solver(flavour: str = "original", **kwargs) -> Solver:
            base_solver = MagicMock(_ExplicitPaddingInvariantSolver)
            base_solver.coreset_size = self.coreset_size

            if flavour == "original":
                # As this test was originally. Build a PseudoCoreset that is essentially
                # a Coresubset.

                def mock_reduce(
                    dataset: Data, solver_state: None = None
                ) -> tuple[AbstractCoreset[Data, Data], None]:
                    indices = jnp.arange(base_solver.coreset_size)
                    return PseudoCoreset(dataset[indices], dataset), solver_state

            elif flavour == "coresubset":
                # Do essentially the same as for "original", just make it an actual
                # Coresubset.

                def mock_reduce(
                    dataset: Data, solver_state: None = None
                ) -> tuple[AbstractCoreset[Data, Data], None]:
                    indices = jnp.arange(base_solver.coreset_size)
                    return Coresubset.build(indices, dataset), solver_state

            elif flavour == "pseudo":
                # Make a PseudoCoreset that just contains ones.

                def mock_reduce(
                    dataset: Data, solver_state: None = None
                ) -> tuple[AbstractCoreset[Data, Data], None]:
                    points = jnp.ones(base_solver.coreset_size, dtype=jnp.float32)
                    return PseudoCoreset.build(points, dataset), solver_state

            else:
                raise ValueError(flavour)

            base_solver.reduce = mock_reduce

            final_args = {
                "base_solver": base_solver,
                "leaf_size": self.leaf_size,
                "tree_type": _MockTree,
            }

            # Allow overriding arguments
            final_args.update(**kwargs)

            return MapReduce(**final_args)

        return jtu.Partial(get_solver)

    @override
    @pytest.fixture(scope="class", params=["original", "pseudo", "coresubset"])
    def reduce_problem(
        self,
        request: pytest.FixtureRequest,
        solver_factory: jtu.Partial,
    ) -> _ReduceProblem:
        dataset = jnp.broadcast_to(jnp.arange(self.shape[0])[..., None], self.shape)
        solver_flavour = request.param
        solver_factory.keywords["flavour"] = solver_flavour
        solver: Solver = solver_factory()
        if solver_flavour == "original":
            # Expected procedure:
            # len(dataset) = 128; leaf_size=32
            # (1): 128 -> Partition -> 4x32 -> Reduce -> 4x16 -> Reshape -> 64
            # (2): 64  -> Partition -> 2x32 -> Reduce -> 2x16 -> Reshape -> 32
            # (3): 32  -> Partition -> 1x32 -> Reduce -> 1x16 -> Reshape -> 16
            # Expected sub-coreset values at each step.
            # (1): [:16], [32:48], [64:80], [96:112]
            # (2): [:16], [64:80]
            # (3): [:16] <- 'coreset_size'
            expected_coreset = PseudoCoreset.build(
                dataset[: self.coreset_size], dataset
            )
        elif solver_flavour == "coresubset":
            # As above, but a Coresubset
            expected_coreset = Coresubset.build(jnp.arange(self.coreset_size), dataset)
        elif solver_flavour == "pseudo":
            # PseudoCoreset mock solver just returns ones
            expected_coreset = PseudoCoreset.build(jnp.ones(self.coreset_size), dataset)
        else:
            raise ValueError(solver_flavour)
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
        selected_indices = coreset.indices.data

        assert jnp.any(selected_indices >= coreset_size), (
            "MapReduce should select points beyond the first few"
        )

        # Check if there are indices from different partitions
        partitions_represented = jnp.unique(selected_indices // leaf_size)
        assert len(partitions_represented) > 1, (
            "MapReduce should select points from multiple partitions"
        )

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
        ) -> tuple[Coresubset[Data], None]:
            half_size = interleaved_base_solver.coreset_size // 2
            indices = jnp.arange(interleaved_base_solver.coreset_size)
            forward_indices = indices[:half_size]
            backward_indices = -(indices[:half_size] + 1)
            interleaved_indices = jnp.stack(
                [forward_indices, backward_indices], axis=1
            ).ravel()

            if interleaved_base_solver.coreset_size % 2 != 0:
                interleaved_indices = jnp.append(interleaved_indices, half_size)
            return Coresubset.build(interleaved_indices, dataset), solver_state

        interleaved_base_solver.reduce = interleaved_mock_reduce

        original_data = Data(
            jnp.array(
                [
                    10,
                    20,
                    30,
                    210,
                    40,
                    60,
                    180,
                    90,
                    150,
                    70,
                    120,
                    200,
                    50,
                    140,
                    80,
                    170,
                    100,
                    190,
                    110,
                    160,
                    130,
                ]
            )
        )
        expected_coreset_data = Data(jnp.array([0, 110, 90]))

        coreset, _ = MapReduce(base_solver=interleaved_base_solver, leaf_size=6).reduce(
            original_data
        )
        assert eqx.tree_equal(coreset.points.data == expected_coreset_data.data)


class TestCaratheodoryRecombination(RecombinationSolverTest):
    """Tests for :class:`coreax.solvers.recombination.CaratheodoryRecombination`."""

    @override
    @pytest.fixture(scope="class")
    def solver_factory(self, request: pytest.FixtureRequest) -> jtu.Partial:
        del request
        return jtu.Partial(CaratheodoryRecombination, test_functions=None, rcond=None)


class TestTreeRecombination(RecombinationSolverTest):
    """Tests for :class:`coreax.solvers.recombination.TreeRecombination`."""

    @override
    @pytest.fixture(scope="class")
    def solver_factory(self, request: pytest.FixtureRequest) -> jtu.Partial:
        del request
        return jtu.Partial(
            TreeRecombination, test_functions=None, rcond=None, tree_reduction_factor=3
        )


class TestKernelThinning(ExplicitSizeSolverTest):
    """Test cases for :class:`coreax.solvers.coresubset.KernelThinning`."""

    @override
    @pytest.fixture(scope="class")
    def solver_factory(self, request: pytest.FixtureRequest) -> jtu.Partial:
        del request
        kernel = PCIMQKernel()
        coreset_size = self.shape[0] // 10
        return jtu.Partial(
            KernelThinning,
            coreset_size=coreset_size,
            random_key=self.random_key,
            kernel=kernel,
            delta=0.01,
            sqrt_kernel=kernel,
        )

    def test_kt_half_analytic(self) -> None:
        # pylint: disable=line-too-long
        r"""
        Test the halving step of kernel thinning on analytical example.

        We aim to split [0.7,0.55,0.6,0.65,0.9,0.10,0.11,0.12] into two coresets, S1
        and S2, each containing 4 elements, enforcing two unique coresets.

        First, let S be the full dataset, with S1 and S2 as subsets. S1 will contain
        half the elements, and S2 will contain the other half. Let :math:`k` represent
        the square root kernel. We will use variables labelled :math:`a`, :math:`b`,
        :math:`\\alpha`, :math:`\\sigma`, :math:`\\delta`, and probability, which
        will be updated iteratively to form the coresets.

        We process pairs :math:`(x, y)` sequentially: :math:`(0.7, 0.55)`, then
        :math:`(0.6, 0.65)`, and so on. For each pair, we compute a swap probability
        that determines whether :math:`x` goes to S1 and :math:`y` to S2, or vice
        versa. In either case, both :math:`x` and :math:`y` are added to S.

        We swap x and y with probability equal to swap probability and then add x and y
        to S1 and S2 respectively. For the purpose of analytic test, if swap probability
        is less than 0.5, we do not swap and add x and y to S1 and S2 respectively,
        otherwise we swap x and y and then add x to S1 and y to S2.

        The process is as follows:

        - Start with :math:`\\delta = \\frac{1}{8}` and :math:`\\sigma = 0`.
        - Take a pair :math:`(x, y)`.
        - Compute :math:`b` and :math:`\\alpha`.
        - Compute :math:`a` and update :math:`\\sigma`.
        - Compute probability, update the sets, and proceed to the next pair,
          using the updated :math:`\\sigma`.

        Functions:

        - :func:`b(x, y)` computes the distance measure between two points based
          on the kernel:
          .. math::
              b(x, y) = \\sqrt{k(x, x) + k(y, y) - 2 \\cdot k(x, y)}

        - :func:`get_swap_params(sigma, b_value, delta)` computes the parameters
          required to update :math:`a` and :math:`\\sigma`:
          .. math::
              a = \\max\\left(b \\cdot \\sigma \\cdot \\sqrt{2 \\ln(2 / \\delta)},
                              b^2\\right)
          .. math::
              \\sigma^2_{new} = \\sigma^2 + \\max\\left(
                              \\frac{b^2 (1 + (b^2 - 2a) \\cdot \\sigma^2 / a^2)}{a^2},
                              0\\right)
          .. math::
              \\sigma_{new} = \\sqrt{\\sigma^2_{new}}

        - :func:`alpha(x, y, S, S1)` computes the difference between the total
          kernel sum for all elements in S and the subset S1:
          .. math::
              \\alpha(x, y, S, S1) =
              \\sum_{s \\in S}\\left(k(s, x) - k(s, y)\\right) - 2
              \\sum_{s \\in S1}\\left(k(s, x) - k(s, y)\\right)

        - :func:`get_probability(alpha_val, a)` computes the probability that
          determines the assignment of points to the coresets:
          .. math::
              P = \\min\\left(1, \\max\\left(0, 0.5 \\cdot \\left(1 - \\frac{\\alpha}{a}\\right)\\right)\\right)

        and for square-root-kernel, choose a ``length_scale`` of
        :math:`\frac{1}{\sqrt{2}}` to simplify computations with the
        ``SquaredExponentialKernel``, in particular it becomes:

        .. math::
            k(x, y) = e^{-||x - y||^2}

        Calculations for each pair:

        **Pair (0.7, 0.55):**

        - Inputs: S=[], S1=[], S2=[], sigma=0, delta=1/8.
        - Compute b:
          - .. math::
              b(0.7, 0.55) = \sqrt{k(0.7, 0.7) + k(0.55, 0.55) - 2k(0.7, 0.55)}
              = 0.2109442800283432.
        - Compute alpha:
          - Since S and S1 are empty, alpha = 0.
        - Compute a:
          - .. math::
              a = \max(b \cdot \sigma \cdot \sqrt{2 \ln(2 / \delta)}, b^2)
              = 0.04449748992919922.
        - Update sigma:
          - new_sigma = 0.2109442800283432.
        - Compute probability:
          - .. math::
              p = 0.5 \cdot \left(1 - \frac{\alpha}{a}\right)
              = 0.5.
        - Assign:
          - Since p >= 0.5, assign x=0.7 to S2, y=0.55 to S1, and add both to S.
          - S1 = [0.55], S2 = [0.7], S = [0.7, 0.55].

        ---

        **Pair (0.6, 0.65):**

        - Inputs: S=[0.7, 0.55], S1=[0.55], S2=[0.7], sigma=0.2109442800283432.
        - Compute b:
          - .. math::
              b(0.6, 0.65) = \sqrt{k(0.6, 0.6) + k(0.65, 0.65) - 2k(0.6, 0.65)}
              = 0.07066679745912552.
        - Compute alpha:
          - alpha = -0.014906525611877441.
        - Compute a:
          - a = 0.035102729200688874.
        - Update sigma:
          - new_sigma = 0.2109442800283432.
        - Compute probability:
          - p = 0.7123271822929382.
        - Assign:
          - Since p > 0.5, assign x=0.6 to S2 and y=0.65 to S1, and add both to S.
          - S1 = [0.55, 0.65], S2 = [0.7, 0.6], S = [0.7, 0.55, 0.6, 0.65].

        ---

        **Pair (0.9, 0.1):**

        - Inputs: S=[0.7, 0.55, 0.6, 0.65], S1=[0.55, 0.65], S2=[0.7, 0.6],
          sigma=0.2109442800283432.
        - Compute b:
          - .. math::
              b(0.9, 0.1) = \sqrt{k(0.9, 0.9) + k(0.1, 0.1) - 2k(0.9, 0.1)}
              = 0.9723246097564697.
        - Compute alpha:
          - alpha = 0.12977957725524902.
        - Compute a:
          - a = 0.9454151391983032.
        - Update sigma:
          - new_sigma = 0.9723246097564697.
        - Compute probability:
          - p = 0.43136370182037354.
        - Assign:
          - Since p < 0.5, assign x=0.9 to S1 and y=0.1 to S2, and add both to S.
          - S1 = [0.55, 0.65, 0.9], S2 = [0.7, 0.6, 0.1],
            S = [0.7, 0.55, 0.6, 0.65, 0.9, 0.1].

        ---

        **Pair (0.11, 0.12):**

        - Inputs: S=[0.7, 0.55, 0.6, 0.65, 0.9, 0.1], S1=[0.55, 0.65, 0.9],
          S2=[0.7, 0.6, 0.1], sigma=0.9723246097564697.
        - Compute b:
          - .. math::
              b(0.11, 0.12) = \sqrt{k(0.11, 0.11) + k(0.12, 0.12) - 2k(0.11, 0.12)}
              = 0.014143308624625206.
        - Compute alpha:
          - alpha = 0.008038222789764404.
        - Compute a:
          - a = 0.03238321865838572.
        - Update sigma:
          - new_sigma = 0.9723246097564697.
        - Compute probability:
          - p = 0.3758890628814697.
        - Assign:
          - Since p < 0.5, assign x=0.11 to S1 and y=0.12 to S2, and add both to S.
          - S1 = [0.55, 0.65, 0.9, 0.11], S2 = [0.7, 0.6, 0.1, 0.12],
            S = [0.7, 0.55, 0.6, 0.65, 0.9, 0.1, 0.11, 0.12].

        ---

        **Final result:**
        S1 = [0.55, 0.65, 0.9, 0.11], S2 = [0.7, 0.6, 0.1, 0.12].
        """  # noqa: E501
        # pylint: enable=line-too-long
        length_scale = 1.0 / jnp.sqrt(2)
        kernel = SquaredExponentialKernel()
        sqrt_kernel = SquaredExponentialKernel(length_scale=length_scale)
        delta = 1 / 8
        random_key = jax.random.PRNGKey(seed=0)
        data = Data(jnp.array([0.7, 0.55, 0.6, 0.65, 0.9, 0.10, 0.11, 0.12]))
        thinning_solver = KernelThinning(
            coreset_size=2,
            kernel=kernel,
            random_key=random_key,
            delta=delta,
            sqrt_kernel=sqrt_kernel,
        )

        def deterministic_uniform(_key, _shape=None):
            return 0.5

        # Patch `jax.random.uniform` with `deterministic_uniform`
        with patch("jax.random.uniform", side_effect=deterministic_uniform):
            coresets = [
                jnp.asarray(s.points.data) for s in thinning_solver.kt_half(data)
            ]

        np.testing.assert_array_equal(
            coresets[0], jnp.array([[0.55], [0.65], [0.9], [0.11]])
        )
        np.testing.assert_array_equal(
            coresets[1], jnp.array([[0.7], [0.6], [0.1], [0.12]])
        )


class TestCompressPlusPlus(ExplicitSizeSolverTest):
    """Test cases for :class:`coreax.solvers.coresubset.KernelThinning`."""

    @override
    @pytest.fixture(scope="class")
    def solver_factory(self, request: pytest.FixtureRequest) -> jtu.Partial:
        del request
        kernel = SquaredExponentialKernel()
        coreset_size = self.shape[0] // 8
        return jtu.Partial(
            CompressPlusPlus,
            g=2,
            coreset_size=coreset_size,
            random_key=self.random_key,
            kernel=kernel,
            delta=0.01,
            sqrt_kernel=kernel,
        )

    def test_invalid_g_too_high(self):
        """Test that ValueError is raised when g too high."""
        dataset = Data(jnp.arange(64))  # Create a dataset with 64 elements

        with pytest.raises(
            ValueError,
            match="The over-sampling factor g should be between 0 and 3, inclusive.",
        ):
            solver = CompressPlusPlus(
                g=4,  # Set g to 4, which is outside the valid range (0 to 3)
                coreset_size=3,  # Set coreset_size to 8
                random_key=self.random_key,
                kernel=SquaredExponentialKernel(),
                delta=0.01,
                sqrt_kernel=SquaredExponentialKernel(),
            )
            solver.reduce(dataset)  # Attempt to reduce the dataset

    def test_invalid_coreset_size_incompatible(self):
        """Test that ValueError is raised when coreset_size and g are incompatible."""
        dataset = Data(jnp.arange(64))  # Create a dataset with 64 elements
        g = 0  # Set g to 0
        coreset_size = 17  # Set an incompatible coreset size

        with pytest.raises(
            ValueError,
            match="Coreset size and g are not compatible with the dataset size. "
            "Ensure that the coreset size does not exceed .* or increase g.",
        ):
            solver = CompressPlusPlus(
                g=g,
                coreset_size=coreset_size,
                random_key=self.random_key,
                kernel=SquaredExponentialKernel(),
                delta=0.01,
                sqrt_kernel=SquaredExponentialKernel(),
            )
            solver.reduce(dataset)
