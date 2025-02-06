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

"""Tests for coreset data-structures."""

import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Union
from unittest.mock import MagicMock, Mock

import equinox as eqx
import jax.numpy as jnp
import pytest

from coreax.coreset import Coresubset, PseudoCoreset
from coreax.data import Data, SupervisedData
from coreax.metrics import Metric
from coreax.weights import WeightsOptimiser

# Coresubset
CORESUBSET_INDICES = Data(jnp.arange(5, dtype=jnp.int32)[..., None])
CORESUBSET_POINTS = Data(jnp.arange(5, dtype=jnp.float32)[..., None] * 2)
CORESUBSET_SUPERVISED_POINTS = SupervisedData(
    jnp.arange(5, dtype=jnp.float32)[..., None] * 2,
    jnp.arange(5, dtype=jnp.float32)[..., None] * 3,
)
# Pseudo-coreset
PSEUDO_CORESET_POINTS = Data(jnp.ones(5, dtype=jnp.float32)[..., None] * 5)
# Pre-coreset data
PRE_CORESET_DATA = Data(jnp.arange(10, dtype=jnp.float32)[..., None] * 2)
PRE_CORESET_SUPERVISED_DATA = SupervisedData(
    jnp.arange(10, dtype=jnp.float32)[..., None] * 2,
    jnp.arange(10, dtype=jnp.float32)[..., None] * 3,
)


@dataclass(frozen=True)
class CoresetTestSetup:
    """Dataclass holding parameters for Coreset tests."""

    coreset_type: Union[type[Coresubset], type[PseudoCoreset]]
    coreset_input: Data
    materialised_points: Data
    pre_coreset_data: Union[Data, SupervisedData]


@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(
            CoresetTestSetup(
                coreset_type=Coresubset,
                coreset_input=CORESUBSET_INDICES,
                materialised_points=CORESUBSET_POINTS,
                pre_coreset_data=PRE_CORESET_DATA,
            ),
            id="Coresubset-unsupervised",
        ),
        pytest.param(
            CoresetTestSetup(
                coreset_type=Coresubset,
                coreset_input=CORESUBSET_INDICES,
                materialised_points=CORESUBSET_SUPERVISED_POINTS,
                pre_coreset_data=PRE_CORESET_SUPERVISED_DATA,
            ),
            id="Coresubset-supervised",
        ),
        pytest.param(
            CoresetTestSetup(
                coreset_type=PseudoCoreset,
                coreset_input=PSEUDO_CORESET_POINTS,
                materialised_points=PSEUDO_CORESET_POINTS,
                pre_coreset_data=PRE_CORESET_DATA,
            ),
            id="PseudoCoreset-unsupervised",
        ),
        pytest.param(
            CoresetTestSetup(
                coreset_type=PseudoCoreset,
                coreset_input=PSEUDO_CORESET_POINTS,
                materialised_points=PSEUDO_CORESET_POINTS,
                pre_coreset_data=PRE_CORESET_SUPERVISED_DATA,
            ),
            id="PseudoCoreset-supervised",
        ),
    ],
)
class TestCoresetCommon:
    """Common tests for `PseudoCoreset` and `Coresubset`."""

    def test_deprecated_nodes(
        self,
        setup: CoresetTestSetup,
    ):
        """Test that the now-deprecated `nodes` property works as before."""
        coreset = setup.coreset_type(setup.coreset_input, setup.pre_coreset_data)
        with pytest.warns(DeprecationWarning):
            nodes = coreset.nodes

        if isinstance(coreset, PseudoCoreset):
            assert nodes == coreset.points
        elif isinstance(coreset, Coresubset):
            assert nodes == coreset.indices
        else:
            raise TypeError(type(coreset))

    def test_deprecated_coreset(
        self,
        setup: CoresetTestSetup,
    ):
        """Test that the now-deprecated `coreset` property works as before."""
        coreset = setup.coreset_type(setup.coreset_input, setup.pre_coreset_data)
        with pytest.warns(DeprecationWarning):
            points = coreset.coreset
        assert eqx.tree_equal(points, coreset.points)

    @pytest.mark.parametrize(
        "use_build", [True, False], ids=["use_build", "use_deprecated_init"]
    )
    @pytest.mark.parametrize(
        "is_coreset_input_data",
        [True, False],
        ids=["coreset_input_data", "coreset_input_array"],
    )
    @pytest.mark.parametrize(
        "is_pre_coreset_data_data",
        [True, False],
        ids=["pre_coreset_data", "coreset_input_arrays"],
    )
    def test_build_array_conversion(
        self,
        setup: CoresetTestSetup,
        use_build: bool,
        is_coreset_input_data: bool,
        is_pre_coreset_data_data: bool,
    ):
        """
        Test the behaviour of `build`.

        The nodes and data can be passed as an `Array` or as a `Data` instance. In the
        former case, we expect this array to be automatically converted to a `Data`
        instance.

        We also support passing an `(Array, Array)` tuple, which will be converted to
        a `SupervisedData` instance.

        Also tests that this code still functions using __init__, but that it raises
        a DeprecationWarning.
        """
        if is_coreset_input_data:
            coreset_input_final = setup.coreset_input
        else:
            coreset_input_final = setup.coreset_input.data

        if is_pre_coreset_data_data:
            pre_coreset_data_final = setup.pre_coreset_data
        elif isinstance(setup.pre_coreset_data, SupervisedData):
            pre_coreset_data_final = (
                setup.pre_coreset_data.data,
                setup.pre_coreset_data.supervision,
            )
        else:
            pre_coreset_data_final = setup.pre_coreset_data.data

        if use_build:
            coreset_from_arrays = setup.coreset_type.build(
                coreset_input_final, pre_coreset_data_final
            )
        else:
            # Check we get a deprecation warning, but it still works.
            # Note that if we pass both as Data instances, we shouldn't get any
            # deprecation warning.
            ctx = (
                nullcontext()
                if (is_coreset_input_data and is_pre_coreset_data_data)
                else pytest.warns(DeprecationWarning)
            )
            with ctx:
                coreset_from_arrays = setup.coreset_type(
                    coreset_input_final,  # pyright: ignore[reportArgumentType]
                    pre_coreset_data_final,  # pyright: ignore[reportArgumentType]
                )
        coreset_from_data = setup.coreset_type.build(
            setup.coreset_input, setup.pre_coreset_data
        )
        assert coreset_from_arrays == coreset_from_data

    def test_materialization(self, setup: CoresetTestSetup):
        """Test the coreset materialisation behaviour."""
        coreset = setup.coreset_type(setup.coreset_input, setup.pre_coreset_data)
        assert setup.materialised_points == coreset.points

    def test_len(self, setup: CoresetTestSetup):
        """Test the coreset length."""
        coreset = setup.coreset_type(setup.coreset_input, setup.pre_coreset_data)
        assert len(coreset) == len(setup.coreset_input.data)

    def test_solve_weights(self, setup: CoresetTestSetup):
        """Test the weights solving convenience interface."""
        solver = MagicMock(WeightsOptimiser)
        solved_weights = jnp.full_like(jnp.asarray(setup.coreset_input), 123)
        solver.solve.return_value = solved_weights
        re_weighted_nodes = eqx.tree_at(
            lambda x: x.weights, setup.coreset_input, solved_weights
        )
        coreset = setup.coreset_type(setup.coreset_input, setup.pre_coreset_data)
        coreset_expected = setup.coreset_type(re_weighted_nodes, setup.pre_coreset_data)
        kwargs = {"test": None}
        coreset_solved_weights = coreset.solve_weights(solver, **kwargs)
        assert eqx.tree_equal(coreset_solved_weights, coreset_expected)
        solver.solve.assert_called_with(
            coreset.pre_coreset_data, coreset.points, **kwargs
        )

    def test_compute_metric(self, setup: CoresetTestSetup):
        """Test the metric computation convenience interface."""
        metric = MagicMock(spec=Metric)
        expected_metric = jnp.asarray(123)
        metric.compute = Mock(return_value=expected_metric)
        coreset = setup.coreset_type(setup.coreset_input, setup.pre_coreset_data)
        kwargs = {"test": None}
        coreset_metric = coreset.compute_metric(metric, **kwargs)
        assert eqx.tree_equal(coreset_metric, expected_metric)
        metric.compute.assert_called_with(
            coreset.pre_coreset_data, coreset.points, **kwargs
        )


class TestCoresetErrors:
    """
    Test various error cases common to both coreset classes.

    These are a separate class as they don't need the same parametrisation as the
    main tests.
    """

    @pytest.mark.parametrize("coreset_type", [Coresubset, PseudoCoreset])
    def test_coreset_too_large(
        self, coreset_type: Union[type[Coresubset], type[PseudoCoreset]]
    ):
        """Test we get an appropriate error if the coreset is too large."""
        indices_or_nodes = jnp.arange(10, dtype=jnp.int32)
        data = jnp.arange(5, dtype=jnp.int32)
        with pytest.raises(
            ValueError,
            match=r"len\(points\)",
        ):
            coreset_type.build(indices_or_nodes, data)

    @pytest.mark.parametrize("coreset_type", [Coresubset, PseudoCoreset])
    def test_invalid_pre_coreset_data_type(
        self, coreset_type: Union[type[Coresubset], type[PseudoCoreset]]
    ):
        """Test we get an appropriate error if pre_coreset_data is the wrong type."""
        indices_or_nodes = jnp.arange(10, dtype=jnp.int32)
        with pytest.raises(TypeError, match="pre_coreset_data"):
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=DeprecationWarning)
                coreset_type.build(indices_or_nodes, object())  # pyright: ignore[reportArgumentType, reportCallIssue]

    @pytest.mark.parametrize("coreset_type", [Coresubset, PseudoCoreset])
    def test_invalid_indices_or_points_type(
        self, coreset_type: Union[type[Coresubset], type[PseudoCoreset]]
    ):
        """Test we get an appropriate error if the indices/points are the wrong type."""
        pre_coreset_data = jnp.arange(10, dtype=jnp.int32)
        with pytest.raises(
            TypeError, match="indices" if coreset_type is Coresubset else "nodes"
        ):
            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=DeprecationWarning)
                coreset_type(object(), pre_coreset_data)  # pyright: ignore[reportArgumentType, reportCallIssue]


class TestCoresubset:
    """Tests specific to `coreax.coreset.Coresubset`."""

    def test_unweighted_indices(self):
        """Test the coresubset 'unweighted_indices' property."""
        coresubset = Coresubset(CORESUBSET_INDICES, PRE_CORESET_DATA)
        expected_indices = CORESUBSET_INDICES.data.squeeze()
        assert eqx.tree_equal(expected_indices, coresubset.unweighted_indices)
