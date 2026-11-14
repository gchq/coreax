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

"""Verify shared solver factories preserve benchmark settings and output paths."""

from pathlib import Path
from unittest.mock import Mock

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from benchmark import (
    blobs_benchmark,
    blobs_benchmark_visualiser,
    mnist_benchmark_visualiser,
)
from coreax import Data
from coreax.benchmark_util import (
    IterativeKernelHerding,
    build_solver_factories,
    initialise_solvers,
)
from coreax.kernels import SquaredExponentialKernel, SteinKernel
from coreax.solvers import (
    CompressPlusPlus,
    KernelHerding,
    KernelThinning,
    RandomSample,
    RPCholesky,
    SteinThinning,
)


@pytest.mark.parametrize("size", [1, 4])
@pytest.mark.parametrize("seed", [0, 42])
def test_blob_solver_configuration_is_preserved(size: int, seed: int) -> None:
    """Compare every factory result with the original explicit configurations."""
    kernel = SquaredExponentialKernel(length_scale=1.25)
    stein = SteinKernel(kernel, lambda x: -jnp.asarray(x))
    key = jax.random.PRNGKey(seed)
    sqrt_kernel = kernel.get_sqrt_kernel(dim=2)
    delta = 0.125
    expected = [
        ("KernelHerding", KernelHerding(coreset_size=size, kernel=kernel)),
        ("RandomSample", RandomSample(coreset_size=size, random_key=key)),
        ("RPCholesky", RPCholesky(coreset_size=size, kernel=kernel, random_key=key)),
        (
            "SteinThinning",
            SteinThinning(coreset_size=size, kernel=stein, regularise=True),
        ),
        (
            "KernelThinning",
            KernelThinning(
                coreset_size=size,
                kernel=kernel,
                random_key=key,
                delta=delta,
                sqrt_kernel=sqrt_kernel,
            ),
        ),
        (
            "CompressPlusPlus",
            CompressPlusPlus(
                coreset_size=size,
                kernel=kernel,
                random_key=key,
                delta=delta,
                sqrt_kernel=sqrt_kernel,
                g=4,
            ),
        ),
        (
            "ProbabilisticIterativeHerding",
            IterativeKernelHerding(
                coreset_size=size,
                random_key=key,
                num_iterations=5,
                temperature=0.001,
                kernel=kernel,
                probabilistic=True,
            ),
        ),
        (
            "IterativeHerding",
            IterativeKernelHerding(
                coreset_size=size,
                random_key=key,
                num_iterations=5,
                temperature=0.001,
                kernel=kernel,
                probabilistic=False,
            ),
        ),
        (
            "CubicProbIterativeHerding",
            IterativeKernelHerding(
                coreset_size=size,
                kernel=kernel,
                probabilistic=True,
                temperature=0.001,
                random_key=key,
                num_iterations=10,
                t_schedule=1 / jnp.linspace(10, 100, 10) ** 3,
            ),
        ),
    ]
    actual = blobs_benchmark.setup_solvers(size, kernel, stein, delta, random_seed=seed)
    assert [name for name, _ in actual] == [name for name, _ in expected]
    for (_, result), (_, reference) in zip(actual, expected, strict=True):
        assert eqx.tree_equal(result, reference)


def test_shared_factory_parameters() -> None:
    """Keep thinning dimension, delta and compression oversampling explicit."""
    kernel = SquaredExponentialKernel(length_scale=2.0)
    stein_factory = Mock(return_value=SteinKernel(kernel, lambda x: -jnp.asarray(x)))
    sqrt_kernel = kernel.get_sqrt_kernel(dim=16)
    oversampling = 7
    registry = build_solver_factories(
        kernel,
        stein_factory,
        jax.random.PRNGKey(1),
        sqrt_kernel=sqrt_kernel,
        delta=0.2,
        cpp_oversampling_factor=oversampling,
    )
    for name, factory in registry.items():
        if name != "Stein Thinning":
            factory(4)
    stein_factory.assert_not_called()
    thinning = registry["Kernel Thinning"](4)
    compression = registry["Compress++"](4)
    assert isinstance(thinning, KernelThinning)
    assert isinstance(compression, CompressPlusPlus)
    assert eqx.tree_equal(thinning.sqrt_kernel, sqrt_kernel)
    assert thinning.delta == pytest.approx(0.2)
    assert compression.g == oversampling
    registry["Stein Thinning"](4)
    stein_factory.assert_called_once_with()


def test_score_model_is_cached_per_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fit KDE once per dataset registry and never reuse it across datasets."""
    data = Data(jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 3.0], [4.0, 1.0]]))
    kde = Mock(wraps=jax.scipy.stats.gaussian_kde)
    monkeypatch.setattr(jax.scipy.stats, "gaussian_kde", kde)
    factories = initialise_solvers(data, jax.random.PRNGKey(0), 1)
    kde.assert_not_called()
    first = factories["Stein Thinning"](1)
    second = factories["Stein Thinning"](2)
    kde.assert_called_once()
    assert isinstance(first, SteinThinning)
    assert isinstance(second, SteinThinning)
    assert isinstance(first.kernel, SteinKernel)
    assert first.kernel is second.kernel
    assert jnp.all(jnp.isfinite(first.kernel.score_function(jnp.array([1.0, 1.0]))))
    previous_fits = kde.call_count
    another = initialise_solvers(data, jax.random.PRNGKey(0), 1)
    another["Stein Thinning"](1)
    assert kde.call_count == previous_fits + 1


def test_calibration_sample_limits_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the historical 300-point and 1000-point median heuristic limits."""
    points = jnp.arange(2200.0).reshape(1100, 2)
    shared_median = Mock(return_value=jnp.array(1.0))
    blob_median = Mock(return_value=jnp.array(1.0))
    monkeypatch.setattr("coreax.benchmark_util.median_heuristic", shared_median)
    monkeypatch.setattr(blobs_benchmark, "median_heuristic", blob_median)
    initialise_solvers(Data(points), jax.random.PRNGKey(0), 1)
    blobs_benchmark.setup_kernel(points)
    expected_limits = (300, 1000)
    assert (
        len(shared_median.call_args.args[0]),
        len(blob_median.call_args.args[0]),
    ) == expected_limits


@pytest.mark.parametrize("override", [False, True])
def test_blob_image_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: bool
) -> None:
    """Resolve default image paths from the script rather than the working directory."""
    repo = tmp_path / "checkout"
    monkeypatch.setattr(
        blobs_benchmark_visualiser,
        "__file__",
        str(repo / "benchmark" / "blobs_benchmark_visualiser.py"),
    )
    monkeypatch.chdir(tmp_path)
    save = Mock()
    monkeypatch.setattr(plt, "savefig", save)
    metrics = {
        name: 1.0
        for name in [
            "Time",
            "Weighted_KSD",
            "Unweighted_MMD",
            "Weighted_MMD",
            "Unweighted_KSD",
        ]
    }
    data = {"1": {"KernelHerding": metrics}, "2": {"KernelHerding": metrics}}
    output = tmp_path / "custom" if override else None
    if override:
        blobs_benchmark_visualiser.plot_benchmarking_results(data, output_dir=output)
    else:
        blobs_benchmark_visualiser.plot_benchmarking_results(data)
    expected = output or repo / "examples" / "benchmarking_images"
    assert expected.is_dir()
    assert save.call_count == len(metrics)
    assert {Path(call.args[0]).parent for call in save.call_args_list} == {expected}
    assert not (tmp_path / "examples").exists()


@pytest.mark.parametrize("override", [False, True])
def test_mnist_image_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: bool
) -> None:
    """Resolve input JSON and output images independently of the working directory."""
    repo = tmp_path / "checkout"
    monkeypatch.setattr(
        mnist_benchmark_visualiser,
        "__file__",
        str(repo / "benchmark" / "mnist_benchmark_visualiser.py"),
    )
    monkeypatch.chdir(tmp_path)
    load = Mock(return_value={"solver": {"1": {}}})
    monkeypatch.setattr(mnist_benchmark_visualiser, "load_benchmark_data", load)
    monkeypatch.setattr(
        mnist_benchmark_visualiser, "compute_statistics", Mock(return_value=({}, {}))
    )
    monkeypatch.setattr(
        mnist_benchmark_visualiser, "compute_time_statistics", Mock(return_value={})
    )
    monkeypatch.setattr(mnist_benchmark_visualiser, "plot_performance", Mock())
    monkeypatch.setattr(plt, "figtext", Mock())
    save = Mock()
    monkeypatch.setattr(plt, "savefig", save)
    output = tmp_path / "custom" if override else None
    if override:
        mnist_benchmark_visualiser.main(output_dir=output)
    else:
        mnist_benchmark_visualiser.main()
    expected = output or repo / "examples" / "benchmarking_images"
    assert expected.is_dir()
    assert {Path(call.args[0]).parent for call in save.call_args_list} == {expected}
    assert {Path(call.args[0]).name for call in save.call_args_list} == {
        "mnist_benchmark_accuracy.png",
        "mnist_benchmark_time_taken.png",
    }
    assert {Path(call.args[0]).parent for call in load.call_args_list} == {
        repo / "benchmark"
    }


def test_blob_run_keeps_seed_separate_from_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wire failure probability and random seed to their named solver parameters."""
    points = np.arange(12.0).reshape(6, 2)
    monkeypatch.setattr(blobs_benchmark, "make_blobs", Mock(return_value=(points,)))
    kernel = SquaredExponentialKernel()
    monkeypatch.setattr(blobs_benchmark, "setup_kernel", Mock(return_value=kernel))
    stein = SteinKernel(kernel, lambda x: -jnp.asarray(x))
    monkeypatch.setattr(blobs_benchmark, "setup_stein_kernel", Mock(return_value=stein))
    solver_setup = Mock(return_value=[])
    monkeypatch.setattr(blobs_benchmark, "setup_solvers", solver_setup)
    monkeypatch.setattr(
        blobs_benchmark, "compute_metrics", Mock(return_value={"solver": {"Time": 0.0}})
    )
    monkeypatch.setattr(
        blobs_benchmark, "__file__", str(tmp_path / "blobs_benchmark.py")
    )
    blobs_benchmark.main()
    expected_seeds = {42, 45, 46, 47, 48}
    assert {
        call.kwargs["random_seed"] for call in solver_setup.call_args_list
    } == expected_seeds
    for call in solver_setup.call_args_list:
        assert 0 < call.kwargs["delta"] < 1
    assert (tmp_path / "blobs_benchmark_results.json").is_file()


def test_new_factory_is_included_in_blob_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding a solver to the shared registry makes it available to the blobs run."""
    kernel = SquaredExponentialKernel()
    stein = SteinKernel(kernel, lambda x: -jnp.asarray(x))
    key = jax.random.PRNGKey(0)
    factories = build_solver_factories(
        kernel,
        lambda: stein,
        key,
        sqrt_kernel=kernel.get_sqrt_kernel(dim=2),
        delta=0.1,
        cpp_oversampling_factor=4,
    )
    extra = Mock(return_value=RandomSample(coreset_size=2, random_key=key))
    factories["Additional Solver"] = extra
    monkeypatch.setattr(
        blobs_benchmark, "build_solver_factories", Mock(return_value=factories)
    )
    solvers = blobs_benchmark.setup_solvers(2, kernel, stein, delta=0.1)
    assert solvers[-1][0] == "Additional Solver"
    extra.assert_called_once_with(2)
