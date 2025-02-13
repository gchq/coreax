# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

-

### Fixed

-

### Changed

-

### Removed

-

### Deprecated

-


## [0.4.0]

### Added

- Support for Python 3.13. (https://github.com/gchq/coreax/pull/950)
- Contributor-facing features:
  - Automated testing of the built package before release.
    (https://github.com/gchq/coreax/pull/858)
  - Automated static type checking in CI with Pyright.
    (https://github.com/gchq/coreax/pull/921)
- Added an analytical test for SteinThinning, and associated documentation in
  `tests.unit.test_solvers`. (https://github.com/gchq/coreax/pull/842)
- Added an analytical test for `KernelHerding.refine` on an existing coreset.
  (https://github.com/gchq/coreax/pull/879)
- Added benchmarking scripts:
  - MNIST (train a classifier on coreset of training data, test on testing data).
    (https://github.com/gchq/coreax/pull/802)
  - Blobs (generate synthetic data using `sklearn.datasets.make_blobs` and compare MMD
    and KSD metrics). (https://github.com/gchq/coreax/pull/802)
  - David (extract pixel locations and values from an image and plot coresets side by
    side for visual benchmarking). (https://github.com/gchq/coreax/pull/880)
  - Pounce (extract frames from a video and use coreset algorithms to select the best
    frames). (https://github.com/gchq/coreax/pull/893)
- Benchmarking results added on documentation.
  (https://github.com/gchq/coreax/pull/908)
- `benchmark` dependency group for benchmarking dependencies.
  (https://github.com/gchq/coreax/pull/888)
- `example` dependency group for running example scripts.
  (https://github.com/gchq/coreax/pull/909)
- Added a method `SquaredExponentialKernel.get_sqrt_kernel` which returns a square
  root kernel for the squared exponential kernel.
  (https://github.com/gchq/coreax/pull/883)
- Added a new coreset algorithm Kernel Thinning.
  (https://github.com/gchq/coreax/pull/915)
- Added (loose) lower bounds to all direct dependencies.
  (https://github.com/gchq/coreax/pull/920)
- Added Kernel Thinning to existing benchmarking tests.
  (https://github.com/gchq/coreax/pull/927)
- Added an option to run Kernel Herding probabilistically.
  (https://github.com/gchq/coreax/pull/941)
- Added an example script for iterative and probabilistic refinement using Kernel
  Herding. (https://github.com/gchq/coreax/pull/953)
- Added Compress++ coreset reduction algorithm. 
  (https://github.com/gchq/coreax/issues/934)
- Added an option to run Kernel Herding probabilistically. 
  (https://github.com/gchq/coreax/pull/941)

### Fixed

- `MMD.compute` no longer returns `nan`. (https://github.com/gchq/coreax/issues/855)
- Corrected an implementation error in `coreax.solvers.CaratheodoryRecombination`,
  which caused numerical instability when using either `CaratheodoryRecombination`
  or `TreeRecombination` on GPU machines. (https://github.com/gchq/coreax/pull/874, see
  also https://github.com/gchq/coreax/issues/852 and
  https://github.com/gchq/coreax/issues/853)
- `KernelHerding.refine` correctly computes a refinement of an existing coreset.
  (https://github.com/gchq/coreax/issues/870)
- Pylint pre-commit hook is now configured as the Pylint docs recommend.
  (https://github.com/gchq/coreax/pull/899)
- Type annotations so that core coreax package passes Pyright.
  (https://github.com/gchq/coreax/pull/906)
- Type annotations so that the example scripts pass Pyright.
  (https://github.com/gchq/coreax/pull/921)
- `KernelThinning` now computes swap probability correctly.
  (https://github.com/gchq/coreax/pull/932)
- Incorrectly-implemented tests for the gradients of `PeriodicKernel`.
  (https://github.com/gchq/coreax/pull/936)
- `MapReduce`'s warning about a solver not being padding-invariant is now raised at the
  correct stack level. (https://github.com/gchq/coreax/pull/951)
- `len(coresubset.points)` is no longer incorrect for a coresubset of size 1 from a 2d
  dataset. (https://github.com/gchq/coreax/pull/957)

### Changed

- Moved coverage and performance data from GitHub gist to coreax-metadata repo.
  (https://github.com/gchq/coreax/pull/887)
- **[BREAKING CHANGE]** Equinox dependency version is changed from `<0.11.8` to `>=0.
  11.5`. (https://github.com/gchq/coreax/pull/898)
- **[BREAKING CHANGE]** The `jaxtyping` version is now lower bounded at `v0.2.31` to
  enable `coreax.data.Data` jaxtyping compatibility.
- Refactored the `Coreset` types - instead of `Coreset` and `Coresubset(Coreset)`, we
  now have `AbstractCoreset`, `PseudoCoreset(AbstractCoreset)`, and
  `Coresubset(AbstractCoreset)`. See "Deprecated" below for more details of this change.
  (https://github.com/gchq/coreax/pull/943)

### Removed

- The `coreax.kernel` module, deprecated in v0.3.0, has been removed. The kernels have
  been moved to submodules of `coreax.kernels` - see the "Deprecated" section of v0.3.0
  for more information. (https://github.com/gchq/coreax/pull/958)
- `coreax.util.median_heuristic`, deprecated in v0.3.0 has been removed. This should be
  replaced with `coreax.kernels.util.median_heuristic`.
  for more information. (https://github.com/gchq/coreax/pull/958)

### Deprecated

- Uses of `Coreset` should be replaced with `AbstractCoreset` (for a general coreset,
  such as in a function argument type hint), or `PseudoCoreset` (for the specific case
  of a coreset that is not necessarily a coresubset).
  (https://github.com/gchq/coreax/pull/943)
- Uses of `Coreset.coreset` should be replaced with `Coreset.points`.
  (https://github.com/gchq/coreax/pull/943)
- Uses of `Coreset.nodes` should be replaced with `Coresubset.indices` or
  `PseudoCoreset.points`, depending on whether the coreset is a coresubset or a
  pseudo-coreset. (https://github.com/gchq/coreax/pull/943)
- Passing `Array` or `tuple[Array, Array]` into coreset constructors is now deprecated -
  either pass in `Data` or `SupervisedData` instances, or use the `build()` class
  method which handles the conversion. (https://github.com/gchq/coreax/pull/943)


## [0.3.1]

### Added

- Added an analytical test for RPCholesky, and associated documentation in
  `tests.unit.test_solvers`. (https://github.com/gchq/coreax/pull/822)
- Added a unit test for RPCholesky to check whether the coreset has duplicates.
  (https://github.com/gchq/coreax/pull/836)
- Enabled `jaxtyping` compatible type hinting for `coreax.data.Data`, to indicate the
  expected type and shape of a `Data` objects `Data.data` array attribute. For example
  `Bool[Data, "n d"]` indicates `Data.data` should be an `n d` array of bools.

### Fixed

- `RPCholesky.reduce` in `coreax.solvers.coresubset` now computes the iteration step
  correctly. (https://github.com/gchq/coreax/pull/825)
- `RPCholesky.reduce` in `coreax.solvers.coresubset` now does not produce duplicate
  points in the coreset.(https://github.com/gchq/coreax/pull/836)
- Fixed the example `examples.david_map_reduce_weighted` to prevent errors when
  downsampling is enabled, and to make it run faster.
  (https://github.com/gchq/coreax/pull/821)
- Build includes sub-packages. (https://github.com/gchq/coreax/pull/845)

### Changed

- Test dependency from `opencv-python` to `opencv-python-headless`.
  (https://github.com/gchq/coreax/pull/848)
- Updated installation instructions in README. (https://github.com/gchq/coreax/pull/848)


## [0.3.0] - [YANKED]

Yanked due to build failure.

### Added

- Added Kernel Stein Discrepancy divergence in `coreax.metrics.KSD`.(https://github.com/gchq/coreax/pull/659)
- Added the `coreax.solvers.recombination` module, which provides the following new solvers:
  - `RecombinationSolver`: an abstract base class for recombination solvers.
  - `CaratheodoryRecombination`: a simple deterministic approach to solving recombination problems.
  - `TreeRecombination`: an advanced deterministic approach that utilises `CaratheodoryRecombination`,
    but is faster for solving all but the smallest recombination problems.(https://github.com/gchq/coreax/pull/504)
- Added supervised coreset construction algorithm in `coreax.solvers.GreedyKernelPoints`.(https://github.com/gchq/coreax/pull/686)
- Added `coreax.kernels.PowerKernel` to replace repeated calls of `coreax.kernels.ProductKernel`
within the `**` magic method of `coreax.kernel.ScalarValuedKernel`.(https://github.com/gchq/coreax/pull/708)
- Added scalar-valued kernel functions `coreax.kernels.PoissonKernel` and `coreax.kernels.MaternKernel`.([#742](https://github.com/gchq/coreax/pull/742))
- Added `progress_bar` attribute to `coreax.score_matching.SlicedScoreMatching` to enable or
  disable tqdm progress bar terminal output. Defaults to disabled (`False`).(https://github.com/gchq/coreax/pull/761)
- Added analytical tests for kernel herding, and associated documentation in `tests.unit.test_solvers`.(https://github.com/gchq/coreax/pull/794)
- Added CI workflow for performance testing.
- Added array dimensions to type annotations using jaxtyping.(https://github.com/gchq/coreax/pull/746)
- Added integration test for `coreax.solver.recombination.TreeRecombination`.(https://github.com/gchq/coreax/pull/798)

### Fixed

- Fixed `MapReduce` in `coreax.solvers.composite.py` to keep track of the indices.(https://github.com/gchq/coreax/pull/779)
- Fixed negative weights on `coreax.weights.qp`.(https://github.com/gchq/coreax/pull/698)

### Changed

- Refactored `coreax.inverses.py` functionality into `coreax.least_squares.py`:
  - `coreax.inverses.RegularisedInverseApproximator` replaced by `coreax.least_squares.RegularisedLeastSquaresSolver`.
  - `coreax.inverses.LeastSquaresApproximator` replaced by `coreax.least_squares.MinimalEuclideanNormSolver`.
  - `coreax.inverses.RandomisedEigendecompositionApproximator` replaced by
    `coreax.least_squares.RandomisedEigendecompositionSolver`.(https://github.com/gchq/coreax/pull/700)
- Refactoring of `coreax.kernel.py` into `coreax.kernels` sub-package:
  - `kernels.util.py` holds utility functions relating to kernels e.g. `median_heuristic`.
  - `kernels.base.py` holds the base kernel class `ScalarValuedKernel` (renamed from `Kernel`),
    as well as the base composite classes `UniCompositeKernel` (renamed from `CompositeKernel`),
    `DuoCompositeKernel` (renamed from `PairedKernel`) and the derived duo-composite kernels
    `AdditiveKernel` and `ProductKernel`
  - `coreax.kernels.scalar_valued.py` holds all currently implemented scalar valued kernels e.g.
    `SquaredExponentialKernel`. (https://github.com/gchq/coreax/pull/708)
- Refactored `coreax.weights.py` to make weight solvers generic on data type.(https://github.com/gchq/coreax/pull/709)

### Removed

- `coreax.weights.MMD` - deprecated alias for `coreax.weights.MMDWeightsOptimiser`; deprecated since version 0.2.0.(https://github.com/gchq/coreax/pull/784)
- `coreax.weights.SBQ` - deprecated alias for `coreax.weights.SBQWeightsOptimiser`; deprecated since version 0.2.0.(https://github.com/gchq/coreax/pull/784)
- `coreax.util.squared_distance_pairwise` - deprecated alias for `coreax.util.pairwise(squared_distance)`; deprecated since version 0.2.0.(https://github.com/gchq/coreax/pull/784)
- `coreax.util.pairwise_difference` - deprecated alias for `coreax.util.pairwise(difference)`; deprecated since version 0.2.0.(https://github.com/gchq/coreax/pull/784)

### Deprecated

- All uses of `coreax.kernel.Kernel` should be replaced with `coreax.kernels.base.ScalarValuedKernel`.(https://github.com/gchq/coreax/pull/708)
- All uses of `coreax.kernel.UniCompositeKernel` should be replaced with `coreax.kernels.base.CompositeKernel`.(https://github.com/gchq/coreax/pull/708)
- All uses of `coreax.kernel.PairedKernel` should be replaced with `coreax.kernels.base.DuoCompositeKernel`.(https://github.com/gchq/coreax/pull/708)
- All uses of `coreax.kernel.AdditiveKernel` should be replaced with `coreax.kernels.base.AdditiveKernel`.(https://github.com/gchq/coreax/pull/708)
- All uses of `coreax.kernel.ProductKernel` should be replaced with `coreax.kernels.base.ProductKernel`.(https://github.com/gchq/coreax/pull/708)
- All uses of `coreax.kernel.LinearKernel` should be replaced with `coreax.kernels.scalar_valued.LinearKernel`.(https://github.com/gchq/coreax/pull/708)
- All uses of `coreax.kernel.PolynomialKernel` should be replaced with `coreax.kernels.scalar_valued.PolynomialKernel`.(https://github.com/gchq/coreax/pull/708)
- All uses of `coreax.kernel.SquaredExponentialKernel` should be replaced with `coreax.kernels.scalar_valued.SquaredExponentialKernel`.(https://github.com/gchq/coreax/pull/708)
- All uses of `coreax.kernel.ExponentialKernel` should be replaced with `coreax.kernels.scalar_valued.ExponentialKernel`.(https://github.com/gchq/coreax/pull/708)
- All uses of `coreax.kernel.RationalQuadraticKernel` should be replaced with `coreax.kernels.scalar_valued.RationalQuadraticKernel`.(https://github.com/gchq/coreax/pull/708)
- All uses of `coreax.kernel.PeriodicKernel` should be replaced with `coreax.kernels.scalar_valued.PeriodicKernel`.(https://github.com/gchq/coreax/pull/708)
- All uses of `coreax.kernel.LocallyPeriodicKernel` should be replaced with `coreax.kernels.scalar_valued.LocallyPeriodicKernel`.(https://github.com/gchq/coreax/pull/708)
- All uses of `coreax.kernel.LaplacianKernel` should be replaced with `coreax.kernels.scalar_valued.LaplacianKernel`.(https://github.com/gchq/coreax/pull/708)
- All uses of `coreax.kernel.SteinKernel` should be replaced with `coreax.kernels.scalar_valued.SteinKernel`.(https://github.com/gchq/coreax/pull/708)
- All uses of `coreax.kernel.PCIMQKernel` should be replaced with `coreax.kernels.scalar_valued.PCIMQKernel`.(https://github.com/gchq/coreax/pull/708)
- All uses of `coreax.util.median_heuristic` should be replaced with `coreax.kernels.util.median_heuristic`.(https://github.com/gchq/coreax/pull/708)



## [0.2.1]

### Added

- Pyright to development tools (code does not pass yet)

### Fixed

- Nitpicks in documentation build
- Incorrect package version number

### Changed

- Augmented unroll parameter to be consistent with block size in MMD metric


## [0.2.0]

### Added

- Badge to README to show code coverage percentage.
- Support for Python 3.12.
- Added a deterministic, iterative, and greedy coreset algorithm which targets the
  Kernelised Stein Discrepancy via `coreax.solvers.coresubset.SteinThinning`.
- Added a stochastic, iterative, and greedy coreset algorithm which approximates the Gramian of a given kernel function
via `coreax.solvers.coresubset.RPCholesky`.
- Added `coreax.util.sample_batch_indices` that allows one to sample an array of indices for batching.
- Added kernel classes `coreax.kernel.AdditiveKernel` and `coreax.kernel.ProductKernel` that
allow for arbitrary composition of positive semi-definite kernels to produce new positive semi-definite kernels.
- Added additional kernel functions: `coreax.kernel.Linear`, `coreax.kernel.Polynomial`, `coreax.kernel.RationalQuadratic`,
 `coreax.kernel.Periodic`, `coreax.kernel.LocallyPeriodic`.
- Added capability to approximate the inverses of arrays via least-squares (`coreax.inverses.LeastSquaresApproximator`)
or randomised eigendecomposition (`coreax.inverses.RandomisedEigendecompositionApproximator`) all inheriting
from `coreax.inverses.RegularisedInverseApproximator`,
- Refactor of package to a functional style to allow for JIT-compilation of the codebase in the largest possible scope:
  - Added data classes `coreax.data.Data` and `coreax.data.SupervisedData` that draw
    distinction between supervised and unsupervised datasets, and handle weighted data.
    Replaces `coreax.data.DataReader` and `coreax.data.ArrayData`.
  - Added `coreax.solvers.base.Solver` to replace functionality in `coreax.refine.py`, `coreax.coresubset.py` and
  `coreax.reduction.py`. In particular, `coreax.solvers.base.CoresubsetSolver` parents coresubset
  algorithms, `coreax.solvers.base.RefinementSolver` parents coresubset algorithms which support refinement
  post-reduction, `coreax.solvers.base.ExplicitSizeSolver` parents all coreset algorithms which
  return a coreset of a specific size.
  - `coreax.reduction.MapReduce` functionality moved to `coreax.solvers.composite.MapReduce`, now
  JIT-compilable via promise described in `coreax.solvers.base.PaddingInvariantSolver`.
  - Moved all coresubset algorithms in `coreax.coresubset.py` to `coreax.solvers.coresubset.py`.
  - All coreset algorithms now return a `coreax.coreset.Coreset` rather than modifying a `coreax.reduction.Coreset` in-place.
- Use Equinox instead of manually constructing pytrees.

### Fixed

- Wording improvements in README.
- Documentation now builds without warnings.
- GitHub workflow runs automatically after Pre-commit autoupdate.

### Changed

- Documentation has been rearranged.
- Renamed `coreax.weights.MMD` to `coreax.weights.MMDWeightsOptimiser` and added deprecation warning.
- Renamed `coreax.weights.SBQ` to `coreax.weights.SBQWeightsOptimiser` and added deprecation warning.
- `requirements-*.txt` will no longer be updated frequently, thereby providing stable versions.
- Single requirements files covering all supported Python versions.
- All references to `kernel_matrix_row_{sum,mean}` have been replaced with `Gramian row-mean`.
- `coreax.networks.ScoreNetwork` now allows the user to specify number of hidden layers.
- Classes in `weights.py` and `score_matching.py` now inherit from `equinox.Module`.
- Performance tests replaced by `jit_variants` tests, which checks whether a function
  has been compiled for reuse.
- Replace some pygrep-hooks with ruff equivalents.
- Use Pytest fixtures instead of unittest style.

### Removed

- Bash script to run integration tests has been removed. `pytest tests/integration` should now work as expected.
- Tests for `coreax.kernels.Kernel.{calculate, update}_kernel_matrix_row_sum`.
- `coreax.util.KernelComputeType`; use `Callable[[ArrayLike, ArrayLike], Array]` instead.
- `coreax.kernels.Kernel.calculate_kernel_matrix_row_{sum,mean}`; use `coreax.kernels.Kernel.gramian_row_mean`.
- `coreax.kernels.Kernel.updated_kernel_matrix_row_sum`; use `coreax.kernels.Kernel.gramian_row_mean` if possible.
- `coreax.data.DataReader` and `coreax.data.ArrayData`; use `coreax.data.Data` and `coreax.data.SupervisedData`.
- `coreax.refine.py` and `coreax.coresubset.py` removed; use `coreax.solvers.base.RefinementSolver` or
`coreax.solvers.base.CoresubsetSolver` to define coreset algorithms in `coreax.solvers.coresubset`.
- `coreax.reduction` removed, use `coreax.solvers.base.ExplicitSizeSolver` in place of `coreax.reduction.SizeReduce` and
`coreax.solvers.composite.MapReduce` in place of `coreax.reduction.MapReduce`. Use `coreax.coreset.Coreset` and
`coreax.coreset.Coresubset` in place of `coreax.reduction.Coreset`.

### Deprecated

- All uses of `coreax.weights.MMD` should be replaced with `coreax.weights.MMDWeightsOptimiser`.
- All uses of `coreax.weights.SBQ` should be replaced with `coreax.weights.SBQWeightsOptimiser`.
- All uses of `coreax.util.squared_distance_pairwise` should be replaced with `coreax.util.pairwise(squared_distance)`.
- All uses of `coreax.util.pairwise_difference` should be replaced with `coreax.util.pairwise(difference)`.


## [0.1.0]

### Added

- Base Coreax package using Object-Oriented Programming incorporating:
  - coreset methods: kernel herding, random sample
  - reduction strategies: size reduce, map reduce
  - kernels: squared exponential, Laplacian, PCIMQ, Stein
  - refinement: regular, reverse, random
  - metrics: MMD
  - approximations of kernel matrix row sum mean: random, ANNchor, Nystrom
  - weights optimisers: SBQ, MMD
  - score matching: sliced score matching, kernel density estimation
  - I/O: array data not requiring any preprocessing
- Near-complete unit test coverage.
- Example scripts for coreset generation, which may be called as integration tests.
- Bash script to run integration tests in sequence to avoid Jax errors.
- Detailed documentation for the Coreax package published to Read the Docs.
- README.md including an overview of what coresets are, setup instructions, a how-to
  guide, example applications and an overview of features coming soon.
- Support for Python 3.9-3.11.
- Project configuration and dependencies through pyproject.toml.
- Requirements files providing a pinned set of dependencies that are known to work for
  each supported Python version.
- Mark Coreax as typed.
- This changelog to make it easier for users and contributors to see precisely what
  notable changes have been made between each release of the project.
- FAQ.md to address any commonly asked questions.
- Contributor guidelines, code of conduct, license and security policy.
- Git configuration.
- GitHub Actions to run unit tests on Windows, macOS and Ubuntu for supported Python
  versions.
- Pre-commit checks to run the following, also checked by GitHub Actions:
  - black
  - isort
  - pylint
  - cspell spell check with custom dictionaries for library names, people and
    miscellaneous
  - pyroma
  - pydocstyle
  - assorted file format and encoding checks

### Deprecated

- Look-before-you-leap validation of all input to public functions

[//]: # (## [M.m.p])

[//]: # (### Added)
[//]: # (This is where features that have been added should be noted.)

[//]: # (### Fixed)
[//]: # (This is where fixes should be noted.)

[//]: # (### Changed)
[//]: # (This is where changes from previous versions should be noted.)

[//]: # (### Removed)
[//]: # (This is where elements which have been removed should be noted.)

[//]: # (### Deprecated)
[//]: # (This is where existing but deprecated elements should be noted.)

[Unreleased]: https://github.com/gchq/coreax/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/gchq/coreax/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/gchq/coreax/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/gchq/coreax/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/gchq/coreax/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/gchq/coreax/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/gchq/coreax/releases/tag/v0.1.0
