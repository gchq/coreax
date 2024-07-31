# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added Kernelised Stein Discrepancy divergence in `coreax.metrics.KSD`.
- Added the `coreax.solvers.recombination` module, which provides the following new solvers:
  - `RecombinationSolver`: an abstract base class for recombination solvers.
  - `CaratheodoryRecombination`: a simple deterministic approach to solving recombination problems.
  - `TreeRecombination`: an advanced deterministic approach that utilises `CaratheodoryRecombination`,
    but provides superior performance for solving all but the smallest recombination problems.

### Fixed



### Changed
- Refactored `coreax.inverses.py` functionality into `coreax.least_squares.py`:
  - `coreax.inverses.RegularisedInverseApproximator` replaced by `coreax.least_squares.RegularisedLeastSquaresSolver`
  - `coreax.inverses.LeastSquaresApproximator` replaced by `coreax.least_squares.MinimalEuclideanNormSolver`
  - `coreax.inverses.RandomisedEigendecompositionApproximator` replaced by `coreax.least_squares.RandomisedEigendecompositionSolver`



### Removed



### Deprecated




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

[Unreleased]: https://github.com/gchq/coreax/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/gchq/coreax/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/gchq/coreax/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/gchq/coreax/releases/tag/v0.1.0
