# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Badge to README to show code coverage percentage.
- Support for Python 3.12.
- Additional data classes `coreax.data.Data` and `coreax.data.SupervisedData` that draw
distinction between supervised and unsupervised datasets, and handle weighted data.
- Additional kernel classes `coreax.kernel.AdditiveKernel` and
`coreax.kernel.ProductKernel` that allow for arbitrary composition of positive
semi-definite kernels to produce new positive semi-definite kernels.
- Added kernel classes `coreax.kernel.Linear`, `coreax.kernel.Polynomial`, `coreax.kernel.RationalQuadratic`,
`coreax.kernel.Polynomial`, `coreax.kernel.Periodic`, `coreax.kernel.LocallyPeriodic`

### Fixed

- Wording improvements in README.
- Documentation now builds without warnings.
- GitHub workflow runs automatically after Pre-commit autoupdate.
- `coreax.kernel.Kernel.length_scale` and `coreax.kernel.Kernel.output_scale` are treated as dynamic elements of the kernel pytree.

### Changed

- Documentation has been rearranged.
- Renamed `coreax.weights.MMD` to `coreax.weights.MMDWeightsOptimiser` and added deprecation warning.
- Renamed `coreax.weights.SBQ` to `coreax.weights.SBQWeightsOptimiser` and added deprecation warning.
- `requirements-*.txt` will no longer be updated frequently, thereby providing stable versions.
- Single requirements files covering all supported Python versions.
- All references to `kernel_matrix_row_{sum,mean}` have been replaced with `Gramian row-mean`.

### Removed
- Bash script to run integration tests has been removed. `pytest tests/integration` should now work as expected.
- Tests for `coreax.kernels.Kernel.{calculate, update}_kernel_matrix_row_sum`.
- `coreax.util.KernelComputeType`; use `Callable[[ArrayLike, ArrayLike], Array]` instead.
- `coreax.kernels.Kernel.calculate_kernel_matrix_row_{sum,mean}`; use `coreax.kernels.Kernel.gramian_row_mean`.
- `coreax.kernels.Kernel.updated_kernel_matrix_row_sum`; use `coreax.kernels.Kernel.gramian_row_mean` if possible.


### Deprecated

- All uses of `coreax.weights.MMD` should be replaced with `coreax.weights.MMDWeightsOptimiser`.
- All uses of `coreax.weights.SBQ` should be replaced with `coreax.weights.SBQWeightsOptimiser`.


## [0.1.0] - 2024-02-16

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

[//]: # (## [M.m.p] - YYYY-MM-DD)

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

[Unreleased]: https://github.com/gchq/coreax/compare/v0.1.0...HEAD
[//]: # ([0.1.1]: https://github.com/gchq/coreax/compare/v0.1.1...v0.1.0)
[0.1.0]: https://github.com/gchq/coreax/releases/tag/v0.1.0
