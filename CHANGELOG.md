# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Fixed

### Changed

### Removed

### Deprecated

## [0.1.0-beta] - 2024-02-16

### Added

- Base Coreax package using Object-Oriented Programming.
- Near-complete unit test coverage.
- Example scripts for coreset generation, which may be called as integration tests.
- Bash script to run integration tests in sequence to avoid Jax errors.
- Detailed documentation for the Coreax package published to Read the Docs.
- README.md including an overview of what coresets are, setup instructions, a how-to guide, example applications and features coming soon.
- Support for Python 3.9-3.11.
- Project configuration and dependencies through pyproject.toml.
- Requirements files providing a pinned set of dependencies that are known to work for each supported Python version.
- This changelog to make it easier for users and contributors to see precisely what notable changes have been made between each release of the project.
- FAQ.md to address any commonly asked questions.
- Contributor guidelines, code of conduct and license.
- GitHub Actions to run unit tests on Windows, macOS and Ubuntu for supported Python versions.
- Pre-commit checks to run the following, also checked by GitHub Actions:
  - black
  - isort
  - pylint
  - cspell spell check with custom dictionaries for library names, people and miscellaneous
  - pyroma
  - pydocstyle
  - assorted file format checks

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

[Unreleased]: https://github.com/gchq/coreax/compare/v0.1.0-beta...HEAD
[//]: # ([0.1.0]: https://github.com/gchq/coreax/compare/v0.1.0-beta...v0.1.0)
[0.1.0-beta]: https://github.com/gchq/coreax/releases/tag/v0.1.0-beta
