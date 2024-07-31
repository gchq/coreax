<div align="center">
    <img alt="Coreax logo" src="https://raw.githubusercontent.com/gchq/coreax/main/documentation/assets/Logo.svg">
</div>

# Coreax

[![Unit Tests](https://github.com/gchq/coreax/actions/workflows/unittests.yml/badge.svg)](https://github.com/gchq/coreax/actions/workflows/unittests.yml)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Ftp832944%2F51dd332be75961a7dc903c67718028e1%2Fraw%2Fcoreax_coverage.json)](https://github.com/gchq/coreax/actions/workflows/coverage.yml)
[![Pre-commit Checks](https://github.com/gchq/coreax/actions/workflows/pre_commit_checks.yml/badge.svg)](https://github.com/gchq/coreax/actions/workflows/pre_commit_checks.yml)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Python version](https://img.shields.io/pypi/pyversions/coreax.svg)](https://pypi.org/project/coreax)
[![PyPI](https://img.shields.io/pypi/v/coreax)](https://pypi.org/project/coreax)
![Beta](https://img.shields.io/badge/pre--release-beta-red)

_© Crown Copyright GCHQ_

Coreax is a library for **coreset algorithms**, written in <a href="https://jax.readthedocs.io/en/latest/notebooks/quickstart.html" target="_blank">JAX</a> for fast execution and GPU support.

## About Coresets

For $n$ points in $d$ dimensions, a coreset algorithm takes an $n \times d$ data set and
reduces it to $m \ll n$ points whilst attempting to preserve the statistical properties
of the full data set. The algorithm maintains the dimension of the original data set.
Thus the $m$ points, referred to as the **coreset**, are also $d$-dimensional.

The $m$ points need not be in the original data set. We refer to the special case where
all selected points are in the original data set as a **coresubset**.

Some algorithms return the $m$ points with weights, so that importance can be
attributed to each point in the coreset. The weights, $w_i$ for $i=1,...,m$, are often
chosen from the simplex. In this case, they are non-negative and sum to 1:
$w_i >0$ $\forall i$ and $\sum_{i} w_i =1$.

Please see [the documentation](https://coreax.readthedocs.io/en/latest/quickstart.html) for some in-depth examples.


##  Example applications

### Choosing pixels from an image

In the example below, we reduce the original 180x215
pixel image (38,700 pixels in total) to a coreset approximately 20% of this size.
(Left) original image.
(Centre) 8,000 coreset points chosen using Stein kernel herding, with point size a
function of weight.
(Right) 8,000 points chosen randomly.
Run `examples/david_map_reduce_weighted.py` to  replicate.

![](https://raw.githubusercontent.com/gchq/coreax/main/examples/data/david_coreset.png)


### Video event detection

Here we identify representative frames such that most of the
useful information in a video is preserved.
Run `examples/pounce.py` to replicate.

|                                 Original                                 |                                     Coreset                                      |
|:------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|
| ![](https://raw.githubusercontent.com/gchq/coreax/main/examples/pounce/pounce.gif) | ![](https://raw.githubusercontent.com/gchq/coreax/main/examples/pounce/pounce_coreset.gif) |


# Setup
Before installing coreax, make sure JAX is installed. Be sure to install the preferred
version of JAX for your system.

Install [JAX](https://jax.readthedocs.io/en/latest/installation.html) noting that there
are (currently) different setup paths for CPU and GPU use:
```shell
$ python3 -m pip install jax
```

Install Coreax:
```shell
$ python3 -m pip install coreax
```

Optionally, install additional dependencies required to run the examples:
```shell
$ python3 -m pip install coreax[test]
```

Should the installation fail, try again using stable pinned package versions. Note that
these versions may be rather outdated, although we endeavour to avoid versions with
known vulnerabilities. To install Coreax:
```shell
$ python3 -m pip install --no-dependencies -r requirements.txt
```

To run the examples, use `requirements-test.txt` instead.

# Release cycle
We anticipate two release types: feature releases and security releases. Security
releases will be issued as needed in accordance with the
[security policy](https://github.com/gchq/coreax/security/policy). Feature releases will
be issued as appropriate, dependent on the feature pipeline and development priorities.

# Coming soon
Some features coming soon include:
* Coordinate bootstrapping for high-dimensional data.
* Other coreset-style algorithms, including recombination, as means
to reducing a large dataset whilst maintaining properties of the underlying distribution.
