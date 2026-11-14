
<img alt="Coreax logo" src="https://raw.githubusercontent.com/gchq/coreax/main/documentation/assets/Logo.svg" width="250px" height="auto">
<hr>

[![Unit Tests](https://github.com/gchq/coreax/actions/workflows/unittests.yml/badge.svg)](https://github.com/gchq/coreax/actions/workflows/unittests.yml)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fgchq%2Fcoreax-metadata%2Frefs%2Fheads%2Fmain%2Fcoverage%2Fcoreax_coverage.json)](https://github.com/gchq/coreax/actions/workflows/coverage.yml)
[![Pre-commit Checks](https://github.com/gchq/coreax/actions/workflows/pre_commit_checks.yml/badge.svg)](https://github.com/gchq/coreax/actions/workflows/pre_commit_checks.yml)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Python version](https://img.shields.io/pypi/pyversions/coreax.svg)](https://pypi.org/project/coreax)
[![PyPI](https://img.shields.io/pypi/v/coreax)](https://pypi.org/project/coreax)

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
(Centre) 8,000 coreset points chosen using kernel thinning.
(Right) 8,000 points chosen randomly.
Run `benchmark/david_benchmark.py` to  replicate.

|      Original      |      Coreset      |      Random      |
|:------------------:|:-----------------:|:----------------:|
| ![][DavidOriginal] | ![][DavidCoreset] | ![][DavidRandom] |

[DavidOriginal]: https://raw.githubusercontent.com/gchq/coreax/refs/heads/main/examples/data/david_original.png
[DavidCoreset]: https://raw.githubusercontent.com/gchq/coreax/refs/heads/main/examples/data/david_kt.png
[DavidRandom]: https://raw.githubusercontent.com/gchq/coreax/refs/heads/main/examples/data/david_random.png

### Video event detection

Here we identify representative frames such that most of the
useful information in a video is preserved.
Run `examples/pounce.py` to replicate.

|      Original       |      Coreset       |
|:-------------------:|:------------------:|
| ![][PounceOriginal] | ![][PounceCoreset] |

[PounceOriginal]: https://raw.githubusercontent.com/gchq/coreax/refs/heads/main/examples/pounce/pounce.gif
[PounceCoreset]: https://raw.githubusercontent.com/gchq/coreax/refs/heads/main/examples/pounce/pounce_coreset.gif


# Setup

Install Coreax from PyPI by adding `coreax` to your project dependencies or running
```shell
pip install coreax
```

Coreax uses JAX. It installs the CPU version by default, but if you have a GPU or TPU,
see the
[JAX installation instructions](https://jax.readthedocs.io/en/latest/installation.html)
for options available to take advantage of the power of your system. For example, if you
have an NVIDIA GPU on Linux, add `jax[cuda12]` to your project dependencies or run
```shell
pip install jax[cuda12]
```

There are additional [dependency groups](https://packaging.python.org/en/latest/specifications/dependency-groups/):
* `dev` includes all the tools and packages a developer of Coreax might need (includes all the below groups).
* `doc` is for compiling the Sphinx documentation;
* `tests` is used to run the tests (includes benchmark);
* `benchmark` is required to run benchmarking (includes example);
* `example` contains all dependencies for the example scripts;

Note that the `test` and `dev` dependencies include `opencv-python-headless`, which is
the headless version of OpenCV and is incompatible with other versions of OpenCV. If you
wish to use an alternative version, remove `opencv-python-headless` and select an
alternative from the
[OpenCV documentation](https://pypi.org/project/opencv-python-headless/).

Should the installation of Coreax fail, you can see the versions used by the Coreax
development team in `uv.lock`. You can transfer these to your own project as follows.
First, [install UV](https://docs.astral.sh/uv/getting-started/installation/). Then,
clone the repo from [GitHub](https://github.com/gchq/coreax). Next, run
```shell
uv export --format requirements-txt
```
which will generate a `requirements.txt`. Install this in your own project before trying
to install Coreax itself,
```shell
pip install -r requirements.txt
pip install coreax
```

# Release cycle

We anticipate two release types: feature releases and security releases. Security
releases will be issued as needed in accordance with the
[security policy](https://github.com/gchq/coreax/security/policy). Feature releases will
be issued as appropriate, dependent on the feature pipeline and development priorities.
