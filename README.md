<div align="center">
<img alt="Coreax logo" src="https://raw.githubusercontent.com/gchq/coreax/main/documentation/assets/Logo.svg">
</div>

# Coreax

[![Unit Tests and Code Coverage Assessment](https://github.com/gchq/coreax/actions/workflows/unittests.yml/badge.svg)](https://github.com/gchq/coreax/actions/workflows/unittests.yml)
[![Pre-commit Checks](https://github.com/gchq/coreax/actions/workflows/pre_commit_checks.yml/badge.svg)](https://github.com/gchq/coreax/actions/workflows/pre_commit_checks.yml)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Python version](https://img.shields.io/pypi/pyversions/coreax.svg)](https://pypi.org/project/coreax)
[![PyPI](https://img.shields.io/pypi/v/coreax)](https://pypi.org/project/coreax)
![Beta](https://img.shields.io/badge/pre--release-beta-red)

_© Crown Copyright GCHQ_

Coreax is a library for **coreset algorithms**, written in <a href="https://jax.readthedocs.io/en/latest/notebooks/quickstart.html" target="_blank">JAX</a> for fast execution and GPU support.

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

## Quick example
Consider $n=10,000$ points drawn from six $2$-D multivariate Gaussian distributions. We
wish to reduce this to only 100 points, whilst maintaining underlying statistical
properties. We achieve this by generating a coreset, setting $m=100$.
We plot the underlying data (blue) as-well as the coreset
points (red), which are plotted sequentially based on the order the algorithm selects
them in. The coreset points are weighted (size of point) to optimally reconstruct the
underlying distribution. Run `examples/herding_stein_weighted.py` to replicate.

We compare the coreset to the full original dataset by calculating the maximum
mean discrepancy (<a href="https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions#Measuring_distance_between_distributions" target="_blank">MMD</a>).
This key property is an integral probability metric, measuring
the distance between the empirical distributions of the full dataset and the coreset.
A good coreset algorithm produces a coreset that has significantly smaller MMD
than randomly sampling the same number of points from the original data, as is the case
in the example below.

|                                     Kernel herding                                      |                                     Random sample                                     |
|:---------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------:|
| ![](https://github.com/gchq/coreax/blob/main/examples/data/coreset_seq/coreset_seq.gif) | ![](https://github.com/gchq/coreax/blob/main/examples/data/random_seq/random_seq.gif) |


# Example applications
**Choosing pixels from an image**: In the example below, we reduce the original 180x215
pixel image (38,700 pixels in total) to a coreset approximately 20% of this size.
(Left) original image.
(Centre) 8,000 coreset points chosen using Stein kernel herding, with point size a
function of weight.
(Right) 8,000 points chosen randomly.
Run `examples/david_map_reduce_weighted.py` to  replicate.

![](https://github.com/gchq/coreax/blob/main/examples/data/david_coreset.png)


**Video event detection**: Here we identify representative frames such that most of the
useful information in a video is preserved.
Run `examples/pounce.py` to replicate.

|                                 Original                                 |                                     Coreset                                      |
|:------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|
| ![](https://github.com/gchq/coreax/blob/main/examples/pounce/pounce.gif) | ![](https://github.com/gchq/coreax/blob/main/examples/pounce/pounce_coreset.gif) |


# Setup
Before installing coreax, make sure JAX is installed. Be sure to install the preferred
version of JAX for your system.
1. Install [JAX](https://jax.readthedocs.io/en/latest/installation.html), noting that there are (currently) different setup paths for CPU and GPU use.
2. Install coreax by cloning the repo and then running `pip install .` from your local coreax directory.
3. To install additional optional dependencies required to run the examples in `examples` use `pip install .[test]` instead.

# A how-to guide
Here are some of the most commonly used classes and methods in the library.

## Kernel herding
Kernel herding is one (greedy) approach to coreset construction.
A `coreax.coresubset.KernelHerding` object is created by supplying a
`coreax.kernel.Kernel` object, such as a `SquaredExponentialKernel`. A coreset is
generated by calling the `fit` method on the kernel herding object.

Note that, throughout the codebase, there are block versions of herding for fitting
within memory constraints. These methods partition the data into blocks before carrying
out the coreset algorithm, restricting the maximum size of variables handled in the process.
```python
from sklearn.datasets import make_blobs
import numpy as np
import jax.random

from coreax import (
    ArrayData,
    KernelHerding,
    SizeReduce,
    SquaredExponentialKernel,
)
from coreax.kernel import median_heuristic

# Generate some data
num_data_points = 10_000
num_features = 2
num_cluster_centers = 6
random_seed = 1989
x, _ = make_blobs(
    num_data_points,
    n_features=num_features,
    centers=num_cluster_centers,
    random_state=random_seed,
)

# Request 100 coreset points
coreset_size = 100

# Setup the original data object
data = ArrayData.load(x)

# Set the bandwidth parameter of the kernel using a median heuristic derived from
# at most 1000 random samples in the data.
num_samples_length_scale = min(num_data_points, 1_000)
generator = np.random.default_rng(random_seed)
idx = generator.choice(num_data_points, num_samples_length_scale, replace=False)
length_scale = median_heuristic(x[idx])

# Compute a coreset using kernel herding with a squared exponential kernel.
herding_key = jax.random.key(random_seed)
herding_object = KernelHerding(
    herding_key, kernel=SquaredExponentialKernel(length_scale=length_scale)
)
herding_object.fit(
    original_data=data, strategy=SizeReduce(coreset_size=coreset_size)
)

# The herding object now has the coreset, and the indices of the original data
# that makeup the coreset as populated attributes
print(herding_object.coreset)
print(herding_object.coreset_indices)
```

## Kernel herding with weighting
A coreset can be weighted, a so-called **weighted coreset**, to attribute importance to
each point and to better approximate the underlying data distribution.
Optimal weights can be determined by implementing a
`coreax.weights.WeightsOptimiser`, such as the `MMDWeightsOptimiser`.
```python
from coreax import (
    KernelHerding,
    SizeReduce,
    SquaredExponentialKernel,
)
from coreax.weights import MMD as MMDWeightsOptimiser

# Define a kernel
kernel = SquaredExponentialKernel(length_scale=length_scale)

# Define a weights optimiser to learn optimal weights for the coreset after creation
weights_optimiser = MMDWeightsOptimiser(kernel=kernel)

# Compute a coreset using kernel herding with a squared exponential kernel.
herding_object = KernelHerding(
    herding_key,
    kernel=kernel,
    weights_optimiser=weights_optimiser
)
herding_object.fit(
    original_data=data, strategy=SizeReduce(coreset_size=coreset_size)
)

# Determine optimal weights for the coreset
herding_weights = herding_object.solve_weights()
```

## Kernel herding with refine
To improve the quality of a coreset, a **refine** step can be added.
These functions work by substituting points from the coreset with points from the
original dataset such that the MMD decreases. This improves the
coreset quality because the refined coreset better captures the
underlying distribution of the original data, as measured by the reduced MMD.

There are several different approaches to refining a coreset, which can be found in the
classes and methods in `coreax.refine`. In the example below, we create a refiner object,
pass it to the herding object, and then call the refine method.
```python
from coreax import (
    KernelHerding,
    SizeReduce,
    SquaredExponentialKernel,
)
from coreax.refine import RefineRegular

# Define a refinement object
refiner = RefineRegular()

# Compute a coreset using kernel herding with a squared exponential kernel.
herding_object = KernelHerding(
    herding_key,
    kernel=SquaredExponentialKernel(length_scale=length_scale),
    refine_method=refiner
)
herding_object.fit(
    original_data=data, strategy=SizeReduce(coreset_size=coreset_size)
)

# Refine the coreset to improve quality
herding_object.refine()

# The herding object now has the refined coreset, and the indices of the original
# data that makeup the refined coreset as populated attributes
print(herding_object.coreset)
print(herding_object.coreset_indices)
```

## Scalable herding
For large $n$ or $d$, you may run into time or memory issues. The class
`coreax.reduction.MapReduce` uses partitioning to tractably compute an approximate
coreset in reasonable time.
There is a necessary impact on coreset quality, for a dramatic improvement in computation time.
These methods can be used by simply replacing `coreax.reduction.SizeReduce` in the
previous examples with `MapReduce` and setting the parameter `leaf_size` in line with
memory requirements.

```python
from coreax.coresubset import KernelHerding
from coreax.kernel import SquaredExponentialKernel
from coreax.reduction import MapReduce

# Compute a coreset using kernel herding with a squared exponential kernel.
herding_object = KernelHerding(
    herding_key, kernel=SquaredExponentialKernel(length_scale=length_scale),
)
herding_object.fit(
    original_data=data,
    strategy=MapReduce(coreset_size=coreset_size, leaf_size=200)
)
```

For large $d$, it is usually worth reducing dimensionality using PCA. See `examples/pounce_map_reduce.py`
for an example.

## Stein kernel herding
We have implemented a version of kernel herding that uses a **Stein kernel**, which
targets [kernelised Stein discrepancy (KSD)](https://arxiv.org/abs/1602.03253) rather than MMD.
This can often give better integration error in practice, but it can be slower than
using a simpler kernel targeting MMD.
To use Stein kernel herding, we have to define a
continuous approximation to the discrete measure, e.g. using kernel density estimation (KDE),
or an estimate the score function $\nabla \log f_X(\mathbf{x})$ of a continuous PDF from
a finite set of samples.
In this example, we use a Stein kernel with a squared exponential base
kernel, computing the score function explicitly.
```python
import numpy as np

from coreax import (
    SquaredExponentialKernel,
    SteinKernel,
    KernelDensityMatching,
)

# Select indices to form a subset of data for learning score function
generator = np.random.default_rng(random_seed)
idx = generator.choice(len(data), subset_size, replace=False)
data_subset = data[idx, :]

# Learn a score function from the subset of the data, through a kernel density
# estimation applied to a subset of the data.
kernel_density_score_matcher = KernelDensityMatching(
    length_scale=length_scale, kde_data=data_subset
)
score_function = kernel_density_score_matcher.match()

# Define a kernel to use for herding
herding_kernel = SteinKernel(
    SquaredExponentialKernel(length_scale=length_scale),
    score_function=score_function,
)

# Compute a coreset using kernel herding with a Stein kernel
herding_object = KernelHerding(herding_key, kernel=herding_kernel)
herding_object.fit(
        original_data=data, strategy=SizeReduce(coreset_size=coreset_size)
    )
```

## Score matching
The score function, $\nabla \log f_X(\mathbf{x})$, of a distribution is the derivative
of the log-density function. This function is required when evaluating Stein kernels.
However, it can be difficult to specify analytically in practice.

To resolve this, we have implemented an approximation of the score function using a
neural network as in <a href="https://arxiv.org/abs/1905.07088" target="_blank">Song et al. (2019)</a>.
This approximate score function can then be passed directly to a Stein kernel, removing
any requirement for analytical derivation. More details on score matching methods
implemented are found in `coreax.score_matching`.
```python
import numpy as np

from coreax import (
    SteinKernel,
    SlicedScoreMatching,
)
from coreax.kernel import PCIMQKernel

# Select indices to form a subset of data for learning score function
generator = np.random.default_rng(random_seed)
idx = generator.choice(len(data), subset_size, replace=False)
data_subset = data[idx, :]

# Learn a score function from a subset of the data, through approximation using a neural
# network applied to a subset of the data
score_key = jax.random.key(random_seed)
sliced_score_matcher = SlicedScoreMatching(
    score_key,
    random_generator=jax.random.rademacher,
    use_analytic=True,
    num_epochs=10,
    num_random_vectors=1,
    sigma=1.0,
    gamma=0.95,
)
score_function = sliced_score_matcher.match(data_subset)

# Define a kernel to use for herding
herding_kernel = SteinKernel(
    PCIMQKernel(length_scale=length_scale),
    score_function=score_function,
)
```

# Release cycle
We anticipate two release types: feature releases and security releases. Security
releases will be issued as needed in accordance with the
[security policy](https://github.com/gchq/coreax/security/policy). Feature releases will
be issued as appropriate, dependent on the feature pipeline and development priorities.

# Coming soon
Some features coming soon include:
- Coordinate bootstrapping for high-dimensional data.
- Other coreset-style algorithms, including kernel thinning and recombination, as means
to reducing a large dataset whilst maintaining properties of the underlying distribution.
