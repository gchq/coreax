# Coreax

[![Unit Tests and Code Coverage Assessment](https://github.com/gchq/coreax/actions/workflows/unittests.yml/badge.svg)](https://github.com/gchq/coreax/actions/workflows/unittests.yml)
[![Pre-commit Checks](https://github.com/gchq/coreax/actions/workflows/pre_commit_checks.yml/badge.svg)](https://github.com/gchq/coreax/actions/workflows/pre_commit_checks.yml)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

_© Crown Copyright GCHQ_

Coreax is a library for **coreset algorithms**, written in [Jax](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) for fast execution and GPU support.

A coreset algorithm takes a $n \times d$ data set and reduces it to $m \ll n$ points
whilst attempting to preserve the statistical properties of the full data set.
Some algorithms return the $m$ points with weights, such that importance can be
attributed to each point. These are often chosen from the simplex, i.e. such that they
are non-negative and sum to 1.

## Quick example
Here are $n=10,000$ points drawn from six $2$-D multivariate Gaussian distributions. The
coreset size, which we set, is $m=100$.
A coreset is generated, and we plot both the underlying data, as-well as the coreset
points that have been weighted optimally to reconstruct the underling distribution.
Run `examples/herding_stein_weighted.py` to replicate.

![](examples/data/coreset_seq/coreset_seq.gif)
![](examples/data/random_seq/random_seq.gif)

The key property to observe is the maximum mean discrepancy (MMD) between the coreset
and full dataset.
This is an integral probability metric, which measures the distance between the
empirical distributions of the full dataset and the coreset.
For coreset algorithms, we would like this to be significantly smaller than random
sampling (as above).

# Example applications
**Choosing pixels from an image**: In the below, we request ~20% of the original pixels.
(Left) original image.
(Centre) 8000 coreset points chosen using Stein kernel herding, with point size a
function of weight.
(Right) 8000 points chosen randomly.
Run `examples/david_map_reduce_weighted.py` to  replicate.

![](examples/data/david_coreset.png)


**Video event detection**: Here we identify representative frames such that most of the
useful information in a video is preserved.
Run `examples/pounce.py` to replicate.

![](examples/pounce/pounce.gif)
![](examples/pounce/pounce_coreset.gif)


# Setup
Be sure to install Jax, and to install the preferred version for your system.
1. Install [Jax](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html), noting that there are (currently) different setup paths for CPU and GPU use.
2. Install coreax from this directory `pip install .`

# A how-to guide
Here are some of the most commonly used classes and methods in the library.

## Kernel herding
Kernel herding is one (greedy) approach to coreset construction.
A `coreax.coresubset.KernelHerding` object can be created by supplying a
`coreax.kernel.Kernel`, and a coreset can be generated by calling the `fit` method on
this object.
Throughout the codebase, there are block versions of herding for fitting within memory
constraints.
```python
from sklearn.datasets import make_blobs
import numpy as np

from coreax.coresubset import KernelHerding
from coreax.data import ArrayData
from coreax.kernel import SquaredExponentialKernel, median_heuristic
from coreax.reduction import SizeReduce

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
generator = np.random.default_rng(1_989)
idx = generator.choice(num_data_points, num_samples_length_scale, replace=False)
length_scale = median_heuristic(x[idx])

# Compute a coreset using kernel herding with a Squared exponential kernel.
herding_object = KernelHerding(
    kernel=SquaredExponentialKernel(length_scale=length_scale)
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
Coresets can be weighted such that the weighted coreset better approximates the
underlying data distribution.
Optimal weights can be determined by using any implementation of
`coreax.weights.WeightsOptimiser`.
```python
from coreax.coresubset import KernelHerding
from coreax.kernel import SquaredExponentialKernel
from coreax.reduction import SizeReduce
from coreax.refine import RefineRegular
from coreax.weights import MMD as MMDWeightsOptimiser

# Define a kernel
kernel = SquaredExponentialKernel(length_scale=length_scale)

# Define a weights optimiser to learn optimal weights for the coreset after creation
weights_optimiser = MMDWeightsOptimiser(kernel=kernel)

# Compute a coreset using kernel herding with a Squared exponential kernel.
herding_object = KernelHerding(
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
A **refine** step can be added to kernel herding to improve performance.
These functions post-process the coreset swapping out points with from the original
dataset that directly improve the coreset's MMD. See the classes and methods in
`coreax.refine`. In the above example, we can simply define a Refiner object, pass it to
the herding object, and then call the refine method.
```python
from coreax.coresubset import KernelHerding
from coreax.kernel import SquaredExponentialKernel
from coreax.reduction import SizeReduce
from coreax.refine import RefineRegular

# Define a refinement object
refiner = RefineRegular()

# Compute a coreset using kernel herding with a Squared exponential kernel.
herding_object = KernelHerding(
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

## Stein kernel herding
We have implemented a version of kernel herding that uses a **Stein kernel**, which
targets [kernelised Stein discrepancy (KSD)](https://arxiv.org/abs/1602.03253) rather than MMD.
This can often give better integration error in practice, but it can be slower than
using a simpler kernel targeting MMD.
To use Stein kernel herding, we have to define a
continuous approximation to the discrete measure, e.g. using a KDE, or estimate the
score function $\nabla \log f_X(\mathbf{x})$ of a continuous PDF from a finite set of
samples.
In this example, we use a Stein kernel with an inverse multi-quadric base
kernel; computing the score function explicitly.
```python
from coreax.kernel import SteinKernel, SquaredExponentialKernel
from coreax.score_matching import KernelDensityMatching

# Learn a score function from the a subset of the data, through a kernel density
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
```

## Scalable herding
For large $n$ or $d$, you may run into time or memory issues.
`coreax.reduction.MapReduce` uses partitioning to tractably compute an approximate
coreset in reasonable time.
There is a necessary impact on coreset quality, but we get dramatic improvement in
computation.
Map reduce can be used by simply replacing `coreax.reduction.SizeReduce` in the
previous examples.
```python
from coreax.coresubset import KernelHerding
from coreax.kernel import SquaredExponentialKernel
from coreax.reduction import MapReduce

# Compute a coreset using kernel herding with a Squared exponential kernel.
herding_object = KernelHerding(
    kernel=SquaredExponentialKernel(length_scale=length_scale),
)
herding_object.fit(
    original_data=data,
    strategy=MapReduce(coreset_size=coreset_size, leaf_size=20)
)
```
For large $d$, it is usually worth reducing dimensionality using PCA.

## Score matching
The score function, $\nabla \log f_X(\mathbf{x})$, of a distribution is the derivative
of the log-density function. This function is required when evaluating Stein kernels.
However, it may be difficult to analytically specify in practice. To avoid this, we have
implemented {cite:p}`ssm` to approximate the
score function with a neural network. See `coreax.score_matching` for implementations.
This approximation to the true score function can then be passed directly to a Stein
kernel, removing any requirement for the analytical derivation.
```python
from jax.random import rademacher

from coreax.kernel import PCIMQKernel, SteinKernel
from coreax.score_matching import SlicedScoreMatching

# Learn a score function from a subset of the data, through approximation using a neural
# network applied to a subset of the data
sliced_score_matcher = SlicedScoreMatching(
    random_generator=rademacher,
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
Some of the features coming soon include
- Coordinate bootstrapping for high-dimensional data.
- Other coreset-style algorithms, including kernel thinning and recombination.
