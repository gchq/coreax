# Coreax

[![Unit Tests](https://github.com/gchq/coreax/actions/workflows/unittests.yml/badge.svg)](https://github.com/gchq/coreax/actions/workflows/unittests.yml)
[![Pre-commit Checks](https://github.com/gchq/coreax/actions/workflows/pre_commit_checks.yml/badge.svg)](https://github.com/gchq/coreax/actions/workflows/pre_commit_checks.yml)
[![Code Coverage Assessment](https://github.com/gchq/coreax/actions/workflows/code_coverage_assessment.yml/badge.svg)](https://github.com/gchq/coreax/actions/workflows/code_coverage_assessment.yml)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

_© Crown Copyright GCHQ_

Coreax is a library for **coreset algorithms**, written in [Jax](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) for fast execution and GPU support.

A coreset algorithm takes a $n \times d$ data set and reduces it to $m \ll n$ points whilst attempting to preserve the statistical properties of the full data set. Some algorithms return the $m$ points with weights, such that importance can be attributed to each point. These are often chosen from the simplex, i.e. such that they are non-negative and sum to 1.

## Quick example
Here are $n=10,000$ points drawn from six $2$-D multivariate Gaussian distributions. The coreset size, which we set, is $m=100$. Run `examples/weighted_herding.py` to replicate.

![](examples/data/coreset_seq/coreset_seq.gif)
![](examples/data/random_seq/random_seq.gif)

The key property to observe is the maximum mean discrepancy (MMD) between the coreset and full set. This is an integral probability metric, which measures the distance between the empirical distributions of the full dataset and the coreset. For coreset algorithms, we would like this to be significantly smaller than random sampling (as above).

# Example applications
**Choosing pixels from an image**: In the below, we request ~20% of the original pixels. (Centre) 8000 coreset points chosen using Stein kernel herding, with point size a function of weight. (Right) 8000 points chosen randomly. Run `examples/david.py` to replicate.

![](examples/data/david_coreset.png)


**Video event detection**: Here we identify representative frames such that most of the useful information in a video is preserved. Run `examples/pounce.py` to replicate.

![](examples/pounce/pounce.gif)
![](examples/pounce/pounce_coreset.gif)


# Setup
Be sure to install Jax, and to install the preferred version for your system.
1. Install [Jax](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html), noting that there are (currently) different setup paths for CPU and GPU use.
2. Install coreax from this directory `pip install .`

# A how-to guide
Here are some of the most commonly used functions in the library.

## Kernel herding
Basic kernel herding can be invoked using `kernel_herding_block` from `coreax.kernel_herding`. All the algorithms are currently variants of [kernel herding](https://arxiv.org/abs/1203.3472), and they will return indices into the original dataset. You can optionally compute weights to infer the importance of each coreset point. The "block" refers to block processing the Gram matrix to avoid GPU memory issues (there are "full matrix" versions of these functions, which will work if your GPU has enough memory to hold an $n \times n$ Gram matrix).

```python
from coreax.kernel_herding import kernel_herding_block
from coreax.kernel import rbf_kernel, median_heuristic
from coreax.weights import qp
from coreax.metrics import mmd_weight_block

from sklearn.datasets import make_blobs
import jax.numpy as jnp
import numpy as np

# create some data
n = 10000
m = 100
d = 2
X, _ = make_blobs(n, n_features=d, centers=6, random_state=32)

# choose the base kernel bandwidth using a median heuristic
N = min(X.shape[0], 1000)
idx = np.random.choice(X.shape[0], N, replace=False)
nu = median_heuristic(X[idx])

# define a kernel
k = lambda x, y : rbf_kernel(x, y, jnp.float32(nu)**2)/(nu * jnp.sqrt(2. * jnp.pi))

# get a coreset and some weights
coreset, Kc, Kbar = kernel_herding_block(X, m, k)
weights = qp(Kc + 1e-10, Kbar)

# assess performance using MMD
mmd = mmd_weight_block(X, X[coreset], jnp.ones(n), weights, k)
print("MMD: %.6f" % mmd)
```

## Kernel herding with refine
A **refine** step can be added to kernel herding to improve performance. These functions post-process the coreset swapping out points with from the original dataset that improve directly the coreset's MMD. See the functions in `coreax.kernel_herding_refine`. In the above example, we can simply replacing the `kernel_herding_block` function with
```python
from coreax.kernel_herding_refine import kernel_herding_refine_block
coreset = kernel_herding_refine_block(X, m, k):
```

## Stein kernel herding
We have implemented a version of kernel herding that uses a **Stein kernel**, which targets [kernelised Stein discrepancy (KSD)](https://arxiv.org/abs/1602.03253) rather than MMD. This can often give better integration error in practice, but it can be slower than using a simpler kernel targeting MMD. To use Stein kernel herding, we have to define a continuous approximation to the discrete measure, e.g. using a KDE, or estimate the score function $\nabla \log f_X(\mathbf{x})$ of a continuous PDF from a finite set of samples. In this example, we use a Stein kernel with an inverse multi-quadric base kernel; computing the score function explicitly (score matching coming soon). Again, there are block versions for fitting within memory constraints.
```python
from coreax.kernel import stein_kernel_pc_imq_element, rbf_grad_log_f_x
from coreax.kernel_herding import stein_kernel_herding_block

coreset, Kc, Kbar = stein_kernel_herding_block(X, m, stein_kernel_pc_imq_element, rbf_grad_log_f_x, nu=nu)
```

## Scalable herding
For large $n$ or $d$, you may run into time or memory issues. `scalable_herding` in `coreax.kernel_herding` uses partitioning to tractably compute an approximate coreset in reasonable time. There is a necessary impact on coreset quality, but we get dramatic improvement in computation.

For some idea of performance, `scalable_herding` on a `ml.p3.8xlarge` gives the following (post-JIT) timings for 2D data with an RBF kernel.
```
|n   | Time |
|----|------|
|10^5|5s    |
|10^6|40s   |
|10^7|7m35s |
|10^8|1hr7m |
```
For large $d$, it is usually worth reducing dimensionality using PCA.

# Release cycle
We anticipate two release types: feature releases and security releases. Security
releases will be issued as needed in accordance with the
[security policy](https://github.com/gchq/coreax/security/policy). Feature releases will
be issued as appropriate, dependent on the feature pipeline and development priorities.

# Coming soon
Some of the features coming soon include
- Score matching to estimate $\nabla \log f_X(\mathbf{x})$ in the Stein kernel.
- Coordinate bootstrapping for high-dimensional data.
- Other coreset-style algorithms, including kernel thinning and recombination.
