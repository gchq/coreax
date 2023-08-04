# Coreax
_© Crown Copyright GCHQ_

Coreax is a library for **coreset algorithms**, written in [Jax](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) for fast execution and GPU support. 

A coreset algorithm takes a $n \times d$ data set and reduces it to $m \ll n$ points whilst attempting to preserve the statistical properties of the full data set. Some algorithms return the $m$ points with weights, such that importance can be attributed to each point. These are often chosen from the simplex, i.e. such that they are non-negative and sum to 1.

## Quick example
Here are $n=10,000$ points drawn from $6$ $2$-D Gaussians. The coreset size, which we set, is $m=100$. Run `examples/weighted_herding.py` to replicate.
<p float="left">
<img src="examples/data/coreset_seq/coreset_seq.gif" width="40%"/>
<img src="examples/data/random_seq/random_seq.gif" width="40%"/>
</p>

The key property to observe is the maximum mean discrepancy (MMD) between the coreset and full set. This is an integral probability metric, which measures the distance between the empirical distributions of the full dataset and the coreset. For coreset algorithms, we would like this to be significantly smaller than random sampling (as above).

# Example applications
**Choosing pixels from an image**: In the below, we request ~20% of the original pixels. (Centre) 8000 coreset points chosen using Stein kernel herding, with point size a function of weight. (Right) 8000 points chosen randomly. Run `examples/david.py` to replicate.
<img src="examples/data/david_coreset.png" width="100%">

**Video event detection**: Here we identify representative frames such that most of the useful information in a video is preserved. Run `examples/pounce.py` to replicate.
<p float="left">
<img src="examples/pounce/pounce.gif" width="40%">
<img src="examples/pounce/pounce_coreset.gif" width="40%">
</p>

# Setup 
Be sure to install Jax, and to install the preferred version for your system.
1. Install [Jax](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html), noting that there are (currently) different setup paths for CPU and GPU use.
2. Install coreax from this directory `pip install .`

# A how-to guide
Here are some of the most commonly used functions in the library.

## Kernel herding
Basic kernel herding can be invoked using `kernel_herding_block` from `coreax.kernel_herding`. All the algorithms are variants of [kernel herding](https://arxiv.org/abs/1203.3472), and they will return indices into the original dataset. You can optionally compute weights to infer the importance of each coreset point. The "block" refers to block processing the Gram matrix to avoid GPU memory issues (there are "full matrix" versions of these functions, which will work if your GPU has enough memory to hold an $n \times n$ Gram matrix).

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
We have implemented a version of kernel herding that uses a **Stein kernel**, which targets [kernelised Stein discrepancy (KSD)](https://arxiv.org/abs/1602.03253) rather than MMD. This can often give better integration error in practice, but it can be slower than using a simpler kernel targetting MMD. To use Stein kernel herding, we have to define a continuous approximation to the discerete measure, e.g. using a KDE, or estimate the score function $\nabla \log f_X(\mathbf{x})$ of a continuous PDF from a finite set of samples. In this example, we use a Stein kernel with an inverse multi-quadric base kernel; computing the score function explicitly (score matching coming soon). Again, there are block versions for fitting within GPU memory constraints.
```python
from coreax.kernel import stein_kernel_pc_imq_element, rbf_grad_log_f_X
from coreax.kernel_herding import stein_kernel_herding_block

coreset, Kc, Kbar = stein_kernel_herding_block(X, m, stein_kernel_pc_imq_element, rbf_grad_log_f_X, nu=nu)
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
