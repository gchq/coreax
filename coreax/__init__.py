# © Crown Copyright GCHQ
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Coreax library for generation of compressed representations of datasets.

The coreax library contains code to address the following generic problem. Given an
:math:`n \times d` dataset, generate a :math:`m \times d` dataset, with :math:`m << n`
such that the generated dataset contains as much of the information from the original
dataset as possible. The generated dataset is often called a coreset.

To improve the speed of computations, various parts of the codebase utilise JAX's
implementation of JIT compilation. One example of this can be seen in
:meth:`~coreax.kernel.Kernel.compute`. Whilst this significantly increases performance
of the code, care needs to be taken when using JIT decorators on class methods. Whenever
such an instance occurs, one must also define the methods :meth:`_tree_unflatten` and
:meth:`_tree_flatten`, or the JIT functionality on the class will not operate. These
methods handle flattening and constructing the pytree objects that JAX depends on for
computations.

Pytrees are tree-like structures built out of container-like Python objects. The
'leaves' of a pytree correspond to data we wish to perform operations on. Many JAX
functions operate on pytrees, that is take in and output pytrees internally. Further
detail of pytrees and their usage in JAX can be found here:
https://jax.readthedocs.io/en/latest/pytrees.html.

Inside :meth:`_tree_unflatten`, one must define arrays & dynamic values (children) and
auxiliary data (static values) of the class. Inside :meth:`_tree_flatten`, one can pass
these children and auxiliary data to the class. See :class:`@coreax.kernel.Kernel` and
children of this object for example implementations of this.

Further details on pytrees in JAX can be found at
https://jax.readthedocs.io/en/latest/pytrees.html and
https://jax.readthedocs.io/en/latest/faq.html#how-to-use-jit-with-methods.

Performance tests are implemented in tests/performance/ that verify the code is in-fact
faster when using JIT compilation, and should be updated as new JIT functionality is
included in the codebase.

Warning about integration tests using Jax: where multiple unit tests are run from a
parent process, Jax is likely to re-use compiled tracers if present. This can result in
errors where data types within the tracers differ (as they are likely to do in
integration tests). Unfortunately, this doesn't seem to be rectified by multiprocessing,
so a simple method to run a batch of tests from separate parent processes is to run or
loop over them individually. We currently do this using
``tests/integration/test_run.sh``.
"""

__version__ = "0.1.0"

# pylint: disable=unused-import
from coreax.coresubset import KernelHerding, RandomSample
from coreax.data import ArrayData
from coreax.kernel import SquaredExponentialKernel, SteinKernel
from coreax.metrics import MMD
from coreax.reduction import MapReduce, SizeReduce
from coreax.score_matching import KernelDensityMatching, SlicedScoreMatching

# pylint: enable=unused-import
