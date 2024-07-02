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

"""

__version__ = "0.2.1"

# pylint: disable=unused-import
from coreax.approximation import (
    ANNchorApproximateKernel,
    ApproximateKernel,
    MonteCarloApproximateKernel,
    NystromApproximateKernel,
)
from coreax.coreset import Coreset, Coresubset
from coreax.data import Data, SupervisedData
from coreax.kernel import (
    CompositeKernel,
    Kernel,
    LaplacianKernel,
    PCIMQKernel,
    SquaredExponentialKernel,
    SteinKernel,
)
from coreax.metrics import MMD
from coreax.score_matching import KernelDensityMatching, SlicedScoreMatching
# pylint: enable=unused-import
