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

"""Solvers for generating coresets."""

from coreax.kernels.base import (
    AdditiveKernel,
    DuoCompositeKernel,
    PowerKernel,
    ProductKernel,
    ScalarValuedKernel,
    UniCompositeKernel,
)
from coreax.kernels.scalar_valued import (
    ExponentialKernel,
    LaplacianKernel,
    LinearKernel,
    LocallyPeriodicKernel,
    MaternKernel,
    PCIMQKernel,
    PeriodicKernel,
    PoissonKernel,
    PolynomialKernel,
    RationalQuadraticKernel,
    SquaredExponentialKernel,
    SteinKernel,
)
from coreax.kernels.util import median_heuristic

__all__ = [
    "ScalarValuedKernel",
    "ExponentialKernel",
    "LaplacianKernel",
    "LinearKernel",
    "LocallyPeriodicKernel",
    "PCIMQKernel",
    "PeriodicKernel",
    "PolynomialKernel",
    "RationalQuadraticKernel",
    "SquaredExponentialKernel",
    "AdditiveKernel",
    "ProductKernel",
    "SteinKernel",
    "median_heuristic",
    "DuoCompositeKernel",
    "UniCompositeKernel",
    "PowerKernel",
    "PoissonKernel",
    "MaternKernel",
]
