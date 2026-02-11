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

from coreax.solvers.base import (
    CoresubsetSolver,
    ExplicitSizeSolver,
    PaddingInvariantSolver,
    PseudoRefinementSolver,
    PseudoSolver,
    RefinementSolver,
    Solver,
)
from coreax.solvers.composite import CompositeSolver, MapReduce
from coreax.solvers.coresubset import (
    CompressPlusPlus,
    GreedyKernelPoints,
    GreedyKernelPointsState,
    HerdingState,
    KernelHerding,
    KernelThinning,
    RandomSample,
    RPCholesky,
    RPCholeskyState,
    SteinThinning,
)
from coreax.solvers.gradient import GradientFlow, JointKernelInducingPoints
from coreax.solvers.recombination import (
    CaratheodoryRecombination,
    RecombinationSolver,
    TreeRecombination,
)

__all__ = [
    "Solver",
    "CoresubsetSolver",
    "RefinementSolver",
    "ExplicitSizeSolver",
    "GradientFlow",
    "JointKernelInducingPoints",
    "PaddingInvariantSolver",
    "PseudoSolver",
    "PseudoRefinementSolver",
    "CompositeSolver",
    "MapReduce",
    "RandomSample",
    "HerdingState",
    "KernelHerding",
    "KernelThinning",
    "SteinThinning",
    "RPCholeskyState",
    "RPCholesky",
    "GreedyKernelPointsState",
    "GreedyKernelPoints",
    "RecombinationSolver",
    "CaratheodoryRecombination",
    "TreeRecombination",
    "CompressPlusPlus",
]
