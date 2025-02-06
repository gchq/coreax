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

"""Abstract base classes for constructing different types of coreset solvers."""

from abc import abstractmethod
from typing import Generic, Optional, TypeVar

import equinox as eqx

from coreax.coreset import AbstractCoreset, Coresubset
from coreax.data import Data, SupervisedData

_Data = TypeVar("_Data", Data, SupervisedData)
_Coreset = TypeVar("_Coreset", bound=AbstractCoreset)
_State = TypeVar("_State")


class Solver(eqx.Module, Generic[_Coreset, _Data, _State]):
    """
    Base class for coreset solvers.

    Solver is generic on the type of data required by the reduce method, and the type of
    coreset returned, providing a convenient means to distinguish between solvers that
    take (weighted) data/supervised data, and those which produce coresets/coresubsets.
    """

    @abstractmethod
    def reduce(
        self, dataset: _Data, solver_state: Optional[_State] = None
    ) -> tuple[_Coreset, _State]:
        r"""
        Reduce 'dataset' to a coreset - solve the coreset problem.

        :param dataset: The (potentially weighted and supervised) data to generate the
            coreset from
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step information
        :return: a tuple of the solved coreset and intermediate solver state information
        """


class CoresubsetSolver(
    Solver[Coresubset[_Data], _Data, _State], Generic[_Data, _State]
):
    """
    Solver which returns a :class:`coreax.coreset.Coresubset`.

    A convenience class for the most common solver type in this package.
    """


class RefinementSolver(CoresubsetSolver[_Data, _State], Generic[_Data, _State]):
    """
    A :class:`~coreax.solvers.CoresubsetSolver` which supports refinement.

    Some solvers assume implicitly/explicitly an initial coresubset on which the
    solution is dependent. Such solvers can be interpreted as refining the initial
    coresubset to produce another (solution) coresubset.

    By providing a 'refine' method, one can compose the results of different solvers
    together, and/or repeatedly apply/chain the result of a refinement based solve.

    .. code-block:: python

        # An example of repeated application/chaining of solutions/solvers.
        result, state = solver.reduce(dataset)
        refined_result, state = refine_solver.refine(result, state)
        re_refined_result, state = refine_solver.refine(refined_result, state)
    """

    @abstractmethod
    def refine(
        self, coresubset: Coresubset[_Data], solver_state: Optional[_State] = None
    ) -> tuple[Coresubset[_Data], _State]:
        """
        Refine a coresubset - swap/update coresubset indices.

        :param coresubset: The coresubset to refine
        :param solver_state: Solution state information, primarily used to cache
            expensive intermediate solution step values.
        :return: a refined coresubset and relevant intermediate solver state information
        """


class ExplicitSizeSolver(
    Solver[_Coreset, _Data, _State], Generic[_Coreset, _Data, _State]
):
    """
    A :class:`Solver` which produces a coreset of an explicitly specified size.

    :param coreset_size: The desired size of the solved coreset
    """

    coreset_size: int = eqx.field(converter=int)

    def __check_init__(self):
        """Check that 'coreset_size' is feasible."""
        if self.coreset_size <= 0:
            raise ValueError("'coreset_size' must be a positive integer")


class PaddingInvariantSolver(
    Solver[_Coreset, _Data, _State], Generic[_Coreset, _Data, _State]
):
    """
    A :class:`Solver` whose results are invariant to zero weighted data.

    In some cases, such as in :class:`coreax.solvers.MapReduce`, there is a need to pad
    data to ensure shape stability. In these cases, we may assign zero weight to the
    padded data points, which allows certain 'padding invariant' solvers to return the
    same values on a call to :meth:`~coreax.solvers.Solver.reduce` as would have been
    returned if no padding were present.

    Inheriting from this class is only a promise by the solver to obey the invariance
    property. Conformity with the property is not checked at runtime.
    """
