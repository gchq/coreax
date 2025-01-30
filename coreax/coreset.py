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

"""Module for defining coreset data structures."""

from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Final,
    Generic,
    Tuple,
    TypeVar,
    Union,
    overload,
    override,
)

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Shaped
from typing_extensions import Self

from coreax.data import Data, SupervisedData, as_data
from coreax.metrics import Metric
from coreax.weights import WeightsOptimiser

if TYPE_CHECKING:
    from typing import Any  # noqa: F401

# `_co` is a well-established suffix for covariant TypeVars
# pylint: disable=invalid-name
_TPointsData_co = TypeVar("_TPointsData_co", Data, SupervisedData, covariant=True)
_TOriginalData_co = TypeVar("_TOriginalData_co", Data, SupervisedData, covariant=True)
# pylint: enable=invalid-name


class AbstractCoreset(eqx.Module, Generic[_TPointsData_co, _TOriginalData_co]):
    r"""
    Abstract base class for coresets.

    A coreset is a reduced set of :math:`\hat{n}` (potentially weighted) data points,
    :math:`\hat{X} := \{(\hat{x}_i, \hat{w}_i)\}_{i=1}^\hat{n}` that, in some sense,
    best represent the "important" properties of a larger set of :math:`n > \hat{n}`
    (potentially weighted) data points :math:`X := \{(x_i, w_i)\}_{i=1}^n`.

    :math:`\hat{x}_i, x_i \in \Omega` represent the data points/nodes and
    :math:`\hat{w}_i, w_i \in \mathbb{R}` represent the associated weights.
    """

    @property
    @abstractmethod
    def points(self) -> _TPointsData_co:
        """The coreset points."""

    @property
    @abstractmethod
    def pre_coreset_data(self) -> _TOriginalData_co:
        """The original data that this coreset is based on."""

    def solve_weights(self, solver: WeightsOptimiser[Data], **solver_kwargs) -> Self:
        """Return a copy of 'self' with weights solved by 'solver'."""
        weights = solver.solve(self.pre_coreset_data, self.points, **solver_kwargs)
        return eqx.tree_at(lambda x: x.nodes.weights, self, weights)

    def compute_metric(
        self, metric: Metric[Data], **metric_kwargs
    ) -> Shaped[Array, ""]:
        """Return metric-distance between `self.pre_coreset_data` and `self.coreset`."""
        return metric.compute(self.pre_coreset_data, self.points, **metric_kwargs)

    def __len__(self) -> int:
        """Return Coreset size/length."""
        return len(self.points)

    def __check_init__(self) -> None:
        """Check that coreset has fewer 'nodes' than the 'pre_coreset_data'."""
        if len(self.points) > len(self.pre_coreset_data):
            raise ValueError(
                "'len(nodes)' cannot be greater than 'len(pre_coreset_data)' "
                "by definition of a Coreset"
            )


class PseudoCoreset(
    AbstractCoreset[Data, _TOriginalData_co], Generic[_TOriginalData_co]
):
    r"""
    Data structure for representing a pseudo-coreset.

    The points of a pseudo-coreset are not necessarily points in the original dataset.

    :param indices: The (weighted) coreset nodes, :math:`I`; these can be
        accessed via :meth:`Coresubset.points`.
    :param pre_coreset_data: The dataset :math:`X` used to construct the coreset.
    """

    # These aren't _constants_ so much as just _read-only_, so it doesn't make sense
    # for them to be in SCREAMING_SNAKE_CASE. Also, even if they are changed to appease
    # Pylint here, Pylint then just complains when they're assigned to in __init__
    # instead!
    # pylint: disable=invalid-name
    _nodes: Final[Data]
    _pre_coreset_data: Final[_TOriginalData_co]
    # pylint: enable=invalid-name

    def __init__(self, nodes: Data, pre_coreset_data: _TOriginalData_co) -> None:
        """Initialise self."""
        self._nodes = nodes
        self._pre_coreset_data = pre_coreset_data

    @classmethod
    @overload
    def build(
        cls, nodes: Union[Data, Array], pre_coreset_data: Union[Data, Array]
    ) -> "PseudoCoreset[Data]": ...

    @classmethod
    @overload
    def build(
        cls,
        nodes: Union[Data, Array],
        pre_coreset_data: Union[SupervisedData, Tuple[Array, Array]],
    ) -> "PseudoCoreset[SupervisedData]": ...

    @classmethod
    def build(
        cls,
        nodes: Union[Data, Array],
        pre_coreset_data: Union[Data, SupervisedData, Array, Tuple[Array, Array]],
    ) -> "PseudoCoreset[Data] | PseudoCoreset[SupervisedData]":
        """Construct a PseudoCoreset from Data or raw Arrays."""
        nodes = as_data(nodes)
        if isinstance(pre_coreset_data, Array):
            pre_coreset_data = as_data(pre_coreset_data)
        elif isinstance(pre_coreset_data, tuple):
            pre_coreset_data = SupervisedData(*pre_coreset_data)

        return PseudoCoreset(nodes, pre_coreset_data)

    @property
    @override
    def points(self) -> Data:
        """Materialised coreset."""
        return self._nodes

    @property
    @override
    def pre_coreset_data(self):
        return self._pre_coreset_data


class Coresubset(
    AbstractCoreset[_TOriginalData_co, _TOriginalData_co], Generic[_TOriginalData_co]
):
    r"""
    Data structure for representing a coresubset.

    A coresubset is a :class:`Coreset`, with the additional condition that the coreset
    data points/nodes must be a subset of the original data points/nodes, such that

    .. math::
        \hat{x}_i = x_i, \forall i \in I,
        I \subset \{1, \dots, n\}, \text{card}(I) = \hat{n}.

    Thus, a coresubset, unlike a coreset, ensures that feasibility constraints on the
    support of the measure are maintained :cite:`litterer2012recombination`.

    In coresubsets, the dataset reduction can be implicit (setting weights/nodes to zero
    for all :math:`i \notin I`) or explicit (removing entries from the weight/node
    arrays). The implicit approach is useful when input/output array shape stability is
    required (E.G. for some JAX transformations); the explicit approach is more similar
    to a standard coreset.

    :param indices: The (weighted) coresubset node indices, :math:`I`; the materialised
        coresubset nodes should only be accessed via :meth:`Coresubset.points`.
    :param pre_coreset_data: The dataset :math:`X` used to construct the coreset.
    """

    # These aren't _constants_ so much as just _read-only_, so it doesn't make sense
    # for them to be in SCREAMING_SNAKE_CASE. Also, even if they are changed to appease
    # Pylint here, Pylint then just complains when they're assigned to in __init__
    # instead!
    # pylint: disable=invalid-name
    _indices: Final[Data]
    _pre_coreset_data: Final[_TOriginalData_co]
    # pylint: enable=invalid-name

    def __init__(self, indices: Data, pre_coreset_data: _TOriginalData_co) -> None:
        """Handle type conversion of ``indices`` and ``pre_coreset_data``."""
        self._indices = indices
        self._pre_coreset_data = pre_coreset_data

    @classmethod
    @overload
    def build(
        cls, nodes: Union[Data, Array], pre_coreset_data: Union[Data, Array]
    ) -> "Coresubset[Data]": ...

    @classmethod
    @overload
    def build(
        cls,
        nodes: Union[Data, Array],
        pre_coreset_data: Union[SupervisedData, Tuple[Array, Array]],
    ) -> "Coresubset[SupervisedData]": ...

    @classmethod
    def build(
        cls,
        nodes: Union[Data, Array],
        pre_coreset_data: Union[Data, SupervisedData, Array, Tuple[Array, Array]],
    ) -> "Coresubset[Data] | Coresubset[SupervisedData]":
        """Construct a Coresubset from Data or raw Arrays."""
        nodes = as_data(nodes)
        if isinstance(pre_coreset_data, Array):
            pre_coreset_data = as_data(pre_coreset_data)
        elif isinstance(pre_coreset_data, tuple):
            pre_coreset_data = SupervisedData(*pre_coreset_data)

        return Coresubset(nodes, pre_coreset_data)

    @property
    @override
    def points(self) -> _TOriginalData_co:
        """Materialise the coresubset from the indices and original data."""
        coreset_data = self.pre_coreset_data[self.unweighted_indices]
        return eqx.tree_at(lambda x: x.weights, coreset_data, self._indices.weights)

    @property
    def unweighted_indices(self) -> Shaped[Array, " n"]:
        """Unweighted Coresubset indices - attribute access helper."""
        return jnp.squeeze(self._indices.data)

    @property
    @override
    def pre_coreset_data(self):
        return self._pre_coreset_data
