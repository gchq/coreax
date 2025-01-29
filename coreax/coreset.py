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

from typing import TYPE_CHECKING, Generic, TypeVar, Union

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Shaped
from typing_extensions import Self

from coreax.data import Data, as_data, as_supervised_data
from coreax.metrics import Metric
from coreax.weights import WeightsOptimiser

if TYPE_CHECKING:
    from typing import Any  # noqa: F401

_Data = TypeVar("_Data", bound=Data)


class Coreset(eqx.Module, Generic[_Data]):
    r"""
    Data structure for representing a coreset.

    A coreset is a reduced set of :math:`\hat{n}` (potentially weighted) data points,
    :math:`\hat{X} := \{(\hat{x}_i, \hat{w}_i)\}_{i=1}^\hat{n}` that, in some sense,
    best represent the "important" properties of a larger set of :math:`n > \hat{n}`
    (potentially weighted) data points :math:`X := \{(x_i, w_i)\}_{i=1}^n`.

    :math:`\hat{x}_i, x_i \in \Omega` represent the data points/nodes and
    :math:`\hat{w}_i, w_i \in \mathbb{R}` represent the associated weights.

    :param nodes: The (weighted) coreset nodes, :math:`\hat{x}_i`; once instantiated,
        the nodes should only be accessed via :meth:`Coresubset.coreset`
    :param pre_coreset_data: The dataset :math:`X` used to construct the coreset.
    """

    nodes: _Data
    pre_coreset_data: _Data

    def __init__(
        self, nodes: Union[_Data, Array], pre_coreset_data: Union[_Data, Array]
    ) -> None:
        """Handle type conversion of ``nodes`` and ``pre_coreset_data``."""
        if isinstance(nodes, Array):
            self.nodes = as_data(nodes)
        elif isinstance(nodes, tuple):
            self.nodes = as_supervised_data(nodes)
        else:
            self.nodes = nodes

        if isinstance(pre_coreset_data, Array):
            self.pre_coreset_data = as_data(pre_coreset_data)
        elif isinstance(pre_coreset_data, tuple):
            self.pre_coreset_data = as_supervised_data(pre_coreset_data)
        else:
            self.pre_coreset_data = pre_coreset_data

    def __check_init__(self) -> None:
        """Check that coreset has fewer 'nodes' than the 'pre_coreset_data'."""
        if len(self.nodes) > len(self.pre_coreset_data):
            raise ValueError(
                "'len(nodes)' cannot be greater than 'len(pre_coreset_data)' "
                "by definition of a Coreset"
            )

    def __len__(self) -> int:
        """Return Coreset size/length."""
        return len(self.nodes)

    @property
    def coreset(self) -> _Data:
        """Materialised coreset."""
        return self.nodes

    def solve_weights(self, solver: WeightsOptimiser[_Data], **solver_kwargs) -> Self:
        """Return a copy of 'self' with weights solved by 'solver'."""
        weights = solver.solve(self.pre_coreset_data, self.coreset, **solver_kwargs)
        return eqx.tree_at(lambda x: x.nodes.weights, self, weights)

    def compute_metric(
        self, metric: Metric[_Data], **metric_kwargs
    ) -> Shaped[Array, ""]:
        """Return metric-distance between `self.pre_coreset_data` and `self.coreset`."""
        return metric.compute(self.pre_coreset_data, self.coreset, **metric_kwargs)


class Coresubset(Coreset[_Data], Generic[_Data]):
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

    :param nodes: The (weighted) coresubset node indices, :math:`I`; the materialised
        coresubset nodes should only be accessed via :meth:`Coresubset.coreset`.
    :param pre_coreset_data: The dataset :math:`X` used to construct the coreset.
    """

    # Unlike on Coreset, contains indices of coreset rather than coreset itself
    nodes: _Data

    def __init__(
        self, nodes: Union[Data, Array], pre_coreset_data: Union[_Data, Array]
    ):
        """Handle typing of ``nodes`` being a `Data` instance."""
        # nodes type can't technically be cast to _Data but do so anyway to avoid a
        # significant amount of boilerplate just for type checking
        super().__init__(
            nodes,  # pyright: ignore [reportArgumentType]
            pre_coreset_data,
        )

    @property
    def coreset(self) -> _Data:
        """Materialise the coresubset from the indices and original data."""
        coreset_data = self.pre_coreset_data[self.unweighted_indices]
        return eqx.tree_at(lambda x: x.weights, coreset_data, self.nodes.weights)

    @property
    def unweighted_indices(self) -> Shaped[Array, " n"]:
        """Unweighted Coresubset indices - attribute access helper."""
        return jnp.squeeze(self.nodes.data)
