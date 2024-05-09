"""Module for defining coreset data structures."""

from __future__ import annotations

from typing import Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Shaped

from coreax.data import Data

_Data = TypeVar("_Data", bound=Data)


def _convert_to_weighted(data: Data | Array):
    if isinstance(data, Data):
        return data
    return Data(data)


class Coreset(eqx.Module, Generic[_Data]):
    r"""
    Data structure for representing a coreset.

    TLDR: a coreset is a reduced set of :math:`\hat{n}` (potentially weighted) data
    points that, in some sense, best represent the "important" properties of a larger
    set of :math:`n > \hat{n}` (potentially weighted) data points.

    Given a dataset :math:`X = \{x_i\}_{i=1}^n, x \in \Omega`, where each node is paired
    with a non-negative (probability) weight :math:`w_i \in \mathbb{R} \ge 0`, there
    exists an implied discrete (probability) measure over :math:`\Omega`

    .. math:
        \eta_n = \sum_{i=1}^{n} w_i \delta_{x_i}.

    If we then specify a set of test-functions :math:`\Phi = {\phi_1, \dots, \phi_M}`,
    where :math:`\phi_i \colon \Omega \to \mathbb{R}`, which somehow capture the
    "important" properties of the data, then there also exists an implied push-forward
    measure over :math:`\mathbb{R}^M`

    .. math:
        \mu_n = \sum_{i=1}^{n} w_i \delta_{\Phi(x_i)}.

    A coreset is simply a reduced measure containing :math:`\hat{n} < n` updated nodes
    :math:`\hat{x}_i` and weights :math:`\hat{w}_i`, such that the push-forward measure
    of the coreset :math:`\nu_\hat{n}` has (approximately for some algorithms) the same
    "centre-of-mass" as the push-forward measure for the original data :math:`\mu_n`

    .. math:
        \text{CoM}(\mu_n) = \text{CoM}(\nu_\hat{n}),
        \text{CoM}(\nu_\hat{n}) = \int_\Omega \Phi(\omega) d\nu_\hat{x}(\omega),
        \text{CoM}(\nu_\hat{n}) = \sum_{i=1}^\hat{n} \hat{w}_i \delta_{\Phi(\hat{x}_i)}.

    .. note:
        Depending on the algorithm, the test-functions may be explicitly specified by
        the user, or implicitly defined by the algorithm's specific objectives.

    :param nodes: The (weighted) coreset nodes, math:`x_i \in \text{supp}(\nu_\hat{n})`;
        once instantiated, the nodes should be accessed via :meth:`Coresubset.coreset`
    :param pre_coreset_data: The dataset :math:`X` used to construct the coreset.
    """

    nodes: Data = eqx.field(converter=_convert_to_weighted)
    pre_coreset_data: _Data

    def __len__(self):
        """Return Coreset size/length."""
        return len(self.nodes)

    @property
    def coreset(self) -> Data:
        """Materialised coreset."""
        return self.nodes


class Coresubset(Coreset[_Data], Generic[_Data]):
    r"""
    Data structure for representing a coresubset.

    A coresubset is a :class`Coreset`, with the additional condition that the support of
    the reduced measure (the coreset), must be a subset of the support of the original
    measure (the original data), such that

    .. math:
        \hat{x}_i = x_i, \forall i \in I,
        I \subset \{1, \dots, n\}, text{card}(I) = \hat{n}.

    Thus, a coresubset, unlike a corset, ensures that feasibility constraints on the
    support of the measure are maintained :cite:`litterer2012recombination`. This is
    vital if, for example, the test-functions are only defined on the support of the
    original measure/nodes, rather than all of :math:`\Omega`.

    In coresubsets, the measure reduction can be implicit (setting weights/nodes to
    zero for all :math:`i \in I \ {1, \dots, n}`) or explicit (removing entries from the
    weight/node arrays). The implicit approach is useful when input/output array shape
    stability is required (E.G. for some JAX transformations); the explicit approach is
    more similar to a standard coreset.

    :param nodes: The (weighted) coresubset node indices, :math:`I`; the materialised
        coresubset nodes should be accessed via :meth:`Coresubset.coreset`.
    :param pre_coreset_data: The dataset :math:`X` used to construct the coreset.
    """

    # Incompatibility between Pylint and eqx.field. Pyright handles this correctly.
    # pylint: disable=no-member
    @property
    def coreset(self) -> Data:
        """Materialise the coresubset from the indices and original data."""
        coreset_nodes = self.pre_coreset_data.data[self.unweighted_indices]
        return Data(coreset_nodes, self.nodes.weights)

    @property
    def unweighted_indices(self) -> Shaped[Array, " n"]:
        """Unweighted Coresubset indices - attribute access helper."""
        return jnp.squeeze(self.nodes.data)

    # pylint: enable=no-member