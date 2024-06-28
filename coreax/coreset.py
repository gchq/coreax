"""Module for defining coreset data structures."""

from typing import TYPE_CHECKING, Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Shaped
from typing_extensions import Self

from coreax.data import Data, SupervisedData, as_data
from coreax.metrics import Metric
from coreax.weights import WeightsOptimiser

if TYPE_CHECKING:
    from typing import Any  # noqa: F401

_Data = TypeVar("_Data", bound=Data)


class Coreset(eqx.Module, Generic[_Data]):
    r"""
    Data structure for representing a coreset.

    **TLDR:** a coreset is a reduced set of :math:`\hat{n}` (potentially weighted) data
    points that, in some sense, best represent the "important" properties of a larger
    set of :math:`n > \hat{n}` (potentially weighted) data points.

    For a dataset :math:`\{(x_i, w_i)\}_{i=1}^n`, where each node :math:`x_i \in \Omega`
    is paired with a non-negative weight :math:`w_i \in \mathbb{R} \ge 0`, there exists
    an implied (discrete) measure :math:`\nu_n = \sum_{i=1}^{n} w_i \delta_{x_i}` on
    :math:`\Omega`. While not very useful on its own, when combined with a set of
    :math:`\nu_n`-integrable test-functions :math:`\Phi = \{ \phi_1, \dots, \phi_M \}`,
    where :math:`\phi_i\ \colon\ \Omega \to \mathbb{R}`, the measure :math:`\nu_n`
    implies the following push-forward measure over :math:`\mathbb{R}^M`

    .. math::
        \begin{align}
            \mu_n &:= \Phi_* \nu_n,\\
            \mu_n &= \sum_{i=1}^{n} w_i \delta_{\Phi(x_i)}.
        \end{align}

    We assume, that for some choice of test-functions, the "important" properties of
    :math:`\nu_n` (the original dataset) are encoded in the "centre-of-mass" of the
    pushed-forward measure :math:`\mu_n`

    .. math::
        \begin{align}
            \text{CoM}(\mu_n) &:= \sum_{i}^{n} w_i \Phi(x_i),\\
            \text{CoM}(\mu_n) &= \int_\Omega \phi_j(\omega) d\mu_n.\
        \end{align}

    .. note::
        Depending on the coreset solver, the test-functions may be explicitly specified
        by the user (the user makes a choice about what properties are "important"), or
        implicitly defined by the solvers's specific objectives (the solver specifies
        what properties are "important").

    A coreset is simply a reduced measure :math:`\hat{\nu}_\hat{n}`, whose push-forward
    :math:`\hat{\mu}_\hat{n} := \Phi_* \hat{\nu}_\hat{n}` has, approximately in some
    cases, the same "centre-of-mass" as the push-forward measure of the original dataset

    .. math::
        \hat{\nu}_\hat{n} := \sum_{i=1}^\hat{n} \hat{w}_i \delta_{\hat{x}_i}, \quad
        \text{CoM}(\hat{\mu}_\hat{n}) = \text{CoM}(\mu_n),

    where :math:`\hat{x}_i \in \Omega` and :math:`\hat{w}_i \in \mathbb{R} \ge 0`. In
    preserving the "centre-of-mass", the coreset satisfies

    .. math::
        \int_\Omega f(\omega)\ d\mu_n = \int_\Omega f(\omega)\ d\hat{\mu}_\hat{n},

    for all functions :math:`f \in \text{span}(\Phi)`. I.E. integration against the
    push-forward of the original dataset and the push-forward of the coreset is
    identical for all functions in the span of the test-functions.

    :param nodes: The (weighted) coreset nodes, :math:`\hat{x}_i`; once instantiated,
        the nodes should only be accessed via :meth:`Coresubset.coreset`
    :param pre_coreset_data: The dataset :math:`X` used to construct the coreset.
    """

    nodes: Data = eqx.field(converter=as_data)
    pre_coreset_data: _Data

    def __check_init__(self):
        """Check that coreset has fewer 'nodes' than the 'pre_coreset_data'."""
        if len(self.nodes) > len(self.pre_coreset_data):
            raise ValueError(
                "'len(nodes)' cannot be greater than 'len(pre_coreset_data)' "
                "by definition of a Coreset"
            )

    def __len__(self):
        """Return Coreset size/length."""
        return len(self.nodes)

    @property
    def coreset(self) -> Data:
        """Materialised coreset."""
        return self.nodes

    def solve_weights(self, solver: WeightsOptimiser, **solver_kwargs) -> Self:
        """Return a copy of 'self' with weights solved by 'solver'."""
        weights = solver.solve(self.pre_coreset_data, self.coreset, **solver_kwargs)
        return eqx.tree_at(lambda x: x.nodes.weights, self, weights)

    def compute_metric(self, metric: Metric, **metric_kwargs) -> Array:
        """Return metric-distance between `self.pre_coreset_data` and `self.coreset`."""
        return metric.compute(self.pre_coreset_data, self.coreset, **metric_kwargs)


class Coresubset(Coreset[_Data], Generic[_Data]):
    r"""
    Data structure for representing a coresubset.

    A coresubset is a :class`Coreset`, with the additional condition that the support of
    the reduced measure (the coreset), must be a subset of the support of the original
    measure (the original data), such that

    .. math::
        \hat{x}_i = x_i, \forall i \in I,
        I \subset \{1, \dots, n\}, text{card}(I) = \hat{n}.

    Thus, a coresubset, unlike a coreset, ensures that feasibility constraints on the
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
        coreset_data = self.pre_coreset_data.data[self.unweighted_indices]
        if isinstance(self.pre_coreset_data, SupervisedData):
            coreset_supervision = self.pre_coreset_data.supervision[
                self.unweighted_indices
            ]
            return SupervisedData(
                data=coreset_data,
                supervision=coreset_supervision,
                weights=self.nodes.weights,
            )
        return Data(data=coreset_data, weights=self.nodes.weights)

    @property
    def unweighted_indices(self) -> Shaped[Array, " n"]:
        """Unweighted Coresubset indices - attribute access helper."""
        return jnp.squeeze(self.nodes.data)

    # pylint: enable=no-member
