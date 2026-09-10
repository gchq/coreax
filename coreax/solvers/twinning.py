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

"""Nearest-neighbour Data Twinning for supervised and unsupervised datasets."""

from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from typing_extensions import override

from coreax.coreset import Coresubset
from coreax.data import Data, SupervisedData
from coreax.solvers.base import CoresubsetSolver

_Data = TypeVar("_Data", Data, SupervisedData)


class DataTwinning(CoresubsetSolver[_Data, None]):
    r"""
    Select a representative subset using Data Twinning.

    Implements Algorithm 1 of :cite:`vakayil2022twinning`. Standardise the input
    columns, select a point and remove its nearest neighbours, then start the next
    group at the remaining point closest to the last group's farthest neighbour.
    For supervised data, both features and responses determine distances.

    The result contains ``ceil(len(dataset) / thinning_factor)`` distinct indices.
    The final group may be smaller than ``thinning_factor``. Constant columns are
    centred without rescaling. Distance ties are resolved by original row order.
    Only uniform, positive dataset weights are supported.

    Neighbour searches use exhaustive, JIT-compatible array operations rather than
    a spatial tree. For ``n`` selected points and ``N`` rows with ``d`` columns,
    time is O(n N (d + log N)) and working memory is O(N d + n). No full pairwise
    distance matrix is stored. This does not have the tree-based implementation's
    near-linear average-case runtime.

    :param thinning_factor: Positive integer group size; one point is kept per group
    :param start_index: First row to select, or ``None`` to start at the point
        farthest from the standardised centroid
    """

    thinning_factor: int = 2
    start_index: int | None = None

    def __check_init__(self) -> None:
        """Validate static configuration."""
        if (
            isinstance(self.thinning_factor, bool)
            or not isinstance(self.thinning_factor, int)
            or self.thinning_factor < 1
        ):
            raise ValueError("'thinning_factor' must be a positive integer")
        if self.start_index is not None and (
            isinstance(self.start_index, bool)
            or not isinstance(self.start_index, int)
            or self.start_index < 0
        ):
            raise ValueError("'start_index' must be a non-negative integer or None")

    @override
    def reduce(
        self, dataset: _Data, solver_state: None = None
    ) -> tuple[Coresubset[_Data], None]:
        """
        Return the smaller twin as indices into the unchanged original dataset.

        :param dataset: Finite, real, uniformly weighted data to reduce
        :param solver_state: Unused; this deterministic solver has no cached state
        :return: Selected coresubset and ``None``
        """
        points = jnp.asarray(dataset)
        expected_ndim = 2
        if points.ndim != expected_ndim or 0 in points.shape:
            raise ValueError("DataTwinning requires non-empty two-dimensional data")
        if jnp.iscomplexobj(points) or jnp.iscomplexobj(dataset.weights):
            raise ValueError("DataTwinning requires real data and weights")
        count = len(dataset)
        if self.start_index is not None and self.start_index >= count:
            raise ValueError("'start_index' must be smaller than the dataset size")
        points = points.astype(jnp.result_type(points, 0.0))
        weights = jnp.asarray(dataset.weights)
        points = eqx.error_if(
            points,
            jnp.any(~jnp.isfinite(points))
            | jnp.any(~jnp.isfinite(weights))
            | jnp.any(weights <= 0)
            | jnp.any(weights != weights[0]),
            "DataTwinning requires finite data and uniform positive weights",
        )
        centred = points - jnp.mean(points, axis=0)
        scales = jnp.std(points, axis=0)
        points = centred / jnp.where(scales > 0, scales, 1)
        first = (
            jnp.asarray(self.start_index)
            if self.start_index is not None
            else jnp.argmax(jnp.sum(points**2, axis=1))
        )
        size = (count + self.thinning_factor - 1) // self.thinning_factor
        neighbour_count = min(self.thinning_factor - 1, count - 1)

        def select(
            index: int, state: tuple[Array, Array, Array]
        ) -> tuple[Array, Array, Array]:
            """Select a representative and remove its group before finding the next."""
            current, available, selected = state
            selected = selected.at[index].set(current)
            available = available.at[current].set(False)
            distances = jnp.sum((points - points[current]) ** 2, axis=1)
            neighbours = jnp.argsort(
                jnp.where(available, distances, jnp.inf), stable=True
            )[:neighbour_count]
            anchor = current
            if neighbour_count:
                remaining = jnp.minimum(jnp.sum(available), neighbour_count)
                anchor = jnp.where(
                    remaining > 0, neighbours[jnp.maximum(remaining - 1, 0)], current
                )
                available = available.at[neighbours].set(False)
            distances = jnp.sum((points - points[anchor]) ** 2, axis=1)
            next_index = jnp.argmin(jnp.where(available, distances, jnp.inf))
            return next_index, available, selected

        _, _, indices = jax.lax.fori_loop(
            0,
            size,
            select,
            (first, jnp.ones(count, dtype=bool), jnp.zeros(size, dtype=first.dtype)),
        )
        return Coresubset.build(indices, dataset), solver_state
