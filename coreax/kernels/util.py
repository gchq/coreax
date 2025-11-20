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

"""Utility functions for the kernels subpackage."""

from math import ceil

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array
from jaxtyping import Shaped

from coreax.data import Data, _atleast_2d_consistent, as_data
from coreax.util import pairwise, squared_distance, tree_zero_pad_leading_axis


def _block_data_convert(
    x: Data | Shaped[Array, " n d"], block_size: int | None
) -> tuple[Data, int]:
    """Convert 'x' into padded and weight normalized blocks of size 'block_size'."""
    x = as_data(x).normalize(preserve_zeros=True)
    block_size = len(x) if block_size is None else min(max(int(block_size), 1), len(x))
    padding = ceil(len(x) / block_size) * block_size - len(x)
    padded_x = tree_zero_pad_leading_axis(x, padding)

    def _reshape(x: Array) -> Array:
        _, *remaining_shape = jnp.shape(x)
        try:
            return x.reshape(-1, block_size, *remaining_shape)
        except ZeroDivisionError as err:
            if 0 in x.shape:
                raise ValueError("'x' must not be empty") from err
            raise

    return jtu.tree_map(_reshape, padded_x, is_leaf=eqx.is_array), len(x)


def median_heuristic(
    x: Shaped[Array, " n d"] | Shaped[Array, " n"] | Shaped[Array, ""] | float | int,
) -> Shaped[Array, ""]:
    """
    Compute the median heuristic for setting kernel bandwidth.

    Analysis of the performance of the median heuristic can be found in
    :cite:`garreau2018median`.

    :param x: Input array of vectors
    :return: Bandwidth parameter, computed from the median heuristic, as a
        zero-dimensional array
    """
    # Format inputs
    x = _atleast_2d_consistent(x)
    # Calculate square distances as an upper triangular matrix
    square_distances = jnp.triu(pairwise(squared_distance)(x, x), k=1)
    # Calculate the median of the square distances
    median_square_distance = jnp.median(
        square_distances[jnp.triu_indices_from(square_distances, k=1)]
    )

    return jnp.sqrt(median_square_distance / 2.0)
