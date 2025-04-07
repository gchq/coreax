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

"""Data-structures for representing weighted and/or supervised data."""

from collections.abc import Sequence
from typing import Union, overload

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import jit
from jaxtyping import Array, Shaped
from typing_extensions import Self


@overload
def _atleast_2d_consistent(
    arrays: Shaped[Array, ""] | float | int,
) -> Shaped[Array, " 1 1"]: ...


@overload
def _atleast_2d_consistent(
    arrays: Sequence[Shaped[Array, ""] | float | int],
) -> list[Shaped[Array, " 1 1"]]: ...


@overload
def _atleast_2d_consistent(  # pyright:ignore[reportOverlappingOverload]
    arrays: Shaped[Array, " n"],
) -> Shaped[Array, " n 1"]: ...


@overload
def _atleast_2d_consistent(  # pyright:ignore[reportOverlappingOverload]
    arrays: Shaped[Array, " n d *p"],
) -> Shaped[Array, " n d *p"]: ...


@overload
def _atleast_2d_consistent(  # pyright:ignore[reportOverlappingOverload]
    arrays: Sequence[
        Shaped[Array, " _n _d _*p"]
        | Shaped[Array, " _n"]
        | Shaped[Array, ""]
        | float
        | int
    ],
) -> list[
    Shaped[Array, " _n _d _*p"] | Shaped[Array, " _n 1"] | Shaped[Array, " 1 1"]
]: ...


@jit
def _atleast_2d_consistent(  # pyright:ignore[reportOverlappingOverload]
    *arrays: Shaped[Array, " n d *p"]
    | Shaped[Array, " n"]
    | Shaped[Array, ""]
    | float
    | int
    | Sequence[
        Shaped[Array, " _n _d _*p"]
        | Shaped[Array, " _n"]
        | Shaped[Array, ""]
        | float
        | int
    ],
) -> (
    Shaped[Array, " n d *p"]
    | Shaped[Array, " n 1"]
    | Shaped[Array, " 1 1"]
    | list[Shaped[Array, " _n _d _*p"] | Shaped[Array, " _n"] | Shaped[Array, ""]]
):
    r"""
    Given an array or sequence of arrays ensure they are at least 2-dimensional.

    Float and integer types are cast as zero-dimensional input arrays, giving 1x1
    output.

    .. note::

        This function differs from :func:`jax.numpy.atleast_2d` in that it converts
        1-dimensional ``n``-vectors into arrays of shape ``(n, 1)`` rather than
        ``(1, n)``.

    :param arrays: Singular array or sequence of arrays
    :return: At least 2-dimensional array or list of at least 2-dimensional arrays
    """
    # If we have been given just one array, return as an array, not list
    if len(arrays) == 1:
        array = jnp.asarray(arrays[0], copy=False)
        if len(array.shape) == 1:
            return jnp.expand_dims(array, 1)
        return jnp.array(array, copy=False, ndmin=2)

    _arrays = [jnp.asarray(array, copy=False) for array in arrays]
    return [
        jnp.expand_dims(array, 1)
        if len(array.shape) == 1
        else jnp.array(array, copy=False, ndmin=2)
        for array in _arrays
    ]


class Data(eqx.Module):
    r"""
    Class for representing unsupervised data.

    A dataset of size `n` consists of a set of pairs :math:`\{(x_i, w_i)\}_{i=1}^n`
    where :math:`x_i\in\mathbb{R}^d` are the features or inputs and :math:`w_i` are
    weights.

    .. note::
        `n`-vector inputs for `data` are interpreted as `n` points in 1-dimension and
        converted to a `(n, 1)` array.

    Compatible with :func:`jaxtyping.jaxtyped` -- :class:`Data` is interpreted as an
    array type, whose shape is the expected shape of :attr:`Data.data`.

    .. note::
        A `Data` object whose :attr:`Data.data` is expected to be a floating point array
        with shape `a b`, can be type hinted as `x: Float[Data, " a b"] = ...`.

    :param data: An :math:`n \times d` array defining the features of the unsupervised
        dataset
    :param weights: An :math:`n`-vector of weights where each element of the weights
        vector is paired with the corresponding index of the data array, forming the
        pair :math:`(x_i, w_i)`; if passed a scalar weight, it will be broadcast to an
        :math:`n`-vector. the default value of :data:`None` sets the weights to
        the ones vector (implies a scalar weight of one)
    """

    data: Shaped[Array, " n d"] = eqx.field(converter=_atleast_2d_consistent)
    weights: Shaped[Array, " n"] | Shaped[Array, ""] | int | float

    def __init__(
        self,
        data: Shaped[Array, " n d"],
        weights: Shaped[Array, " n"] | Shaped[Array, ""] | int | float | None = None,
    ):
        """Initialise `Data` class, handle non-Array weight attribute."""
        self.data = _atleast_2d_consistent(data)
        n = self.data.shape[0]
        self.weights = jnp.broadcast_to(1 if weights is None else weights, n)

    def __getitem__(self, key) -> Self:
        """Support `Array` style indexing of `Data` objects."""
        return jtu.tree_map(lambda x: x[key], self)

    @overload
    def __jax_array__(
        self: "Data",
    ) -> Shaped[Array, " n d"]: ...

    @overload
    def __jax_array__(  # pyright:ignore[reportOverlappingOverload]
        self: "SupervisedData",
    ) -> Shaped[Array, " n d+p"]: ...

    def __jax_array__(
        self: Union["Data", "SupervisedData"],
    ) -> Shaped[Array, " n d"] | Shaped[Array, " n d+p"]:
        """
        Return value of `jnp.asarray(Data(...))` and `jnp.asarray(SupervisedData(...))`.

        .. note::

            When ``self`` is a `SupervisedData` instance `jnp.asarray` will return
            a single array where the ``supervision`` array has been
            right-concatenated onto the``data`` array.
        """
        if isinstance(self, SupervisedData):
            return jnp.hstack((self.data, self.supervision))
        return self.data

    def __len__(self) -> int:
        """Return data length."""
        return len(self.data)

    @property
    def dtype(self):
        """Return dtype of data; used for jaxtyping annotations."""
        return self.data.dtype

    @property
    def shape(self):
        """Return shape of data; used for jaxtyping annotations."""
        return self.data.shape

    def normalize(self, *, preserve_zeros: bool = False) -> Self:
        """
        Return a copy of ``self`` with ``weights`` that sum to one.

        :param preserve_zeros: If to preserve zero valued weights; when all weights are
            zero valued, the 'normalized' copy will **sum to zero, not one**.
        :return: A copy of 'self' with normalized 'weights'
        """
        normalized_weights = self.weights / jnp.sum(self.weights)
        if preserve_zeros:
            normalized_weights = jnp.nan_to_num(normalized_weights)
        return eqx.tree_at(lambda x: x.weights, self, normalized_weights)


class SupervisedData(Data):
    r"""
    Class for representing supervised data.

    A supervised dataset of size `n` consists of a set of triples
    :math:`\{(x_i, y_i, w_i)\}_{i=1}^n` where :math:`x_i\in\mathbb{R}^d` are the
    features or inputs, :math:`y_i\in\mathbb{R}^p` are the responses or outputs, and
    :math:`w_i` are weights which correspond to the pairs :math:`(x_i, y_i)`.

    .. note::
        `n`-vector inputs for `data` and `supervision` are interpreted as `n` points in
        1-dimension and converted to a `(n, 1)` array.

    :param data: An :math:`n \times d` array defining the features of the supervised
        dataset paired with the corresponding index of the supervision
    :param supervision: An :math:`n \times p` array defining the responses of the
        supervised dataset paired with the corresponding index of the data
    :param weights: An :math:`n`-vector of weights where each element of the weights
        vector is is paired with the corresponding index of the data and supervision
        array, forming the triple :math:`(x_i, y_i, w_i)`; if passed a scalar weight,
        it will be broadcast to an :math:`n`-vector. the default value of :data:`None`
        sets the weights to the ones vector (implies a scalar weight of one)
    """

    supervision: Shaped[Array, " n p"] = eqx.field(converter=_atleast_2d_consistent)

    def __init__(
        self,
        data: Shaped[Array, " n d"],
        supervision: Shaped[Array, " n p"],
        weights: Shaped[Array, " n"] | Shaped[Array, ""] | int | float | None = None,
    ):
        """Initialise SupervisedData class."""
        self.supervision = supervision
        super().__init__(data, weights)

    def __check_init__(self):
        """Check leading dimensions of supervision and data match."""
        if self.supervision.shape[0] != self.data.shape[0]:
            raise ValueError(
                "Leading dimensions of 'supervision' and 'data' must be equal"
            )


def as_data(x: Shaped[Array, " n d"] | Data) -> Data:
    """Cast ``x`` to a `Data` instance."""
    return x if isinstance(x, Data) else Data(x)


def as_supervised_data(
    xy: tuple[Shaped[Array, " n d"], Shaped[Array, " n p"]] | SupervisedData,
) -> SupervisedData:
    """Cast ``xy`` to a `SupervisedData` instance."""
    return xy if isinstance(xy, SupervisedData) else SupervisedData(*xy)
