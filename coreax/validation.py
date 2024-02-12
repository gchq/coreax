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

"""
Functionality to validate data passed throughout coreax.

The functions within this module are intended to be used as a means to validate inputs
passed to classes, functions and methods throughout the coreax codebase.
"""

# Support annotations with | in Python < 3.10
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from jax import Array, dtypes
from jax.typing import ArrayLike
from typing_extensions import TypeAlias

KeyArray: TypeAlias = Array
KeyArrayLike: TypeAlias = ArrayLike

T = TypeVar("T")
U = TypeVar("U")


def validate_in_range(
    x: T,
    object_name: str,
    strict_inequalities: bool,
    lower_bound: T | None = None,
    upper_bound: T | None = None,
) -> None:
    """
    Verify that a given input is in a specified range.

    :param x: Variable we wish to verify lies in the specified range
    :param object_name: Name of ``x`` to display if limits are broken
    :param strict_inequalities: If :data:`True`, checks are applied using strict
        inequalities, otherwise they are not
    :param lower_bound: Lower limit placed on ``x``, or :data:`None`
    :param upper_bound: Upper limit placed on ``x``, or :data:`None`
    :raises ValueError: Raised if ``x`` does not fall between ``lower_limit`` and
        ``upper_limit``
    :raises TypeError: Raised if x cannot be compared to a value using ``>``, ``>=``,
        ``<`` or ``<=``
    """
    try:
        if strict_inequalities:
            if lower_bound is not None and x <= lower_bound:
                raise ValueError(f"{object_name} must be strictly above {lower_bound}")
            if upper_bound is not None and x >= upper_bound:
                raise ValueError(f"{object_name} must be strictly below {upper_bound}")
        else:
            if lower_bound is not None and x < lower_bound:
                raise ValueError(f"{object_name} must be {lower_bound} or above")
            if upper_bound is not None and x > upper_bound:
                raise ValueError(f"{object_name} must be {upper_bound} or lower")
    except TypeError as exc:
        if strict_inequalities:
            raise TypeError(
                f"{object_name} must have a valid comparison < and > implemented"
            ) from exc
        raise TypeError(
            f"{object_name} must have a valid comparison <= and >= implemented"
        ) from exc


def validate_is_instance(
    x: object,
    object_name: str,
    expected_type: type | tuple[type],
) -> None:
    """
    Verify that a given object is of a given type.

    .. note:: This code should work if a :class:`~types.UnionType` is passed to
        ``expected_type`` but this is untested while this library continues to support
        Python < 3.10.

    :func:`cast_as_type` should generally be used where possible with this function
    reserved for classes or other object types that do not have a reliable caster.

    :param x: Object we wish to validate
    :param object_name: Name of ``x`` to display if it is not of type ``expected_type``
    :param expected_type: Expected type of ``x``, can be a tuple to specify a
        choice of valid types
    :raises TypeError: Raised if ``x`` is not of type ``expected_type``
    """
    try:
        is_valid_type = isinstance(x, expected_type)
    except TypeError as exc:
        raise TypeError(
            "expected_type must be a type, tuple of types or a union"
        ) from exc

    if not is_valid_type:
        raise TypeError(f"{object_name} must be of type {expected_type}")


def cast_as_type(x: U, object_name: str, type_caster: Callable[[U], T]) -> T:
    """
    Cast an object as a specified type.

    :param x: Variable to cast as specified type
    :param object_name: Name of the object being considered
    :param type_caster: Callable that ``x`` will be passed
    :return: ``x``, but cast as the type specified by ``type_caster``
    :raises TypeError: Raised if ``x`` cannot be cast using ``type_caster``
    """
    try:
        return type_caster(x)
    except (TypeError, ValueError) as exc:
        error_text = f"{object_name} cannot be cast using {type_caster}: \n"
        if hasattr(exc, "message"):
            error_text += exc.message
        else:
            error_text += str(exc)
        raise TypeError(error_text) from exc


def validate_array_size(
    x: T, object_name: str, dimension: int, expected_size: int
) -> None:
    """
    Validate the size of an array dimension.

    :param x: Variable with a dimension
    :param object_name: Name of ``x`` to display if ``dimension`` is not size
        ``expected_size``
    :param dimension: The dimension to check meets ``expected_size``
    :param expected_size: The expected size of ``dimension``
    :raises ValueError: Raised if the ``dimension`` of ``x`` is not of size
        ``expected_size``
    """
    if not x.shape[dimension] == expected_size:
        raise ValueError(
            f"Dimension {dimension} of {object_name} is not the expected size of "
            f"{expected_size}"
        )


# Deprecation Warning: This will be removed by #420
def validate_key_array(x: KeyArrayLike, object_name: str) -> None:
    """
    Validate that ``x`` is a sub-dtype of jax.dtypes.prng_key.

    :param x: Variable to check
    :param object_name: Semantic name of the object ``x``.
    :raises TypeError: Raised if ``x`` is not a sub-dtype of ``jax.dtype.prng_key``
    """
    if not isinstance(x, KeyArray) or not dtypes.issubdtype(x.dtype, dtypes.prng_key):
        raise TypeError(
            f"{object_name} is not a typed JAX PRNG, for more detail see "
            "https://jax.readthedocs.io/en/latest/jep/9263-typed-keys.html"
        )
