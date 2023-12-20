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
# TODO: Remove once no longer supporting old code
from __future__ import annotations

from collections.abc import Callable
from types import UnionType
from typing import Any, TypeVar

T = TypeVar("T")


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
            if lower_bound is not None and not x > lower_bound:
                raise ValueError(f"{object_name} must be strictly above {lower_bound}")
            if upper_bound is not None and not x < upper_bound:
                raise ValueError(f"{object_name} must be strictly below {upper_bound}")
        else:
            if lower_bound is not None and not x >= lower_bound:
                raise ValueError(f"{object_name} must be {lower_bound} or above")
            if upper_bound is not None and not x <= upper_bound:
                raise ValueError(f"{object_name} must be {upper_bound} or lower")
    except TypeError:
        if strict_inequalities:
            raise TypeError(
                f"{object_name} must have a valid comparison < and > implemented"
            )
        else:
            raise TypeError(
                f"{object_name} must have a valid comparison <= and >= implemented"
            )


def validate_is_instance(
    x: object,
    object_name: str,
    expected_type: type | UnionType | tuple[type | UnionType | None, ...] | None,
) -> None:
    """
    Verify that a given object is of a given type.

    Unlike built-in :func:`isinstance`, :data:`None` may be passed to `expected_type`.

    :func:`cast_as_type` should generally be used where possible with this function
    reserved for classes or other object types that do not have a reliable caster.

    :param x: Object we wish to validate
    :param object_name: Name of ``x`` to display if it is not of type ``expected_type``
    :param expected_type: Expected type of ``x``, can be a tuple or union to specify a
        choice of valid types
    :raises TypeError: Raised if ``x`` is not of type ``expected_type``
    """
    # None is not handled by isinstance so check separately, although is ok if inside a
    # union
    if expected_type is None:
        valid = x is None
    else:
        # Check if a tuple of types containing None is given
        if isinstance(expected_type, tuple) and None in expected_type:
            if x is None:
                # Valid: return here to avoid more intricate if-else statements
                return
            # Filter None from the tuple of expected types - may appear multiple times
            expected_type_without_none = tuple(
                t for t in expected_type if t is not None
            )
        else:
            expected_type_without_none = expected_type

        # Try-except to guard against a still invalid expected_type in isinstance
        try:
            valid = isinstance(x, expected_type_without_none)
        except TypeError:
            raise TypeError("expected_type must be a type, tuple of types or a union")

    if not valid:
        raise TypeError(f"{object_name} must be of type {expected_type}")


def cast_as_type(x: Any, object_name: str, type_caster: Callable) -> Any:
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
    except (TypeError, ValueError) as e:
        error_text = f"{object_name} cannot be cast using {type_caster}: \n"
        if hasattr(e, "message"):
            error_text += e.message
        else:
            error_text += str(e)
        raise TypeError(error_text)


def validate_array_size(x: T, object_name: str, expected_size: int) -> None:
    """
    Verify that an array is of a certain size.

    :param x: Variable we wish to verify the size of
    :param object_name: Name of ``x`` to display if it is not of size``expected_size``
    :param expected_size: The expected size of ``x``
    :raises ValueError: Raised if ``x`` is not of size ``expected_size``
    """
    if not x.shape[0] == expected_size:
        raise ValueError(f"{object_name} is not the expected size of {expected_size}")
