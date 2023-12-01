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
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")


def validate_in_range(
    x: T,
    variable_name: str,
    lower_limit: T = -np.inf,
    upper_limit: T = np.inf,
) -> None:
    """
    Verify that a given input is in a specified range.

    :param x: Variable we wish to verify lies in the specified range
    :param variable_name: Name of ``x`` to display if limits are broken
    :param lower_limit: Lower limit placed on ``x``
    :param upper_limit: Upper limit placed on ``x``
    :raises ValueError: Raised if ``x`` does not fall between ``lower_limit`` and
        ``upper_limit``
    """
    if not lower_limit < x < upper_limit:
        raise ValueError(
            f"{variable_name} must be between {lower_limit} and {upper_limit}. "
            f"Given value {x}."
        )


def validate_variable_is_instance(
    x: Any, variable_name: str, expected_type: Any
) -> None:
    """
    Verify that a given variable is of a given type.

    :param x: Variable we wish to verify lies in the specified range
    :param variable_name: Name of ``x`` to display if limits are broken
    :param expected_type: The expected type of ``x``
    :raises TypeError: Raised if ``x`` is not of type ``expected_type``
    """
    if not isinstance(x, expected_type):
        raise TypeError(f"{variable_name} must be of type {expected_type}.")


def cast_variable_as_type(x: Any, variable_name: str, type_caster: Callable) -> Any:
    """
    Cast a variable as a specified type.

    :param x: Variable to cast as specified type
    :param variable_name: Name of the variable being considered
    :param type_caster: Callable that ``x`` will be passed
    :return: ``x``, but cast as the type specified by ``type_caster``
    :raises TypeError: Raised if ``x`` cannot be cast using ``type_caster``
    """
    try:
        return type_caster(x)
    except Exception as e:
        error_text = (
            f"{variable_name} cannot be cast using {type_caster}. "
            f"Given value {x}.\n"
        )
        if hasattr(e, "message"):
            error_text += e.message
        else:
            error_text += str(e)
        raise TypeError(error_text)
