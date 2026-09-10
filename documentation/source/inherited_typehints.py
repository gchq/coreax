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

"""Complete method documentation from compatible annotated base methods."""

import inspect
from typing import Any

from sphinx.application import Sphinx
from sphinx_autodoc_typehints import process_docstring
from typing_extensions import get_overloads


def inherit_method_types(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    lines: list[str],
) -> None:
    """Render base annotations for unannotated overrides with matching parameters."""
    if (
        what != "method"
        or not inspect.isfunction(obj)
        or obj.__annotations__
        or get_overloads(obj)
    ):
        return
    module = inspect.getmodule(obj)
    qualified_name = getattr(obj, "__qualname__", "")
    parts = qualified_name.split(".")
    owner: Any = module
    for part in parts[:-1]:
        owner = getattr(owner, part, None)
    if not inspect.isclass(owner):
        return

    parameters = inspect.signature(obj).parameters
    layout = [(parameter.name, parameter.kind) for parameter in parameters.values()]
    for base in owner.__mro__[1:]:
        candidate = base.__dict__.get(parts[-1])
        if isinstance(candidate, (staticmethod, classmethod)):
            candidate = candidate.__func__
        if not inspect.isfunction(candidate):
            continue
        base_parameters = inspect.signature(candidate).parameters
        base_layout = [
            (parameter.name, parameter.kind) for parameter in base_parameters.values()
        ]
        if layout != base_layout:
            return
        if candidate.__annotations__ or get_overloads(candidate):
            process_docstring(app, what, name, candidate, options, lines)
            return


def setup(app: Sphinx) -> dict[str, bool]:
    """Run after the normal type-hint processor without changing runtime objects."""
    app.connect("autodoc-process-docstring", inherit_method_types, priority=600)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
