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

"""Build documentation to test type hints inherited by method overrides."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sphinx_autodoc_typehints") is None,
    reason="Documentation dependencies are required",
)
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(name="rendered_methods", scope="module")
def build_method_docs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Compile real kernels and small inheritance fixtures in a separate process."""
    source = tmp_path_factory.mktemp("typehints-docs")
    (source / "conf.py").write_text(
        "import sys\n"
        f"sys.path[:0] = [{str(source)!r}, {str(ROOT)!r}, "
        f"{str(ROOT / 'documentation/source')!r}]\n"
        "extensions = ['sphinx.ext.autodoc', 'sphinx_autodoc_typehints', "
        "'inherited_typehints']\n"
        "autodoc_typehints = 'description'\n",
        encoding="utf-8",
    )
    (source / "example_types.py").write_text(
        "from typing import overload\n"
        "class Base:\n"
        "    def plain(self, value: int) -> str:\n"
        '        """:param value: Input value."""\n'
        "    @overload\n"
        "    def overloaded(self, value: int) -> str: ...\n"
        "    @overload\n"
        "    def overloaded(self, value: str) -> int: ...\n"
        "    def overloaded(self, value):\n"
        '        """:param value: Input value."""\n'
        "class Child(Base):\n"
        "    def plain(self, value): ...\n"
        "    def overloaded(self, value): ...\n"
        "class Grandchild(Child):\n"
        "    def plain(self, value): ...\n"
        "class Explicit(Base):\n"
        "    def plain(self, value: bytes) -> bytes: ...\n"
        "class OwnTypes(Base):\n"
        "    @overload\n"
        "    def overloaded(self, value: bytes) -> bool: ...\n"
        "    @overload\n"
        "    def overloaded(self, value: bool) -> bytes: ...\n"
        "    def overloaded(self, value): ...\n"
        "class Changed(Base):\n"
        "    def plain(self, value, other): ...\n",
        encoding="utf-8",
    )
    names = [
        "coreax.kernels.ScalarValuedKernel.compute",
        "coreax.kernels.SquaredExponentialKernel.compute_elementwise",
        "coreax.kernels.SquaredExponentialKernel.grad_y_elementwise",
        "example_types.Base.overloaded",
        "example_types.Child.plain",
        "example_types.Child.overloaded",
        "example_types.Grandchild.plain",
        "example_types.Explicit.plain",
        "example_types.Changed.plain",
        "example_types.OwnTypes.overloaded",
    ]
    (source / "index.rst").write_text(
        "Type hints\n==========\n\n"
        + "\n\n".join(f".. automethod:: {name}" for name in names)
        + "\n",
        encoding="utf-8",
    )
    output = source / "build"
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "xml", "-W", str(source), str(output)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    document = ElementTree.parse(output / "index.xml")
    methods = {}
    for node in document.iter("desc"):
        signature = node.find("desc_signature")
        if signature is not None and signature.get("ids"):
            methods[signature.get("ids", "")] = " ".join(node.itertext())
    return methods


@pytest.mark.parametrize(
    "name",
    [
        "coreax.kernels.SquaredExponentialKernel.compute_elementwise",
        "coreax.kernels.SquaredExponentialKernel.grad_y_elementwise",
    ],
)
def test_concrete_kernel_types(rendered_methods: dict[str, str], name: str) -> None:
    """Unannotated kernel overrides retain their base array shape annotations."""
    assert "Shaped" in rendered_methods[name]
    assert "Array" in rendered_methods[name]


@pytest.mark.parametrize("owner", ["Base", "Child"])
def test_overloads(rendered_methods: dict[str, str], owner: str) -> None:
    """Display distinct input/output overloads, including inherited overloads."""
    body = rendered_methods[f"example_types.{owner}.overloaded"]
    assert "Overloads" in body
    assert "int" in body and "str" in body


@pytest.mark.parametrize("owner", ["Child", "Grandchild"])
def test_inherited_annotations(rendered_methods: dict[str, str], owner: str) -> None:
    """Walk the method resolution order through unannotated intermediate classes."""
    body = rendered_methods[f"example_types.{owner}.plain"]
    assert "int" in body and "str" in body


def test_explicit_annotations_win(rendered_methods: dict[str, str]) -> None:
    """A subclass's explicit annotation takes precedence over its base contract."""
    body = rendered_methods["example_types.Explicit.plain"]
    assert "bytes" in body
    assert "int" not in body


def test_changed_signature_is_not_inherited(rendered_methods: dict[str, str]) -> None:
    """Do not apply an incompatible base method's types to a new signature."""
    body = rendered_methods["example_types.Changed.plain"]
    assert "other" in body
    assert "int" not in body


def test_explicit_overloads_win(rendered_methods: dict[str, str]) -> None:
    """Do not append inherited overloads when the override declares its own."""
    body = rendered_methods["example_types.OwnTypes.overloaded"]
    assert body.count("Overloads") == 1
    assert "bytes" in body and "bool" in body
    assert "int" not in body and "str" not in body


def test_base_kernel_overloads(rendered_methods: dict[str, str]) -> None:
    """Preserve overloads already rendered for annotated base methods."""
    body = rendered_methods["coreax.kernels.ScalarValuedKernel.compute"]
    assert body.count("Overloads") == 1
    assert "Shaped" in body and "Array" in body
