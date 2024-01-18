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

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Support annotations with | in Python < 3.10
# TODO: Remove once no longer supporting old code
from __future__ import annotations

import pathlib
import shutil
import sys
from collections.abc import Generator
from typing import Any

import sphinx.config
from jax.typing import ArrayLike

import coreax
import coreax.util as cu

CONF_FILE_PATH = pathlib.Path(__file__).absolute()
SOURCE_FOLDER_PATH = CONF_FILE_PATH.parent
DOCS_FOLDER_PATH = SOURCE_FOLDER_PATH.parent
REPO_FOLDER_PATH = DOCS_FOLDER_PATH.parent
EXAMPLES = "examples"

sys.path.extend([str(DOCS_FOLDER_PATH), str(SOURCE_FOLDER_PATH), str(REPO_FOLDER_PATH)])

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Coreax"
copyright = "UK Crown"
author = "GCHQ"
version = "v" + coreax.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# sphinx extensions
extensions = [
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# file sources
source_suffix = [".rst", ".md"]

# enable MyST extension to parse HTML image objects in .md files
myst_enable_extensions = ["html_image", "dollarmath", "amsmath"]

# BibTex references path
bibtex_bibfiles = ["references.bib"]

templates_path = ["_templates"]

# Display type annotations only in compiled description.
autodoc_typehints = "description"

autodoc_default_options = {
    "members": True,
    "class": True,
    "member-order": "bysource",
    "private-members": True,
    "undoc-members": True,
    "show_inheritance": True,
}

# set Inter-sphinx mapping to link to external documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "jaxopt": ("https://jaxopt.github.io/stable/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "optax": ("https://optax.readthedocs.io/en/latest/", None),
}

# Specify custom types for autodoc_type_hints

# TODO: Once no longer supporting Python <3.10, drop try statement as quotes should
# never be required
try:
    # pylint: disable=unsupported-binary-operation
    OptionalArrayLike = ArrayLike | None
except TypeError:
    OptionalArrayLike = "ArrayLike | None"

autodoc_custom_types: dict[Any, str] = {
    cu.KernelComputeType: ":obj:`~coreax.util.KernelComputeType`",
    ArrayLike: ":data:`~jax.typing.ArrayLike`",
    OptionalArrayLike: ":data:`~jax.typing.ArrayLike` | :data:`None`",
}


# Specify the typehints_formatter for custom types for autodoc_type_hints
def typehints_formatter(annotation: Any, _: sphinx.config.Config) -> str | None:
    """Properly replace custom type aliases."""
    return autodoc_custom_types.get(annotation)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]


# create local copies of example image files
examples_source = REPO_FOLDER_PATH / EXAMPLES
examples_dest = SOURCE_FOLDER_PATH / EXAMPLES

if examples_dest.exists():
    # makes sure we don't keep files that should have been deleted
    shutil.rmtree(examples_dest)
examples_dest.mkdir()


def walk(source_folder: pathlib.Path) -> Generator:
    """Generate the file names in a directory tree by walking the tree top-down."""
    sub_directories = list(d for d in source_folder.iterdir() if d.is_dir())
    files = list(f for f in source_folder.iterdir() if f.is_file())
    yield source_folder, sub_directories, files
    for s in sub_directories:
        yield from walk(s)


def copy_filtered_files(
    source_folder: pathlib.Path,
    destination_folder: pathlib.Path,
    file_types: set[str] = frozenset(),
):
    r"""Copy the contents of a folder across if they have a particular type."""

    for root, dirs, files in walk(source_folder):
        for dr in dirs:
            pathlib.Path(
                str(root).replace(str(source_folder), str(destination_folder))
            ).joinpath(dr.stem).mkdir()
        for file in files:
            if file.suffix in file_types:
                source_filename = root.joinpath(file)
                dest_filename = str(source_filename).replace(
                    str(source_folder), str(destination_folder)
                )
                print(source_filename)
                print(dest_filename)
                shutil.copyfile(str(source_filename), str(dest_filename))


# copy example files across
copy_filtered_files(examples_source, examples_dest, file_types={".gif", ".png"})
