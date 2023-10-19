# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pathlib
import sys
from typing import TypeAlias

import sphinx
from jax import random
from jax.typing import ArrayLike

import coreax
import coreax.util as cu

CONF_FILE_PATH = pathlib.Path(__file__).absolute()
SOURCE_FOLDER_PATH = CONF_FILE_PATH.parent
DOCS_FOLDER_PATH = SOURCE_FOLDER_PATH.parent
REPO_FOLDER_PATH = DOCS_FOLDER_PATH.parent

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
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__, __init__",
    "show_inheritance": True,
}

# set Inter-sphinx mapping to link to external documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "optax": ("https://optax.readthedocs.io/en/latest/", None),
}

# specify custom types for autodoc_type_hints
autodoc_custom_types: dict[TypeAlias, str] = {
    cu.KernelFunction: ":obj:`~coreax.util.KernelFunction`",
    cu.KernelFunctionWithGrads: ":obj:`~coreax.util.KernelFunctionWithGrads`",
    ArrayLike: ":data:`~jax.typing.ArrayLike`",
    ArrayLike | None: ":data:`~jax.typing.ArrayLike` | :data:`None`",
}


# specify the typehints_formatter for custom types for autodoc_type_hints
def typehints_formatter(annotation: str, config: sphinx.config.Config) -> str | None:
    """Properly replace custom type aliases."""
    return autodoc_custom_types.get(annotation)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
