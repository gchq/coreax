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
"""Configuration details for Sphinx documentation."""

import collections
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional
from unittest import mock

import sphobjinv
import tqdm

# https://docs.github.com/en/actions/learn-github-actions/variables,
# see the "Default environment variables" section
RUNNING_IN_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"

CONF_FILE_PATH = Path(__file__).absolute()
SOURCE_FOLDER_PATH = CONF_FILE_PATH.parent
DOCS_FOLDER_PATH = SOURCE_FOLDER_PATH.parent
REPO_FOLDER_PATH = DOCS_FOLDER_PATH.parent
EXAMPLES = "examples"

TQDM_CUSTOM_PATH = SOURCE_FOLDER_PATH / "tqdm.inv"

sys.path.extend([str(DOCS_FOLDER_PATH), str(SOURCE_FOLDER_PATH), str(REPO_FOLDER_PATH)])


# pylint: disable=wrong-import-position
for module_name in ("jaxopt",):
    # only needed to import coreax, not actually used on import
    sys.modules[module_name] = mock.Mock()
from ref_style import STYLE_NAME  # needed to fix citations within the docstrings

import coreax  # Cannot import until after package has been added to path
import coreax.util

# pylint: enable=wrong-import-position


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# pylint: disable=invalid-name
# pylint: disable=redefined-builtin
project = "Coreax"
copyright = "UK Crown"
author = "GCHQ"
version = "v" + coreax.__version__
# pylint: enable=redefined-builtin
# pylint: enable=invalid-name

# mocked by Sphinx to reduce requirements for building docs
autodoc_mock_imports = ["cv2", "imageio", "matplotlib"]


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_toolbox.wikipedia",
    "sphinxcontrib.bibtex",
]


# pylint: disable=invalid-name
toc_object_entries_show_parents = "hide"  # don't show prefix in secondary TOC
# pylint: enable=invalid-name


# pylint: disable=invalid-name
bibtex_default_style = STYLE_NAME
bibtex_references_path = SOURCE_FOLDER_PATH / "references.bib"
bibtex_bibfiles = [bibtex_references_path.relative_to(SOURCE_FOLDER_PATH).as_posix()]
# pylint: enable=invalid-name


# pylint: disable=invalid-name
autodoc_typehints = (
    "description"  # Display type annotations only in compiled description.
)
# pylint: enable=invalid-name

autodoc_default_options = {
    "members": True,
    "class": True,
    "member-order": "bysource",
    "private-members": False,
    "undoc-members": True,
    "show-inheritance": True,
    "exclude-members": ",".join(
        (  # Use this join syntax to make positions of commas clear and consistent
            "_abc_impl",
            "_parent_ref",
            "_state",
            "hidden_dim",
            "output_dim",
            "name",
            "parent",
            "scope",
            "_asdict",
            "_field_defaults",
            "_fields",
            "_make",
            "_replace",
        )
    ),
}

if RUNNING_IN_GITHUB_ACTIONS:
    linkcheck_ignore = ["https://stackoverflow.com"]

# set Inter-sphinx mapping to link to external documentation
intersphinx_mapping = {  # linking to external documentation
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "jaxopt": ("https://jaxopt.github.io/stable", None),
    "jaxtyping": ("https://docs.kidger.site/jaxtyping", None),
    "flax": ("https://flax-linen.readthedocs.io/en/latest", None),
    "optax": ("https://optax.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "tqdm": ("https://tqdm.github.io/docs", str(TQDM_CUSTOM_PATH)),
    "equinox": ("https://docs.kidger.site/equinox", None),
    "umap": ("https://umap-learn.readthedocs.io/en/latest", None),
}

nitpick_ignore = [
    ("py:class", "flax.core.scope.Scope"),
    ("py:class", "flax.linen.module._Sentinel"),
    ("py:class", "coreax.solvers.coresubset._Data"),
    ("py:class", "coreax.solvers.composite._Coreset"),
    ("py:class", "coreax.solvers.composite._Data"),
    ("py:class", "coreax.solvers.composite._State"),
    ("py:class", "Array"),
    ("py:class", "typing.Self"),
    ("py:class", "jaxtyping.Shaped"),
    ("py:class", "jaxtyping.Shaped[Array, 'n *d']"),
    ("py:class", "jaxtyping.Shaped[ndarray, 'n *d']"),
    ("py:class", "jaxtyping.Shaped[Array, 'n d']"),
    ("py:class", "jaxtyping.Real[Array, 'm-1']"),
    ("py:class", "jaxtyping.Shaped[ndarray, 'n d']"),
    ("py:class", "jaxtyping.Shaped[Array, 'n *p']"),
    ("py:class", "jaxtyping.Shaped[Array, 'n p']"),
    ("py:class", "jaxtyping.Shaped[Array, 'n']"),
    ("py:class", "jaxtyping.Shaped[ndarray, 'n']"),
    ("py:class", "jax._src.typing.SupportsDType"),
    ("py:class", "'n d'"),
    ("py:class", "'n p'"),
    # TODO: Remove once no longer supporting Numpy < 2
    # https://github.com/gchq/coreax/issues/674
    ("py:class", "numpy.bool_"),
    ("py:class", "equinox._module.Module"),
    ("py:class", "coreax.coreset._Data"),
    ("py:class", "tqdm.std.tqdm"),
    ("py:obj", "coreax.util.T"),
    ("py:obj", "coreax.coreset._TPointsData"),
    ("py:obj", "coreax.coreset._TPointsData_co"),
    ("py:class", "coreax.coreset._TPointsData_co"),
    ("py:obj", "coreax.coreset._TOriginalData_co"),
    ("py:class", "coreax.coreset._TOriginalData_co"),
    ("py:obj", "coreax.solvers.composite._Data"),
    ("py:obj", "coreax.solvers.composite._Coreset"),
    ("py:obj", "coreax.solvers.composite._State"),
    ("py:obj", "coreax.solvers.base._Data"),
    ("py:obj", "coreax.solvers.base._State"),
    ("py:obj", "coreax.solvers.base._Coreset"),
    ("py:obj", "coreax.solvers.coresubset._Data"),
    ("py:obj", "coreax.solvers.coresubset._State"),
    ("py:obj", "coreax.solvers.coresubset._Coreset"),
    ("py:obj", "coreax.solvers.recombination._Data"),
    ("py:obj", "coreax.solvers.recombination._State"),
    ("py:obj", "coreax.weights._Data"),
    ("py:obj", "coreax.metrics._Data"),
    ("py:obj", "coreax.solvers.coresubset._SupervisedData"),
    ("py:obj", "coreax.util.T"),
    ("py:class", "pathlib._local.Path"),
]

nitpick_ignore_regex = [
    ("py:class", r"jaxtyping\.Shaped\[Array, '.*']"),
]

# custom references for tqdm, which does not support intersphinx
tqdm_refs: dict[str, dict[str, str]] = {
    "py:class": {
        "tqdm.tqdm": "tqdm/#tqdm-objects",
    }
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# pylint: disable=invalid-name
html_theme = "furo"
html_logo = "../assets/Logo.svg"
html_favicon = "../assets/LogoMark.svg"
# pylint: enable=invalid-name

html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]

html_theme_options = {
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/gchq/coreax/",
    "source_branch": "main",
    "source_directory": "documentation/source/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/gchq/coreax/",
            "class": "fa-brands fa-github fa-2x",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/coreax/",
            "class": "fa-brands fa-python fa-2x",
        },
        {
            "name": "Changelog",
            "url": "https://github.com/gchq/coreax/blob/main/CHANGELOG.md",
            "class": "fa-solid fa-scroll fa-2x",
        },
    ],
}


def create_custom_inv_file(
    module: ModuleType,
    custom_refs: dict[str, dict[str, str]],
    file_name: Optional[str] = None,
) -> None:
    """
    Create an objects.inv file containing custom routes.

    :param module: The module to which this file will refer.
    :param custom_refs: A nested mapping of references to be included. See the example
        below.
    :param file_name: The name of the created file (which is created in the same
        directory as this file). If none, the name of the module is used (with the
        suffix .inv).

    For more info, see https://sphobjinv.readthedocs.io/en/latest/customfile.html

    :Example:
        >>> import collections
        >>> custom_refs = {
        ...     "py:class:" : {
        ...         "collections.Counter": "/path/to/collections.html#Counter",
        ...     }
        ... }
        >>> create_custom_inv_file(collections, custom_refs)

        This creates a file named "collections.inv" which contains a single reference
        to :py:class:`collections.Counter` pointing at the **path**
        "/path/to/collections.html#Counter".
        Intersphinx will combine this path with the path passed to the standard mapping.
    """
    inventory = sphobjinv.Inventory()
    inventory.project = module.__name__
    inventory.version = getattr(module, "__version__", None)

    for domain_and_role, mapping in custom_refs.items():
        domain, role = domain_and_role.split(":")
        for name, uri in mapping.items():
            # pylint: disable=abstract-class-instantiated
            inventory.objects.append(
                sphobjinv.DataObjStr(
                    # Pyright does not detect this as a dataclass
                    name=name,  # pyright: ignore [reportCallIssue]
                    domain=domain,  # pyright: ignore [reportCallIssue]
                    role=role,  # pyright: ignore [reportCallIssue]
                    priority=str(1),  # pyright: ignore [reportCallIssue]
                    uri=uri,  # pyright: ignore [reportCallIssue]
                    dispname="-",  # pyright: ignore [reportCallIssue]
                )
            )
            # pylint: enable=abstract-class-instantiated

    raw_inventory_bytes = inventory.data_file(contract=True)
    compressed_inventory_bytes = sphobjinv.compress(raw_inventory_bytes)

    if file_name is None:
        file_name = f"{module.__name__}.inv"
    sphobjinv.writebytes(SOURCE_FOLDER_PATH / file_name, compressed_inventory_bytes)


if not TQDM_CUSTOM_PATH.exists():
    print("Creating custom inventory file for tqdm")
    create_custom_inv_file(tqdm, tqdm_refs)


# pylint: disable=unused-argument
# pylint: disable=unidiomatic-typecheck
# pylint: disable=protected-access


def remove_namedtuple_attrib_docstring(app, what, name, obj, skip, options):
    """Combine fields and parameters into a single entry for NamedTuple classes."""
    if type(obj) is collections._tuplegetter:  # pyright: ignore [reportAttributeAccessIssue]
        return True
    return skip


# pylint: enable=protected-access
# pylint: enable=unused-argument
# pylint: enable=unidiomatic-typecheck


def setup(app):
    """Register custom functions."""
    app.connect("autodoc-skip-member", remove_namedtuple_attrib_docstring)
