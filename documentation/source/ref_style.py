"""
Contains a custom reference style for Coreax.

See https://github.com/mcmtroffaes/sphinxcontrib-bibtex
Version: 2.6.2
Path: test/roots/test-bibliography_style_label_1/conf.py
"""
from __future__ import annotations

from typing import Generator, Iterable

from pybtex.database import Entry
from pybtex.plugin import register_plugin
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.labels import BaseLabelStyle

STYLE_NAME = "bib_key"


# pylint: disable=too-few-public-methods
class MyLabelStyle(BaseLabelStyle):
    """A label style which formats labels to use the original key in the .bib file."""

    def format_labels(
        self, sorted_entries: Iterable[Entry]
    ) -> Generator[str, None, None]:
        """Yield formatted labels for each entry."""
        for entry in sorted_entries:
            yield entry.key


class _BibKeyStyle(UnsrtStyle):
    default_label_style = MyLabelStyle


register_plugin("pybtex.style.formatting", STYLE_NAME, _BibKeyStyle)
# pylint: enable=too-few-public-methods
