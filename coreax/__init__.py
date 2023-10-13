# __init__.py
import importlib.metadata as metadata

# this pulls the version from pyproject.toml
__version__ = metadata.version(__package__)
