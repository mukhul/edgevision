"""Checks that the package is installed and importable the way users will use it."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import edgevision


def test_package_is_installed_with_matching_version() -> None:
    # Fails if the package was never installed (pip install -e ".[dev]"), or if
    # __version__ changed since installation and a reinstall is needed.
    assert metadata.version("edgevision") == edgevision.__version__


def test_package_is_imported_from_src_directory() -> None:
    # With the src/ layout, the import must resolve to src/edgevision, not to a
    # stray copy elsewhere on the path.
    package_dir = Path(edgevision.__file__).resolve().parent
    assert package_dir.parent.name == "src"
    assert package_dir.name == "edgevision"
