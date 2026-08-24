"""Tests for the package metadata and importability."""

import chainset


def test_package_version_is_available() -> None:
    """The package exposes a non-empty ``__version__``."""
    assert chainset.__version__
