"""Smoke test proving pytest and the package import work."""

from __future__ import annotations

from wsindex import __version__, greet


def test_greet_default() -> None:
    """The default greeting includes the package version."""
    assert greet() == f"Hello, world, from WSIndex {__version__}!"


def test_greet_custom_name() -> None:
    """A custom name is echoed in the greeting."""
    assert greet("dev") == f"Hello, dev, from WSIndex {__version__}!"
