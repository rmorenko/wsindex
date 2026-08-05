"""Smoke test proving pytest and the package import work."""

from __future__ import annotations

import pytest

from wsindex import __version__, greet, main


def test_greet_default() -> None:
    """The default greeting includes the package version."""
    assert greet() == f"Hello, world, from WSIndex {__version__}!"


def test_greet_custom_name() -> None:
    """A custom name is echoed in the greeting."""
    assert greet("dev") == f"Hello, dev, from WSIndex {__version__}!"


def test_main_prints_greeting(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The console entry point parses argv and prints the greeting."""
    monkeypatch.setattr("sys.argv", ["wsindex", "tester"])
    main()
    assert capsys.readouterr().out == f"Hello, tester, from WSIndex {__version__}!\n"
