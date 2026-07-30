"""WSIndex — minimal package skeleton.

This module intentionally contains a single small, fully typed function so the
toolchain (uv, ruff, mypy, pytest, packaging, CI) can be verified end to end
before the real indexing engine is written. See ARCH_en.md for the target design.
"""

from __future__ import annotations

import argparse

__version__ = "0.1.0"


def greet(name: str = "world") -> str:
    """Return a friendly greeting used to smoke-test the toolchain."""
    return f"Hello, {name}, from WSIndex {__version__}!"


def main() -> None:
    """Entry point for the ``wsindex`` console script."""
    parser = argparse.ArgumentParser(prog="wsindex", description="WSIndex skeleton CLI.")
    parser.add_argument("name", nargs="?", default="world", help="who to greet")
    args = parser.parse_args()
    print(greet(args.name))
