"""Guards for the documents a person maintains by hand.

Nothing here reads the code. These tests exist because the documents
under `docs/` are written rather than generated, and a written document
goes stale silently — the two guards in `test_cli.py` were both added
the day something had already drifted.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DESIGN = Path("docs/design")

HEADING = re.compile(r"^(#{1,6}) +(\d+(?:\.\d+)*)?")


def pairs() -> list[tuple[Path, Path]]:
    """Every `x.md` under `docs/` that has an `x.ru.md` beside it."""
    return [
        (english, ru)
        for english in sorted(Path("docs").rglob("*.md"))
        if not english.name.endswith(".ru.md") and (ru := english.with_suffix(".ru.md")).exists()
    ]


def skeleton(path: Path) -> list[tuple[int, str | None]]:
    """A document's headings as (level, section number) — its shape, not its words.

    Language-independent on purpose: the heading *text* differs between
    the two versions and is supposed to, while the level and the section
    number are the same document or they are two different documents.
    """
    shape: list[tuple[int, str | None]] = []
    fenced = False
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if found := HEADING.match(line):
            shape.append((len(found.group(1)), found.group(2)))
    return shape


def test_there_are_pairs_to_check() -> None:
    """The guard below is vacuous if the glob stops matching."""
    assert len(pairs()) >= 3


@pytest.mark.parametrize("english, russian", pairs(), ids=lambda p: p.name)
def test_a_translation_keeps_the_shape_of_its_original(english: Path, russian: Path) -> None:
    """`x.ru.md` has the same sections as `x.md`, in the same order.

    This is the drift that hides: `architecture.ru.md` sat at 603 lines
    against an English version of 168, still describing Tensorus as the
    index database nine months after ADR-7 replaced it, and nothing
    failed — the file was valid Markdown, spell-checked and formatted.
    Nobody reads both halves of a bilingual document in the same sitting,
    which is exactly why a machine should.

    Headings only. Translating the prose is a person's job and no test
    can check it, but a section that exists in one language and not the
    other is a fact about the file, and it is the shape a document loses
    first.
    """
    assert skeleton(russian) == skeleton(
        english
    ), f"{russian} and {english} no longer describe the same document"
