"""References out of the repository, into trackers and urls.

A commit message names a ticket; a doc links a page. Recognising those is
a cheap bridge — nothing is downloaded — so that "why" can reach the
ticket without a connector, and a connector can bring contents later.

The measurement that shaped this is worth restating, because it is the
whole reason the config carries the patterns. A built-in `PROJ-123` rule
matched 198 times on this repository and every hit was an internal
number: ADR-7, FR-111, ADR-5. `[A-Z]+-\\d+` is not a Jira key, it is also
`UTF-8` and `ISO-8601`. Which prefixes name a tracker is knowledge only
the workspace has.
"""

from pathlib import Path

import pytest

from wsindex.config import Config
from wsindex.ingest.link_extract import links_for
from wsindex.links import LinkKind
from wsindex.model import Chunk, Kind

JIRA = "https://jira.example.invalid/browse/PROJ-{key}"
ISSUE = "https://github.com/org/repo/issues/{key}"


def message(text: str, *, kind: Kind = Kind.COMMIT, start: int = 1) -> Chunk:
    return Chunk(
        repo="r",
        path="commits/2026-09-09-abc1234",
        lang="git-commit",
        kind=kind,
        symbol="abc1234",
        node_type="commit",
        start_line=start,
        end_line=start,
        text=text,
    )


def refs(chunk: Chunk, templates: dict[str, str] | None = None) -> list[tuple[str, str | None]]:
    return [
        (link.name, link.url)
        for link in links_for([chunk], references=templates)
        if link.kind is LinkKind.REFERENCES
    ]


# --- bare urls: the one kind that needs no configuration -----------------


def test_a_bare_url_resolves_to_itself() -> None:
    found = refs(message("See https://example.invalid/docs for the rationale."))
    assert found == [("https://example.invalid/docs", "https://example.invalid/docs")]


def test_trailing_punctuation_is_not_part_of_the_url() -> None:
    # Measured while building this: without stripping, a url from a
    # markdown sentence came out as `http://localhost:8080`,` — a link
    # that resolves to nothing.
    for text, expected in [
        ("see https://example.invalid/a.", "https://example.invalid/a"),
        ("see (https://example.invalid/b)", "https://example.invalid/b"),
        ("see `https://example.invalid/c`,", "https://example.invalid/c"),
        ("see https://example.invalid/d;", "https://example.invalid/d"),
    ]:
        assert refs(message(text))[0][1] == expected, text


def test_a_url_path_survives_intact() -> None:
    found = refs(message("https://github.com/org/repo/pull/42"))
    assert found[0][1] == "https://github.com/org/repo/pull/42"


def test_documents_carry_references_too() -> None:
    found = refs(message("Read https://example.invalid/guide", kind=Kind.DOC))
    assert found == [("https://example.invalid/guide", "https://example.invalid/guide")]


# --- configured prefixes -------------------------------------------------


def test_a_declared_prefix_resolves_through_its_template() -> None:
    found = refs(message("Fixes PROJ-123 at last"), {"PROJ-": JIRA})
    assert found == [("PROJ-123", "https://jira.example.invalid/browse/PROJ-123")]


def test_an_issue_prefix_works_the_same_way() -> None:
    found = refs(message("Closes #42"), {"#": ISSUE})
    assert found == [("#42", "https://github.com/org/repo/issues/42")]


def test_several_prefixes_coexist() -> None:
    found = refs(message("PROJ-1 and #2"), {"PROJ-": JIRA, "#": ISSUE})
    assert {name for name, _ in found} == {"PROJ-1", "#2"}


def test_an_undeclared_prefix_is_not_a_reference() -> None:
    # The finding that shaped the design: ADR-7 is a document of ours,
    # not a ticket. With no prefix declared, it is simply not a reference.
    assert refs(message("Per ADR-7 the store is LanceDB")) == []


def test_declaring_one_prefix_does_not_recognise_another() -> None:
    # Even with Jira configured, `ADR-7` stays invisible: the prefix is
    # `PROJ-`, not "anything that looks like a key".
    found = refs(message("Per ADR-7, fixes PROJ-9"), {"PROJ-": JIRA})
    assert [name for name, _ in found] == ["PROJ-9"]


def test_a_reference_records_the_line_it_was_written_on() -> None:
    # A file line, not an offset inside the chunk: the point of storing
    # one is to send a person to it.
    chunk = message("subject\n\nFixes PROJ-7\n", start=10)
    links = links_for([chunk], references={"PROJ-": JIRA})
    assert [link.line for link in links] == [12]


# --- what is deliberately not recorded -----------------------------------


def test_code_contributes_no_references() -> None:
    # A url in code is not a pointer someone wrote for a reader; it is
    # usually an endpoint, which the drift rule already has an opinion
    # about. Keeping the two apart keeps both readable.
    code = Chunk(
        repo="r",
        path="a.py",
        lang="python",
        kind=Kind.CODE,
        symbol=None,
        node_type=None,
        start_line=1,
        end_line=1,
        text='BASE = "https://api.example.invalid"',
    )
    assert [link for link in links_for([code]) if link.kind is LinkKind.REFERENCES] == []


def test_nothing_is_recognised_without_configuration_except_urls() -> None:
    found = refs(message("Fixes PROJ-123 and #42, see https://example.invalid"))
    assert found == [("https://example.invalid", "https://example.invalid")]


# --- the config side -----------------------------------------------------


def test_references_default_to_empty() -> None:
    # Zero false positives out of the box, which the 198-match
    # measurement is the argument for.
    assert Config.default("demo").references == {}


def test_references_round_trip_through_the_file(tmp_path: Path) -> None:
    config = Config.default("demo")
    config._data["references"] = {"PROJ-": JIRA}
    path = config.save(tmp_path / "wsindex.toml")

    Config.reset()
    assert Config(path).references == {"PROJ-": JIRA}


def test_a_config_without_the_section_still_loads(tmp_path: Path) -> None:
    # Same defaulted-section policy as `store` and `rank`: configs
    # written before this existed must keep working.
    import tomli_w

    data = Config.default("demo").to_dict()
    del data["references"]
    path = tmp_path / "wsindex.toml"
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    Config.reset()
    assert Config(path).references == {}


@pytest.mark.parametrize("template", ["https://x.invalid/{key}", "https://x.invalid/PROJ-{key}"])
def test_the_template_receives_the_digits(template: str) -> None:
    found = refs(message("PROJ-77"), {"PROJ-": template})
    assert found[0][1] == template.format(key="77")
