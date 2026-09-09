"""Materialization tests: real git repositories, fake sources.

The split is deliberate. Git is real here for the same reason it is real
in `test_git_state`: the claims being made — "an unchanged document
produces no commit", "a dropped url deletes its file" — are claims about
what git does, and a mock would only confirm our idea of it.

The *sources* are fake, because what is being tested is the
materialization, not the network. `probes/step29b` already established
what the real ones hand back, and re-establishing it on every test run
would make the suite depend on GitHub being up.
"""

from pathlib import Path, PurePosixPath
from typing import ClassVar

import pytest

from wsindex.connectors import BUILTIN, Connector, ConnectorSpec, Document, DocumentNotFound
from wsindex.ingest.git_state import decode_path, run_git
from wsindex.snapshot import SnapshotReport, document_path, materialize, render

ANY_URL = ConnectorSpec(type="fake", url_pattern="https://*")


class FakeConnector(Connector):
    """Answers from a dict the test fills; raises for anything missing."""

    documents: ClassVar[dict[str, Document]] = {}

    def matches(self, url: str) -> bool:
        return url.startswith("https://")

    def fetch(self, url: str) -> Document:
        document = self.documents.get(url)
        if document is None:
            raise DocumentNotFound(f"{url} is not there, or not visible")
        return document


@pytest.fixture(autouse=True)
def fake_source(monkeypatch: pytest.MonkeyPatch) -> dict[str, Document]:
    """Route every https url to `FakeConnector`, and hand back its store."""
    documents: dict[str, Document] = {}
    monkeypatch.setattr(FakeConnector, "documents", documents)
    monkeypatch.setitem(BUILTIN, "fake", FakeConnector)
    return documents


@pytest.fixture(autouse=True)
def hermetic_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """No user config, and deliberately no identity in the environment.

    The second half is the point: a snapshot must commit on a machine
    where nobody has ever run `git config user.email`, which is the
    normal state of a CI runner or a container.
    """
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    for leftover in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "EMAIL"):
        monkeypatch.delenv(leftover, raising=False)


def doc(url: str, *, title: str = "A page", text: str = "Body text.", **metadata: str) -> Document:
    return Document(url=url, title=title, text=text, metadata=metadata)


def git(root: Path, *args: str) -> str:
    return decode_path(run_git(root, *args)).strip()


# --- document_path: a url is external input that names a file -------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # The ordinary case: the tree reads like the site it came from.
        ("https://github.com/org/repo/issues/7", "github.com/org/repo/issues/7.md"),
        # A fragment names a place inside a document, not another document.
        ("https://docs.example.com/guide#install", "docs.example.com/guide.md"),
        # A directory-shaped url still names a page, and `guide/` must not
        # collide with `guide`.
        ("https://docs.example.com/guide/", "docs.example.com/guide/index.md"),
        ("https://docs.example.com/guide", "docs.example.com/guide.md"),
        # Nothing but a host.
        ("https://example.com", "example.com/index.md"),
        # A name that is already markdown does not get a second suffix.
        ("https://raw.example.com/org/repo/README.md", "raw.example.com/org/repo/README.md"),
        # The host is case-insensitive; lowercasing it keeps one host one
        # directory.
        ("https://EXAMPLE.com/Page", "example.com/Page.md"),
        # A port is part of the address, and `:` is not a filename.
        ("https://example.com:8443/page", "example.com-8443/page.md"),
        # Unicode survives: `\w` is unicode-aware, and git stores UTF-8.
        ("https://example.com/Дизайн/Заметка", "example.com/Дизайн/Заметка.md"),
    ],
)
def test_a_url_becomes_the_path_it_looks_like(url: str, expected: str) -> None:
    assert document_path(url) == PurePosixPath(expected)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/../../etc/passwd",
        "https://example.com/%2e%2e/%2e%2e/etc/passwd",
        "https://example.com/a/./b/../../../../c",
        "https://example.com/..%2f..%2fetc/passwd",
    ],
)
def test_no_url_can_escape_the_snapshot(url: str, tmp_path: Path) -> None:
    # The property, not the exact spelling: whatever the path turns out
    # to be, resolving it under the root must stay under the root.
    resolved = (tmp_path / document_path(url)).resolve()
    assert resolved.is_relative_to(tmp_path.resolve())


def test_a_credential_in_the_url_stays_out_of_the_path() -> None:
    # netloc would have carried the token into a directory name — in a
    # git repository someone may push.
    path = document_path("https://user:s3cret@example.com/page")
    assert "s3cret" not in str(path)
    assert path == PurePosixPath("example.com/page.md")


def test_a_query_string_is_part_of_the_document_name() -> None:
    # Dropping it would have quietly made two wiki pages one file.
    home = document_path("https://example.com/wiki?page=Home")
    other = document_path("https://example.com/wiki?page=Other")
    assert home != other
    assert "Home" in str(home)


def test_a_very_long_name_is_truncated_and_still_unique() -> None:
    first = document_path("https://example.com/" + "n" * 300 + "a")
    second = document_path("https://example.com/" + "n" * 300 + "b")
    assert first != second
    assert all(len(part.encode()) <= 255 for part in first.parts)


def test_a_malformed_port_does_not_raise() -> None:
    # The url still names a document; the host is only a directory name.
    assert document_path("https://example.com:notaport/page").name == "page.md"


# --- render: what gets committed -----------------------------------------


def test_the_document_is_markdown_with_its_source_in_frontmatter() -> None:
    rendered = render(doc("https://x/1", title="Add CI", text="Run tests.", source="github"))
    assert rendered == (
        "---\n"
        'url: "https://x/1"\n'
        'title: "Add CI"\n'
        'source: "github"\n'
        "---\n"
        "\n"
        "# Add CI\n"
        "\n"
        "Run tests.\n"
    )


def test_nothing_about_the_fetch_is_recorded() -> None:
    # The whole "git log is the source's history" claim rests on this: a
    # timestamp would make every sync a diff.
    rendered = render(doc("https://x/1"))
    assert "fetched" not in rendered.lower()
    assert "date" not in rendered.split("---")[1].lower()


def test_a_title_with_punctuation_does_not_break_the_frontmatter() -> None:
    rendered = render(doc("https://x/1", title='Fix: the "quoted" case'))
    assert 'title: "Fix: the \\"quoted\\" case"' in rendered


def test_a_body_that_already_opens_with_its_title_is_not_repeated() -> None:
    rendered = render(doc("https://x/1", title="Guide", text="# Guide\n\nWords."))
    assert rendered.count("# Guide") == 1


def test_metadata_order_does_not_depend_on_the_source() -> None:
    one = render(doc("https://x/1", state="open", author="a"))
    other = render(doc("https://x/1", author="a", state="open"))
    assert one == other


# --- materialize: the pass as a whole ------------------------------------


def test_a_first_sync_creates_the_repository_and_commits(
    tmp_path: Path, fake_source: dict[str, Document]
) -> None:
    fake_source["https://example.com/a"] = doc("https://example.com/a")
    root = tmp_path / "snapshot"

    report = materialize(root, urls=list(fake_source), specs=[ANY_URL])

    assert report.added == 1
    assert report.commit is not None
    assert (root / "example.com" / "a.md").exists()
    assert git(root, "log", "--format=%s") == "sync: 1 added"


def test_the_commit_is_wsindexs_own(tmp_path: Path, fake_source: dict[str, Document]) -> None:
    # No identity is configured anywhere (see `hermetic_git`), so this
    # also proves a snapshot commits on a bare machine.
    fake_source["https://example.com/a"] = doc("https://example.com/a")
    root = tmp_path / "snapshot"

    materialize(root, urls=list(fake_source), specs=[ANY_URL])

    assert git(root, "log", "--format=%an <%ae>") == "wsindex <wsindex@localhost>"


def test_an_unchanged_document_produces_no_commit(
    tmp_path: Path, fake_source: dict[str, Document]
) -> None:
    # The invariant the whole design rests on: a repeat sync is free, and
    # the log records the source's changes rather than the sync's.
    fake_source["https://example.com/a"] = doc("https://example.com/a")
    root = tmp_path / "snapshot"
    materialize(root, urls=list(fake_source), specs=[ANY_URL])

    report = materialize(root, urls=list(fake_source), specs=[ANY_URL])

    assert report == SnapshotReport(unchanged=1, commit=None)
    assert git(root, "rev-list", "--count", "HEAD") == "1"


def test_an_edited_document_is_one_more_commit(
    tmp_path: Path, fake_source: dict[str, Document]
) -> None:
    root = tmp_path / "snapshot"
    fake_source["https://example.com/a"] = doc("https://example.com/a", text="First.")
    materialize(root, urls=list(fake_source), specs=[ANY_URL])
    fake_source["https://example.com/a"] = doc("https://example.com/a", text="Second.")

    report = materialize(root, urls=list(fake_source), specs=[ANY_URL])

    assert (report.updated, report.added) == (1, 0)
    assert git(root, "rev-list", "--count", "HEAD") == "2"
    assert "Second." in git(root, "show", "HEAD:example.com/a.md")


def test_a_url_dropped_from_the_config_is_deleted(
    tmp_path: Path, fake_source: dict[str, Document]
) -> None:
    root = tmp_path / "snapshot"
    fake_source["https://example.com/a"] = doc("https://example.com/a")
    fake_source["https://example.com/b"] = doc("https://example.com/b")
    materialize(root, urls=list(fake_source), specs=[ANY_URL])

    report = materialize(root, urls=["https://example.com/a"], specs=[ANY_URL])

    assert report.removed == 1
    assert not (root / "example.com" / "b.md").exists()
    assert git(root, "log", "--format=%s", "-1") == "sync: 1 removed"


def test_a_failed_fetch_keeps_the_file_it_could_not_refresh(
    tmp_path: Path, fake_source: dict[str, Document]
) -> None:
    # The distinction that makes the history trustworthy: a timeout is
    # not a deletion. Only a url the config no longer names is.
    root = tmp_path / "snapshot"
    fake_source["https://example.com/a"] = doc("https://example.com/a")
    materialize(root, urls=list(fake_source), specs=[ANY_URL])
    fake_source.clear()

    report = materialize(root, urls=["https://example.com/a"], specs=[ANY_URL])

    assert report.removed == 0
    assert (root / "example.com" / "a.md").exists()
    assert [url for url, _ in report.failed] == ["https://example.com/a"]
    assert report.commit is None


def test_an_unroutable_url_is_reported_not_raised(
    tmp_path: Path, fake_source: dict[str, Document]
) -> None:
    fake_source["https://example.com/a"] = doc("https://example.com/a")

    report = materialize(
        tmp_path / "snapshot",
        urls=["https://example.com/a", "ftp://example.com/b"],
        specs=[ANY_URL],
    )

    assert report.added == 1
    assert report.failed == (("ftp://example.com/b", "no connector claims it"),)


def test_two_urls_that_name_one_file_are_not_silently_merged(
    tmp_path: Path, fake_source: dict[str, Document]
) -> None:
    # Letting the second win would make the snapshot depend on config
    # order and lose a document without saying so.
    for url in ("https://example.com/a#one", "https://example.com/a#two"):
        fake_source[url] = doc(url)

    report = materialize(tmp_path / "snapshot", urls=list(fake_source), specs=[ANY_URL])

    assert report.added == 1
    assert len(report.failed) == 1
    assert "same file" in report.failed[0][1]


def test_a_directory_that_is_not_ours_is_not_taken_over(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "important.txt").write_text("mine", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to take it over"):
        materialize(root, urls=[], specs=[ANY_URL])

    assert (root / "important.txt").read_text(encoding="utf-8") == "mine"


def test_an_untracked_file_is_not_pruned(tmp_path: Path, fake_source: dict[str, Document]) -> None:
    # Pruning reads `git ls-files`, so it can only delete what a previous
    # sync committed — never a file someone dropped in the directory.
    root = tmp_path / "snapshot"
    fake_source["https://example.com/a"] = doc("https://example.com/a")
    materialize(root, urls=list(fake_source), specs=[ANY_URL])
    (root / "scratch.txt").write_text("mine", encoding="utf-8")

    materialize(root, urls=[], specs=[ANY_URL])

    # Still there, and still git's business only when someone says so:
    # a stray swept into a commit would be tracked, and the run after
    # that would prune it as a document no url produces.
    assert (root / "scratch.txt").exists()
    assert "scratch.txt" not in git(root, "ls-files")


def test_an_empty_snapshot_is_a_repository_with_no_commits(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"

    report = materialize(root, urls=[], specs=[ANY_URL])

    assert report == SnapshotReport()
    assert (root / ".git").is_dir()


def test_the_summary_says_what_moved() -> None:
    assert SnapshotReport(added=2, updated=1).summary() == "2 added, 1 updated"
    assert SnapshotReport(unchanged=3).summary() == "up to date (3 documents)"
    assert SnapshotReport(added=1, failed=(("u", "why"),)).summary() == "1 added; 1 failed"


def test_a_document_with_crlf_settles_after_one_sync(
    tmp_path: Path, fake_source: dict[str, Document]
) -> None:
    # Found live, against a real GitHub issue whose body carries one
    # `\r\n`. Comparing with `read_text` translated the endings back, so
    # the file never matched what had just been written from it: every
    # sync reported "1 updated" and committed nothing, forever.
    root = tmp_path / "snapshot"
    fake_source["https://example.com/a"] = doc("https://example.com/a", text="One\r\nTwo\r\n")
    materialize(root, urls=list(fake_source), specs=[ANY_URL])

    report = materialize(root, urls=list(fake_source), specs=[ANY_URL])

    assert report == SnapshotReport(unchanged=1, commit=None)
    # And the file itself has one kind of line ending, not two.
    assert b"\r" not in (root / "example.com" / "a.md").read_bytes()
