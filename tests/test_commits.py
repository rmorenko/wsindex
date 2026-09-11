"""Commit messages as corpus, blame as edges.

A repository's reasoning lives in its commit messages and nowhere else —
the step-26 spike proved that on this repo, for "why is dedup before
embedding". These tests pin the two halves that make it searchable: the
messages become chunks, and blame becomes `BLAMED_BY` links from the code
to the commit that last wrote it.

Every case drives real git. Blame in particular has a failure mode no
stub would reproduce: it cannot blame a file that is in no commit, and a
full pass indexes untracked files by design.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from wsindex.config import Config, Repository
from wsindex.embed import FakeEmbedder
from wsindex.ingest import commits as commits_module
from wsindex.ingest.commits import (
    COMMIT_LANG,
    blame_links,
    blame_map,
    commit_chunks,
    read_commits,
)
from wsindex.ingest.git_state import GIT_TIMEOUT
from wsindex.links import LinkKind, LinkStore
from wsindex.model import Kind, SearchFilter, SourceFile
from wsindex.pipeline import Pipeline
from wsindex.store import LanceDBStore

Committer = Callable[..., None]


@pytest.fixture
def commit(monkeypatch: pytest.MonkeyPatch) -> Committer:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    for name in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{name}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{name}_EMAIL", "test@example.invalid")

    def run(root: Path, message: str = "snapshot") -> None:
        if not (root / ".git").exists():
            subprocess.run(
                ["git", "init", "-q", "--initial-branch=main"],
                cwd=root,
                check=True,
                capture_output=True,
            )
        subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-qm", message, "--allow-empty"],
            cwd=root,
            check=True,
            capture_output=True,
        )

    return run


@pytest.fixture
def repo(tmp_path: Path, commit: Committer) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def f():\n    return 1\n")
    commit(root, "feat: add f\n\nBecause the caller needed a number.")
    return root


# --- reading commits -----------------------------------------------------


def test_a_full_pass_reads_the_whole_history(repo: Path, commit: Committer) -> None:
    commit(repo, "second")
    found = read_commits(repo, since=None)
    assert [c.message.splitlines()[0] for c in found] == ["second", "feat: add f"]


def test_an_incremental_pass_reads_only_what_is_new(repo: Path, commit: Committer) -> None:
    first = read_commits(repo, since=None)[0]
    commit(repo, "second")
    commit(repo, "third")

    found = read_commits(repo, since=first.sha)
    assert [c.message for c in found] == ["third", "second"]


def test_nothing_new_reads_nothing(repo: Path) -> None:
    head = read_commits(repo, since=None)[0]
    assert read_commits(repo, since=head.sha) == []


def test_a_full_pass_is_bounded(repo: Path, commit: Committer) -> None:
    # History is unbounded; the questions asked of it are not.
    for i in range(4):
        commit(repo, f"extra {i}")
    assert len(read_commits(repo, since=None, limit=2)) == 2


def test_the_body_survives_intact(repo: Path) -> None:
    # Records are separated by control characters, not newlines: a
    # message contains newlines by definition, and a line-oriented split
    # would tear bodies apart — losing exactly the reasoning worth
    # indexing.
    found = read_commits(repo, since=None)[0]
    assert "Because the caller needed a number." in found.message


# --- commits as chunks ---------------------------------------------------


def test_a_commit_becomes_one_chunk(repo: Path) -> None:
    # One chunk, not windowed: the reasoning in a message is one
    # argument, and splitting it would scatter what `why` hands back.
    chunks = commit_chunks(read_commits(repo, since=None), repo="r")
    assert len(chunks) == 1
    assert "Because the caller needed a number." in chunks[0].text


def test_a_commit_chunk_is_identifiable(repo: Path) -> None:
    found = read_commits(repo, since=None)[0]
    chunk = commit_chunks([found], repo="r")[0]
    assert chunk.kind is Kind.COMMIT
    assert chunk.lang == COMMIT_LANG
    assert chunk.symbol == found.short
    assert chunk.node_type == "commit"


def test_the_synthetic_path_carries_the_date(repo: Path) -> None:
    # A commit has no path on disk and every chunk needs one. Encoding
    # the date there keeps search output readable and history sorted,
    # without a new column on every chunk in the store.
    chunk = commit_chunks(read_commits(repo, since=None), repo="r")[0]
    assert chunk.path.startswith("commits/")
    assert chunk.path.count("-") >= 3  # yyyy-mm-dd-<sha>


def test_an_empty_message_yields_no_chunk(repo: Path, commit: Committer) -> None:
    subprocess.run(
        ["git", "commit", "-q", "--allow-empty", "--allow-empty-message", "-m", ""],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    found = read_commits(repo, since=None)
    assert len(commit_chunks(found, repo="r")) < len(found)


# --- blame edges ---------------------------------------------------------


def test_blame_links_a_chunk_to_the_commit_that_wrote_it(repo: Path) -> None:
    from wsindex.ingest import chunk_file

    commits = read_commits(repo, since=None)
    messages = commit_chunks(commits, repo="r")
    known = {commits[0].sha: messages[0].id}
    chunks = chunk_file(
        (repo / "a.py").read_text(),
        SourceFile(repo="r", path="a.py", lang="python", kind=Kind.CODE),
    )

    blamed = blame_map(repo, ["a.py"])
    links = blame_links(chunks=chunks, known=known, by_line=blamed["a.py"])
    assert links
    assert {link.kind for link in links} == {LinkKind.BLAMED_BY}
    assert links[0].name == commits[0].short
    # Resolved at write time: both ends exist by now, so there is nothing
    # to defer the way code-to-config has to.
    assert links[0].dst_chunk_id == messages[0].id


def test_an_untracked_file_yields_no_blame_and_no_crash(repo: Path) -> None:
    # The bug this guards: a full pass indexes untracked files by design
    # (`ls-files --others` lists them), and git cannot blame
    # a file that is in no commit. Letting that raise would mean one new
    # file breaks indexing for the whole workspace.
    from wsindex.ingest import chunk_file

    (repo / "fresh.py").write_text("def g():\n    return 2\n")
    chunks = chunk_file(
        "def g():\n    return 2\n",
        SourceFile(repo="r", path="fresh.py", lang="python", kind=Kind.CODE),
    )
    blamed = blame_map(repo, ["fresh.py"])
    assert blamed["fresh.py"] == {}
    assert blame_links(chunks=chunks, known={}, by_line=blamed["fresh.py"]) == []


def test_a_commit_outside_this_run_still_gets_an_edge(repo: Path) -> None:
    # Knowing *which* commit answers "when did this change" even when the
    # message was indexed by an earlier run.
    from wsindex.ingest import chunk_file

    chunks = chunk_file(
        (repo / "a.py").read_text(),
        SourceFile(repo="r", path="a.py", lang="python", kind=Kind.CODE),
    )
    links = blame_links(chunks=chunks, known={}, by_line=blame_map(repo, ["a.py"])["a.py"])
    assert links
    assert links[0].dst_chunk_id is None


def test_no_chunks_means_no_links(repo: Path) -> None:
    assert blame_links(chunks=[], known={}, by_line={}) == []


def test_nothing_to_blame_forks_nothing(repo: Path) -> None:
    # The pool is opened per index run, including runs where the
    # incremental path found no changed file.
    assert blame_map(repo, []) == {}


def test_every_file_is_blamed_once_and_kept_apart(repo: Path, commit: Committer) -> None:
    # The point of the pool: many files in flight at once, each answer
    # still landing under its own path.
    (repo / "b.py").write_text("def h():\n    return 3\n")
    commit(repo, "feat: add h")
    blamed = blame_map(repo, ["a.py", "b.py"])
    assert set(blamed) == {"a.py", "b.py"}
    assert len(set(blamed["a.py"].values()) | set(blamed["b.py"].values())) == 2


# --- through a real index run --------------------------------------------


@pytest.fixture
def indexed(tmp_path: Path, repo: Path) -> tuple[Pipeline, LinkStore]:
    config = Config.default("test")
    config.add_repo(Repository(id="r", path=str(repo)))
    links = LinkStore(tmp_path / "idx")
    pipeline = Pipeline(
        store=LanceDBStore(uri=str(tmp_path / "db"), embedder=FakeEmbedder()),
        state_dir=tmp_path / "state",
        links=links,
    )
    return pipeline, links


def test_indexing_reports_commits_apart_from_chunks(
    indexed: tuple[Pipeline, LinkStore],
) -> None:
    # Counted separately on purpose: folding them in would make
    # `files: 1  chunks: 3` fail to add up, since one came from no file.
    pipeline, _ = indexed
    report = pipeline.index()
    assert report.commits == 1
    assert report.files == 1


def test_a_commit_message_is_searchable(indexed: tuple[Pipeline, LinkStore]) -> None:
    pipeline, _ = indexed
    pipeline.index()
    hits = pipeline.search(
        "why the caller needed a number", k=5, filters=SearchFilter(kind=(Kind.COMMIT,))
    )
    assert hits
    assert "Because the caller needed a number." in hits[0].metadata["text"]


def test_commits_can_be_excluded_from_search(indexed: tuple[Pipeline, LinkStore]) -> None:
    # The reason COMMIT is its own kind rather than a DOC: without it
    # there would be no way to search the code without the history.
    pipeline, _ = indexed
    pipeline.index()
    hits = pipeline.search("return", k=10, filters=SearchFilter(kind=(Kind.CODE,)))
    assert hits
    assert all(hit.metadata["kind"] != Kind.COMMIT.value for hit in hits)


def test_indexing_records_blame_edges(indexed: tuple[Pipeline, LinkStore]) -> None:
    pipeline, links = indexed
    pipeline.index()
    rows = links._db.execute(
        "SELECT COUNT(*) FROM links WHERE kind = ?", (LinkKind.BLAMED_BY.value,)
    ).fetchone()
    assert rows[0] > 0


def test_a_second_run_indexes_no_commits_twice(indexed: tuple[Pipeline, LinkStore]) -> None:
    pipeline, _ = indexed
    pipeline.index()
    assert pipeline.index().commits == 0


def test_a_new_commit_is_picked_up_incrementally(
    indexed: tuple[Pipeline, LinkStore], repo: Path, commit: Committer
) -> None:
    pipeline, _ = indexed
    pipeline.index()
    (repo / "a.py").write_text("def f():\n    return 2\n")
    commit(repo, "fix: return two instead")

    report = pipeline.index()
    assert report.commits == 1
    hits = pipeline.search("return two instead", k=5, filters=SearchFilter(kind=(Kind.COMMIT,)))
    assert any("return two instead" in hit.metadata["text"] for hit in hits)


def test_commit_chunks_survive_a_full_pass(
    indexed: tuple[Pipeline, LinkStore], tmp_path: Path
) -> None:
    # A full pass reconciles the whole dataset against what the tree
    # produces. Commit chunks come from no file, so without counting them
    # as fresh the reconciling delete would reap every one of them.
    pipeline, _ = indexed
    pipeline.index()
    (tmp_path / "state" / "state.json").unlink()

    pipeline.index()
    kept = pipeline.search("caller", k=10, filters=SearchFilter(kind=(Kind.COMMIT,)))
    assert kept


def test_a_blame_failure_names_the_file(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # It does reach the caller — `map` re-raises when the results are
    # walked — but it used to arrive through `concurrent.futures` with no
    # idea which of a hundred files caused it.
    real = commits_module._blame

    def flaky(root: Path, rel_path: str) -> bytes | None:
        if rel_path == "b.py":
            raise RuntimeError("blame blew up")
        return real(root, rel_path)

    monkeypatch.setattr(commits_module, "_blame", flaky)

    with pytest.raises(RuntimeError, match=r"blaming b\.py.*blame blew up"):
        blame_map(repo, ["a.py", "b.py"])


# --- how far back a full pass reaches -------------------------------------


def test_the_cap_is_read_when_the_log_is_read_not_when_the_module_loads(
    repo: Path, commit: Committer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`limit` must not be a keyword default, and this is why.

    Python binds a keyword default once, when the function is defined, so
    `limit: int = MAX_COMMITS` would read the module attribute exactly
    once ever — and `[index] max_commits` could never reach it. A probe
    set the constant, re-indexed three times and got three identical
    answers before this was noticed.
    """
    for n in range(4):
        (repo / "a.py").write_text(f"x = {n}\n")
        commit(repo, f"change {n}")
    monkeypatch.setattr(commits_module, "MAX_COMMITS", 2)

    assert len(read_commits(repo, since=None)) == 2


def test_an_explicit_limit_wins_over_the_default(repo: Path, commit: Committer) -> None:
    for n in range(4):
        (repo / "a.py").write_text(f"x = {n}\n")
        commit(repo, f"change {n}")

    assert len(read_commits(repo, since=None, limit=3)) == 3


def test_an_incremental_run_is_bounded_by_the_diff_not_the_cap(
    repo: Path, commit: Committer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cap exists to stop a first pass reading a decade of history. A
    # run that knows where it left off reads what arrived since, and a
    # cap that also applied there would silently skip commits.
    first = read_commits(repo, since=None)[0].sha
    for n in range(5):
        (repo / "a.py").write_text(f"x = {n}\n")
        commit(repo, f"later {n}")
    monkeypatch.setattr(commits_module, "MAX_COMMITS", 1)

    assert len(read_commits(repo, since=first)) == 5


def test_a_huge_batch_does_not_overflow_the_child_timeout() -> None:
    # `GIT_TIMEOUT * len(rel_paths)` reaches `poll()`, which takes
    # milliseconds as a 32-bit int, so past 17 896 files it raised
    # `OverflowError` before a single file was read. The field trial met
    # that on two real corpora — syncthing's 33 048 pre-rendered docs and
    # Ladybird's 19 253 sources — and both ended with an empty store.
    poll_ceiling = (2**31 - 1) / 1000

    for files in (18_000, 33_048, 1_000_000):
        assert commits_module.child_timeout(files) <= poll_ceiling


def test_a_small_batch_still_scales_with_its_size() -> None:
    # The ceiling must not flatten the ordinary case into one constant:
    # four files should not be given the same grace as four thousand.
    assert commits_module.child_timeout(4) < commits_module.child_timeout(40)
    assert commits_module.child_timeout(4) == pytest.approx(GIT_TIMEOUT * 4)
