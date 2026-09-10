"""`wsindex refs` and `wsindex why`, the consumers of links.

Everything the two commands show was recorded by earlier steps — drift
edges (26), blame edges and commit chunks (27), external references
(27b). These tests are about turning that into something a person reads.

Worth naming what `refs` is *not*. The plan calls it `refs <symbol>`,
suggesting "who calls this function". Code-to-code edges are deferred
until they can be shown to pay for their noise (ADR-9 measured 11% of
resolvable call names as ambiguous), so there are no callers to list. The
inverted index that does exist is over the link store by name: ports,
tickets, commits, urls.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wsindex.cli import app
from wsindex.paths import CONFIG_FILE, ENV_OVERRIDE

runner = CliRunner()

Committer = Callable[..., None]


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A workspace whose repo has drift, history and a reference."""
    monkeypatch.delenv(ENV_OVERRIDE, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    for name in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{name}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{name}_EMAIL", "test@example.invalid")
    monkeypatch.chdir(tmp_path)

    repo = tmp_path / "svc"
    repo.mkdir()
    (repo / "client.py").write_text(
        'BASE_URL = "http://localhost:8080"\n\n\ndef connect():\n    return BASE_URL\n'
    )
    (repo / "docker-compose.yml").write_text('services:\n  app:\n    ports:\n      - "9000:9000"\n')
    subprocess.run(
        ["git", "init", "-q", "--initial-branch=main"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "commit",
            "-qm",
            "feat: add the client\n\nPoints at the service. Implements PROJ-412.\n"
            "Co-Authored-By: Nobody <nobody@example.invalid>",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    config = tmp_path / CONFIG_FILE
    config.write_text(
        config.read_text().replace(
            "[references]\n", '[references]\n"PROJ-" = "https://jira.invalid/browse/PROJ-{key}"\n'
        )
    )
    runner.invoke(app, ["add-repo", "svc", str(repo)])
    runner.invoke(app, ["index"])
    return tmp_path


# --- refs: the inverted index --------------------------------------------


def test_refs_shows_who_reads_a_port(workspace: Path) -> None:
    result = runner.invoke(app, ["refs", "8080"])
    assert result.exit_code == 0
    assert "read by" in result.output
    assert "client.py:1" in result.output


def test_refs_says_when_nothing_declares_it(workspace: Path) -> None:
    # The drift report narrowed to one name. Someone asking about a port
    # is exactly who needs to know it answers to nothing.
    result = runner.invoke(app, ["refs", "8080"])
    assert "drifted" in result.output


def test_refs_shows_a_declaration(workspace: Path) -> None:
    result = runner.invoke(app, ["refs", "9000"])
    assert "declared by" in result.output
    assert "docker-compose.yml" in result.output


def test_refs_finds_an_external_reference(workspace: Path) -> None:
    result = runner.invoke(app, ["refs", "PROJ-412"])
    assert result.exit_code == 0
    assert "https://jira.invalid/browse/PROJ-412" in result.output


def test_refs_on_a_commit_lists_what_it_wrote(workspace: Path) -> None:
    # Inverted blame: ask about a commit and the answer is the files it
    # touched — hence the label reads "wrote", not "written by".
    sha = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=workspace / "svc",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    result = runner.invoke(app, ["refs", sha])
    assert "wrote" in result.output
    assert "client.py" in result.output


def test_refs_on_an_unknown_name_says_so(workspace: Path) -> None:
    result = runner.invoke(app, ["refs", "nothing-names-this"])
    assert result.exit_code == 0
    assert "no links named" in result.output


def test_refs_needs_a_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path / "absent.toml"))
    result = runner.invoke(app, ["refs", "8080"])
    assert result.exit_code == 1
    assert "wsindex init" in result.output


# --- why: definition -> blame -> reasoning -------------------------------


def test_why_reaches_the_commit_message(workspace: Path) -> None:
    result = runner.invoke(app, ["why", "connect"])
    assert result.exit_code == 0
    assert "connect" in result.output
    assert "written by" in result.output
    assert "Points at the service." in result.output


def test_why_points_at_the_definition(workspace: Path) -> None:
    result = runner.invoke(app, ["why", "connect"])
    assert "svc/client.py:" in result.output


def test_why_drops_git_trailers(workspace: Path) -> None:
    # The body is the answer; a wall of `Co-Authored-By` is noise between
    # the reader and the next commit.
    result = runner.invoke(app, ["why", "connect"])
    assert "Co-Authored-By" not in result.output


def test_why_shows_what_the_commit_pointed_at(workspace: Path) -> None:
    # The bridge out of the repository: the commit named a ticket, and
    # `why` hands back the url without any connector having run.
    result = runner.invoke(app, ["why", "connect"])
    assert "https://jira.invalid/browse/PROJ-412" in result.output


def test_why_and_refs_agree_that_not_found_is_an_answer(workspace: Path) -> None:
    # They used to disagree: `why` exited 1 and `refs` exited 0 for the
    # same situation, which a script finds out the hard way. Looking and
    # not finding is an answer; code 1 is for not being able to look.
    why_result = runner.invoke(app, ["why", "no_such_symbol_anywhere"])
    refs_result = runner.invoke(app, ["refs", "no_such_name_anywhere"])
    assert why_result.exit_code == refs_result.exit_code == 0
    assert "no definition found" in why_result.output


def test_why_needs_a_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path / "absent.toml"))
    result = runner.invoke(app, ["why", "connect"])
    assert result.exit_code == 1
    assert "wsindex init" in result.output


def test_why_survives_a_definition_with_no_blame(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An untracked file has no history, so its chunks carry no
    # blame edges. `why` says so rather than printing an empty section.
    (workspace / "svc" / "fresh.py").write_text("def brandnew():\n    return 1\n")
    runner.invoke(app, ["index"])

    result = runner.invoke(app, ["why", "brandnew"])
    assert result.exit_code == 0
    assert "no commit recorded" in result.output


def test_why_reports_a_commit_whose_message_is_not_indexed(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Blame can name a commit an earlier run indexed, or one outside the
    # window. Knowing *which* commit still answers "when did this
    # change", so the edge is kept and labelled.
    monkeypatch.setattr(
        "wsindex.store.lancedb.LanceDBStore.chunk_text", lambda self, dataset_name, *, ids: {}
    )
    result = runner.invoke(app, ["why", "connect"])
    assert result.exit_code == 0
    assert "message not indexed" in result.output


def test_why_labels_a_commit_from_before_this_run(workspace: Path) -> None:
    # Editing *inside* a function gives it a new chunk id while some of
    # its lines still belong to the older commit. An incremental run
    # indexes only the new commits, so that older edge has no
    # destination — and is kept anyway, because knowing *which* commit
    # still answers "when did this change" (see `blame_links`).
    #
    # Appending a new function would not do it: the existing chunk keeps
    # its id, and the link written by the first run survives intact.
    repo = workspace / "svc"
    (repo / "client.py").write_text(
        'BASE_URL = "http://localhost:8080"\n\n\ndef connect():\n    log = 1\n    return BASE_URL\n'
    )
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-qm", "feat: log the call"], cwd=repo, check=True, capture_output=True
    )
    runner.invoke(app, ["index"])

    result = runner.invoke(app, ["why", "connect"])
    assert result.exit_code == 0
    assert "written by" in result.output
    # The older commit is named even though this run did not index its
    # message: an edge with no destination is still an answer.
    assert "message not indexed" in result.output


# --- one answer, three renderings -----------------------------------------


def test_why_is_one_library_call(workspace: Path) -> None:
    # It used to be assembled separately by `wsindex why` and the MCP
    # tool, from the same three moves — and they had already drifted:
    # one showed three definitions, the other all of them, and only one
    # showed what a commit pointed at.
    from wsindex.cli import build_pipeline

    definitions = build_pipeline().why("connect")

    assert definitions, "the fixture workspace defines `connect`"
    assert all(definition.hit.symbol for definition in definitions)


def test_a_definition_carries_the_commits_that_wrote_it(workspace: Path) -> None:
    from wsindex.cli import build_pipeline

    first = build_pipeline().why("connect")[0]

    assert first.commits, "indexing built blame edges for it"
    assert all(author.commit for author in first.commits)


def test_why_without_links_still_answers(workspace: Path) -> None:
    # A pipeline built without a LinkStore is a valid pipeline; it simply
    # has no authorship to report.
    from wsindex.cli.composition import build_store, config_or_default
    from wsindex.pipeline import Pipeline

    config = config_or_default()
    bare = Pipeline(store=build_store(config), state_dir=config.index_dir, links=None)

    definitions = bare.why("connect")

    assert definitions
    assert all(definition.commits == () for definition in definitions)
