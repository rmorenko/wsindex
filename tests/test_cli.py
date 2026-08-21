"""CLI tests via CliRunner: no subprocesses, CWD controlled per test.

Every command starts from disk state only (fresh-process model), so each
test chdirs into its own tmp_path and drives the full loop through files.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from wsindex.cli import WSINDEX_TOML, app
from wsindex.config import Provider, load_config
from wsindex.embed.embedder import FakeEmbedder

runner = CliRunner()

PY_TEXT = "def f():\n    return 1"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Empty CWD of its own for every test + a small repo to register."""
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "repo1"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "main.py").write_text(PY_TEXT + "\n")
    (repo / "README.md").write_text("# Title\nalpha body\n")
    return tmp_path


def test_init_creates_config(workspace: Path) -> None:
    result = runner.invoke(app, ["init", "ws"])
    assert result.exit_code == 0
    assert (workspace / WSINDEX_TOML).exists()
    assert "created" in result.output


def test_init_provider_option_reaches_the_file(workspace: Path) -> None:
    # Regression: this line once got lost in a refactor, and every workspace
    # silently initialized with the real model — green tests, 100x slower.
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    assert load_config(workspace / WSINDEX_TOML).provider == Provider.FAKE


def test_init_refuses_to_overwrite(workspace: Path) -> None:
    assert runner.invoke(app, ["init", "ws"]).exit_code == 0
    assert runner.invoke(app, ["init", "other"]).exit_code == 1


def test_command_without_config_fails(workspace: Path) -> None:
    for cmd in (["status"], ["index"], ["search", "x"], ["add-repo", "r", "."]):
        assert runner.invoke(app, cmd).exit_code == 1


def test_add_repo_and_status(workspace: Path) -> None:
    runner.invoke(app, ["init", "ws"])
    assert runner.invoke(app, ["add-repo", "repo1", str(workspace / "repo1")]).exit_code == 0
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "repo1" in result.output


def test_add_repo_duplicate_id_fails(workspace: Path) -> None:
    runner.invoke(app, ["init", "ws"])
    runner.invoke(app, ["add-repo", "repo1", str(workspace / "repo1")])
    assert runner.invoke(app, ["add-repo", "repo1", "elsewhere"]).exit_code == 1


def test_index_then_search_end_to_end(workspace: Path) -> None:
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    runner.invoke(app, ["add-repo", "repo1", str(workspace / "repo1")])

    result = runner.invoke(app, ["index"])
    assert result.exit_code == 0
    assert "files: 2" in result.output
    assert "written: 2" in result.output

    result = runner.invoke(app, ["search", PY_TEXT, "-k", "1"])
    assert result.exit_code == 0
    assert "repo1/src/main.py:1-2" in result.output
    assert "1.000" in result.output

    # Second index run: dedup writes nothing.
    result = runner.invoke(app, ["index"])
    assert "written: 0" in result.output


def test_search_without_index_says_no_results(workspace: Path) -> None:
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    runner.invoke(app, ["add-repo", "repo1", str(workspace / "repo1")])
    result = runner.invoke(app, ["search", "anything"])
    assert result.exit_code == 0
    assert "no results" in result.output


def test_status_with_no_repos_hints_add_repo(workspace: Path) -> None:
    runner.invoke(app, ["init", "ws"])
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "repos: none" in result.output


def test_index_warns_about_missing_repo(workspace: Path) -> None:
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    runner.invoke(app, ["add-repo", "ghost", str(workspace / "does-not-exist")])
    result = runner.invoke(app, ["index"])
    assert result.exit_code == 0  # missing repo is a warning, not a failure


def test_st_provider_builds_st_embedder(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, str] = {}

    class StubST(FakeEmbedder):
        # Same constructor signature as the real class; dim must match
        # config.dim (384) or the composition-root guard rejects it.
        def __init__(self, model_name: str) -> None:
            super().__init__(dim=384)
            created["model"] = model_name

    # Patch where the name is looked up: cli.py imported its own reference.
    monkeypatch.setattr("wsindex.cli.SentenceTransformerEmbedder", StubST)
    runner.invoke(app, ["init", "ws"])  # default provider is sentence-transformers
    runner.invoke(app, ["add-repo", "repo1", str(workspace / "repo1")])
    result = runner.invoke(app, ["index"])
    assert result.exit_code == 0
    assert created["model"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_embedder_dim_mismatch_is_rejected(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class WrongDimST(FakeEmbedder):
        def __init__(self, model_name: str) -> None:
            super().__init__(dim=8)

    monkeypatch.setattr("wsindex.cli.SentenceTransformerEmbedder", WrongDimST)
    runner.invoke(app, ["init", "ws"])
    runner.invoke(app, ["add-repo", "repo1", str(workspace / "repo1")])
    result = runner.invoke(app, ["index"])
    assert result.exit_code == 1
    assert "dim" in result.output
