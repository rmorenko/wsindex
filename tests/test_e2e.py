"""End-to-end acceptance test: clone a real repository, index it, search it.

Runs under the `slow` marker: needs git, network for the clone, the ml
extra and the real embedding model. The corpus and query are overridable,
so the test can point at any repository:

    WSINDEX_E2E_REPO  (default: https://github.com/tensorus/tensorus)
    WSINDEX_E2E_DIR   (default: ~/.cache/wsindex-e2e/<repo name>)
    WSINDEX_E2E_QUERY (default: "where are tensors stored on disk")

The clone is shallow and cached between runs; delete the directory to
re-fetch. The semantic assertion is soft on purpose — it checks that the
storage layer surfaces in the top hits for the default corpus, not exact
ranks: acceptance verifies the promise, unit tests verify the mechanics.
"""

import os
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wsindex.cli import app

pytestmark = pytest.mark.slow

REPO_URL = os.environ.get("WSINDEX_E2E_REPO", "https://github.com/tensorus/tensorus")
QUERY = os.environ.get("WSINDEX_E2E_QUERY", "where are tensors stored on disk")

runner = CliRunner()


def _corpus_dir() -> Path:
    override = os.environ.get("WSINDEX_E2E_DIR")
    if override:
        return Path(override).expanduser()
    name = REPO_URL.rstrip("/").rsplit("/", 1)[-1]
    return Path.home() / ".cache" / "wsindex-e2e" / name


def test_clone_index_and_search_a_real_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("sentence_transformers")
    corpus = _corpus_dir()
    if not corpus.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(corpus)],
            check=True,
            capture_output=True,
        )

    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "e2e"]).exit_code == 0
    assert runner.invoke(app, ["add-repo", "corpus", str(corpus)]).exit_code == 0

    result = runner.invoke(app, ["index"])
    assert result.exit_code == 0
    assert "written:" in result.output

    result = runner.invoke(app, ["search", QUERY, "-k", "5"])
    assert result.exit_code == 0
    hits = [line for line in result.output.splitlines() if line.strip()]
    assert hits and hits[0] != "no results"
    if "WSINDEX_E2E_QUERY" not in os.environ:
        # Default corpus + default query: the storage layer must show up.
        assert any("storage" in hit for hit in hits), result.output
