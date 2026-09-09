"""CLI tests via CliRunner: no subprocesses, CWD controlled per test.

Every command starts from disk state only (fresh-process model), so each
test chdirs into its own tmp_path and drives the full loop through files.

`repo1` is a real git repository: `index` is incremental against git
(Этап 8) and treats anything else as a config error.
"""

import subprocess
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wsindex.cli import app
from wsindex.config import Provider
from wsindex.connectors import BUILTIN, Connector, Document, DocumentNotFound
from wsindex.embed import FakeEmbedder
from wsindex.paths import CONFIG_FILE, ENV_OVERRIDE

runner = CliRunner()

PY_TEXT = "def f():\n    return 1"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Empty CWD of its own for every test + a small git repo to register.

    Drops the `$WSINDEX_CONFIG` guard the autouse fixture installs: these
    tests drive the real four-mode resolver, and tmp_path being the CWD is
    what keeps it away from the developer's own config.
    """
    monkeypatch.delenv(ENV_OVERRIDE, raising=False)
    # A hermetic git identity: these tests must not read the developer's
    # ~/.gitconfig, nor need one to exist.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.invalid")
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "repo1"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "main.py").write_text(PY_TEXT + "\n")
    (repo / "README.md").write_text("# Title\nalpha body\n")
    for args in (
        ["init", "-q", "--initial-branch=main"],
        ["add", "-A"],
        ["commit", "-qm", "first"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    return tmp_path


def test_init_creates_config(workspace: Path) -> None:
    result = runner.invoke(app, ["init", "ws"])
    assert result.exit_code == 0
    assert (workspace / CONFIG_FILE).exists()
    assert "created" in result.output


def test_init_provider_option_reaches_the_file(workspace: Path) -> None:
    # Regression: this line once got lost in a refactor, and every workspace
    # silently initialized with the real model — green tests, 100x slower.
    # Read the file, not Config(): the option only counts once it lands there.
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    data = tomllib.loads((workspace / CONFIG_FILE).read_text())
    assert data["embeddings"]["provider"] == Provider.FAKE.value


def test_init_refuses_to_overwrite(workspace: Path) -> None:
    assert runner.invoke(app, ["init", "ws"]).exit_code == 0
    assert runner.invoke(app, ["init", "other"]).exit_code == 1


def test_commands_needing_a_config_file_fail_without_one(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An override pointing at a missing file makes discovery come up empty
    # no matter what config dirs exist on the machine running the tests.
    monkeypatch.setenv(ENV_OVERRIDE, str(workspace / "absent.toml"))
    for cmd in (["index"], ["search", "x"], ["add-repo", "r", "."]):
        result = runner.invoke(app, cmd)
        assert result.exit_code == 1, cmd
        assert "wsindex init" in result.output


def test_status_without_config_reports_defaults(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No config is not a failure for `status`: Config falls back to the
    # built-in defaults, the CLI warns, and status says what it is showing.
    monkeypatch.setenv(ENV_OVERRIDE, str(workspace / "absent.toml"))
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "warning: no wsindex config found" in result.output
    assert "built-in defaults" in result.output
    assert not (workspace / CONFIG_FILE).exists()


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
    captured: dict[str, object] = {}

    class StubST(FakeEmbedder):
        # Same constructor signature as the real class; dim must match
        # config.dim (384) or the composition-root guard rejects it.
        def __init__(self, model_name: str | None = None, cache_folder: Path | None = None) -> None:
            super().__init__(dim=384)
            captured["model"] = model_name
            captured["cache_folder"] = cache_folder

    # Patch where the name is looked up: cli.py imported its own reference.
    monkeypatch.setattr("wsindex.cli.SentenceTransformerEmbedder", StubST)
    runner.invoke(app, ["init", "ws"])  # default provider is sentence-transformers
    runner.invoke(app, ["add-repo", "repo1", str(workspace / "repo1")])
    result = runner.invoke(app, ["index"])
    assert result.exit_code == 0
    # The model name is not forwarded any more — the embedder reads it from
    # Config (test_embedder.py). The cache dir still is: it is a paths
    # concern, shared by every workspace, and not a config field at all.
    assert captured["model"] is None
    # cli passes an explicit models subdir under $XDG_CACHE_HOME/wsindex/
    # so the wsindex-owned cache is namespaced (see ADR-8 amendment).
    cache_folder = captured["cache_folder"]
    assert isinstance(cache_folder, Path)
    assert cache_folder.name == "models"
    assert cache_folder.parent.name == "wsindex"


def test_embedder_dim_mismatch_is_rejected(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class WrongDimST(FakeEmbedder):
        def __init__(self, model_name: str | None = None, cache_folder: Path | None = None) -> None:
            super().__init__(dim=8)

    monkeypatch.setattr("wsindex.cli.SentenceTransformerEmbedder", WrongDimST)
    runner.invoke(app, ["init", "ws"])
    runner.invoke(app, ["add-repo", "repo1", str(workspace / "repo1")])
    result = runner.invoke(app, ["index"])
    assert result.exit_code == 1
    assert "dim" in result.output


# --- step 19g: scope + filter flags on `search` ----------------------------


def _bootstrap_two_repos(workspace: Path) -> None:
    """A workspace with two indexed repos so the scope flags have room to act."""
    repo2 = workspace / "repo2"
    (repo2 / "src").mkdir(parents=True)
    (repo2 / "src" / "main.py").write_text("print('two')\n")
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    runner.invoke(app, ["add-repo", "repo1", str(workspace / "repo1")])
    runner.invoke(app, ["add-repo", "repo2", str(repo2)])
    runner.invoke(app, ["index"])


def test_search_repo_flag_narrows_output(workspace: Path) -> None:
    _bootstrap_two_repos(workspace)
    result = runner.invoke(app, ["search", "def f", "--repo", "repo1", "-k", "5"])
    assert result.exit_code == 0
    assert "repo1/" in result.output
    assert "repo2/" not in result.output


def test_search_unknown_repo_exits_with_error(workspace: Path) -> None:
    _bootstrap_two_repos(workspace)
    result = runner.invoke(app, ["search", "def f", "--repo", "ghost"])
    assert result.exit_code == 1
    assert "unknown repo id" in result.output


def test_search_lang_flag_filters(workspace: Path) -> None:
    _bootstrap_two_repos(workspace)
    result = runner.invoke(app, ["search", "def f", "--lang", "python", "-k", "5"])
    assert result.exit_code == 0
    for line in result.output.splitlines():
        # every hit line has the form `repo/path:... score first_line`
        if ".md" in line:
            pytest.fail(f"markdown hit leaked through --lang python: {line!r}")


def test_search_path_glob_filters(workspace: Path) -> None:
    _bootstrap_two_repos(workspace)
    result = runner.invoke(app, ["search", "def f", "--path", "src/*.py", "-k", "5"])
    assert result.exit_code == 0
    for line in result.output.splitlines():
        if "/" in line and ":" in line:
            assert "src/" in line, f"path filter failed to constrain: {line!r}"


# --- step 22в: wsindex compact -------------------------------------------


def test_compact_reports_reclaimed_space(workspace: Path) -> None:
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    runner.invoke(app, ["add-repo", "repo1", str(workspace / "repo1")])
    runner.invoke(app, ["index"])
    # Churn the index so there is history to prune: edit, commit, re-index.
    (workspace / "repo1" / "src" / "main.py").write_text("def g():\n    return 2\n")
    subprocess.run(["git", "add", "-A"], cwd=workspace / "repo1", check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-qm", "edit"], cwd=workspace / "repo1", check=True, capture_output=True
    )
    runner.invoke(app, ["index"])

    result = runner.invoke(app, ["compact"])

    assert result.exit_code == 0
    assert "reclaimed" in result.output
    assert "versions" in result.output


def test_compact_keeps_search_working(workspace: Path) -> None:
    # Reclaiming space must not cost rows — the guard that matters most.
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    runner.invoke(app, ["add-repo", "repo1", str(workspace / "repo1")])
    runner.invoke(app, ["index"])
    runner.invoke(app, ["compact"])

    result = runner.invoke(app, ["search", PY_TEXT, "-k", "3"])
    assert result.exit_code == 0
    assert "src/main.py" in result.output


def test_compact_keep_days_retains_history(workspace: Path) -> None:
    # Nothing in a fresh index is a day old, so a one-day window prunes
    # nothing. This is the escape hatch for a shared store with readers.
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    runner.invoke(app, ["add-repo", "repo1", str(workspace / "repo1")])
    runner.invoke(app, ["index"])

    result = runner.invoke(app, ["compact", "--keep-days", "1"])

    assert result.exit_code == 0
    before, after = (
        int(n) for n in result.output.split("; ")[1].split(" versions")[0].split(" -> ")
    )
    assert after >= before


def test_compact_without_a_config_exits_with_a_hint(workspace: Path) -> None:
    result = runner.invoke(app, ["compact"])
    assert result.exit_code == 1
    assert "needs a config file" in result.output


def test_compact_says_so_when_size_cannot_be_measured(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A remote store cannot be walked from here. The command must say
    # that, not print a 0 that reads like "nothing was freed".
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    runner.invoke(app, ["add-repo", "repo1", str(workspace / "repo1")])
    runner.invoke(app, ["index"])
    monkeypatch.setattr("wsindex.store.lancedb.LanceDBStore._on_disk_bytes", lambda self: None)

    result = runner.invoke(app, ["compact"])

    assert result.exit_code == 0
    assert "not measurable" in result.output
    assert "reclaimed" not in result.output


# --- step 22b: wsindex sync ----------------------------------------------


@pytest.fixture
def origin(tmp_path: Path, workspace: Path) -> Path:
    """A bare repo standing in for a real remote, with one commit in it."""
    bare = tmp_path / "origin.git"
    bare.mkdir()

    def git(root: Path, *args: str) -> None:
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    git(bare, "init", "-q", "--bare", "--initial-branch=main")
    git(workspace / "repo1", "remote", "add", "origin", str(bare))
    git(workspace / "repo1", "push", "-q", "-u", "origin", "main")
    return bare


def test_add_repo_records_the_remote(workspace: Path) -> None:
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    result = runner.invoke(
        app, ["add-repo", "r", "/checkouts/r", "--remote", "https://example.invalid/r.git"]
    )
    assert result.exit_code == 0
    assert "https://example.invalid/r.git" in result.output
    data = tomllib.loads((workspace / CONFIG_FILE).read_text())
    assert data["repos"][0]["remote"] == "https://example.invalid/r.git"


def test_sync_clones_a_missing_working_copy_then_indexes(workspace: Path, origin: Path) -> None:
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    target = workspace / "cloned"
    runner.invoke(app, ["add-repo", "cloned", str(target), "--remote", str(origin)])

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 0
    assert "cloned: cloned" in result.output
    assert (target / "src" / "main.py").is_file()
    # sync ends in an index run, so the freshly cloned code is searchable.
    assert "files:" in result.output
    assert "src/main.py" in runner.invoke(app, ["search", PY_TEXT, "-k", "1"]).output


def test_sync_no_index_only_updates_working_copies(workspace: Path, origin: Path) -> None:
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    target = workspace / "cloned"
    runner.invoke(app, ["add-repo", "cloned", str(target), "--remote", str(origin)])

    result = runner.invoke(app, ["sync", "--no-index"])

    assert result.exit_code == 0
    assert (target / "src" / "main.py").is_file()
    assert "files:" not in result.output


def test_sync_leaves_a_dirty_checkout_alone_and_exits_nonzero(
    workspace: Path, origin: Path
) -> None:
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    runner.invoke(app, ["add-repo", "repo1", str(workspace / "repo1"), "--remote", str(origin)])
    (workspace / "repo1" / "src" / "main.py").write_text("def mine():\n    return 'work'\n")

    result = runner.invoke(app, ["sync", "--no-index"])

    assert result.exit_code == 1
    assert "uncommitted changes" in result.output
    assert "def mine()" in (workspace / "repo1" / "src" / "main.py").read_text()


def test_sync_reports_an_unusable_remote_per_repo(workspace: Path) -> None:
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    runner.invoke(
        app, ["add-repo", "broken", str(workspace / "nope"), "--remote", str(workspace / "absent")]
    )

    result = runner.invoke(app, ["sync", "--no-index"])

    assert result.exit_code == 1
    assert "broken: failed" in result.output


def test_sync_without_any_remote_says_so(workspace: Path) -> None:
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    runner.invoke(app, ["add-repo", "repo1", str(workspace / "repo1")])

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 0
    assert "nothing to sync" in result.output


def test_sync_skips_repos_without_a_remote(workspace: Path, origin: Path) -> None:
    # A checkout the user maintains themselves must be left alone even
    # when another repo in the same workspace is synced.
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    runner.invoke(app, ["add-repo", "repo1", str(workspace / "repo1")])
    runner.invoke(app, ["add-repo", "cloned", str(workspace / "cloned"), "--remote", str(origin)])

    result = runner.invoke(app, ["sync", "--no-index"])

    assert result.exit_code == 0
    assert "cloned: cloned" in result.output
    assert "repo1:" not in result.output


def test_sync_still_indexes_after_a_skip_but_exits_nonzero(workspace: Path, origin: Path) -> None:
    # A repo it declined to touch is still a repo worth indexing — the
    # working copy just is not the remote's. The exit code carries the
    # warning so a script notices; the index run happens regardless.
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    runner.invoke(app, ["add-repo", "repo1", str(workspace / "repo1"), "--remote", str(origin)])
    (workspace / "repo1" / "scratch.py").write_text("print('scratch')\n")

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 1
    assert "uncommitted changes" in result.output
    assert "files:" in result.output  # the index run still happened
    # `--lang python`, not top-1: since step 27 the corpus holds commit
    # messages too, and with the fake embedder a commit can outscore the
    # file being looked for. The claim is that the file was indexed.
    found = runner.invoke(app, ["search", "scratch", "-k", "5", "--lang", "python"])
    assert "scratch" in found.output


def test_a_malformed_config_is_an_error_not_a_traceback(workspace: Path) -> None:
    # The module's own rule: a traceback in the output is always a bug.
    # Duplicating a section is easy to do by appending a snippet from a
    # README, and it used to produce a raw TOMLDecodeError traceback.
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    config = workspace / CONFIG_FILE
    config.write_text(config.read_text() + "\n[references]\n")

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "cannot read the wsindex config" in result.output
    assert "Traceback" not in result.output


def test_a_config_missing_a_required_section_is_an_error_too(workspace: Path) -> None:
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    config = workspace / CONFIG_FILE
    config.write_text('[workspace]\nname = "ws"\nbackend = "local"\n')

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 1
    assert "cannot read the wsindex config" in result.output


# --- step 29a: wsindex fetch ---------------------------------------------


def test_fetch_prints_a_document(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from wsindex.connectors import Document

    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    config = workspace / CONFIG_FILE
    config.write_text(
        config.read_text() + '\n[[connectors]]\ntype = "generic-http"\nurl_pattern = "https://*"\n'
    )
    monkeypatch.setattr(
        "wsindex.connectors.http.GenericHttpConnector.fetch",
        lambda self, url: Document(
            url=url, title="Guide", text="body text", metadata={"content_type": "text/html"}
        ),
    )

    result = runner.invoke(app, ["fetch", "https://example.invalid/guide"])
    assert result.exit_code == 0
    assert "title: Guide" in result.output
    assert "content_type: text/html" in result.output
    assert "body text" in result.output


def test_fetch_says_when_nothing_claims_the_url(workspace: Path) -> None:
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    result = runner.invoke(app, ["fetch", "https://example.invalid/guide"])
    assert result.exit_code == 1
    assert "no connector claims" in result.output


def test_fetch_reports_a_connector_error(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from wsindex.connectors import DocumentNotFound

    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    config = workspace / CONFIG_FILE
    config.write_text(
        config.read_text() + '\n[[connectors]]\ntype = "generic-http"\nurl_pattern = "https://*"\n'
    )

    def missing(self: object, url: str) -> None:
        raise DocumentNotFound(f"{url} is not there, or not visible")

    monkeypatch.setattr("wsindex.connectors.http.GenericHttpConnector.fetch", missing)
    result = runner.invoke(app, ["fetch", "https://example.invalid/gone"])
    assert result.exit_code == 1
    assert "not there, or not visible" in result.output
    assert "Traceback" not in result.output


def test_fetch_needs_a_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path / "absent.toml"))
    result = runner.invoke(app, ["fetch", "https://example.invalid/a"])
    assert result.exit_code == 1
    assert "wsindex init" in result.output


# --- snapshot repos: sync materializes, index reads them unchanged --------


class _StubConnector(Connector):
    """One document, so the CLI loop can be driven without the network."""

    def matches(self, url: str) -> bool:
        return url.startswith("https://")

    def fetch(self, url: str) -> Document:
        if url.endswith("/missing"):
            raise DocumentNotFound(f"{url} is not there, or not visible")
        return Document(
            url=url,
            title="Retention policy",
            text="Snapshots are pruned after ninety days.",
            metadata={"source": "stub"},
        )


@pytest.fixture
def stub_connector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(BUILTIN, "stub", _StubConnector)


def declare_connector(workspace: Path) -> None:
    """Append a `[[connectors]]` entry — the syntax the README documents."""
    config = workspace / CONFIG_FILE
    config.write_text(
        config.read_text() + '\n[[connectors]]\ntype = "stub"\nurl_pattern = "https://*"\n'
    )


def test_add_repo_registers_a_snapshot(workspace: Path) -> None:
    runner.invoke(app, ["init", "ws", "--provider", "fake"])

    result = runner.invoke(
        app,
        ["add-repo", "docs", "snap", "--source", "connector", "--url", "https://example.com/a"],
    )

    assert result.exit_code == 0
    entry = tomllib.loads((workspace / CONFIG_FILE).read_text())["repos"][0]
    assert entry["source"] == "connector"
    assert entry["urls"] == ["https://example.com/a"]


def test_a_repo_cannot_be_both_pulled_and_materialized(workspace: Path) -> None:
    runner.invoke(app, ["init", "ws", "--provider", "fake"])

    result = runner.invoke(
        app,
        [
            "add-repo",
            "docs",
            "snap",
            "--source",
            "connector",
            "--remote",
            "https://x.invalid/r.git",
        ],
    )

    assert result.exit_code == 1
    assert "not both" in result.output
    assert not tomllib.loads((workspace / CONFIG_FILE).read_text())["repos"]


def test_sync_materializes_a_snapshot_and_index_reads_it(
    workspace: Path, stub_connector: None
) -> None:
    # The point of the whole step: nothing in `index` or `search` knows
    # that this repo was fetched rather than checked out.
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    declare_connector(workspace)
    runner.invoke(
        app,
        [
            "add-repo",
            "docs",
            "snap",
            "--source",
            "connector",
            "--url",
            "https://wiki.example.com/retention",
        ],
    )

    synced = runner.invoke(app, ["sync"])

    assert synced.exit_code == 0
    assert "docs: 1 added" in synced.output
    assert (workspace / "snap" / "wiki.example.com" / "retention.md").exists()
    found = runner.invoke(app, ["search", "how long are snapshots kept"])
    assert "wiki.example.com/retention.md" in found.output


def test_a_second_sync_of_unchanged_documents_commits_nothing(
    workspace: Path, stub_connector: None
) -> None:
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    declare_connector(workspace)
    runner.invoke(
        app,
        [
            "add-repo",
            "docs",
            "snap",
            "--source",
            "connector",
            "--url",
            "https://wiki.example.com/a",
        ],
    )
    runner.invoke(app, ["sync"])

    result = runner.invoke(app, ["sync"])

    assert "docs: up to date (1 documents)" in result.output
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=workspace / "snap",
        check=True,
        capture_output=True,
        text=True,
    )
    assert len(log.stdout.splitlines()) == 1


def test_sync_names_the_document_it_could_not_fetch(workspace: Path, stub_connector: None) -> None:
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    declare_connector(workspace)
    runner.invoke(
        app,
        [
            "add-repo",
            "docs",
            "snap",
            "--source",
            "connector",
            "--url",
            "https://wiki.example.com/a",
            "--url",
            "https://wiki.example.com/missing",
        ],
    )

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 1
    assert "1 failed" in result.output
    assert "https://wiki.example.com/missing" in result.output
    # The one that worked still landed: a failure is per document.
    assert (workspace / "snap" / "wiki.example.com" / "a.md").exists()


def test_sync_reports_a_snapshot_it_could_not_write(workspace: Path, stub_connector: None) -> None:
    # The directory exists and holds something that is not ours, so
    # materialization refuses it — per repo, like an unusable remote.
    runner.invoke(app, ["init", "ws", "--provider", "fake"])
    declare_connector(workspace)
    (workspace / "snap").mkdir()
    (workspace / "snap" / "notes.txt").write_text("mine")
    runner.invoke(
        app,
        [
            "add-repo",
            "docs",
            "snap",
            "--source",
            "connector",
            "--url",
            "https://wiki.example.com/a",
        ],
    )

    result = runner.invoke(app, ["sync"])

    assert result.exit_code == 1
    assert "docs: failed" in result.output
    assert (workspace / "snap" / "notes.txt").exists()
