"""Server tests: a real app over a real pipeline, no network.

`TestClient` drives the actual ASGI application — the routing, the
dependencies and the auth are the things under test, and a stub of them
would only confirm the stub. What is faked is the embedder, as
everywhere else in this suite.

The pipeline is passed in rather than discovered, which is what
`create_app(pipeline_factory=...)` exists for: a test must not build the
CLI's composition root and load a real model.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from wsindex.config import Config, Provider, Repository
from wsindex.connectors import BUILTIN, Connector, Document
from wsindex.embed import FakeEmbedder
from wsindex.pipeline import Pipeline
from wsindex.server import create_app
from wsindex.server.api import Busy, RunLog, Writer
from wsindex.store import LanceDBStore

PY_TEXT = "def greet(name):\n    return f'hello {name}'\n"


class _StubConnector(Connector):
    """One document, so a snapshot sync can be driven without the network."""

    def matches(self, url: str) -> bool:
        return url.startswith("https://")

    def fetch(self, url: str) -> Document:
        return Document(url=url, title="Policy", text="Kept ninety days.", metadata={})


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A config with one small git repo, installed as the process config."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.invalid")
    repo = tmp_path / "repo1"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "main.py").write_text(PY_TEXT)
    for args in (["init", "-q", "--initial-branch=main"], ["add", "-A"], ["commit", "-qm", "one"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    Config.reset()
    config = Config.default("srv", provider=Provider.FAKE)
    config.add_repo(Repository(id="repo1", path=str(repo)))
    config.save(tmp_path / "wsindex.toml")
    Config.reset()
    Config(tmp_path / "wsindex.toml")
    return tmp_path


@pytest.fixture
def pipeline(workspace: Path) -> Pipeline:
    store = LanceDBStore(uri=str(workspace / ".wsindex"), embedder=FakeEmbedder())
    return Pipeline(store=store, state_dir=workspace / "state")


@pytest.fixture
def app(pipeline: Pipeline) -> FastAPI:
    # Held separately from the client: `TestClient.app` is typed as the
    # bare ASGI callable, so reaching for `.state` through it is a lie
    # the type checker is right to reject.
    return create_app(pipeline_factory=lambda: pipeline)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as running:
        yield running


@pytest.fixture
def secured(pipeline: Pipeline) -> Iterator[TestClient]:
    with TestClient(create_app(pipeline_factory=lambda: pipeline, token="s3cret")) as running:
        yield running


# --- the endpoints mirror the CLI ----------------------------------------


def test_health_needs_no_token(secured: TestClient) -> None:
    # A probe is not a reader: a load balancer must be able to ask
    # whether the process is up without holding a credential.
    assert secured.get("/healthz").json() == {"status": "ok"}


def test_index_then_search(client: TestClient) -> None:
    indexed = client.post("/index").json()
    assert indexed["files"] == 1
    assert indexed["written"] > 0

    found = client.get("/search", params={"q": "how does it greet", "k": 3}).json()

    # Not a claim about ranking — the embedder here is the deterministic
    # fake, so the order is arbitrary. The claim is that the file is in
    # the corpus and that a hit carries the fields the CLI prints, so
    # two interfaces describe one result the same way.
    assert found["count"] >= 1
    hit = next(h for h in found["hits"] if h["path"].endswith("main.py"))
    assert hit["repo"] == "repo1"
    assert hit["lang"] == "python"
    assert hit["start_line"] >= 1
    assert "greet" in hit["text"]


def test_search_filters_reach_the_store(client: TestClient) -> None:
    client.post("/index")

    nothing = client.get("/search", params={"q": "greet", "lang": "rust"}).json()
    something = client.get("/search", params={"q": "greet", "lang": "python"}).json()

    assert nothing["count"] == 0
    assert something["count"] >= 1


def test_an_unknown_repo_is_the_callers_mistake(client: TestClient) -> None:
    # The CLI exits 1 on this; 400 is the same sentence in HTTP.
    response = client.get("/search", params={"q": "x", "repo": "nope"})
    assert response.status_code == 400
    assert "nope" in response.json()["detail"]


def test_status_describes_the_workspace_and_its_runs(client: TestClient) -> None:
    client.post("/index")

    status = client.get("/status").json()

    assert status["workspace"] == "srv"
    assert [repo["id"] for repo in status["repos"]] == ["repo1"]
    assert status["indexing"] is False
    assert status["runs"][0]["kind"] == "index"


# --- authentication -------------------------------------------------------


def test_without_a_token_nothing_is_served(secured: TestClient) -> None:
    for method, path in (("get", "/search?q=x"), ("post", "/index"), ("get", "/status")):
        assert getattr(secured, method)(path).status_code == 401


def test_a_wrong_token_is_refused_the_same_way(secured: TestClient) -> None:
    # No hint about which half was wrong: telling one caller "unknown
    # token" and another "no token" has told both something.
    wrong = secured.get("/search?q=x", headers={"Authorization": "Bearer nope"})
    missing = secured.get("/search?q=x")
    assert wrong.status_code == missing.status_code == 401
    assert wrong.json() == missing.json()


def test_the_right_token_is_let_through(secured: TestClient) -> None:
    response = secured.get("/status", headers={"Authorization": "Bearer s3cret"})
    assert response.status_code == 200


# --- one writer ------------------------------------------------------------


def test_a_second_indexing_run_is_refused_not_queued(client: TestClient, app: FastAPI) -> None:
    # ADR-10: two runs of the same repo do the same work twice, and a
    # caller who waits learns nothing. 409, because the resource is in a
    # state that forbids the request.
    app.state.writer._lock.acquire()
    try:
        response = client.post("/index")
    finally:
        app.state.writer._lock.release()

    assert response.status_code == 409
    assert "already in progress" in response.json()["detail"]


def test_the_lock_is_released_when_a_run_fails() -> None:
    writer = Writer()
    with pytest.raises(RuntimeError), writer.held():
        raise RuntimeError("boom")
    assert not writer.busy


def test_the_lock_refuses_rather_than_waits() -> None:
    writer = Writer()
    with writer.held(), pytest.raises(Busy), writer.held():
        pass


# --- the run log -----------------------------------------------------------


def test_the_run_log_keeps_the_last_few() -> None:
    log = RunLog(limit=3)
    for n in range(5):
        log.record("index", 0.0, {"files": n})

    assert [entry["files"] for entry in log.entries] == [4, 3, 2]


# --- the admin page --------------------------------------------------------


def test_the_admin_page_lists_the_repos(client: TestClient) -> None:
    page = client.get("/admin").text
    assert "repo1" in page
    assert "Sync and re-index" in page


def test_the_admin_page_escapes_what_it_shows(pipeline: Pipeline, workspace: Path) -> None:
    # A repo id reaches the page from a config file, and a config file is
    # something a person edits — the page renders it, so it escapes it.
    config = Config()
    config.add_repo(Repository(id="<script>alert(1)</script>", path="/x"))
    with TestClient(create_app(pipeline_factory=lambda: pipeline)) as client:
        page = client.get("/admin").text
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_adding_a_repo_from_the_page_writes_the_config(client: TestClient, workspace: Path) -> None:
    response = client.post(
        "/admin/add-repo",
        data={"repo_id": "added", "path": "/checkouts/added", "remote": ""},
        follow_redirects=False,
    )

    assert response.status_code == 303
    # Written through Config, so the file the CLI reads next is this one.
    Config.reset()
    assert [repo.id for repo in Config(workspace / "wsindex.toml").repos] == ["repo1", "added"]


def test_a_duplicate_repo_id_is_rejected_by_the_page_too(client: TestClient) -> None:
    response = client.post("/admin/add-repo", data={"repo_id": "repo1", "path": "/x", "remote": ""})
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


# --- the scheduler ---------------------------------------------------------


def test_the_hook_syncs_and_indexes(client: TestClient) -> None:
    # The body is ignored on purpose: a push says "something changed",
    # and the run is incremental anyway.
    response = client.post("/hooks/sync", json={"anything": "at all"})

    assert response.status_code == 200
    detail = response.json()
    assert detail["files"] == 1
    # A local checkout has no remote and no connector, so nothing was
    # pulled — and the run log says which kind of run it was.
    assert detail["synced"] == {}
    assert client.get("/status").json()["runs"][0]["kind"] == "sync"


def test_the_hook_is_refused_while_a_run_is_going(client: TestClient, app: FastAPI) -> None:
    app.state.writer._lock.acquire()
    try:
        assert client.post("/hooks/sync").status_code == 409
    finally:
        app.state.writer._lock.release()


def test_no_interval_means_no_ticker(client: TestClient, app: FastAPI) -> None:
    # A server that starts pulling remotes the moment it boots is a
    # surprise, and this one spends somebody's rate limit.
    assert app.state.ticker is None


def test_a_configured_interval_starts_one(pipeline: Pipeline) -> None:
    Config()._data["server"] = {"interval": 3600}
    configured = create_app(pipeline_factory=lambda: pipeline)
    ticker = configured.state.ticker
    try:
        assert ticker is not None
        assert ticker.interval == 3600
    finally:
        ticker.stop()


def test_a_failing_tick_is_logged_and_does_not_kill_the_thread(app: FastAPI) -> None:
    # The ticker is the only thing keeping the index current; a thread
    # that dies on one bad cycle leaves a server answering from a corpus
    # that quietly stops advancing.
    from wsindex.server import scheduler

    ticker = scheduler.Ticker(app, interval=0.01)

    def explode(_: FastAPI) -> dict[str, Any]:
        raise RuntimeError("network gone")

    # Patched by name rather than injected: the ticker calls the module
    # attribute, and that call is what the test is about.
    original = scheduler.sync_and_index
    scheduler.sync_and_index = explode  # type: ignore[assignment]
    try:
        ticker.start()
        deadline = 2.0
        while deadline > 0 and not app.state.runs.entries:
            import time

            time.sleep(0.05)
            deadline -= 0.05
    finally:
        ticker.stop()
        scheduler.sync_and_index = original

    assert app.state.runs.entries
    assert "network gone" in app.state.runs.entries[0]["error"]


def test_the_default_factory_is_the_cli_composition_root(workspace: Path) -> None:
    # `create_app()` with no factory must build the same pipeline the CLI
    # builds, or the two interfaces are two engines (ADR-10). Works here
    # because the workspace config asks for the fake provider.
    with TestClient(create_app()) as client:
        assert client.post("/index").json()["files"] == 1


def test_the_sync_button_runs_one(client: TestClient, app: FastAPI) -> None:
    response = client.post("/admin/sync", follow_redirects=False)

    assert response.status_code == 303  # a refresh must not re-post
    assert app.state.runs.entries[0]["kind"] == "sync"


def test_the_sync_button_on_a_busy_server_just_shows_the_page(
    client: TestClient, app: FastAPI
) -> None:
    app.state.writer._lock.acquire()
    try:
        response = client.post("/admin/sync", follow_redirects=False)
    finally:
        app.state.writer._lock.release()

    assert response.status_code == 303
    assert not app.state.runs.entries


def test_the_page_shows_what_a_sync_did_and_what_failed(client: TestClient, app: FastAPI) -> None:
    app.state.runs.record("sync", 0.0, {"files": 2, "written": 1, "synced": {"a": "updated"}})
    app.state.runs.record("sync", 0.0, {"error": "GitCommandError: no route to host"})

    page = client.get("/admin").text

    assert "no route to host" in page
    assert "a: updated" in page


def test_adding_a_repo_needs_a_config_file(tmp_path: Path) -> None:
    # A server on built-in defaults has nowhere to save; better a clear
    # 400 than a write that silently goes nowhere. The pipeline is built
    # against that same file-less config on purpose: the server writes to
    # the workspace it is serving, not to whatever the singleton holds.
    Config.reset()
    fileless = Config.default("nofile", provider=Provider.FAKE)
    store = LanceDBStore(uri=str(tmp_path / "db"), embedder=FakeEmbedder())
    pipeline = Pipeline(store=store, state_dir=tmp_path / "state", config=fileless)

    with TestClient(create_app(pipeline_factory=lambda: pipeline)) as client:
        response = client.post("/admin/add-repo", data={"repo_id": "x", "path": "/p", "remote": ""})

    assert response.status_code == 400
    assert "no config file" in response.json()["detail"]


def test_a_sync_pulls_a_remote_and_reports_the_outcome(client: TestClient, workspace: Path) -> None:
    origin = workspace / "origin.git"
    origin.mkdir()
    subprocess.run(
        ["git", "init", "-q", "--bare", "--initial-branch=main"],
        cwd=origin,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", str(origin)],
        cwd=workspace / "repo1",
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "push", "-q", "-u", "origin", "main"],
        cwd=workspace / "repo1",
        check=True,
        capture_output=True,
    )
    Config()._data["repos"][0]["remote"] = str(origin)

    synced = client.post("/hooks/sync").json()["synced"]

    assert synced == {"repo1": "up to date"}


def test_a_sync_materializes_a_snapshot_repo(
    client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(BUILTIN, "stub", _StubConnector)
    Config()._data["connectors"] = [{"type": "stub", "url_pattern": "https://*"}]
    Config()._data["repos"].append(
        {
            "id": "docs",
            "path": str(workspace / "snap"),
            "source": "connector",
            "urls": ["https://wiki.example.com/a"],
        }
    )

    synced = client.post("/hooks/sync").json()["synced"]

    assert synced["docs"] == "1 added"
    assert (workspace / "snap" / "wiki.example.com" / "a.md").exists()


def test_an_unreachable_remote_is_reported_not_raised(client: TestClient, workspace: Path) -> None:
    # One broken remote must not strand the repos after it, and a server
    # that died on a network blip would need a human to notice. The
    # index that follows still runs, because the working copy is fine.
    Config()._data["repos"][0]["remote"] = "https://example.invalid/r.git"

    detail = client.post("/hooks/sync").json()

    assert "skipped" in detail["synced"]["repo1"] or "failed" in detail["synced"]["repo1"]
    assert detail["files"] >= 0


def test_a_path_that_is_not_a_checkout_is_the_configs_fault(
    client: TestClient, app: FastAPI, workspace: Path
) -> None:
    # The CLI prints this and exits 1. A 500 would say the server broke,
    # when the answer is in the config file.
    plain = workspace / "plain"
    plain.mkdir()
    Config()._data["repos"][0]["path"] = str(plain)

    response = client.post("/index")

    assert response.status_code == 400
    assert "git repos only" in response.json()["detail"]
    # And it is in the log, or a scheduled run would fail invisibly.
    assert "not a git repository" in app.state.runs.entries[0]["error"]


def test_a_snapshot_that_cannot_be_written_is_reported_per_repo(
    client: TestClient, app: FastAPI, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The directory exists and holds something that is not ours, so
    # materialization refuses it — per repo, like an unreachable remote.
    monkeypatch.setitem(BUILTIN, "stub", _StubConnector)
    (workspace / "taken").mkdir()
    (workspace / "taken" / "notes.txt").write_text("mine")
    Config()._data["connectors"] = [{"type": "stub", "url_pattern": "https://*"}]
    Config()._data["repos"].append(
        {
            "id": "docs",
            "path": str(workspace / "taken"),
            "source": "connector",
            "urls": ["https://wiki.example.com/a"],
        }
    )

    response = client.post("/hooks/sync")

    # Materialization refused the directory, and the index that follows
    # then refuses it too — it is not a checkout. So the call fails, but
    # nothing is lost: the log holds both halves of why.
    assert response.status_code == 400
    entry = app.state.runs.entries[0]
    assert "failed" in entry["synced"]["docs"]
    assert "git repos only" in entry["error"]
    assert (workspace / "taken" / "notes.txt").exists()


def test_a_broken_remote_is_reported_per_repo(client: TestClient, workspace: Path) -> None:
    # A real checkout — so indexing it is fine — pointed at a remote that
    # does not answer. One unreachable remote must not strand the run.
    second = workspace / "second"
    second.mkdir()
    (second / "mod.py").write_text("def other():\n    return 2\n")
    for args in (
        ["init", "-q", "--initial-branch=main"],
        ["add", "-A"],
        ["commit", "-qm", "two"],
        # Configured in git too, or `git fetch` has nothing to try and
        # reach and the sync ends in a skip instead of the failure this
        # is about.
        ["remote", "add", "origin", "https://example.invalid/r.git"],
    ):
        subprocess.run(["git", *args], cwd=second, check=True, capture_output=True)
    Config()._data["repos"].append(
        {"id": "other", "path": str(second), "remote": "https://example.invalid/r.git"}
    )

    synced = client.post("/hooks/sync").json()["synced"]

    # Reported, and the run went on to index the repo that was fine.
    assert "failed" in synced["other"]


def test_the_hook_says_which_path_is_not_a_checkout(client: TestClient, workspace: Path) -> None:
    plain = workspace / "plain"
    plain.mkdir()
    Config()._data["repos"][0]["path"] = str(plain)

    response = client.post("/hooks/sync")

    assert response.status_code == 400
    assert "git repos only" in response.json()["detail"]


# --- MCP over the other transport -----------------------------------------


def test_the_mcp_endpoint_is_where_it_is_documented(client: TestClient) -> None:
    # Regression, found against a real client: the SDK's sub-app routes
    # `/mcp` of its own, so mounting it at `/mcp` naively puts the
    # endpoint at `/mcp/mcp` — and the client reports "Session
    # terminated", which reads as a protocol fault rather than a 404.
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        headers={"Accept": "application/json, text/event-stream"},
    )

    assert response.status_code != 404
    # And nothing is served one level deeper, which is where the bug put it.
    assert client.post("/mcp/mcp", json={}).status_code == 404


def test_a_workspace_without_the_mcp_extra_still_serves(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An optional adapter that is missing must cost its own endpoint, not
    # the server.
    import builtins

    real_import = builtins.__import__

    def refuse(name: str, *args: object, **kwargs: object) -> Any:
        if name == "wsindex.mcp_server":
            raise ImportError("no mcp extra here")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", refuse)
    with TestClient(create_app(pipeline_factory=lambda: pipeline)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.post("/mcp", json={}).status_code == 404
