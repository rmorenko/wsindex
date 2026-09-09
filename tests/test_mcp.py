"""MCP tests: the tools, called the way a client calls them.

Through `FastMCP.call_tool` rather than by invoking the Python functions
directly — the schema, the argument coercion and the error shape are
what an agent actually meets, and calling the closures would skip all
three.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP

from wsindex.config import Config, Provider
from wsindex.embed import FakeEmbedder
from wsindex.links import LinkStore
from wsindex.mcp_server import build
from wsindex.pipeline import Pipeline
from wsindex.store import LanceDBStore

PY_TEXT = "def greet(name):\n    return f'hello {name}'\n"
CONFIG_TEXT = "port = 8080\n"


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.invalid")
    repo = tmp_path / "repo1"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "main.py").write_text(PY_TEXT)
    (repo / "settings.toml").write_text(CONFIG_TEXT)
    for args in (["init", "-q", "--initial-branch=main"], ["add", "-A"], ["commit", "-qm", "one"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    Config.reset()
    config = Config.default("agents", provider=Provider.FAKE)
    config.add_repo("repo1", path=str(repo))
    config.save(tmp_path / "wsindex.toml")
    Config.reset()
    Config(tmp_path / "wsindex.toml")
    return tmp_path


@pytest.fixture
def server(workspace: Path) -> FastMCP:
    # Built the way `wsindex.cli._build_pipeline` builds it: state and
    # links both live in `index_dir`, and the tools read them from there.
    index_dir = Config().index_dir
    store = LanceDBStore(uri=str(index_dir), embedder=FakeEmbedder())
    pipeline = Pipeline(store=store, state_dir=index_dir, links=LinkStore(index_dir))
    pipeline.index()
    return build(pipeline)


async def call(server: FastMCP, tool: str, **arguments: Any) -> Any:
    """Call a tool the way a client does, and hand back the structured result."""
    _, structured = await server.call_tool(tool, arguments)
    return structured


# --- what the client sees --------------------------------------------------


@pytest.mark.anyio
async def test_the_tools_are_the_three_worth_calling(server: FastMCP) -> None:
    # `index` is deliberately absent: a tool an agent may call again
    # without thinking should not be minutes of CPU and somebody's git
    # remotes.
    assert {tool.name for tool in await server.list_tools()} == {"search", "refs", "why"}


@pytest.mark.anyio
async def test_search_returns_hits_with_locations(server: FastMCP) -> None:
    result = await call(server, "search", query="how does it greet", k=5)

    assert result["count"] >= 1
    hit = next(h for h in result["hits"] if h["path"].endswith("main.py"))
    # The same shape the HTTP API returns, so an agent that has seen one
    # recognises the other.
    assert hit["repo"] == "repo1"
    assert hit["start_line"] >= 1
    assert "greet" in hit["text"]


@pytest.mark.anyio
async def test_search_takes_the_same_filters_as_the_cli(server: FastMCP) -> None:
    nothing = await call(server, "search", query="greet", lang=["rust"])
    something = await call(server, "search", query="greet", lang=["python"])

    assert nothing["count"] == 0
    assert something["count"] >= 1


@pytest.mark.anyio
async def test_a_wrong_kind_says_which_ones_exist(server: FastMCP) -> None:
    # An agent can fix this on its next turn only if it is told.
    with pytest.raises(Exception, match="kind must be one of"):
        await call(server, "search", query="x", kind=["prose"])


@pytest.mark.anyio
async def test_refs_answers_who_names_a_port(server: FastMCP) -> None:
    result = await call(server, "refs", name="8080")

    assert result["name"] == "8080"
    assert result["count"] >= 1
    assert {link["relation"] for link in result["links"]} <= {
        "read by",
        "declared by",
        "mentioned in",
        "wrote",
    }


@pytest.mark.anyio
async def test_refs_on_an_unknown_name_is_empty_not_an_error(server: FastMCP) -> None:
    result = await call(server, "refs", name="nothing-names-this")
    assert result == {
        "name": "nothing-names-this",
        "count": 0,
        "links": [],
        "unresolved": False,
    }


@pytest.mark.anyio
async def test_why_walks_from_a_definition_to_its_commits(server: FastMCP) -> None:
    result = await call(server, "why", symbol="greet")

    assert result["symbol"] == "greet"
    definition = result["definitions"][0]
    assert definition["path"].endswith("main.py")
    # The commit that wrote those lines, with the message it carried.
    assert definition["commits"]
    assert definition["commits"][0]["message"].startswith("one")


@pytest.mark.anyio
async def test_why_on_something_that_is_not_there(server: FastMCP) -> None:
    result = await call(server, "why", symbol="no_such_function_anywhere")
    assert result["definitions"] == [] or all(
        d["symbol"] != "no_such_function_anywhere" for d in result["definitions"]
    )
