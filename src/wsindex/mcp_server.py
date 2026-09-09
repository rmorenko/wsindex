"""MCP: the workspace index as tools an agent can call.

The answer to the "IDE plugin" the scope excluded. An agent client —
Claude Code, an editor's assistant — speaks MCP already, so a plugin per
editor is a plugin nobody has to write: point the client at
`wsindex mcp` and it can search the workspace.

Three tools, and they are the three commands worth calling from outside:

- `search` — the same funnel as `wsindex search`, filters included.
- `refs` — everything that names a port, a ticket, a commit, a url.
- `why` — the commits that wrote a definition, and their reasoning.

`index` is deliberately absent. A tool an agent can call should be one it
can call again without thinking, and re-indexing a workspace is minutes
of CPU and somebody's git remotes. Indexing is a decision, and the CLI,
the scheduler and `POST /index` are all better places to make it.

Two transports, one set of tools
--------------------------------
`build()` returns a `FastMCP` and knows nothing about how it is reached.
`wsindex mcp` runs it over stdio, which is what an editor spawns; the
same object also exposes a streamable-HTTP app, so a workspace already
running `wsindex serve` can offer MCP on the same port. The
tool bodies are written once, which is the whole point of building it
this way rather than twice.

The pipeline is built once and shared, exactly as the HTTP server builds
it — and searches refresh the store first (ADR-10), so an agent holding
a long session does not answer from the corpus as it was when the editor
started.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from wsindex.config import Config
from wsindex.links import KIND_LABELS, LinkKind, LinkStore
from wsindex.model import Kind, SearchFilter

if TYPE_CHECKING:  # pragma: no cover - import-time only, for annotations
    from mcp.server.fastmcp import FastMCP

    from wsindex.pipeline import Pipeline

INSTRUCTIONS = """\
Semantic search over the repositories of a developer workspace. Results
carry exact file:line locations, so quote them rather than paraphrasing.

Use `search` for "where/how does X work" questions; narrow it with
`repo`, `lang`, `kind` or `path` when the workspace is large. Use `refs`
to find everything that names a port, a ticket or a commit, and `why` to
find the commits that wrote a definition — that is usually where the
reasoning behind a design lives.
"""


def build(pipeline: Pipeline | None = None) -> FastMCP:
    """Assemble the MCP server over one shared Pipeline.

    Args:
        pipeline: Already built, or None to build the CLI's — the same
            composition root the server and the shell use, so no
            interface gets its own engine.

    Returns:
        The server, ready for either transport.
    """
    from mcp.server.fastmcp import FastMCP

    if pipeline is None:
        from wsindex.cli import _build_pipeline

        pipeline = _build_pipeline()
    engine = pipeline
    server: FastMCP = FastMCP(name="wsindex", instructions=INSTRUCTIONS)

    @server.tool()
    def search(
        query: Annotated[str, Field(description="What to look for, in natural language")],
        k: Annotated[int, Field(description="How many hits", ge=1, le=50)] = 10,
        repo: Annotated[str | None, Field(description="Restrict to one repo id")] = None,
        lang: Annotated[list[str] | None, Field(description="Languages (OR)")] = None,
        kind: Annotated[
            list[str] | None, Field(description="code, config, doc or commit (OR)")
        ] = None,
        path: Annotated[str | None, Field(description="Path glob, e.g. src/*.py")] = None,
        symbol: Annotated[str | None, Field(description="Substring of the symbol name")] = None,
    ) -> dict[str, Any]:
        """Search the workspace by meaning; returns chunks with file:line."""
        try:
            kinds = tuple(Kind(value) for value in kind or ())
        except ValueError as exc:
            # Named back to the caller: an agent can fix a wrong enum on
            # its next turn if it is told which values exist.
            raise ValueError(f"kind must be one of {', '.join(k.value for k in Kind)}") from exc
        candidate = SearchFilter(lang=tuple(lang or ()), kind=kinds, path=path, symbol=symbol)
        hits = engine.search(
            query, k=k, repo=repo, filters=None if candidate.is_empty else candidate
        )
        return {"count": len(hits), "hits": [hit.to_json() for hit in hits]}

    @server.tool()
    def refs(
        name: Annotated[str, Field(description="A port, ticket, commit sha or url")],
    ) -> dict[str, Any]:
        """Everything that names this: who reads a port, which commits mention a ticket."""
        with LinkStore(Config().index_dir) as links:
            edges = links.by_name(name)
        found = [
            {
                "relation": KIND_LABELS[edge.kind],
                "repo": edge.repo,
                "path": edge.path,
                "line": edge.line,
                "url": edge.url,
            }
            for edge in edges
        ]
        drifted = any(edge.kind is LinkKind.READS_KEY for edge in edges) and not any(
            edge.kind is LinkKind.DECLARES for edge in edges
        )
        return {"name": name, "count": len(found), "links": found, "unresolved": drifted}

    @server.tool()
    def why(
        symbol: Annotated[str, Field(description="A function, class or method name")],
    ) -> dict[str, Any]:
        """The commits that wrote a definition, and what they said about it."""
        found = engine.search(symbol, k=3, filters=SearchFilter(symbol=symbol))
        if not found:
            return {"symbol": symbol, "definitions": []}
        definitions = []
        with LinkStore(Config().index_dir) as links:
            for hit in found:
                meta = hit.metadata
                commits = []
                for edge in links.out_of([str(hit.native_id)], kind=LinkKind.BLAMED_BY):
                    message = None
                    if edge.dst_chunk_id is not None:
                        message = engine.store.chunk_text(edge.repo, ids=[edge.dst_chunk_id]).get(
                            edge.dst_chunk_id
                        )
                    commits.append({"commit": edge.name, "message": message})
                definitions.append(
                    {
                        "symbol": meta.get("symbol"),
                        "repo": str(meta["repo"]),
                        "path": str(meta["path"]),
                        "start_line": int(meta["start_line"]),
                        "end_line": int(meta["end_line"]),
                        "commits": commits,
                    }
                )
        return {"symbol": symbol, "definitions": definitions}

    return server


__all__ = ["INSTRUCTIONS", "build"]
