"""MCP: the workspace index as tools an agent can call.

The answer to "is there an IDE plugin": an agent client speaks MCP
already, so no plugin has to exist. Point it at `wsindex mcp`.

Three tools — `search`, `refs`, `why` — the three commands worth calling
from outside. `index` is deliberately absent: a tool an agent may call
again without thinking should not be minutes of CPU and somebody's git
remotes.

`build()` knows nothing about transport. `wsindex mcp` runs it over
stdio, which is what an editor spawns; `wsindex serve` mounts the same
object on HTTP. One implementation of the tools, two ways in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

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
        from wsindex.cli import build_pipeline

        pipeline = build_pipeline()
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
        # Named, because an agent reporting "there is no such code" on a
        # workspace half of which was never indexed is worse than an
        # agent that says it does not know.
        skipped = engine.unsearched(repo)
        return {
            "count": len(hits),
            "hits": [hit.to_json() for hit in hits],
            "unsearched": list(skipped),
        }

    @server.tool()
    def refs(
        name: Annotated[str, Field(description="A port, ticket, commit sha or url")],
    ) -> dict[str, Any]:
        """Everything that names this: who reads a port, which commits mention a ticket."""
        with LinkStore(engine.config.index_dir) as links:
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
        with LinkStore(engine.config.index_dir) as links:
            for hit in found:
                commits = []
                for edge in links.out_of([str(hit.native_id)], kind=LinkKind.BLAMED_BY):
                    message = (
                        engine.commit_message(edge.repo, edge.dst_chunk_id)
                        if edge.dst_chunk_id is not None
                        else None
                    )
                    commits.append({"commit": edge.name, "message": message})
                definitions.append(
                    {
                        "symbol": hit.symbol,
                        "repo": hit.repo,
                        "path": hit.path,
                        "start_line": hit.start_line,
                        "end_line": hit.end_line,
                        "commits": commits,
                    }
                )
        return {"symbol": symbol, "definitions": definitions}

    return server


__all__ = ["INSTRUCTIONS", "build"]
