"""Commands that read the index: search, refs, why."""

import re
from typing import Annotated

import typer

from wsindex.cli.composition import (
    build_links,
    build_pipeline,
    config_or_default,
    require_config_file,
)
from wsindex.links import KIND_LABELS, LinkKind
from wsindex.model import Kind, SearchFilter
from wsindex.pipeline import Authorship
from wsindex.ui import render_hits


def search(
    query: str,
    top: Annotated[int, typer.Option("--top", "-k", help="How many hits")] = 10,
    repo: Annotated[str | None, typer.Option("--repo", help="Restrict to a single repo id")] = None,
    lang: Annotated[
        list[str] | None,
        typer.Option("--lang", help="Restrict to a language (repeat for OR)"),
    ] = None,
    kind: Annotated[
        list[Kind] | None,
        typer.Option("--kind", help="Restrict to a Kind (code/config/doc; repeat for OR)"),
    ] = None,
    path: Annotated[
        str | None,
        typer.Option("--path", help="Path glob (`*`, `?` wildcards)"),
    ] = None,
    symbol: Annotated[
        str | None, typer.Option("--symbol", help="Substring of the chunk symbol")
    ] = None,
) -> None:
    """Search all indexed repos, best hits first.

    Scope flags stack: --repo narrows the dataset list, structural
    filters (--lang/--kind/--path/--symbol) go down to the store as a
    prefilter (top-k over the filtered subset, not slashed out of it).
    """
    pipeline = build_pipeline()
    candidate = SearchFilter(
        lang=tuple(lang or ()),
        kind=tuple(kind or ()),
        path=path,
        symbol=symbol,
    )
    filters: SearchFilter | None = None if candidate.is_empty else candidate
    try:
        hits = pipeline.search(query, k=top, repo=repo, filters=filters)
        skipped = pipeline.unsearched(repo)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if skipped:
        # Before the results, not after: this changes how they should be
        # read. A repo that was never indexed takes no part in any
        # search, so "nothing found there" was never something this
        # answer had the standing to say.
        typer.echo(
            "warning: not searched (never indexed): "
            + ", ".join(skipped)
            + " — run `wsindex index`",
            err=True,
        )
    if not hits:
        typer.echo("no results")
        return
    render_hits(hits)


def refs(name: str) -> None:
    """Everything that names something: a port, a ticket, a commit, a url.

    The inverted index over the links `index` recorded. Ask it about a
    port and it answers who reads it and who publishes it; about a
    ticket, which commits mention it; about a function, where it is
    defined and which files name it.

    That last one is not "who calls this": a name in a comment or a
    string counts, and nothing resolves which definition a use refers to.
    Code-to-code *call* edges remain deferred until they can be shown to
    pay for their noise (ADR-9 measured 11% of resolvable call names as
    ambiguous). What is here is the cheaper claim — this name occurs
    here — which is a search result rather than a call graph.
    """
    config = config_or_default()
    require_config_file(config)
    with build_links(config) as links:
        edges = links.by_name(name)
    if not edges:
        typer.echo(f"no links named {name!r}")
        return
    typer.echo(name)
    for kind, label in KIND_LABELS.items():
        group = [edge for edge in edges if edge.kind is kind]
        if not group:
            continue
        typer.echo(f"  {label}:")
        for edge in group:
            suffix = f"  -> {edge.url}" if edge.url else ""
            typer.echo(f"    {edge.repo}/{edge.path}:{edge.line}{suffix}")
    if any(edge.kind is LinkKind.READS_KEY for edge in edges) and not any(
        edge.kind is LinkKind.DECLARES for edge in edges
    ):
        # The drift report, narrowed to one name. Worth saying here too:
        # someone asking about a port is exactly who needs to know.
        typer.echo("  (nothing declares it — code and configuration have drifted)")


def why(symbol: str) -> None:
    """Why a definition looks the way it does: the commits that wrote it.

    Definition -> blame edges -> commit messages, plus whatever those
    commits pointed at outside the repository. The reasoning behind a
    design decision usually lives in a commit message and nowhere else;
    this is the path to it.
    """
    config = config_or_default()
    require_config_file(config)
    definitions = build_pipeline().why(symbol)
    if not definitions:
        # Exit 0, like `refs`. Looking and not finding is an answer, and
        # the two commands used to disagree about that — `why` exited 1
        # where `refs` exited 0 for the same situation, which is the kind
        # of difference a script discovers the hard way. Code 1 is kept
        # for "could not look".
        typer.echo(f"no definition found for {symbol!r}")
        return
    for definition in definitions:
        typer.echo(f"{definition.hit.symbol}  {definition.hit.location}")
        if not definition.commits:
            typer.echo("  (no commit recorded — run `wsindex index` to build blame edges)")
            continue
        typer.echo("  written by:")
        for author in definition.commits:
            _echo_commit(author)
        typer.echo("")


_TRAILER = re.compile(r"^[A-Z][A-Za-z-]+:\s")
"""A git trailer — `Co-Authored-By:`, `Signed-off-by:`, `Reviewed-by:`.
Metadata about the commit, not the reasoning behind the code, and `why`
asks about the latter."""


def _reasoning(message: str) -> list[str]:
    """A commit message with its trailers cut off, blank lines dropped.

    The body is the answer `why` exists to give, so it is kept whole —
    but a wall of `Co-Authored-By` at the end is noise between the reader
    and the next commit.
    """
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    while len(lines) > 1 and _TRAILER.match(lines[-1]):
        lines.pop()
    return lines


def _echo_commit(author: Authorship) -> None:
    """One commit behind a definition: its subject, then the rest of it."""
    if author.message is None:
        # Blamed to a commit an earlier run indexed, or one outside the
        # window. Knowing which commit still answers "when did this
        # change" — see `blame_links`.
        typer.echo(f"    {author.commit}  (message not indexed)")
        return
    lines = _reasoning(author.message)
    typer.echo(f"    {author.commit}  {lines[0]}")
    for line in lines[1:]:
        typer.echo(f"        {line}")
    for reference in author.references:
        typer.echo(f"        see {reference.name} -> {reference.url}")
