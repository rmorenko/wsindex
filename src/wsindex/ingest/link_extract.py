"""Finding links in chunks that were just produced.

One rule so far, the one the step-26 spike measured at zero false alarms:
**code names a port, configuration publishes one.** It is the rule that
would have caught the 8080/8000 bug — `config.py` defaulting to
`http://localhost:8080` while `docker-compose.yml` published 8000 — and
it is deliberately the only one here. ADR-9 lists the other edge kinds
and the evidence each still owes.

Why the chunk text and not the parse tree
-----------------------------------------
ADR-9 said edges would come from "a second visitor over the same parse".
Implementing it showed the value rule does not need a parse at all: a
`host:port` inside a string is recognisable in text, and the chunks have
just been built, so their text is already in hand. That is cheaper than
the ADR's plan, not more expensive, and it keeps the extractor out of
`ast_chunks`, which would otherwise have to hand its tree back.

It does widen the net: a port written in a comment is matched too. For a
drift detector that is a feature — a comment claiming the service runs on
8080 has drifted just as badly as code claiming it — and the measurement
below says it costs nothing. A rule that needs the tree (calls, imports)
is when the tree gets threaded through, and not before.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from wsindex.links import Link, LinkKind
from wsindex.model import Chunk, Kind

_URL_PORT = re.compile(r"https?://[\w.\-]+:(\d{2,5})\b")
"""A port a piece of code expects to reach. Anchored to a url on
purpose: a bare number is not a claim about anything, and matching one
would drown the report the moment a file mentions a timeout."""

_PUBLISHED_PORT = re.compile(r"^\s*-?\s*[\"']?(\d{2,5}):(\d{2,5})[\"']?\s*$")
"""A compose port mapping, `"8000:7860"`. The *host* side is what code
can reach, so that is what counts as declared — the container side is an
implementation detail of the service."""

_CONFIG_PORT_KEY = re.compile(r"(?:^|[^\w])port\s*[:=]\s*[\"']?(\d{2,5})\b", re.IGNORECASE)
"""`port: 8000` or `port = 8000` in a config file: a declaration too."""


def _references(text: str) -> Iterable[tuple[str, int]]:
    for offset, line in enumerate(text.splitlines(), start=0):
        for port in _URL_PORT.findall(line):
            yield port, offset


def _declarations(text: str) -> Iterable[tuple[str, int]]:
    for offset, line in enumerate(text.splitlines(), start=0):
        mapping = _PUBLISHED_PORT.match(line)
        if mapping is not None:
            yield mapping.group(1), offset
            continue
        for port in _CONFIG_PORT_KEY.findall(line):
            yield port, offset


def links_for(chunks: Iterable[Chunk]) -> list[Link]:
    """Links found in one file's freshly built chunks.

    Which side a chunk contributes follows its `kind`, which is the
    walker's answer to "what sort of file is this": code *references*,
    configuration *declares*. A doc file contributes neither — prose
    naming a port is not a claim either side can be held to.

    Args:
        chunks: The chunks a single file just produced. Each carries the
            id the link will be anchored to and the line it starts at.

    Returns:
        The links, anchored to the chunks they were found in.
    """
    found: list[Link] = []
    for chunk in chunks:
        if chunk.kind is Kind.CODE:
            kind, pairs = LinkKind.READS_KEY, _references(chunk.text)
        elif chunk.kind is Kind.CONFIG:
            kind, pairs = LinkKind.DECLARES, _declarations(chunk.text)
        else:
            continue
        for name, offset in pairs:
            found.append(
                Link(
                    src_chunk_id=chunk.id,
                    kind=kind,
                    name=name,
                    # The chunk knows where it starts; the offset is
                    # within it. Reporting a file line is the whole
                    # point of storing one.
                    line=chunk.start_line + offset,
                )
            )
    return found
