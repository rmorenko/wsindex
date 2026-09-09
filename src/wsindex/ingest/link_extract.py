"""Finding links in chunks that were just produced.

Two rules. The first is the one the step-26 spike measured at zero false
alarms:
**code names a port, configuration publishes one.** It is the rule that
would have caught the 8080/8000 bug — `config.py` defaulting to
`http://localhost:8080` while `docker-compose.yml` published 8000 — and
it is deliberately narrow. ADR-9 lists the other edge kinds and the
evidence each still owes.

The second (step 27b) reaches *outside* the repository: a commit message
or a document naming a ticket, an issue or a url. Nothing is downloaded —
it is a cheap bridge, so "why" can reach a ticket without a connector,
and stage 10b can bring the contents later.

Which prefixes count is entirely the workspace's to say. A built-in
`PROJ-123` rule is not possible: measured on this repository it matched
198 times and every hit was an internal number — ADR-7, FR-111 — not a
ticket. See `Config.references`.

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
from collections.abc import Iterable, Mapping

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

_BARE_URL = re.compile(r"https?://[^\s<>\"'`]+")
"""A url written out in prose. Self-resolving — it needs no template and
no configuration, which is why it is the one reference kind that works
out of the box."""

_URL_TRAILERS = ").,;:!?`'\"]>"
"""Punctuation a url in prose collects and does not own. Measured: without
stripping these, a link stored from a markdown sentence came out as
`http://localhost:8080`,` — a url that resolves to nothing."""


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


def _reference_links(chunk: Chunk, templates: Mapping[str, str]) -> list[Link]:
    """External references in one commit message or document.

    A reference is recorded only when it can be *resolved*: a bare url,
    which resolves to itself, or a configured prefix, which resolves
    through its template. Recording an unresolvable `#123` would be
    recording noise — the whole value of this edge is that it reaches
    something.

    Args:
        chunk: A COMMIT or DOC chunk.
        templates: `Config.references`, prefix -> url template.

    Returns:
        One link per reference found, url filled in.
    """
    found: list[Link] = []
    for offset, line in enumerate(chunk.text.splitlines()):
        for match in _BARE_URL.finditer(line):
            url = match.group(0).rstrip(_URL_TRAILERS)
            found.append(
                Link(
                    src_chunk_id=chunk.id,
                    kind=LinkKind.REFERENCES,
                    name=url,
                    line=chunk.start_line + offset,
                    url=url,
                )
            )
        for prefix, template in templates.items():
            for digits in re.findall(re.escape(prefix) + r"(\d+)\b", line):
                found.append(
                    Link(
                        src_chunk_id=chunk.id,
                        kind=LinkKind.REFERENCES,
                        name=f"{prefix}{digits}",
                        line=chunk.start_line + offset,
                        url=template.format(key=digits),
                    )
                )
    return found


def links_for(
    chunks: Iterable[Chunk], *, references: Mapping[str, str] | None = None
) -> list[Link]:
    """Links found in one file's freshly built chunks.

    What a chunk contributes follows its `kind`, which is the walker's
    answer to "what sort of file is this". Code *references* a port and
    configuration *declares* one; prose is held to neither, because a
    sentence naming a port is not a claim anyone can be held to. What
    prose does carry is pointers outward, so documents and commit
    messages contribute `REFERENCES` instead.

    Args:
        chunks: The chunks a single file just produced. Each carries the
            id the link will be anchored to and the line it starts at.
        references: `Config.references`. Commit messages and documents
            are scanned for external references against it; without it
            only bare urls are found, since nothing else can be resolved.

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
            # COMMIT and DOC. Prose points outward rather than making
            # claims about ports, so it contributes the other edge kind.
            found += _reference_links(chunk, references or {})
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
