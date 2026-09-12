r"""Finding links in chunks the chunker has already built.

Three kinds, all cheap because the text is already in hand.

The first is about names: a chunk *defines* the symbol the syntax tree
gave it, and *mentions* the other names it uses. Both sides are stored
unresolved and meet in `by_name`, because this sees one file at a time
and the definition is usually in another repository. It is not a call
graph and does not pretend to be one — a name in a comment counts.

The second stays inside the repository: code *reads* a port or a key,
configuration *declares* one, and a read with no declaration is drift.
Only the pair ADR-9 measured at zero false alarms is here.

The third reaches outside: a commit message or document naming a
ticket, an issue or a url. Nothing is downloaded — it is a bridge, so
"why" can reach a ticket without a connector.

Which prefixes count is the workspace's to say. A built-in `[A-Z]+-\d+`
rule looks like a Jira key and also matches `ADR-7`, `UTF-8` and
`ISO-8601`: measured on this repository it produced 198 matches and not
one was a ticket.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from wsindex.links import OCCURRENCE_ORDER, Link, LinkKind, Occurrence
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

_CONFIG_KEY = re.compile(r"^[\s\-]*[\"']?([A-Za-z_][A-Za-z0-9_.\-]*)[\"']?\s*[:=]")
"""The *name* a config file publishes, not only the value. One pattern
for yaml, toml, json, ini and properties, because what they share is the
only part this needs: a name at the head of a line, then `:` or `=`.

This is where the drift pair was starving. `_declarations` matched ports
and nothing else, so `max_retries: 3` produced no link at all and 8% of
a workspace's config keys were known to the store — measured on
caddyserver, 10 of 120 sampled. Ports were never the interesting half;
they were the half a regular expression could reach.

Safe to be this permissive because it runs only on CONFIG chunks, where
a name before a colon is a key by grammar rather than by guess. That is
also why no `_COMPOUND` test applies here: the filter that separates a
name from a keyword is needed in code and pointless in a config file."""

_MIN_KEY = 3
"""Shorter than `_MIN_NAME` on purpose. Four exists to stop `get`, `run`
and `new` earning edges in code; a config key is a name by grammar, so
the guard is not needed and the cost of it is real — `ssl`, `env`, `dsn`
and `api` are settings people ask about."""

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
"""A token that could be a name. Deliberately not language-aware: the
chunker registers sixteen code grammars and a per-language lexer for each
would be sixteen things to keep right, for an edge that is a search hint
rather than a call graph."""

_COMPOUND = re.compile(r"(?:[a-z0-9][A-Z])|_[A-Za-z0-9]")
"""An internal word boundary — `ServeHTTP`, `read_config`. This is the
filter that stands in for the symbol table `links_for` cannot have, and
it was chosen by measurement. Against an oracle that knew every symbol in
caddyserver, storing *every* token of four characters or more kept all
810 cross-file names and cost 186 100 edges — nineteen per chunk, which
is 1.9M rows on a 100k-chunk workspace and past what this schema sizes
itself for. Requiring a boundary keeps 664 of the 810 and costs 31 839.
Eighteen per cent of the links for a sixth of the rows.

What it rejects is the argument for it: every keyword of every language
here — `return`, `import`, `error`, `string` — is a single lowercase
word, so this needs no per-language stop-list to avoid linking them."""

_COMMENT_LINE = re.compile(r"^\s*(?://|#|\*|/\*|--|;)")
"""A line that is prose about code. Covers the comment openers of every
grammar registered here, which is cheaper than asking sixteen parsers and
wrong only for a `#` that is a shell directive or a C preprocessor line —
neither of which is a use of a symbol either."""

_IMPORT_LINE = re.compile(r"^\s*(?:import|from|package|use|require|include|#include)\b")
"""A line that brings a name into scope rather than using it."""

_QUOTED = re.compile(r"[\"'`]")
"""Cheap gate before the real test: most lines have no quote at all, and
`_in_string` is the one classifier that has to walk a line."""

_MIN_NAME = 4
"""Shorter names are not worth an edge: `get`, `run`, `new`, `id`. With a
boundary already required this is nearly redundant — measured, raising it
to six changed caddyserver by 371 edges of 31 839 — but it is the cheaper
of the two tests and runs first."""

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
    """What one config chunk publishes: the values, and the names.

    Both, and a line gives both — `port: 8000` declares the port 8000
    and also declares that this file has a `port` setting. The first
    answers "who else uses 8000", the second answers "where is this
    setting configured", and only the first was ever recorded.
    """
    for offset, line in enumerate(text.splitlines(), start=0):
        key = _CONFIG_KEY.match(line)
        if key is not None and len(key.group(1)) >= _MIN_KEY:
            yield key.group(1), offset
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


def _in_string(line: str, start: int) -> bool:
    """Is the character at `start` inside a string literal?

    A scan rather than a parse: quote state toggled left to right with
    backslash escapes honoured. It knows nothing of raw strings,
    heredocs or a quote inside a comment, and does not need to — the
    caller asks about comments first, and being wrong labels an edge
    `code` instead of `string`, which changes an order in a report and
    nothing else.
    """
    quote = ""
    escaped = False
    for index, char in enumerate(line):
        if index == start:
            return bool(quote)
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif quote:
            quote = "" if char == quote else quote
        elif char in "\"'`":
            quote = char
    return False


def _occurrence(line: str, *, start: int, end: int) -> Occurrence:
    """What sort of occurrence the name at `start:end` is.

    Decidable from the line alone, which is the property that matters:
    it is why this can be recorded one file at a time, where resolving
    a call to its definition cannot.
    """
    if _COMMENT_LINE.match(line):
        return Occurrence.COMMENT
    if _IMPORT_LINE.match(line):
        return Occurrence.IMPORT
    if _QUOTED.search(line) and _in_string(line, start):
        return Occurrence.STRING
    if line[end:].lstrip().startswith("("):
        return Occurrence.CALL
    return Occurrence.CODE


def _symbol_links(chunk: Chunk) -> list[Link]:
    """What one code chunk defines, and which other names it uses.

    Both sides unresolved, which is the same shape the config pair has
    and for the same reason: this extractor is handed one file, and the
    definition of a name it mentions is usually in another repository
    entirely. Names meet in `by_name` at query time, where the whole
    workspace is visible.

    A name is recorded once per chunk. Recording every occurrence would
    multiply the store by how often a variable is used inside its own
    function, which tells a reader nothing they cannot see once the file
    is open.

    Which one is kept is not "the first": a Python docstring precedes
    the body, so the first occurrence of a name is often prose about it
    while the call is eight lines down. The *best* one is kept instead —
    highest `OCCURRENCE_ORDER`, earliest line breaking a tie — so stored
    is the one worth being sent to, and `via` describes that line.

    Args:
        chunk: A CODE chunk. Its `symbol` is what the syntax tree named
            it, dotted for methods (`Cls.method`).

    Returns:
        At most one `DEFINES`, plus one `MENTIONS` per distinct name.
    """
    found: list[Link] = []
    # The bare name, because that is how a call site spells it. The
    # qualified form is what the store holds for `search --symbol`; here
    # it would join with nothing.
    defined = chunk.symbol.rsplit(".", 1)[-1] if chunk.symbol else None
    if defined and len(defined) >= _MIN_NAME:
        found.append(
            Link(
                src_chunk_id=chunk.id,
                kind=LinkKind.DEFINES,
                name=defined,
                line=chunk.start_line,
            )
        )
    best: dict[str, tuple[int, int, Occurrence]] = {}
    for offset, line in enumerate(chunk.text.splitlines()):
        for match in _IDENTIFIER.finditer(line):
            name = match.group(0)
            if len(name) < _MIN_NAME or name == defined or _COMPOUND.search(name) is None:
                continue
            via = _occurrence(line, start=match.start(), end=match.end())
            candidate = (OCCURRENCE_ORDER[via], offset, via)
            if name not in best or candidate < best[name]:
                best[name] = candidate
    found += [
        Link(
            src_chunk_id=chunk.id,
            kind=LinkKind.MENTIONS,
            name=name,
            line=chunk.start_line + offset,
            via=via,
        )
        for name, (_, offset, via) in best.items()
    ]
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

    Code additionally contributes the name pair — what it defines and
    what it names — which is the only edge here that does not come from
    a regular expression over text, and the reason `refs` has more than
    a handful of answerable names.

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
            found += _symbol_links(chunk)
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
