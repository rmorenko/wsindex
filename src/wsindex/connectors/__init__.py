"""Fetching one external document, and deciding who fetches it.

A connector is a *pointed pull*, not a crawler: given a url — one a
reference pointed at, or one a person typed — bring back that document.
Nothing walks a wiki space, which is what keeps the cost of an external
source proportional to what the repository mentions.

Which connector answers a url is configuration:

    [[connectors]]
    type = "github"
    url_pattern = "https://github.com/myorg/*"
    token_env = "GITHUB_TOKEN"

Secrets never live in the config — the discipline the S3 backend keeps
(ADR-7). The config names an *environment variable*.

`DocumentNotFound` is deliberately vague about why: GitHub answers 404
for a repository you lack access to exactly as for one that does not
exist, so claiming "no such document" would be a guess.
"""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase


class ConnectorError(RuntimeError):
    """A connector could not do its job.

    A subclass of RuntimeError rather than ValueError: the config was
    fine, the world did not cooperate. The CLI turns it into a message.
    """


class DocumentNotFound(ConnectorError):
    """The url named nothing this connector could reach.

    Deliberately vague about *why*, because the source often is. GitHub
    answers 404 for a repository you lack access to exactly as it does
    for one that does not exist — a policy, not an accident — so
    claiming "no such document" would be a guess.
    """


@dataclass(frozen=True, kw_only=True)
class Document:
    """One fetched document: the text, and enough to say where it came from.

    Attributes:
        url: The canonical human-facing url. Not always the one asked
            for — a source may normalize, and the canonical one is what
            belongs in a link.
        title: Short human name. Empty when the source has none.
        text: The document body. Markdown where the source speaks it,
            otherwise plain text; turning richer formats into markdown
            is `wsindex.snapshot`'s job, not the fetch's.
        metadata: Whatever the source knows about the document — author,
            dates, state. Strings throughout, because this ends up in
            frontmatter or a sidecar, and a schema per source would be a
            schema the core has to know.
    """

    url: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ConnectorSpec:
    """One `[[connectors]]` entry: who to use, for which urls, with what.

    Attributes:
        type: Which connector implementation — a key of `BUILTIN`, or a
            name a plugin registers.
        url_pattern: Glob the url must match, `*` and `?` as in a shell.
            A glob rather than a regex because these are written by hand
            in a config file, and `https://github.com/myorg/*` is what
            someone means.
        token_env: Name of the environment variable holding the token,
            or None for a source that needs none. The *name*, never the
            value: a config file is committed, a token is not.
    """

    type: str
    url_pattern: str
    token_env: str | None = None

    def matches(self, url: str) -> bool:
        """True when this entry claims the url."""
        return fnmatchcase(url, self.url_pattern)

    def token(self) -> str | None:
        """The configured token, read from the environment.

        Raises:
            ConnectorError: A token variable was named and is not set.
                Silently falling back to an anonymous request is the
                worse failure: GitHub then answers 404, which reads as
                "no such document" and sends the user looking in the
                wrong place entirely.
        """
        if self.token_env is None:
            return None
        value = os.environ.get(self.token_env)
        if not value:
            raise ConnectorError(
                f"${self.token_env} is not set, and the connector for "
                f"{self.url_pattern!r} is configured to use it"
            )
        return value


class Connector(ABC):
    """Fetches one document from one kind of source.

    Two methods, because routing and fetching are different questions:
    `matches` is asked of every connector until one says yes, and only
    then is `fetch` called. Keeping them apart is what lets a url be
    routed without anything being requested over the network.
    """

    def __init__(self, spec: ConnectorSpec) -> None:
        """Bind the connector to the config entry that selected it.

        Args:
            spec: The `[[connectors]]` entry; carries the token name and
                the pattern this instance answers for.
        """
        self.spec = spec

    @abstractmethod
    def matches(self, url: str) -> bool:
        """True when this connector can fetch the url.

        Narrower than the config's pattern: the pattern says which urls
        the *user* routed here, this says which the connector actually
        understands. GitHub's connector answers for an issue url and not
        for a repository's front page, whatever the pattern allows.
        """

    @abstractmethod
    def fetch(self, url: str) -> Document:
        """Retrieve one document.

        Args:
            url: The document's url; `matches` has already said yes.

        Returns:
            The document, with the source's own metadata attached.

        Raises:
            DocumentNotFound: The source has nothing there — or nothing
                this token may see.
            ConnectorError: The source could not be reached, answered
                something unusable, or refused.
        """


ConnectorFactory = Callable[[ConnectorSpec], Connector]

BUILTIN: dict[str, ConnectorFactory] = {}
"""Connector implementations shipped with wsindex, by `type` name.

Filled at the bottom of this module, after the implementations import.
Plugin entry points are merged into the same map on first use (step
29v), which is why it is a plain dict rather than a set of imports at
each call site."""


_plugins_loaded = False
_plugin_lock = threading.Lock()
"""Held while plugins load. The flag alone is check-then-act, and the
window between reading it and setting it is a couple of bytecodes wide —
narrow enough that four threads racing it never hit it in a measurement,
and real enough to close for the price of one uncontended acquire per
process. Two threads that did hit it would both load, and the second
would report every type as "already registered by another plugin": a
conflict that does not exist."""


def _ensure_plugins() -> None:
    """Load installed connector plugins once, before the first routing.

    Not at import time, which is where this started and where it does
    not work. A plugin must `from wsindex.connectors import Connector` at
    module level to subclass it; if importing that package ended by
    importing plugins, a program that imported the *plugin* first would
    re-enter it half-executed and the entry point would resolve to
    nothing. The loader would report it — "import failed, partially
    initialized module" — and skip, which is a silently disabled plugin
    dressed up as a warning nobody reads.

    Deferring to the first `route` breaks the cycle: by the time a url
    needs a connector, every module involved has finished importing.
    """
    global _plugins_loaded
    if _plugins_loaded:
        return
    with _plugin_lock:
        if _plugins_loaded:
            return
        # Set inside the lock and before loading: a plugin's import may
        # reach back into this module, and once round the loop is enough.
        _plugins_loaded = True
        from wsindex.connectors.plugins import load_connectors

        load_connectors(BUILTIN)


def route(
    url: str,
    specs: Sequence[ConnectorSpec],
    *,
    registry: Mapping[str, ConnectorFactory] | None = None,
) -> Connector | None:
    """The connector configured to answer this url, or None.

    First match wins, in config order — the same rule a routing table
    anywhere keeps, and the one a person writing the file expects. A
    spec whose `type` names nothing installed is skipped rather than
    raising: the user may have a plugin on another machine, and one
    unusable entry must not disable the rest.

    Args:
        url: The url to route.
        specs: `Config.connectors`, in file order.
        registry: Type name -> implementation; defaults to the
            process-wide `BUILTIN`, which is what every caller wants.
            Injectable for the same reason `load_connectors` takes one:
            a test that routed through the global table would have to
            mutate it.

    Returns:
        A connector bound to the winning spec, or None when nothing
        claims the url or the claimant does not understand it.
    """
    if registry is None:
        _ensure_plugins()
    for spec in specs:
        if not spec.matches(url):
            continue
        factory = (BUILTIN if registry is None else registry).get(spec.type)
        if factory is None:
            continue
        connector = factory(spec)
        if connector.matches(url):
            return connector
    return None


from wsindex.connectors.github import GitHubConnector  # noqa: E402
from wsindex.connectors.http import GenericHttpConnector  # noqa: E402

BUILTIN.update(
    {
        # Order is documentation, not behaviour — routing is by config.
        # generic-http first because it is the fallback shape: anything
        # that is already text needs no source-specific knowledge.
        "generic-http": GenericHttpConnector,
        "github": GitHubConnector,
    }
)

SHIPPED = frozenset(BUILTIN)
"""The type names that came in the box, frozen before any plugin loads.

A separate name from `BUILTIN` because the two answer different
questions: `BUILTIN` is "what can this process route to", which grows
with what is installed, and this is "what does wsindex itself claim",
which does not. A plugin may not take one of these — see
`wsindex.connectors.plugins`."""

__all__ = [
    "BUILTIN",
    "Connector",
    "ConnectorError",
    "ConnectorFactory",
    "ConnectorSpec",
    "Document",
    "DocumentNotFound",
    "GenericHttpConnector",
    "GitHubConnector",
    "route",
]
