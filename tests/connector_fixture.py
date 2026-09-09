"""Stand-in connector plugins, loaded through a real EntryPoint.

Lives beside the tests rather than in `src/` so nothing ships it, and is
importable because pytest puts the test directory on `sys.path`. That is
what lets `test_connector_plugins.py` exercise `EntryPoint.load()` for
real instead of stubbing the one step where the mechanism lives.

Each object here is one thing a plugin author can get wrong, plus the
one they can get right.
"""

from wsindex.connectors import Connector, Document


class FixtureConnector(Connector):
    """A plugin that works."""

    def matches(self, url: str) -> bool:
        return url.startswith("https://fixture.invalid/")

    def fetch(self, url: str) -> Document:
        return Document(url=url, title="Fixture", text="Fixture body.", metadata={"source": "fix"})


class OtherConnector(FixtureConnector):
    """A second working one, for load-order cases."""


class HalfDoneConnector(Connector):
    """A subclass that forgot `fetch`, so it can never be constructed."""

    def matches(self, url: str) -> bool:
        return True


class Unrelated:
    """Not a Connector at all — the commonest mistyped entry point."""


NOT_A_CONNECTOR = "wsindex.connectors:Connector"
"""A string where a class belongs: what an author writes when they think
the entry point value is repeated in the module."""
