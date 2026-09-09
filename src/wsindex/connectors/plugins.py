"""Finding connector plugins: entry points in, registered types out.

The seam the language plugins use (`wsindex.ingest.plugins`), pointed at
a different registry. A distribution advertises a `Connector` subclass
and the name it advertises it under is the `type` a config asks for:

    [project.entry-points."wsindex.connectors"]
    notion = "wsindex_connector_notion:NotionConnector"

Nothing scans a filesystem: a connector becomes available by being
*installed*, and the core keeps no list of them.

Everything checkable is checked at load, because the failure this exists
to prevent is a plugin that installs cleanly and then does nothing. A
type name belongs to whoever registered it first, so a plugin claiming
`github` is skipped — a config saying `type = "github"` was written
against the built-in.

Warnings, not exceptions: someone else's broken package must not cost the
user their workspace.
"""

from __future__ import annotations

import warnings
from importlib.metadata import entry_points

from wsindex.connectors import BUILTIN, SHIPPED, Connector, ConnectorFactory

ENTRY_POINT_GROUP = "wsindex.connectors"
"""The group a plugin declares its connectors under."""


class ConnectorLoadWarning(UserWarning):
    """A connector plugin was skipped. Its own category so callers can filter it."""


def _warn(message: str) -> None:
    warnings.warn(message, ConnectorLoadWarning, stacklevel=3)


def load_connectors(
    registry: dict[str, ConnectorFactory] | None = None,
    *,
    group: str = ENTRY_POINT_GROUP,
) -> tuple[str, ...]:
    """Register every connector advertised under `group`.

    Called once per process, from `route` the first time a url needs a
    connector — not at import time; see `wsindex.connectors._ensure_plugins`
    for the cycle that forbids it.

    Args:
        registry: Where to register; defaults to the process-wide
            `BUILTIN`. Tests pass their own to avoid leaking into the
            rest of the run.
        group: Entry point group to read; overridden in tests.

    Returns:
        The type names actually registered, in load order.
    """
    target = BUILTIN if registry is None else registry
    loaded: list[str] = []
    for entry_point in entry_points(group=group):
        name = entry_point.name
        if name in target:
            # Whoever got there first keeps the name — the built-ins,
            # because they are in place before any plugin loads, and
            # among plugins whichever the metadata yields first.
            # Overwriting would change what an existing `type = "..."`
            # means without anything in the config having changed. The
            # two messages differ because the fixes do: one is a plugin
            # to report, the other is two plugins to choose between.
            clash = (
                "is a built-in connector"
                if name in SHIPPED
                else "is already registered by another plugin"
            )
            _warn(
                f"connector plugin {name!r} skipped: the type {name!r} {clash}, "
                "and a config naming it means the one already there"
            )
            continue
        try:
            obj = entry_point.load()
        except Exception as exc:
            # Deliberately broad: this is third-party code, and every way
            # it can fail to import — a missing dependency, a syntax
            # error, a module-level assert — must cost a warning rather
            # than the whole process.
            _warn(f"connector plugin {name!r} skipped: import failed — {exc}")
            continue
        if not (isinstance(obj, type) and issubclass(obj, Connector)):
            # A plain factory function would work at runtime, but nothing
            # about it can be checked until it is called — and by then
            # the user is looking at a failed fetch instead of a warning
            # naming the plugin. A class is checkable now.
            _warn(
                f"connector plugin {name!r} skipped: expected a Connector subclass, "
                f"got {type(obj).__name__ if not isinstance(obj, type) else obj.__name__}"
            )
            continue
        missing = sorted(getattr(obj, "__abstractmethods__", ()))
        if missing:
            # A subclass that left `matches` or `fetch` abstract cannot be
            # constructed at all, so routing to it would raise a TypeError
            # from inside `route` — far from the plugin that caused it.
            _warn(
                f"connector plugin {name!r} skipped: {obj.__name__} does not implement "
                + ", ".join(missing)
            )
            continue
        target[name] = obj
        loaded.append(name)
    return tuple(loaded)


__all__ = ["ENTRY_POINT_GROUP", "ConnectorLoadWarning", "load_connectors"]
