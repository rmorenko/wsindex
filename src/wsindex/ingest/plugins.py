"""Finding language plugins: entry points in, registered languages out.

A distribution advertises objects for other packages to find by
declaring entry points in its build config:

    [project.entry-points."wsindex.languages"]
    lua = "wsindex_lang_lua:LANGUAGES"

The build backend copies those into the installed distribution's
metadata, and `importlib.metadata` reads the metadata of everything on
`sys.path` — so a plugin becomes visible by being *installed*, not by
being listed in our config.

The advertised object may be one `LanguageSpec` or an iterable of them,
so a plugin covering several dialects needs one entry point.

A broken plugin is a warning and a skip, never an exception: someone
else's package failing to import must not cost the user their workspace,
and every failure mode here is one the core cannot fix.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator
from importlib.metadata import entry_points
from typing import Any

from wsindex.ingest.languages import LanguageRegistry, LanguageSpec

ENTRY_POINT_GROUP = "wsindex.languages"
"""The group a plugin declares its languages under."""


class PluginLoadWarning(UserWarning):
    """A plugin was skipped. Its own category so callers can filter it."""


def _warn(message: str) -> None:
    # stacklevel points past this helper and its caller, at whoever
    # triggered the load — for us that is an import, so the exact frame
    # is not useful, but the category is what callers filter on anyway.
    warnings.warn(message, PluginLoadWarning, stacklevel=3)


def _as_specs(name: str, loaded: object) -> Iterator[LanguageSpec]:
    """Interpret what an entry point resolved to as language specs.

    One spec or an iterable of them; anything else is a plugin bug worth
    naming, because the alternative is a plugin that installs cleanly and
    silently indexes nothing.

    Args:
        name: Entry point name, for the warning message.
        loaded: Whatever `EntryPoint.load()` returned.

    Yields:
        The specs found; nothing at all when the object is unusable.
    """
    if isinstance(loaded, LanguageSpec):
        yield loaded
        return
    # `str` and `bytes` are iterable, so without this they would be taken
    # for a container and reported one character at a time — 26 warnings
    # for a single mistyped entry point. They are never a bag of specs.
    if isinstance(loaded, str | bytes) or not isinstance(loaded, Iterable):
        _warn(
            f"language plugin {name!r} skipped: expected a LanguageSpec or an "
            f"iterable of them, got {type(loaded).__name__}"
        )
        return
    items: list[Any]
    try:
        items = list(loaded)
    except Exception as exc:  # a plugin's generator may raise anything
        _warn(f"language plugin {name!r} skipped: iterating its specs failed — {exc}")
        return
    for item in items:
        if not isinstance(item, LanguageSpec):
            _warn(
                f"language plugin {name!r} skipped an entry: expected a "
                f"LanguageSpec, got {type(item).__name__}"
            )
            continue
        yield item


def load_plugins(registry: LanguageRegistry, *, group: str = ENTRY_POINT_GROUP) -> tuple[str, ...]:
    """Register every language advertised under `group`.

    Called once, when the ingest package finishes assembling itself, so a
    plugin is live for anything that imports wsindex — the CLI, a test, a
    library caller — without anyone having to remember this function.

    Failures are per entry point when the load itself fails (there is
    nothing to salvage) and per spec afterwards: a plugin offering three
    languages, one of which collides with an installed one, contributes
    the other two. A plugin is a bag of languages, and one bad language
    does not invalidate its siblings.

    Args:
        registry: Registry to add the languages to.
        group: Entry point group to read; overridden in tests.

    Returns:
        Names of the languages actually registered, in load order.
    """
    loaded: list[str] = []
    for entry_point in entry_points(group=group):
        try:
            obj = entry_point.load()
        except Exception as exc:
            # Deliberately broad: this is third-party code, and every way
            # it can fail to import — a missing dependency, a syntax
            # error, a module-level assert — must cost a warning rather
            # than the whole process.
            _warn(f"language plugin {entry_point.name!r} skipped: import failed — {exc}")
            continue
        for spec in _as_specs(entry_point.name, obj):
            try:
                registry.register(spec)
            except ValueError as exc:
                # The specification's own rules (see `LanguageRegistry.register`):
                # malformed, or claiming a name or suffix someone already has.
                _warn(f"language plugin {entry_point.name!r} skipped {spec.name!r}: {exc}")
                continue
            loaded.append(spec.name)
    return tuple(loaded)
