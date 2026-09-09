"""Connector loader tests: real EntryPoint objects, real imports.

Same shape as `test_plugins.py`, and for the same reason: `load()` is the
mechanism under test, so it is real, and only *discovery* is patched —
installing a distribution per case is the example package's job, not a
unit test's. The fixtures live in `tests/connector_fixture.py`, which
pytest puts on `sys.path`.

Every case registers into a throwaway dict. A test that mutated the
process-wide `BUILTIN` would leak into the rest of the run.
"""

import importlib.util
import subprocess
import sys
from collections.abc import Callable
from importlib.metadata import EntryPoint

import pytest

from connector_fixture import FixtureConnector
from wsindex.connectors import Connector, ConnectorFactory, ConnectorSpec, route
from wsindex.connectors.plugins import (
    ENTRY_POINT_GROUP,
    ConnectorLoadWarning,
    load_connectors,
)

GROUP = ENTRY_POINT_GROUP


def ep(name: str, value: str) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=GROUP)


@pytest.fixture
def advertise(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Pretend the given entry points are installed on this machine."""

    def install(*points: EntryPoint) -> None:
        monkeypatch.setattr(
            "wsindex.connectors.plugins.entry_points",
            lambda group: tuple(p for p in points if p.group == group),
        )

    return install


@pytest.fixture
def registry() -> dict[str, ConnectorFactory]:
    return {}


# --- the happy path ------------------------------------------------------


def test_a_plugin_registers_its_type(
    registry: dict[str, ConnectorFactory], advertise: Callable[..., None]
) -> None:
    advertise(ep("fixture", "connector_fixture:FixtureConnector"))

    assert load_connectors(registry, group=GROUP) == ("fixture",)
    assert "fixture" in registry


def test_the_entry_point_name_is_the_config_type(
    registry: dict[str, ConnectorFactory], advertise: Callable[..., None]
) -> None:
    # The whole integration: a `type =` in wsindex.toml reaches a class
    # in a package the core has never heard of.
    advertise(ep("fixture", "connector_fixture:FixtureConnector"))
    load_connectors(registry, group=GROUP)
    spec = ConnectorSpec(type="fixture", url_pattern="https://fixture.invalid/*")

    connector = route("https://fixture.invalid/page", [spec], registry=registry)

    assert isinstance(connector, Connector)
    assert connector.fetch("https://fixture.invalid/page").title == "Fixture"


def test_several_plugins_load_independently(
    registry: dict[str, ConnectorFactory], advertise: Callable[..., None]
) -> None:
    advertise(
        ep("broken", "connector_fixture_broken:Anything"),
        ep("fixture", "connector_fixture:FixtureConnector"),
    )

    with pytest.warns(ConnectorLoadWarning):
        assert load_connectors(registry, group=GROUP) == ("fixture",)


# --- a broken plugin is a warning, never an exception --------------------


def test_a_plugin_that_cannot_be_imported_is_skipped(
    registry: dict[str, ConnectorFactory], advertise: Callable[..., None]
) -> None:
    advertise(ep("broken", "connector_fixture_broken:Anything"))

    with pytest.warns(ConnectorLoadWarning, match="import failed"):
        assert load_connectors(registry, group=GROUP) == ()
    assert not registry


def test_a_plugin_pointing_at_nothing_is_skipped(
    registry: dict[str, ConnectorFactory], advertise: Callable[..., None]
) -> None:
    advertise(ep("missing", "connector_fixture:NoSuchName"))

    with pytest.warns(ConnectorLoadWarning, match="import failed"):
        assert load_connectors(registry, group=GROUP) == ()


def test_something_that_is_not_a_connector_is_skipped(
    registry: dict[str, ConnectorFactory], advertise: Callable[..., None]
) -> None:
    advertise(ep("wrong", "connector_fixture:NOT_A_CONNECTOR"))

    with pytest.warns(ConnectorLoadWarning, match="expected a Connector subclass"):
        assert load_connectors(registry, group=GROUP) == ()


def test_a_class_that_is_not_a_connector_is_named_in_the_warning(
    registry: dict[str, ConnectorFactory], advertise: Callable[..., None]
) -> None:
    advertise(ep("wrong", "connector_fixture:Unrelated"))

    with pytest.warns(ConnectorLoadWarning, match="got Unrelated"):
        load_connectors(registry, group=GROUP)


def test_an_unfinished_connector_is_caught_at_load_not_at_fetch(
    registry: dict[str, ConnectorFactory], advertise: Callable[..., None]
) -> None:
    # It is a Connector subclass and it installs cleanly; it just cannot
    # be constructed. Without this check the TypeError would surface from
    # inside `route`, with nothing pointing at the plugin.
    advertise(ep("halfdone", "connector_fixture:HalfDoneConnector"))

    with pytest.warns(ConnectorLoadWarning, match="does not implement fetch"):
        assert load_connectors(registry, group=GROUP) == ()


# --- the box wins --------------------------------------------------------


def test_a_plugin_may_not_take_a_builtin_name(
    registry: dict[str, ConnectorFactory], advertise: Callable[..., None]
) -> None:
    # A config saying `type = "github"` was written against the built-in.
    # Silently changing what that word means is worse than not loading.
    registry["github"] = FixtureConnector
    advertise(ep("github", "connector_fixture:OtherConnector"))

    with pytest.warns(ConnectorLoadWarning, match="is a built-in connector"):
        assert load_connectors(registry, group=GROUP) == ()
    assert registry["github"] is FixtureConnector


def test_two_plugins_claiming_one_type_do_not_shadow_each_other(
    registry: dict[str, ConnectorFactory], advertise: Callable[..., None]
) -> None:
    # Same rule as for a built-in, and the same reason: the config says
    # `type = "fixture"` and must keep meaning one thing.
    advertise(
        ep("fixture", "connector_fixture:FixtureConnector"),
        ep("fixture", "connector_fixture:OtherConnector"),
    )

    with pytest.warns(ConnectorLoadWarning, match="another plugin"):
        assert load_connectors(registry, group=GROUP) == ("fixture",)
    assert registry["fixture"] is FixtureConnector


# --- when the loading happens ---------------------------------------------


def test_plugins_load_on_the_first_route_not_on_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Importing wsindex must not import plugins: a plugin imports
    # wsindex, so doing it on the way up re-enters the plugin
    # half-executed and silently disables it.
    import wsindex.connectors as package

    calls: list[int] = []
    monkeypatch.setattr(package, "_plugins_loaded", False)
    monkeypatch.setattr(
        "wsindex.connectors.plugins.load_connectors", lambda registry=None: calls.append(1)
    )

    package.route("https://example.invalid/a", [])
    package.route("https://example.invalid/b", [])

    assert calls == [1]


def test_a_plugin_imported_first_still_registers() -> None:
    # The regression itself, in the only place it can be reproduced: a
    # fresh interpreter, with the import order that used to break it.
    # Skipped rather than asserted when the example is not installed —
    # the claim is about wsindex, not about this machine.
    if importlib.util.find_spec("wsindex_connector_notion") is None:
        pytest.skip("example connector plugin not installed (uv run poe example-plugin)")
    program = (
        "import wsindex_connector_notion\n"
        "from wsindex.connectors import ConnectorSpec, route\n"
        "spec = ConnectorSpec(type='notion', url_pattern='https://www.notion.so/*')\n"
        "url = 'https://www.notion.so/o/Page-1f2e3d4c5b6a7890abcdef1234567890'\n"
        "assert route(url, [spec]) is not None, 'plugin did not register'\n"
    )
    subprocess.run([sys.executable, "-W", "error::UserWarning", "-c", program], check=True)


def test_the_group_is_the_one_the_documentation_names() -> None:
    assert ENTRY_POINT_GROUP == "wsindex.connectors"
