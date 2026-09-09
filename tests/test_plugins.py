"""Loader tests: real EntryPoint objects, real imports, real failures.

The one thing worth not faking here is `EntryPoint.load()` — it is the
mechanism the step is about, and a stub would only prove that our stub
returns what we told it to. So the entry points below are genuine
`importlib.metadata.EntryPoint` values pointing at `tests/plugin_fixture*`,
which pytest puts on `sys.path`. What is patched is only *discovery*:
`entry_points()` reads the metadata of installed distributions, and
installing a package per test case is step 25's job, not a unit test's.

Every case registers into a throwaway `LanguageRegistry`. A test that
mutated the process-wide `REGISTRY` would leak into the rest of the run.
"""

from collections.abc import Callable
from importlib.metadata import EntryPoint

import pytest

from wsindex.ingest.languages import LanguageRegistry, LanguageSpec
from wsindex.ingest.plugins import ENTRY_POINT_GROUP, PluginLoadWarning, load_plugins
from wsindex.model import Kind

GROUP = ENTRY_POINT_GROUP


def ep(name: str, value: str) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=GROUP)


@pytest.fixture
def advertise(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Pretend the given entry points are installed on this machine."""

    def install(*points: EntryPoint) -> None:
        monkeypatch.setattr(
            "wsindex.ingest.plugins.entry_points",
            lambda group: tuple(p for p in points if p.group == group),
        )

    return install


@pytest.fixture
def registry() -> LanguageRegistry:
    return LanguageRegistry()


# --- the happy path ------------------------------------------------------


def test_a_plugin_registers_its_language(
    registry: LanguageRegistry, advertise: Callable[..., None]
) -> None:
    advertise(ep("go", "plugin_fixture:GO"))
    assert load_plugins(registry, group=GROUP) == ("fixture-go",)
    assert registry.get("fixture-go") is not None


def test_one_entry_point_may_carry_several_languages(
    registry: LanguageRegistry, advertise: Callable[..., None]
) -> None:
    # A plugin covering a language and its template dialect declares one
    # entry point, not two.
    advertise(ep("go", "plugin_fixture:PAIR"))
    assert load_plugins(registry, group=GROUP) == ("fixture-go", "fixture-tmpl")


def test_a_loaded_language_reaches_the_walker_and_the_chunker(
    registry: LanguageRegistry, advertise: Callable[..., None]
) -> None:
    # The point of the whole step: an installed package the core has
    # never heard of changes which files get indexed.
    advertise(ep("go", "plugin_fixture:GO"))
    load_plugins(registry, group=GROUP)
    from pathlib import Path

    matched = registry.match(Path("server.fgo"))
    assert matched is not None
    assert (matched.name, matched.kind) == ("fixture-go", Kind.CODE)


def test_nothing_installed_is_not_an_error(
    registry: LanguageRegistry, advertise: Callable[..., None]
) -> None:
    advertise()
    assert load_plugins(registry, group=GROUP) == ()


def test_only_the_requested_group_is_read(
    registry: LanguageRegistry, advertise: Callable[..., None]
) -> None:
    advertise(EntryPoint(name="go", value="plugin_fixture:GO", group="somebody.else"))
    assert load_plugins(registry, group=GROUP) == ()


# --- a broken plugin costs a warning, never the process ------------------


def test_a_plugin_that_fails_to_import_is_skipped(
    registry: LanguageRegistry, advertise: Callable[..., None]
) -> None:
    # The most common failure by far: a missing transitive dependency.
    advertise(ep("broken", "plugin_fixture_broken:ANY"))
    with pytest.warns(PluginLoadWarning, match="import failed"):
        assert load_plugins(registry, group=GROUP) == ()


def test_a_missing_attribute_is_skipped(
    registry: LanguageRegistry, advertise: Callable[..., None]
) -> None:
    # A plugin built against a wsindex that named things differently.
    advertise(ep("typo", "plugin_fixture:NO_SUCH_NAME"))
    with pytest.warns(PluginLoadWarning, match="import failed"):
        assert load_plugins(registry, group=GROUP) == ()


def test_an_object_that_is_not_a_spec_is_skipped(
    registry: LanguageRegistry, advertise: Callable[..., None]
) -> None:
    advertise(ep("wrong", "plugin_fixture:NOT_A_SPEC"))
    with pytest.warns(PluginLoadWarning, match="expected a LanguageSpec"):
        assert load_plugins(registry, group=GROUP) == ()


def test_a_malformed_spec_is_skipped(
    registry: LanguageRegistry, advertise: Callable[..., None]
) -> None:
    # The specification's own rules do the rejecting (step 23); the
    # loader only decides that it costs a warning rather than a crash.
    advertise(ep("bad", "plugin_fixture:BROKEN"))
    with pytest.warns(PluginLoadWarning, match="must be lowercase and start with"):
        assert load_plugins(registry, group=GROUP) == ()


def test_a_conflicting_language_is_skipped(
    registry: LanguageRegistry, advertise: Callable[..., None]
) -> None:
    registry.register(LanguageSpec(name="mine", kind=Kind.CODE, suffixes=(".fgo",)))
    advertise(ep("go", "plugin_fixture:GO"))
    with pytest.warns(PluginLoadWarning, match="already claimed by 'mine'"):
        assert load_plugins(registry, group=GROUP) == ()
    # The installed language kept its suffix.
    assert registry.get("fixture-go") is None


def test_a_bad_spec_does_not_sink_its_siblings(
    registry: LanguageRegistry, advertise: Callable[..., None]
) -> None:
    # A plugin is a bag of languages: one unusable entry must not cost
    # the others. MIXED is (a valid spec, the integer 42).
    advertise(ep("mixed", "plugin_fixture:MIXED"))
    with pytest.warns(PluginLoadWarning, match="got int"):
        assert load_plugins(registry, group=GROUP) == ("fixture-go",)


def test_one_broken_plugin_does_not_stop_the_next(
    registry: LanguageRegistry, advertise: Callable[..., None]
) -> None:
    advertise(ep("broken", "plugin_fixture_broken:ANY"), ep("go", "plugin_fixture:GO"))
    with pytest.warns(PluginLoadWarning):
        assert load_plugins(registry, group=GROUP) == ("fixture-go",)


def test_the_warning_names_the_plugin(
    registry: LanguageRegistry, advertise: Callable[..., None]
) -> None:
    # Whoever sees this has to know which package to blame or uninstall.
    advertise(ep("suspicious-package", "plugin_fixture_broken:ANY"))
    with pytest.warns(PluginLoadWarning, match="suspicious-package"):
        load_plugins(registry, group=GROUP)


# --- what the core itself declares ---------------------------------------


def test_the_group_name_is_the_documented_one() -> None:
    # Plugins hardcode this string in their pyproject; renaming it breaks
    # every plugin already published.
    assert ENTRY_POINT_GROUP == "wsindex.languages"


def test_wsindex_advertises_no_language_plugins_of_its_own() -> None:
    # Built-ins are registered directly, not through entry points: the
    # core must not pay discovery to find what it already ships.
    #
    # Asks wsindex's own metadata rather than the whole environment —
    # otherwise installing any plugin (which is exactly what step 25
    # does) would fail this.
    from importlib.metadata import distribution

    declared = [point for point in distribution("wsindex").entry_points if point.group == GROUP]
    assert declared == []


def test_a_bare_string_warns_once_not_per_character(
    registry: LanguageRegistry, advertise: Callable[..., None]
) -> None:
    # Regression: `str` is iterable, so a mistyped entry point pointing at
    # a string used to be walked character by character — 26 warnings for
    # one mistake, none of which said anything useful.
    advertise(ep("wrong", "plugin_fixture:NOT_A_SPEC"))
    with pytest.warns(PluginLoadWarning) as caught:
        assert load_plugins(registry, group=GROUP) == ()
    assert len(caught) == 1
    assert "got str" in str(caught[0].message)


def test_a_spec_generator_that_raises_is_skipped(
    registry: LanguageRegistry, advertise: Callable[..., None]
) -> None:
    # A plugin that computes its languages can fail halfway through.
    advertise(ep("angry", "plugin_fixture:ANGRY"))
    with pytest.warns(PluginLoadWarning, match="iterating its specs failed"):
        assert load_plugins(registry, group=GROUP) == ()
