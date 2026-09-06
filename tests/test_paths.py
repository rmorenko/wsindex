"""Path resolver tests: every mode in isolation + priority order between them.

The resolver is a pure module — no CWD games in production code — so tests
inject `cwd` explicitly and patch `$XDG_*` via monkeypatch. `HOME` is also
patched: platformdirs falls back to `$HOME/.config` when `XDG_CONFIG_HOME`
is unset, and an unpatched HOME would leak the developer's real config.
"""

from pathlib import Path

import pytest

from wsindex.paths import (
    ConfigLocation,
    Mode,
    find_config,
    resolve_cache_dir,
    resolve_index_dir,
    searched_paths,
    user_config_file,
    workspace_config_path,
    workspace_id,
)


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with a clean env: no override, no leaked XDG_*.

    HOME is also redirected into tmp_path so platformdirs' fallback
    (`$HOME/.config`) cannot accidentally point at the developer's real
    dotfiles and turn tests hermetic-in-name-only.
    """
    monkeypatch.delenv("WSINDEX_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_DIRS", str(tmp_path / "etc-xdg"))


def _write(path: Path, content: str = "x = 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


# --- workspace mode -------------------------------------------------------


def test_workspace_mode_finds_config_in_cwd(tmp_path: Path) -> None:
    _write(tmp_path / "wsindex.toml")
    location = find_config(cwd=tmp_path)
    assert location is not None
    assert location.mode == Mode.WORKSPACE
    assert location.path == tmp_path / "wsindex.toml"


def test_workspace_mode_walks_up_from_subdirectory(tmp_path: Path) -> None:
    # Roman's user story: `wsindex search` should work from a subpackage
    # of the project, not only from the workspace root. Same behaviour as
    # `git status` — the tool finds its marker in an ancestor directory.
    _write(tmp_path / "wsindex.toml")
    deep = tmp_path / "src" / "wsindex" / "ingest"
    deep.mkdir(parents=True)
    location = find_config(cwd=deep)
    assert location is not None
    assert location.mode == Mode.WORKSPACE
    assert location.path == tmp_path / "wsindex.toml"


# --- env override --------------------------------------------------------


def test_env_override_wins_over_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = _write(tmp_path / "explicit.toml")
    _write(tmp_path / "wsindex.toml")  # would win in workspace mode
    monkeypatch.setenv("WSINDEX_CONFIG", str(override))
    location = find_config(cwd=tmp_path)
    assert location is not None
    assert location.mode == Mode.OVERRIDE
    assert location.path == override


def test_env_override_missing_returns_none_not_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If the user set WSINDEX_CONFIG to a specific file, silently
    # falling back to a workspace or XDG config would hide the typo.
    _write(tmp_path / "wsindex.toml")  # would satisfy workspace mode
    monkeypatch.setenv("WSINDEX_CONFIG", str(tmp_path / "does-not-exist.toml"))
    assert find_config(cwd=tmp_path) is None


# --- user (XDG_CONFIG_HOME) mode -----------------------------------------


def test_user_mode_via_xdg_config_home(tmp_path: Path) -> None:
    user_config = _write(tmp_path / "config" / "wsindex" / "config.toml")
    empty_cwd = tmp_path / "empty"
    empty_cwd.mkdir()
    location = find_config(cwd=empty_cwd)
    assert location is not None
    assert location.mode == Mode.USER
    assert location.path == user_config


# --- system (XDG_CONFIG_DIRS) mode --------------------------------------


def test_system_mode_via_xdg_config_dirs(tmp_path: Path) -> None:
    system_config = _write(tmp_path / "etc-xdg" / "wsindex" / "config.toml")
    empty_cwd = tmp_path / "empty"
    empty_cwd.mkdir()
    location = find_config(cwd=empty_cwd)
    assert location is not None
    assert location.mode == Mode.SYSTEM
    assert location.path == system_config


# --- priority order between modes ---------------------------------------


def test_workspace_beats_user_and_system(tmp_path: Path) -> None:
    _write(tmp_path / "config" / "wsindex" / "config.toml")
    _write(tmp_path / "etc-xdg" / "wsindex" / "config.toml")
    workspace = _write(tmp_path / "wsindex.toml")
    location = find_config(cwd=tmp_path)
    assert location is not None
    assert location.mode == Mode.WORKSPACE
    assert location.path == workspace


def test_user_beats_system(tmp_path: Path) -> None:
    user_config = _write(tmp_path / "config" / "wsindex" / "config.toml")
    _write(tmp_path / "etc-xdg" / "wsindex" / "config.toml")
    empty_cwd = tmp_path / "empty"
    empty_cwd.mkdir()
    location = find_config(cwd=empty_cwd)
    assert location is not None
    assert location.mode == Mode.USER
    assert location.path == user_config


# --- nothing found ------------------------------------------------------


def test_no_config_anywhere(tmp_path: Path) -> None:
    empty_cwd = tmp_path / "empty"
    empty_cwd.mkdir()
    assert find_config(cwd=empty_cwd) is None


# --- resolve_index_dir --------------------------------------------------


def test_resolve_index_dir_workspace_mode_sits_next_to_config(tmp_path: Path) -> None:
    config = tmp_path / "wsindex.toml"
    location = ConfigLocation(path=config, mode=Mode.WORKSPACE)
    assert resolve_index_dir(location, "myws") == tmp_path / ".wsindex"


def test_resolve_index_dir_override_mode_sits_next_to_config(tmp_path: Path) -> None:
    # Override behaves like workspace for index placement: the user
    # picked an explicit config, so the index goes with it.
    config = tmp_path / "elsewhere" / "custom.toml"
    location = ConfigLocation(path=config, mode=Mode.OVERRIDE)
    assert resolve_index_dir(location, "myws") == tmp_path / "elsewhere" / ".wsindex"


def test_resolve_index_dir_user_mode_uses_xdg_data_with_workspace_id(tmp_path: Path) -> None:
    config = tmp_path / "config" / "wsindex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text("x = 1\n")
    location = ConfigLocation(path=config, mode=Mode.USER)
    resolved = resolve_index_dir(location, "myws")
    assert resolved.parent == tmp_path / "data" / "wsindex"
    assert resolved.name.startswith("myws-")
    assert resolved.name == workspace_id(config, "myws")


def test_resolve_index_dir_system_mode_still_writes_to_user_data(tmp_path: Path) -> None:
    # System configs are shared/read-only; every user needs their own
    # writable index under $XDG_DATA_HOME regardless of where the
    # config came from.
    config = tmp_path / "etc-xdg" / "wsindex" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text("x = 1\n")
    location = ConfigLocation(path=config, mode=Mode.SYSTEM)
    resolved = resolve_index_dir(location, "myws")
    assert resolved.parent == tmp_path / "data" / "wsindex"
    assert resolved.name.startswith("myws-")


def test_workspace_id_deterministic_for_same_path(tmp_path: Path) -> None:
    # Same config in the same place → same id, always. The whole point
    # of a stable id is that consecutive runs write to the same dir.
    config = tmp_path / "wsindex.toml"
    config.write_text("x = 1\n")
    assert workspace_id(config, "demo") == workspace_id(config, "demo")


def test_workspace_id_differs_by_config_path(tmp_path: Path) -> None:
    # The real collision case: two user-mode workspaces with the same
    # `name` but different config paths (e.g. isolated via distinct
    # $XDG_CONFIG_HOME values) must get distinct index dirs.
    config_a = tmp_path / "a" / "wsindex.toml"
    config_a.parent.mkdir()
    config_a.write_text("x = 1\n")
    config_b = tmp_path / "b" / "wsindex.toml"
    config_b.parent.mkdir()
    config_b.write_text("x = 1\n")
    assert workspace_id(config_a, "demo") != workspace_id(config_b, "demo")


def test_workspace_id_differs_by_name(tmp_path: Path) -> None:
    # Same config, different `name` → different id (the human prefix
    # changes even before the hash tail is inspected).
    config = tmp_path / "wsindex.toml"
    config.write_text("x = 1\n")
    assert workspace_id(config, "alpha") != workspace_id(config, "beta")


def test_workspace_id_is_filesystem_safe(tmp_path: Path) -> None:
    # blake2s + hex → nothing that could confuse a shell or a filesystem.
    config = tmp_path / "wsindex.toml"
    config.write_text("x = 1\n")
    identifier = workspace_id(config, "demo")
    assert all(c.isalnum() or c == "-" for c in identifier)
    assert identifier.startswith("demo-")


def test_workspace_id_resolves_symlinks(tmp_path: Path) -> None:
    # If two "different" paths resolve to the same real file (symlink),
    # they must share one id — otherwise a symlinked config would
    # silently maintain a second, ghost index.
    real = tmp_path / "real.toml"
    real.write_text("x = 1\n")
    link = tmp_path / "link.toml"
    link.symlink_to(real)
    assert workspace_id(real, "demo") == workspace_id(link, "demo")


# --- helpers ------------------------------------------------------------


def test_workspace_config_path_is_wsindex_toml_in_cwd(tmp_path: Path) -> None:
    assert workspace_config_path(cwd=tmp_path) == tmp_path / "wsindex.toml"


def test_user_config_file_lives_under_xdg_config_home(tmp_path: Path) -> None:
    assert user_config_file() == tmp_path / "config" / "wsindex" / "config.toml"


def test_resolve_cache_dir_is_under_xdg_cache_home(tmp_path: Path) -> None:
    # Per-user, not per-workspace: model files are shared across workspaces
    # that use the same model id — the whole point of a cache. `rm -rf` on
    # the returned path must be a no-op-in-effect for correctness.
    assert resolve_cache_dir() == tmp_path / "cache" / "wsindex"


def test_searched_paths_mentions_every_mode(tmp_path: Path) -> None:
    paths = searched_paths(cwd=tmp_path)
    text = "\n".join(paths)
    assert "$WSINDEX_CONFIG" in text
    assert "workspace" in text
    assert str(tmp_path) in text
    assert "user" in text
    assert "system" in text


def test_searched_paths_shows_override_value_when_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WSINDEX_CONFIG", "/some/explicit/path.toml")
    paths = searched_paths(cwd=tmp_path)
    assert any("/some/explicit/path.toml" in line for line in paths)
