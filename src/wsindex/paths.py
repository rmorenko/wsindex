"""Where the workspace config and its index live on disk.

Path resolution has four modes, checked in strict priority order:

1. `WSINDEX_CONFIG` env — explicit override, wins over everything.
2. Workspace mode — walk from CWD up to `/` for a `wsindex.toml`,
   like `git` finds `.git/`. Index lives in `.wsindex/` next to the
   config: the natural place for a per-project tool.
3. User mode — `$XDG_CONFIG_HOME/wsindex/config.toml` via platformdirs.
   Index lives in `$XDG_DATA_HOME/wsindex/<workspace-name>/`: for
   `pipx install` and other from-anywhere invocations.
4. System mode — `$XDG_CONFIG_DIRS/wsindex/config.toml`. Read-only
   admin defaults; index still goes to user data (system prefix may
   be read-only, and multi-user systems must not share one index).

The resolver is a pure module: no CLI concerns, no exit codes. It
returns a `ConfigLocation | None`; the caller decides how to complain.
"""

import hashlib
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from platformdirs import site_config_path, user_cache_path, user_config_path, user_data_path

APP_NAME = "wsindex"
CONFIG_FILE = "wsindex.toml"
USER_CONFIG_FILE = "config.toml"
WORKSPACE_INDEX_DIR = ".wsindex"
ENV_OVERRIDE = "WSINDEX_CONFIG"
WORKSPACE_ID_HASH_LEN = 8  # bytes of blake2s in hex → 16 hex chars


class Mode(StrEnum):
    """How a config was located; drives where the index dir lives."""

    OVERRIDE = "override"
    WORKSPACE = "workspace"
    USER = "user"
    SYSTEM = "system"


@dataclass(frozen=True, kw_only=True)
class ConfigLocation:
    """Where the resolver found the config, and by which mode.

    Attributes:
        path: The resolved config file; guaranteed to exist at the
            moment `find_config` returned it (races are the caller's
            problem — the CLI reads it immediately).
        mode: Which resolution mode won; `resolve_index_dir` uses it.
    """

    path: Path
    mode: Mode


def find_config(cwd: Path | None = None) -> ConfigLocation | None:
    """Locate a wsindex config, or None if there is none anywhere.

    An explicit `$WSINDEX_CONFIG` that points at a non-existent file
    returns None (not an ancestor-walk fallback): the user meant that
    file — silently ignoring the override would hide the mistake.

    Args:
        cwd: Starting directory for the workspace-mode walk; defaults
            to `Path.cwd()`. Injected for tests, not for production.

    Returns:
        The winning `ConfigLocation`, or None if no config exists.
    """
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        candidate = Path(override)
        return ConfigLocation(path=candidate, mode=Mode.OVERRIDE) if candidate.is_file() else None

    start = cwd or Path.cwd()
    for parent in [start, *start.parents]:
        candidate = parent / CONFIG_FILE
        if candidate.is_file():
            return ConfigLocation(path=candidate, mode=Mode.WORKSPACE)

    user = user_config_path(APP_NAME) / USER_CONFIG_FILE
    if user.is_file():
        return ConfigLocation(path=user, mode=Mode.USER)

    system = site_config_path(APP_NAME) / USER_CONFIG_FILE
    if system.is_file():
        return ConfigLocation(path=system, mode=Mode.SYSTEM)

    return None


def searched_paths(cwd: Path | None = None) -> list[str]:
    """Human-readable list of every path `find_config` would check.

    Used by the CLI to compose a helpful "no config found" message
    that tells the user exactly where wsindex looked.
    """
    override = os.environ.get(ENV_OVERRIDE)
    start = cwd or Path.cwd()
    return [
        f"$WSINDEX_CONFIG ({override or 'not set'})",
        f"workspace: {start} and its parents up to /",
        f"user: {user_config_path(APP_NAME) / USER_CONFIG_FILE}",
        f"system: {site_config_path(APP_NAME) / USER_CONFIG_FILE}",
    ]


def workspace_id(config_path: Path, workspace_name: str) -> str:
    """Stable, unique id for a workspace: `<name>-<hash-of-config-path>`.

    Used as the subdirectory name under `$XDG_DATA_HOME/wsindex/` in
    user and system mode. Two configs with the same `name` but different
    file paths (typical when a user parameterizes `$XDG_CONFIG_HOME` to
    isolate environments) must get separate index dirs — collision here
    would silently merge two workspaces into one shared index. The name
    stays as a human-readable prefix so `ls ~/.local/share/wsindex/`
    remains browsable; the hash tail guarantees uniqueness.

    The hash is of the **resolved** absolute config path, so a symlink
    to the config resolves to the same id as the target. Renaming or
    moving the config produces a new id — the old index dir is left as
    an orphan on disk; nothing points at it, `rm -rf` is safe.

    Args:
        config_path: The `ConfigLocation.path` (whatever `find_config`
            returned).
        workspace_name: `Config.name` — the human prefix.

    Returns:
        Filesystem-safe workspace id, e.g. `"demo-a3f9c2d1e5b7a4f8"`.
    """
    canonical = str(config_path.resolve())
    digest = hashlib.blake2s(canonical.encode(), digest_size=WORKSPACE_ID_HASH_LEN).hexdigest()
    return f"{workspace_name}-{digest}"


def resolve_index_dir(location: ConfigLocation, workspace_name: str) -> Path:
    """Where the index database lives for a given config location.

    Workspace/override configs put the index in `.wsindex/` next to
    the config — the per-project convention, isolated by directory.
    User/system configs push the index to `$XDG_DATA_HOME/wsindex/
    <workspace-id>/`, where the id is `<name>-<hash>` (see
    `workspace_id`): system prefixes are often read-only, and multiple
    user-mode configs (different `$XDG_CONFIG_HOME`s pointing at
    workspaces with the same name) must not collide.

    Args:
        location: Config location that `find_config` produced.
        workspace_name: `Config.name`; used to compose the workspace
            id in user/system mode.

    Returns:
        Absolute path to the index directory (may not exist yet;
        the store creates it on first write).
    """
    if location.mode in (Mode.WORKSPACE, Mode.OVERRIDE):
        return location.path.parent / WORKSPACE_INDEX_DIR
    return user_data_path(APP_NAME) / workspace_id(location.path, workspace_name)


def workspace_config_path(cwd: Path | None = None) -> Path:
    """Target of `wsindex init` (workspace mode): `./wsindex.toml`."""
    return (cwd or Path.cwd()) / CONFIG_FILE


def user_config_file() -> Path:
    """Target of `wsindex init --user`: `$XDG_CONFIG_HOME/wsindex/config.toml`."""
    return user_config_path(APP_NAME) / USER_CONFIG_FILE


def resolve_cache_dir() -> Path:
    """Root of wsindex-owned cache: `$XDG_CACHE_HOME/wsindex/`.

    Per-user, not per-workspace: cached models (~90 MB each) are the
    same across every workspace that uses the same model id — sharing
    them across workspaces is a feature, not a bug. Callers pick a
    subdirectory (e.g. `resolve_cache_dir() / "models"`); this module
    stays out of the "what kind of cache" business.

    XDG contract: everything under this directory can be deleted at
    any time without breaking wsindex — the next run will re-download
    what it needs.
    """
    return user_cache_path(APP_NAME)
