"""Workspace configuration: the `wsindex.toml` file (defaults, load, save).

The config is the single source of truth for a workspace: which vector-store
backend to use, which embedding model, and which repositories to index. The
TOML layout (sections `workspace`, `embeddings`, `tensorus`, `repos`) lives
only in `to_dict`/`from_dict`, so the file format has one definition per
direction.

Module-level singleton: exactly one Config is "current" per process, stored
in the private module variable `_current`. `load_config(path)` publishes as
a side effect; `set_config(cfg)` publishes explicitly (for configs built in
memory, e.g. `wsindex init`); `get_config()` returns the current one or
raises. Tests reset the slot between cases via the autouse fixture in
`tests/conftest.py` — otherwise module state would leak across tests since
pytest never reimports modules.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import tomli_w


class Backend(StrEnum):
    """Vector store selector; StrEnum so the value round-trips through TOML as-is."""

    TENSORUS = "tensorus"
    LOCAL = "local"


class Provider(StrEnum):
    """Embedder selector: deterministic fake for tests, real model for work."""

    FAKE = "fake"
    SENTENCE_TRANSFORMERS = "sentence-transformers"


@dataclass(frozen=True, kw_only=True)
class Repository:
    """One indexed repository.

    Attributes:
        id: Stable unique name; doubles as the dataset name in the store.
        path: Repository root directory, absolute or workspace-relative.
    """

    id: str
    path: str


@dataclass(kw_only=True)
class Config:
    """In-memory form of `wsindex.toml`. Mutable: `add_repo` edits it in place.

    Attributes:
        name: Workspace name; identification only, nothing derives from it.
        backend: Which VectorStore the composition root builds.
        provider: Which Embedder the composition root builds.
        model: Embedding model name (both local and server-side).
        dim: Vector dimensionality the model produces.
        base_url: Tensorus server root; unused by the local backend.
        metric: Similarity metric datasets are created with.
        repos: Repositories to index, in search merge-order.
    """

    name: str
    backend: Backend
    provider: Provider
    model: str
    dim: int
    base_url: str
    metric: str
    repos: list[Repository]

    @classmethod
    def default_config(cls, name: str) -> Config:
        """Config for a fresh workspace: Tensorus backend, MiniLM model, no repos.

        Args:
            name: Workspace name to bake into the config.

        Returns:
            A new Config with project defaults; the caller saves it to disk.
        """
        return cls(
            name=name,
            backend=Backend.TENSORUS,
            provider=Provider.SENTENCE_TRANSFORMERS,
            model="sentence-transformers/all-MiniLM-L6-v2",
            dim=384,
            base_url="http://localhost:8000",
            metric="cosine",
            repos=[],
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for saving.

        Returns:
            Nested dict in the `wsindex.toml` section layout (`workspace`,
            `embeddings`, `tensorus`, `repos`) — input for tomli_w.
        """
        return {
            "workspace": {"name": self.name, "backend": self.backend},
            "embeddings": {"model": self.model, "dim": self.dim, "provider": self.provider},
            "tensorus": {"base_url": self.base_url, "metric": self.metric},
            "repos": [{"id": r.id, "path": r.path} for r in self.repos],
        }

    def add_repo(self, repo_id: str, *, path: str) -> None:
        """Register a repository in the config (in memory; saving is separate).

        Args:
            repo_id: Unique repo id; becomes the dataset name.
            path: Repository root directory.

        Raises:
            ValueError: The id is already registered — ids name datasets,
                so a duplicate would silently merge two repos into one.
        """
        if any(r.id == repo_id for r in self.repos):
            raise ValueError(f"repo id already exists: {repo_id}")
        self.repos.append(Repository(id=repo_id, path=path))

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Config:
        """Inverse of `to_dict`.

        Args:
            config_dict: Parsed TOML in the `wsindex.toml` section layout.

        Returns:
            The equivalent Config; `repos` may be absent and defaults to [].

        Raises:
            KeyError: A required section or key is missing — strict on
                purpose, a half-read config must not survive silently.
        """
        ws = config_dict["workspace"]
        emb = config_dict["embeddings"]
        ts = config_dict["tensorus"]
        return cls(
            name=ws["name"],
            backend=Backend(ws["backend"]),
            provider=Provider(emb["provider"]),
            model=emb["model"],
            dim=emb["dim"],
            base_url=ts["base_url"],
            metric=ts["metric"],
            repos=[Repository(**r) for r in config_dict.get("repos", [])],
        )


_current: Config | None = None


def get_config() -> Config:
    """Return the process-wide current Config.

    Raises:
        RuntimeError: No config has been loaded or published yet. The
            composition root must call `load_config` (file-based) or
            `set_config` (in-memory) before any subsystem asks.
    """
    if _current is None:
        raise RuntimeError("config not loaded; call load_config() or set_config() first")
    return _current


def set_config(config: Config) -> None:
    """Publish `config` as the process-wide current Config.

    Use this when the object was built without touching disk (typically
    `wsindex init` calls `Config.default_config(...)` then wants it
    available via `get_config`). File-based flows should use
    `load_config` — it publishes as a side effect.
    """
    global _current
    _current = config


def reset_config() -> None:
    """Clear the module singleton. Intended for test isolation only.

    pytest never reimports modules between tests, so `_current` would
    otherwise leak across cases. The autouse fixture in
    `tests/conftest.py` calls this before and after every test.
    """
    global _current
    _current = None


def save_config(config: Config, *, path: Path) -> None:
    """Serialize the config to disk as UTF-8 TOML.

    Args:
        config: Config to write.
        path: Target file, overwritten if present.
    """
    path.write_text(tomli_w.dumps(config.to_dict()), encoding="utf-8")


def load_config(path: Path) -> Config:
    """Parse a `wsindex.toml` file into a Config and publish it as current.

    Side effect: installs the returned object as the module singleton, so
    `get_config()` returns this object afterwards. Combined so the
    composition root has one obvious moment where "the config becomes
    current" — splitting load and publish would invite forgetting the
    second call and getting `RuntimeError` from a subsystem later.

    Args:
        path: File to read.

    Returns:
        The parsed Config (same object `get_config()` now returns).

    Raises:
        FileNotFoundError: The file does not exist.
        KeyError: A required section or key is missing (see `from_dict`).
    """
    config_dict = tomllib.loads(path.read_text())
    config = Config.from_dict(config_dict)
    set_config(config)
    return config
