"""Workspace configuration: the `wsindex.toml` file (defaults, load, save).

The config is the single source of truth for a workspace: which vector-store
backend to use, which embedding model, and which repositories to index. The
TOML layout (sections `workspace`, `embeddings`, `tensorus`, `repos`) is also
the in-memory form — `Config` keeps the parsed document as a dict and exposes
one read-only property per field, so the file format has a single definition
and every reader still goes through a typed accessor.

`Config` is a singleton class: `__new__` caches the one instance on the class,
so `Config()` anywhere in the process returns the same object and no subsystem
needs the config threaded through its constructor. The first construction
decides where the document comes from — `Config(path)` reads that file,
`Config()` discovers one (see `wsindex.paths.find_config`). Nothing on disk is
not an error: it means the built-in `Config.DEFAULT` and `is_default` set.
Saying so out loud belongs to whoever has a user to say it to — the CLI warns
in `wsindex.cli._config`; this module never writes to a stream. No file is
ever written as a side effect either; only `wsindex init` creates one.

Tests drop the cached instance between cases via `Config.reset()` — pytest
never reimports modules, so an instance cached on the class outlives a test
exactly like module-level state would.

Deliberately not a dataclass. A dataclass is a value object: `__eq__` by
fields, one instance per distinct value. A singleton is an identity object:
one instance, period. Generating value semantics for a type that has exactly
one instance is a contradiction, and `__init__` has to be hand-written anyway
because it must be idempotent — Python calls it on every `Config(...)`,
including the calls `__new__` answered from the cache.
"""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

import tomli_w

from wsindex.paths import ConfigLocation, Mode, find_config, resolve_index_dir


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

    A value object, unlike `Config`: repositories are compared and passed
    around by value, and there are many of them.

    Attributes:
        id: Stable unique name; doubles as the dataset name in the store.
        path: Repository root directory, absolute or workspace-relative.
    """

    id: str
    path: str


class Config:
    """The workspace config, as one process-wide instance.

    Holds the parsed `wsindex.toml` document in `_data` and reads every
    field through a property, so the TOML layout is spelled out once per
    field instead of leaking into callers. Mutable in exactly one way:
    `add_repo` appends to the document.

    Construct it with `Config()` — the first call loads, the rest hand back
    the same object. `Config.default(...)` replaces the instance with an
    in-memory default (that is what `wsindex init` needs), `Config.reset()`
    drops it (that is what tests need).
    """

    DEFAULT: ClassVar[dict[str, Any]] = {
        "workspace": {"name": "default", "backend": Backend.LOCAL.value},
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dim": 384,
            "provider": Provider.SENTENCE_TRANSFORMERS.value,
        },
        "tensorus": {"base_url": "http://localhost:8000", "metric": "cosine"},
        "repos": [],
    }
    """Project defaults, in the on-disk layout. Never handed out directly:
    it is a mutable dict shared by the whole process, and `add_repo` would
    otherwise append to the constant itself — every user takes a deepcopy."""

    REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "workspace": ("name", "backend"),
        "embeddings": ("model", "dim", "provider"),
        "tensorus": ("base_url", "metric"),
    }
    """Sections and keys a config file must contain; `repos` is optional."""

    _instance: ClassVar[Config | None] = None

    # Bare annotations, not assignments: `__init__` uses `hasattr(self,
    # "_data")` to tell "already loaded" from "fresh instance", which only
    # works while the class itself carries no such attribute.
    _data: dict[str, Any]
    _location: ConfigLocation | None

    def __new__(cls, path: Path | None = None) -> Config:
        """Return the one instance of `cls`, creating it on the first call.

        Args:
            path: Ignored here — Python passes the constructor arguments to
                `__new__` and `__init__` alike, and `__init__` is the one
                that decides what to read.

        Returns:
            The cached instance; every `Config(...)` in the process gets
            the same object back.
        """
        # `cls.__dict__`, not `cls._instance`: a class attribute is visible
        # through subclasses, so `Sub()` would otherwise be handed the
        # parent's instance instead of building its own.
        instance: Config | None = cls.__dict__.get("_instance")
        if instance is None:
            # No `*args` here. `object.__new__` accepts the class and
            # nothing else as soon as `__new__` is overridden — forwarding
            # the constructor arguments raises TypeError.
            instance = super().__new__(cls)
            cls._instance = instance
        return instance

    def __init__(self, path: Path | None = None) -> None:
        """Load the document on the first construction; do nothing after.

        Python calls `__init__` on every `Config(...)`, including the calls
        `__new__` served from the cache. Without the guard below, a bare
        `Config()` somewhere downstream would re-run discovery over an
        already-loaded config and silently swap it out.

        Args:
            path: Read this file. When None, discover one. Either way a
                missing file is not an error: it falls back to `DEFAULT`
                and sets `is_default`, and writes nothing to disk.

        Raises:
            KeyError: The file exists but a required section or key is
                missing (see `_validate`).
            ValueError: The file exists but `backend`, `provider` or a
                `repos` entry holds something unusable.
        """
        if hasattr(self, "_data"):
            return
        if path is not None:
            # Same mode an explicit `$WSINDEX_CONFIG` gets: the caller named
            # the file, so the index belongs next to it.
            location = ConfigLocation(path=path, mode=Mode.OVERRIDE) if path.is_file() else None
        else:
            location = find_config()
        if location is None:
            self._data = copy.deepcopy(self.DEFAULT)
        else:
            data: dict[str, Any] = tomllib.loads(location.path.read_text(encoding="utf-8"))
            self._validate(data)
            data.setdefault("repos", [])
            self._data = data
        self._location = location

    @classmethod
    def _validate(cls, data: dict[str, Any]) -> None:
        """Reject a half-read document before it becomes the live config.

        Strict on purpose: a silent default would mask a typo in the file,
        and the whole workspace would quietly index with the wrong model.

        Args:
            data: Freshly parsed TOML.

        Raises:
            KeyError: A required section or key is missing.
            ValueError: `backend` or `provider` names no enum member, or a
                `repos` entry lacks `id`/`path`.
        """
        for section, keys in cls.REQUIRED.items():
            if section not in data:
                raise KeyError(section)
            for key in keys:
                if key not in data[section]:
                    raise KeyError(f"{section}.{key}")
        # Constructing the enums is the check; the properties do it again on
        # access, but a bad value must fail at load, not at first read.
        Backend(data["workspace"]["backend"])
        Provider(data["embeddings"]["provider"])
        for repo in data.get("repos", []):
            if "id" not in repo or "path" not in repo:
                raise ValueError(f"repo entry needs both 'id' and 'path': {repo!r}")

    @classmethod
    def default(
        cls,
        name: str,
        *,
        backend: Backend | None = None,
        provider: Provider | None = None,
    ) -> Config:
        """Replace the instance with an in-memory config built from `DEFAULT`.

        What `wsindex init` needs: project defaults for a workspace that has
        no file yet, regardless of what happens to be on disk around the
        current directory.

        Args:
            name: Workspace name to bake in.
            backend: Overrides the default backend when given.
            provider: Overrides the default provider when given.

        Returns:
            The new current Config — what `Config()` returns from here on.
        """
        cls.reset()
        # `cls.__new__(cls)` publishes the instance without running
        # `__init__`: there is no file to read, the document comes from the
        # constant below.
        config = cls.__new__(cls)
        data = copy.deepcopy(cls.DEFAULT)
        data["workspace"]["name"] = name
        if backend is not None:
            data["workspace"]["backend"] = backend.value
        if provider is not None:
            data["embeddings"]["provider"] = provider.value
        config._data = data
        config._location = None
        return config

    @classmethod
    def reset(cls) -> None:
        """Drop the cached instance. Intended for test isolation only.

        pytest never reimports modules, so an instance cached on the class
        would leak into the next test; the autouse fixture in
        `tests/conftest.py` calls this around every case. The CLI calls it
        too, in its callback, because one command must always start from
        disk state (see `wsindex.cli`).
        """
        cls._instance = None

    @property
    def location(self) -> ConfigLocation | None:
        """Where the document was read from, or None if these are defaults."""
        return self._location

    @property
    def path(self) -> Path | None:
        """The config file backing this object, or None when it has none."""
        return self._location.path if self._location is not None else None

    @property
    def is_default(self) -> bool:
        """True when no file was found and `DEFAULT` is what you are reading."""
        return self._location is None

    @property
    def index_dir(self) -> Path:
        """Where the index database for this workspace lives.

        Follows the config location, not the config contents (see
        `wsindex.paths.resolve_index_dir`): `.wsindex/` next to a workspace
        config, `$XDG_DATA_HOME/wsindex/<id>/` for a user or system one.

        Raises:
            ValueError: This config came from `DEFAULT`, so there is no
                location to hang an index off — there is no workspace yet.
        """
        if self._location is None:
            raise ValueError("no index dir: this config was built from defaults")
        return resolve_index_dir(self._location, self.name)

    @property
    def name(self) -> str:
        """Workspace name; identification only, nothing derives from it."""
        return str(self._data["workspace"]["name"])

    @property
    def backend(self) -> Backend:
        """Which VectorStore the composition root builds."""
        return Backend(self._data["workspace"]["backend"])

    @property
    def provider(self) -> Provider:
        """Which Embedder the composition root builds."""
        return Provider(self._data["embeddings"]["provider"])

    @property
    def model(self) -> str:
        """Embedding model name (both local and server-side)."""
        return str(self._data["embeddings"]["model"])

    @property
    def dim(self) -> int:
        """Vector dimensionality the model produces."""
        return int(self._data["embeddings"]["dim"])

    @property
    def base_url(self) -> str:
        """Tensorus server root; unused by the local backend."""
        return str(self._data["tensorus"]["base_url"])

    @property
    def metric(self) -> str:
        """Similarity metric datasets are created with."""
        return str(self._data["tensorus"]["metric"])

    @property
    def repos(self) -> list[Repository]:
        """Repositories to index, in search merge-order.

        Built fresh on every access, so mutating the returned list does
        nothing to the config — `add_repo` is the way in.
        """
        return [
            Repository(id=str(repo["id"]), path=str(repo["path"])) for repo in self._data["repos"]
        ]

    def add_repo(self, repo_id: str, *, path: str) -> None:
        """Register a repository in the document (in memory; `save` is separate).

        Args:
            repo_id: Unique repo id; becomes the dataset name.
            path: Repository root directory.

        Raises:
            ValueError: The id is already registered — ids name datasets,
                so a duplicate would silently merge two repos into one.
        """
        if any(repo.id == repo_id for repo in self.repos):
            raise ValueError(f"repo id already exists: {repo_id}")
        self._data["repos"].append({"id": repo_id, "path": path})

    def to_dict(self) -> dict[str, Any]:
        """The document as it would be written.

        Returns:
            A deep copy in the `wsindex.toml` section layout, so callers
            cannot reach through it and mutate the live config.
        """
        return copy.deepcopy(self._data)

    def save(self, path: Path | None = None) -> Path:
        """Write the document to disk as UTF-8 TOML.

        Saving does not change where this object reads from: the next
        process discovers the file normally.

        Args:
            path: Target file, overwritten if present. When None, writes
                back to where the config was read from.

        Returns:
            The path written.

        Raises:
            ValueError: No path given and this config came from `DEFAULT`,
                so there is nowhere to write back to.
        """
        target = path or self.path
        if target is None:
            raise ValueError("no path to save to: this config was built from defaults")
        target.write_text(tomli_w.dumps(self._data), encoding="utf-8")
        return target
