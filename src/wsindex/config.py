"""Workspace configuration: the `wsindex.toml` file (defaults, load, save).

The config is the single source of truth for a workspace: which vector-store
backend to use, which embedding model, and which repositories to index. The
TOML layout (sections `workspace`, `embeddings`, `store`, `repos`) lives
only in `to_dict`/`from_dict`, so the file format has one definition per
direction.
"""

import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import tomli_w

DEFAULT_URI = ".wsindex"


class Backend(StrEnum):
    """Vector store selector; StrEnum so the value round-trips through TOML as-is.

    A single member since ADR-7 removed Tensorus — kept as an enum so a
    future backend is a data change, not an API change.
    """

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
        store_uri: LanceDB location — a local path or an s3:// uri.
        backend: Which VectorStore the composition root builds.
        provider: Which Embedder the composition root builds.
        model: Embedding model name.
        dim: Vector dimensionality the model produces.
        metric: Similarity metric datasets are created with.
        repos: Repositories to index, in search merge-order.
    """

    name: str
    store_uri: str
    backend: Backend
    provider: Provider
    model: str
    dim: int
    metric: str
    repos: list[Repository]

    @classmethod
    def default_config(cls, name: str) -> "Config":
        """Config for a fresh workspace: local backend, MiniLM model, no repos.

        Args:
            name: Workspace name to bake into the config.

        Returns:
            A new Config with project defaults; the caller saves it to disk.
        """
        return cls(
            name=name,
            backend=Backend.LOCAL,
            provider=Provider.SENTENCE_TRANSFORMERS,
            model="sentence-transformers/all-MiniLM-L6-v2",
            dim=384,
            metric="cosine",
            store_uri=DEFAULT_URI,
            repos=[],
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for saving.

        Returns:
            Nested dict in the `wsindex.toml` section layout (`workspace`,
            `embeddings`, `store`, `repos`) — input for tomli_w.
        """
        return {
            "workspace": {"name": self.name, "backend": self.backend},
            "embeddings": {"model": self.model, "dim": self.dim, "provider": self.provider},
            "store": {"uri": self.store_uri, "metric": self.metric},
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
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Inverse of `to_dict`.

        Args:
            config_dict: Parsed TOML in the `wsindex.toml` section layout.

        Returns:
            The equivalent Config; `repos` may be absent and defaults to [].

        Raises:
            KeyError: A required section or key is missing — strict on
                purpose, a half-read config must not survive silently.
                The `store` section is the one defaulted exception
                (format evolution without a migrator).
            ValueError: The config is from the removed tensorus era.
        """
        ws = config_dict["workspace"]
        if "tensorus" in config_dict or ws.get("backend") == "tensorus":
            raise ValueError(
                "this wsindex.toml is from the tensorus era, which ADR-7 removed — "
                "recreate it with `wsindex init` and re-index (ids are deterministic, "
                "re-indexing is cheap)"
            )
        emb = config_dict["embeddings"]
        store = config_dict.get("store", {})
        return cls(
            name=ws["name"],
            backend=Backend(ws["backend"]),
            provider=Provider(emb["provider"]),
            model=emb["model"],
            dim=emb["dim"],
            metric=store.get("metric", "cosine"),
            repos=[Repository(**r) for r in config_dict.get("repos", [])],
            store_uri=store.get("uri", DEFAULT_URI),
        )


def save_config(config: Config, *, path: Path) -> None:
    """Serialize the config to disk as UTF-8 TOML.

    Args:
        config: Config to write.
        path: Target file, overwritten if present.
    """
    path.write_text(tomli_w.dumps(config.to_dict()), encoding="utf-8")


def load_config(path: Path) -> Config:
    """Parse a `wsindex.toml` file into a Config.

    Args:
        path: File to read.

    Returns:
        The parsed Config.

    Raises:
        FileNotFoundError: The file does not exist.
        KeyError: A required section or key is missing (see `from_dict`).
    """
    config_dict = tomllib.loads(path.read_text())
    return Config.from_dict(config_dict)
