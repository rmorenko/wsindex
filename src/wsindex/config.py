"""Workspace configuration: the `wsindex.toml` file (defaults, load, save).

The config is the single source of truth for a workspace: which vector-store
backend to use, which embedding model, and which repositories to index. The
TOML layout (sections `workspace`, `embeddings`, `tensorus`, `repos`) lives
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
        store_uri: Relative path to local lancedb or s3 URI
        backend: Which VectorStore the composition root builds.
        provider: Which Embedder the composition root builds.
        model: Embedding model name (both local and server-side).
        dim: Vector dimensionality the model produces.
        base_url: Tensorus server root; unused by the local backend.
        metric: Similarity metric datasets are created with.
        repos: Repositories to index, in search merge-order.
    """

    name: str
    store_uri: str
    backend: Backend
    provider: Provider
    model: str
    dim: int
    base_url: str
    metric: str
    repos: list[Repository]

    @classmethod
    def default_config(cls, name: str) -> "Config":
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
            store_uri=DEFAULT_URI,
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
            "store": {"uri": self.store_uri},
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
        """
        ws = config_dict["workspace"]
        emb = config_dict["embeddings"]
        ts = config_dict["tensorus"]
        store_uri = config_dict.get("store", {}).get("uri", DEFAULT_URI)
        return cls(
            name=ws["name"],
            backend=Backend(ws["backend"]),
            provider=Provider(emb["provider"]),
            model=emb["model"],
            dim=emb["dim"],
            base_url=ts["base_url"],
            metric=ts["metric"],
            repos=[Repository(**r) for r in config_dict.get("repos", [])],
            store_uri=store_uri,
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
