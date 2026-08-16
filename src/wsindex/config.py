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


class Backend(StrEnum):
    """Vector store selector; StrEnum so the value round-trips through TOML as-is."""

    TENSORUS = "tensorus"
    LOCAL = "local"


class Provider(StrEnum):
    FAKE = "fake"
    SENTENCE_TRANSFORMERS = "sentence-transformers"


@dataclass(frozen=True, kw_only=True)
class Repository:
    """One indexed repository: a stable id (used as the dataset name) and its path."""

    id: str
    path: str


@dataclass(kw_only=True)
class Config:
    """In-memory form of `wsindex.toml`. Mutable: `add_repo` edits it in place."""

    name: str
    backend: Backend
    provider: Provider
    model: str
    dim: int
    base_url: str
    metric: str
    repos: list[Repository]

    @classmethod
    def default_config(cls, name: str) -> "Config":
        """Config for a fresh workspace: Tensorus backend, MiniLM model, no repos."""
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
        """Nested dict in the `wsindex.toml` section layout — input for tomli_w."""
        return {
            "workspace": {"name": self.name, "backend": self.backend},
            "embeddings": {"model": self.model, "dim": self.dim, "provider": self.provider},
            "tensorus": {"base_url": self.base_url, "metric": self.metric},
            "repos": [{"id": r.id, "path": r.path} for r in self.repos],
        }

    def add_repo(self, repo_id: str, *, path: str) -> None:
        """Register a repository; ids must be unique because they name datasets."""
        if any(r.id == repo_id for r in self.repos):
            raise ValueError(f"repo id already exists: {repo_id}")
        self.repos.append(Repository(id=repo_id, path=path))

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Inverse of `to_dict`; raises KeyError if a required section is missing."""
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


def save_config(config: Config, *, path: Path) -> None:
    """Serialize the config to `path` as UTF-8 TOML."""
    path.write_text(tomli_w.dumps(config.to_dict()), encoding="utf-8")


def load_config(path: Path) -> Config:
    """Parse a `wsindex.toml` file into a Config."""
    config_dict = tomllib.loads(path.read_text())
    return Config.from_dict(config_dict)
