import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import tomli_w


class Backend(StrEnum):
    TENSORUS = "tensorus"
    LOCAL = "local"


@dataclass(frozen=True)
class RepoEntry:
    id: str
    path: str


@dataclass
class Config:
    name: str
    backend: Backend
    model: str
    dim: int
    base_url: str
    metric: str
    repos: list[RepoEntry]

    @classmethod
    def default_config(cls, name: str) -> "Config":
        return cls(
            name=name,
            backend=Backend.TENSORUS,
            model="sentence-transformers/all-MiniLM-L6-v2",
            dim=384,
            base_url="http://localhost:8080",
            metric="cosine",
            repos=[],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": {"name": self.name, "backend": self.backend},
            "embeddings": {"model": self.model, "dim": self.dim},
            "tensorus": {"base_url": self.base_url, "metric": self.metric},
            "repos": [{"id": r.id, "path": r.path} for r in self.repos],
        }

    def add_repo(self, repo_id: str, path: str) -> None:
        if any(r.id == repo_id for r in self.repos):
            raise ValueError(f"repo id already exists: {repo_id}")
        self.repos.append(RepoEntry(id=repo_id, path=path))

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        ws = config_dict["workspace"]
        emb = config_dict["embeddings"]
        ts = config_dict["tensorus"]
        return cls(
            name=ws["name"],
            backend=Backend(ws["backend"]),
            model=emb["model"],
            dim=emb["dim"],
            base_url=ts["base_url"],
            metric=ts["metric"],
            repos=[RepoEntry(**r) for r in config_dict.get("repos", [])],
        )


def save_config(config: Config, path: Path) -> None:
    path.write_text(tomli_w.dumps(config.to_dict()), encoding="utf-8")


def load_config(path: Path) -> Config:
    config_dict = tomllib.loads(path.read_text())
    return Config.from_dict(config_dict)
