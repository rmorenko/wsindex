import tomllib
from pathlib import Path

import pytest

from wsindex.config import Backend, Config, load_config, save_config


def test_roundtrip(tmp_path: Path) -> None:
    config = Config.default_config("demo")
    config.add_repo(repo_id="test1", path="path1/test1")
    config.add_repo(repo_id="test2", path="path2/test2")
    save_config(config, tmp_path / "wsindex.toml")
    assert config == load_config(tmp_path / "wsindex.toml")


def test_default_config(tmp_path: Path) -> None:
    config = Config.default_config("demo")
    assert config.backend == Backend.TENSORUS
    assert config.dim == 384
    assert config.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.base_url == "http://localhost:8080"
    assert config.metric == "cosine"
    assert config.repos == []


def test_add_repo_duplicate_raises() -> None:

    config = Config.default_config("demo")
    config.add_repo(repo_id="test1", path="path1/test1")
    with pytest.raises(ValueError):
        config.add_repo(repo_id="test1", path="path1/test1")


def test_saved_file_is_valid_toml(tmp_path: Path) -> None:
    config = Config.default_config("demo")
    config.add_repo(repo_id="test1", path="path1/test1")
    config.add_repo(repo_id="test2", path="path2/test2")
    save_config(config, tmp_path / "wsindex.toml")
    data = tomllib.loads((tmp_path / "wsindex.toml").read_text())
    assert data["workspace"]["backend"] == "tensorus"
    assert data["repos"][0]["id"] == "test1"
