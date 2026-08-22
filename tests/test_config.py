import tomllib
from pathlib import Path

import pytest
import tomli_w

from wsindex.config import DEFAULT_URI, Backend, Config, Provider, load_config, save_config


def test_roundtrip(tmp_path: Path) -> None:
    config = Config.default_config("demo")
    config.add_repo(repo_id="test1", path="path1/test1")
    config.add_repo(repo_id="test2", path="path2/test2")
    # Non-default provider, so the roundtrip proves the field really travels
    # through the file instead of passing on a hardcoded default.
    config.provider = Provider.FAKE
    save_config(config, path=tmp_path / "wsindex.toml")
    assert config == load_config(tmp_path / "wsindex.toml")


def test_default_config(tmp_path: Path) -> None:
    config = Config.default_config("demo")
    assert config.backend == Backend.LOCAL
    assert config.dim == 384
    assert config.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.metric == "cosine"
    assert config.provider == Provider.SENTENCE_TRANSFORMERS
    assert config.repos == []
    assert config.store_uri == DEFAULT_URI
    assert not config.rank_enabled
    assert config.rank_model == "cross-encoder/ms-marco-MiniLM-L6-v2"


def test_tensorus_era_config_is_rejected(tmp_path: Path) -> None:
    # Pre-ADR-7 configs carry a `[tensorus]` section and backend="tensorus";
    # loading one must fail loudly with a hint to re-init, not silently.
    legacy = {
        "workspace": {"name": "demo", "backend": "tensorus"},
        "embeddings": {"model": "m", "dim": 8, "provider": "fake"},
        "tensorus": {"base_url": "http://x", "metric": "cosine"},
    }
    path = tmp_path / "wsindex.toml"
    path.write_text(tomli_w.dumps(legacy), encoding="utf-8")
    with pytest.raises(ValueError, match="tensorus era"):
        load_config(path)


def test_missing_store_section_gets_default_uri() -> None:
    # Configs written before the [store] section existed must keep loading;
    # this is the only defaulted key — everything else stays strict.
    config_dict = Config.default_config("demo").to_dict()
    del config_dict["store"]
    assert Config.from_dict(config_dict).store_uri == DEFAULT_URI


def test_add_repo_duplicate_raises() -> None:
    config = Config.default_config("demo")
    config.add_repo(repo_id="test1", path="path1/test1")
    with pytest.raises(ValueError):
        config.add_repo(repo_id="test1", path="path1/test1")


def test_config_without_provider_is_rejected(tmp_path: Path) -> None:
    # Strict schema on purpose: a silent default would mask typos in the file.
    data = Config.default_config("demo").to_dict()
    del data["embeddings"]["provider"]
    path = tmp_path / "wsindex.toml"
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(path)


def test_unknown_provider_value_is_rejected(tmp_path: Path) -> None:
    data = Config.default_config("demo").to_dict()
    data["embeddings"]["provider"] = "nonsense"
    path = tmp_path / "wsindex.toml"
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_saved_file_is_valid_toml(tmp_path: Path) -> None:
    config = Config.default_config("demo")
    config.add_repo(repo_id="test1", path="path1/test1")
    config.add_repo(repo_id="test2", path="path2/test2")
    save_config(config, path=tmp_path / "wsindex.toml")
    data = tomllib.loads((tmp_path / "wsindex.toml").read_text())
    assert data["workspace"]["backend"] == "local"
    assert data["repos"][0]["id"] == "test1"
