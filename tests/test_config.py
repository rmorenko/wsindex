import tomllib
from pathlib import Path

import pytest
import tomli_w

from wsindex.config import (
    Backend,
    Config,
    Provider,
    get_config,
    load_config,
    reset_config,
    save_config,
    set_config,
)


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
    assert config.backend == Backend.TENSORUS
    assert config.dim == 384
    assert config.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.base_url == "http://localhost:8000"
    assert config.metric == "cosine"
    assert config.provider == Provider.SENTENCE_TRANSFORMERS
    assert config.repos == []


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
    assert data["workspace"]["backend"] == "tensorus"
    assert data["repos"][0]["id"] == "test1"


# Module-level singleton: get/set/reset + load_config side effect.
# The autouse fixture in conftest.py resets the module slot around every
# test, so each case starts with `get_config()` in the "not loaded" state.


def test_get_config_without_load_raises() -> None:
    # No load_config/set_config called in this test — slot must be empty.
    with pytest.raises(RuntimeError):
        get_config()


def test_set_config_publishes_to_module_slot() -> None:
    cfg = Config.default_config("demo")
    set_config(cfg)
    # `is`, not `==`: singleton means one object shared, not equal values.
    assert get_config() is cfg


def test_load_config_publishes_to_module_slot(tmp_path: Path) -> None:
    original = Config.default_config("demo")
    save_config(original, path=tmp_path / "wsindex.toml")
    loaded = load_config(tmp_path / "wsindex.toml")
    assert get_config() is loaded


def test_reset_config_clears_slot() -> None:
    set_config(Config.default_config("demo"))
    reset_config()
    with pytest.raises(RuntimeError):
        get_config()


def test_two_independent_configs_dont_collide() -> None:
    # The point of dropping the __new__ hack: two Config(...) calls now
    # produce two distinct objects. Before the refactor this failed —
    # `a` and `b` shared the same instance and `a.name` became "b".
    a = Config.default_config("a")
    b = Config.default_config("b")
    assert a is not b
    assert a.name == "a"
    assert b.name == "b"
