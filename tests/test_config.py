"""Config tests: the document (defaults, roundtrip, validation) and the
singleton (one instance, idempotent init, reset, defaults fallback).

The autouse fixture in conftest.py drops the cached instance around every
test and rigs discovery to find nothing, so each case starts from "nothing
constructed yet, no config anywhere" and only sees the files it writes.
"""

import tomllib
from pathlib import Path

import pytest
import tomli_w

from wsindex.config import Backend, Config, Provider, Repository
from wsindex.paths import Mode


def test_roundtrip(tmp_path: Path) -> None:
    # Non-default provider, so the roundtrip proves the field really travels
    # through the file instead of passing on a hardcoded default.
    config = Config.default("demo", provider=Provider.FAKE)
    config.add_repo("test1", path="path1/test1")
    config.add_repo("test2", path="path2/test2")
    saved = config.to_dict()
    path = config.save(tmp_path / "wsindex.toml")

    Config.reset()
    assert Config(path).to_dict() == saved


def test_default_values() -> None:
    config = Config.default("demo")
    assert config.name == "demo"
    assert config.backend == Backend.LOCAL
    assert config.provider == Provider.SENTENCE_TRANSFORMERS
    assert config.dim == 384
    assert config.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.base_url == "http://localhost:8000"
    assert config.metric == "cosine"
    assert config.repos == []
    assert config.is_default


def test_default_overrides_reach_the_document() -> None:
    config = Config.default("demo", backend=Backend.TENSORUS, provider=Provider.FAKE)
    assert config.backend == Backend.TENSORUS
    assert config.to_dict()["workspace"]["backend"] == "tensorus"
    assert config.to_dict()["embeddings"]["provider"] == "fake"


def test_add_repo_duplicate_raises() -> None:
    config = Config.default("demo")
    config.add_repo("test1", path="path1/test1")
    with pytest.raises(ValueError, match="already exists"):
        config.add_repo("test1", path="path1/test1")


def test_repos_property_is_a_copy() -> None:
    config = Config.default("demo")
    config.repos.append(Repository(id="ghost", path="nowhere"))
    assert config.repos == []


def test_config_without_provider_is_rejected(tmp_path: Path) -> None:
    # Strict schema on purpose: a silent default would mask typos in the file.
    data = Config.default("demo").to_dict()
    del data["embeddings"]["provider"]
    path = tmp_path / "wsindex.toml"
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    Config.reset()
    with pytest.raises(KeyError):
        Config(path)


def test_config_without_section_is_rejected(tmp_path: Path) -> None:
    data = Config.default("demo").to_dict()
    del data["tensorus"]
    path = tmp_path / "wsindex.toml"
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    Config.reset()
    with pytest.raises(KeyError):
        Config(path)


def test_unknown_provider_value_is_rejected(tmp_path: Path) -> None:
    data = Config.default("demo").to_dict()
    data["embeddings"]["provider"] = "nonsense"
    path = tmp_path / "wsindex.toml"
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    Config.reset()
    with pytest.raises(ValueError, match="nonsense"):
        Config(path)


def test_repo_entry_without_path_is_rejected(tmp_path: Path) -> None:
    data = Config.default("demo").to_dict()
    data["repos"] = [{"id": "orphan"}]
    path = tmp_path / "wsindex.toml"
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    Config.reset()
    with pytest.raises(ValueError, match="'id' and 'path'"):
        Config(path)


def test_file_without_repos_section_loads_as_empty(tmp_path: Path) -> None:
    data = Config.default("demo").to_dict()
    del data["repos"]
    path = tmp_path / "wsindex.toml"
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    Config.reset()
    assert Config(path).repos == []


def test_saved_file_is_valid_toml(tmp_path: Path) -> None:
    config = Config.default("demo")
    config.add_repo("test1", path="path1/test1")
    config.add_repo("test2", path="path2/test2")
    config.save(tmp_path / "wsindex.toml")
    data = tomllib.loads((tmp_path / "wsindex.toml").read_text())
    assert data["workspace"]["backend"] == "local"
    assert data["repos"][0]["id"] == "test1"


def test_save_without_a_path_needs_a_source_file(tmp_path: Path) -> None:
    config = Config.default("demo")
    with pytest.raises(ValueError, match="built from defaults"):
        config.save()

    path = config.save(tmp_path / "wsindex.toml")
    Config.reset()
    loaded = Config(path)
    loaded.add_repo("late", path="somewhere")
    assert loaded.save() == path
    assert "late" in path.read_text()


# The singleton itself: one instance, an __init__ that runs once, reset.


def test_config_is_one_instance(tmp_path: Path) -> None:
    path = Config.default("demo").save(tmp_path / "wsindex.toml")
    Config.reset()
    # `is`, not `==`: singleton means one object shared, not equal values.
    assert Config(path) is Config()


def test_second_construction_does_not_reload(tmp_path: Path) -> None:
    path = Config.default("from-file").save(tmp_path / "wsindex.toml")
    Config.reset()
    Config(path)
    # Discovery is rigged to find nothing; a second __init__ pass would
    # have replaced the loaded document with DEFAULT.
    assert Config().name == "from-file"
    assert not Config().is_default


def test_reset_drops_the_instance() -> None:
    first = Config.default("demo")
    Config.reset()
    assert Config.default("demo") is not first


def test_missing_file_falls_back_to_defaults(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = Config()
    assert config.is_default
    assert config.location is None
    assert config.path is None
    assert config.name == "default"
    # A fallback must not materialize anything: only `wsindex init` writes.
    assert list(tmp_path.iterdir()) == []
    # Nor say anything: reporting it needs a user, which this module has
    # no idea about — the CLI does it in `_config`.
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_missing_explicit_path_falls_back_to_defaults(tmp_path: Path) -> None:
    missing = tmp_path / "nope.toml"
    config = Config(missing)
    assert config.is_default
    assert not missing.exists()


def test_explicit_path_gets_override_mode(tmp_path: Path) -> None:
    # Mode drives where the index lives; an explicitly named file behaves
    # like $WSINDEX_CONFIG — index next to the config, not in XDG data.
    path = Config.default("demo").save(tmp_path / "wsindex.toml")
    Config.reset()
    location = Config(path).location
    assert location is not None
    assert location.mode == Mode.OVERRIDE
    assert location.path == path


def test_index_dir_follows_the_config_location(tmp_path: Path) -> None:
    path = Config.default("demo").save(tmp_path / "wsindex.toml")
    Config.reset()
    # Explicit path behaves like $WSINDEX_CONFIG: index next to the config.
    assert Config(path).index_dir == tmp_path / ".wsindex"


def test_index_dir_without_a_config_file_raises() -> None:
    config = Config.default("demo")
    with pytest.raises(ValueError, match="built from defaults"):
        _ = config.index_dir


def test_default_constant_is_never_mutated() -> None:
    first = Config.default("first")
    first.add_repo("leak", path="somewhere")
    Config.reset()
    # Without the deepcopy in `default`, add_repo above would have appended
    # straight into Config.DEFAULT and every later config would carry it.
    assert Config.default("second").repos == []
    assert Config.DEFAULT["repos"] == []
    assert Config.DEFAULT["workspace"]["name"] == "default"


def test_to_dict_cannot_reach_the_live_document() -> None:
    config = Config.default("demo")
    config.to_dict()["workspace"]["name"] = "hijacked"
    assert config.name == "demo"


def test_subclass_gets_its_own_instance() -> None:
    class Sub(Config):
        pass

    # `_instance` is a plain class attribute, so a naive `hasattr(cls,
    # "_instance")` check would hand Sub() the parent's instance.
    assert Sub() is not Config()
    assert isinstance(Sub(), Sub)
