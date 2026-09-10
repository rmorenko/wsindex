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

from wsindex.config import (
    DEFAULT_RANK_MODEL,
    Backend,
    Config,
    Provider,
    Repository,
    RepoSource,
)
from wsindex.model import Kind
from wsindex.paths import Mode


def test_roundtrip(tmp_path: Path) -> None:
    # Non-default provider, so the roundtrip proves the field really travels
    # through the file instead of passing on a hardcoded default.
    config = Config.default("demo", provider=Provider.FAKE)
    config.add_repo(Repository(id="test1", path="path1/test1"))
    config.add_repo(Repository(id="test2", path="path2/test2"))
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
    assert config.metric == "cosine"
    assert config.repos == []
    assert not config.rank_enabled
    assert config.rank_model == DEFAULT_RANK_MODEL
    assert config.is_default


def test_default_overrides_reach_the_document() -> None:
    config = Config.default("demo", backend=Backend.LOCAL, provider=Provider.FAKE)
    assert config.provider == Provider.FAKE
    assert config.to_dict()["workspace"]["backend"] == "local"
    assert config.to_dict()["embeddings"]["provider"] == "fake"


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
    Config.reset()
    with pytest.raises(ValueError, match="tensorus era"):
        Config(path)


def test_store_uri_defaults_to_the_index_dir(tmp_path: Path) -> None:
    # No `[store] uri` in the document means "wherever this workspace's
    # index belongs" — a relative default would follow the CWD instead.
    data = Config.default("demo").to_dict()
    assert "uri" not in data["store"]
    path = tmp_path / "wsindex.toml"
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    Config.reset()
    assert Config(path).store_uri == str(tmp_path / ".wsindex")


def test_explicit_store_uri_wins(tmp_path: Path) -> None:
    # The escape hatch for shared storage: an s3 uri must survive intact.
    data = Config.default("demo").to_dict()
    data["store"]["uri"] = "s3://bucket/prefix"
    path = tmp_path / "wsindex.toml"
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    Config.reset()
    assert Config(path).store_uri == "s3://bucket/prefix"


def test_missing_store_and_rank_sections_still_load(tmp_path: Path) -> None:
    # Configs written before these sections existed must keep loading:
    # they are the defaulted exceptions, everything else stays strict.
    data = Config.default("demo").to_dict()
    del data["store"]
    del data["rank"]
    path = tmp_path / "wsindex.toml"
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    Config.reset()
    config = Config(path)
    assert config.metric == "cosine"
    assert not config.rank_enabled
    assert config.rank_model == DEFAULT_RANK_MODEL
    assert config.store_uri == str(tmp_path / ".wsindex")


def test_add_repo_duplicate_raises() -> None:
    config = Config.default("demo")
    config.add_repo(Repository(id="test1", path="path1/test1"))
    with pytest.raises(ValueError, match="already exists"):
        config.add_repo(Repository(id="test1", path="path1/test1"))


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
    del data["embeddings"]
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
    config.add_repo(Repository(id="test1", path="path1/test1"))
    config.add_repo(Repository(id="test2", path="path2/test2"))
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
    loaded.add_repo(Repository(id="late", path="somewhere"))
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


def test_store_uri_without_a_config_file_raises() -> None:
    # store_uri falls through to index_dir, so it inherits the same rule:
    # no file on disk means there is no workspace to hold an index.
    config = Config.default("demo")
    with pytest.raises(ValueError, match="built from defaults"):
        _ = config.store_uri


def test_default_constant_is_never_mutated() -> None:
    first = Config.default("first")
    first.add_repo(Repository(id="leak", path="somewhere"))
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


# --- an optional remote per repo -----------------------------------------


def test_repo_without_a_remote_has_none() -> None:
    config = Config.default("demo")
    config.add_repo(Repository(id="local", path="/checkouts/local"))
    assert config.repos[0].remote is None


def test_remote_round_trips_through_the_file(tmp_path: Path) -> None:
    config = Config.default("demo")
    config.add_repo(
        Repository(id="upstream", path="/checkouts/up", remote="https://example.invalid/r.git")
    )
    path = config.save(tmp_path / "wsindex.toml")

    Config.reset()
    assert Config(path).repos[0].remote == "https://example.invalid/r.git"


def test_omitted_remote_is_absent_from_the_document() -> None:
    # TOML has no null and tomli_w refuses to write one, so "no remote"
    # has to mean "no key" rather than an explicit None.
    config = Config.default("demo")
    config.add_repo(Repository(id="local", path="/checkouts/local"))
    assert "remote" not in config.to_dict()["repos"][0]


def test_empty_remote_string_reads_as_no_remote(tmp_path: Path) -> None:
    # A hand-edited `remote = ""` means the user cleared it, not that
    # sync should try to clone from an empty url.
    data = Config.default("demo").to_dict()
    data["repos"] = [{"id": "r", "path": "/p", "remote": ""}]
    path = tmp_path / "wsindex.toml"
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    Config.reset()
    assert Config(path).repos[0].remote is None


# --- snapshot repos ------------------------------------------------------


def write(tmp_path: Path, repos: list[dict[str, object]]) -> Path:
    data = Config.default("demo").to_dict()
    data["repos"] = repos
    path = tmp_path / "wsindex.toml"
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    Config.reset()
    return path


def test_a_snapshot_repo_round_trips(tmp_path: Path) -> None:
    config = Config.default("demo")
    config.add_repo(
        Repository(id="docs", path="snap", source=RepoSource.CONNECTOR, urls=("https://x/a",))
    )
    path = config.save(tmp_path / "wsindex.toml")

    Config.reset()
    repo = Config(path).repos[0]
    assert repo.is_snapshot
    assert repo.urls == ("https://x/a",)


def test_an_ordinary_repo_is_not_a_snapshot() -> None:
    config = Config.default("demo")
    config.add_repo(Repository(id="r", path="/p"))
    assert not config.repos[0].is_snapshot


def test_a_repo_with_two_owners_is_rejected(tmp_path: Path) -> None:
    # A working copy is either pulled or materialized. Both would mean
    # sync fast-forwards the files it just wrote, or the reverse.
    path = write(tmp_path, [{"id": "r", "path": "p", "source": "connector", "remote": "u"}])
    with pytest.raises(ValueError, match="not both"):
        Config(path)


def test_urls_without_a_source_are_a_typo_worth_naming(tmp_path: Path) -> None:
    path = write(tmp_path, [{"id": "r", "path": "p", "urls": ["https://x/a"]}])
    with pytest.raises(ValueError, match="no 'source'"):
        Config(path)


def test_an_unknown_source_is_rejected_at_load(tmp_path: Path) -> None:
    # Strict, unlike `connectors`: a repo that fails to load is a repo
    # that silently stops being indexed.
    path = write(tmp_path, [{"id": "r", "path": "p", "source": "telepathy"}])
    with pytest.raises(ValueError, match="telepathy"):
        Config(path)


def test_add_repo_refuses_the_same_contradiction() -> None:
    config = Config.default("demo")
    with pytest.raises(ValueError, match="not both"):
        config.add_repo(Repository(id="r", path="p", source=RepoSource.CONNECTOR, remote="u"))
    assert not config.repos


# --- per-repo markup -----------------------------------------------------


def test_per_repo_markup_round_trips(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        [
            {
                "id": "app",
                "path": "p",
                "ignore": ["dist/*", "*.min.js"],
                "formats": {".sql": {"lang": "sql", "kind": "code"}},
            }
        ],
    )
    repo = Config(path).repos[0]
    assert repo.ignore == ("dist/*", "*.min.js")
    assert repo.formats == {".sql": ("sql", Kind.CODE)}


def test_an_unknown_repo_key_is_refused(tmp_path: Path) -> None:
    # The failure this format is most likely to produce: a misspelled
    # `ignores` that silently indexes everything it was meant to exclude.
    path = write(tmp_path, [{"id": "app", "path": "p", "ignores": ["dist/*"]}])
    with pytest.raises(ValueError, match="unknown key"):
        Config(path)


def test_a_suffix_without_its_dot_is_refused(tmp_path: Path) -> None:
    # ".sql" is a suffix, "sql" matches no file at all.
    path = write(
        tmp_path, [{"id": "a", "path": "p", "formats": {"sql": {"lang": "s", "kind": "code"}}}]
    )
    with pytest.raises(ValueError, match="start with a dot"):
        Config(path)


def test_a_format_needs_both_lang_and_kind(tmp_path: Path) -> None:
    path = write(tmp_path, [{"id": "a", "path": "p", "formats": {".sql": {"lang": "sql"}}}])
    with pytest.raises(ValueError, match="needs both"):
        Config(path)


def test_a_file_may_not_be_marked_as_a_commit(tmp_path: Path) -> None:
    # Commit chunks come from git history and have no path on disk;
    # a file claiming that kind would collide with them in every filter.
    path = write(
        tmp_path, [{"id": "a", "path": "p", "formats": {".txt": {"lang": "t", "kind": "commit"}}}]
    )
    with pytest.raises(ValueError, match="not a file kind"):
        Config(path)


def test_an_unknown_kind_is_refused(tmp_path: Path) -> None:
    path = write(
        tmp_path, [{"id": "a", "path": "p", "formats": {".txt": {"lang": "t", "kind": "prose"}}}]
    )
    with pytest.raises(ValueError, match="prose"):
        Config(path)


def test_add_repo_records_ignore_globs() -> None:
    config = Config.default("demo")
    config.add_repo(Repository(id="app", path="p", ignore=("dist/*",)))
    assert config.repos[0].ignore == ("dist/*",)
    assert config.to_dict()["repos"][0]["ignore"] == ["dist/*"]


def test_a_saved_config_can_be_hand_edited(tmp_path: Path) -> None:
    # The trap `connectors = []` fell into, and `repos` had
    # been in it from the start: tomli_w writes a short repo entry as
    # `repos = [{...}]`, and TOML forbids attaching `[[repos]]` or
    # `[repos.formats]` to a static array. Whatever a person is told to
    # hand-edit has to be a shape they can hand-edit.
    config = Config.default("demo")
    config.add_repo(Repository(id="app", path="p", ignore=("dist/*",)))
    config._data["repos"][0]["formats"] = {".sql": {"lang": "sql", "kind": "code"}}
    path = config.save(tmp_path / "wsindex.toml")

    text = path.read_text(encoding="utf-8")
    assert "[[repos]]" in text
    assert "repos = [" not in text
    # And the hand-editing itself works: append a second repo by hand.
    path.write_text(text + '\n[[repos]]\nid = "second"\npath = "q"\n', encoding="utf-8")

    Config.reset()
    repos = Config(path).repos
    assert [r.id for r in repos] == ["app", "second"]
    assert repos[0].formats == {".sql": ("sql", Kind.CODE)}


def test_a_config_with_no_repos_leaves_room_for_one(tmp_path: Path) -> None:
    # `repos = []` would be the same trap: a static empty array cannot
    # become an array of tables.
    path = Config.default("demo").save(tmp_path / "wsindex.toml")
    assert "repos" not in path.read_text(encoding="utf-8")

    path.write_text(
        path.read_text(encoding="utf-8") + '\n[[repos]]\nid = "a"\npath = "p"\n', encoding="utf-8"
    )
    Config.reset()
    assert [r.id for r in Config(path).repos] == ["a"]


def test_max_commits_defaults_to_absent_rather_than_to_a_number(tmp_path: Path) -> None:
    """None, not `MAX_COMMITS`, and the reason is layering.

    Config sits under everything and is imported by every command.
    Reaching up into `wsindex.ingest` for a default would put 30 ms of
    tree-sitter and model imports behind `wsindex --help`; `read_commits`
    applies the default instead, where it is documented.
    """
    Config.reset()
    config = Config.default("w")

    assert config.max_commits is None


def test_max_commits_is_read_from_the_index_section(tmp_path: Path) -> None:
    Config.reset()
    config = Config.default("w")
    config._data["index"] = {"max_commits": 250}

    assert config.max_commits == 250


def test_the_index_section_is_accepted_by_the_validator(tmp_path: Path) -> None:
    # The schema is generated from the same constants the validator uses,
    # so a section the generator does not know is a section that makes a
    # valid config unloadable.
    path = tmp_path / "wsindex.toml"
    path.write_text(
        '[workspace]\nname = "w"\nbackend = "local"\n\n'
        '[embeddings]\nmodel = "m"\ndim = 8\nprovider = "fake"\n\n'
        "[index]\nmax_commits = 250\n"
    )
    Config.reset()

    assert Config(path).max_commits == 250
