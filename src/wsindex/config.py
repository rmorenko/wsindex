"""Workspace configuration: the `wsindex.toml` file (defaults, load, save).

The config is the single source of truth for a workspace: which vector-store
backend to use, which embedding model, and which repositories to index. The
TOML layout (sections `workspace`, `embeddings`, `store`, `rank`, `repos`) is
also the in-memory form — `Config` keeps the parsed document as a dict and
exposes one read-only property per field, so the file format has a single
definition and every reader still goes through a typed accessor.

`Config` is a singleton class: `__new__` caches the one instance on the class,
so `Config()` anywhere in the process returns the same object and no subsystem
needs the config threaded through its constructor. The first construction
decides where the document comes from — `Config(path)` reads that file,
`Config()` discovers one (see `wsindex.paths.find_config`). Nothing on disk is
not an error: it means the built-in `Config.DEFAULT` and `is_default` set.
Saying so out loud belongs to whoever has a user to say it to — the CLI warns
in `wsindex.cli._config`; this module never writes to a stream. No file is
ever written as a side effect either; only `wsindex init` creates one.

`workspace` and `embeddings` are strict: a missing key there is a typo, and a
silent default would index the whole workspace with the wrong model. `store`
and `rank` are the defaulted exceptions — they arrived after the first configs
were written, and defaulting them is what lets the format evolve without a
migrator.

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
import json
import tomllib
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import blake2b
from pathlib import Path
from typing import Any, ClassVar

import tomli_w

from wsindex.connectors import ConnectorSpec
from wsindex.model import Kind
from wsindex.paths import ConfigLocation, Mode, find_config, resolve_index_dir

DEFAULT_RANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


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


class RepoSource(StrEnum):
    """Who fills a repository's working copy.

    Absent means the user does — a checkout they maintain, which sync
    leaves alone unless it has a `remote`. An enum with one member for
    the same reason `Backend` has one: a second kind of generated repo
    should be a data change.
    """

    CONNECTOR = "connector"


@dataclass(frozen=True, kw_only=True)
class Repository:
    """One indexed repository.

    A value object, unlike `Config`: repositories are compared and passed
    around by value, and there are many of them.

    Attributes:
        id: Stable unique name; doubles as the dataset name in the store.
        path: Repository root directory, absolute or workspace-relative.
        remote: Clone url `wsindex sync` keeps `path` up to date from, or
            None for a working copy the user manages themselves. Optional
            because the two cases are both normal: a repo you already have
            checked out needs no url, a repo the workspace should fetch
            for itself does.
        source: Set to `connector` for a snapshot repository — one whose
            files `wsindex sync` writes by fetching `urls` and
            committing them (see `wsindex.snapshot`). Mutually exclusive
            with `remote`:
            a working copy has exactly one owner.
        urls: The documents a snapshot repository holds. Meaningless
            without `source`, and rejected there — a url list on a git
            repo is a typo, not a preference.
        ignore: Path globs this repo excludes, on top of the walker's
            own pruning. Narrows only — there is no way to widen, by
            design (see `wsindex.ingest.walker._skip_dir`).
        formats: Suffix -> (lang, kind) for this repo, overriding the
            language registry. What to index is a property of a
            repository: `.component.html` is source in an Angular repo
            and generated noise in a Python one, and a global table
            cannot be right for both.
    """

    id: str
    path: str
    remote: str | None = None
    source: RepoSource | None = None
    urls: tuple[str, ...] = ()
    ignore: tuple[str, ...] = ()
    formats: dict[str, tuple[str, Kind]] = field(default_factory=dict)

    @property
    def is_snapshot(self) -> bool:
        """True when sync materializes this repo instead of pulling it."""
        return self.source is RepoSource.CONNECTOR

    @property
    def markup_key(self) -> str:
        """A fingerprint of what this repo indexes, for the index state.

        The commit alone cannot answer "is the index still right": edit
        `formats` and the same tree yields a different set of files,
        while git reports nothing changed at all. Incremental indexing
        would then skip the repo forever — which it did, until a live run
        caught it.

        A hash rather than the values, because `state.json` is a cache
        and a repo may carry fifty globs. Sorted before hashing so
        reordering the config is not a change.
        """
        payload = json.dumps(
            {
                "ignore": sorted(self.ignore),
                "formats": {k: list(v) for k, v in sorted(self.formats.items())},
            },
            sort_keys=True,
        )
        return blake2b(payload.encode(), digest_size=8).hexdigest()


def _repo_entry(repo: Repository) -> dict[str, Any]:
    """A repo as the TOML document holds it.

    Empty values are left out rather than written as null: TOML has no
    null, and `tomli_w` refuses to write one.
    """
    entry: dict[str, Any] = {"id": repo.id, "path": repo.path}
    if repo.remote is not None:
        entry["remote"] = repo.remote
    if repo.source is not None:
        entry["source"] = repo.source.value
    if repo.urls:
        entry["urls"] = list(repo.urls)
    if repo.ignore:
        entry["ignore"] = list(repo.ignore)
    if repo.formats:
        entry["formats"] = {
            suffix: {"lang": lang, "kind": kind.value}
            for suffix, (lang, kind) in repo.formats.items()
        }
    return entry


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
        # No `uri` on purpose: absent means "follow the config location"
        # (see `store_uri`). Baking `.wsindex` in would freeze a
        # CWD-relative path into a user-mode config, which is read from
        # `$XDG_CONFIG_HOME` and may run from anywhere.
        "store": {"metric": "cosine"},
        "rank": {"enabled": False, "model": DEFAULT_RANK_MODEL},
        # Empty on purpose. A built-in `PROJ-123` pattern is not
        # possible: measured on this repository it matched 198 times and
        # every hit was an internal number — ADR-7, FR-111 — not a
        # ticket. Which prefixes name a tracker is knowledge only the
        # workspace has, so nothing is recognized until it says.
        "references": {},
        "repos": [],
    }
    """Project defaults, in the on-disk layout. Never handed out directly:
    it is a mutable dict shared by the whole process, and `add_repo` would
    otherwise append to the constant itself — every user takes a deepcopy."""

    REQUIRED: ClassVar[dict[str, tuple[str, ...]]] = {
        "workspace": ("name", "backend"),
        "embeddings": ("model", "dim", "provider"),
    }
    """Sections and keys a config file must contain; `store`, `rank` and
    `repos` are optional and default per-key."""

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
                `repos` entry holds something unusable, or the document is
                from the removed tensorus era.
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
            ValueError: The document is from the removed tensorus era,
                `backend` or `provider` names no enum member, or a `repos`
                entry lacks `id`/`path`.
        """
        # Before the schema check, not after: a pre-ADR-7 file is complete
        # and valid on its own terms, so every other check would pass and
        # the user would get `ValueError: 'tensorus'` from the enum instead
        # of a sentence telling them what to do.
        if "tensorus" in data or data.get("workspace", {}).get("backend") == "tensorus":
            raise ValueError(
                "this wsindex.toml is from the tensorus era, which ADR-7 removed — "
                "recreate it with `wsindex init` and re-index (ids are deterministic, "
                "re-indexing is cheap)"
            )
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
            cls._validate_repo(repo)

    REPO_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"id", "path", "remote", "source", "urls", "ignore", "formats"}
    )
    """Every key a `[[repos]]` entry may carry. Closed on purpose: a
    misspelled `ignores` that silently indexed everything is the failure
    this format is most likely to produce, and the only cheap way to catch
    it is to refuse what we do not recognize."""

    @staticmethod
    def _validate_repo(repo: dict[str, Any]) -> None:
        """Check one `[[repos]]` entry beyond its required keys.

        Strict, unlike `connectors`, which skips a broken entry. The
        asymmetry is deliberate: a connector that fails to load routes
        nothing, while a repo that fails to load is a repo that silently
        stops being indexed — and the user would go looking for the
        missing search results, not for the typo.

        Args:
            repo: One parsed repo table.

        Raises:
            ValueError: An unknown key, `source` names no known kind, a
                snapshot also has a `remote` (two owners for one working
                copy), `urls` appear on a repo that is not a snapshot, or
                a `formats` entry is malformed.
        """
        unknown = sorted(set(repo) - Config.REPO_KEYS)
        if unknown:
            raise ValueError(
                f"repo {repo.get('id', '?')!r} has unknown key(s) {', '.join(unknown)} — "
                f"a repo entry may hold: {', '.join(sorted(Config.REPO_KEYS))}"
            )
        Config._validate_formats(repo)
        source = repo.get("source")
        if source is not None:
            # Constructing the enum is the check, as it is for `backend`.
            RepoSource(source)
            if repo.get("remote"):
                raise ValueError(
                    f"repo {repo['id']!r} has both 'source' and 'remote' — a working copy "
                    "is either pulled from a remote or materialized by connectors, not both"
                )
        elif repo.get("urls"):
            raise ValueError(
                f"repo {repo['id']!r} lists 'urls' but has no 'source' — add "
                'source = "connector" to materialize them'
            )

    @staticmethod
    def _validate_formats(repo: dict[str, Any]) -> None:
        """Check the per-repo suffix table, which is all hand-written.

        Every rule below describes a mapping that would otherwise fail
        silently — a suffix without its dot matches nothing, a `kind` the
        model does not have selects nothing — and silence here surfaces
        later as "why is my file not indexed?", the hardest question this
        tool can be asked.

        Args:
            repo: One parsed repo table.

        Raises:
            ValueError: A malformed suffix, a missing `lang`/`kind`, an
                unknown `kind`, or `commit`, which no file may claim.
        """
        for suffix, entry in (repo.get("formats") or {}).items():
            where = f"repo {repo.get('id', '?')!r}, formats[{suffix!r}]"
            if not suffix.startswith(".") or len(suffix) < 2:
                raise ValueError(f"{where}: a suffix must start with a dot, like '.sql'")
            if not isinstance(entry, dict) or not entry.get("lang") or not entry.get("kind"):
                raise ValueError(
                    f"{where}: needs both 'lang' and 'kind', e.g. "
                    '{ lang = "sql", kind = "code" }'
                )
            kind = Kind(entry["kind"])
            if kind is Kind.COMMIT:
                # Commit chunks are synthesized from git history and have
                # no path on disk; a file claiming to be one would collide
                # with that corpus in every filter that mentions it.
                raise ValueError(f"{where}: 'commit' is not a file kind — use code, config or doc")

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
        """Embedding model name."""
        return str(self._data["embeddings"]["model"])

    @property
    def dim(self) -> int:
        """Vector dimensionality the model produces."""
        return int(self._data["embeddings"]["dim"])

    @property
    def store_uri(self) -> str:
        """LanceDB location — a local path or an `s3://` uri.

        Absent from the document means "wherever this workspace's index
        belongs", i.e. `index_dir`: that keeps the default correct for a
        user-mode config, which is discovered from `$XDG_CONFIG_HOME` and
        must not drop a `.wsindex/` into whatever directory the user
        happened to run from. Set `[store] uri` explicitly to point the
        workspace at shared storage; credentials come from the AWS_* env.

        Raises:
            ValueError: No explicit uri and no config file to derive one
                from (see `index_dir`).
        """
        uri = self._data.get("store", {}).get("uri")
        return str(uri) if uri is not None else str(self.index_dir)

    @property
    def metric(self) -> str:
        """Similarity metric datasets are created with."""
        return str(self._data.get("store", {}).get("metric", "cosine"))

    @property
    def rank_enabled(self) -> bool:
        """Whether search runs the cross-encoder reranking stage."""
        return bool(self._data.get("rank", {}).get("enabled", False))

    @property
    def rank_model(self) -> str:
        """Cross-encoder model the reranking stage loads."""
        return str(self._data.get("rank", {}).get("model", DEFAULT_RANK_MODEL))

    @property
    def references(self) -> dict[str, str]:
        r"""Prefix -> url template for external references.

        Maps the literal prefix a reference is written with to the url it
        resolves to, `{key}` standing for the digits:

            [references]
            "PROJ-" = "https://jira.example.com/browse/PROJ-{key}"
            "#" = "https://github.com/org/repo/issues/{key}"
            "!" = "https://gitlab.example.com/org/repo/-/merge_requests/{key}"

        A prefix rather than a named pattern because a named one has to
        guess. `[A-Z]+-\d+` looks like a Jira key and also matches
        `ADR-7`, `UTF-8` and `ISO-8601`; measured on this repository it
        produced 198 matches and not one was a ticket. Declaring the
        prefixes is the difference between that and zero false positives,
        and it costs the user one line each.
        """
        raw = self._data.get("references", {})
        return {str(prefix): str(template) for prefix, template in raw.items()}

    @property
    def server_token_env(self) -> str | None:
        """Name of the variable holding the server's bearer token.

            [server]
            token_env = "WSINDEX_TOKEN"
            interval = 900

        The name, never the value — the rule connectors keep and the S3
        store keeps (ADR-7). Absent means an open server,
        which is a decision someone has to write down rather than a
        default someone can fall into: `wsindex serve` says so out loud.
        """
        value = self._data.get("server", {}).get("token_env")
        return str(value) if value else None

    @property
    def server_interval(self) -> float:
        """Seconds between automatic sync cycles; 0 disables them.

        Zero by default. A server that starts pulling remotes on its own
        the moment it boots is a surprise, and this particular surprise
        spends somebody's rate limit.
        """
        return float(self._data.get("server", {}).get("interval", 0) or 0)

    @property
    def connectors(self) -> list[ConnectorSpec]:
        """External document sources, in routing order.

            [[connectors]]
            type = "github"
            url_pattern = "https://github.com/myorg/*"
            token_env = "GITHUB_TOKEN"

        First match wins, which is why order is preserved rather than
        sorted: a person writing this file expects the specific entry
        above the catch-all to be the one that answers.

        `token_env` names an environment variable, never a token. A
        config file is committed; a credential is not — the same rule
        the S3 store keeps (ADR-7) and `wsindex sync` keeps for remotes.

        Entries missing `type` or `url_pattern` are skipped: an
        incomplete one routes nothing, and refusing to load the whole
        config over it would take the workspace down for a typo in a
        section nothing else depends on.

        Absent from `DEFAULT` deliberately, unlike every other optional
        section. `wsindex init` would otherwise write `connectors = []`,
        and TOML does not let a static array become an array of tables —
        so the one documented way to add an entry, appending a
        `[[connectors]]` block, would fail on a freshly initialized
        workspace. An empty default that blocks the only way to fill it
        is worse than no default.
        """
        found: list[ConnectorSpec] = []
        for entry in self._data.get("connectors", []):
            if not entry.get("type") or not entry.get("url_pattern"):
                continue
            token_env = entry.get("token_env")
            found.append(
                ConnectorSpec(
                    type=str(entry["type"]),
                    url_pattern=str(entry["url_pattern"]),
                    token_env=str(token_env) if token_env else None,
                )
            )
        return found

    @property
    def repos(self) -> list[Repository]:
        """Repositories to index, in search merge-order.

        Built fresh on every access, so mutating the returned list does
        nothing to the config — `add_repo` is the way in.
        """
        return [
            Repository(
                id=str(repo["id"]),
                path=str(repo["path"]),
                # `or None`: an empty string in a hand-edited file means
                # "no remote", not a url that will fail at clone time.
                remote=str(repo["remote"]) if repo.get("remote") else None,
                source=RepoSource(repo["source"]) if repo.get("source") else None,
                urls=tuple(str(url) for url in repo.get("urls", [])),
                ignore=tuple(str(pattern) for pattern in repo.get("ignore", [])),
                formats={
                    str(suffix).lower(): (str(entry["lang"]), Kind(entry["kind"]))
                    for suffix, entry in (repo.get("formats") or {}).items()
                },
            )
            for repo in self._data["repos"]
        ]

    def add_repo(self, repo: Repository) -> None:
        """Register a repository in the document (in memory; `save` is separate).

        Takes the whole `Repository` rather than its fields one by one.
        Two parallel lists of the same six things drift: `formats` was
        added to the dataclass and forgotten here, so the only way to set
        it through this class was to reach into the parsed document.

        Args:
            repo: The entry to add.

        Raises:
            ValueError: The id is already registered — ids name datasets,
                so a duplicate would silently merge two repos into one —
                or the entry breaks a rule `_validate_repo` enforces on
                load, checked here so a bad `add-repo` fails now rather
                than on the next command.
        """
        if any(existing.id == repo.id for existing in self.repos):
            raise ValueError(f"repo id already exists: {repo.id}")
        entry = _repo_entry(repo)
        self._validate_repo(entry)
        self._data["repos"].append(entry)

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
        target.write_text(self._render(), encoding="utf-8")
        return target

    def _render(self) -> str:
        """The document as TOML, with repos as `[[repos]]` sections.

        tomli_w decides that per entry by line length: a repo of just an
        id and a path fits on one line, so it writes `repos = [{...}]` —
        and a static array cannot take a sub-table. `[repos.formats]`,
        the documented way to mark up a repository, is then a syntax
        error on the very file `wsindex add-repo` just wrote.

        The same trap `connectors` fell into, and the same
        rule: whatever a person is told to hand-edit must be a shape they
        can hand-edit. So the array of tables is written here rather than
        left to a heuristic, while every value still goes through
        tomli_w — quoting a key like `".sql"` is not something to
        reimplement.
        """
        parts = [tomli_w.dumps({key: value for key, value in self._data.items() if key != "repos"})]
        for repo in self._data.get("repos", []):
            flat = {key: value for key, value in repo.items() if not isinstance(value, dict)}
            nested = {key: value for key, value in repo.items() if isinstance(value, dict)}
            parts.append("[[repos]]\n" + tomli_w.dumps(flat))
            if nested:
                # Rendered under the `repos` name so the headers come out
                # as `[repos.formats...]`, which TOML attaches to the
                # last `[[repos]]` — the entry just written above.
                parts.append(tomli_w.dumps({"repos": nested}))
        return "\n".join(parts)
