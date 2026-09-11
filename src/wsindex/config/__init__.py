"""Workspace configuration: the `wsindex.toml` file, as one live object.

`Config` keeps the parsed document as a dict and exposes one read-only
property per field, so the file format has a single definition and every
reader still goes through a typed accessor.

A singleton class: `__new__` caches the one instance, so `Config()`
anywhere in the process returns the same object and no subsystem needs
the config threaded through its constructor. Nothing on disk is not an
error — it means `DEFAULT` and `is_default` set; saying so out loud
belongs to whoever has a user to talk to.

What a document is made of lives in `.schema`, what makes one valid in
`.validate`, how one is written back out in `.document`.

Not a dataclass, deliberately: a dataclass is a value object, one
instance per distinct value; a singleton is an identity object. `reset()`
drops the instance, which is what test isolation needs.
"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any, ClassVar

from wsindex.config.document import render, repo_entry
from wsindex.config.schema import (
    DEFAULT_RANK_MODEL,
    Backend,
    LinksBackend,
    Provider,
    Repository,
    RepoSource,
)
from wsindex.config.validate import validate, validate_repo
from wsindex.connectors import ConnectorSpec
from wsindex.model import Kind
from wsindex.paths import ConfigLocation, Mode, find_config, resolve_index_dir


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
            validate(data)
            data.setdefault("repos", [])
            self._data = data
        self._location = location

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

    def _setting(self, section: str, key: str, default: Any = None) -> Any:
        """One optional key of one optional section.

        The eight optional settings were each written as
        `self._data.get("store", {}).get("uri")` — a two-step reach whose
        every repetition is a chance to forget the second default. The
        raw document stays the source of truth: a section wsindex does
        not know about survives a `save`, which a typed model of the file
        would quietly drop.
        """
        return self._data.get(section, {}).get(key, default)

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
    def query_prefix(self) -> str:
        """What this model wants in front of a question and not a passage.

        Empty for the default model and for every symmetric one, which is
        why it is optional. Asymmetric models document their own and lose
        real accuracy without it — see `SentenceTransformerEmbedder`.
        """
        value = self._setting("embeddings", "query_prefix")
        return str(value) if value is not None else ""

    @property
    def max_seq(self) -> int | None:
        """Cap on the model's input window, or None for whatever it declares.

        A memory setting, not a quality one: a model advertising 8192
        tokens allocates for 8192, and one of them asked for a 96 GiB
        buffer to encode chunks of a few hundred characters.
        """
        value = self._setting("embeddings", "max_seq")
        return int(value) if value is not None else None

    @property
    def trust_remote_code(self) -> bool:
        """Whether this model may run its own code from the hub.

        Off unless a workspace says otherwise, and that is a deliberate
        piece of friction rather than an oversight: executing code
        downloaded from a model host sits badly beside a tool whose
        premise is that nothing leaves the machine. Some code-specialized
        models need it, so it is reachable — by a line somebody typed.
        """
        return bool(self._setting("embeddings", "trust_remote_code") or False)

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
        uri = self._setting("store", "uri")
        return str(uri) if uri is not None else str(self.index_dir)

    @property
    def metric(self) -> str:
        """Similarity metric datasets are created with."""
        return str(self._setting("store", "metric", "cosine"))

    @property
    def rank_enabled(self) -> bool:
        """Whether search runs the cross-encoder reranking stage."""
        return bool(self._setting("rank", "enabled", False))

    @property
    def rank_model(self) -> str:
        """Cross-encoder model the reranking stage loads."""
        return str(self._setting("rank", "model", DEFAULT_RANK_MODEL))

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
    def stats_enabled(self) -> bool:
        """Whether searches are written to the local log.

            [stats]
            enabled = false

        On by default, because the quality loop it feeds needs data and
        the data is already on this machine — it sits in the same 0700
        directory as the index, which holds the source code itself. Off
        is one line, and `wsindex stats --forget` empties it. What makes
        the default defensible is that nothing leaves: strictly local is
        a rule here, with a test that a search opens no sockets.
        """
        return bool(self._setting("stats", "enabled", True))

    @property
    def max_commits(self) -> int | None:
        """How far back a full pass indexes history, or None for the default.

        The one constant in the ingest path whose right value is a
        property of the repository rather than of the machine, which is
        why it is the one that became configurable: "how much history
        should be searchable here" is a question somebody can answer
        about their own project, unlike "how many blame processes to
        run". Measured cost, so it can be chosen rather than guessed:
        about 1.4 s and 2.5 MB per thousand commits.

        None rather than `MAX_COMMITS`, so that this module does not
        import `wsindex.ingest` to learn a number. Config sits under
        everything and is imported by every command; reaching up into the
        ingest layer for a default would put 30 ms of tree-sitter and
        model imports behind `wsindex --help`. The default stays where it
        is documented, and `read_commits` applies it.
        """
        setting = self._setting("index", "max_commits")
        return int(setting) if setting is not None else None

    @property
    def links_backend(self) -> LinksBackend:
        """Where the link store lives: `sqlite` (default) or `postgres`.

            [links]
            backend = "postgres"
            dsn_env = "WSINDEX_LINKS_DSN"

        SQLite is the default and needs no service; it is what tests run
        on and what an offline machine gets. Postgres is for a workspace
        whose index is shared — links are workspace data, derived
        entirely from content, so leaving them on one machine while
        `[store] uri` puts the vectors in S3 was an asymmetry.
        """
        return LinksBackend(self._setting("links", "backend", LinksBackend.SQLITE.value))

    @property
    def links_dsn_env(self) -> str | None:
        """Name of the variable holding the Postgres connection string.

        The name, never the value — a DSN carries a password, and this
        is the rule `token_env`, the connectors and the S3 store all
        keep (ADR-7).
        """
        value = self._setting("links", "dsn_env")
        return str(value) if value else None

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
        value = self._setting("server", "token_env")
        return str(value) if value else None

    @property
    def server_interval(self) -> float:
        """Seconds between automatic sync cycles; 0 disables them.

        Zero by default. A server that starts pulling remotes on its own
        the moment it boots is a surprise, and this particular surprise
        spends somebody's rate limit.
        """
        return float(self._setting("server", "interval", 0) or 0)

    @property
    def connectors(self) -> list[ConnectorSpec]:
        """External document sources, in routing order.

            [[connectors]]
            type = "github"
            url_pattern = "https://github.com/myorg/*"
            token_env = "GITHUB_TOKEN"

        First match wins, so order is preserved rather than sorted: a
        person writing this file expects the specific entry above the
        catch-all to answer. `token_env` names a variable, never a token.

        An entry missing `type` or `url_pattern` is skipped rather than
        fatal: it routes nothing, and a typo here must not take the
        workspace down.

        Absent from `DEFAULT` deliberately. `wsindex init` would
        otherwise write `connectors = []`, and TOML does not let a static
        array become an array of tables — so the one documented way to
        add an entry would fail on a fresh workspace.
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
        entry = repo_entry(repo)
        validate_repo(entry)
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
        target.write_text(render(self._data), encoding="utf-8")
        return target


__all__ = [
    "DEFAULT_RANK_MODEL",
    "Backend",
    "Config",
    "LinksBackend",
    "Provider",
    "RepoSource",
    "Repository",
]
