"""The vocabulary of a workspace: what a `wsindex.toml` can say.

The types a config document is made of, with no reading, writing or
validating in sight — those are `wsindex.config`, `.document` and
`.validate`. Kept apart because they change for different reasons: a new
backend is a change here, a new rule about repos is a change there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import blake2b

from wsindex.model import Kind

DEFAULT_RANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class Backend(StrEnum):
    """Vector store selector; StrEnum so the value round-trips through TOML as-is.

    A single member since ADR-7 removed Tensorus — kept as an enum so a
    future backend is a data change, not an API change.
    """

    LOCAL = "local"


class LinksBackend(StrEnum):
    """Where the links live.

    SQLite is the default and needs no service — it is what an offline
    machine and every test get. Postgres is for a workspace whose index
    is shared: links are workspace data, derived entirely from content,
    so leaving them on one machine while `[store] uri` puts the vectors
    in S3 was an asymmetry rather than a design (see ADR-11).
    """

    SQLITE = "sqlite"
    POSTGRES = "postgres"


class Provider(StrEnum):
    """Where vectors come from: a fake for tests, a local model, or a
    hosted one.

    `REMOTE` is the only member that sends your code anywhere, which is
    why it has to be typed out in a config file before it can happen.
    See `wsindex.embed.remote` for what it buys and what it costs."""

    FAKE = "fake"
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    REMOTE = "remote"


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
