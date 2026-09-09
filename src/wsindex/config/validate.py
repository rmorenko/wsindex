"""What a config document must satisfy before it becomes the live config.

Strict on purpose: a silent default masks a typo, and the whole workspace
would then index with the wrong model or quietly stop indexing a repo.
Every rule here describes a document that would otherwise fail later and
somewhere else.
"""

from __future__ import annotations

from typing import Any

from wsindex.config.schema import Backend, Provider, RepoSource
from wsindex.model import Kind

REQUIRED: dict[str, tuple[str, ...]] = {
    "workspace": ("name", "backend"),
    "embeddings": ("model", "dim", "provider"),
}
"""Sections and keys a config file must contain. `store`, `rank` and
`repos` are optional and default per key — they arrived after the first
configs were written, and defaulting them is what lets the format evolve
without a migrator."""

REPO_KEYS: frozenset[str] = frozenset(
    {"id", "path", "remote", "source", "urls", "ignore", "formats"}
)
"""Every key a `[[repos]]` entry may carry. Closed on purpose: a
misspelled `ignores` that silently indexed everything is the failure this
format is most likely to produce, and the only cheap way to catch it is
to refuse what we do not recognize."""


def validate(data: dict[str, Any]) -> None:
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
    for section, keys in REQUIRED.items():
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
        validate_repo(repo)


def validate_repo(repo: dict[str, Any]) -> None:
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
    unknown = sorted(set(repo) - REPO_KEYS)
    if unknown:
        raise ValueError(
            f"repo {repo.get('id', '?')!r} has unknown key(s) {', '.join(unknown)} — "
            f"a repo entry may hold: {', '.join(sorted(REPO_KEYS))}"
        )
    validate_formats(repo)
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


def validate_formats(repo: dict[str, Any]) -> None:
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
                f"{where}: needs both 'lang' and 'kind', e.g. {{ lang = \"sql\", kind = \"code\" }}"
            )
        kind = Kind(entry["kind"])
        if kind is Kind.COMMIT:
            # Commit chunks are synthesized from git history and have
            # no path on disk; a file claiming to be one would collide
            # with that corpus in every filter that mentions it.
            raise ValueError(f"{where}: 'commit' is not a file kind — use code, config or doc")
