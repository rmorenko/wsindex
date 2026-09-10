"""A JSON Schema for `wsindex.toml`, derived from the rules that enforce it.

The config is already validated strictly, with messages that say what to
do — but only when a command runs. This says the same things in the
editor, while somebody is typing: which keys a section takes, which
values `backend` and `provider` accept, and that `ignores` is not a word.

Derived, not written twice. Every constant here comes from
`wsindex.config.validate` or the enums in `wsindex.config.schema`, so a
new `Provider` member or a new key in `REPO_KEYS` reaches the schema
without anybody remembering to. A hand-kept copy would drift, and a
schema that lies is worse than none: it would underline correct config.

The generated document lives at `wsindex.schema.json` in the repository
root, and a test asserts the two agree — that is what makes "derived"
true rather than aspirational.
"""

from __future__ import annotations

from typing import Any

from wsindex.config.schema import (
    DEFAULT_RANK_MODEL,
    Backend,
    LinksBackend,
    Provider,
    RepoSource,
)
from wsindex.config.validate import REPO_KEYS
from wsindex.model import Kind

SCHEMA_FILE = "wsindex.schema.json"
"""Where the generated document is kept, relative to the repository root."""

SCHEMA_ID = "https://raw.githubusercontent.com/rmorenko/wsindex/main/wsindex.schema.json"
"""The address an editor is pointed at with a `#:schema` line. A url
rather than a path because a workspace config lives anywhere on disk,
and a relative path would only be right for this repository."""


def _enum(
    values: type[Backend] | type[LinksBackend] | type[Provider] | type[RepoSource] | type[Kind],
) -> list[str]:
    """The members of a StrEnum, in declaration order, as plain strings."""
    return [member.value for member in values]


def _repo_properties() -> dict[str, Any]:
    """One entry per key `REPO_KEYS` allows, in the order it reads best."""
    return {
        "id": {"type": "string", "description": "Dataset name; unique within the workspace."},
        "path": {"type": "string", "description": "Working copy on this machine."},
        "remote": {
            "type": "string",
            "description": "Clone url; `wsindex sync` keeps `path` current from it.",
        },
        "source": {
            "type": "string",
            "enum": _enum(RepoSource),
            "description": "`connector` makes this a snapshot repo built from `urls`.",
        },
        "urls": {
            "type": "array",
            "items": {"type": "string"},
            "description": 'Documents to materialize; needs `source = "connector"`.',
        },
        "ignore": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Path globs this repo excludes; `*` crosses directories.",
        },
        "formats": {
            "type": "object",
            "description": "Suffix (with the dot) -> language and kind, for this repo only.",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "lang": {"type": "string"},
                    "kind": {"type": "string", "enum": _enum(Kind)},
                },
                "required": ["lang", "kind"],
                "additionalProperties": False,
            },
        },
    }


def _section(properties: dict[str, Any], *, required: list[str] | None = None) -> dict[str, Any]:
    """One closed table of the config document.

    Closed everywhere, exactly as `validate` is: an unrecognized key is
    the mistake this format invites most, and refusing it is the only
    cheap way to catch it.
    """
    section: dict[str, Any] = {"type": "object"}
    if required is not None:
        section["required"] = required
    # After `required`, before `properties`: JSON object order carries no
    # meaning to a validator, but the generated file is checked in and a
    # refactor that reshuffles it produces a diff that says nothing.
    section["additionalProperties"] = False
    section["properties"] = properties
    return section


def _workspace() -> dict[str, Any]:
    """`[workspace]`: what this index is called and what stores it."""
    return _section(
        {
            "name": {"type": "string", "description": "Workspace name."},
            "backend": {"type": "string", "enum": _enum(Backend)},
        },
        required=["name", "backend"],
    )


def _embeddings() -> dict[str, Any]:
    """`[embeddings]`: the model, and the width the store is built for."""
    return _section(
        {
            "model": {"type": "string", "description": "Model id, e.g. a hub name."},
            "dim": {
                "type": "integer",
                "minimum": 1,
                "description": "Vector width; must match what the model produces.",
            },
            "provider": {"type": "string", "enum": _enum(Provider)},
        },
        required=["model", "dim", "provider"],
    )


def _store() -> dict[str, Any]:
    """`[store]`: where the index lives, and how similarity is measured."""
    return _section(
        {
            "uri": {
                "type": "string",
                "description": "Where the index lives; a path, or s3://bucket/prefix. "
                "Defaults to `.wsindex/` beside this file.",
            },
            "metric": {"type": "string", "enum": ["cosine"], "default": "cosine"},
        }
    )


def _rank() -> dict[str, Any]:
    """`[rank]`: the cross-encoder, off by default."""
    return _section(
        {
            "enabled": {
                "type": "boolean",
                "default": False,
                "description": "Re-rank with a cross-encoder; loads a second model.",
            },
            "model": {"type": "string", "default": DEFAULT_RANK_MODEL},
        }
    )


def _links() -> dict[str, Any]:
    """`[links]`: which store answers `refs`, `why` and the drift report."""
    return _section(
        {
            "backend": {
                "type": "string",
                "enum": _enum(LinksBackend),
                "default": LinksBackend.SQLITE.value,
                "description": "sqlite needs no service; postgres is for a shared index.",
            },
            "dsn_env": {
                "type": "string",
                "description": "Name of the variable holding the Postgres connection "
                "string — the name, never the string.",
            },
        }
    )


def _server() -> dict[str, Any]:
    """`[server]`: the token's variable name, and the sync interval."""
    return _section(
        {
            "token_env": {
                "type": "string",
                "description": "Name of the variable holding the bearer token — "
                "the name, never the token.",
            },
            "interval": {
                "type": "number",
                "minimum": 0,
                "default": 0,
                "description": "Seconds between automatic syncs; 0 turns them off.",
            },
        }
    )


def _connectors() -> dict[str, Any]:
    """`[[connectors]]`: which fetcher claims which urls."""
    return {
        "type": "array",
        "items": _section(
            {
                "type": {"type": "string", "description": "Connector name, e.g. `github`."},
                "url_pattern": {
                    "type": "string",
                    "description": "Glob the url must match; `*` and `?` as in a shell.",
                },
                "token_env": {
                    "type": "string",
                    "description": "Name of the variable holding this source's token.",
                },
            },
            required=["type", "url_pattern"],
        ),
    }


def build() -> dict[str, Any]:
    """The whole schema, as a plain dict ready to be written as JSON.

    Returns:
        A draft 2020-12 document describing `wsindex.toml`.

    Raises:
        RuntimeError: `REPO_KEYS` holds a key no section here describes.
            A key the validator accepts and this does not would make the
            schema underline a config that works, and that is worse than
            having no schema — better to fail the generator.
    """
    repo_properties = _repo_properties()
    unknown = REPO_KEYS - set(repo_properties)
    if unknown:
        raise RuntimeError(f"REPO_KEYS holds keys this schema does not describe: {sorted(unknown)}")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "wsindex workspace configuration",
        "type": "object",
        "required": ["workspace", "embeddings"],
        "additionalProperties": False,
        "properties": {
            "workspace": _workspace(),
            "embeddings": _embeddings(),
            "store": _store(),
            "links": _links(),
            "rank": _rank(),
            "server": _server(),
            "references": {
                "type": "object",
                "description": 'Prefix -> url template, e.g. PROJ = "https://tracker/browse/{id}".',
                "additionalProperties": {"type": "string"},
            },
            "repos": {
                "type": "array",
                "items": _section(repo_properties, required=["id", "path"]),
            },
            "connectors": _connectors(),
        },
    }


__all__ = ["SCHEMA_FILE", "SCHEMA_ID", "build"]
