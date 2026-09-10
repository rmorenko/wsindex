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

from wsindex.config.schema import DEFAULT_RANK_MODEL, Backend, Provider, RepoSource
from wsindex.config.validate import REPO_KEYS
from wsindex.model import Kind

SCHEMA_FILE = "wsindex.schema.json"
"""Where the generated document is kept, relative to the repository root."""

SCHEMA_ID = "https://raw.githubusercontent.com/rmorenko/wsindex/main/wsindex.schema.json"
"""The address an editor is pointed at with a `#:schema` line. A url
rather than a path because a workspace config lives anywhere on disk,
and a relative path would only be right for this repository."""


def _enum(values: type[Backend] | type[Provider] | type[RepoSource] | type[Kind]) -> list[str]:
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


def build() -> dict[str, Any]:
    """The whole schema, as a plain dict ready to be written as JSON.

    Returns:
        A draft 2020-12 document describing `wsindex.toml`.
    """
    repo_properties = _repo_properties()
    unknown = REPO_KEYS - set(repo_properties)
    if unknown:
        # A key the validator accepts and this does not would make the
        # schema underline a config that works. Better to fail the
        # generator than to ship that.
        raise RuntimeError(f"REPO_KEYS holds keys this schema does not describe: {sorted(unknown)}")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "wsindex workspace configuration",
        "type": "object",
        "required": ["workspace", "embeddings"],
        "additionalProperties": False,
        "properties": {
            "workspace": {
                "type": "object",
                "required": ["name", "backend"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "description": "Workspace name."},
                    "backend": {"type": "string", "enum": _enum(Backend)},
                },
            },
            "embeddings": {
                "type": "object",
                "required": ["model", "dim", "provider"],
                "additionalProperties": False,
                "properties": {
                    "model": {"type": "string", "description": "Model id, e.g. a hub name."},
                    "dim": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Vector width; must match what the model produces.",
                    },
                    "provider": {"type": "string", "enum": _enum(Provider)},
                },
            },
            "store": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "uri": {
                        "type": "string",
                        "description": "Where the index lives; a path, or s3://bucket/prefix. "
                        "Defaults to `.wsindex/` beside this file.",
                    },
                    "metric": {"type": "string", "enum": ["cosine"], "default": "cosine"},
                },
            },
            "rank": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "default": False,
                        "description": "Re-rank with a cross-encoder; loads a second model.",
                    },
                    "model": {"type": "string", "default": DEFAULT_RANK_MODEL},
                },
            },
            "server": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
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
                },
            },
            "references": {
                "type": "object",
                "description": 'Prefix -> url template, e.g. PROJ = "https://tracker/browse/{id}".',
                "additionalProperties": {"type": "string"},
            },
            "repos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "path"],
                    # Closed on purpose, exactly as `validate_repo` is: a
                    # misspelled `ignores` that silently indexes
                    # everything is the mistake this format invites most.
                    "additionalProperties": False,
                    "properties": repo_properties,
                },
            },
            "connectors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["type", "url_pattern"],
                    "additionalProperties": False,
                    "properties": {
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
                },
            },
        },
    }


__all__ = ["SCHEMA_FILE", "SCHEMA_ID", "build"]
