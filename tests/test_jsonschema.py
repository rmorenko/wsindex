"""The generated schema must be valid, checked in, and agree with the validator.

Three separate claims, and only the first is about JSON. The other two
are what make "derived from the rules" true rather than aspirational: a
schema that has drifted from `wsindex.config.validate` would underline a
config that works, which is worse than having no schema at all.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from wsindex.config.jsonschema import SCHEMA_FILE, build
from wsindex.config.schema import Backend, Provider, RepoSource
from wsindex.config.validate import REPO_KEYS, REQUIRED, validate

GOOD = """
[workspace]
name = "ws"
backend = "local"

[embeddings]
model = "sentence-transformers/all-MiniLM-L6-v2"
dim = 384
provider = "sentence-transformers"

[store]
metric = "cosine"

[rank]
enabled = true

[server]
token_env = "WSINDEX_TOKEN"
interval = 900

[references]
PROJ = "https://tracker/browse/{id}"

[[repos]]
id = "app"
path = "~/app"
ignore = ["vendor/*"]

[repos.formats.".sql"]
lang = "sql"
kind = "code"

[[connectors]]
type = "github"
url_pattern = "https://github.com/org/*"
token_env = "GITHUB_TOKEN"
"""


@pytest.fixture
def checker() -> Draft202012Validator:
    """A validator over the freshly built schema."""
    schema = build()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def errors(checker: Draft202012Validator, text: str) -> list[str]:
    """Schema complaints about one TOML document, as plain messages."""
    return [error.message for error in checker.iter_errors(tomllib.loads(text))]


def test_the_checked_in_schema_is_what_the_generator_produces() -> None:
    # `poe schema` writes it; this is what stops the file and the rules
    # it came from parting company.
    on_disk: dict[str, Any] = json.loads(Path(SCHEMA_FILE).read_text())

    assert on_disk == build(), "run `poe schema`"


def test_a_config_the_validator_accepts_the_schema_accepts_too(
    checker: Draft202012Validator,
) -> None:
    # The direction that matters most: a false complaint in somebody's
    # editor about a config that runs.
    validate(tomllib.loads(GOOD))

    assert errors(checker, GOOD) == []


@pytest.mark.parametrize(
    ("label", "broken", "expected"),
    [
        # The mistake `REPO_KEYS` exists to catch, now caught while typing.
        ("misspelled repo key", ('ignore = ["vendor/*"]', 'ignores = ["vendor/*"]'), "ignores"),
        ("unknown backend", ('backend = "local"', 'backend = "lancedb"'), "lancedb"),
        ("missing required key", ("dim = 384\n", ""), "dim"),
        ("misspelled section", ("[rank]", "[ranking]"), "ranking"),
        ("wrong type", ("dim = 384", 'dim = "384"'), "integer"),
    ],
)
def test_the_schema_catches_what_the_validator_catches(
    checker: Draft202012Validator, label: str, broken: tuple[str, str], expected: str
) -> None:
    found = errors(checker, GOOD.replace(*broken))

    assert found, f"{label} went unnoticed"
    assert any(expected in message for message in found), found


def test_the_schema_covers_every_key_the_validator_allows() -> None:
    # Derived means derived: a key added to REPO_KEYS without a
    # description here would be underlined in an editor despite being
    # valid. `build` refuses rather than shipping that, so this asserts
    # the refusal never has to fire.
    described = set(build()["properties"]["repos"]["items"]["properties"])

    assert described >= REPO_KEYS


@pytest.mark.parametrize(
    ("section", "keys"),
    sorted((section, keys) for section, keys in REQUIRED.items()),
)
def test_required_sections_are_required_in_the_schema(section: str, keys: tuple[str, ...]) -> None:
    schema = build()

    assert section in schema["required"]
    assert set(keys) <= set(schema["properties"][section]["required"])


@pytest.mark.parametrize(
    ("path", "enum"),
    [
        (("workspace", "backend"), Backend),
        (("embeddings", "provider"), Provider),
    ],
)
def test_enum_values_come_from_the_enums(path: tuple[str, str], enum: type[Backend]) -> None:
    # A new Provider member must reach the schema without anybody
    # remembering to edit it.
    section, key = path
    listed = build()["properties"][section]["properties"][key]["enum"]

    assert listed == [member.value for member in enum]


def test_repo_source_values_come_from_the_enum() -> None:
    listed = build()["properties"]["repos"]["items"]["properties"]["source"]["enum"]

    assert listed == [member.value for member in RepoSource]
