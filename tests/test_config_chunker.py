"""Tests for config AST chunking: TOML/YAML/JSON/Dockerfile spans and gaps.

Language tests skip themselves when the grammar is absent (base install);
the no-grammar guard test runs everywhere.
"""

import pytest

from wsindex.ingest.ast_chunker import CONFIG_PARSERS, chunk_config
from wsindex.model import Chunk, Kind

TOML = """\
root_key = "top-level pair"  # inline comment

# section comment
[tool.ruff]
line-length = 100
select = ["E", "F"]

[[repos]]
id = "one"

[[repos]]
id = "two"
"""

YAML = """\
# header comment
name: ci
on:
  push:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
---
second_doc: true
"""

JSON = """\
{
  "name": "demo",
  "scripts": {
    "build": "tsc",
    "test": "jest"
  },
  "keywords": ["a", "b"]
}
"""

DOCKER = """\
# build stage
FROM python:3.12 AS builder
RUN pip install uv && \\
    uv sync

FROM python:3.12-slim
COPY --from=builder /app /app
CMD ["python", "-m", "app"]
"""

SAMPLES = [("toml", TOML), ("yaml", YAML), ("json", JSON), ("dockerfile", DOCKER)]


def _chunk(text: str, lang: str) -> list[Chunk]:
    if lang not in CONFIG_PARSERS:
        pytest.skip(f"no {lang} grammar installed")
    return chunk_config(text, repo="r", path=f"cfg.{lang}", lang=lang, kind=Kind.CONFIG)


def _shape(chunks: list[Chunk]) -> list[tuple[int, int, str | None, str | None]]:
    return [(c.start_line, c.end_line, c.symbol, c.node_type) for c in chunks]


def test_toml_tables_become_chunks() -> None:
    assert _shape(_chunk(TOML, "toml")) == [
        (1, 3, None, None),  # leading pair + section comment
        (4, 6, "tool.ruff", "table"),
        (8, 9, "repos", "table_array_element"),
        (11, 12, "repos", "table_array_element"),
    ]


def test_yaml_top_level_keys_across_documents() -> None:
    assert _shape(_chunk(YAML, "yaml")) == [
        (1, 1, None, None),  # header comment
        (2, 2, "name", "block_mapping_pair"),
        (3, 5, "on", "block_mapping_pair"),
        (6, 8, "jobs", "block_mapping_pair"),
        (9, 9, None, None),  # the --- separator
        (10, 10, "second_doc", "block_mapping_pair"),
    ]


def test_json_top_level_keys_without_quotes() -> None:
    assert _shape(_chunk(JSON, "json")) == [
        (1, 1, None, None),  # opening brace
        (2, 2, "name", "pair"),
        (3, 6, "scripts", "pair"),
        (7, 7, "keywords", "pair"),
        (8, 8, None, None),  # closing brace
    ]


def test_dockerfile_stages_grouped_by_from() -> None:
    assert _shape(_chunk(DOCKER, "dockerfile")) == [
        (1, 1, None, None),  # comment before the first stage
        (2, 4, "builder", "stage"),
        (6, 8, "python:3.12-slim", "stage"),
    ]


def test_json_top_level_array_degrades_to_gap_chunks() -> None:
    chunks = _chunk('[1, 2,\n {"k": 3}]\n', "json")
    assert _shape(chunks) == [(1, 2, None, None)]


@pytest.mark.parametrize(("lang", "sample"), SAMPLES)
def test_every_nonblank_line_lands_in_exactly_one_chunk(lang: str, sample: str) -> None:
    lines = sample.splitlines()
    owners = [0] * (len(lines) + 1)
    for chunk in _chunk(sample, lang):
        for i in range(chunk.start_line, chunk.end_line + 1):
            owners[i] += 1
    for i, line in enumerate(lines, start=1):
        expected = 1 if line.strip() else owners[i]  # blank lines may be nobody's
        assert owners[i] == expected, f"{lang} line {i}: {line!r}"


@pytest.mark.parametrize(("lang", "sample"), SAMPLES)
def test_chunk_text_is_a_verbatim_slice(lang: str, sample: str) -> None:
    lines = sample.splitlines()
    for chunk in _chunk(sample, lang):
        assert chunk.text == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])


@pytest.mark.parametrize(("lang", "sample"), SAMPLES)
def test_metadata_flows_through(lang: str, sample: str) -> None:
    chunk = _chunk(sample, lang)[0]
    expected = ("r", f"cfg.{lang}", lang, Kind.CONFIG)
    assert (chunk.repo, chunk.path, chunk.lang, chunk.kind) == expected


def test_unknown_lang_is_rejected_with_hint() -> None:
    # Runs everywhere: "ini" is never registered, no parser gets touched.
    with pytest.raises(RuntimeError, match="--extra ast"):
        chunk_config("key = 1\n", repo="r", path="s.ini", lang="ini", kind=Kind.CONFIG)
