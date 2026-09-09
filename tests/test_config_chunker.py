"""Tests for config AST chunking: TOML/YAML/JSON/Dockerfile spans and gaps.

Language tests skip themselves when the grammar is absent (base install);
the no-grammar guard test runs everywhere.
"""

import pytest

from wsindex.ingest.chunker import chunk_file
from wsindex.ingest.languages import REGISTRY
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

XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<!-- what this module builds -->
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>demo</artifactId>
  <version>1.0.0</version>
  <properties>
    <java.version>21</java.version>
  </properties>
</project>
"""

BIG_XML = """\
<project>
  <artifactId>demo</artifactId>
  <dependencies>
{deps}  </dependencies>
</project>
""".format(
    deps="".join(
        f"    <dependency>\n      <artifactId>lib{n}</artifactId>\n    </dependency>\n"
        for n in range(20)
    )
)


SAMPLES = [
    ("toml", TOML),
    ("yaml", YAML),
    ("json", JSON),
    ("dockerfile", DOCKER),
    ("xml", XML),
    ("xml", BIG_XML),
]


def _chunk(text: str, lang: str) -> list[Chunk]:
    if REGISTRY.parser(lang) is None:
        pytest.skip(f"no {lang} grammar installed")
    return chunk_file(text, repo="r", path=f"cfg.{lang}", lang=lang, kind=Kind.CONFIG)


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


def test_unregistered_lang_falls_back_to_text_chunks() -> None:
    # Runs everywhere: "ini" is never registered, so there is no parser
    # and no extractor, and the dispatcher windows the file instead.
    # This used to raise; the registry made the raise unreachable, since
    # `ast_chunks` now takes a parser rather than looking one up.
    chunks = chunk_file("key = 1\n", repo="r", path="s.ini", lang="ini", kind=Kind.CONFIG)
    assert len(chunks) == 1
    assert chunks[0].node_type is None


# --- XML: the shape the probe argued for ---------------------------------


def test_xml_children_of_the_root_are_the_unit_not_the_root() -> None:
    # The trap `probes/step17j` found: an XML file has exactly one
    # top-level element, so "top-level elements" — the rule every other
    # extractor here follows — would make each file a single chunk.
    # Measured on Maven's own pom: `<project>` is 1287 of 1306 lines.
    assert _shape(_chunk(XML, "xml")) == [
        # The prolog, the comment and the root's own opening tag: the
        # walk claims the root's children, so its tags fall to gaps.
        (1, 3, None, None),
        (4, 10, "project", "elements"),
        (11, 11, None, None),  # </project>
    ]


def test_xml_small_siblings_share_a_chunk() -> None:
    # Tomcat's web.xml has 1029 children of the root, four lines each.
    # One chunk each would be a thousand embeddings for one file.
    chunks = _chunk(XML, "xml")
    body = next(c for c in chunks if c.symbol == "project")
    assert "<modelVersion>" in body.text
    assert "<properties>" in body.text


def test_xml_an_element_too_big_for_a_chunk_is_descended_into() -> None:
    # And the other end: Maven's pom has a <dependencyManagement> of 514
    # lines, which as one chunk is mostly past what the model reads.
    symbols = [c.symbol for c in _chunk(BIG_XML, "xml")]
    assert "project/dependencies" in symbols
    assert all(c.end_line - c.start_line + 1 <= 41 for c in _chunk(BIG_XML, "xml"))


def test_xml_a_lone_element_is_named_by_its_path() -> None:
    # What makes `--symbol project/dependencies` worth typing.
    named = next(c for c in _chunk(BIG_XML, "xml") if c.symbol is not None)
    assert (named.symbol, named.node_type) == ("project/artifactId", "element")


def test_xml_self_closing_elements_are_named() -> None:
    text = '<beans>\n  <import resource="other.xml"/>\n</beans>\n'
    named = next(c for c in _chunk(text, "xml") if c.symbol is not None)
    assert named.symbol == "beans/import"


def test_xml_that_does_not_parse_still_yields_chunks() -> None:
    # The parser is error-tolerant and the gap pass is total, so a
    # half-written config is indexed rather than dropped.
    text = "<project>\n  <artifactId>demo\n</project>\n"
    chunks = _chunk(text, "xml")
    assert chunks
    assert "\n".join(c.text for c in chunks).count("artifactId") == 1


def test_xml_a_container_tag_never_becomes_a_chunk_of_its_own() -> None:
    # Found live, on Spring Petclinic's pom. Descending into
    # <dependencies> left its own two tag lines as chunks, and searching
    # for "where is the dependency version configured" returned that
    # two-line `<dependencies>` and its closing tag above every actual
    # dependency: the shortest chunk made of nothing but the container's
    # name is the strongest lexical match and the weakest answer.
    chunks = _chunk(BIG_XML, "xml")
    runs = [c for c in chunks if c.symbol == "project/dependencies"]
    assert runs[0].text.lstrip().startswith("<dependencies>")
    assert runs[-1].text.rstrip().endswith("</dependencies>")
    assert not any(c.text.strip() in ("<dependencies>", "</dependencies>") for c in chunks)
