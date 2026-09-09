"""Tests for the plugin specification: what a LanguageSpec must satisfy.

Every rejection below describes a spec that would otherwise fail
silently — a language nothing can select, an extractor nothing can call,
a grammar nothing reads. The point of `register` being strict is that the
plugin author learns at registration rather than wondering later why
their files are not in the index.

Validation runs against a throwaway `LanguageRegistry` per test — a case
that mutated the process-wide `REGISTRY` would leak into every later test
in the session. The two end-to-end cases at the bottom do have to use the
real one, since the point is that the walker and the chunker see it; they
restore it afterwards.
"""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from wsindex.ingest.ast.core import Span, def_span
from wsindex.ingest.chunker import chunk_file
from wsindex.ingest.languages import (
    BUILTIN_LANGUAGES,
    REGISTRY,
    GrammarSpec,
    LanguageRegistry,
    LanguageSpec,
)
from wsindex.ingest.walker import inspect_file
from wsindex.model import Kind


def fake_spans(root: object, lines: list[str], covered: list[bool]) -> list[Span]:
    """A stand-in extractor; never called by the validation tests."""
    return []


GRAMMAR = GrammarSpec(module="tree_sitter_nonexistent", getter="language")


@pytest.fixture
def registry() -> LanguageRegistry:
    return LanguageRegistry()


# --- what a valid spec looks like ----------------------------------------


def test_a_doc_language_needs_only_a_suffix(registry: LanguageRegistry) -> None:
    registry.register(LanguageSpec(name="asciidoc", kind=Kind.DOC, suffixes=(".adoc",)))
    assert registry.match(Path("guide.adoc")) is not None
    assert registry.get("asciidoc") is not None


def test_a_code_language_carries_grammar_and_extractor(registry: LanguageRegistry) -> None:
    spec = LanguageSpec(
        name="go", kind=Kind.CODE, suffixes=(".go",), grammar=GRAMMAR, spans=fake_spans
    )
    registry.register(spec)
    assert spec.is_ast
    assert registry.extractor("go") is fake_spans


def test_a_language_matched_by_exact_filename(registry: LanguageRegistry) -> None:
    registry.register(LanguageSpec(name="makefile", kind=Kind.CONFIG, filenames=("Makefile",)))
    matched = registry.match(Path("sub/dir/Makefile"))
    assert matched is not None
    assert matched.name == "makefile"


def test_suffix_wins_over_filename(registry: LanguageRegistry) -> None:
    # A file called `Dockerfile.py` is Python, not a Dockerfile: the
    # suffix is the more specific signal, and the walker relied on that
    # order before the table moved in here.
    registry.register(LanguageSpec(name="py", kind=Kind.CODE, suffixes=(".py",)))
    registry.register(LanguageSpec(name="docker", kind=Kind.CONFIG, filenames=("Dockerfile",)))
    matched = registry.match(Path("Dockerfile.py"))
    assert matched is not None
    assert matched.name == "py"


def test_unknown_file_matches_nothing(registry: LanguageRegistry) -> None:
    registry.register(LanguageSpec(name="py", kind=Kind.CODE, suffixes=(".py",)))
    assert registry.match(Path("image.png")) is None
    assert registry.match(Path("LICENSE")) is None


def test_suffix_match_is_case_insensitive(registry: LanguageRegistry) -> None:
    registry.register(LanguageSpec(name="py", kind=Kind.CODE, suffixes=(".py",)))
    assert registry.match(Path("Module.PY")) is not None


# --- what register refuses, and why --------------------------------------


def test_a_spec_matching_no_file_is_rejected(registry: LanguageRegistry) -> None:
    # Neither suffixes nor filenames: nothing could ever select it.
    with pytest.raises(ValueError, match="matches no files"):
        registry.register(LanguageSpec(name="ghost", kind=Kind.CODE))


def test_a_nameless_spec_is_rejected(registry: LanguageRegistry) -> None:
    with pytest.raises(ValueError, match="needs a name"):
        registry.register(LanguageSpec(name="", kind=Kind.DOC, suffixes=(".x",)))


def test_a_suffix_without_a_dot_is_rejected(registry: LanguageRegistry) -> None:
    # `match` compares against `Path.suffix`, which always carries the
    # dot — "py" would silently never match anything.
    with pytest.raises(ValueError, match="must be lowercase and start with"):
        registry.register(LanguageSpec(name="py", kind=Kind.CODE, suffixes=("py",)))


def test_an_uppercase_suffix_is_rejected(registry: LanguageRegistry) -> None:
    # `match` lowercases what it looks up, so ".PY" would never be found.
    with pytest.raises(ValueError, match="must be lowercase and start with"):
        registry.register(LanguageSpec(name="py", kind=Kind.CODE, suffixes=(".PY",)))


def test_a_grammar_without_an_extractor_is_rejected(registry: LanguageRegistry) -> None:
    with pytest.raises(ValueError, match="go together"):
        registry.register(
            LanguageSpec(name="go", kind=Kind.CODE, suffixes=(".go",), grammar=GRAMMAR)
        )


def test_an_extractor_without_a_grammar_is_rejected(registry: LanguageRegistry) -> None:
    with pytest.raises(ValueError, match="go together"):
        registry.register(
            LanguageSpec(name="go", kind=Kind.CODE, suffixes=(".go",), spans=fake_spans)
        )


def test_a_doc_language_with_a_grammar_is_rejected(registry: LanguageRegistry) -> None:
    # `chunk_file` sends every DOC file to the text chunker, so the
    # grammar would be dead weight the author never learns about.
    with pytest.raises(ValueError, match="chunked as text"):
        registry.register(
            LanguageSpec(
                name="md", kind=Kind.DOC, suffixes=(".md",), grammar=GRAMMAR, spans=fake_spans
            )
        )


def test_a_duplicate_name_is_rejected(registry: LanguageRegistry) -> None:
    registry.register(LanguageSpec(name="py", kind=Kind.CODE, suffixes=(".py",)))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(LanguageSpec(name="py", kind=Kind.CODE, suffixes=(".py3",)))


def test_a_stolen_suffix_is_rejected(registry: LanguageRegistry) -> None:
    # Two languages claiming `.py` would make the winner depend on
    # registration order, i.e. on which plugin happened to load first.
    registry.register(LanguageSpec(name="py", kind=Kind.CODE, suffixes=(".py",)))
    with pytest.raises(ValueError, match="already claimed by 'py'"):
        registry.register(LanguageSpec(name="py2", kind=Kind.CODE, suffixes=(".py",)))


def test_a_stolen_filename_is_rejected(registry: LanguageRegistry) -> None:
    registry.register(LanguageSpec(name="docker", kind=Kind.CONFIG, filenames=("Dockerfile",)))
    with pytest.raises(ValueError, match="already claimed by 'docker'"):
        registry.register(LanguageSpec(name="other", kind=Kind.CONFIG, filenames=("Dockerfile",)))


def test_a_rejected_spec_leaves_the_registry_untouched(registry: LanguageRegistry) -> None:
    registry.register(LanguageSpec(name="py", kind=Kind.CODE, suffixes=(".py",)))
    with pytest.raises(ValueError):
        registry.register(LanguageSpec(name="broken", kind=Kind.CODE, suffixes=("nodot",)))
    assert [spec.name for spec in registry.specs] == ["py"]


# --- grammars stay optional ----------------------------------------------


def test_an_uninstalled_grammar_yields_no_parser(registry: LanguageRegistry) -> None:
    # The `ast` extra is optional, so naming a grammar is a statement of
    # intent, not a runtime dependency. The chunker falls back to text.
    registry.register(
        LanguageSpec(
            name="go", kind=Kind.CODE, suffixes=(".go",), grammar=GRAMMAR, spans=fake_spans
        )
    )
    assert registry.parser("go") is None
    # The extractor is still there — it simply has nothing to read.
    assert registry.extractor("go") is fake_spans


def test_a_language_without_a_grammar_has_no_parser(registry: LanguageRegistry) -> None:
    registry.register(LanguageSpec(name="md", kind=Kind.DOC, suffixes=(".md",)))
    assert registry.parser("md") is None
    assert registry.extractor("md") is None


def test_unknown_language_has_neither(registry: LanguageRegistry) -> None:
    assert registry.parser("klingon") is None
    assert registry.extractor("klingon") is None


def test_registering_after_a_lookup_is_picked_up(registry: LanguageRegistry) -> None:
    # Parsers are built lazily and dropped on register, which is what
    # will let step 24 load plugins after this module was imported.
    registry.register(LanguageSpec(name="md", kind=Kind.DOC, suffixes=(".md",)))
    assert registry.parser("md") is None  # forces the parser table to build
    registry.register(
        LanguageSpec(
            name="go", kind=Kind.CODE, suffixes=(".go",), grammar=GRAMMAR, spans=fake_spans
        )
    )
    assert registry.get("go") is not None
    assert registry.match(Path("main.go")) is not None


# --- the built-ins, through the same door --------------------------------


def test_builtins_are_registered() -> None:
    names = {spec.name for spec in REGISTRY.specs}
    assert {"python", "rust", "typescript", "java"} <= names
    assert {"toml", "yaml", "json", "dockerfile"} <= names
    assert {"markdown", "rst", "text"} <= names


def test_builtins_all_pass_their_own_validation() -> None:
    # The specification has to be satisfiable by what ships with it: if
    # a rule rejected a built-in, the rule would be wrong.
    fresh = LanguageRegistry()
    for spec in BUILTIN_LANGUAGES:
        fresh.register(spec)
    assert len(fresh.specs) == len(BUILTIN_LANGUAGES)


@pytest.mark.parametrize(
    ("filename", "lang", "kind"),
    [
        ("m.py", "python", Kind.CODE),
        ("m.rs", "rust", Kind.CODE),
        ("m.ts", "typescript", Kind.CODE),
        ("M.java", "java", Kind.CODE),
        ("pyproject.toml", "toml", Kind.CONFIG),
        ("ci.yml", "yaml", Kind.CONFIG),
        ("ci.yaml", "yaml", Kind.CONFIG),
        ("data.json", "json", Kind.CONFIG),
        ("Dockerfile", "dockerfile", Kind.CONFIG),
        ("README.md", "markdown", Kind.DOC),
        ("doc.rst", "rst", Kind.DOC),
        ("notes.txt", "text", Kind.DOC),
    ],
)
def test_builtin_files_are_matched(filename: str, lang: str, kind: Kind) -> None:
    spec = REGISTRY.match(Path(filename))
    assert spec is not None
    assert (spec.name, spec.kind) == (lang, kind)


def test_def_span_is_part_of_the_published_helper_set() -> None:
    # A plugin's extractor is expected to use these rather than mark
    # coverage by hand; this pins them as public API of the spec.
    assert callable(def_span)
    assert Span(start_line=1, end_line=1, symbol=None, node_type=None).start_line == 1


# --- the claim the specification makes: register once, every stage sees it


@pytest.fixture
def register_globally() -> Iterator[Callable[[LanguageSpec], None]]:
    """Add a language to the process registry, then take it back out.

    Reaches into `_specs` to restore, which no production code does and
    no plugin should: the registry is append-only by design, because a
    language disappearing mid-run would mean chunks whose `lang` nothing
    can explain. A test still has to undo itself, and swapping the module
    attribute would not work — `walker` and `chunker` hold a direct
    reference to this object, which is exactly why registering works at
    all.
    """
    saved = dict(REGISTRY._specs)
    yield REGISTRY.register
    REGISTRY._specs.clear()
    REGISTRY._specs.update(saved)
    REGISTRY._parsers = None


def test_a_registered_language_is_walked_and_chunked(
    tmp_path: Path, register_globally: Callable[[LanguageSpec], None]
) -> None:
    # The whole claim of the specification in one test: one object, and
    # both the walker (which files count) and the chunker (how they are
    # split) start honouring it. Before this step the walker's table was
    # a module constant and no plugin could reach it.
    # An invented suffix, not a real language: `.lua` used to stand in
    # here and stopped being unclaimed the moment the example plugin was
    # installed. A registry test must not depend on the environment.
    register_globally(LanguageSpec(name="invented", kind=Kind.CODE, suffixes=(".invented",)))
    (tmp_path / "main.invented").write_text("greet()\n")

    walked = inspect_file(tmp_path, "main.invented")
    assert walked is not None
    assert (walked.rel_path, walked.lang, walked.kind) == ("main.invented", "invented", Kind.CODE)

    chunks = chunk_file(
        (tmp_path / "main.invented").read_text(),
        repo="r",
        path="main.invented",
        lang="invented",
        kind=Kind.CODE,
    )
    # No grammar declared, so it lands on the text chunker — which is the
    # documented fallback, not a failure.
    assert chunks
    assert all(chunk.lang == "invented" for chunk in chunks)


def test_an_unregistered_suffix_is_still_skipped(tmp_path: Path) -> None:
    # The negative half: the walker did not simply start accepting
    # everything when its table moved into the registry.
    (tmp_path / "main.invented").write_text("x = 1\n")
    assert REGISTRY.match(Path("main.invented")) is None
    assert inspect_file(tmp_path, "main.invented") is None
