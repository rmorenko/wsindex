"""A stand-in language plugin, loaded through a real EntryPoint.

Lives beside the tests rather than in `src/` so nothing ships it, and is
importable because pytest puts the test directory on `sys.path`. That is
what lets `test_plugins.py` exercise `EntryPoint.load()` for real instead
of stubbing the one step where the mechanism actually lives.
"""

from collections.abc import Iterator

from wsindex.ingest.ast import Span, def_span, symbol_name
from wsindex.ingest.languages import GrammarSpec, LanguageSpec
from wsindex.model import Kind


def go_spans(root: object, lines: list[str], covered: list[bool]) -> list[Span]:
    """Top-level function declarations become chunks of their own."""
    out: list[Span] = []
    for child in root.named_children:  # type: ignore[attr-defined]
        if child.type == "function_declaration":
            name = symbol_name(child)
            if name is not None:
                out.append(
                    def_span(child, covered=covered, symbol=name, node_type="function_declaration")
                )
    return out


GO = LanguageSpec(
    name="fixture-go",
    kind=Kind.CODE,
    suffixes=(".fgo",),
    grammar=GrammarSpec(module="tree_sitter_nonexistent_go", getter="language"),
    spans=go_spans,
)

TEMPLATES = LanguageSpec(name="fixture-tmpl", kind=Kind.DOC, suffixes=(".ftmpl",))

PAIR = (GO, TEMPLATES)

NOT_A_SPEC = "this is not a LanguageSpec"

MIXED = (GO, 42)

BROKEN = LanguageSpec(name="fixture-broken", kind=Kind.CODE, suffixes=("nodot",))


def _angry_specs() -> Iterator[LanguageSpec]:
    """A generator that dies partway: a plugin computing specs at import."""
    yield GO
    raise RuntimeError("computing the rest of my languages failed")


ANGRY = _angry_specs()
