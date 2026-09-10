"""AST chunking: the language-agnostic mechanism and the per-language policy.

`core` holds the mechanism — parse, let a language claim the parts it
cares about, turn everything left into gap chunks. Each sibling module is
one language's policy: which nodes deserve a chunk of their own and where
their symbol comes from.

Deliberately thin. Which languages exist, which grammar parses them and
which extractor reads them is not decided here — that is a `LanguageSpec`
in `wsindex.ingest.languages`, so a plugin can add one from outside the
package. This package would otherwise have to import that registry, and
the registry imports these modules to build it.

What it does export is the toolkit a `spans` extractor is written with —
the same one the built-in languages use. A plugin should reach for these
rather than mark coverage by hand: `def_span` for a definition,
`gap_spans` for the lines inside it that no nested definition claimed,
`unwrap` for grammars that hide a definition inside a decorator or an
export statement.
"""

from wsindex.ingest.ast.core import (
    HAS_TREE_SITTER,
    PARSE_ERROR,
    Span,
    ast_chunks,
    def_span,
    gap_spans,
    line_span,
    mark_covered,
    symbol_name,
    unwrap,
)

__all__ = [
    "HAS_TREE_SITTER",
    "PARSE_ERROR",
    "Span",
    "ast_chunks",
    "def_span",
    "gap_spans",
    "line_span",
    "mark_covered",
    "symbol_name",
    "unwrap",
]
