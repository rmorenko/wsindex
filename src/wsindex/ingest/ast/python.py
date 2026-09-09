"""Python policy: decorated defs unwrap, methods get qualified symbols."""

from __future__ import annotations

from wsindex.ingest.ast.nested import NestedPolicy, extractor

POLICY = NestedPolicy(
    types=("class_definition",),
    members=("function_definition",),
    standalone=("function_definition",),
    wrappers=("decorated_definition",),
)
"""`function_definition` is both a member and standalone: the same node
is a method inside a class and a function outside one. Decorators wrap
the definition, so the chunk is the wrapper — a decorator without what it
decorates is not a passage anyone wants back."""

spans = extractor(POLICY)
"""Functions and methods with a qualified symbol (`Cls.method`); class
lines no method claimed — the header, the docstring, the attributes —
carry the class name. Oversized functions stay whole on purpose."""
