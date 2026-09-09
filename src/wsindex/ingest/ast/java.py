"""Java policy: classes hold members, annotations come for free."""

from __future__ import annotations

from wsindex.ingest.ast.nested import NestedPolicy, extractor

POLICY = NestedPolicy(
    types=("class_declaration",),
    members=("method_declaration", "constructor_declaration"),
    standalone=("interface_declaration", "enum_declaration", "record_declaration"),
)
"""Annotations live inside the declaration node (its `modifiers` child),
so spans include them without asking. Javadoc comments are siblings and
stay in gap chunks — accepted debt, like JSDoc for typescript."""

spans = extractor(POLICY)
"""Classes descend into methods and constructors (`Cls.method`);
interfaces, enums and records stay whole."""
