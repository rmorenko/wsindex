"""Kotlin policy: classes and objects hold members, top-level funs stand alone.

Kotlin keeps a type's members in a `class_body` child rather than a
`body` field, which is what `NestedPolicy.body_types` exists for. There
is no namespace to recurse into: `package` is a header, not a container,
so a file's declarations are already at the top level.

An `interface` parses as a `class_declaration` here, so it needs no row
of its own — it simply arrives with the classes.
"""

from wsindex.ingest.ast.nested import NestedPolicy, extractor

POLICY = NestedPolicy(
    types=("class_declaration", "object_declaration"),
    members=("function_declaration", "property_declaration", "secondary_constructor"),
    standalone=("function_declaration",),
    body_types=("class_body", "enum_class_body"),
)

spans = extractor(POLICY)
