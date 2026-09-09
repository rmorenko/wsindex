"""Rust policy: prelude siblings stick to definitions, impls act as classes."""

from __future__ import annotations

from wsindex.ingest.ast.nested import NestedPolicy, extractor

POLICY = NestedPolicy(
    types=("impl_item",),
    members=("function_item",),
    standalone=("function_item", "struct_item", "enum_item", "trait_item"),
    separator="::",
    # An impl block has no `name`: what it implements *for* is its `type`.
    # A trait impl is therefore named by the implementing type, and the
    # trait itself is ignored on purpose — `Display for Greeter` is found
    # by looking for `Greeter`.
    type_name_field="type",
    preamble=("attribute_item", "line_comment"),
)
"""`#[derive(...)]` and `///` docs are siblings of the definition below
them, not children, so they are attached by widening the span backwards
— `#[test] fn it_works` is one chunk, not two."""

spans = extractor(POLICY)
"""Free functions, types and impl methods; method symbols use the native
`Type::method` separator. An impl block is handled like a class: methods
first, leftover lines (header, closing brace) carry the type name."""
