"""PHP policy: namespaces recurse, classes and traits hold their methods.

A namespace can be braced or file-scoped; only the braced form is a node
with a body to descend into, and the file-scoped form leaves its
declarations at the top level, so both are covered without a special case.

Traits and interfaces are types rather than standalone: a trait exists to
carry methods, and searching for one method of it is the normal case.
"""

from wsindex.ingest.ast.nested import NestedPolicy, extractor

POLICY = NestedPolicy(
    containers=("namespace_definition",),
    types=("class_declaration", "trait_declaration", "interface_declaration"),
    members=("method_declaration",),
    standalone=("function_definition", "enum_declaration"),
)

spans = extractor(POLICY)
