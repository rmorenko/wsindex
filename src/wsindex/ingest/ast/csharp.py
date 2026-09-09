"""C# policy: namespaces recurse, classes and structs hold their members.

Everything nests under a namespace — file-scoped (`namespace App;`) or
braced — so the walk has to descend or a whole file yields one chunk.
Records and enums are claimed whole: a record's members are its
constructor parameters, and splitting an enum by member would produce
chunks too small to mean anything.
"""

from wsindex.ingest.ast.nested import NestedPolicy, extractor

POLICY = NestedPolicy(
    containers=("namespace_declaration", "file_scoped_namespace_declaration"),
    types=("class_declaration", "struct_declaration", "interface_declaration"),
    members=(
        "method_declaration",
        "constructor_declaration",
        "property_declaration",
        "operator_declaration",
    ),
    standalone=("record_declaration", "enum_declaration", "delegate_declaration"),
)

spans = extractor(POLICY)
