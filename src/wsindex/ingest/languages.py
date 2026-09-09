"""What a language is, to wsindex: the plugin specification and its registry.

First step of Этап 9. Until now a language was spread across four
unrelated tables — two parser tables, two extractor tables — plus a fifth
in the walker that nothing else could see. Adding one meant editing three
files and knowing which; adding one *from outside the package* was simply
impossible, because the walker's suffix table was a module constant.

A `LanguageSpec` is that scattered knowledge as one value: how to
recognize the file, what to call the language, and (for code and configs)
which grammar parses it and which extractor turns its tree into spans.
Register a spec and every stage picks it up — the walker starts selecting
those files, the chunker starts routing them.

That is the whole contract a plugin has to satisfy. The loader that finds
plugins through entry points is step 24; this module only defines what it
will hand over, and validates it. `register` is deliberately strict and
raises: a malformed spec is a bug in the plugin, and the loader is the
right place to decide that one bad plugin should be a warning rather than
a dead workspace.

Grammars stay optional in the way they already were. A spec may name a
grammar module that is not installed (the `ast` extra is optional); the
registry then has no parser for it and the chunker falls back to plain
text windows. Declaring the grammar is a statement of intent, not a
runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from wsindex.ingest.ast import (
    c,
    configs,
    cpp,
    csharp,
    go,
    html,
    java,
    kotlin,
    php,
    python,
    ruby,
    rust,
    sfc,
    typescript,
)
from wsindex.ingest.ast.core import HAS_TREE_SITTER, SpanExtractor
from wsindex.ingest.ast.sfc import Section, SectionSplitter
from wsindex.model import Kind

if TYPE_CHECKING:
    from tree_sitter import Parser


@dataclass(frozen=True, kw_only=True)
class GrammarSpec:
    """Where a tree-sitter grammar comes from.

    Named indirectly, by module and function, rather than as an imported
    `Language`: the module may not be installed, and a spec must be
    declarable without importing it (see the module docstring).

    Attributes:
        module: Importable module name, e.g. `"tree_sitter_python"`.
        getter: Zero-argument attribute of that module returning the
            grammar pointer, e.g. `"language"`. Grammars that ship
            several languages use a specific one, like
            `"language_typescript"`.
    """

    module: str
    getter: str


@dataclass(frozen=True, kw_only=True)
class LanguageSpec:
    """Everything wsindex needs in order to index one language.

    Attributes:
        name: The `lang` value stored on every chunk and matched by
            `--lang`. Unique across the registry.
        kind: Which chunker the files are routed to. CODE and CONFIG go
            through the AST path when a grammar is available; DOC always
            goes through the text chunker.
        suffixes: File suffixes that identify the language, lowercase and
            dot-prefixed (`".py"`).
        filenames: Exact file names, for languages whose files carry no
            useful suffix (`"Dockerfile"`).
        grammar: Grammar that parses the language, or None for a language
            chunked as text.
        spans: Extractor turning that grammar's trees into spans, or None.
        sections: Splitter for a *container* language — one file holding
            several languages, like a `.vue` component. Given the
            container's tree it returns the parts, and the chunker
            recurses into each as its own language. Mutually exclusive
            with `spans`: a file is either chunked or split, not both.
    """

    name: str
    kind: Kind
    suffixes: tuple[str, ...] = ()
    filenames: tuple[str, ...] = ()
    grammar: GrammarSpec | None = None
    spans: SpanExtractor | None = None
    sections: SectionSplitter | None = None

    @property
    def is_ast(self) -> bool:
        """True when this language declares an AST chunking path."""
        return self.grammar is not None and self.spans is not None

    @property
    def is_container(self) -> bool:
        """True when this language is split into other languages."""
        return self.grammar is not None and self.sections is not None


class LanguageRegistry:
    """The languages this process knows, and the tables derived from them.

    Parsers are built lazily and cached, then dropped whenever a spec is
    registered. That ordering is what lets step 24 load plugins after
    this module is imported: nothing has been computed from the specs
    until something asks.

    Plugins are loaded on the same terms, and for a sharper reason than
    tidiness. A plugin has to `import wsindex...` at module level to
    subclass or build what it registers. If the import of wsindex ended
    by loading plugins, then a program that imported the *plugin* first
    would re-enter it while it was still half-executed, and the entry
    point would resolve to nothing — a silently disabled plugin, which
    is the exact failure the loader exists to prevent. Deferring to
    first use breaks the cycle: by the time anything asks this registry
    a question, every module involved has finished importing.
    """

    def __init__(self, *, plugins: bool = False) -> None:
        """Build a registry.

        Args:
            plugins: Load installed language plugins on first use. True
                for the process-wide `REGISTRY` and false everywhere
                else — a throwaway registry in a test must contain only
                what the test put in it.
        """
        self._specs: dict[str, LanguageSpec] = {}
        self._parsers: dict[str, Parser] | None = None
        self._pending_plugins = plugins

    def _ensure_plugins(self) -> None:
        """Load installed plugins once, before the first question."""
        if not self._pending_plugins:
            return
        # Cleared first: `load_plugins` calls `register`, and a plugin
        # importing wsindex must not send us back around this loop.
        self._pending_plugins = False
        from wsindex.ingest.plugins import load_plugins

        load_plugins(self)

    def register(self, spec: LanguageSpec) -> None:
        """Add a language, rejecting anything that could not work.

        Strict on purpose. Every rule below describes a spec that would
        otherwise fail silently — a language nothing can select, an
        extractor nothing can call — and silence here surfaces much later
        as "why is my file not indexed?".

        Args:
            spec: The language to add.

        Raises:
            ValueError: The spec is malformed, or it collides with a
                language already registered.
        """
        if not spec.name:
            raise ValueError("language spec needs a name")
        if not spec.suffixes and not spec.filenames:
            raise ValueError(f"{spec.name!r} matches no files: give it suffixes or filenames")
        for suffix in spec.suffixes:
            if not suffix.startswith(".") or suffix != suffix.lower():
                raise ValueError(
                    f"{spec.name!r}: suffix {suffix!r} must be lowercase and start with '.'"
                )
        if spec.spans is not None and spec.sections is not None:
            # A container's parts are chunked by their own languages; an
            # extractor on top of that would claim the same lines twice.
            raise ValueError(
                f"{spec.name!r}: give `spans` or `sections`, not both — a container "
                f"file is split, and its parts are chunked by their own languages"
            )
        if spec.sections is not None and spec.grammar is None:
            raise ValueError(f"{spec.name!r}: `sections` needs a `grammar` to split with")
        if spec.sections is None and (spec.grammar is None) != (spec.spans is None):
            # One without the other can never run: an extractor needs a
            # tree to read, and a tree nobody reads produces no chunks.
            raise ValueError(
                f"{spec.name!r}: `grammar` and `spans` go together — give both or neither"
            )
        if spec.kind is Kind.DOC and spec.is_ast:
            # `chunk_file` sends every DOC file to the text chunker, so a
            # grammar here would be quietly ignored. Refusing beats that.
            raise ValueError(
                f"{spec.name!r}: DOC languages are chunked as text; a grammar would be unused"
            )
        if spec.name in self._specs:
            raise ValueError(f"language {spec.name!r} is already registered")
        for suffix in spec.suffixes:
            owner = self._owner_of(suffix=suffix)
            if owner is not None:
                raise ValueError(f"suffix {suffix!r} is already claimed by {owner!r}")
        for filename in spec.filenames:
            owner = self._owner_of(filename=filename)
            if owner is not None:
                raise ValueError(f"filename {filename!r} is already claimed by {owner!r}")
        self._specs[spec.name] = spec
        # Anything derived from the specs is now stale.
        self._parsers = None

    def _owner_of(self, *, suffix: str | None = None, filename: str | None = None) -> str | None:
        """Name of the language already claiming this suffix or filename."""
        for spec in self._specs.values():
            if suffix is not None and suffix in spec.suffixes:
                return spec.name
            if filename is not None and filename in spec.filenames:
                return spec.name
        return None

    @property
    def specs(self) -> tuple[LanguageSpec, ...]:
        """Every registered language, in registration order."""
        self._ensure_plugins()
        return tuple(self._specs.values())

    def get(self, name: str) -> LanguageSpec | None:
        """The spec registered under `name`, or None."""
        self._ensure_plugins()
        return self._specs.get(name)

    def match(self, path: Path) -> LanguageSpec | None:
        """The language of a file, by suffix or exact name; None = skip.

        Suffix first, then exact name — the same order the walker used
        when this table was a pair of module constants.

        Args:
            path: File to identify; only its name is read.

        Returns:
            The matching spec, or None when no language claims the file.
        """
        self._ensure_plugins()
        suffix = path.suffix.lower()
        for spec in self._specs.values():
            if suffix and suffix in spec.suffixes:
                return spec
        for spec in self._specs.values():
            if path.name in spec.filenames:
                return spec
        return None

    def parser(self, name: str) -> Parser | None:
        """The parser for a language, or None when its grammar is absent.

        None is a normal answer, not an error: grammars ship in the
        optional `ast` extra, so a spec can name one that is not
        installed. The chunker falls back to text windows.
        """
        self._ensure_plugins()
        return self._build_parsers().get(name)

    def extractor(self, name: str) -> SpanExtractor | None:
        """The span extractor for a language, or None if it has none."""
        self._ensure_plugins()
        spec = self._specs.get(name)
        return spec.spans if spec is not None else None

    def _build_parsers(self) -> dict[str, Parser]:
        """Instantiate every grammar that imports; cache until re-register."""
        if self._parsers is None:
            self._parsers = self._load()
        return self._parsers

    def _load(self) -> dict[str, Parser]:
        if not HAS_TREE_SITTER:  # pragma: no cover - base install only (CI matrix)
            return {}
        # Imported here, not at module scope: `HAS_TREE_SITTER` is False
        # on a base install and the names would not exist.
        import importlib

        from tree_sitter import Language, Parser

        parsers: dict[str, Parser] = {}
        for spec in self._specs.values():
            if spec.grammar is None:
                continue
            try:
                module = importlib.import_module(spec.grammar.module)
            except ImportError:  # pragma: no cover - partial grammar install
                continue
            parsers[spec.name] = Parser(Language(getattr(module, spec.grammar.getter)()))
        return parsers


__all__ = [
    "BUILTIN_LANGUAGES",
    "REGISTRY",
    "GrammarSpec",
    "LanguageRegistry",
    "LanguageSpec",
    # Re-exported from `ast.core`, where it lives so the language modules
    # can name it without importing this registry back.
    "Section",
    "SectionSplitter",
    "SpanExtractor",
]

REGISTRY = LanguageRegistry(plugins=True)
"""The process-wide registry. Step 24's plugin loader appends to it."""


BUILTIN_LANGUAGES: tuple[LanguageSpec, ...] = (
    # Code: the ARCH §1 corpus. Each one is a grammar plus the policy
    # module that says which of its nodes deserve a chunk.
    LanguageSpec(
        name="python",
        kind=Kind.CODE,
        suffixes=(".py",),
        grammar=GrammarSpec(module="tree_sitter_python", getter="language"),
        spans=python.spans,
    ),
    LanguageSpec(
        name="rust",
        kind=Kind.CODE,
        suffixes=(".rs",),
        grammar=GrammarSpec(module="tree_sitter_rust", getter="language"),
        spans=rust.spans,
    ),
    LanguageSpec(
        name="typescript",
        kind=Kind.CODE,
        suffixes=(".ts",),
        # tree-sitter-typescript ships two grammars in one module; `.tsx`
        # would need `language_tsx` and is not indexed today.
        grammar=GrammarSpec(module="tree_sitter_typescript", getter="language_typescript"),
        spans=typescript.spans,
    ),
    LanguageSpec(
        name="javascript",
        kind=Kind.CODE,
        # `.jsx` too: the JavaScript grammar parses JSX, so React sources
        # need no second language. `.mjs`/`.cjs` are the module-flavoured
        # spellings Node introduced.
        suffixes=(".js", ".jsx", ".mjs", ".cjs"),
        grammar=GrammarSpec(module="tree_sitter_javascript", getter="language"),
        # Same policy object as TypeScript, not a copy of it: the node
        # names the extractor matches on (`function_declaration`,
        # `class_declaration`, `lexical_declaration`, `export_statement`)
        # are shared across the whole family. TS-only nodes simply never
        # appear in a JS tree.
        spans=typescript.spans,
    ),
    LanguageSpec(
        name="tsx",
        kind=Kind.CODE,
        suffixes=(".tsx",),
        # A language of its own only because the grammar is: TSX is a
        # separate parser inside tree-sitter-typescript, and one spec
        # binds one grammar. The policy is TypeScript's, unchanged.
        grammar=GrammarSpec(module="tree_sitter_typescript", getter="language_tsx"),
        spans=typescript.spans,
    ),
    LanguageSpec(
        name="go",
        kind=Kind.CODE,
        suffixes=(".go",),
        grammar=GrammarSpec(module="tree_sitter_go", getter="language"),
        spans=go.spans,
    ),
    LanguageSpec(
        name="c",
        kind=Kind.CODE,
        # `.h` goes to C rather than C++ on the usual convention. A C++
        # header written as `.h` is still indexed — the parser is error
        # tolerant — but its classes fall to the gap pass. `.hpp` is the
        # unambiguous spelling and belongs to C++ below.
        suffixes=(".c", ".h"),
        grammar=GrammarSpec(module="tree_sitter_c", getter="language"),
        spans=c.spans,
    ),
    LanguageSpec(
        name="cpp",
        kind=Kind.CODE,
        suffixes=(".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"),
        grammar=GrammarSpec(module="tree_sitter_cpp", getter="language"),
        spans=cpp.spans,
    ),
    LanguageSpec(
        name="java",
        kind=Kind.CODE,
        suffixes=(".java",),
        grammar=GrammarSpec(module="tree_sitter_java", getter="language"),
        spans=java.spans,
    ),
    # Configs: chunked by their top-level structure rather than by defs.
    LanguageSpec(
        name="csharp",
        kind=Kind.CODE,
        suffixes=(".cs",),
        grammar=GrammarSpec(module="tree_sitter_c_sharp", getter="language"),
        spans=csharp.spans,
    ),
    LanguageSpec(
        name="kotlin",
        kind=Kind.CODE,
        # `.kts` is a Kotlin script — same grammar, same policy.
        suffixes=(".kt", ".kts"),
        grammar=GrammarSpec(module="tree_sitter_kotlin", getter="language"),
        spans=kotlin.spans,
    ),
    LanguageSpec(
        name="php",
        kind=Kind.CODE,
        suffixes=(".php",),
        # `language_php` parses a full file including the `<?php` tag;
        # the package also ships a tag-less variant we do not want.
        grammar=GrammarSpec(module="tree_sitter_php", getter="language_php"),
        spans=php.spans,
    ),
    LanguageSpec(
        name="ruby",
        kind=Kind.CODE,
        suffixes=(".rb",),
        filenames=("Rakefile", "Gemfile"),
        grammar=GrammarSpec(module="tree_sitter_ruby", getter="language"),
        spans=ruby.spans,
    ),
    LanguageSpec(
        name="html",
        kind=Kind.CODE,
        # Angular component templates are plain HTML files, so they need
        # no dialect: `foo.component.html` is matched by `.html` here.
        suffixes=(".html", ".htm"),
        grammar=GrammarSpec(module="tree_sitter_html", getter="language"),
        spans=html.spans,
    ),
    LanguageSpec(
        name="vue",
        kind=Kind.CODE,
        suffixes=(".vue",),
        # A container, not a language: the HTML grammar parses the
        # skeleton and `sections` hands each part to the language it is
        # actually written in. See `wsindex.ingest.ast.sfc`.
        grammar=GrammarSpec(module="tree_sitter_html", getter="language"),
        sections=sfc.sections,
    ),
    LanguageSpec(
        name="svelte",
        kind=Kind.CODE,
        suffixes=(".svelte",),
        # Same splitter as Vue: the difference between the two is where
        # the markup sits, not how the file is structured.
        grammar=GrammarSpec(module="tree_sitter_html", getter="language"),
        sections=sfc.sections,
    ),
    LanguageSpec(
        name="toml",
        kind=Kind.CONFIG,
        suffixes=(".toml",),
        grammar=GrammarSpec(module="tree_sitter_toml", getter="language"),
        spans=configs.toml_spans,
    ),
    LanguageSpec(
        name="yaml",
        kind=Kind.CONFIG,
        suffixes=(".yaml", ".yml"),
        grammar=GrammarSpec(module="tree_sitter_yaml", getter="language"),
        spans=configs.yaml_spans,
    ),
    LanguageSpec(
        name="json",
        kind=Kind.CONFIG,
        suffixes=(".json",),
        grammar=GrammarSpec(module="tree_sitter_json", getter="language"),
        spans=configs.json_spans,
    ),
    LanguageSpec(
        name="dockerfile",
        kind=Kind.CONFIG,
        filenames=("Dockerfile",),
        grammar=GrammarSpec(module="tree_sitter_dockerfile", getter="language"),
        spans=configs.dockerfile_spans,
    ),
    # Docs: no grammar by design — the text chunker splits them by
    # headers, and `register` refuses a DOC grammar that nothing calls.
    LanguageSpec(name="markdown", kind=Kind.DOC, suffixes=(".md",)),
    LanguageSpec(name="rst", kind=Kind.DOC, suffixes=(".rst",)),
    LanguageSpec(name="text", kind=Kind.DOC, suffixes=(".txt",)),
)
"""The languages wsindex ships with. Only the ARCH §1 corpus is listed on
purpose: every extra format is a future obligation for the chunkers —
which is exactly what a plugin now lets someone take on themselves."""


for _spec in BUILTIN_LANGUAGES:
    REGISTRY.register(_spec)
