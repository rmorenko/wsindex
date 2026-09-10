"""Shared test helpers, importable without pytest's path tricks.

Not `conftest.py`: `from conftest import ...` works only because the
default `prepend` import mode puts the rootdir on `sys.path`, and pytest
has said `importlib` is where it is heading. A plain module beside the
tests is imported by the same rules as the code under test.
"""

import pytest


def needs_grammar(*langs: str) -> pytest.MarkDecorator:
    """Skip unless every named tree-sitter grammar is installed.

    Forty-four tests repeated `pytest.mark.skipif(not has_grammar(...),
    reason="needs the ast extra")`, and three files each defined their
    own `has_grammar`. The condition really does differ per test — the
    grammars are separate packages — but the shape and the words did not.

    The reason names the grammar, which the copied string never did: a
    skipped run used to say "needs the ast extra" and leave you to work
    out which of eight was missing.

    Args:
        langs: Language names as the registry knows them.

    Returns:
        The mark to apply.
    """
    from wsindex.ingest.languages import REGISTRY

    missing = [lang for lang in langs if REGISTRY.parser(lang) is None]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"needs the ast extra ({', '.join(missing) or ', '.join(langs)})",
    )
