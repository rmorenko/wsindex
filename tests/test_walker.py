"""Tests for repository walking, filtering and lang/kind detection."""

from pathlib import Path

import pytest

from wsindex.ingest.walker import MAX_FILE_SIZE, walk_repo
from wsindex.model import Kind


def make_repo(root: Path) -> None:
    """Build a synthetic repo tree covering every walker rule.

    Indexable (must be walked):
        src/main.py, app.ts, Dockerfile, pyproject.toml, README.md
    Traps (must be skipped):
        .git/x/cfg.json      - ignored dir; `.git` itself has no files,
                               so it catches pruning done in the wrong loop
        node_modules/c.yaml  - ignored dir
        __pycache__/m.py     - ignored dir
        logo.png             - binary (NUL byte in prefix)
        big.txt              - exceeds MAX_FILE_SIZE
        data.xyz             - unknown extension
    """
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("def f(): pass")
    (root / "app.ts").write_text("let x = 1")
    (root / "Dockerfile").write_text("FROM python:3.11")
    (root / "pyproject.toml").write_text('[project]\nname = "x"')
    (root / "README.md").write_text("# Demo")

    (root / ".git" / "x").mkdir(parents=True)
    (root / ".git" / "x" / "cfg.json").write_text("{}")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "c.yaml").write_text("a: 1")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "m.py").write_text("x = 1")
    (root / "logo.png").write_bytes(b"\x89PNG\x00\x01")
    (root / "big.txt").write_text("x" * (MAX_FILE_SIZE + 1))
    (root / "data.xyz").write_text("mystery")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Fresh synthetic repo per test (composes with the built-in tmp_path)."""
    make_repo(tmp_path)
    return tmp_path


def test_walks_expected_files(repo: Path) -> None:
    rel_paths = {f.rel_path for f in walk_repo(repo)}
    assert rel_paths == {
        "src/main.py",
        "app.ts",
        "Dockerfile",
        "pyproject.toml",
        "README.md",
    }


def test_skips_ignored_dirs(repo: Path) -> None:
    rel_paths = {f.rel_path for f in walk_repo(repo)}
    assert not {p for p in rel_paths if p.startswith((".git", "node_modules", "__pycache__"))}


def test_skips_binary_and_large_files(repo: Path) -> None:
    rel_paths = {f.rel_path for f in walk_repo(repo)}
    assert "logo.png" not in rel_paths  # NUL byte in prefix -> binary
    assert "big.txt" not in rel_paths  # MAX_FILE_SIZE + 1 bytes


def test_skips_unknown_extensions(repo: Path) -> None:
    rel_paths = {f.rel_path for f in walk_repo(repo)}
    assert "data.xyz" not in rel_paths


@pytest.mark.parametrize(
    ("rel_path", "lang", "kind"),
    [
        ("src/main.py", "python", Kind.CODE),
        ("pyproject.toml", "toml", Kind.CONFIG),
        ("README.md", "markdown", Kind.DOC),
        ("Dockerfile", "dockerfile", Kind.CONFIG),
    ],
)
def test_detects_lang_and_kind(repo: Path, rel_path: str, lang: str, kind: Kind) -> None:
    by_path = {f.rel_path: (f.lang, f.kind) for f in walk_repo(repo)}
    assert by_path[rel_path] == (lang, kind)


def test_rel_path_is_posix_relative(repo: Path) -> None:
    # rel_path feeds chunk_id, so it must never leak the absolute tmp root.
    rel_paths = {f.rel_path for f in walk_repo(repo)}
    for rel_path in rel_paths:
        assert not rel_path.startswith("/")
