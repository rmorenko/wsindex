"""Tests for the indexing policy: which files count, and as what.

`selected` below is how the pipeline actually decides — it lists paths
and asks `inspect_file` about each. A `walk_repo` generator used to
answer the same question by traversing the tree itself; it went away with
step 29, having outlived the pipeline's use of it by four steps.
"""

from pathlib import Path

import pytest

from wsindex.ingest.walker import (
    IGNORED_DIRS,
    MAX_FILE_SIZE,
    WalkedFile,
    _skip_dir,
    inspect_file,
)
from wsindex.model import Kind


def selected(root: Path) -> list[WalkedFile]:
    """Every file under `root` the policy accepts, as the pipeline asks it.

    Paths first, then one `inspect_file` each — the same shape as an
    index run, which lists with git and inspects what it gets back.
    """
    return [
        walked
        for path in sorted(root.rglob("*"))
        if (walked := inspect_file(root, path.relative_to(root).as_posix())) is not None
    ]


def make_repo(root: Path) -> None:
    """Build a synthetic repo tree covering every walker rule.

    Indexable (must be walked):
        src/main.py, app.ts, App.java, Dockerfile, pyproject.toml, README.md
    Traps (must be skipped):
        .git/x/cfg.json      - ignored dir; `.git` itself has no files,
                               so it catches pruning done in the wrong loop
        node_modules/c.yaml  - ignored dir
        __pycache__/m.py     - ignored dir
        logo.png             - unknown extension AND binary
        fake.txt             - indexable extension but binary content
        big.txt              - exceeds MAX_FILE_SIZE
        data.xyz             - unknown extension
    """
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("def f(): pass")
    (root / "app.ts").write_text("let x = 1")
    (root / "App.java").write_text("class App {}")
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
    (root / "fake.txt").write_bytes(b"looks like text\x00but is not")
    (root / "big.txt").write_text("x" * (MAX_FILE_SIZE + 1))
    (root / "data.xyz").write_text("mystery")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Fresh synthetic repo per test (composes with the built-in tmp_path)."""
    make_repo(tmp_path)
    return tmp_path


def test_walks_expected_files(repo: Path) -> None:
    rel_paths = {f.rel_path for f in selected(repo)}
    assert rel_paths == {
        "src/main.py",
        "app.ts",
        "App.java",
        "Dockerfile",
        "pyproject.toml",
        "README.md",
    }


def test_skips_ignored_dirs(repo: Path) -> None:
    rel_paths = {f.rel_path for f in selected(repo)}
    assert not {p for p in rel_paths if p.startswith((".git", "node_modules", "__pycache__"))}


def test_skips_binary_and_large_files(repo: Path) -> None:
    rel_paths = {f.rel_path for f in selected(repo)}
    # logo.png is rejected by extension before the binary sniff even runs;
    # fake.txt is the case that actually exercises is_binary.
    assert "logo.png" not in rel_paths
    assert "fake.txt" not in rel_paths  # NUL byte in prefix -> binary
    assert "big.txt" not in rel_paths  # MAX_FILE_SIZE + 1 bytes


def test_skips_unknown_extensions(repo: Path) -> None:
    rel_paths = {f.rel_path for f in selected(repo)}
    assert "data.xyz" not in rel_paths


@pytest.mark.parametrize(
    ("rel_path", "lang", "kind"),
    [
        ("src/main.py", "python", Kind.CODE),
        ("App.java", "java", Kind.CODE),
        ("pyproject.toml", "toml", Kind.CONFIG),
        ("README.md", "markdown", Kind.DOC),
        ("Dockerfile", "dockerfile", Kind.CONFIG),
    ],
)
def test_detects_lang_and_kind(repo: Path, rel_path: str, lang: str, kind: Kind) -> None:
    by_path = {f.rel_path: (f.lang, f.kind) for f in selected(repo)}
    assert by_path[rel_path] == (lang, kind)


def test_rel_path_is_posix_relative(repo: Path) -> None:
    # rel_path feeds chunk_id, so it must never leak the absolute tmp root.
    rel_paths = {f.rel_path for f in selected(repo)}
    for rel_path in rel_paths:
        assert not rel_path.startswith("/")


def test_skips_arbitrary_hidden_directory(tmp_path: Path) -> None:
    # The dot-prefix rule is a policy, not a list — any hidden dir the
    # walker has never heard of must still be pruned. This is the case
    # the refactor targeted: previously the rule lived inline in the
    # walk loop, far from IGNORED_DIRS; a reader saw the constant and
    # missed the implicit second policy.
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.py").write_text("x = 1\n")
    assert selected(tmp_path) == []


def test_skip_dir_hides_dot_prefixed_names() -> None:
    assert _skip_dir(".git")
    assert _skip_dir(".venv")
    assert _skip_dir(".anything-new-and-hidden")


def test_skip_dir_hides_ignored_names() -> None:
    for name in IGNORED_DIRS:
        assert _skip_dir(name)


def test_skip_dir_lets_regular_directories_through() -> None:
    assert not _skip_dir("src")
    assert not _skip_dir("tests")
    assert not _skip_dir("docs")


def test_ignored_dirs_does_not_repeat_the_hidden_rule() -> None:
    # A dot-prefixed name in IGNORED_DIRS would be redundant: startswith(".")
    # catches it first. Keeping the list free of dot-names is the invariant
    # that made the refactor worth doing.
    assert not any(name.startswith(".") for name in IGNORED_DIRS)


# --- step 22: the per-file policy, asked about one named path ------------


def test_the_policy_is_the_same_however_a_path_arrives(tmp_path: Path) -> None:
    # Incremental runs ask about the handful of paths git reported as
    # changed; full runs ask about every path git lists. Both go through
    # `inspect_file`, and this pins that the answer depends on the file
    # rather than on how the run found it.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("print('a')\n")
    (tmp_path / "notes.md").write_text("# notes\n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\x00\x00")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.py").write_text("print('dep')\n")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.py").write_text("print('s')\n")

    every_path = [
        str(p.relative_to(tmp_path).as_posix()) for p in tmp_path.rglob("*") if p.is_file()
    ]
    one_at_a_time = {p for p in every_path if inspect_file(tmp_path, p) is not None}
    all_at_once = {f.rel_path for f in selected(tmp_path)}
    assert one_at_a_time == all_at_once == {"src/a.py", "notes.md"}


def test_inspect_file_skips_a_pruned_directory(tmp_path: Path) -> None:
    # git reports a tracked file under node_modules/ happily; the walker
    # never descends there, so inspect_file must reject it by path.
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.py").write_text("print('dep')\n")
    assert inspect_file(tmp_path, "node_modules/dep.py") is None


def test_inspect_file_skips_a_hidden_directory(tmp_path: Path) -> None:
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "mod.py").write_text("print('v')\n")
    assert inspect_file(tmp_path, ".venv/mod.py") is None


def test_inspect_file_on_a_missing_path_is_none(tmp_path: Path) -> None:
    # A path can vanish between git reporting it and this call.
    assert inspect_file(tmp_path, "gone.py") is None


def test_inspect_file_on_a_directory_is_none(tmp_path: Path) -> None:
    (tmp_path / "weird.py").mkdir()
    assert inspect_file(tmp_path, "weird.py") is None


def test_inspect_file_skips_an_oversized_file(tmp_path: Path) -> None:
    (tmp_path / "huge.py").write_text("x" * (MAX_FILE_SIZE + 1))
    assert inspect_file(tmp_path, "huge.py") is None


def test_inspect_file_skips_a_binary_file(tmp_path: Path) -> None:
    (tmp_path / "blob.py").write_bytes(b"print('a')\x00binary")
    assert inspect_file(tmp_path, "blob.py") is None


def test_inspect_file_skips_an_unreadable_file(tmp_path: Path) -> None:
    target = tmp_path / "locked.py"
    target.write_text("print('locked')\n")
    target.chmod(0o000)
    try:
        assert inspect_file(tmp_path, "locked.py") is None
    finally:
        target.chmod(0o644)


def test_inspect_file_returns_lang_and_kind(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("print('m')\n")
    walked = inspect_file(tmp_path, "mod.py")
    assert walked is not None
    assert (walked.rel_path, walked.lang, walked.kind) == ("mod.py", "python", Kind.CODE)
