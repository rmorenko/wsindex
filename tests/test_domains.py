"""Reading a repository's shape, and the arithmetic that makes it a reading.

The spike that admitted this candidate set its own bar: the domains found
have to match the author's intuition about their own code. That intuition
is the package layout, so what is tested here is the machinery that
compares the two — above all `package_of`, whose first version was off by
one and turned a nine-package project into a forty-three-package one,
which made the baseline 2% and every number after it meaningless.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from wsindex.config import Config, Repository
from wsindex.domains import (
    COUPLED_FROM,
    NEIGHBOURS,
    analyse,
    branching_depth,
    package_of,
    source_prefix,
)
from wsindex.embed import FakeEmbedder
from wsindex.pipeline import Pipeline
from wsindex.store import LanceDBStore

# --- naming the package a file belongs to ---------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/wsindex/store/base.py", "store"),
        ("src/wsindex/ingest/ast/core.py", "ingest"),
        ("src/wsindex/pipeline.py", "(root)"),
        ("src/wsindex/__init__.py", "(root)"),
    ],
)
def test_a_file_is_named_by_its_directory_not_its_own_name(path: str, expected: str) -> None:
    # The off-by-one that started this: `parts[3]` for the four-part path
    # above is `base.py`, so every file became its own package.
    assert package_of(path, depth=2) == expected


def test_the_depth_is_derived_from_the_layout_not_assumed() -> None:
    """A constant here fits one project and misreads the next.

    `src/wsindex/store/base.py` wants level 2 and `src/alpha/one.py`
    wants level 1. A default of either reports the other as a
    single-package repository, which is what the first version did.
    """
    deep = ["src/wsindex/store/base.py", "src/wsindex/cli/init.py", "src/wsindex/pipeline.py"]
    shallow = ["src/alpha/one.py", "src/beta/two.py", "src/six.py"]

    assert branching_depth(deep) == 2
    assert branching_depth(shallow) == 1
    assert package_of("src/alpha/one.py", depth=branching_depth(shallow)) == "alpha"


def test_a_level_with_one_name_distinguishes_nothing() -> None:
    # Every file is under `src`, so `src` cannot be anybody's package.
    assert branching_depth(["src/only/deep/a.py", "src/only/deep/b.py"]) == 2


# --- the report -----------------------------------------------------------
#
# The semantic half cannot be tested here and is not pretended to be: the
# suite runs on the fake embedder, whose vectors are deterministic noise,
# so "meaning recovers the layout" would measure nothing. That claim was
# established by a spike on this repository — 57% against an 11% baseline
# — and is recorded where the module explains itself. What is testable is
# the machinery: the arithmetic, the guards, and the co-change half,
# which reads git and does not care about vectors at all.


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository with three packages and a history that couples two."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "test@example.invalid")

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    layout = {
        "src/alpha/one.py": "def one():\n    return 1\n",
        "src/alpha/two.py": "def two():\n    return 2\n",
        "src/beta/three.py": "def three():\n    return 3\n",
        "src/beta/four.py": "def four():\n    return 4\n",
        "src/gamma/five.py": "def five():\n    return 5\n",
        "src/six.py": "def six():\n    return 6\n",
        "tests/test_seven.py": "def test_seven():\n    assert True\n",
    }
    for rel, text in layout.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    git("init", "-q", "--initial-branch=main")
    git("add", "-A")
    git("commit", "-qm", "first")
    # Five commits touching one file in alpha and one in beta together:
    # coupling across a package boundary that no import would show.
    for n in range(5):
        (tmp_path / "src/alpha/one.py").write_text(f"def one():\n    return {n}\n")
        (tmp_path / "src/beta/three.py").write_text(f"def three():\n    return {n}\n")
        git("add", "-A")
        git("commit", "-qm", f"together {n}")
    return tmp_path


@pytest.fixture
def analysed(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> Pipeline:
    home = tmp_path_factory.mktemp("idx")
    Config.reset()
    config = Config.default("dom")
    config._data["store"] = {"uri": str(home / "db")}
    config.add_repo(Repository(id="r", path=str(repo)))
    pipeline = Pipeline(
        store=LanceDBStore(uri=str(home / "db"), embedder=FakeEmbedder(dim=8)),
        state_dir=home,
        config=config,
    )
    pipeline.index()
    return pipeline


def test_packages_are_counted_by_directory(analysed: Pipeline) -> None:
    found = analyse(analysed, repo="r", prefix="src/")

    assert found.packages == {"alpha": 2, "beta": 2, "gamma": 1, "(root)": 1}
    assert found.files == 6


def test_the_prefix_keeps_tests_out_of_the_picture(analysed: Pipeline) -> None:
    # A report that counted a project's tests would describe the project
    # plus everything it happens to contain.
    only_src = analyse(analysed, repo="r", prefix="src/")
    everything = analyse(analysed, repo="r", prefix="")

    assert everything.files > only_src.files


def test_files_that_keep_changing_together_are_reported(analysed: Pipeline) -> None:
    # Five commits touched one alpha file and one beta file together, and
    # nothing imports anything — which is the point: this signal sees
    # coupling a call graph cannot.
    found = analyse(analysed, repo="r", prefix="src/")

    pairs = {(c.left, c.right) for c in found.coupled}
    assert ("src/alpha/one.py", "src/beta/three.py") in pairs
    assert all(c.commits >= COUPLED_FROM for c in found.coupled)


def test_coupling_inside_one_package_is_not_news(analysed: Pipeline) -> None:
    # Two files in the same package changing together is what a package
    # is. Only crossings are worth a line in the report.
    found = analyse(analysed, repo="r", prefix="src/")

    depth = branching_depth([c.left for c in found.coupled])
    assert all(
        package_of(c.left, depth=depth) != package_of(c.right, depth=depth) for c in found.coupled
    )


def test_an_unknown_repo_is_the_callers_mistake(analysed: Pipeline) -> None:
    with pytest.raises(ValueError, match="no repo"):
        analyse(analysed, repo="not-a-repo")


def test_a_repository_too_small_to_have_domains_says_so(analysed: Pipeline) -> None:
    # Five files cannot have five neighbours each, and inventing domains
    # from four comparisons would be worse than saying nothing.
    found = analyse(analysed, repo="r", prefix="src/alpha")

    assert found.files < NEIGHBOURS + 1
    assert (found.strangers, found.coupled) == ((), ())


def test_coupling_needs_more_than_a_coincidence() -> None:
    # One sweeping rename touches twenty files; that is not a design fact.
    assert COUPLED_FROM > 1


# --- where a repository keeps the code it is about ------------------------


def test_a_gem_keeps_its_code_in_lib() -> None:
    # The old default was `src/`, which is this project's layout and
    # almost nobody else's. Measured on the pinned corpus, deriving it
    # took the repositories that can report packages from 5 of 22 to 18.
    assert source_prefix(["lib/a.rb", "lib/b/c.rb", "spec/a_spec.rb"]) == "lib/"


def test_a_go_module_keeps_it_at_the_root() -> None:
    # Nothing in common between `cmd/` and `internal/`, so the answer is
    # the whole repository — which is that repository's actual shape, not
    # a failure to find a prefix.
    assert source_prefix(["cmd/main.go", "internal/run.go", "server.go"]) == ""


def test_tests_do_not_decide_where_the_code_lives() -> None:
    # The one job the prefix has. Without dropping them, a repo with more
    # test files than source files would be described by its tests.
    paths = ["src/one.py", "src/two.py", "tests/a.py", "tests/b.py", "tests/c.py"]

    assert source_prefix(paths) == "src/"


def test_a_repository_that_is_only_tests_is_read_whole() -> None:
    # Dropping every path would leave nothing to derive from, and
    # answering "src/" there would be inventing a layout.
    assert source_prefix(["tests/a.py", "spec/b.rb"]) == ""


def test_nothing_indexed_derives_nothing() -> None:
    assert source_prefix([]) == ""


def test_a_deeply_nested_layout_keeps_all_of_what_is_shared() -> None:
    # Java puts everything under `src/main/java`, and stopping at `src/`
    # would pull `src/test/java` back in.
    paths = [
        "src/main/java/org/app/store/One.java",
        "src/main/java/org/app/web/Two.java",
        "src/test/java/org/app/store/OneTest.java",
    ]

    assert source_prefix(paths) == "src/main/java/org/app/"
