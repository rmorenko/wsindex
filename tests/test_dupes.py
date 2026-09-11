"""Finding the same code twice, by what it is made of.

The mechanism was chosen by measurement and the numbers live in the
module. What is tested here is the machinery and, above all, the two
decisions that make the report readable rather than true-but-useless: the
threshold, which was placed by reading what sits on each side of it, and
the grouping, which turns four thousand pairs between two copies of a
vendored library into one line.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from wsindex.config import Config, Repository
from wsindex.dupes import COMMON_SHINGLE, MIN_TOKENS, SHINGLE, _pairs, find, fingerprint
from wsindex.embed import FakeEmbedder
from wsindex.pipeline import Pipeline
from wsindex.store import LanceDBStore

WORDS = " ".join(f"name_{n} value_{n} result_{n}" for n in range(40))


# --- the fingerprint ------------------------------------------------------


def test_identical_text_has_an_identical_fingerprint() -> None:
    assert fingerprint(WORDS) == fingerprint(WORDS)


def test_reformatting_changes_nothing() -> None:
    # Identifiers only, so indentation, line breaks and punctuation — the
    # first things a copy changes — are invisible to this.
    spaced = WORDS.replace(" ", "\n    ").replace("name_", "name_")

    assert fingerprint(WORDS) == fingerprint(spaced)


def test_renaming_one_identifier_costs_only_the_shingles_around_it() -> None:
    # The reason shingles overlap: an edit should cost a few hashes, not
    # the alignment of everything after it.
    edited = WORDS.replace("value_20", "renamed_thing", 1)

    shared = fingerprint(WORDS) & fingerprint(edited)

    assert len(shared) >= len(fingerprint(WORDS)) - SHINGLE - 1


def test_something_too_short_to_fingerprint_has_no_fingerprint() -> None:
    # A four-line accessor is identical to a thousand others and says
    # nothing; removing it here keeps it out of every later stage.
    assert fingerprint("one two three") == set()
    assert len(fingerprint(" ".join(f"w{n}" for n in range(MIN_TOKENS - 1)))) == 0


# --- the report -----------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository with a vendored copy, a hand copy, and an original."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "test@example.invalid")

    def body(name: str, seed: int, salt: str = "") -> str:
        """A function whose identifiers depend on `seed`.

        Distinct per seed, and that matters: the first version of this
        fixture varied only the function name, so all twelve "vendored"
        modules were copies of each other as well as of their mirror —
        and the report correctly found five groups where the test
        expected one. Real vendored files differ from each other.
        """
        lines = [f"def {name}(payload_{seed}, options_{seed}, registry_{seed}):"]
        lines += [
            f"    value_{seed}_{n} = registry_{seed}.lookup_{seed}("
            f"payload_{seed}, options_{seed}, 'field_{seed}_{n}')"
            for n in range(14)
        ]
        lines.append(f"    return {salt or f'value_{seed}_0'}")
        return "\n".join(lines) + "\n"

    layout = {
        # Twelve files copied wholesale into two places — a vendored tree.
        **{f"vendor/lib/mod_{n}.py": body(f"helper_{n}", n) for n in range(12)},
        **{f"third_party/lib/mod_{n}.py": body(f"helper_{n}", n) for n in range(12)},
        # One function copied by hand and lightly edited.
        "app/report.py": body("report", 99),
        "app/summary.py": body("summary", 99, salt="value_99_1"),
        # And something with nothing in common.
        "app/unrelated.py": "def unrelated():\n    return 'nothing like the others at all'\n",
    }
    for rel, text in layout.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    for args in (["init", "-q", "--initial-branch=main"], ["add", "-A"], ["commit", "-qm", "x"]):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


@pytest.fixture
def indexed(repo: Path, tmp_path_factory: pytest.TempPathFactory) -> Pipeline:
    home = tmp_path_factory.mktemp("dupes")
    Config.reset()
    config = Config.default("dup")
    config._data["store"] = {"uri": str(home / "db")}
    config.add_repo(Repository(id="r", path=str(repo)))
    pipeline = Pipeline(
        store=LanceDBStore(uri=str(home / "db"), embedder=FakeEmbedder(dim=8)),
        state_dir=home,
        config=config,
    )
    pipeline.index()
    return pipeline


def test_a_copied_directory_is_one_line_not_a_hundred(indexed: Pipeline) -> None:
    """The decision the whole report is built around.

    Twelve copied files are one fact — somebody vendored a library — and
    printing twelve pairs instead of one line is how a duplication report
    becomes something nobody opens twice.
    """
    found = find(indexed, repo="r")

    vendored = [b for b in found.between if b.wholesale]
    assert len(vendored) == 1
    assert {vendored[0].left, vendored[0].right} == {"vendor/lib", "third_party/lib"}
    assert len(vendored[0].pairs) >= 10


def test_a_hand_copy_is_reported_separately(indexed: Pipeline) -> None:
    found = find(indexed, repo="r")

    handmade = [b for b in found.between if not b.wholesale]
    paths = {(p.left, p.right) for b in handmade for p in b.pairs}

    assert ("app/report.py", "app/summary.py") in paths or (
        "app/summary.py",
        "app/report.py",
    ) in paths


def test_code_with_nothing_in_common_is_not_reported(indexed: Pipeline) -> None:
    found = find(indexed, repo="r")

    assert not any(
        "unrelated.py" in (pair.left, pair.right) for group in found.between for pair in group.pairs
    )


def test_a_pair_carries_the_lines_to_look_at(indexed: Pipeline) -> None:
    # A report that names two files without saying where in them is a
    # report that makes somebody grep.
    found = find(indexed, repo="r")
    pair = found.between[0].pairs[0]

    assert pair.left_lines[0] >= 1
    assert pair.left_lines[1] >= pair.left_lines[0]


def test_raising_the_threshold_narrows_the_report(indexed: Pipeline) -> None:
    loose = find(indexed, repo="r", minimum=0.3)
    strict = find(indexed, repo="r", minimum=0.95)

    assert sum(len(b.pairs) for b in strict.between) <= sum(len(b.pairs) for b in loose.between)


def test_an_unknown_repo_is_the_callers_mistake(indexed: Pipeline) -> None:
    with pytest.raises(ValueError, match="no repo"):
        find(indexed, repo="not-a-repo")


def test_one_directory_repeating_itself_says_so(indexed: Pipeline) -> None:
    # A tree of generated files duplicating itself is a real and common
    # shape, and a different fact from two directories mirroring each
    # other — so the report distinguishes them rather than printing the
    # same path twice.
    found = find(indexed, repo="r", minimum=0.3)

    assert any(group.within for group in found.between) or all(
        group.left != group.right for group in found.between
    )


def test_two_chunks_of_one_file_are_not_a_copy(indexed: Pipeline) -> None:
    """Windows inside a long file overlap by construction.

    The sliding chunker shares lines between neighbours on purpose, so
    every long file would otherwise report itself as a duplicate of
    itself — thousands of findings that are the chunker working.
    """
    found = find(indexed, repo="r", minimum=0.3)

    assert all(pair.left != pair.right for group in found.between for pair in group.pairs)


# --- the two filters that keep the report finishable and honest -----------


def test_a_shingle_everybody_has_is_dropped() -> None:
    """Boilerplate connects the whole repository to itself.

    A licence header or a framework's call signature appears everywhere.
    Kept in the index it makes candidate generation quadratic again —
    which is the cost the index exists to avoid — and fills the report
    with `public function`.
    """
    shared = fingerprint(WORDS)
    crowd = {f"c{n}": set(shared) for n in range(COMMON_SHINGLE + 5)}
    chunks = {name: (f"file_{name}.py", "", (1, 2)) for name in crowd}

    assert _pairs(chunks, crowd, minimum=0.1) == []


def test_two_windows_of_one_file_are_not_a_copy() -> None:
    # The sliding chunker shares lines between neighbours on purpose, so
    # without this every long file reports itself.
    marks = fingerprint(WORDS)
    chunks = {"a": ("same.py", "", (1, 40)), "b": ("same.py", "", (30, 70))}

    assert _pairs(chunks, {"a": marks, "b": marks}, minimum=0.1) == []


def test_two_windows_of_different_files_are_a_copy() -> None:
    # The control for the test above: the only difference is the path.
    marks = fingerprint(WORDS)
    chunks = {"a": ("one.py", "", (1, 40)), "b": ("two.py", "", (1, 40))}

    assert len(_pairs(chunks, {"a": marks, "b": marks}, minimum=0.1)) == 1
