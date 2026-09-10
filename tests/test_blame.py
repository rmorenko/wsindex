"""The blame pass, and the child process that now does most of it.

Two paths to one answer, so the tests that matter most are the ones that
put them side by side: a routing optimisation is only safe while both
routes are indistinguishable from outside. The rest is about the child
failing — a helper that can stop an index run is worse than no helper,
so every way it can go wrong has to end in the same answer, slower.

The corpus here is deliberately larger than four files. Everything else
in this suite works on one or two, which means everything else in this
suite exercises the in-process path and none of this.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from wsindex.ingest import blame as blaming
from wsindex.ingest import commits as commits_module
from wsindex.ingest.commits import BLAME_WORKERS, HELPER_FROM, _blame, blame_map

REAL_RUN = subprocess.run
"""`subprocess.run` as it was before any test replaced it."""


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository with more files than `HELPER_FROM`, in two commits."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    for name in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{name}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{name}_EMAIL", "test@example.invalid")

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q", "--initial-branch=main")
    for n in range(HELPER_FROM + 2):
        (tmp_path / f"file{n}.py").write_text(f"def f{n}():\n    return {n}\n")
    git("add", "-A")
    git("commit", "-qm", "first")
    # A second commit, so the map has more than one sha in it and a test
    # asserting "which commit wrote this line" can actually be wrong.
    (tmp_path / "file0.py").write_text("def f0():\n    return 'changed'\n")
    git("add", "-A")
    git("commit", "-qm", "second")
    return tmp_path


@pytest.fixture
def tracked(repo: Path) -> list[str]:
    return sorted(path.name for path in repo.glob("*.py"))


def here(root: Path, paths: list[str]) -> dict[str, dict[int, str]]:
    """The in-process route, called directly."""
    return blaming.blame_files(root, paths, workers=BLAME_WORKERS, blame=_blame)


# --- the two routes must not be distinguishable ---------------------------


def test_the_child_and_this_process_agree(repo: Path, tracked: list[str]) -> None:
    # The whole justification for routing at all. If these can differ,
    # the optimisation is a bug that only shows up on large repositories.
    assert blame_map(repo, tracked) == here(repo, tracked)


def test_every_line_is_attributed_to_the_commit_that_wrote_it(
    repo: Path, tracked: list[str]
) -> None:
    blamed = blame_map(repo, tracked)

    untouched = set(blamed["file1.py"].values())
    edited = set(blamed["file0.py"].values())

    # `file0.py` was rewritten by the second commit, but only its second
    # line changed — so it carries both shas, and the one it shares with
    # `file1.py` is the first commit.
    assert len(untouched) == 1
    assert len(edited) == 2
    assert untouched < edited


def test_a_file_with_no_history_maps_to_nothing_either_way(repo: Path, tracked: list[str]) -> None:
    # A full pass indexes untracked files too. git refuses to blame them,
    # and that has to stay a normal answer rather than a failed run.
    (repo / "brand_new.py").write_text("x = 1\n")
    paths = [*tracked, "brand_new.py"]

    assert blame_map(repo, paths)["brand_new.py"] == {}
    assert here(repo, paths)["brand_new.py"] == {}


def test_a_path_that_is_not_utf_8_survives_the_round_trip(repo: Path, tracked: list[str]) -> None:
    """JSON crosses the pipe, and a path from git is bytes.

    `decode_path` puts lone surrogates in the string with
    `surrogateescape`; those cannot be encoded, but `json.dumps` escapes
    them to `\\udcXX` and hands back exactly what went in. The failure
    mode this guards is an index run that dies on one oddly-named file.

    Tested at the protocol rather than on disk: this filesystem refuses
    to *create* such a name ("Illegal byte sequence"), while git on a
    filesystem that allows it hands the name straight through. What has
    to survive is the pipe, and that is what is asserted — the name comes
    back as a key, with an empty map because no such file is committed
    here.
    """
    odd = "b\udcff.py"
    paths = [*tracked, odd]

    through_child = blame_map(repo, paths)

    assert odd in through_child
    assert through_child == here(repo, paths)


def test_nothing_to_blame_is_not_an_error(repo: Path) -> None:
    assert blame_map(repo, []) == {}
    assert here(repo, []) == {}


# --- the routing ----------------------------------------------------------


def test_a_small_batch_never_starts_a_process(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Below the threshold the child costs more than the forks it saves,
    # and the everyday sync — one changed file — is below it.
    monkeypatch.setattr(
        commits_module, "_in_child", lambda *a: pytest.fail("a child was started for a small batch")
    )

    assert blame_map(repo, ["file0.py"]) != {}


def test_a_large_batch_goes_to_the_child(
    repo: Path, tracked: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Asserted by behaviour, not by spying: if the in-process blamer is
    # broken and the answer still arrives, it came from the child.
    def refuse(root: Path, path: str) -> bytes | None:
        raise AssertionError("this batch should have gone to the child")

    monkeypatch.setattr(commits_module, "_blame", refuse)

    assert len(blame_map(repo, tracked)) == len(tracked)


# --- the child failing must cost time, never correctness ------------------


def test_no_interpreter_falls_back_to_this_process(
    repo: Path, tracked: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "executable", str(repo / "not-an-interpreter"))

    assert blame_map(repo, tracked) == here(repo, tracked)


def test_a_child_that_exits_badly_falls_back(
    repo: Path, tracked: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = repo / "broken_helper.py"
    broken.write_text("import sys; sys.exit(9)\n")
    monkeypatch.setattr(blaming, "__file__", str(broken))

    assert blame_map(repo, tracked) == here(repo, tracked)


def test_a_child_that_writes_nonsense_falls_back(
    repo: Path, tracked: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The failure a `check=True` alone would miss: exit 0, useless output.
    liar = repo / "lying_helper.py"
    liar.write_text("print('not json at all')\n")
    monkeypatch.setattr(blaming, "__file__", str(liar))

    assert blame_map(repo, tracked) == here(repo, tracked)


def test_a_child_that_hangs_is_given_up_on(
    repo: Path, tracked: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeper = repo / "sleeping_helper.py"
    sleeper.write_text("import time; time.sleep(30)\n")
    monkeypatch.setattr(blaming, "__file__", str(sleeper))
    monkeypatch.setattr(commits_module, "GIT_TIMEOUT", 0.2)

    assert blame_map(repo, tracked) == here(repo, tracked)


def test_a_missing_helper_file_falls_back(
    repo: Path, tracked: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(blaming, "__file__", str(repo / "gone.py"))

    assert blame_map(repo, tracked) == here(repo, tracked)


# --- the helper on its own ------------------------------------------------


def test_the_helper_answers_a_request_on_stdin(repo: Path, tracked: list[str]) -> None:
    request = json.dumps({"root": str(repo), "paths": tracked, "workers": 2, "timeout": 30.0})

    finished = subprocess.run(
        [sys.executable, blaming.__file__],
        input=request.encode("ascii"),
        capture_output=True,
        check=True,
    )

    answered = json.loads(finished.stdout)
    assert sorted(answered) == sorted(tracked)
    assert all(sha for lines in answered.values() for sha in lines.values())


def test_the_helper_imports_nothing_from_this_package() -> None:
    """The reason it is fast enough to be worth starting.

    `import wsindex.ingest` costs 30 ms of unrelated imports, measured,
    against a 21 ms bare interpreter — which would move the break-even
    from four files to nearer ten. A stray `from wsindex...` here would
    not fail any other test; it would quietly make the threshold wrong.
    """
    source = Path(blaming.__file__).read_text()

    offenders = [
        line for line in source.splitlines() if line.startswith(("import wsindex", "from wsindex"))
    ]

    assert offenders == []


def test_the_parser_reads_a_porcelain_header() -> None:
    raw = b"\n".join(
        [
            b"1111111111111111111111111111111111111111 1 1 2",
            b"\tdef f():",
            b"2222222222222222222222222222222222222222 2 2",
            b"\t    return 1",
        ]
    )

    assert blaming.parse(raw) == {1: "1" * 40, 2: "2" * 40}


def test_a_blamer_that_raises_names_the_file() -> None:
    # Through `concurrent.futures` an exception arrives with a traceback
    # and no idea which of a hundred files caused it.
    def boom(root: Path, path: str) -> bytes | None:
        raise OSError("disk went away")

    with pytest.raises(RuntimeError, match=re.escape("blaming file0.py: OSError: disk went away")):
        blaming.blame_files(Path("/tmp"), ["file0.py"], workers=1, blame=boom)


def test_the_helper_reads_stdin_and_writes_stdout(
    repo: Path,
    tracked: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`main` in this process, so its lines are measured as well as run.

    The subprocess test above proves the file really works when run by
    path — which is how it is used — but coverage cannot see into a
    child, and "82% covered" would then mean "the helper is untested" to
    every reader afterwards.
    """
    request = {"root": str(repo), "paths": tracked, "workers": 2, "timeout": 30.0}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(request)))

    assert blaming.main() == 0

    answered = json.loads(capsys.readouterr().out)
    assert sorted(answered) == sorted(tracked)


def test_the_helpers_own_git_reports_a_refusal_as_none(repo: Path) -> None:
    # The untracked case, at the level the child sees it: git exits
    # non-zero and that has to become "no history", not an exception.
    (repo / "untracked.py").write_text("z = 3\n")

    assert blaming.plain_git(repo, "untracked.py", timeout=30.0) is None


def test_the_helpers_own_git_returns_porcelain_for_a_tracked_file(repo: Path) -> None:
    raw = blaming.plain_git(repo, "file1.py", timeout=30.0)

    assert raw is not None
    assert blaming.parse(raw) != {}


# --- the rule the child exists to keep ------------------------------------


def spawns_during_index(root: Path, home: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    """How many processes an index run starts *from this process*."""
    from wsindex.config import Config, Repository
    from wsindex.embed import FakeEmbedder
    from wsindex.links import LinkStore
    from wsindex.pipeline import Pipeline
    from wsindex.store import LanceDBStore

    started = 0

    def counted(args: Any, **kw: Any) -> Any:
        nonlocal started
        if isinstance(args, list) and args and args[0] == "git":
            started += 1
        return REAL_RUN(args, **kw)

    # `REAL_RUN` from import time, not from here: this runs twice in one
    # test, and reading `subprocess.run` now would capture the previous
    # wrapper and count through a chain of them.
    monkeypatch.setattr(subprocess, "run", counted)
    Config.reset()
    config = Config.default("spawns")
    config.add_repo(Repository(id="r", path=str(root)))
    # `links` is not optional here: with no link store the pipeline skips
    # the blame pass altogether, and the first version of this measured a
    # run that never blamed anything. The falsification test below is what
    # caught it.
    with LinkStore(home / "idx") as links:
        Pipeline(
            store=LanceDBStore(uri=str(home / "db"), embedder=FakeEmbedder()),
            state_dir=home / "state",
            links=links,
        ).index()
    return started


def test_indexing_starts_a_fixed_number_of_processes_whatever_the_repo_size(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property the child exists to preserve, as a test rather than a memory.

    Not a count anybody has to keep up to date — a *shape*. Blame used to
    start one process per indexed file from the process holding the
    embedding model, and on macOS every one of those costs 2.43 ms of
    address-space teardown that no amount of threading parallelises. The
    cure moved them into a child; what must stay true is that the engine
    process starts a number of processes that does not grow with the
    repository.

    A linter, a `ripgrep`, a second git pass added to the ingest path
    would reintroduce the problem silently and pass every other test in
    this suite. This one fails.
    """
    small = spawns_during_index(repo, tmp_path / "small", monkeypatch)

    for n in range(20):
        (repo / f"extra{n}.py").write_text(f"def extra{n}():\n    return {n}\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "more"], cwd=repo, check=True, capture_output=True)

    large = spawns_during_index(repo, tmp_path / "large", monkeypatch)

    assert large == small, (
        f"indexing {20} more files started {large - small} more processes from the "
        "engine process; they belong in the blame helper's child"
    )


def test_without_the_child_that_number_grows_with_the_repository(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same measurement with the child switched off, so the guard above
    is known not to be vacuous.

    A test that cannot fail proves nothing, and this one names what the
    child actually buys: with it disabled, twenty more files mean twenty
    more processes started from the process holding the model.
    """
    monkeypatch.setattr(commits_module, "HELPER_FROM", 10**6)

    small = spawns_during_index(repo, tmp_path / "small", monkeypatch)
    for n in range(20):
        (repo / f"extra{n}.py").write_text(f"def extra{n}():\n    return {n}\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "more"], cwd=repo, check=True, capture_output=True)

    large = spawns_during_index(repo, tmp_path / "large", monkeypatch)

    # At least one more process per added file. Not exactly twenty: the
    # second run also reads a diff the first one had no commit for, and
    # what this has to establish is the *growth*, not a census.
    assert large - small >= 20
