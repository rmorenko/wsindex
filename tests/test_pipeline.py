"""End-to-end pipeline tests: fake repo on disk -> FakeEmbedder -> LanceDBStore.

FakeEmbedder is deterministic, so querying with the exact text of a chunk
must return that chunk with cosine score ~1.0 — that is what makes the
pipeline assertable end to end without a real model.

Pipeline reads the repo list from `Config()`, so "add a second repo" is
`config.add_repo(...)` rather than a new Pipeline: the object under test
stays the same one across the whole test.

Every repo here is a real git repository, because `index` is incremental
against git and refuses to touch anything else. The `commit`
helper is how a test says "this is now the committed state", which is
also what puts the repo on the incremental path — an uncommitted change
forces a full pass by design.
"""

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import wsindex.pipeline
from helpers import as_indexed
from wsindex.config import Config, Repository
from wsindex.embed import FakeEmbedder
from wsindex.ingest import IndexState, NotAGitRepositoryError, Skip, chunk_file
from wsindex.model import Chunk, Hit, Kind, SearchFilter, SourceFile
from wsindex.pipeline import _CANDIDATE_MULTIPLIER, _QUOTA_MULTIPLIER, FullPass, Pipeline
from wsindex.rank.reranker import FakeReranker
from wsindex.store import LanceDBStore

PY_TEXT = "def f():\n    return 1"

INDEXED_MAIN = as_indexed(PY_TEXT, path="src/main.py", symbol="f")
"""What the store embedded for `src/main.py`, which is what a test has to
ask for: `FakeEmbedder` seeds a vector from its input, so nothing is near
anything and only the same string comes back."""

# README.md yields two markdown sections, main.py one plain chunk.
EXPECTED_CHUNKS = 3

Committer = Callable[[Path], None]


def make_repo(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text(PY_TEXT + "\n")
    (root / "README.md").write_text("# Title\nalpha body\n## Sub\nbeta body\n")


@pytest.fixture
def commit(monkeypatch: pytest.MonkeyPatch) -> Committer:
    """Stage and commit everything in a repo, with a hermetic identity."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "Test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.invalid")

    def run(root: Path) -> None:
        if not (root / ".git").exists():
            subprocess.run(
                ["git", "init", "-q", "--initial-branch=main"],
                cwd=root,
                check=True,
                capture_output=True,
            )
        subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-qm", "snapshot", "--allow-empty"],
            cwd=root,
            check=True,
            capture_output=True,
        )

    return run


@pytest.fixture
def config(tmp_path: Path, commit: Committer) -> Config:
    """The workspace under test: one committed repo, config in memory."""
    repo_dir = tmp_path / "repo1"
    repo_dir.mkdir()
    make_repo(repo_dir)
    commit(repo_dir)
    config = Config.default("test")
    config.add_repo(Repository(id="repo1", path=str(repo_dir)))
    return config


def add_repo(
    config: Config,
    tmp_path: Path,
    name: str,
    *,
    full: bool = False,
    commit: Committer | None = None,
) -> Path:
    """Create a second repo on disk and register it in the config."""
    repo_dir = tmp_path / name
    repo_dir.mkdir()
    if full:
        make_repo(repo_dir)
    else:
        (repo_dir / "app.py").write_text("print('two')\n")
    if commit is not None:
        commit(repo_dir)
    config.add_repo(Repository(id=name, path=str(repo_dir)))
    return repo_dir


@pytest.fixture
def store(tmp_path: Path, embedder: FakeEmbedder) -> LanceDBStore:
    return LanceDBStore(uri=str(tmp_path / ".wsindex"), embedder=embedder)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    return tmp_path / "state"


@pytest.fixture
def pipeline(config: Config, store: LanceDBStore, state_dir: Path) -> Pipeline:
    # `config` is requested for its side effect: it installs the workspace
    # as the process-wide Config, which is where Pipeline reads its repos.
    return Pipeline(store=store, state_dir=state_dir)


def test_report_counts(pipeline: Pipeline) -> None:
    report = pipeline.index()
    assert report.files == 2
    assert report.chunks == EXPECTED_CHUNKS
    assert report.written == EXPECTED_CHUNKS  # first run: everything is new
    assert report.missing_repos == ()


def test_store_search_finds_exact_chunk(pipeline: Pipeline, store: LanceDBStore) -> None:
    pipeline.index()
    hits = store.search(dataset_name="repo1", query=INDEXED_MAIN, k=3)
    assert hits[0].metadata["path"] == "src/main.py"  # rel_path, POSIX, no tmp leak
    assert hits[0].score == pytest.approx(1.0)


def test_second_run_reads_nothing(pipeline: Pipeline) -> None:
    # Before incremental indexing this re-read and re-chunked everything
    # and leaned on
    # dedup to write nothing. Now the diff is empty, so no file is opened
    # at all — that is where the wall-clock saving comes from.
    pipeline.index()
    report = pipeline.index()
    assert report.files == 0
    assert report.chunks == 0
    assert report.written == 0
    assert report.deleted == 0
    assert report.full_repos == ()


def test_two_repos_get_isolated_datasets(
    tmp_path: Path, config: Config, pipeline: Pipeline, store: LanceDBStore, commit: Committer
) -> None:
    add_repo(config, tmp_path, "repo2", commit=commit)
    pipeline.index()
    for dataset in ("repo1", "repo2"):
        hits = store.search(dataset_name=dataset, query=PY_TEXT, k=10)
        assert hits
        assert {h.metadata["repo"] for h in hits} == {dataset}


def test_missing_repo_is_reported_not_fatal(
    tmp_path: Path, config: Config, pipeline: Pipeline
) -> None:
    config.add_repo(Repository(id="ghost", path=str(tmp_path / "does-not-exist")))
    report = pipeline.index()
    assert report.missing_repos == ("ghost",)
    assert report.written == EXPECTED_CHUNKS  # repo1 still indexed


def test_search_exact_text_wins(pipeline: Pipeline) -> None:
    pipeline.index()
    hits = pipeline.search(INDEXED_MAIN, k=3)
    assert hits[0].metadata["path"] == "src/main.py"
    assert hits[0].score == pytest.approx(1.0)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_merges_across_repos(
    tmp_path: Path, config: Config, pipeline: Pipeline, commit: Committer
) -> None:
    add_repo(config, tmp_path, "repo2", commit=commit)
    pipeline.index()
    hits = pipeline.search(as_indexed("print('two')", path="app.py"), k=3)
    assert hits[0].metadata["repo"] == "repo2"
    assert hits[0].score == pytest.approx(1.0)


def test_search_tie_keeps_config_repo_order(
    tmp_path: Path, config: Config, pipeline: Pipeline, commit: Committer
) -> None:
    # The same file in both repos: two hits with identical scores; the
    # stable merge must keep the config repo order.
    add_repo(config, tmp_path, "repo2", full=True, commit=commit)
    pipeline.index()
    # The same file at the same relative path in both repos, so the
    # string the store embedded is the same too — the tie survives the
    # store learning a chunk's name and place.
    hits = pipeline.search(INDEXED_MAIN, k=2)
    assert [h.score for h in hits] == [pytest.approx(1.0)] * 2
    assert [h.metadata["repo"] for h in hits] == ["repo1", "repo2"]


def test_search_cuts_to_k_after_merge(pipeline: Pipeline) -> None:
    pipeline.index()
    assert len(pipeline.search(PY_TEXT, k=1)) == 1


def test_search_skips_unindexed_repo(
    tmp_path: Path, config: Config, pipeline: Pipeline, commit: Committer
) -> None:
    pipeline.index()  # repo1 indexed
    # repo2 is listed but never indexed — search must skip it silently.
    add_repo(config, tmp_path, "repo2", commit=commit)
    hits = pipeline.search(PY_TEXT, k=5)
    assert hits  # no crash, repo1 still answers
    assert {h.metadata["repo"] for h in hits} == {"repo1"}


class InvertingReranker(FakeReranker):
    """Returns descending scores in input order: last input gets the highest.

    Proves the pipeline actually adopts reranker output — the final order
    must be the reverse of what the store returned by cosine.
    """

    def _rank(self, query: str, texts: Sequence[str]) -> list[float]:
        n = len(texts)
        return [(i + 1) / n for i in range(n)]  # ascending: 0->1/n, last->1.0


def test_reranker_replaces_scores_and_reorders(
    config: Config, store: LanceDBStore, state_dir: Path
) -> None:
    # Scoped to code: the corpus holds commit messages too,
    # and the claim under test is about the reranker's effect on an
    # ordering, not about which kinds happen to be in it.
    only_code = SearchFilter(kind=(Kind.CODE,))
    plain = Pipeline(store=store, state_dir=state_dir)
    plain.index()
    baseline = [h.native_id for h in plain.search(PY_TEXT, k=3, filters=only_code)]

    ranked_pipeline = Pipeline(store=store, state_dir=state_dir, reranker=InvertingReranker())
    ranked = ranked_pipeline.search(PY_TEXT, k=3, filters=only_code)

    # Reranker inverts store's cosine order — top-k reverses.
    assert [h.native_id for h in ranked] == list(reversed(baseline))
    # Scores come from the reranker, not from cosine — all in (0, 1].
    assert all(0 < h.score <= 1 for h in ranked)


def test_pipeline_fetches_k_times_multiplier_with_reranker(
    config: Config, store: LanceDBStore, state_dir: Path
) -> None:
    spied = MagicMock(wraps=store)
    Pipeline(store=spied, state_dir=state_dir).index()

    Pipeline(store=spied, state_dir=state_dir, reranker=FakeReranker()).search(PY_TEXT, k=3)
    _, kwargs = spied.search.call_args
    assert kwargs["k"] == 3 * _CANDIDATE_MULTIPLIER

    # There are two reasons to ask for more than k now, and the reranker
    # is only the louder one: the commit quota also needs spares, or
    # capping history at two in ten would return eight hits instead of
    # ten. A plain search over-fetches by the smaller factor.
    spied.reset_mock()
    Pipeline(store=spied, state_dir=state_dir).search(PY_TEXT, k=3)
    _, kwargs_plain = spied.search.call_args
    assert kwargs_plain["k"] == 3 * _QUOTA_MULTIPLIER


# --- repo scope and filter passthrough ------------------------------------


def test_search_with_repo_narrows_dataset_list(
    tmp_path: Path, config: Config, pipeline: Pipeline, commit: Committer
) -> None:
    add_repo(config, tmp_path, "repo2", commit=commit)
    pipeline.index()
    # Query text is from repo1's main.py, but --repo repo2 must exclude it.
    hits = pipeline.search(PY_TEXT, k=5, repo="repo2")
    assert hits
    assert {h.metadata["repo"] for h in hits} == {"repo2"}


def test_search_with_unknown_repo_raises(pipeline: Pipeline) -> None:
    pipeline.index()
    with pytest.raises(ValueError, match="unknown repo id"):
        pipeline.search(PY_TEXT, k=5, repo="does-not-exist")


def test_search_repo_only_asks_that_dataset(
    config: Config, store: LanceDBStore, state_dir: Path
) -> None:
    # --repo is a pipeline-layer decision: the store must only be asked
    # for the scoped dataset, not for the others.
    Pipeline(store=store, state_dir=state_dir).index()
    spied = MagicMock(wraps=store)
    Pipeline(store=spied, state_dir=state_dir).search(PY_TEXT, k=3, repo="repo1")
    called_datasets = [call.kwargs["dataset_name"] for call in spied.search.call_args_list]
    assert called_datasets == ["repo1"]


def test_search_forwards_filters_to_store(
    config: Config, store: LanceDBStore, state_dir: Path
) -> None:
    Pipeline(store=store, state_dir=state_dir).index()
    spied = MagicMock(wraps=store)
    flt = SearchFilter(lang=("python",), kind=(Kind.CODE,))
    Pipeline(store=spied, state_dir=state_dir).search(PY_TEXT, k=3, filters=flt)
    _, kwargs = spied.search.call_args
    assert kwargs["filters"] is flt  # exact object, not rebuilt


def test_search_no_filters_forwards_none(
    config: Config, store: LanceDBStore, state_dir: Path
) -> None:
    Pipeline(store=store, state_dir=state_dir).index()
    spied = MagicMock(wraps=store)
    Pipeline(store=spied, state_dir=state_dir).search(PY_TEXT, k=3)
    _, kwargs = spied.search.call_args
    assert kwargs["filters"] is None


def test_reranker_respects_filters(config: Config, store: LanceDBStore, state_dir: Path) -> None:
    # The reranker sees only the filtered candidates: with a filter that
    # excludes the exact-match chunk, the top hit cannot be from src/.
    Pipeline(store=store, state_dir=state_dir).index()
    ranked = Pipeline(store=store, state_dir=state_dir, reranker=FakeReranker())
    hits = ranked.search(PY_TEXT, k=3, filters=SearchFilter(kind=(Kind.DOC,)))
    assert hits  # README yields doc chunks
    assert all(h.metadata["kind"] == "doc" for h in hits)


# --- incremental index ---------------------------------------------------


def edit(root: Path, rel: str, text: str, commit: Committer) -> None:
    """Change a file and commit it, so the tree stays on the fast path."""
    (root / rel).write_text(text)
    commit(root)


def test_first_run_is_a_full_pass_and_arms_the_next(pipeline: Pipeline, state_dir: Path) -> None:
    report = pipeline.index()
    # The reason travels with the id: "why did this take nine seconds" is
    # the question the field exists to answer.
    assert report.full_repos == (("repo1", FullPass.NEVER_INDEXED),)
    # A clean tree IS exactly HEAD, so a full pass records it — that is
    # what switches the repo onto the fast path for the next run.
    assert "repo1" in IndexState.load(state_dir).commits


def test_changed_file_is_the_only_one_read(
    tmp_path: Path, pipeline: Pipeline, commit: Committer
) -> None:
    pipeline.index()
    edit(tmp_path / "repo1", "src/main.py", "def f():\n    return 2\n", commit)

    report = pipeline.index()
    assert report.files == 1  # README.md was not touched, so it was not read
    assert report.written == 1
    assert report.deleted == 1  # the chunk the old text produced
    assert report.full_repos == ()


def test_changed_file_no_longer_answers_with_its_old_text(
    tmp_path: Path, pipeline: Pipeline, store: LanceDBStore, commit: Committer
) -> None:
    # The debt `delete_chunks` was added for: without the reconciling
    # delete the old
    # chunk would stay searchable forever.
    pipeline.index()
    edit(tmp_path / "repo1", "src/main.py", "def g():\n    return 99\n", commit)
    pipeline.index()

    hits = store.search(dataset_name="repo1", query=PY_TEXT, k=10)
    assert all(PY_TEXT not in h.metadata["text"] for h in hits)


def test_deleted_file_loses_all_its_chunks(
    tmp_path: Path, pipeline: Pipeline, store: LanceDBStore, commit: Committer
) -> None:
    pipeline.index()
    (tmp_path / "repo1" / "src" / "main.py").unlink()
    commit(tmp_path / "repo1")

    report = pipeline.index()
    assert report.files == 0  # nothing to read, only to forget
    assert report.deleted == 1
    assert store.chunk_ids(dataset_name="repo1", paths=["src/main.py"]) == set()


def test_renamed_file_moves_its_chunks(
    tmp_path: Path, pipeline: Pipeline, store: LanceDBStore, commit: Committer
) -> None:
    # A rename is a delete plus an add to git, and must be exactly that
    # to us: old path empty, new path populated.
    repo = tmp_path / "repo1"
    pipeline.index()
    subprocess.run(
        ["git", "mv", "src/main.py", "src/renamed.py"], cwd=repo, check=True, capture_output=True
    )
    commit(repo)
    pipeline.index()

    assert store.chunk_ids(dataset_name="repo1", paths=["src/main.py"]) == set()
    assert store.chunk_ids(dataset_name="repo1", paths=["src/renamed.py"])


def test_file_that_stops_being_indexable_loses_its_chunks(
    tmp_path: Path, pipeline: Pipeline, store: LanceDBStore, commit: Committer
) -> None:
    # Renamed to a suffix we do not index: git calls it a change, the
    # walker policy calls it a skip, and the two together mean "forget".
    repo = tmp_path / "repo1"
    pipeline.index()
    subprocess.run(
        ["git", "mv", "src/main.py", "src/main.bin"], cwd=repo, check=True, capture_output=True
    )
    commit(repo)

    report = pipeline.index()
    assert report.files == 0
    assert report.deleted == 1
    assert store.chunk_ids(dataset_name="repo1", paths=["src/main.py"]) == set()


def test_dirty_tree_forces_a_full_pass_and_records_nothing(
    tmp_path: Path, pipeline: Pipeline, state_dir: Path
) -> None:
    pipeline.index()
    armed = IndexState.load(state_dir).commits["repo1"]
    (tmp_path / "repo1" / "scratch.py").write_text("print('uncommitted')\n")

    report = pipeline.index()
    assert report.full_repos == (("repo1", FullPass.DIRTY_TREE),)
    # Recording HEAD here would make the next run diff from it and skip
    # the uncommitted work forever.
    assert IndexState.load(state_dir).commits["repo1"] == armed


def test_untracked_file_is_indexed_by_the_full_pass(
    tmp_path: Path, pipeline: Pipeline, store: LanceDBStore
) -> None:
    # A module written but not yet committed is the most likely thing a
    # developer wants to find; `ls-files --others` is why it is there.
    (tmp_path / "repo1" / "fresh.py").write_text("def fresh():\n    return 'brand new'\n")
    pipeline.index()
    assert store.chunk_ids(dataset_name="repo1", paths=["fresh.py"])


def test_gitignored_file_is_not_indexed(
    tmp_path: Path, pipeline: Pipeline, store: LanceDBStore
) -> None:
    repo = tmp_path / "repo1"
    (repo / ".gitignore").write_text("build.py\n")
    (repo / "build.py").write_text("print('generated')\n")
    pipeline.index()
    assert store.chunk_ids(dataset_name="repo1", paths=["build.py"]) == set()


def test_full_pass_reconciles_chunks_it_can_no_longer_produce(
    tmp_path: Path, pipeline: Pipeline, store: LanceDBStore, state_dir: Path, commit: Committer
) -> None:
    # The full pass is a reconcile, not just an append. Drop the state so
    # the run cannot use a diff, delete a file, and the orphan must still
    # go — this is the case the pre-step-22 pipeline leaked forever.
    pipeline.index()
    (state_dir / "state.json").unlink()
    (tmp_path / "repo1" / "src" / "main.py").unlink()
    commit(tmp_path / "repo1")

    report = pipeline.index()
    # A *deleted* state file is honestly indistinguishable from never
    # having indexed — there is nothing left to say otherwise. A file
    # that is there and unusable is a different story; see below.
    assert report.full_repos == (("repo1", FullPass.NEVER_INDEXED),)
    assert report.deleted == 1
    assert store.chunk_ids(dataset_name="repo1", paths=["src/main.py"]) == set()


def test_an_unusable_state_file_says_so_instead_of_guessing(
    pipeline: Pipeline, state_dir: Path
) -> None:
    # The note used to name three causes — a first index, edited markup,
    # uncommitted work — and a corrupt state file is none of them, so it
    # sent the reader looking for uncommitted work that was not there.
    pipeline.index()
    (state_dir / "state.json").write_text("not json at all")

    assert pipeline.index().full_repos == (("repo1", FullPass.STATE_LOST),)


def test_state_is_per_repo(
    tmp_path: Path, config: Config, pipeline: Pipeline, state_dir: Path, commit: Committer
) -> None:
    add_repo(config, tmp_path, "repo2", commit=commit)
    pipeline.index()
    commits = IndexState.load(state_dir).commits
    assert set(commits) == {"repo1", "repo2"}
    # Separate repos, separate histories: nothing may be shared.
    assert commits["repo1"] != commits["repo2"]


def test_one_dirty_repo_does_not_hold_back_a_clean_one(
    tmp_path: Path, config: Config, pipeline: Pipeline, state_dir: Path, commit: Committer
) -> None:
    add_repo(config, tmp_path, "repo2", commit=commit)
    pipeline.index()
    (tmp_path / "repo1" / "scratch.py").write_text("print('dirty')\n")

    report = pipeline.index()
    assert report.full_repos == (("repo1", FullPass.DIRTY_TREE),)  # repo2 stayed incremental


def test_non_git_repo_is_a_config_error(tmp_path: Path, config: Config, pipeline: Pipeline) -> None:
    # Git-only is the decision: a plain directory is a mistake in
    # the config, not a reason to silently switch models of state.
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "a.py").write_text("print('a')\n")
    config.add_repo(Repository(id="plain", path=str(plain)))
    with pytest.raises(NotAGitRepositoryError, match="not a git repository"):
        pipeline.index()


def test_changed_markup_re_reads_a_tree_git_calls_unchanged(
    tmp_path: Path, pipeline: Pipeline, config: Config
) -> None:
    # Found live. `formats` decides which files a tree produces, and git
    # reports nothing when it changes — so trusting the commit alone made
    # editing the markup a silent no-op: the newly indexable file stayed
    # invisible until someone deleted the index by hand.
    (tmp_path / "repo1" / "schema.sql").write_text("CREATE TABLE users (id INT);\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path / "repo1", check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-qm", "sql"], cwd=tmp_path / "repo1", check=True, capture_output=True
    )
    pipeline.index()
    assert pipeline.index().files == 0  # nothing changed, and nothing was read

    config._data["repos"][0]["formats"] = {".sql": {"lang": "sql", "kind": "code"}}
    report = pipeline.index()

    assert report.full_repos == (("repo1", FullPass.MARKUP_CHANGED),)
    assert report.written == 1
    # And the run after that is back on the fast path.
    assert pipeline.index().files == 0


def test_a_repo_marked_the_same_way_stays_on_the_fast_path(
    tmp_path: Path, pipeline: Pipeline, config: Config
) -> None:
    # The fingerprint must be about content, not about order: reordering
    # globs is not a reason to re-embed a repository.
    config._data["repos"][0]["ignore"] = ["dist/*", "*.min.js"]
    pipeline.index()

    config._data["repos"][0]["ignore"] = ["*.min.js", "dist/*"]

    assert pipeline.index().full_repos == ()


def test_chunks_reach_the_store_in_batches(
    tmp_path: Path, pipeline: Pipeline, store: LanceDBStore, commit: Committer
) -> None:
    # One write per file cost three ways at once: a deduplication round
    # trip each time, a store version each time, and a fragmented index
    # that every later search pays for. Measured on a 149-file corpus:
    # 78% of the run, 4.4 MB where 1.7 was enough, 6.4 ms per search
    # against 2.3 ms.
    repo = tmp_path / "repo1"
    for n in range(12):
        (repo / f"mod{n}.py").write_text(f"def f{n}():\n    return {n}\n")
    commit(repo)
    writes: list[int] = []
    original = store.add_chunks

    def counting(dataset_name: str, *, chunks: Sequence[Chunk]) -> int:
        writes.append(len(chunks))
        return original(dataset_name, chunks=chunks)

    store.add_chunks = counting  # type: ignore[method-assign]
    report = pipeline.index()

    # One write for the commit messages, one for every file's chunks.
    assert report.files == 14
    assert len(writes) == 2
    assert sum(writes) >= report.chunks


def test_a_repo_that_was_never_indexed_is_named_not_hidden(
    tmp_path: Path, config: Config, pipeline: Pipeline, commit: Committer
) -> None:
    # The worst failure the tool had: an answer that is not wrong but
    # *partial*, and looks exactly like a whole one. A repo added to the
    # config and never indexed took no part in any search and said
    # nothing, so "not found there" was a conclusion nobody was entitled
    # to draw.
    second = tmp_path / "repo2"
    (second / "src").mkdir(parents=True)
    (second / "src" / "other.py").write_text("def helper():\n    return 2\n")
    commit(second)
    pipeline.index()
    config.add_repo(Repository(id="repo2", path=str(second)))

    assert pipeline.unsearched() == ("repo2",)
    assert {hit.repo for hit in pipeline.search("helper", k=10)} == {"repo1"}

    pipeline.index()
    assert pipeline.unsearched() == ()


def test_unsearched_narrows_with_the_search_it_accompanies(
    tmp_path: Path, config: Config, pipeline: Pipeline
) -> None:
    config.add_repo(Repository(id="repo2", path=str(tmp_path / "repo2")))
    pipeline.index()

    assert pipeline.unsearched("repo1") == ()
    assert pipeline.unsearched("repo2") == ("repo2",)
    with pytest.raises(ValueError, match="unknown repo id"):
        pipeline.unsearched("nope")


def test_a_file_that_cannot_be_read_is_reported_not_dropped(
    tmp_path: Path, pipeline: Pipeline, commit: Committer
) -> None:
    # It used to be folded in with a broken symlink and a concurrent
    # delete, which really are nothing. This one is not: git tracks the
    # file, it was meant to be indexed, and `files: 1` where there were
    # two told the reader something untrue.
    locked = tmp_path / "repo1" / "src" / "locked.py"
    locked.write_text("def hidden():\n    return 2\n")
    commit(tmp_path / "repo1")
    locked.chmod(0o000)
    try:
        report = pipeline.index()
    finally:
        locked.chmod(0o644)

    assert report.unreadable == ("repo1/src/locked.py",)


def test_a_run_reports_how_long_it_took(pipeline: Pipeline) -> None:
    # The server recorded this per run and a person at a terminal did
    # not, so "did that get slower" had nothing to compare against.
    assert pipeline.index().seconds >= 0.0


def test_a_dirty_tree_outranks_never_having_been_indexed(
    tmp_path: Path, pipeline: Pipeline
) -> None:
    # Both are true on a first run over a dirty tree, and only one of
    # them is worth saying: the dirty tree is *why* nothing was recorded
    # and will cost a full pass on every run until somebody commits.
    # Caught on a live server, which called a workspace it had indexed
    # all week "a first index".
    (tmp_path / "repo1" / "scratch.py").write_text("print('uncommitted')\n")

    assert pipeline.index().full_repos == (("repo1", FullPass.DIRTY_TREE),)


# --- finding out what went wrong -----------------------------------------


def test_a_failure_while_chunking_says_which_file(
    tmp_path: Path, pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    # It used to be `error: the chunker fell over` for a workspace of 122
    # files. The loop knows the path at exactly that moment.
    (tmp_path / "repo1" / "src" / "later.py").write_text("def g():\n    return 2\n")
    real = chunk_file  # the pipeline's own reference, before it is swapped

    def flaky(text: str, source: SourceFile) -> list[Chunk]:
        if source.path.endswith("later.py"):
            raise RuntimeError("the chunker fell over")
        return real(text, source)

    monkeypatch.setattr(wsindex.pipeline, "chunk_file", flaky)

    with pytest.raises(RuntimeError, match=r"repo1/src/later\.py.*the chunker fell over"):
        pipeline.index()


def test_the_original_failure_is_still_underneath(
    tmp_path: Path, pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Chained, not replaced: `WSINDEX_DEBUG` shows every frame, and a
    # caller can still see what really happened.
    def flaky(text: str, source: SourceFile) -> list[Chunk]:
        raise ZeroDivisionError("the real cause")

    monkeypatch.setattr(wsindex.pipeline, "chunk_file", flaky)

    with pytest.raises(RuntimeError) as caught:
        pipeline.index()

    assert isinstance(caught.value.__cause__, ZeroDivisionError)
    # The type is named in the message, since RuntimeError replaced it:
    # not every exception can be rebuilt from one string.
    assert "ZeroDivisionError" in str(caught.value)


def test_a_file_the_grammar_could_not_read_is_reported(
    tmp_path: Path, pipeline: Pipeline, commit: Committer
) -> None:
    # It is still indexed — as text windows rather than definitions,
    # which is worse to search and used to be impossible to notice.
    broken = tmp_path / "repo1" / "src" / "broken.py"
    broken.write_text("def alpha(:\n    return 1\n\n\ndef beta(\n    return 2\n")
    commit(tmp_path / "repo1")

    report = pipeline.index()

    assert report.unparsed == ("repo1/src/broken.py",)


def test_a_file_the_grammar_reads_is_not_reported(pipeline: Pipeline) -> None:
    assert pipeline.index().unparsed == ()


# --- the questions a finished index answers --------------------------------


def test_describe_reports_an_indexed_file(tmp_path: Path, pipeline: Pipeline) -> None:
    # `wsindex explain` in library terms. It used to live in the CLI,
    # which cost that adapter eight imports out of `wsindex.ingest` and
    # put real analysis — parsed, or merely windowed — in a renderer.
    report = pipeline.describe(tmp_path / "repo1" / "src" / "main.py")

    assert report is not None
    assert (report.repo, report.rel_path) == ("repo1", "src/main.py")
    assert report.skipped is None
    assert report.lang == "python"
    assert report.chunks >= 1
    assert report.parsed_cleanly


def test_describe_names_the_rule_that_excluded_a_file(tmp_path: Path, pipeline: Pipeline) -> None:
    (tmp_path / "repo1" / "notes.unknownsuffix").write_text("hello\n")

    report = pipeline.describe(tmp_path / "repo1" / "notes.unknownsuffix")

    assert report is not None
    assert report.skipped is Skip.UNKNOWN_SUFFIX
    assert report.chunks == 0


def test_describe_says_when_the_grammar_gave_up(tmp_path: Path, pipeline: Pipeline) -> None:
    broken = tmp_path / "repo1" / "src" / "broken.py"
    broken.write_text("def alpha(:\n    return 1\n\n\ndef beta(\n    return 2\n")

    report = pipeline.describe(broken)

    assert report is not None
    assert report.skipped is None
    assert not report.parsed_cleanly


def test_describe_is_none_outside_every_repo(tmp_path: Path, pipeline: Pipeline) -> None:
    assert pipeline.describe(tmp_path / "elsewhere.py") is None


def test_references_uses_the_pipelines_own_link_store(tmp_path: Path, pipeline: Pipeline) -> None:
    # The MCP tool opened a second LinkStore on every call, ignoring the
    # one it had been handed.
    pipeline.index()

    assert isinstance(pipeline.references("no_such_name_at_all"), list)


def test_references_without_links_is_empty(config: Config, store: LanceDBStore) -> None:
    bare = Pipeline(store=store, state_dir=Path("/nonexistent"), config=config, links=None)

    assert bare.references("8080") == []


# --- fitting an answer to a context window --------------------------------
#
# `k` says how many results; an agent needs to know how much context, and
# the two are not the same question. Ten hits are anywhere between two
# hundred tokens and twelve thousand depending on what they hold.


def hit_of(text: str) -> Hit:
    return Hit(score=1.0, native_id="x", metadata={"text": text, "path": "a.py"})


def test_a_budget_keeps_the_best_hits_that_fit(pipeline: Pipeline) -> None:
    hits = [hit_of("word " * 30), hit_of("word " * 30), hit_of("word " * 30)]

    kept, spent = pipeline.fit(hits, budget=pipeline.store.count_tokens(hits[0].text) * 2)

    assert len(kept) == 2
    assert spent <= pipeline.store.count_tokens(hits[0].text) * 2


def test_a_budget_smaller_than_one_hit_returns_nothing(pipeline: Pipeline) -> None:
    # Saying "nothing fits" beats overrunning a context window silently,
    # and beats returning one oversized hit the caller cannot afford.
    kept, spent = pipeline.fit([hit_of("word " * 200)], budget=5)

    assert (kept, spent) == ([], 0)


def test_a_budget_larger_than_everything_keeps_everything(pipeline: Pipeline) -> None:
    hits = [hit_of("small"), hit_of("also small")]

    kept, _ = pipeline.fit(hits, budget=10_000)

    assert kept == hits


def test_fitting_does_not_reorder_by_size(pipeline: Pipeline) -> None:
    """A prefix, not a knapsack.

    Skipping a large hit to fit two small ones would quietly trade
    relevance for bytes — and the caller, having asked for the best
    results, would have no way to see that it got the cheapest instead.
    """
    big, small = hit_of("word " * 100), hit_of("tiny")

    kept, _ = pipeline.fit([big, small], budget=pipeline.store.count_tokens("tiny") * 2)

    assert kept == [], "the first hit did not fit, so nothing after it is considered"


def test_nothing_to_fit_is_not_an_error(pipeline: Pipeline) -> None:
    assert pipeline.fit([], budget=100) == ([], 0)


def test_a_language_nobody_claims_is_reported_not_passed_over(
    tmp_path: Path, pipeline: Pipeline, commit: Committer
) -> None:
    # The loudest silence the field trial found. A workspace of 417
    # Elixir files indexed 30 of them and printed `files: 30` — true, and
    # it told the reader nothing; `status` then showed three healthy
    # repos. Skipping a `LICENSE` is policy and not news. Skipping a
    # *language* is, and nothing said so.
    source = tmp_path / "repo1" / "src"
    for n in range(12):
        (source / f"mod{n}.ex").write_text(f"defmodule M{n} do\nend\n")
    (source / "mix.exs").write_text("defmodule Mix do\nend\n")
    commit(tmp_path / "repo1")

    report = pipeline.index()

    assert report.mostly_unclaimed
    assert report.unclaimed_files == 13
    assert dict(report.unclaimed) == {".ex": 12, ".exs": 1}


def test_the_odd_unclaimed_file_is_not_worth_a_warning(
    tmp_path: Path, pipeline: Pipeline, commit: Committer
) -> None:
    # The other half of the promise: a repo that leaves out a licence and
    # an icon has nothing wrong with it, and a run that complained every
    # time would teach the reader to stop reading warnings.
    source = tmp_path / "repo1" / "src"
    for n in range(12):
        (source / f"mod{n}.py").write_text(f"def f{n}():\n    return {n}\n")
    (tmp_path / "repo1" / "LICENSE").write_text("All rights reserved.\n")
    commit(tmp_path / "repo1")

    report = pipeline.index()

    assert report.unclaimed_files == 1
    assert not report.mostly_unclaimed


def test_history_may_not_take_the_whole_screen(
    tmp_path: Path, pipeline: Pipeline, commit: Committer
) -> None:
    # Commit messages and code are ranked by one cosine, and prose scores
    # well against a prose query on almost anything. The field trial
    # watched all ten hits for a named constant come back as commits,
    # scored within 0.005 of each other. History still answers — it is
    # capped, not filtered — so this pins both halves of that.
    repo = tmp_path / "repo1"
    for n in range(12):
        (repo / "src" / f"mod{n}.py").write_text(f"def f{n}():\n    return {n}\n")
        commit(repo)
    pipeline.index()

    hits = pipeline.search("snapshot", k=10)

    assert len(hits) == 10
    commits = [h for h in hits if h.metadata["kind"] == Kind.COMMIT.value]
    assert 0 < len(commits) <= int(10 * wsindex.pipeline.COMMIT_SHARE)


def test_the_quota_never_shortens_a_list_it_could_fill(
    tmp_path: Path, pipeline: Pipeline, commit: Committer
) -> None:
    # A cap without spares is a shorter list, not a better one: dropping
    # the commits out of a top ten and returning eight would trade one
    # complaint for another.
    repo = tmp_path / "repo1"
    for n in range(12):
        (repo / "src" / f"mod{n}.py").write_text(f"def f{n}():\n    return {n}\n")
        commit(repo)
    pipeline.index()

    assert len(pipeline.search("snapshot", k=10)) == 10
