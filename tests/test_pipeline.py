"""End-to-end pipeline tests: fake repo on disk -> FakeEmbedder -> LocalStore.

FakeEmbedder is deterministic, so querying with the exact text of a chunk
must return that chunk with cosine score ~1.0 — that is what makes the
pipeline assertable end to end without a real model.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from wsindex.config import Repository
from wsindex.embed import FakeEmbedder
from wsindex.pipeline import Pipeline
from wsindex.store import LocalStore

PY_TEXT = "def f():\n    return 1"

# README.md yields two markdown sections, main.py one plain chunk.
EXPECTED_CHUNKS = 3


def make_repo(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text(PY_TEXT + "\n")
    (root / "README.md").write_text("# Title\nalpha body\n## Sub\nbeta body\n")


@pytest.fixture
def repo1(tmp_path: Path) -> Repository:
    repo_dir = tmp_path / "repo1"
    repo_dir.mkdir()
    make_repo(repo_dir)
    return Repository(id="repo1", path=str(repo_dir))


@pytest.fixture
def store(tmp_path: Path, embedder: FakeEmbedder) -> LocalStore:
    return LocalStore(tmp_path / ".wsindex", embedder=embedder)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def pipeline(repo1: Repository, store: LocalStore) -> Pipeline:
    return Pipeline(repos=[repo1], metric="cosine", store=store)


def test_report_counts(pipeline: Pipeline) -> None:
    report = pipeline.index()
    assert report.files == 2
    assert report.chunks == EXPECTED_CHUNKS
    assert report.written == EXPECTED_CHUNKS  # first run: everything is new
    assert report.missing_repos == ()


def test_store_search_finds_exact_chunk(pipeline: Pipeline, store: LocalStore) -> None:
    pipeline.index()
    hits = store.search(dataset_name="repo1", query=PY_TEXT, k=3)
    assert hits[0].metadata["path"] == "src/main.py"  # rel_path, POSIX, no tmp leak
    assert hits[0].score == pytest.approx(1.0)


def test_second_run_writes_nothing(pipeline: Pipeline) -> None:
    pipeline.index()
    report = pipeline.index()
    # Files are walked and chunked again, but dedup by chunk id writes nothing.
    assert report.files == 2
    assert report.chunks == EXPECTED_CHUNKS
    assert report.written == 0


def test_two_repos_get_isolated_datasets(
    tmp_path: Path, pipeline: Pipeline, store: LocalStore
) -> None:
    repo2 = tmp_path / "repo2"
    repo2.mkdir()
    (repo2 / "app.py").write_text("print('two')\n")
    pipeline = replace(pipeline, repos=[*pipeline.repos, Repository(id="repo2", path=str(repo2))])
    pipeline.index()
    for dataset in ("repo1", "repo2"):
        hits = store.search(dataset_name=dataset, query=PY_TEXT, k=10)
        assert hits
        assert {h.metadata["repo"] for h in hits} == {dataset}


def test_missing_repo_is_reported_not_fatal(tmp_path: Path, pipeline: Pipeline) -> None:
    pipeline = replace(
        pipeline,
        repos=[*pipeline.repos, Repository(id="ghost", path=str(tmp_path / "does-not-exist"))],
    )
    report = pipeline.index()
    assert report.missing_repos == ("ghost",)
    assert report.written == EXPECTED_CHUNKS  # repo1 still indexed


def test_search_exact_text_wins(pipeline: Pipeline) -> None:
    pipeline.index()
    hits = pipeline.search(PY_TEXT, k=3)
    assert hits[0].metadata["path"] == "src/main.py"
    assert hits[0].score == pytest.approx(1.0)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_merges_across_repos(tmp_path: Path, pipeline: Pipeline) -> None:
    repo2 = tmp_path / "repo2"
    repo2.mkdir()
    (repo2 / "app.py").write_text("print('two')\n")
    pipeline = replace(pipeline, repos=[*pipeline.repos, Repository(id="repo2", path=str(repo2))])
    pipeline.index()
    hits = pipeline.search("print('two')", k=3)
    assert hits[0].metadata["repo"] == "repo2"
    assert hits[0].score == pytest.approx(1.0)


def test_search_tie_keeps_config_repo_order(tmp_path: Path, pipeline: Pipeline) -> None:
    # The same file in both repos: two hits with identical scores; the
    # stable merge must keep the config repo order.
    repo2 = tmp_path / "repo2"
    repo2.mkdir()
    make_repo(repo2)
    pipeline = replace(pipeline, repos=[*pipeline.repos, Repository(id="repo2", path=str(repo2))])
    pipeline.index()
    hits = pipeline.search(PY_TEXT, k=2)
    assert [h.score for h in hits] == [pytest.approx(1.0)] * 2
    assert [h.metadata["repo"] for h in hits] == ["repo1", "repo2"]


def test_search_cuts_to_k_after_merge(pipeline: Pipeline) -> None:
    pipeline.index()
    assert len(pipeline.search(PY_TEXT, k=1)) == 1


def test_search_skips_unindexed_repo(tmp_path: Path, pipeline: Pipeline) -> None:
    pipeline.index()  # repo1 indexed
    repo2 = tmp_path / "repo2"
    repo2.mkdir()
    (repo2 / "app.py").write_text("print('two')\n")
    # repo2 is listed but never indexed — search must skip it silently.
    pipeline = replace(pipeline, repos=[*pipeline.repos, Repository(id="repo2", path=str(repo2))])
    hits = pipeline.search(PY_TEXT, k=5)
    assert hits  # no crash, repo1 still answers
    assert {h.metadata["repo"] for h in hits} == {"repo1"}
