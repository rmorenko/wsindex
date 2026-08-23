"""End-to-end pipeline tests: fake repo on disk -> FakeEmbedder -> LanceDBStore.

FakeEmbedder is deterministic, so querying with the exact text of a chunk
must return that chunk with cosine score ~1.0 — that is what makes the
pipeline assertable end to end without a real model.
"""

from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from wsindex.config import Backend, Config, Provider
from wsindex.embed.embedder import FakeEmbedder
from wsindex.model import Kind, SearchFilter
from wsindex.pipeline import _CANDIDATE_MULTIPLIER, Pipeline
from wsindex.rank.reranker import FakeReranker
from wsindex.store.lancedb import LanceDBStore

PY_TEXT = "def f():\n    return 1"

# README.md yields two markdown sections, main.py one plain chunk.
EXPECTED_CHUNKS = 3


def make_repo(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text(PY_TEXT + "\n")
    (root / "README.md").write_text("# Title\nalpha body\n## Sub\nbeta body\n")


@pytest.fixture
def config(tmp_path: Path) -> Config:
    repo_dir = tmp_path / "repo1"
    repo_dir.mkdir()
    make_repo(repo_dir)
    cfg = Config(
        name="ws",
        backend=Backend.LOCAL,
        provider=Provider.SENTENCE_TRANSFORMERS,
        model="fake",
        dim=8,
        metric="cosine",
        repos=[],
        store_uri=str(tmp_path / "db"),
        rank_enabled=False,
        rank_model="cross-encoder/ms-marco-MiniLM-L6-v2",
    )
    cfg.add_repo("repo1", path=str(repo_dir))
    return cfg


@pytest.fixture
def store(tmp_path: Path, embedder: FakeEmbedder) -> LanceDBStore:
    return LanceDBStore(uri=str(tmp_path / ".wsindex"), embedder=embedder)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def pipeline(config: Config, store: LanceDBStore) -> Pipeline:
    return Pipeline(config=config, store=store)


def test_report_counts(pipeline: Pipeline) -> None:
    report = pipeline.index()
    assert report.files == 2
    assert report.chunks == EXPECTED_CHUNKS
    assert report.written == EXPECTED_CHUNKS  # first run: everything is new
    assert report.missing_repos == ()


def test_store_search_finds_exact_chunk(pipeline: Pipeline, store: LanceDBStore) -> None:
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
    tmp_path: Path, pipeline: Pipeline, store: LanceDBStore
) -> None:
    repo2 = tmp_path / "repo2"
    repo2.mkdir()
    (repo2 / "app.py").write_text("print('two')\n")
    pipeline.config.add_repo("repo2", path=str(repo2))
    pipeline.index()
    for dataset in ("repo1", "repo2"):
        hits = store.search(dataset_name=dataset, query=PY_TEXT, k=10)
        assert hits
        assert {h.metadata["repo"] for h in hits} == {dataset}


def test_missing_repo_is_reported_not_fatal(tmp_path: Path, pipeline: Pipeline) -> None:
    pipeline.config.add_repo("ghost", path=str(tmp_path / "does-not-exist"))
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
    pipeline.config.add_repo("repo2", path=str(repo2))
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
    pipeline.config.add_repo("repo2", path=str(repo2))
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
    pipeline.config.add_repo("repo2", path=str(repo2))  # in the config, never indexed
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


def test_reranker_replaces_scores_and_reorders(config: Config, store: LanceDBStore) -> None:
    plain = Pipeline(config=config, store=store)
    plain.index()
    baseline = [h.native_id for h in plain.search(PY_TEXT, k=3)]

    ranked_pipeline = Pipeline(config=config, store=store, reranker=InvertingReranker())
    ranked = ranked_pipeline.search(PY_TEXT, k=3)

    # Reranker inverts store's cosine order — top-k reverses.
    assert [h.native_id for h in ranked] == list(reversed(baseline))
    # Scores come from the reranker, not from cosine — all in (0, 1].
    assert all(0 < h.score <= 1 for h in ranked)


def test_pipeline_fetches_k_times_multiplier_with_reranker(
    config: Config, store: LanceDBStore
) -> None:
    spied = MagicMock(wraps=store)
    Pipeline(config=config, store=spied).index()

    Pipeline(config=config, store=spied, reranker=FakeReranker()).search(PY_TEXT, k=3)
    _, kwargs = spied.search.call_args
    assert kwargs["k"] == 3 * _CANDIDATE_MULTIPLIER

    spied.reset_mock()
    Pipeline(config=config, store=spied).search(PY_TEXT, k=3)
    _, kwargs_plain = spied.search.call_args
    assert kwargs_plain["k"] == 3


# --- step 19g: repo scope + filter passthrough -----------------------------


def test_search_with_repo_narrows_dataset_list(tmp_path: Path, pipeline: Pipeline) -> None:
    repo2 = tmp_path / "repo2"
    repo2.mkdir()
    (repo2 / "app.py").write_text("print('two')\n")
    pipeline.config.add_repo("repo2", path=str(repo2))
    pipeline.index()
    # Query text is from repo1's main.py, but --repo repo2 must exclude it.
    hits = pipeline.search(PY_TEXT, k=5, repo="repo2")
    assert hits
    assert {h.metadata["repo"] for h in hits} == {"repo2"}


def test_search_with_unknown_repo_raises(pipeline: Pipeline) -> None:
    pipeline.index()
    with pytest.raises(ValueError, match="unknown repo id"):
        pipeline.search(PY_TEXT, k=5, repo="does-not-exist")


def test_search_repo_only_asks_that_dataset(config: Config, store: LanceDBStore) -> None:
    # --repo is a pipeline-layer decision: the store must only be asked
    # for the scoped dataset, not for the others.
    Pipeline(config=config, store=store).index()
    spied = MagicMock(wraps=store)
    Pipeline(config=config, store=spied).search(PY_TEXT, k=3, repo="repo1")
    called_datasets = [call.kwargs["dataset_name"] for call in spied.search.call_args_list]
    assert called_datasets == ["repo1"]


def test_search_forwards_filters_to_store(config: Config, store: LanceDBStore) -> None:
    Pipeline(config=config, store=store).index()
    spied = MagicMock(wraps=store)
    flt = SearchFilter(lang=("python",), kind=(Kind.CODE,))
    Pipeline(config=config, store=spied).search(PY_TEXT, k=3, filters=flt)
    _, kwargs = spied.search.call_args
    assert kwargs["filters"] is flt  # exact object, not rebuilt


def test_search_no_filters_forwards_none(config: Config, store: LanceDBStore) -> None:
    Pipeline(config=config, store=store).index()
    spied = MagicMock(wraps=store)
    Pipeline(config=config, store=spied).search(PY_TEXT, k=3)
    _, kwargs = spied.search.call_args
    assert kwargs["filters"] is None


def test_reranker_respects_filters(config: Config, store: LanceDBStore) -> None:
    # The reranker sees only the filtered candidates: with a filter that
    # excludes the exact-match chunk, the top hit cannot be from src/.
    Pipeline(config=config, store=store).index()
    ranked = Pipeline(config=config, store=store, reranker=FakeReranker())
    hits = ranked.search(PY_TEXT, k=3, filters=SearchFilter(kind=(Kind.DOC,)))
    assert hits  # README yields doc chunks
    assert all(h.metadata["kind"] == "doc" for h in hits)
