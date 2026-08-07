"""End-to-end pipeline tests: fake repo on disk -> FakeEmbedder -> LocalStore.

FakeEmbedder is deterministic, so querying with the exact text of a chunk
must return that chunk with cosine score ~1.0 — that is what makes the
pipeline assertable end to end without a real model.
"""

from pathlib import Path

import pytest

from wsindex.config import Backend, Config
from wsindex.embed.embedder import FakeEmbedder
from wsindex.pipeline import index
from wsindex.store.local import LocalStore

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
        model="fake",
        dim=8,
        base_url="",
        metric="cosine",
        repos=[],
    )
    cfg.add_repo("repo1", str(repo_dir))
    return cfg


@pytest.fixture
def store(tmp_path: Path) -> LocalStore:
    return LocalStore(tmp_path / ".wsindex")


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


def test_report_counts(config: Config, store: LocalStore, embedder: FakeEmbedder) -> None:
    report = index(config, store, embedder)
    assert report.files == 2
    assert report.chunks == EXPECTED_CHUNKS
    assert report.written == EXPECTED_CHUNKS  # first run: everything is new
    assert report.missing_repos == ()


def test_search_finds_exact_chunk(
    config: Config, store: LocalStore, embedder: FakeEmbedder
) -> None:
    index(config, store, embedder)
    query = embedder.embed([PY_TEXT])[0]
    hits = store.search("repo1", query, k=3)
    assert hits[0].metadata["path"] == "src/main.py"  # rel_path, POSIX, no tmp leak
    assert hits[0].score == pytest.approx(1.0)


def test_second_run_writes_nothing(
    config: Config, store: LocalStore, embedder: FakeEmbedder
) -> None:
    index(config, store, embedder)
    report = index(config, store, embedder)
    # Files are walked and chunked again, but dedup by chunk id writes nothing.
    assert report.files == 2
    assert report.chunks == EXPECTED_CHUNKS
    assert report.written == 0


def test_two_repos_get_isolated_datasets(
    tmp_path: Path, config: Config, store: LocalStore, embedder: FakeEmbedder
) -> None:
    repo2 = tmp_path / "repo2"
    repo2.mkdir()
    (repo2 / "app.py").write_text("print('two')\n")
    config.add_repo("repo2", str(repo2))
    index(config, store, embedder)
    for dataset in ("repo1", "repo2"):
        hits = store.search(dataset, embedder.embed([PY_TEXT])[0], k=10)
        assert hits
        assert {h.metadata["repo"] for h in hits} == {dataset}


def test_missing_repo_is_reported_not_fatal(
    tmp_path: Path, config: Config, store: LocalStore, embedder: FakeEmbedder
) -> None:
    config.add_repo("ghost", str(tmp_path / "does-not-exist"))
    report = index(config, store, embedder)
    assert report.missing_repos == ("ghost",)
    assert report.written == EXPECTED_CHUNKS  # repo1 still indexed
