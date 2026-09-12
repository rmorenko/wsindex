"""Link store and the lifetime rule ADR-9 owed.

The debt this closes: a link is keyed by `chunk_id`, and a chunk id is
`sha256(text, path)` — it does not survive an edit. Without deleting a
chunk's links when the chunk goes, two things rot. The store grows edges
pointing at nothing forever, and those orphans are indistinguishable from
real dangling links, so the drift report fills with references from code
that no longer exists and stops being worth reading.

Most of what follows tests the store in isolation; the end-to-end cases
at the bottom drive a real `Pipeline.index` over a real git repo, because
the guarantee is about the two deletions happening together and only a
real run proves that.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from wsindex.config import Config, Repository
from wsindex.embed import FakeEmbedder
from wsindex.links import Link, LinkKind, LinkStore
from wsindex.model import Chunk, Kind
from wsindex.pipeline import Pipeline
from wsindex.store import LanceDBStore

Committer = Callable[[Path], None]


def link(src: str, kind: LinkKind, name: str, line: int = 1) -> Link:
    return Link(src_chunk_id=src, kind=kind, name=name, line=line)


@pytest.fixture
def links(tmp_path: Path) -> LinkStore:
    return LinkStore(tmp_path / "idx")


# --- the store on its own ------------------------------------------------


def test_links_round_trip(links: LinkStore) -> None:
    assert links.add_links([link("a", LinkKind.READS_KEY, "8080")], repo="r", path="a.py") == 1
    assert links.count() == 1


def test_adding_the_same_link_twice_writes_once(links: LinkStore) -> None:
    # Idempotent for the same reason `add_chunks` is: a re-index must not
    # double what it finds.
    one = [link("a", LinkKind.READS_KEY, "8080")]
    links.add_links(one, repo="r", path="a.py")
    assert links.add_links(one, repo="r", path="a.py") == 0
    assert links.count() == 1


def test_adding_nothing_is_not_an_error(links: LinkStore) -> None:
    assert links.add_links([], repo="r", path="a.py") == 0


def test_the_database_survives_reopening(tmp_path: Path) -> None:
    with LinkStore(tmp_path / "idx") as first:
        first.add_links([link("a", LinkKind.READS_KEY, "8080")], repo="r", path="a.py")
    with LinkStore(tmp_path / "idx") as second:
        assert second.count() == 1


# --- the drift query -----------------------------------------------------


def test_an_unanswered_reference_is_drift(links: LinkStore) -> None:
    links.add_links([link("code", LinkKind.READS_KEY, "8080")], repo="r", path="config.py")
    found = links.dangling()
    assert [(d.name, d.path) for d in found] == [("8080", "config.py")]


def test_a_reference_a_config_declares_is_not_drift(links: LinkStore) -> None:
    links.add_links([link("code", LinkKind.READS_KEY, "8000")], repo="r", path="config.py")
    links.add_links([link("cfg", LinkKind.DECLARES, "8000")], repo="r", path="compose.yml")
    assert links.dangling() == []


def test_a_declaration_alone_is_never_drift(links: LinkStore) -> None:
    # A config publishing a port nobody reads is not a problem.
    links.add_links([link("cfg", LinkKind.DECLARES, "9999")], repo="r", path="compose.yml")
    assert links.dangling() == []


def test_drift_is_ordered_for_reading(links: LinkStore) -> None:
    links.add_links(
        [link("b", LinkKind.READS_KEY, "1", line=9), link("a", LinkKind.READS_KEY, "2", line=2)],
        repo="r",
        path="z.py",
    )
    links.add_links([link("c", LinkKind.READS_KEY, "3", line=5)], repo="r", path="a.py")
    assert [d.path for d in links.dangling()] == ["a.py", "z.py", "z.py"]


# --- deletion: the debt itself -------------------------------------------


def test_deleting_a_chunk_takes_its_links(links: LinkStore) -> None:
    links.add_links(
        [link("gone", LinkKind.READS_KEY, "8080"), link("kept", LinkKind.READS_KEY, "9090")],
        repo="r",
        path="a.py",
    )
    assert links.delete_by_source(["gone"]) == 1
    assert [d.name for d in links.dangling()] == ["9090"]


def test_deleting_nothing_is_not_an_error(links: LinkStore) -> None:
    assert links.delete_by_source([]) == 0


def test_deleting_an_unknown_chunk_is_a_no_op(links: LinkStore) -> None:
    links.add_links([link("a", LinkKind.READS_KEY, "1")], repo="r", path="a.py")
    assert links.delete_by_source(["never-existed"]) == 0
    assert links.count() == 1


def test_losing_a_declaration_brings_drift_back(links: LinkStore) -> None:
    # The lifetime rule and the feature are the same mechanism: delete
    # the config that published a port and the code reading it becomes
    # dangling on the next query, with nothing else to update.
    links.add_links([link("code", LinkKind.READS_KEY, "8000")], repo="r", path="config.py")
    links.add_links([link("cfg", LinkKind.DECLARES, "8000")], repo="r", path="compose.yml")
    assert links.dangling() == []

    links.delete_by_source(["cfg"])
    assert [d.name for d in links.dangling()] == ["8000"]


# --- end to end, through a real index run --------------------------------


@pytest.fixture
def commit(monkeypatch: pytest.MonkeyPatch) -> Committer:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    for name in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{name}_NAME", "Test")
        monkeypatch.setenv(f"GIT_{name}_EMAIL", "test@example.invalid")

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
def workspace(tmp_path: Path, commit: Committer) -> tuple[Pipeline, LinkStore, Path]:
    """The 8080/8000 shape, reconstructed: code says 8080, compose says 8000."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "config.py").write_text('BASE_URL = "http://localhost:8080"\n')
    (repo / "docker-compose.yml").write_text('services:\n  app:\n    ports:\n      - "8000:7860"\n')
    commit(repo)
    config = Config.default("test")
    config.add_repo(Repository(id="r", path=str(repo)))
    store = LinkStore(tmp_path / "idx")
    pipeline = Pipeline(
        store=LanceDBStore(uri=str(tmp_path / "db"), embedder=FakeEmbedder()),
        state_dir=tmp_path / "state",
        links=store,
    )
    return pipeline, store, repo


def test_indexing_finds_the_8080_drift(
    workspace: tuple[Pipeline, LinkStore, Path],
) -> None:
    # The step-26 exam, now as production behaviour rather than a probe.
    pipeline, links, _ = workspace
    pipeline.index()
    found = links.dangling()
    assert [(d.name, d.path) for d in found] == [("8080", "config.py")]


def test_a_run_says_so_when_a_prune_turns_out_to_have_cost_something(
    workspace: tuple[Pipeline, LinkStore, Path],
    commit: Committer,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The safety net pruning is only defensible with, driven through a
    # real run. Nothing re-extracts the mentions — their files did not
    # change — so the alternative is `refs` answering with a definition
    # and no uses, which reads as a fact about the code.
    pipeline, links, repo = workspace
    pipeline.index()
    links.add_links([link("gone", LinkKind.MENTIONS, "handle_request")], repo="r", path="old.py")
    links.prune_unjoinable()
    assert links.by_name("handle_request") == []

    (repo / "server.py").write_text("def handle_request():\n    return 1\n")
    commit(repo)
    with caplog.at_level("WARNING", logger="wsindex.pipeline"):
        pipeline.index()

    assert "run a full re-index" in caplog.text
    # And the debt clears, or the warning outlives the loss and becomes
    # something people learn to scroll past.
    caplog.clear()
    with caplog.at_level("WARNING", logger="wsindex.pipeline"):
        pipeline.index()
    assert "run a full re-index" not in caplog.text


def test_fixing_the_code_clears_the_drift(
    workspace: tuple[Pipeline, LinkStore, Path], commit: Committer
) -> None:
    pipeline, links, repo = workspace
    pipeline.index()
    (repo / "config.py").write_text('BASE_URL = "http://localhost:8000"\n')
    commit(repo)

    pipeline.index()
    assert links.dangling() == []


def test_a_rewritten_file_leaves_no_orphan_links(
    workspace: tuple[Pipeline, LinkStore, Path], commit: Committer
) -> None:
    # The debt, asserted directly: every link's source must still be a
    # chunk that exists. An orphan here is indistinguishable from real
    # drift, which is what would poison the report.
    pipeline, links, repo = workspace
    pipeline.index()
    (repo / "config.py").write_text('BASE_URL = "http://localhost:8000"\n')
    commit(repo)
    pipeline.index()

    sources = {row[0] for row in links._db.execute("SELECT DISTINCT src_chunk_id FROM links")}
    live = pipeline.store.chunk_ids(dataset_name="r")
    assert sources <= live


def test_deleting_the_config_brings_drift_back_through_a_real_run(
    workspace: tuple[Pipeline, LinkStore, Path], commit: Committer
) -> None:
    # End to end, the direction that proves the two halves are one
    # mechanism: fix the code, then delete the config it now agrees with.
    pipeline, links, repo = workspace
    pipeline.index()
    (repo / "config.py").write_text('BASE_URL = "http://localhost:8000"\n')
    commit(repo)
    pipeline.index()
    assert links.dangling() == []

    (repo / "docker-compose.yml").unlink()
    commit(repo)
    pipeline.index()
    assert [d.name for d in links.dangling()] == ["8000"]


def test_a_pipeline_without_a_link_store_still_indexes(tmp_path: Path, commit: Committer) -> None:
    # Links are optional: nothing about indexing depends on them.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text('X = "http://localhost:8080"\n')
    commit(repo)
    config = Config.default("test")
    config.add_repo(Repository(id="r", path=str(repo)))
    pipeline = Pipeline(
        store=LanceDBStore(uri=str(tmp_path / "db"), embedder=FakeEmbedder()),
        state_dir=tmp_path / "state",
    )
    assert pipeline.index().written > 0


# --- what each side of the rule contributes ------------------------------


def test_a_port_in_prose_is_neither_side_of_the_drift_rule() -> None:
    # A sentence naming a port is not a claim anyone can be held to, so a
    # doc chunk is neither a reference nor a declaration. Since
    # external references arrived,
    # it does contribute a REFERENCES link for the url itself, which is a
    # different assertion entirely.
    from wsindex.ingest.link_extract import links_for
    from wsindex.model import Chunk, Kind

    doc = Chunk(
        repo="r",
        path="README.md",
        lang="markdown",
        kind=Kind.DOC,
        symbol=None,
        node_type=None,
        start_line=1,
        end_line=1,
        text="The service runs on http://localhost:8080 by default.",
    )
    kinds = {link.kind for link in links_for([doc])}
    assert LinkKind.READS_KEY not in kinds
    assert LinkKind.DECLARES not in kinds


def test_link_lines_are_file_lines_not_chunk_lines() -> None:
    # A link points a person at a file; a chunk offset would send them to
    # the wrong place in every chunk but the first.
    from wsindex.ingest.link_extract import links_for
    from wsindex.model import Chunk, Kind

    chunk = Chunk(
        repo="r",
        path="a.py",
        lang="python",
        kind=Kind.CODE,
        symbol=None,
        node_type=None,
        start_line=40,
        end_line=42,
        text='def f():\n    return "http://localhost:8080"\n',
    )
    assert [link.line for link in links_for([chunk])] == [41]


def test_a_port_key_in_a_config_is_a_declaration() -> None:
    # Not every config publishes ports the compose way. `port: 8000` in
    # a yaml or `port = 8000` in a toml declares one just as much.
    from wsindex.ingest.link_extract import links_for
    from wsindex.model import Chunk, Kind

    cfg = Chunk(
        repo="r",
        path="app.yaml",
        lang="yaml",
        kind=Kind.CONFIG,
        symbol=None,
        node_type=None,
        start_line=1,
        end_line=1,
        text="server:\n  port: 8000\n",
    )
    # The key names are declarations too, since config keys arrived; the
    # port *value* is what this case is about.
    assert (LinkKind.DECLARES, "8000") in [(link.kind, link.name) for link in links_for([cfg])]


# --- pruning what can never join -----------------------------------------


def test_a_mention_nothing_defines_is_dropped(links: LinkStore) -> None:
    # Two thirds of them, measured: 18 871 of caddyserver's 28 545
    # mentions name the standard library or a vendored package, and
    # those can never be half of a join.
    links.add_links(
        [
            link("c1", LinkKind.MENTIONS, "WriteHeader"),
            link("c1", LinkKind.MENTIONS, "ownThing", line=2),
            link("c2", LinkKind.DEFINES, "ownThing"),
        ],
        repo="r",
        path="a.go",
    )

    assert links.prune_unjoinable() == 1
    assert {edge.name for edge in links.by_name("ownThing")} == {"ownThing"}
    assert links.by_name("WriteHeader") == []


def test_a_mention_a_config_anchors_is_kept(links: LinkStore) -> None:
    # Caught by re-asking a case an earlier measurement had answered: a
    # first version of the prune asked only about DEFINES, and
    # `refs trusted_proxies` went from four uses to none. A config key
    # anchors a mention just as a definition does, and that join — a
    # json key meeting a Go identifier — is the one thing `refs` does
    # that `grep` cannot.
    links.add_links([link("c1", LinkKind.MENTIONS, "TrustedProxies")], repo="r", path="a.go")
    links.add_links([link("c2", LinkKind.DECLARES, "trusted_proxies")], repo="r", path="a.json")

    assert links.prune_unjoinable() == 0
    assert len(links.by_name("trusted_proxies")) == 2


def test_pruning_matches_across_spellings(links: LinkStore) -> None:
    # The mention is `MaxRetries` and the declaration is `max_retries`.
    # Comparing raw names here would delete exactly the edges the
    # normalised join exists to keep.
    links.add_links(
        [link("c1", LinkKind.MENTIONS, "MaxRetries"), link("c2", LinkKind.DEFINES, "max_retries")],
        repo="r",
        path="a.go",
    )

    assert links.prune_unjoinable() == 0


def test_a_definition_arriving_later_is_reported_not_swallowed(links: LinkStore) -> None:
    # The one way pruning becomes a wrong answer: a repo joins the
    # workspace and defines a name whose uses were already dropped. The
    # files holding them have not changed, so nothing re-extracts them,
    # and `refs` would show a definition with no uses — which reads as a
    # fact about the code rather than as a hole in the index.
    links.add_links([link("c1", LinkKind.MENTIONS, "SharedThing")], repo="a", path="a.go")
    links.prune_unjoinable()

    links.add_links([link("c2", LinkKind.DEFINES, "SharedThing")], repo="b", path="b.go")

    assert links.rescued(["sharedthing"]) == {"sharedthing"}


def test_a_settled_debt_stops_being_reported(links: LinkStore) -> None:
    # Otherwise the warning outlives the loss: a full re-index puts the
    # mentions back, and a warning that never clears is one people learn
    # to scroll past.
    links.add_links([link("c1", LinkKind.MENTIONS, "SharedThing")], repo="a", path="a.go")
    links.prune_unjoinable()
    links.add_links(
        [link("c2", LinkKind.DEFINES, "SharedThing"), link("c3", LinkKind.MENTIONS, "SharedThing")],
        repo="b",
        path="b.go",
    )

    links.prune_unjoinable()

    assert links.rescued(["sharedthing"]) == set()


def test_pruning_leaves_a_database_from_before_normalisation_alone(tmp_path: Path) -> None:
    # Those rows have NULL in `norm`, every comparison against it is
    # false, and a prune that read that as "nothing defines it" would
    # empty the table of an index it was only asked to shrink.
    store = LinkStore(tmp_path / "idx")
    store.add_links([link("c1", LinkKind.MENTIONS, "Whatever")], repo="r", path="a.go")
    store._db.execute("UPDATE links SET norm = NULL")
    store._db.commit()

    assert store.prune_unjoinable() == 0
    assert store.count() == 1
    store.close()


def test_only_mentions_are_pruned(links: LinkStore) -> None:
    # A commit nobody references, a url in a doc, a port a config
    # publishes — none of those is half of a pair, and all of them are
    # answers on their own.
    links.add_links(
        [
            link("c1", LinkKind.DECLARES, "8000"),
            link("c2", LinkKind.REFERENCES, "https://example.invalid"),
            link("c3", LinkKind.BLAMED_BY, "3964bb7"),
        ],
        repo="r",
        path="compose.yml",
    )

    assert links.prune_unjoinable() == 0
    assert links.count() == 3


# --- settings, and the spellings they are asked about under ---------------


def config(text: str, *, path: str = "app.yaml", line: int = 1) -> Chunk:
    return Chunk(
        repo="r",
        path=path,
        lang="yaml",
        kind=Kind.CONFIG,
        symbol=None,
        node_type=None,
        start_line=line,
        end_line=line,
        text=text,
    )


def test_a_config_declares_its_keys_and_not_only_its_ports() -> None:
    # The hole this closes: `_declarations` matched ports and nothing
    # else, so `max_retries: 3` produced no link and 8% of a workspace's
    # config keys were known to the store — 10 of 120 sampled on
    # caddyserver. Ports were never the interesting half, only the half
    # a regular expression could reach.
    from wsindex.ingest.link_extract import links_for

    found = links_for([config("server:\n  max_retries: 3\n  port: 8000\n")])

    assert {(link.kind, link.name) for link in found} == {
        (LinkKind.DECLARES, "server"),
        (LinkKind.DECLARES, "max_retries"),
        (LinkKind.DECLARES, "port"),
        (LinkKind.DECLARES, "8000"),
    }


def test_a_short_key_is_a_setting_even_though_a_short_symbol_is_not() -> None:
    # Code needs a length guard because `get` and `run` are not names
    # worth an edge. A config key is a name by grammar, and `ssl`, `env`
    # and `dsn` are settings people ask about.
    from wsindex.ingest.link_extract import links_for

    assert {link.name for link in links_for([config("ssl: true\nenv: prod\n")])} == {"ssl", "env"}


def test_a_setting_is_found_under_the_other_half_s_spelling(links: LinkStore) -> None:
    # The one thing `refs` can do that `grep` cannot: `rg -w max_retries`
    # misses `MaxRetries`, and `rg -i` misses it too, because they differ
    # by more than case. Asking someone to guess which half of their own
    # system spells it which way is asking them to know the answer first.
    from wsindex.ingest.link_extract import links_for

    links.add_links(links_for([config("max_retries: 3")]), repo="r", path="app.yaml")
    links.add_links(links_for([code("if attempt < cfg.MaxRetries {")]), repo="r", path="retry.go")

    found = {(edge.kind, edge.name) for edge in links.by_name("max_retries")}
    assert found == {(LinkKind.DECLARES, "max_retries"), (LinkKind.MENTIONS, "MaxRetries")}


def test_the_spelling_that_was_asked_for_comes_first(links: LinkStore) -> None:
    # A variant arriving above an exact hit reads as the exact answer,
    # and the reader has no way to tell without checking the file.
    from wsindex.ingest.link_extract import links_for

    links.add_links(links_for([code("x := MaxRetries")]), repo="r", path="a.go")
    links.add_links(links_for([code("y := max_retries")]), repo="r", path="b.go")

    assert [edge.name for edge in links.by_name("max_retries")] == ["max_retries", "MaxRetries"]


def test_a_database_from_before_normalisation_is_still_searchable(tmp_path: Path) -> None:
    # `norm` is NULL in rows written by an older wsindex, and a query
    # that only asked `norm = ?` would answer "no links named that" for
    # an index full of them. Silence is the worst failure here: it reads
    # as a fact about the code.
    store = LinkStore(tmp_path / "idx")
    store.add_links([link("c1", LinkKind.DECLARES, "max_retries")], repo="r", path="app.yaml")
    store._db.execute("UPDATE links SET norm = NULL")
    store._db.commit()

    assert [edge.name for edge in store.by_name("max_retries")] == ["max_retries"]
    store.close()


# --- the name pair -------------------------------------------------------


def code(text: str, *, symbol: str | None = None, path: str = "a.go", line: int = 1) -> Chunk:
    return Chunk(
        repo="r",
        path=path,
        lang="go",
        kind=Kind.CODE,
        symbol=symbol,
        node_type=None,
        start_line=line,
        end_line=line,
        text=text,
    )


def test_a_definition_and_a_use_of_the_same_name_meet(links: LinkStore) -> None:
    # The join this pair exists for, and the thing the config pair could
    # almost never do: across five workspaces exactly three names of
    # 6 896 had both a READS_KEY and a DECLARES. Both sides are stored
    # unresolved, because the extractor sees one file and the definition
    # is usually in another repository entirely.
    from wsindex.ingest.link_extract import links_for

    definition = links_for([code("func ServeHTTP() {}", symbol="ServeHTTP")])
    use = links_for([code("h.ServeHTTP(w, r)", path="b.go")])
    links.add_links(definition, repo="r", path="a.go")
    links.add_links(use, repo="r", path="b.go")

    found = {(edge.kind, edge.path) for edge in links.by_name("ServeHTTP")}
    assert found == {(LinkKind.DEFINES, "a.go"), (LinkKind.MENTIONS, "b.go")}


def test_a_chunk_does_not_mention_the_name_it_defines() -> None:
    # Otherwise every definition answers its own query and `refs` reports
    # the definition twice, once under each label.
    from wsindex.ingest.link_extract import links_for

    found = links_for([code("func ServeHTTP() {\n  return ServeHTTP\n}", symbol="ServeHTTP")])

    assert [edge.kind for edge in found] == [LinkKind.DEFINES]


def test_a_keyword_is_not_a_name_worth_an_edge() -> None:
    # The filter that makes this affordable, and it is a measured choice
    # rather than taste. Storing every token of four characters or more
    # kept all 810 of caddyserver's cross-file names but cost 186 100
    # edges — nineteen per chunk, 1.9M rows on a 100k-chunk workspace.
    # Requiring an internal word boundary keeps 664 of the 810 for
    # 31 839 edges, and it needs no per-language stop-list precisely
    # because keywords are single lowercase words in all sixteen
    # grammars.
    from wsindex.ingest.link_extract import links_for

    found = links_for([code("return errors.New(string(value))\nreadConfig()\n")])

    assert [edge.name for edge in found] == ["readConfig"]


def test_a_name_is_recorded_once_at_its_first_line() -> None:
    # A variable used nine times inside its own function is one fact, not
    # nine, and the nine tell a reader nothing they cannot see once the
    # file is open.
    from wsindex.ingest.link_extract import links_for

    found = links_for([code("x := maxRetries\ny := maxRetries\n", line=40)])

    assert [(edge.name, edge.line) for edge in found] == [("maxRetries", 40)]


def test_each_sort_of_occurrence_is_told_apart() -> None:
    # Why this is recorded rather than filtered on: measured on
    # caddyserver, 51% of mentions are calls and 42% are other real code
    # — receivers, struct literals, field accesses. A call graph would
    # throw that 42% away to remove the 6% that is comment, import and
    # string. Labelling keeps both readers.
    from wsindex.ingest.link_extract import links_for
    from wsindex.links import Occurrence

    found = links_for(
        [
            code(
                "import loggingPkg\n"
                "// see requestCount for why\n"
                'label := "metricName"\n'
                "total := requestTotal\n"
                "computeSum(total)\n"
            )
        ]
    )

    assert {edge.name: edge.via for edge in found} == {
        "loggingPkg": Occurrence.IMPORT,
        "requestCount": Occurrence.COMMENT,
        "metricName": Occurrence.STRING,
        "requestTotal": Occurrence.CODE,
        "computeSum": Occurrence.CALL,
    }


def test_the_best_occurrence_wins_not_the_first() -> None:
    # A docstring precedes the body, so the first occurrence of a name is
    # often prose about it while the call is further down. Keeping the
    # first would send a reader to the comment and label the edge
    # `comment`, which is the least useful answer to "where is this
    # used".
    from wsindex.ingest.link_extract import links_for
    from wsindex.links import Occurrence

    found = links_for([code("# wraps parseHeader\nx = 1\nparseHeader(x)\n", line=10)])

    assert [(edge.via, edge.line) for edge in found] == [(Occurrence.CALL, 12)]


def test_the_occurrence_survives_a_round_trip(links: LinkStore) -> None:
    # It is a column, and a column that is written and not read is the
    # ordinary way a field like this quietly becomes decorative.
    from wsindex.ingest.link_extract import links_for
    from wsindex.links import Occurrence

    links.add_links(links_for([code("computeSum(total)")]), repo="r", path="a.go")

    assert [edge.via for edge in links.by_name("computeSum")] == [Occurrence.CALL]


def test_a_kind_with_nothing_to_say_about_placement_says_nothing(links: LinkStore) -> None:
    # `via` is an attribute of MENTIONS. A DEFINES edge is the definition
    # wherever it sits, and a default of `code` would read as a measured
    # claim rather than as an absence.
    from wsindex.ingest.link_extract import links_for

    links.add_links(links_for([code("func doThing() {}", symbol="doThing")]), repo="r", path="a.go")

    assert [edge.via for edge in links.by_name("doThing")] == [None]


def test_a_method_is_stored_under_its_bare_name() -> None:
    # The store holds `Cls.method` for `search --symbol`, but a call site
    # spells it `method`, and a qualified name here would join with
    # nothing.
    from wsindex.ingest.link_extract import links_for

    found = links_for([code("def run_once(self): pass", symbol="Scheduler.run_once")])

    assert [(edge.kind, edge.name) for edge in found] == [(LinkKind.DEFINES, "run_once")]


def test_a_database_from_before_a_column_existed_still_reads(tmp_path: Path) -> None:
    # `CREATE TABLE IF NOT EXISTS` does nothing to a table that already
    # exists, so a links.db written before `url` arrived kept its old
    # shape and every read failed with `no such column`. Found the hard
    # way, on an index built one step earlier.
    import sqlite3

    (tmp_path / "idx").mkdir()
    old = sqlite3.connect(tmp_path / "idx" / "links.db")
    old.execute(
        "CREATE TABLE links (src_chunk_id TEXT NOT NULL, kind TEXT NOT NULL, "
        "name TEXT NOT NULL, line INTEGER NOT NULL, dst_chunk_id TEXT, "
        "repo TEXT NOT NULL, path TEXT NOT NULL, "
        "PRIMARY KEY (src_chunk_id, kind, name, line))"
    )
    old.execute("INSERT INTO links VALUES ('c1', 'reads_key', '8080', 1, NULL, 'r', 'a.py')")
    old.commit()
    old.close()

    with LinkStore(tmp_path / "idx") as store:
        edges = store.by_name("8080")
        assert [(edge.name, edge.url) for edge in edges] == [("8080", None)]
        # And it still accepts rows that use the new column.
        store.add_links(
            [
                Link(
                    src_chunk_id="c2",
                    kind=LinkKind.REFERENCES,
                    name="#7",
                    line=2,
                    url="https://x.invalid/7",
                )
            ],
            repo="r",
            path="b.md",
        )
        assert store.by_name("#7")[0].url == "https://x.invalid/7"


def test_by_name_is_the_inverted_index(links: LinkStore) -> None:
    links.add_links([link("code", LinkKind.READS_KEY, "8000")], repo="r", path="a.py")
    links.add_links([link("cfg", LinkKind.DECLARES, "8000")], repo="r", path="compose.yml")
    links.add_links([link("other", LinkKind.READS_KEY, "9999")], repo="r", path="b.py")

    found = links.by_name("8000")
    assert {edge.kind for edge in found} == {LinkKind.READS_KEY, LinkKind.DECLARES}
    assert {edge.path for edge in found} == {"a.py", "compose.yml"}


def test_by_name_on_an_unknown_name_is_empty(links: LinkStore) -> None:
    assert links.by_name("nothing") == []


def test_out_of_reads_links_leaving_a_chunk(links: LinkStore) -> None:
    links.add_links(
        [link("c1", LinkKind.READS_KEY, "8000"), link("c1", LinkKind.BLAMED_BY, "abc1234", line=2)],
        repo="r",
        path="a.py",
    )
    assert len(links.out_of(["c1"])) == 2
    assert [e.name for e in links.out_of(["c1"], kind=LinkKind.BLAMED_BY)] == ["abc1234"]


def test_out_of_nothing_is_empty(links: LinkStore) -> None:
    assert links.out_of([]) == []
