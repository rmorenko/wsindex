"""Indexing and search pipeline: repos in, Hits out.

The pipeline sees only the VectorStore contract and works in plain text —
embedding is the store's private business. The concrete backend is built
at the edge (the CLI composition root) and injected through the
constructor; which repositories to index and with which metric comes from
`Config()`, read at call time. One process serves one workspace, so
threading those two values through the composition root only to hand them
back unchanged was ceremony.

This module is the engine and nothing else. What a run *produces* — its
tallies, its report, the rule that decides why a repo was read whole,
the shapes an answer comes back in — lives in `wsindex.run`, and is
re-exported from here so that no caller had to move when the two
separated. They separated when this file reached four roles and 992
lines, which is the threshold its own docstring had named one review
earlier.

`index` is incremental against git. Both of its paths — the
full pass and the incremental one — end in the same two lines: add the
chunks the working tree currently produces, then delete every stored
chunk in scope that it did not. Only the scope differs (a few changed
paths, or the whole dataset), which is why the reconciling delete is
written once. That also fixes the debt the full pass carried before
the store used to only ever add, so a file that shrank or vanished
left its old chunks in the index forever.
"""

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

from wsindex.config import Config, Repository
from wsindex.ingest import (
    IndexState,
    RepoDiff,
    WalkedFile,
    chunk_file,
    diff_since,
    examine,
    has_uncommitted_changes,
)
from wsindex.ingest.commits import blame_links, blame_map, commit_chunks, read_commits
from wsindex.ingest.link_extract import links_for
from wsindex.links import Edge, LinkKind, LinkStore
from wsindex.model import Chunk, Hit, SearchFilter, SourceFile
from wsindex.rank.reranker import Reranker
from wsindex.run import (
    _WRITE_BATCH,
    Authorship,
    Definition,
    FileReport,
    FullPass,
    IndexReport,
    Reference,
    _blaming,
    _History,
    _selection,
    _Tally,
    _Totals,
    _unparsed,
    _why_full,
    _Written,
)
from wsindex.stats import SearchLog
from wsindex.store import VectorStore

_CANDIDATE_MULTIPLIER = 4

log = logging.getLogger(__name__)
"""Silent unless somebody attaches a handler; `wsindex serve` does.
What is logged here is what an operator asks about afterwards — which
repo was read, how much of it, how long, and what could not be read at
all. The CLI says the same things in its own words to a person who is
watching; a log is for the reader who was not."""


@dataclass(frozen=True, kw_only=True)
class Pipeline:
    """The wired system: a store, plus whatever the current Config says.

    Frozen on purpose: a Pipeline is a bundle of dependencies, not state —
    nothing may accumulate between calls.

    Attributes:
        store: Any VectorStore backend; the pipeline never looks behind
            the contract.
        config: The workspace this pipeline indexes. A parameter rather
            than a `Config()` call inside the methods: the dependency is
            real either way, and a constructor that does not mention it
            is a constructor that lies. `Config` is a singleton, so the
            default is the same object the rest of the process sees — and
            a repo added at runtime is still picked up, because it is
            added to that object.
        state_dir: Where `index` keeps the per-repo "last indexed commit"
            file. Supplied by the composition root because it is a
            location, not a policy — the same reason the store gets its
            uri from there (see `wsindex.cli.build_pipeline`).
        reranker: Optional second stage of the search funnel. Present
            means `search` over-fetches candidates and re-scores them;
            None means the store's own ranking is the answer.
        stats: Where searches are recorded, or None to record nothing.
            Local by construction and by rule — see `wsindex.stats`.
        links: Where code-to-config edges are recorded, or None to skip
            link extraction entirely. When present, `index` writes the
            links a file yields and — this is the part that matters —
            deletes the links of every chunk it deletes (ADR-9).
    """

    store: VectorStore
    state_dir: Path
    config: Config = field(default_factory=Config)
    reranker: Reranker | None = None
    links: LinkStore | None = None
    stats: SearchLog | None = None

    def index(self, *, progress: Callable[[str], None] | None = None) -> IndexReport:
        """Index every repo into its own dataset (dataset name = repo id).

        Incremental against git: a repo goes down the fast path when it
        has been indexed before *and* its working tree is clean. Then only
        the files git reports as changed are read, and the chunks they no
        longer produce are deleted.

        Args:
            progress: Called with each repo id as that repo is reached,
                so a caller can show that silence is work. None keeps the
                run silent, which is what a pipe wants.

        A dirty working tree forces a full pass, and the run records no
        new commit for that repo. This is not pessimism, it is the only
        honest answer: a diff between two commits cannot see an
        uncommitted edit or an untracked file, but the full pass lists
        them (`ls-files --others`) and indexes both, so trusting the diff
        would leave the index describing a tree that never existed.
        Recording HEAD anyway would make the *next* run skip those same
        invisible changes forever.

        Decisions fixed here: files are read with errors="replace" so a
        stray non-UTF-8 file cannot abort the run; a repo whose directory
        does not exist goes to `missing_repos` and is skipped (an existing
        repo with zero indexable files is NOT missing).

        The new commit is recorded per repo, right after that repo's
        chunks are in the store — not once at the end. A crash halfway
        through a five-repo workspace must not cost the four that
        finished, and must not claim the one that did not.

        Returns:
            Totals across all repos; see IndexReport field docs.

        Raises:
            NotAGitRepositoryError: A configured repo is not a git
                repository root. Git-only is a decision: a
                fallback to plain walking would mean two models of
                state, so this is a config error with a message.
        """
        started = time.monotonic()
        config = self.config
        state = IndexState.load(self.state_dir)
        tally = _Tally()
        for repo in config.repos:
            # The engine says *who* is being read; what to draw with that
            # is the caller's business (see `wsindex.ui`). A plain
            # callable rather than an event system: one caller, one fact.
            if progress is not None:
                progress(repo.id)
            root = Path(repo.path)
            if not root.is_dir():
                log.warning("skipping %s: %s does not exist", repo.id, root)
                tally.missing.append(repo.id)
                continue
            self.store.create_dataset(dataset_name=repo.id, metric=config.metric)
            diff, dirty = self._plan(repo, root=root, state=state)
            if diff.full:
                reason = _why_full(repo, state=state, dirty=dirty)
                log.info("full pass for %s: %s", repo.id, reason)
                tally.full.append((repo.id, reason))
            if diff.full or diff.changed or diff.deleted:
                totals = self._index_repo(repo, root=root, diff=diff, config=config)
                tally.add(repo.id, totals)
                log.info(
                    "indexed %s: %d files, %d chunks, %d written, %d deleted",
                    repo.id,
                    totals.files,
                    totals.chunks,
                    totals.written,
                    totals.deleted,
                )
                for path in totals.unreadable:
                    log.warning("could not read %s/%s", repo.id, path)
            # Nothing moved and nothing to reconcile: no read, no chunking,
            # no store round trip. That is the whole point of the step.
            if not dirty:
                # A clean tree is exactly `diff.head`, whether we got here
                # by a delta or by re-reading everything — so a first full
                # pass is what switches this repo onto the fast path. A
                # dirty tree records nothing: we indexed content that no
                # commit describes, and claiming HEAD would make the next
                # run skip those same changes forever.
                state = state.with_commit(repo.id, diff.head, markup=repo.markup_key)
                state.save(self.state_dir)
        log.info(
            "index finished in %.2fs: %d files, %d chunks",
            time.monotonic() - started,
            tally.files,
            tally.chunks,
        )
        return tally.report(seconds=round(time.monotonic() - started, 2))

    def _plan(self, repo: Repository, *, root: Path, state: IndexState) -> tuple[RepoDiff, bool]:
        """Work out what to read for one repo, and whether HEAD describes it.

        Two separate questions, easy to conflate:

        - *Can a delta be trusted?* Only with a commit to diff from and a
          clean tree. Answered by `diff.full`, which the diff itself
          reports — a `since` that no longer resolves silently yields a
          full listing, and only `diff_since` knows that happened.
        - *May HEAD be recorded as indexed?* Whenever the tree is clean,
          full pass or not. This is what puts a freshly indexed repo onto
          the fast path for the next run.
        - *Is the markup the same one that produced the index?* A commit
          says nothing about which files the config selected from it.

        Returns:
            The diff to apply, and whether the working tree is dirty.
        """
        # Tracked edits, staged files and untracked files alike: any of
        # them makes the working tree differ from every commit, so no
        # commit-to-commit diff can describe what we are about to index.
        dirty = has_uncommitted_changes(root)
        # A third question, and the one a live run found missing: *is the
        # policy the same?* Editing `formats` changes which files this
        # tree produces while git reports nothing at all, so trusting the
        # commit alone made a markup change a silent no-op.
        remarked = state.markup.get(repo.id) != repo.markup_key
        since = None if dirty or remarked else state.commits.get(repo.id)
        return diff_since(root, since=since), dirty

    def _index_repo(
        self, repo: Repository, *, root: Path, diff: RepoDiff, config: Config
    ) -> _Totals:
        """Index what changed in one repo, then forget what it no longer holds.

        Four steps, in the only order they work in: decide what to read,
        note what the store holds now, write, subtract. Reading the
        stored ids *before* writing is what makes the last step mean
        "what was here when we started" — after the write the new ids
        would cancel out and the intent would be invisible.

        Args:
            repo: The repo being indexed; its id names the dataset.
            root: Repository root.
            diff: What to look at. `diff.full` widens the reconciling
                delete from "the paths listed here" to "the whole
                dataset", which is what makes a full pass also clean up
                files that vanished while nobody was watching.
            config: The workspace, for its reference templates.

        Returns:
            Totals for this repo.
        """
        picked = _selection(repo, root=root, changed=diff.changed, deleted=diff.deleted)
        scope = (
            None
            if diff.full
            else [*(walked.rel_path for walked in picked.indexable), *picked.forget]
        )
        stored = self.store.chunk_ids(dataset_name=repo.id, paths=scope)

        history = self._index_commits(repo, root=root, since=diff.since, config=config)
        files = self._index_files(
            repo, root=root, walked=picked.indexable, history=history, config=config
        )
        deleted = self._forget(repo, stale=sorted(stored - files.ids - history.ids))
        return _Totals(
            files=files.files,
            chunks=files.chunks,
            written=files.written,
            deleted=deleted,
            commits=history.written,
            unreadable=tuple(picked.unreadable),
            unparsed=files.unparsed,
            unclaimed=tuple(picked.unclaimed.most_common()),
        )

    def _index_commits(
        self, repo: Repository, *, root: Path, since: str | None, config: Config
    ) -> _History:
        """Index the repo's commit messages, and note where each one landed.

        Before the files, because blame edges point at these chunk ids
        and `git log` over a whole history costs milliseconds.
        """
        commits = read_commits(root, since=since, limit=config.max_commits)
        messages = commit_chunks(commits, repo=repo.id)
        # Keyed off the chunk's symbol rather than zipping: `commit_chunks`
        # drops commits with an empty message, so the two lists are not
        # guaranteed to line up.
        by_short = {message.symbol: message.id for message in messages}
        written = self.store.add_chunks(dataset_name=repo.id, chunks=messages)
        if self.links is not None and messages:
            # A commit message is where a ticket gets named, so the
            # outward references live here more than anywhere.
            self.links.add_links(
                links_for(messages, references=config.references),
                repo=repo.id,
                path="commits",
            )
        return _History(
            written=written,
            # Commit chunks are never stale: a commit is immutable, so
            # the chunk it produced can only be re-derived identically.
            # They still count as fresh, or a full pass — whose `stored`
            # covers the whole dataset — would reap every one of them.
            ids={message.id for message in messages},
            by_sha={c.sha: by_short[c.short] for c in commits if c.short in by_short},
        )

    def _index_files(
        self,
        repo: Repository,
        *,
        root: Path,
        walked: list[WalkedFile],
        history: _History,
        config: Config,
    ) -> _Written:
        """Chunk and link every file worth reading, writing in batches.

        A write per file is what this used to do, and it cost three ways
        at once. Each `add_chunks` asks the store which ids it already
        holds, so 149 files meant 149 round trips — 78% of an indexing
        run, more than chunking and writing together. Each also opens a
        Lance version, so the index carried 149 of them and took 4.4 MB
        where 1.7 was enough. And a fragmented index is slower to read:
        6.4 ms per search against 2.3 ms after compaction.

        Batching by chunk count rather than by repo keeps the memory
        bound a constant: a whole repo in flight is ~125 MB at 100k
        chunks, `_WRITE_BATCH` is a few megabytes.
        """
        totals = _Written()
        batch: list[Chunk] = []
        # Every blame at once rather than one per file in turn: they are
        # independent processes, and waiting for them one after another
        # was 83% of an indexing run.
        blames = (
            blame_map(root, [entry.rel_path for entry in walked]) if self.links is not None else {}
        )
        for entry in walked:
            source = SourceFile(repo=repo.id, path=entry.rel_path, lang=entry.lang, kind=entry.kind)
            with _blaming(repo.id, entry.rel_path):
                text = (root / entry.rel_path).read_text(encoding="utf-8", errors="replace")
                chunks: list[Chunk] = chunk_file(text, source)
                totals = totals.read(
                    chunks=len(chunks),
                    ids={chunk.id for chunk in chunks},
                    path=entry.rel_path if _unparsed(chunks) else None,
                )
                # Links are per file by nature — they name the file they
                # were found in — so they are recorded as the file is
                # read, not when its chunks happen to reach the store.
                self._link(
                    chunks,
                    source=source,
                    history=history,
                    references=config.references,
                    by_line=blames.get(entry.rel_path, {}),
                )
            batch += chunks
            if len(batch) >= _WRITE_BATCH:
                totals = totals.wrote(self.store.add_chunks(dataset_name=repo.id, chunks=batch))
                batch = []
        if batch:
            totals = totals.wrote(self.store.add_chunks(dataset_name=repo.id, chunks=batch))
        return totals

    def _link(
        self,
        chunks: list[Chunk],
        *,
        source: SourceFile,
        history: _History,
        references: dict[str, str],
        by_line: dict[int, str],
    ) -> None:
        """Record what one file's chunks name, and who wrote their lines.

        The repo id comes from `source`, which already carries it — a
        separate parameter for the same value is one more thing that can
        disagree with itself.
        """
        if self.links is None:
            return
        self.links.add_links(
            links_for(chunks, references=references), repo=source.repo, path=source.path
        )
        # Blame is the expensive half, so it is paid per *indexed* file —
        # which the incremental path already keeps down to what changed.
        self.links.add_links(
            blame_links(chunks=chunks, known=history.by_sha, by_line=by_line),
            repo=source.repo,
            path=source.path,
        )

    def _forget(self, repo: Repository, *, stale: list[str]) -> int:
        """Delete chunks the tree no longer produces, and their links.

        The same set, in the same breath. A link that outlives its chunk
        is not merely stale: it is indistinguishable from a real dangling
        link, so the drift report would fill with references from code
        that no longer exists (ADR-9).
        """
        if not stale:
            return 0
        deleted = self.store.delete_chunks(dataset_name=repo.id, ids=stale)
        if self.links is not None:
            self.links.delete_by_source(stale)
        return deleted

    def why(self, symbol: str, *, limit: int = 3) -> list[Definition]:
        """Definitions of `symbol`, each with the commits that wrote it.

        Lives here because two adapters wanted it and each built it
        itself — `wsindex why` and the MCP tool — from the same three
        moves: find the definitions, walk their BLAMED_BY edges, fetch
        each commit's message. They had already drifted (one showed three
        definitions, the other all of them), which is what a rule in
        ADR-10 exists to prevent: an interface that cannot be written as
        a library call means the library is missing something.

        The `symbol` filter is a prefilter, so the store narrows to
        exactly the matching chunks and ranking only breaks ties among
        them.

        Args:
            symbol: Name to look for; matched as a substring.
            limit: How many definitions to return, best match first.

        Returns:
            The definitions, empty when nothing matches. A definition
            with no commits is normal — links may be off, or the repo
            may not have been indexed since blame edges existed.
        """
        found = self.search(symbol, k=limit, filters=SearchFilter(symbol=symbol))
        if not found or self.links is None:
            return [Definition(hit=hit, commits=()) for hit in found]
        return [Definition(hit=hit, commits=tuple(self._authors(hit))) for hit in found]

    def _authors(self, hit: Hit) -> Iterator[Authorship]:
        """The commits a blame edge attributes this chunk to."""
        assert self.links is not None
        for edge in self.links.out_of([str(hit.native_id)], kind=LinkKind.BLAMED_BY):
            if edge.dst_chunk_id is None:
                yield Authorship(commit=edge.name, message=None)
                continue
            pointed = self.links.out_of([edge.dst_chunk_id], kind=LinkKind.REFERENCES)
            yield Authorship(
                commit=edge.name,
                message=self.commit_message(edge.repo, edge.dst_chunk_id),
                references=tuple(Reference(name=ref.name, url=ref.url) for ref in pointed),
            )

    def describe(self, path: Path) -> FileReport | None:
        """What the index knows about one file, or None if it owns none.

        `wsindex explain` in library terms. It used to do this itself,
        which cost the CLI eight imports out of `wsindex.ingest` and put
        real analysis — is this file parsed or merely windowed — in an
        adapter.

        Args:
            path: The file, absolute or relative to the current
                directory.

        Returns:
            The report, or None when the path lies outside every
            configured repo.
        """
        target = path.expanduser().resolve()
        for repo in self.config.repos:
            root = Path(repo.path).expanduser().resolve()
            if root != target and root not in target.parents:
                continue
            rel = target.relative_to(root).as_posix()
            found = examine(root, rel, ignore=repo.ignore, formats=repo.formats)
            if not isinstance(found, WalkedFile):
                return FileReport(repo=repo.id, rel_path=rel, skipped=found)
            return self._read_report(repo.id, target, found)
        return None

    @staticmethod
    def _read_report(repo_id: str, target: Path, found: WalkedFile) -> FileReport:
        """Chunk one file to see what it becomes; the second half of `describe`."""
        text = target.read_text(encoding="utf-8", errors="replace")
        source = SourceFile(repo=repo_id, path=found.rel_path, lang=found.lang, kind=found.kind)
        chunks = chunk_file(text, source)
        return FileReport(
            repo=repo_id,
            rel_path=found.rel_path,
            skipped=None,
            lang=found.lang,
            kind=found.kind,
            chunks=len(chunks),
            symbols=sum(1 for chunk in chunks if chunk.symbol),
            parsed_cleanly=not _unparsed(chunks),
        )

    def references(self, name: str) -> list[Edge]:
        """Every link that names this thing — a port, a ticket, a sha.

        A thin pass-through, and it earns its place by removing a
        second connection: the MCP tool opened its own `LinkStore` on
        every call, ignoring the one this Pipeline was handed. The
        resource was injected and then bypassed.

        Args:
            name: Exactly as it was recorded — `8080`, `PROJ-412`.

        Returns:
            The edges, ordered by file then line; empty when links are
            switched off for this pipeline.
        """
        return [] if self.links is None else self.links.by_name(name)

    def commit_message(self, repo: str, chunk_id: str) -> str | None:
        """The text of one indexed commit message, or None if it is gone.

        A question about the workspace, answered here rather than by
        callers reaching through `pipeline.store` — which two of them
        did, each writing the same three lines. None is ordinary: a
        commit chunk that a later run re-indexed away is not an error.

        Args:
            repo: Repo id the commit belongs to.
            chunk_id: Id of the commit's own chunk, as a blame edge holds it.

        Returns:
            The message, or None when the store no longer has it.
        """
        return self.store.chunk_text(repo, ids=[chunk_id]).get(chunk_id)

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        repo: str | None = None,
        filters: SearchFilter | None = None,
    ) -> list[Hit]:
        """Global top-k across all config repos, best score first.

        Merge policy lives here and only here: every dataset is asked for
        k hits (the global top may sit entirely in one repo, so asking for
        less is wrong), then one stable sort merges and cuts to k — on
        equal scores the given repo order wins, which keeps results
        deterministic. A repo that was never indexed (store raises
        ValueError) contributes zero hits here: not yet indexed is a
        normal state, not an error. It is not a *silent* state, though —
        ask `unsearched()` and tell the reader, because an answer that
        skipped half the workspace must not look like one that did not.

        Repo scope is applied here (dataset list), structural filters go
        down to the store as a prefilter — reranker sees only the
        filtered candidates, so the funnel stays consistent (ADR-7).

        Args:
            query: Query text; embedding is the store's business.
            k: Maximum number of hits in the merged result.
            repo: Restrict to a single repo id; unknown id is an error,
                not a silent empty result.
            filters: Structural filters passed through to the store.

        Returns:
            At most k hits across all (scoped) repos, best score first.

        Raises:
            ValueError: `repo` is set but not present in the config.
        """
        started = time.perf_counter()
        repos = self._scope(repo)
        # Before reading, not after: a store holds the version it opened
        # at, so a long-lived process would answer from the corpus as it
        # was when it started and never fail doing it (ADR-10). Costs
        # ~4 ms; a CLI never notices and a server cannot do without it.
        self.store.refresh()
        n = _CANDIDATE_MULTIPLIER if self.reranker else 1
        all_hits: list[Hit] = []
        for r in repos:
            try:
                hits = self.store.search(dataset_name=r.id, query=query, k=k * n, filters=filters)
            except ValueError:
                continue
            all_hits.extend(hits)
        if self.reranker:
            scores = self.reranker.rank(query, [hit.text for hit in all_hits])
            all_hits = [replace(h, score=s) for h, s in zip(all_hits, scores, strict=True)]
        best = sorted(all_hits, key=lambda h: h.score, reverse=True)[:k]
        if self.stats is not None:
            # After the answer is computed, and unable to affect it: a
            # note about a question must not be able to break answering
            # it. `searched` swallows its own failures for the same
            # reason. Measured at 0.05 ms against an 8 ms search —
            # 0.7%, which is below the run-to-run noise of the search
            # itself.
            self.stats.searched(
                query,
                k=k,
                repo=repo,
                hits=len(best),
                top_score=best[0].score if best else None,
                ms=round((time.perf_counter() - started) * 1000, 2),
                reranked=self.reranker is not None,
            )
        return best

    def _scope(self, repo: str | None) -> list[Repository]:
        """The repos a query covers, or a named error for an unknown id."""
        if repo is None:
            return list(self.config.repos)
        scoped = [r for r in self.config.repos if r.id == repo]
        if not scoped:
            raise ValueError(f"unknown repo id: {repo!r}")
        return scoped

    def fit(self, hits: list[Hit], *, budget: int) -> tuple[list[Hit], int]:
        """The longest prefix of `hits` that costs at most `budget` tokens.

        `k` answers "how many results"; an agent needs "how much context",
        and the two are not the same question. Ten hits are anywhere
        between two hundred tokens and twelve thousand depending on what
        they happen to contain, and today the caller finds out only after
        it has already spent them. Measured on the acceptance criteria,
        plain search fills a thousand tokens with fifteen chunks and
        already holds the expected answer for all ten queries — so this
        is not about finding more, it is about not overrunning.

        A prefix rather than a knapsack: hits arrive best-first, and
        skipping a large one to fit two small ones would quietly reorder
        relevance to save bytes. Dropping the tail is a decision the
        caller can see; re-ranking by size is not.

        Args:
            hits: What `search` returned, best first.
            budget: Maximum tokens the texts may cost together.

        Returns:
            The hits that fit, and what they cost. An empty list when even
            the first does not fit — the caller asked for less than one
            result is worth, and saying so beats overrunning silently.
        """
        kept: list[Hit] = []
        spent = 0
        for hit in hits:
            cost = self.store.count_tokens(hit.text)
            if spent + cost > budget:
                break
            kept.append(hit)
            spent += cost
        return kept, spent

    def unsearched(self, repo: str | None = None) -> tuple[str, ...]:
        """Configured repos the store has never heard of, in config order.

        The other half of `search`. A repo that was added to the config
        and never indexed — or whose indexing failed a month ago — takes
        no part in any search and says nothing about it, so every answer
        since has been quietly partial. This is what lets the caller put
        that in words.

        Asked separately rather than returned from `search` because a
        Pipeline is frozen and a search returns hits; the store call
        behind this reads one small registry table.

        Args:
            repo: Restrict to one repo id, matching the search it
                accompanies. An unknown id raises, exactly as it does
                there.

        Returns:
            The ids, in the order the config lists them; empty when the
            whole workspace is searchable.

        Raises:
            ValueError: `repo` is set but not present in the config.
        """
        known = self.store.datasets()
        return tuple(r.id for r in self._scope(repo) if r.id not in known)


__all__ = [
    # Re-exported from `wsindex.run`, which is where they live now: every
    # adapter already imports them from here, and a split of this module
    # is not a reason for four other files to change.
    "Authorship",
    "Definition",
    "FileReport",
    "FullPass",
    "IndexReport",
    "Pipeline",
    "Reference",
]
