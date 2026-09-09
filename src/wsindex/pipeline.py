"""Indexing and search pipeline: repos in, Hits out.

The pipeline sees only the VectorStore contract and works in plain text —
embedding is the store's private business. The concrete backend is built
at the edge (the CLI composition root) and injected through the
constructor; which repositories to index and with which metric comes from
`Config()`, read at call time. One process serves one workspace, so
threading those two values through the composition root only to hand them
back unchanged was ceremony.

`index` is incremental against git. Both of its paths — the
full pass and the incremental one — end in the same two lines: add the
chunks the working tree currently produces, then delete every stored
chunk in scope that it did not. Only the scope differs (a few changed
paths, or the whole dataset), which is why the reconciling delete is
written once. That also fixes the debt the full pass carried before
the store used to only ever add, so a file that shrank or vanished
left its old chunks in the index forever.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from wsindex.config import Config, Repository
from wsindex.ingest import (
    IndexState,
    RepoDiff,
    WalkedFile,
    chunk_file,
    diff_since,
    has_uncommitted_changes,
    inspect_file,
)
from wsindex.ingest.commits import blame_links, commit_chunks, read_commits
from wsindex.ingest.link_extract import links_for
from wsindex.links import LinkStore
from wsindex.model import Chunk, Hit, SearchFilter, SourceFile
from wsindex.rank.reranker import Reranker
from wsindex.store import VectorStore

_CANDIDATE_MULTIPLIER = 4


@dataclass(frozen=True, kw_only=True)
class _Totals:
    """One repo's contribution to an IndexReport; summed by `index`."""

    files: int
    chunks: int
    written: int
    deleted: int
    commits: int


@dataclass(frozen=True, kw_only=True)
class _History:
    """What indexing one repo's commit messages produced.

    Attributes:
        written: Message chunks the store actually wrote.
        ids: Their chunk ids, which count as fresh (see `_index_commits`).
        by_sha: Commit sha -> its chunk id, which is what a blame edge
            needs to point at the message that explains a line.
    """

    written: int
    ids: set[str]
    by_sha: dict[str, str]


@dataclass(frozen=True, kw_only=True)
class _Written:
    """Running totals over the files of one repo."""

    files: int = 0
    chunks: int = 0
    written: int = 0
    ids: frozenset[str] = frozenset()

    def add(self, *, chunks: int, written: int, ids: set[str]) -> "_Written":
        """This plus one more file."""
        return _Written(
            files=self.files + 1,
            chunks=self.chunks + chunks,
            written=self.written + written,
            ids=self.ids | ids,
        )


def _selection(
    repo: Repository, *, root: Path, changed: tuple[str, ...], deleted: tuple[str, ...]
) -> tuple[list[WalkedFile], list[str]]:
    """Split the changed paths into what to read and what to forget.

    A path that changed into something unindexable — renamed to a `.png`,
    grown past the size limit, turned binary — is a deletion as far as the
    store is concerned, which is why it joins the second list rather than
    being skipped.

    Args:
        repo: The repo, for its own `ignore` and `formats` markup.
        root: Repository root.
        changed: Paths git reports as added or modified.
        deleted: Paths git reports as gone.

    Returns:
        The files worth reading, and the paths whose chunks must go.
    """
    indexable: list[WalkedFile] = []
    forget: list[str] = list(deleted)
    for rel_path in changed:
        walked = inspect_file(root, rel_path, ignore=repo.ignore, formats=repo.formats)
        if walked is None:
            forget.append(rel_path)
            continue
        indexable.append(walked)
    return indexable, forget


@dataclass(frozen=True, kw_only=True)
class IndexReport:
    """Immutable summary of one index() run, totals across all repos.

    Attributes:
        files: How many files were read and chunked. In an incremental
            run this counts the changed files only, so it is a measure
            of work done, not of corpus size.
        chunks: How many chunks those files produced.
        written: How many chunks the store actually wrote; below `chunks`
            means dedup skipped already-stored ones.
        deleted: How many stored chunks were removed as stale — chunks
            of deleted files, and chunks a changed file no longer
            produces.
        commits: How many commit messages were newly indexed. Counted
            apart from `chunks` on purpose: folding them in would make
            `files: 2  chunks: 4` fail to add up for the reader, since
            two of those chunks came from no file at all.
        missing_repos: Ids of configured repos whose directory does not
            exist; they were skipped, not failed on.
        full_repos: Ids of repos that could not go incremental this run
            and were fully re-read. Either they had never been indexed,
            or their working tree is dirty (see `Pipeline.index`).
    """

    files: int
    chunks: int
    written: int
    deleted: int
    commits: int
    missing_repos: tuple[str, ...]
    full_repos: tuple[str, ...]


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
        config = self.config
        state = IndexState.load(self.state_dir)
        files = chunks_count = written = deleted = commits = 0
        missing_repos: list[str] = []
        full_repos: list[str] = []
        for repo in config.repos:
            # The engine says *who* is being read; what to draw with that
            # is the caller's business (see `wsindex.ui`). A plain
            # callable rather than an event system: one caller, one fact.
            if progress is not None:
                progress(repo.id)
            root = Path(repo.path)
            if not root.is_dir():
                missing_repos.append(repo.id)
                continue
            self.store.create_dataset(dataset_name=repo.id, metric=config.metric)
            diff, dirty = self._plan(repo, root=root, state=state)
            if diff.full:
                full_repos.append(repo.id)
            if diff.full or diff.changed or diff.deleted:
                totals = self._index_repo(repo, root=root, diff=diff, config=config)
                files += totals.files
                chunks_count += totals.chunks
                written += totals.written
                deleted += totals.deleted
                commits += totals.commits
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
        return IndexReport(
            files=files,
            chunks=chunks_count,
            written=written,
            deleted=deleted,
            commits=commits,
            missing_repos=tuple(missing_repos),
            full_repos=tuple(full_repos),
        )

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
        indexable, forget = _selection(repo, root=root, changed=diff.changed, deleted=diff.deleted)
        scope = None if diff.full else [*(walked.rel_path for walked in indexable), *forget]
        stored = self.store.chunk_ids(dataset_name=repo.id, paths=scope)

        history = self._index_commits(repo, root=root, since=diff.since, config=config)
        files = self._index_files(repo, root=root, walked=indexable, history=history, config=config)
        deleted = self._forget(repo, stale=sorted(stored - files.ids - history.ids))
        return _Totals(
            files=files.files,
            chunks=files.chunks,
            written=files.written,
            deleted=deleted,
            commits=history.written,
        )

    def _index_commits(
        self, repo: Repository, *, root: Path, since: str | None, config: Config
    ) -> _History:
        """Index the repo's commit messages, and note where each one landed.

        Before the files, because blame edges point at these chunk ids
        and `git log` over a whole history costs milliseconds.
        """
        commits = read_commits(root, since=since)
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
        """Chunk, store and link every file that is worth reading."""
        totals = _Written()
        for entry in walked:
            source = SourceFile(repo=repo.id, path=entry.rel_path, lang=entry.lang, kind=entry.kind)
            text = (root / entry.rel_path).read_text(encoding="utf-8", errors="replace")
            chunks: list[Chunk] = chunk_file(text, source)
            totals = totals.add(
                chunks=len(chunks),
                written=self.store.add_chunks(dataset_name=repo.id, chunks=chunks),
                ids={chunk.id for chunk in chunks},
            )
            self._link(
                chunks,
                root=root,
                source=source,
                history=history,
                references=config.references,
            )
        return totals

    def _link(
        self,
        chunks: list[Chunk],
        *,
        root: Path,
        source: SourceFile,
        history: _History,
        references: dict[str, str],
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
        # Blame is the expensive half (~28 ms/file), so it is paid per
        # *indexed* file — which the incremental path already keeps down
        # to what changed.
        self.links.add_links(
            blame_links(root, rel_path=source.path, chunks=chunks, known=history.by_sha),
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
        ValueError) silently contributes zero hits: not yet indexed is a
        normal state, not an error.

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
        repos = self.config.repos
        if repo is not None:
            repos = [r for r in repos if r.id == repo]
            if not repos:
                raise ValueError(f"unknown repo id: {repo!r}")
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
        return sorted(all_hits, key=lambda h: h.score, reverse=True)[:k]
