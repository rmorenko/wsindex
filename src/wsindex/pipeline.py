"""Indexing and search pipeline: repos in, Hits out.

The pipeline sees only the VectorStore contract and works in plain text —
embedding is the store's private business. The concrete backend is built
at the edge (the CLI composition root) and injected through the
constructor; which repositories to index and with which metric comes from
`Config()`, read at call time. One process serves one workspace, so
threading those two values through the composition root only to hand them
back unchanged was ceremony.

`index` is incremental against git (Этап 8). Both of its paths — the
full pass and the incremental one — end in the same two lines: add the
chunks the working tree currently produces, then delete every stored
chunk in scope that it did not. Only the scope differs (a few changed
paths, or the whole dataset), which is why the reconciling delete is
written once. That also fixes the debt the full pass carried before
step 20: it used to only ever add, so a file that shrank or vanished
left its old chunks in the index forever.
"""

from dataclasses import dataclass, replace
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
from wsindex.model import Chunk, Hit, SearchFilter
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
    nothing may accumulate between calls. The repo list and the metric are
    read from `Config()` inside the methods rather than captured at
    construction, so a config that changed (a repo added, say) is picked
    up by the next call instead of going stale in a field.

    Attributes:
        store: Any VectorStore backend; the pipeline never looks behind
            the contract.
        state_dir: Where `index` keeps the per-repo "last indexed commit"
            file. Supplied by the composition root because it is a
            location, not a policy — the same reason the store gets its
            uri from there (see `wsindex.cli._build_pipeline`).
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
    reranker: Reranker | None = None
    links: LinkStore | None = None

    def index(self) -> IndexReport:
        """Index every repo into its own dataset (dataset name = repo id).

        Incremental against git. A repo goes down the fast path when it
        has been indexed before *and* its working tree is clean; then
        only the files git reports as changed since that commit are read,
        and the chunks those files no longer produce are deleted.

        A dirty working tree forces a full pass, and the run records no
        new commit for that repo. This is not pessimism, it is the only
        honest answer: a diff between two commits cannot see an
        uncommitted edit or an untracked file, but `walk_repo` indexes
        both, so trusting the diff would leave the index describing a
        tree that never existed. Recording HEAD anyway would make the
        *next* run skip those same invisible changes forever.

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
                repository root. Git-only is a decision of Этап 8: a
                fallback to plain walking would mean two models of
                state, so this is a config error with a message.
        """
        config = Config()
        state = IndexState.load(self.state_dir)
        files = chunks_count = written = deleted = commits = 0
        missing_repos: list[str] = []
        full_repos: list[str] = []
        for repo in config.repos:
            root = Path(repo.path)
            if not root.is_dir():
                missing_repos.append(repo.id)
                continue
            self.store.create_dataset(dataset_name=repo.id, metric=config.metric)
            diff, dirty = self._plan(repo, root=root, state=state)
            if diff.full:
                full_repos.append(repo.id)
            if diff.full or diff.changed or diff.deleted:
                totals = self._apply(repo, root=root, diff=diff)
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
                state = state.with_commit(repo.id, diff.head)
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

        Returns:
            The diff to apply, and whether the working tree is dirty.
        """
        # Tracked edits, staged files and untracked files alike: any of
        # them makes the working tree differ from every commit, so no
        # commit-to-commit diff can describe what we are about to index.
        dirty = has_uncommitted_changes(root)
        since = None if dirty else state.commits.get(repo.id)
        return diff_since(root, since=since), dirty

    def _apply(self, repo: Repository, *, root: Path, diff: RepoDiff) -> _Totals:
        """Chunk what changed, then delete what the tree no longer produces.

        The two halves are one thought: `add_chunks` makes the store hold
        everything the current tree says it should, and the delete below
        removes everything in scope that the tree did not just produce.
        A file that lost its tail, a file that was deleted, a file that
        stopped being indexable — all three are the same subtraction.

        Args:
            repo: The repo being indexed; its id names the dataset.
            root: Repository root.
            diff: What to look at. `diff.full` widens the reconciling
                delete from "the paths listed here" to "the whole
                dataset", which is what makes a full pass also clean up
                files that vanished while nobody was watching.

        Returns:
            Totals for this repo.
        """
        indexable: list[WalkedFile] = []
        forget: list[str] = list(diff.deleted)
        for rel_path in diff.changed:
            walked = inspect_file(root, rel_path)
            if walked is None:
                # It changed into something we do not index — renamed to
                # a .png, grown past the size limit, turned binary. Its
                # old chunks have to go, which is exactly "deleted" to us.
                forget.append(rel_path)
                continue
            indexable.append(walked)

        # Read the stored ids BEFORE writing, so the set means "what was
        # here when we started". Reading after would work too (the new
        # ids cancel out), but the intent would be harder to see.
        scope = None if diff.full else [*(w.rel_path for w in indexable), *forget]
        stored = self.store.chunk_ids(dataset_name=repo.id, paths=scope)

        # Commits first: their chunk ids are what blame edges point at,
        # and `git log` over a whole history costs milliseconds.
        commits = read_commits(root, since=diff.since)
        messages = commit_chunks(commits, repo=repo.id)
        # Keyed off the chunk's symbol rather than zipping: `commit_chunks`
        # drops commits with an empty message, so the two lists are not
        # guaranteed to line up.
        by_short = {m.symbol: m.id for m in messages}
        commit_ids = {c.sha: by_short[c.short] for c in commits if c.short in by_short}
        written_commits = self.store.add_chunks(dataset_name=repo.id, chunks=messages)

        files = chunks_count = written = 0
        fresh_ids: set[str] = set()
        for walked in indexable:
            chunks: list[Chunk] = chunk_file(
                text=(root / walked.rel_path).read_text(encoding="utf-8", errors="replace"),
                path=walked.rel_path,
                repo=repo.id,
                lang=walked.lang,
                kind=walked.kind,
            )
            fresh_ids.update(chunk.id for chunk in chunks)
            files += 1
            chunks_count += len(chunks)
            written += self.store.add_chunks(dataset_name=repo.id, chunks=chunks)
            if self.links is not None:
                self.links.add_links(links_for(chunks), repo=repo.id, path=walked.rel_path)
                # Blame is the expensive half of step 27 (~28 ms/file), so
                # it is paid per *indexed* file — which the incremental
                # path already keeps down to what changed.
                self.links.add_links(
                    blame_links(root, rel_path=walked.rel_path, chunks=chunks, known=commit_ids),
                    repo=repo.id,
                    path=walked.rel_path,
                )

        # Commit chunks are never stale: a commit is immutable, so the
        # chunk it produced can only ever be re-derived identically. They
        # must still be counted as fresh, or a full pass — whose `stored`
        # covers the whole dataset — would reap every one of them.
        fresh_ids.update(m.id for m in messages)

        stale = sorted(stored - fresh_ids)
        deleted = self.store.delete_chunks(dataset_name=repo.id, ids=stale) if stale else 0
        if self.links is not None and stale:
            # The same set, in the same breath. A link that outlives its
            # chunk is not merely stale: it is indistinguishable from a
            # real dangling link, so the drift report would fill with
            # references from code that no longer exists (ADR-9).
            self.links.delete_by_source(stale)
        return _Totals(
            files=files,
            chunks=chunks_count,
            written=written,
            deleted=deleted,
            commits=written_commits,
        )

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
        repos = Config().repos
        if repo is not None:
            repos = [r for r in repos if r.id == repo]
            if not repos:
                raise ValueError(f"unknown repo id: {repo!r}")
        n = _CANDIDATE_MULTIPLIER if self.reranker else 1
        all_hits: list[Hit] = []
        for r in repos:
            try:
                hits = self.store.search(dataset_name=r.id, query=query, k=k * n, filters=filters)
            except ValueError:
                continue
            all_hits.extend(hits)
        if self.reranker:
            scores = self.reranker.rank(query, [h.metadata["text"] for h in all_hits])
            all_hits = [replace(h, score=s) for h, s in zip(all_hits, scores, strict=True)]
        return sorted(all_hits, key=lambda h: h.score, reverse=True)[:k]
