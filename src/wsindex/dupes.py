"""The same code in two places, found by what it is made of.

Not by meaning. Vectors were measured against this and lost: real copies
found by the exact method below score 0.994, 0.840 and 0.717 by how
heavily they were edited, while unrelated pairs sit at a median of 0.135
— so the correlation is real, but the tails overlap (unrelated p99 of
0.860 against adapted p90 of 0.864) and a threshold catching adapted
copies flags two or three percent of everything. Near-identical copies,
which embeddings do separate cleanly, are separated exactly and far more
cheaply by a fingerprint. So: fingerprints.

**Token shingles.** Every run of `SHINGLE` identifiers is hashed, and two
chunks are compared by how many hashes they share (Jaccard). Identifiers
rather than characters, so reformatting and whitespace change nothing;
overlapping runs, so an inserted line costs a few shingles rather than
the alignment.

**The report is grouped, and that is the whole design.** Measured on a
twenty-year PHP codebase, the pairs a naive report would print are
overwhelmingly generated classes and vendored libraries — three hundred
pairs between two copies of jQuery UI say one thing, not three hundred.
So pairs are collapsed by the directories they connect, and a directory
pair is one line. What is left after that collapse is the copy-paste a
person can act on: `contrib/forms/dap/report.php` against
`contrib/forms/ped_GI/report.php`, a report function copied per form.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from wsindex.model import Kind

if TYPE_CHECKING:  # pragma: no cover - import-time only, for annotations
    from wsindex.pipeline import Pipeline

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
"""What a shingle is made of. Identifiers only: punctuation, indentation
and line breaks are exactly what a copy changes first."""

SHINGLE = 5
"""Identifiers per shingle.

Long enough that a shared run means something — five consecutive
identifiers in the same order is not a coincidence — and short enough
that an edit inside a function does not destroy every shingle around it."""

MIN_OVERLAP = 0.45
"""Shared-shingle fraction at which a pair is worth printing.

Measured by reading what sits on each side of it, rather than by naming
bands after the number that defines them. Between 0.30 and 0.45 on a real
codebase: generated classes, hundreds of them, with identical
`__construct($data = [])` and `xmlSerialize` bodies — true duplicates
that nobody can act on. From 0.45: a report function copied per form
type, and vendored libraries checked in twice. So the line goes where
generated code stops and copying starts."""

MIN_TOKENS = 60
"""Identifiers a chunk needs before it can be part of a pair.

A four-line accessor is identical to a thousand others and says nothing.
Sixty identifiers is roughly a function worth having copied."""

COMMON_SHINGLE = 40
"""Chunks a shingle may appear in before it is dropped from the index.

Boilerplate — a licence header, a framework's call signature — appears
everywhere and connects everything, which would make candidate
generation quadratic again and the results meaningless. Dropping the
common ones is what keeps this linear-ish; it is the same idea as a stop
word, and it is why the report does not drown in `public function`."""


@dataclass(frozen=True, kw_only=True)
class Pair:
    """Two chunks that share most of what they are made of."""

    left: str
    right: str
    overlap: float
    left_lines: tuple[int, int]
    right_lines: tuple[int, int]


@dataclass(frozen=True, kw_only=True)
class Between:
    """Everything two directories have in common.

    One line per directory pair is the report's whole point: two copies
    of a vendored library share hundreds of chunks and are one fact.

    Attributes:
        left, right: The directories, in path order.
        pairs: The duplicate pairs found between them, worst first.
        files: How many distinct files on the left take part.
    """

    left: str
    right: str
    pairs: tuple[Pair, ...]
    files: int

    @property
    def within(self) -> bool:
        """Whether this is one directory repeating itself.

        A real and common shape — a tree of generated files, a form
        copied per variant — and a different fact from two directories
        mirroring each other, so the report says which.
        """
        return self.left == self.right

    @property
    def cross_repo(self) -> bool:
        """Whether the two sides live in different repositories.

        The fact a workspace-wide tool exists to report, and the one a
        per-repository report could not reach at all. Paths are qualified
        as `repo/path`, so the repository is the first component.
        """
        return self.left.split("/", 1)[0] != self.right.split("/", 1)[0]

    @property
    def wholesale(self) -> bool:
        """Whether this looks like a copied directory rather than copied code.

        Many pairs between two places is a vendored library or a
        generated tree — one decision somebody made once. A handful is
        code a person copied, which is the kind worth reading.
        """
        return len(self.pairs) >= 10


@dataclass(frozen=True, kw_only=True)
class Duplication:
    """What `wsindex dupes` found."""

    repos: tuple[str, ...]
    chunks: int
    compared: int
    between: tuple[Between, ...]


def fingerprint(text: str) -> set[int]:
    """The set of shingle hashes this text is made of.

    Empty when there is not enough of it to fingerprint, which is how a
    short chunk removes itself from consideration without a separate
    check anywhere else.
    """
    tokens = TOKEN.findall(text)
    if len(tokens) < MIN_TOKENS:
        return set()
    return {
        hash(tuple(tokens[index : index + SHINGLE])) for index in range(len(tokens) - SHINGLE + 1)
    }


def find(
    pipeline: Pipeline, *, repo: str | None = None, minimum: float = MIN_OVERLAP
) -> Duplication:
    """Duplicate code across a workspace, grouped by where it lives.

    **Across, not within, and that is the point.** This used to take one
    repository and could therefore not answer the question a multi-repo
    tool exists for. The field trial made that concrete: the workspace
    chosen *because* its three adapter gems are near-copies of each other
    reported almost nothing, and could not have reported otherwise.

    Args:
        pipeline: A pipeline whose store holds the workspace's chunks.
        repo: One repo id to narrow to, or None for every indexed repo.
        minimum: Shared-shingle fraction a pair must reach.

    Returns:
        The report, directory pairs first, each worst-overlap first.
        Paths are qualified as `repo/path`, the same way a search hit is.

    Raises:
        ValueError: `repo` names nothing the store holds.
    """
    known = pipeline.store.datasets()
    if repo is not None and repo not in known:
        raise ValueError(f"no repo {repo!r} in this index")
    repos = [repo] if repo is not None else sorted(known)
    chunks = _code(pipeline, repos=repos)
    prints = {where: fingerprint(text) for where, (_, text, _) in chunks.items()}
    prints = {where: marks for where, marks in prints.items() if marks}
    pairs = _pairs(chunks, prints, minimum=minimum)
    return Duplication(
        repos=tuple(repos),
        chunks=len(chunks),
        compared=len(prints),
        between=_group(pairs),
    )


def _code(
    pipeline: Pipeline, *, repos: list[str]
) -> dict[tuple[str, str], tuple[str, str, tuple[int, int]]]:
    """(repo, chunk id) -> (`repo/path`, text, line range) for the workspace's code.

    **Keyed by the pair, not by the chunk id alone**, and that is not
    defensive: a chunk id is a hash of text and path with no repository
    in it, so a vendored library copied into two repos — which keeps its
    paths — produces the *same* id twice. Keying by the id would make one
    of them overwrite the other and hide the strongest duplication there
    is, an exact copy, from the report meant to find it.
    """
    found: dict[tuple[str, str], tuple[str, str, tuple[int, int]]] = {}
    for repo in repos:
        ids = sorted(pipeline.store.chunk_ids(repo))
        meta = pipeline.store.metadata_of(repo, ids=ids)
        code = [chunk_id for chunk_id, m in meta.items() if m.kind == Kind.CODE.value]
        for chunk_id, text in pipeline.store.chunk_text(repo, ids=code).items():
            entry = meta[chunk_id]
            found[(repo, chunk_id)] = (
                f"{repo}/{entry.path}",
                text,
                (entry.start_line, entry.end_line),
            )
    return found


def _pairs(
    chunks: dict[tuple[str, str], tuple[str, str, tuple[int, int]]],
    prints: dict[tuple[str, str], set[int]],
    *,
    minimum: float,
) -> list[Pair]:
    """Candidate pairs from a shingle index, scored exactly.

    Two stages, and the first is what makes this finishable: an inverted
    index over shingles proposes only chunks that share one, so nothing
    is compared against everything. Shingles held by more than
    `COMMON_SHINGLE` chunks are dropped before that — they are
    boilerplate, they connect the whole repository to itself, and keeping
    them would restore the quadratic behaviour the index exists to avoid.
    """
    postings: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for where, marks in prints.items():
        for mark in marks:
            postings[mark].append(where)
    candidates: Counter[tuple[tuple[str, str], tuple[str, str]]] = Counter()
    for holders in postings.values():
        if len(holders) > COMMON_SHINGLE:
            continue
        for index, left in enumerate(holders):
            for right in holders[index + 1 :]:
                candidates[(left, right) if left < right else (right, left)] += 1
    found: list[Pair] = []
    for (left, right), _ in candidates.items():
        if chunks[left][0] == chunks[right][0]:
            continue  # the same file in the same repo: overlapping windows, not a copy
        a, b = prints[left], prints[right]
        overlap = len(a & b) / len(a | b)
        if overlap >= minimum:
            found.append(
                Pair(
                    left=chunks[left][0],
                    right=chunks[right][0],
                    overlap=round(overlap, 3),
                    left_lines=chunks[left][2],
                    right_lines=chunks[right][2],
                )
            )
    return found


def _group(pairs: list[Pair]) -> tuple[Between, ...]:
    """Collapse pairs into the directory pairs they connect."""
    grouped: dict[tuple[str, str], list[Pair]] = defaultdict(list)
    for pair in pairs:
        left, right = str(Path(pair.left).parent), str(Path(pair.right).parent)
        grouped[(left, right) if left <= right else (right, left)].append(pair)
    between = [
        Between(
            left=left,
            right=right,
            pairs=tuple(sorted(found, key=lambda p: -p.overlap)),
            files=len({p.left for p in found}),
        )
        for (left, right), found in grouped.items()
    ]
    # Wholesale copies first and loudest: they are one decision each, and
    # reading them first is what stops a person scrolling past the
    # handful of hand-copied functions underneath.
    return tuple(sorted(between, key=lambda b: (-len(b.pairs), b.left)))


__all__ = [
    "COMMON_SHINGLE",
    "MIN_OVERLAP",
    "MIN_TOKENS",
    "SHINGLE",
    "Between",
    "Duplication",
    "Pair",
    "find",
    "fingerprint",
]
