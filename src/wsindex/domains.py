"""Where a repository's meaning sits, and where it crosses its own lines.

A different consumer from search: an architect asking "what is this
codebase made of, and what is tangled" rather than a person asking where
something is. The decision of 2026-08-14 — that a graph is not what
search needs — is untouched; this reads the same data for another
question.

Two signals, and the value is in their *disagreement*.

**Semantic.** A file's meaning is the mean of its chunks' vectors. Files
whose meanings are close belong together, whatever directory they are
filed under. Measured on this project: of a file's five nearest
neighbours, 57% share its package against an 11% baseline — so the
vectors do recover the layout somebody chose, which is what makes the
exceptions worth reading.

**Empirical.** Files that keep changing in the same commit are coupled
whether or not anything imports anything. Measured here, pairs that
change together score 0.623 to each other semantically against 0.422 for
all pairs, so the two signals agree well above chance — and where they
*do not*, one of them is telling you something.

A third signal was planned, the structural one: `READS_KEY` links from
code to config. It is not used, and the reason is measured rather than
assumed — the link extractor's whole vocabulary is port numbers, which
came to ten names across five thousand files. It would contribute
nothing here until that vocabulary grows.

**No graph viewer**, deliberately. The interesting output is three short
answers, and an interactive graph is the thing these tools become instead
of answering them.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from wsindex.ingest.git_state import GitCommandError, decode_path, run_git
from wsindex.model import Kind

if TYPE_CHECKING:  # pragma: no cover - import-time only, for annotations
    from wsindex.pipeline import Pipeline

NEIGHBOURS = 5
"""How many nearest files make up a file's neighbourhood.

Five because the question is "where does this file's meaning live", and
one neighbour is an anecdote while twenty is the whole package."""

COUPLED_FROM = 4
"""How many shared commits make a pair worth reporting as coupled.

Below four is coincidence on any repository with a merge in it: two files
touched by one sweeping rename are not a design fact."""

COMMITS_READ = 400
"""How far back co-change is counted. Coupling is a property of how a
project is being worked on now, and a rename from two years ago is
archaeology rather than evidence."""


@dataclass(frozen=True, kw_only=True)
class Coupled:
    """Two files that keep changing together across a package boundary.

    Attributes:
        left, right: The files, in path order.
        commits: How many of the commits read touched both.
        similarity: How close their meanings are, -1 to 1. A high number
            says the coupling is honest — they change together because
            they are about the same thing. A low one is the interesting
            case: something binds them that is not their subject.
    """

    left: str
    right: str
    commits: int
    similarity: float


@dataclass(frozen=True, kw_only=True)
class Stranger:
    """A file whose nearest neighbours are all outside its own package.

    Not a defect by itself. It says the file's subject lives somewhere
    other than its directory, which is sometimes deliberate — and
    sometimes a module that was filed by when it runs rather than by what
    it is about.
    """

    path: str
    package: str
    neighbours: tuple[str, ...]
    similarity: float


@dataclass(frozen=True, kw_only=True)
class Domains:
    """What `wsindex domains` found.

    Attributes:
        repo: The repository read.
        files: How many source files carried enough chunks to place.
        packages: Package name -> how many files it holds.
        agreement: Share of a file's neighbours that share its package,
            averaged. The baseline is one over the number of packages;
            well above it means the layout and the meaning agree, and the
            exceptions below are worth reading. At or near it means they
            do not, and nothing else in this report should be trusted.
        baseline: What `agreement` would be if meaning said nothing.
        coupled: Cross-package pairs that keep changing together.
        strangers: Files whose meaning sits outside their package.
    """

    repo: str
    files: int
    packages: dict[str, int]
    agreement: float
    baseline: float
    coupled: tuple[Coupled, ...]
    strangers: tuple[Stranger, ...]


def branching_depth(paths: Iterable[str]) -> int:
    """The first directory level where the layout actually branches.

    Derived rather than configured, because the alternative is a constant
    that fits one project. `src/wsindex/store/base.py` wants level 2 and
    `src/alpha/one.py` wants level 1, and a default of either is wrong
    for the other — which is how the first version of this reported a
    three-package repository as having one.

    A level with a single name carries no information: every file is
    under `src`, so `src` does not distinguish anything. Descend until a
    level has more than one name, and that is where the packages start.
    """
    depth = 0
    listed = [Path(path).parts for path in paths]
    while depth < 8:  # a guard, not a limit: no layout nests packages this deep
        names = {parts[depth] for parts in listed if len(parts) > depth + 1}
        if not names:
            # Past the deepest file: everything lives in one directory, so
            # the level above is the package and there is exactly one.
            return max(depth - 1, 0)
        if len(names) != 1:
            return depth
        depth += 1
    return depth  # pragma: no cover - eight single-child levels is not a layout


def package_of(path: str, *, depth: int) -> str:
    """The package a path belongs to, or `(root)` for a module above them all.

    `depth` comes from `branching_depth`. Getting it off by one turns
    every file into its own package and every number after it into
    nonsense, which is what the first spike did — it reported 43 packages
    for a project with nine, making the baseline 2%.
    """
    parts = Path(path).parts
    return parts[depth] if len(parts) > depth + 1 else "(root)"


def analyse(pipeline: Pipeline, *, repo: str, prefix: str = "src/") -> Domains:
    """Read one repository's shape from its vectors and its history.

    Args:
        pipeline: A pipeline whose store holds the repository's chunks.
        repo: Repo id, as the config names it.
        prefix: Only paths starting here are considered — a report that
            counted tests and vendored code would describe the repository
            plus everything it happens to contain.

    Returns:
        The report. Empty-ish rather than raising when there is too
        little to say: a repository with three source files has no
        domains, and saying so beats inventing them.

    Raises:
        ValueError: `repo` names nothing the store holds.
    """
    centroids = _centroids(pipeline, repo=repo, prefix=prefix)
    if len(centroids) < NEIGHBOURS + 1:
        shallow = branching_depth(centroids)
        return Domains(
            repo=repo,
            files=len(centroids),
            packages=dict(Counter(package_of(p, depth=shallow) for p in centroids)),
            agreement=0.0,
            baseline=0.0,
            coupled=(),
            strangers=(),
        )
    files = sorted(centroids)
    depth = branching_depth(files)
    packages = Counter(package_of(path, depth=depth) for path in files)
    neighbourhood = {path: _nearest(path, centroids, files) for path in files}
    shares = [
        sum(1 for n in near if package_of(n, depth=depth) == package_of(path, depth=depth))
        / NEIGHBOURS
        for path, near in neighbourhood.items()
    ]
    strangers = tuple(
        Stranger(
            path=path,
            package=package_of(path, depth=depth),
            neighbours=tuple(near),
            similarity=round(_cosine(centroids[path], centroids[near[0]]), 3),
        )
        for path, near in neighbourhood.items()
        if not any(package_of(n, depth=depth) == package_of(path, depth=depth) for n in near)
    )
    return Domains(
        repo=repo,
        files=len(files),
        packages=dict(packages.most_common()),
        agreement=round(statistics.mean(shares), 3),
        baseline=round(1 / len(packages), 3),
        coupled=_coupling(pipeline, repo=repo, centroids=centroids, depth=depth),
        strangers=tuple(sorted(strangers, key=lambda s: -s.similarity)),
    )


def _centroids(pipeline: Pipeline, *, repo: str, prefix: str) -> dict[str, list[float]]:
    """Each source file's mean chunk vector, normalised to unit length.

    The mean, because a file has no vector of its own and its chunks are
    what the store holds. Normalised, so that a dot product is a cosine
    and every comparison below is one multiplication.
    """
    if repo not in pipeline.store.datasets():
        raise ValueError(f"no repo {repo!r} in this index")
    vectors = pipeline.store.vectors(repo, kind=Kind.CODE)
    paths = pipeline.store.paths_of(repo, ids=list(vectors))
    grouped: dict[str, list[list[float]]] = {}
    for chunk_id, vector in vectors.items():
        path = paths.get(chunk_id, "")
        if path.startswith(prefix):
            grouped.setdefault(path, []).append(vector)
    centroids: dict[str, list[float]] = {}
    for path, group in grouped.items():
        mean = [sum(values) / len(group) for values in zip(*group, strict=True)]
        norm = sum(value * value for value in mean) ** 0.5 or 1.0
        centroids[path] = [value / norm for value in mean]
    return centroids


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _nearest(path: str, centroids: dict[str, list[float]], files: list[str]) -> list[str]:
    """The `NEIGHBOURS` files closest in meaning to this one."""
    mine = centroids[path]
    scored = sorted(
        ((_cosine(mine, centroids[other]), other) for other in files if other != path),
        reverse=True,
    )
    return [other for _, other in scored[:NEIGHBOURS]]


def _coupling(
    pipeline: Pipeline, *, repo: str, centroids: dict[str, list[float]], depth: int
) -> tuple[Coupled, ...]:
    """Cross-package pairs that keep changing in the same commit.

    Read from git rather than from the indexed commit chunks: what is
    needed is which *files* a commit touched, and a commit chunk holds
    the message. Returns nothing when the repository is not readable —
    coupling is the optional half of this report.
    """
    root = next((Path(r.path) for r in pipeline.config.repos if r.id == repo), None)
    if root is None:
        return ()
    try:
        # A record separator before each sha, and it is load-bearing.
        # `--name-only` separates commits with a blank line, so splitting
        # on that cuts a sha away from its own file list — and then
        # dropping the block's first line, meaning to drop the sha, drops
        # **the first file of every commit** instead. It hides on a
        # repository whose commits touch many files and shows up at once
        # on a pair. `ingest.commits` uses \x1e for the same reason.
        log = decode_path(
            run_git(root, "log", "--format=\x1e%H", "--name-only", f"-n{COMMITS_READ}")
        )
    except (GitCommandError, OSError):  # pragma: no cover - depends on the checkout
        return ()
    together: Counter[tuple[str, str]] = Counter()
    for block in log.split("\x1e"):
        changed = sorted({line for line in block.splitlines()[1:] if line in centroids})
        for index, left in enumerate(changed):
            for right in changed[index + 1 :]:
                together[(left, right)] += 1
    found = [
        Coupled(
            left=left,
            right=right,
            commits=count,
            similarity=round(_cosine(centroids[left], centroids[right]), 3),
        )
        for (left, right), count in together.items()
        if count >= COUPLED_FROM and package_of(left, depth=depth) != package_of(right, depth=depth)
    ]
    # Loudest first: many shared commits and little shared meaning is the
    # pair worth a conversation, since something binds them that is not
    # their subject.
    return tuple(sorted(found, key=lambda c: (-c.commits, c.similarity)))


__all__ = [
    "COMMITS_READ",
    "COUPLED_FROM",
    "NEIGHBOURS",
    "Coupled",
    "Domains",
    "Stranger",
    "analyse",
    "branching_depth",
    "package_of",
]
