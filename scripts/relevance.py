"""Does it find the right file? The instrument that answers that, and only that.

`acceptance.py` grades ten criteria on one repository and reports 10/10.
The field trial of 2026-09-11 put 204 questions to the same software and
found the answer in the top three for one of them. Both numbers are
correct; they measure different things. Four of the ten acceptance
criteria expect a path fragment sharing a stem with a word in the query
(`stored on disk` -> `storage`), three are identifier lookups, and a
match counts when a *fragment of the path* appears anywhere in the top
five. That instrument cannot see the failure the trial found, and it
certified against it.

So this is a second instrument with three properties the first lacks:

**The questions were written blind.** A tester explored each workspace
with reading and grep only, never running wsindex, and wrote down the
questions a newcomer would ask with the answer found by reading. Six of
the twelve per workspace describe a behaviour using no word that appears
in the answer file, checked as a substring. They are in
`acceptance_corpus/` and are not to be edited to make a run pass — the
same rule `acceptance.py` already lives by, and for the same reason.

**And a second set nobody here wrote at all** (`--harvested`, built by
`scripts/harvest.py`): 355 closed issues paired with the pull requests
that closed them, so the question is a real person's words and the
answer is the file a maintainer actually changed. Blind is a discipline;
this removes the need to trust it. The authored set is better posed and
the harvested set cannot be accused of having been written by the party
being measured, so both are kept and neither replaces the other.

**The corpus is pinned.** The truth is `file:line`, and repositories
move. Every repo is checked out at the sha it had when its questions were
written; a floating clone would rot the answers silently.

**There is a control.** Every question also goes to ripgrep. A question
grep already answers is not evidence for an index, and without the
control this becomes another way of grading wsindex against itself.

Usage:
    uv run poe relevance          # routine tier, 60 questions
    uv run poe relevance --full   # adds dbeaver and icsharpcode, 84
    uv run poe relevance -- --harvested          # 204 real issue titles
    uv run poe relevance -- --harvested --full   # all 355
    uv run poe relevance -- --harvested --dump rows.json   # for a paired test
    uv run poe relevance -- --save scripts/relevance_baseline.json
    uv run poe relevance -- --check scripts/relevance_baseline.json

Environment:
    WSINDEX_RELEVANCE_DIR   corpus cache (default: ~/.cache/wsindex-relevance)
    WSINDEX_MODEL           model id to grade instead of the default
    WSINDEX_QUERY_PREFIX    that model's query instruction, if it wants one
    WSINDEX_TRUST_REMOTE    "1" to let that model run its own code
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wsindex.cli.composition import _reranker, build_store
from wsindex.config import Config, Repository
from wsindex.model import Hit, Kind, SearchFilter
from wsindex.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "acceptance_corpus"
CACHE = Path(os.environ.get("WSINDEX_RELEVANCE_DIR", Path.home() / ".cache" / "wsindex-relevance"))

Repo = dict[str, str]
"""One pinned repository from `acceptance_corpus/corpus.json`."""

Baseline = dict[str, object]
"""A saved run: the model it used, the tier, and the counts per class."""

DEFAULT_K = 10
"""What `wsindex search` gives a person who types nothing extra."""

DEEP_K = 50
"""The deep list, with history filtered out.

Not a proposal for a default — a measurement of what the index holds that
the default does not reach. The gap between the two columns is the cost
of the defaults, and it is the number step 3 of the plan has to move."""

DROWNED_AT = 20
"""Files `rg -l` may return before its answer stops being an answer.

Above this the developer is reading a list, which is the work ranking
exists to remove, so it counts as a partial win for the index rather than
a win for the control."""

RG = os.environ.get("WSINDEX_RG") or shutil.which("rg")
"""The control binary.

`WSINDEX_RG` exists because a machine may have ripgrep only as a shell
function forwarding to a bundled copy, which `which` cannot see. It is
invoked with `argv[0]` forced to `rg` (see `control`), so the escape hatch
works for a wrapper as well as for a real install. When neither is found
the report says the control did not run — scoring it as a miss would
hand wsindex a win it did not earn."""


@dataclass
class Graded:
    """One question, put to both tools."""

    id: str
    klass: str
    text: str
    truth: str
    reachable: bool
    rank: int | None
    deep_rank: int | None
    control: str
    control_files: int
    commits_in_top3: int

    @property
    def hit3(self) -> bool:
        return self.rank is not None and self.rank <= 3

    @property
    def hit10(self) -> bool:
        return self.rank is not None


def covered(pipeline: Pipeline, repo: str, path: str, lines: list[int]) -> bool:
    """Whether any indexed chunk holds the lines the answer lives on.

    Counted apart from ranking, because mixing them hides both. Twelve of
    the eighty-four questions are about files wsindex never indexed —
    eleven in an Elixir workspace with no grammar, one a `Makefile` — and
    scoring those as retrieval failures taxed every model the step 4
    spike graded, equally and invisibly. Worse, it would make a future
    fix to *coverage* look like a gain in *relevance*.

    Measured when this was added: no answer is missed for any other
    reason. Every file that is indexed has a chunk covering its answer,
    so the chunker is not where anything is lost.
    """
    spans = _spans(pipeline, repo)
    lo, hi = lines
    return any(start <= hi and end >= lo for start, end in spans.get(path, ()))


_SPANS: dict[str, dict[str, list[tuple[int, int]]]] = {}


def _spans(pipeline: Pipeline, repo: str) -> dict[str, list[tuple[int, int]]]:
    """Path -> the line ranges this repo has chunks for, read once."""
    if repo not in _SPANS:
        meta = pipeline.store.metadata_of(repo, ids=sorted(pipeline.store.chunk_ids(repo)))
        found: dict[str, list[tuple[int, int]]] = {}
        for entry in meta.values():
            found.setdefault(entry.path, []).append((entry.start_line, entry.end_line))
        _SPANS[repo] = found
    return _SPANS[repo]


@dataclass
class Workspace:
    org: str
    graded: list[Graded] = field(default_factory=list)
    files: int = 0
    chunks: int = 0
    seconds: float = 0.0


def run_git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def materialise(org: str, repos: list[Repo]) -> Path:
    """Clone or update the workspace, every repo at its pinned sha."""
    root = CACHE / org
    root.mkdir(parents=True, exist_ok=True)
    for repo in repos:
        into = root / repo["id"]
        if not (into / ".git").is_dir():
            print(f"    cloning {repo['id']}", flush=True)
            run_git("clone", "--quiet", "--no-tags", repo["url"], str(into))
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=into, capture_output=True, text=True
        ).stdout.strip()
        if not head.startswith(repo["sha"]):
            run_git("checkout", "--quiet", "--detach", repo["sha"], cwd=into)
    return root


def control(root: Path, rg_query: str, truth: str) -> tuple[str, int]:
    """The same question put to ripgrep, exactly as the tester recorded it."""
    if RG is None:
        return "no ripgrep", 0
    argv = shlex.split(rg_query)
    if argv and argv[0] == "rg":
        argv = argv[1:]
    if "-l" not in argv and "--files-with-matches" not in argv:
        argv.append("-l")
    # An explicit path, and it is load-bearing: `rg pattern` with no path
    # searches stdin when stdin is not a terminal, so under a pipe the
    # control waits for input that never comes.
    if not any(a == "." for a in argv):
        argv.append(".")
    # `argv[0]` forced to `rg` rather than to the binary's own path: that
    # is what makes a multi-call wrapper behave as ripgrep.
    done = subprocess.run(
        ["rg", *argv],
        executable=RG,
        cwd=root,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )
    files = [line.strip().removeprefix("./") for line in done.stdout.splitlines() if line.strip()]
    if truth not in files:
        return "missed", len(files)
    return ("found" if len(files) <= DROWNED_AT else "drowned"), len(files)


_STOP = {
    "about",
    "after",
    "again",
    "against",
    "always",
    "because",
    "before",
    "behaviour",
    "behavior",
    "between",
    "cannot",
    "could",
    "doesn",
    "during",
    "error",
    "should",
    "support",
    "there",
    "using",
    "version",
    "issue",
    "problem",
    "returns",
    "something",
    "unexpected",
    "without",
    "working",
    "works",
    "would",
}
"""English a bug report is made of. Grepping `unexpected` measures the
size of the repository, not whether ripgrep can find the answer."""

_QUOTED = re.compile(r"[`\"']([^`\"']{3,60})[`\"']")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def terms(question: str) -> list[str]:
    """What a developer would plausibly grep, most specific first.

    Backticked strings first — a bug report quotes the error it saw, and
    that is the strongest thing anybody has to search with. Then
    identifier-shaped words, then long words. English is dropped.

    Frozen as a rule rather than chosen per question on purpose: a
    control written by somebody who has seen the answer is not a control.
    """
    found: list[str] = [match.group(1).strip() for match in _QUOTED.finditer(question)]
    words = [w for w in _WORD.findall(question) if w.lower() not in _STOP]
    found += [w for w in words if "_" in w or re.search(r"[a-z][A-Z]", w)]
    found += sorted((w for w in words if len(w) >= 6), key=len, reverse=True)
    return list(dict.fromkeys(found))[:8]


def best_control(root: Path, question: str, truth: str) -> tuple[str, int, str]:
    """Ripgrep's best single query drawn from the question's own words.

    Every candidate term is tried and the *friendliest* outcome to
    ripgrep is reported — found beats drowned beats missed, fewer files
    breaks a tie. That is deliberately generous. A harvested question has
    no hand-written `rg_query`, and inventing a weak one would make
    wsindex look good by beating a control nobody would have typed.

    Returns:
        Outcome, how many files that query returned, and the query.
    """
    rank = {"found": 0, "drowned": 1, "missed": 2}
    best = ("missed", 0, "")
    for term in terms(question):
        outcome, count = control(root, f"rg -i -F {shlex.quote(term)}", truth)
        if (rank[outcome], count) < (rank[best[0]], best[1] if best[2] else 10**9):
            best = (outcome, count, term)
    return best


def rank_of(hits: list[Hit], truth: str) -> int | None:
    for index, hit in enumerate(hits, start=1):
        meta = hit.metadata
        if f"{meta.get('repo')}/{meta.get('path')}" == truth:
            return index
    return None


def build(org: str, repos: list[Repo], root: Path) -> Pipeline:
    """A pipeline over one freshly indexed workspace.

    Split from `grade` so a probe can reuse the wiring without copying
    it: a probe that builds its own slightly different pipeline is a
    probe measuring something slightly different.
    """
    # `Config.default` replaces the process-wide instance, which is what a
    # script wants: it must never pick up a real workspace config.
    config = Config.default(f"relevance-{org}")
    # Overridable so a model can be graded through the real configuration
    # path rather than a spike's own wiring — the point of step 5 is that
    # choosing a model is a config line, and this proves it is one.
    if os.environ.get("WSINDEX_MODEL"):
        # Reaching into the document rather than through a setter, and
        # saying so: `Config` has no writer for these, because a workspace
        # writes them once by hand. A measuring script is the one caller
        # that wants to change them per run.
        config._data["embeddings"].update(
            {
                "model": os.environ["WSINDEX_MODEL"],
                "dim": int(os.environ.get("WSINDEX_DIM", "768")),
                "query_prefix": os.environ.get("WSINDEX_QUERY_PREFIX", ""),
                "trust_remote_code": os.environ.get("WSINDEX_TRUST_REMOTE") == "1",
                **(
                    {"max_seq": int(os.environ["WSINDEX_MAX_SEQ"])}
                    if os.environ.get("WSINDEX_MAX_SEQ")
                    else {}
                ),
            }
        )
    if os.environ.get("WSINDEX_PROVIDER"):
        # The hosted path, same three settings a workspace would write.
        # Measuring it through a probe's own client would measure the
        # probe; this way the thing graded is the thing that ships.
        config._data["embeddings"].update(
            {
                "provider": os.environ["WSINDEX_PROVIDER"],
                "url": os.environ.get("WSINDEX_EMBED_URL", ""),
                "token_env": os.environ.get("WSINDEX_EMBED_TOKEN_ENV", ""),
                "input_types": os.environ.get("WSINDEX_INPUT_TYPES") == "1",
            }
        )
    if os.environ.get("WSINDEX_RANK_MODEL"):
        config._data["rank"] = {
            "enabled": True,
            "model": os.environ["WSINDEX_RANK_MODEL"],
            "provider": os.environ.get("WSINDEX_RANK_PROVIDER", "sentence-transformers"),
            "url": os.environ.get("WSINDEX_RANK_URL", ""),
            "token_env": os.environ.get("WSINDEX_RANK_TOKEN_ENV", ""),
        }
    for repo in repos:
        config.add_repo(Repository(id=repo["id"], path=str(root / repo["id"])))
    state = CACHE / ".index" / org
    shutil.rmtree(state, ignore_errors=True)
    state.mkdir(parents=True, exist_ok=True)
    config._data["store"]["uri"] = str(state / "data.lance")
    # Built by the composition root, not here. The real model, the real
    # provider switch, the real reranker wiring — so that changing a
    # default changes what this measures, which is the point of the
    # instrument. A fake embedder, or a second hand-rolled one, would make
    # it grade its own fixture: the failure mode it exists to catch.
    return Pipeline(
        store=build_store(config),
        config=config,
        state_dir=state,
        reranker=_reranker(config),
    )


def grade(org: str, repos: list[Repo], *, harvested: bool = False) -> Workspace:
    """Index one workspace and put its frozen questions to both tools.

    Args:
        org: Which workspace.
        repos: Its pinned repositories.
        harvested: Ask the questions real people asked — issue titles
            paired with the files that closed them (`scripts/harvest.py`)
            — instead of the ones written for this corpus. The reason to
            have both: the authored set is better posed and the harvested
            set is not vulnerable to having been written by the person
            being measured, which is the objection that ends an argument
            about the authored one.
    """
    root = materialise(org, repos)
    source = (CORPUS / "harvested" / f"{org}.json") if harvested else (CORPUS / f"{org}.json")
    questions = json.loads(source.read_text(encoding="utf-8"))["questions"]
    if harvested:
        return _grade_harvested(org, repos, root, questions)
    pipeline = build(org, repos, root)
    started = time.perf_counter()
    report = pipeline.index()
    space = Workspace(
        org=org,
        files=report.files,
        chunks=report.chunks,
        seconds=round(time.perf_counter() - started, 1),
    )

    code_and_doc = SearchFilter(kind=(Kind.CODE, Kind.DOC))
    for item in questions:
        truth = f"{item['truth']['repo']}/{item['truth']['path']}"
        hits = pipeline.search(item["text"], k=DEFAULT_K)
        deep = pipeline.search(item["text"], k=DEEP_K, filters=code_and_doc)
        outcome, count = control(root, item["rg_query"], truth)
        space.graded.append(
            Graded(
                id=item["id"],
                klass=item["class"],
                text=item["text"],
                truth=truth,
                reachable=covered(
                    pipeline, item["truth"]["repo"], item["truth"]["path"], item["truth"]["lines"]
                ),
                rank=rank_of(hits, truth),
                deep_rank=rank_of(deep, truth),
                control=outcome,
                control_files=count,
                commits_in_top3=sum(
                    1 for h in hits[:3] if str(h.metadata.get("path", "")).startswith("commits/")
                ),
            )
        )
    return space


def _grade_harvested(
    org: str, repos: list[Repo], root: Path, questions: list[dict[str, Any]]
) -> Workspace:
    """The same grading, for questions nobody here wrote.

    Three things differ, and each is the honest handling rather than a
    convenience:

    - **The answer is a set.** A pull request that closes an issue often
      touches two or three files and all of them are where to look, so a
      hit is the best rank among them. The authored set has one file and
      a line range; these have neither.
    - **Reachability is per file.** Without line numbers, "did the
      indexer read this at all" is the only reachability question that
      can be asked, and it is still worth asking — a question about a
      file no grammar covers is unanswerable for reasons that have
      nothing to do with ranking.
    - **The control is derived**, by `best_control`, since there is no
      hand-written `rg_query` and writing one now would mean writing it
      with the answer in view.
    """
    pipeline = build(org, repos, root)
    started = time.perf_counter()
    report = pipeline.index()
    space = Workspace(
        org=org,
        files=report.files,
        chunks=report.chunks,
        seconds=round(time.perf_counter() - started, 1),
    )
    code_and_doc = SearchFilter(kind=(Kind.CODE, Kind.DOC))
    indexed = {
        f"{repo['id']}/{meta.path}"
        for repo in repos
        for meta in pipeline.store.metadata_of(
            repo["id"], ids=sorted(pipeline.store.chunk_ids(repo["id"]))
        ).values()
    }
    for item in questions:
        answers = [item["truth"], *item.get("also_valid", [])]
        hits = pipeline.search(item["text"], k=DEFAULT_K)
        deep = pipeline.search(item["text"], k=DEEP_K, filters=code_and_doc)
        ranks = [r for r in (rank_of(hits, a) for a in answers) if r is not None]
        deep_ranks = [r for r in (rank_of(deep, a) for a in answers) if r is not None]
        outcome, count, query = best_control(root, item["text"], item["truth"])
        space.graded.append(
            Graded(
                id=item["id"],
                klass="harvested",
                text=item["text"],
                truth=item["truth"],
                reachable=any(a in indexed for a in answers),
                rank=min(ranks) if ranks else None,
                deep_rank=min(deep_ranks) if deep_ranks else None,
                control=outcome,
                control_files=count,
                commits_in_top3=sum(
                    1 for h in hits[:3] if str(h.metadata.get("path", "")).startswith("commits/")
                ),
            )
        )
        del query
    return space


def report_on(spaces: list[Workspace]) -> str:
    every = [g for space in spaces for g in space.graded]
    lines = ["# Relevance report", ""]
    lines.append(
        f"_{time.strftime('%Y-%m-%d %H:%M')}, {len(every)} questions, "
        f"{len(spaces)} workspaces, pinned corpus._"
    )
    if RG is None:
        lines += ["", "**No ripgrep on this machine — the control did not run.**"]
    lines += [
        "",
        "| Workspace | Files | Chunks | Index | hit@3 | hit@10 | deep | rg found |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for space in spaces:
        n = len(space.graded) or 1
        lines.append(
            f"| `{space.org}` | {space.files} | {space.chunks} | {space.seconds} s | "
            f"{sum(g.hit3 for g in space.graded)}/{n} | {sum(g.hit10 for g in space.graded)}/{n} | "
            f"{sum(g.deep_rank is not None for g in space.graded)}/{n} | "
            f"{sum(g.control == 'found' for g in space.graded)}/{n} |"
        )
    lines += [
        "",
        "## By class",
        "",
        "Reachable means some indexed chunk holds the lines the answer is on. "
        "Anything else is a coverage failure in a retrieval failure's clothes, so "
        "it is counted beside the score rather than folded into it.",
        "",
        "| Class | Questions | Reachable | hit@1 | hit@3 | hit@10 | deep | rg found |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    # Whatever classes this run actually produced, in a fixed order, so a
    # harvested run reports instead of reporting nothing. The first
    # version listed the three authored classes by name and printed an
    # empty table for 204 harvested questions.
    order = {"literal": 0, "descriptive": 1, "cross-repo": 2, "harvested": 3}
    for klass in sorted({g.klass for g in every}, key=lambda k: order.get(k, 9)):
        group = [g for g in every if g.klass == klass]
        if not group:
            continue
        n = len(group)
        lines.append(
            f"| {klass} | {n} | {sum(g.reachable for g in group)} | "
            f"{sum(g.rank == 1 for g in group)} | {sum(g.hit3 for g in group)} | "
            f"{sum(g.hit10 for g in group)} | {sum(g.deep_rank is not None for g in group)} | "
            f"{sum(g.control == 'found' for g in group)} |"
        )
    unreachable = [g for g in every if not g.reachable]
    if unreachable:
        lines += [
            "",
            f"**{len(unreachable)} of {len(every)} questions are unreachable**: no chunk "
            "covers the answer, because the file was never indexed. Not a model's fault, "
            "and no model can move them.",
            "",
            *[f"- `{g.id}` {g.truth}" for g in unreachable],
        ]
    # The class the gate is about: the authored questions grep cannot
    # answer, or — in a harvested run, where every question came from an
    # issue tracker and none was written here — all of them.
    hard = "descriptive" if any(g.klass == "descriptive" for g in every) else "harvested"
    said = [g for g in every if g.klass == hard]
    reachable = [g for g in said if g.reachable]
    if said and reachable:
        share = sum(g.hit10 for g in reachable) / len(reachable)
        found = sum(g.control == "found" for g in said)
        lines += [
            "",
            "## The gate",
            "",
            f"`{hard}` questions: ripgrep found {found} of {len(said)}. The plan of "
            f"2026-09-11 set `hit@10 >= 0.5` on the descriptive class as the line "
            f"between finishing the tool as promised and repositioning it.",
            "",
            f"**hit@10 = {share:.2f}** on {len(reachable)} reachable of {len(said)}.",
        ]
        if hard == "harvested":
            lines += [
                "",
                "These questions were not written here. Each is the title of a closed "
                "issue, and the answer is the file the pull request that closed it "
                "changed — so neither end was produced by anyone measuring this tool. "
                "The control is derived from the question's own words by a frozen rule "
                "(`terms`) and reports ripgrep's *best* attempt, because beating a "
                "query nobody would have typed proves nothing.",
            ]
    lines += ["", "## Misses worth reading", ""]
    for g in every:
        if g.klass == hard and not g.hit10 and g.deep_rank:
            lines.append(
                f"- `{g.id}` {g.truth} — missed at k={DEFAULT_K}, "
                f"rank {g.deep_rank} in the deep list. _{g.text}_"
            )
    return "\n".join(lines) + "\n"


def counted(spaces: list[Workspace]) -> dict[str, int]:
    """Answers found per class — the whole of what a baseline compares."""
    found: dict[str, int] = {}
    for space in spaces:
        for graded in space.graded:
            found[graded.klass] = found.get(graded.klass, 0) + graded.hit10
            found[graded.klass + "@3"] = found.get(graded.klass + "@3", 0) + graded.hit3
    return found


def compare(now: dict[str, int], saved: Baseline) -> list[str]:
    """Every class that answers fewer questions than the baseline did.

    No tolerance band, unlike `scripts/bench.py`: that one measures time,
    which is noisy, and this one measures which file came back, on a
    corpus pinned to a sha with a fixed model. Repeated runs of it have
    been identical. One answer fewer is a regression, not a fluctuation.
    """
    lines: list[str] = []
    was_model = saved.get("model")
    if was_model and was_model != Config().model:
        # A different model is a different question, not a worse answer.
        lines.append(
            f"- **the baseline is from another model**: `{was_model}`, "
            f"now `{Config().model}` — read the rest as trivia"
        )
    found = saved.get("found") or {}
    assert isinstance(found, dict)
    for klass, before in sorted(found.items()):
        after = now.get(klass, 0)
        if after < before:
            lines.append(f"- **{klass}**: {before} -> {after}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="add the two large workspaces (84 questions instead of 60)",
    )
    parser.add_argument(
        "--harvested",
        action="store_true",
        help="ask the questions real people asked (scripts/harvest.py) instead",
    )
    parser.add_argument(
        "--dump",
        type=Path,
        help="Write one row per question, for a paired test the summary cannot support",
    )
    parser.add_argument("--save", type=Path, help="Write these counts as a baseline")
    parser.add_argument(
        "--check",
        type=Path,
        help="Fail if any class answers fewer questions than the baseline",
    )
    args = parser.parse_args()

    corpus = json.loads((CORPUS / "corpus.json").read_text(encoding="utf-8"))
    wanted = {org: c for org, c in corpus.items() if args.full or c["tier"] == "routine"}
    print(f"{len(wanted)} workspaces, corpus cache at {CACHE}", flush=True)

    spaces = []
    for org, c in wanted.items():
        print(f"  {org}", flush=True)
        spaces.append(grade(org, c["repos"], harvested=args.harvested))
    text = report_on(spaces)
    Path("relevance_report.md").write_text(text, encoding="utf-8")
    print("\n" + text)

    if args.dump:
        # Per question, because every comparison worth making here is
        # paired — the same question put to both tools — and a table of
        # totals cannot tell "both found it" from "each found a
        # different half", which is the only thing McNemar reads.
        args.dump.write_text(
            json.dumps(
                [{"org": space.org, **g.__dict__} for space in spaces for g in space.graded],
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"per-question rows written to {args.dump}")

    found = counted(spaces)
    if args.save:
        args.save.write_text(
            json.dumps(
                {"model": Config().model, "full": args.full, "found": found},
                indent=1,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"baseline written to {args.save}")
    if args.check:
        saved = json.loads(args.check.read_text(encoding="utf-8"))
        if saved.get("full") != args.full:
            print(
                f"\nbaseline is for {'--full' if saved.get('full') else 'the routine tier'}; "
                "run the same tier to compare"
            )
            return 1
        worse = compare(found, saved)
        if worse:
            print("\n## Fewer answers than the baseline\n\n" + "\n".join(worse))
            return 1
        print("\nNo class answers fewer questions than the baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
