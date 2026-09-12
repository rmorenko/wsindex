"""Questions asked by people who had never heard of this tool.

The 84 questions in this corpus were written by the person who built
wsindex, after reading the repositories. Every guard around them — the
substring leak rule, freezing them before a run, the ripgrep control —
narrows that problem and none of them removes it. A sceptic can discard
the whole result in one sentence, and would be right to.

This removes it at the source. A closed issue is a question a real
person asked about a real codebase, in their words, before any of this
existed; the pull request that closed it names the files that answer it.
Neither end was written for a search engine.

The rules below are the methodology, and they are frozen here rather
than applied by hand:

- **The pull request must be merged before the pinned commit.** Then the
  code that answers the question is in the tree this corpus indexes.
  (The other direction — harvest fixes merged *after* the pin, so the
  answer cannot have been seen — is the cleaner design and is not
  available: these repositories were pinned days ago and almost nothing
  has landed since.)
- **It must close an issue explicitly**, by `fixes #n` or a synonym, so
  the pairing is the author's claim and not a guess.
- **One to five changed files**, all of which still exist at the pin. A
  pull request touching thirty files has no single answer, and a file
  deleted since is not in the corpus to be found.
- **Not test-only or documentation-only.** "Where do I look" should land
  on the thing that implements the behaviour.
- **The leak rule, unchanged:** no path component of the answer may
  appear as a substring of the question. An issue titled "caddyfile
  parsing is wrong" pointing at `caddyfile.go` tests the filesystem, not
  retrieval.
- **Bots are excluded**, by author type and by a title pattern, because
  a dependabot title is not a question.

Everything is recorded — issue number, pull request number, merge date,
the whole changed-file list — so any claim made from these questions can
be traced back to a url a reader can open.

Usage:
    uv run python scripts/harvest.py            # every org in the corpus
    uv run python scripts/harvest.py caddyserver
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CORPUS = Path(__file__).parent / "acceptance_corpus"
OUT = CORPUS / "harvested"

CACHE = Path(os.environ.get("WSINDEX_RELEVANCE_DIR", Path.home() / ".cache" / "wsindex-relevance"))
"""Where `relevance.py` materialises the pinned repositories. The same
place on purpose: this reads the trees it has already checked out at the
pinned commit, which is how "does this file still exist at the pin" is
answered without a call per file."""

CLOSES = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)\b", re.IGNORECASE)
"""GitHub's own closing keywords. Using them rather than inferring a
pairing means the link is something the pull request author asserted."""

BOT_TITLE = re.compile(r"^(?:bump|chore\(deps\)|build\(deps\)|update .* to v?\d)", re.IGNORECASE)
"""A dependency bump is not a question anybody asked."""

MAX_FILES = 5
"""Above this a pull request is a refactor, not an answer to a question."""

PAGES = 4
"""Pages of 100 merged pull requests to scan per repository. Enough to
reach a year or two back on an active project, and a hard bound on what
one run costs against a 5 000/hour rate limit."""

TEST_OR_DOC = re.compile(r"(^|/)(tests?|testdata|spec|docs?|examples?)(/|$)|_test\.|\.md$")
"""Where an answer should not land. A test asserts the behaviour; the
question is where the behaviour lives."""

CODE_SUFFIXES = {
    ".go",
    ".py",
    ".rb",
    ".rs",
    ".java",
    ".cs",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".ex",
    ".exs",
    ".kt",
    ".swift",
    ".php",
    ".scala",
}


def gh(endpoint: str, **params: Any) -> Any:
    """One GitHub API call, as json.

    Through the `gh` CLI rather than a token in this process: it already
    holds the credential, which keeps this script from being one more
    thing that wants a secret.
    """
    query = "".join(f"&{key}={value}" for key, value in params.items())
    joined = f"{endpoint}?{query[1:]}" if query else endpoint
    done = subprocess.run(
        ["gh", "api", joined], capture_output=True, text=True, stdin=subprocess.DEVNULL
    )
    if done.returncode != 0:
        return None
    return json.loads(done.stdout)


@dataclass
class Harvested:
    """One question, with everything needed to check it was not invented.

    Attributes:
        id: Stable within an org file.
        text: The issue title, as its author typed it.
        truth: `repo/path` of the file the fix changed most. Single,
            because the scorer is, and the harsher choice of the two:
            being counted wrong for finding another file the same fix
            touched understates the tool rather than flattering it.
        also_valid: The rest of the changed files. Kept so a scorer that
            accepts any of them can be written without harvesting again.
        source: Issue and pull request urls, and the merge date.
    """

    id: str
    text: str
    truth: str
    also_valid: list[str] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)


def leaks(question: str, paths: list[str]) -> bool:
    """Does the question give away where the answer is?

    Substring, not whole word: `search` hides inside `ElasticSearch`, and
    an earlier version of this rule using word boundaries let three
    question sets through that should have been rewritten.
    """
    lowered = question.lower()
    for path in paths:
        for part in re.split(r"[/._-]", path):
            if len(part) >= 4 and part.lower() in lowered:
                return True
    return False


def usable(files: list[dict[str, Any]], root: Path) -> list[str]:
    """The changed files that can be an answer, largest change first.

    Ordered by how much of the file the fix touched, so `truth` is the
    file the change is most about rather than whichever GitHub listed
    first.
    """
    kept = []
    for entry in files:
        path = entry["filename"]
        if TEST_OR_DOC.search(path) or Path(path).suffix not in CODE_SUFFIXES:
            continue
        if not (root / path).is_file():
            # Renamed or deleted since. Not in the corpus, so not findable.
            continue
        kept.append((entry.get("changes", 0), path))
    kept.sort(reverse=True)
    return [path for _, path in kept]


def harvest_repo(owner_repo: str, repo_id: str, *, sha: str, root: Path) -> list[Harvested]:
    """Every usable issue/PR pair in one repository."""
    pinned = subprocess.run(
        ["git", "show", "-s", "--format=%cI", sha],
        cwd=root,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    ).stdout.strip()
    found: list[Harvested] = []
    seen: set[int] = set()
    for page in range(1, PAGES + 1):
        pulls = gh(
            f"repos/{owner_repo}/pulls",
            state="closed",
            per_page=100,
            page=page,
            sort="updated",
            direction="desc",
        )
        if not pulls:
            break
        for pull in pulls:
            merged = pull.get("merged_at")
            if not merged or (pinned and merged > pinned):
                continue
            match = CLOSES.search(f"{pull.get('title', '')}\n{pull.get('body') or ''}")
            if not match:
                continue
            number = int(match.group(1))
            if number in seen:
                continue
            seen.add(number)
            issue = gh(f"repos/{owner_repo}/issues/{number}")
            if not issue or issue.get("pull_request") or issue.get("user", {}).get("type") == "Bot":
                continue
            title = (issue.get("title") or "").strip()
            if len(title) < 20 or BOT_TITLE.match(title):
                continue
            files = gh(f"repos/{owner_repo}/pulls/{pull['number']}/files", per_page=100)
            if not files:
                continue
            paths = usable(files, root)
            if not 1 <= len(paths) <= MAX_FILES or leaks(title, paths):
                continue
            found.append(
                Harvested(
                    id=f"{repo_id}-{number}",
                    text=title,
                    truth=f"{repo_id}/{paths[0]}",
                    also_valid=[f"{repo_id}/{path}" for path in paths[1:]],
                    source={
                        "issue": f"https://github.com/{owner_repo}/issues/{number}",
                        "pull": f"https://github.com/{owner_repo}/pull/{pull['number']}",
                        "merged_at": merged,
                    },
                )
            )
    return found


def main(only: str | None) -> None:
    corpus = json.loads((CORPUS / "corpus.json").read_text())
    OUT.mkdir(exist_ok=True)
    for org, spec in corpus.items():
        if only and org != only:
            continue
        questions: list[Harvested] = []
        for repo in spec["repos"]:
            root = CACHE / org / repo["id"]
            if not root.is_dir():
                print(f"  {repo['id']}: not materialised, skipping", file=sys.stderr)
                continue
            owner_repo = repo["url"].removeprefix("https://github.com/").removesuffix(".git")
            got = harvest_repo(owner_repo, repo["id"], sha=repo["sha"], root=root)
            print(f"  {repo['id']:<18} {len(got)}", file=sys.stderr)
            questions += got
        (OUT / f"{org}.json").write_text(
            json.dumps(
                {"org": org, "harvested": True, "questions": [q.__dict__ for q in questions]},
                indent=2,
            )
            + "\n"
        )
        print(f"{org}: {len(questions)} questions", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
