"""Blaming a batch of files — here, or in a small process of its own.

**Nothing from `wsindex` is imported here, and that is the point.** This
module is run two ways: imported normally by `commits.py`, and executed
*by path* as a child (`python .../blame.py`), which skips the package
entirely. `import wsindex.ingest` costs 30 ms of somebody else's imports,
measured, and a helper that pays them is a helper that stops being worth
starting.

Why a child at all. `git blame` is per file, so an index run starts one
process per indexed file — and starting a process *from the process that
holds the embedding model* is where the time goes. Measured on macOS: 117
spawns of `git --version`, a command that does nothing, block a search in
the same process for 1.4 s, and eight threads doing it buy no parallelism
at all (9.7 ms per spawn either way). Moving the spawning into a small
child makes the same storm 27% faster (0.49 s → 0.36 s) *and* leaves the
searcher at its idle latency (487.8 ms → 8.2 ms p50, 14.2 ms worst); with
nobody searching it costs nothing (0.35 s against 0.35 s).

What is *not* established is why that process is a bad one to fork from.
A 1 GB single allocation does not reproduce it and four thousand small
mappings barely do (0.80 s → 0.89 s), so the tidy explanation — "torch
has more memory regions" — is a guess and is not relied on here. The
shape is what was measured, and the shape is enough to act on.

One implementation, two callers, which is why `blame` arrives as an
argument: the parent hands in `run_git` and keeps every guarantee that
function makes about invoking git, while the child hands in the plain one
below. Neither is a second copy of how to read porcelain output.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BLAME_LINE = re.compile(r"^([0-9a-f]{40}) \d+ (\d+)")
"""A porcelain header line: `<sha> <original line> <final line>`."""

Blamer = Callable[[Path, str], bytes | None]
"""Runs `git blame --porcelain` on one file.

Returns raw stdout, or **None** when git refused the file — an untracked
one is in no commit (`fatal: no such path ... in HEAD`), which is a
normal answer rather than a failure. None instead of an exception so that
no exception class has to cross between the parent's git layer and the
child's."""


def blame_files(
    root: Path, paths: Sequence[str], *, workers: int, blame: Blamer
) -> dict[str, dict[int, str]]:
    """Blame every path: `path -> {line -> sha}`.

    Args:
        root: Repository root.
        paths: Files to blame, repo-relative.
        workers: How many at once.
        blame: How to run one; see `Blamer`.

    Returns:
        One entry per path, in the order given. A file with no history
        maps to an empty dict.

    Raises:
        RuntimeError: One file could not be blamed, named. A failure in a
            worker thread does reach the caller — `map` re-raises when
            the results are walked — but it arrives with a traceback
            through `concurrent.futures` and no idea which of a hundred
            files caused it. The name is added where it is known.
    """
    if not paths:
        return {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        blamed = pool.map(_named(root, blame), paths)
        return dict(zip(paths, blamed, strict=True))


def parse(raw: bytes) -> dict[int, str]:
    """Line number -> the sha that last wrote it, from porcelain output.

    One `git blame` per file rather than one per chunk: the porcelain
    output already covers the whole file, and a chunk-sized `-L` range
    would pay the process cost once per chunk instead of once per file.
    """
    by_line: dict[int, str] = {}
    # `surrogateescape`, like every other path git hands back: a file name
    # that is not UTF-8 must survive the round trip rather than raise.
    for line in raw.decode("utf-8", errors="surrogateescape").splitlines():
        header = BLAME_LINE.match(line)
        if header is not None:
            by_line[int(header.group(2))] = header.group(1)
    return by_line


def _named(root: Path, blame: Blamer) -> Callable[[str], dict[int, str]]:
    """One file's blame, with the path attached to any failure."""

    def one(path: str) -> dict[int, str]:
        try:
            raw = blame(root, path)
        except Exception as exc:
            raise RuntimeError(f"blaming {path}: {type(exc).__name__}: {exc}") from exc
        return parse(raw) if raw is not None else {}

    return one


# --- the child ------------------------------------------------------------


def plain_git(root: Path, path: str, *, timeout: float) -> bytes | None:
    """`git blame --porcelain`, with only the standard library.

    The parent's `run_git` is the one place that knows how to invoke git
    safely, and this is deliberately not a second one: it makes no policy
    at all. The read-only lock hint is inherited from the environment the
    parent started this process with, and the timeout arrives in the
    request, so neither is decided twice.

    Anything unusual — git missing, a timeout — is left to raise, which
    ends the child; the parent then does the whole batch itself and gets
    the real error from `run_git` with its proper type. A child that tried
    to report those faithfully would be inventing an error protocol to
    replace one that already works.
    """
    finished = subprocess.run(
        ["git", "blame", "--porcelain", "--", path],
        cwd=root,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    return finished.stdout if finished.returncode == 0 else None


def main() -> int:
    """Read one request from stdin, write the blame map to stdout.

    JSON both ways, and ASCII-escaped by default, which matters: a path
    git handed back may hold lone surrogates, and those survive
    `\\udcXX` escaping where they would not survive being encoded.
    """
    request = json.loads(sys.stdin.read())
    timeout = float(request["timeout"])
    blamed = blame_files(
        Path(request["root"]),
        list(request["paths"]),
        workers=int(request["workers"]),
        blame=lambda root, path: plain_git(root, path, timeout=timeout),
    )
    json.dump(
        {path: {str(n): sha for n, sha in lines.items()} for path, lines in blamed.items()},
        sys.stdout,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - only when run by path, as a child
    raise SystemExit(main())
