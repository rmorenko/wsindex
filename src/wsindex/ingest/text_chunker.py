"""Text chunking: Markdown by headers, everything else by sliding window.

Serves DOC files and is the fallback for every language without an AST
chunker (ARCH §4). Chunk text is always a
verbatim slice of the file — `text` corresponds exactly to lines
`start_line..end_line` — so a search hit can point back at the real location.
"""

import itertools

from wsindex.model import Chunk, SourceFile

WINDOW_LINES = 40
OVERLAP_LINES = 10

MAX_CHUNK_CHARS = 900
"""Characters after which a chunk is cut short, whatever its line count.

**A window measured only in lines is measured in the wrong unit.** The
embedding model reads 256 tokens and stops; everything past that is not
in the vector, so no query can reach it. Measured within one run, a line
the model read is found by its own words 67 times in 133 at median depth
2, while a line past the cut is found 11 times in 60 at median depth 33 —
three times less findable, sixteen times deeper.

And it was not a small edge. Counted exactly over two real repositories,
**74.1% and 32.5% of the indexed text sat past the cut** before this
existed. Of that, the share this cap can reach — windows and the gap pass
— was 38.0% and 5.2%; the cap takes those to 13.5% and **0.1%**. What is
left is inside AST definitions, which stay whole on purpose: a function
is a unit somebody wrote, and halving it changes what a hit means. That
residue is the next question, not this one.

Nine hundred rather than seven, measured the same way: 700 recovers two
further points on the harder corpus and costs 9% more chunks. The whole
cap costs 5% more chunks on a corpus of ordinary code and 41% on one full
of minified vendored assets, where it also cannot finish the job —
openemr's remaining window loss is single lines longer than the cap,
admitted whole because a chunk's text has to stay a verbatim slice of its
line range.

**The first calibration of this number was wrong and is worth recording.**
It asked what share of chunks *shorter* than the cap came under 256
tokens and got 99.9% — a question answered by the thousands of tiny
chunks, not by the ones the cap actually produces, which all sit at it.
The metric was wrong twice over, too: "share of chunks over the limit"
*rises* when a cap splits one enormous chunk into many near-limit ones,
while the text actually lost falls. Content is what matters, so content
is what is counted.

**This encodes a property of the model, not of the workspace.** 256
tokens belongs to `all-MiniLM-L6-v2`. If the default model changes, this
number has to be re-measured — by counting unreachable characters, not
chunks."""


def chunk_plain(
    text: str,
    source: SourceFile,
    *,
    window: int = WINDOW_LINES,
    overlap: int = OVERLAP_LINES,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[Chunk]:
    """Split text into windows of at most `window` lines and `max_chars` characters.

    Whichever limit comes first ends the window. Overlap keeps passages that
    straddle a boundary retrievable from at least one chunk, and stays
    proportional when the character cap cuts a window short: a window of the
    full `window` lines overlaps by exactly `overlap`, and a shorter one by the
    same quarter. A fixed overlap would otherwise turn a five-line window into
    eighty percent repetition.

    A single line longer than `max_chars` is admitted whole rather than split,
    because `text` has to stay a verbatim slice of lines `start_line..end_line`.

    Whitespace-only windows are dropped.
    """
    if overlap >= window:
        raise ValueError("Overlap must be less than window")
    lines = text.splitlines()
    result: list[Chunk] = []
    start = 0
    while start < len(lines):
        end = _window_end(lines, start, window=window, max_chars=max_chars)
        window_text = "\n".join(lines[start:end])
        if window_text.strip():
            result.append(source.chunk(text=window_text, start_line=start + 1, end_line=end))
        if end >= len(lines):
            break
        taken = end - start
        # `taken // 4` reproduces the configured overlap exactly at the full
        # window (40 lines, 10 over) and scales it down with the window,
        # so nothing about the ordinary case changes.
        start += max(1, taken - min(overlap, taken // 4))
    return result


def _window_end(lines: list[str], start: int, *, window: int, max_chars: int) -> int:
    """Exclusive index where the window starting at `start` has to stop.

    Always at least one line past `start`: a line longer than `max_chars`
    on its own cannot be split without breaking the promise that a chunk's
    text is a verbatim slice of its line range.
    """
    size = 0
    end = start
    last = min(start + window, len(lines))
    while end < last:
        # The newline that `join` will put back, for every line but the first.
        grown = size + len(lines[end]) + (1 if end > start else 0)
        if end > start and grown > max_chars:
            break
        size = grown
        end += 1
    return end


def chunk_markdown(text: str, source: SourceFile) -> list[Chunk]:
    """Split Markdown into sections: one chunk per header, plus the preamble.

    A `#` line inside a fenced code block does not start a section. The header
    text becomes the chunk's `symbol`; the preamble (text before the first
    header) has no symbol.
    """
    chunks: list[Chunk] = []
    in_fence = False
    header_idx: list[int] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("```"):
            in_fence = not in_fence
        if (line.startswith("#") and not in_fence) or idx == 0:
            header_idx.append(idx)
    boundaries = [*header_idx, len(lines)]
    for s, e in itertools.pairwise(boundaries):
        section_text = "\n".join(lines[s:e])
        if not section_text.strip():
            continue
        start_line = s + 1
        end_line = e
        first = lines[s]
        symbol = first.lstrip("#").strip() if first.startswith("#") else None
        chunks.append(
            source.chunk(
                text=section_text,
                start_line=start_line,
                end_line=end_line,
                symbol=symbol,
                node_type="section",
            )
        )
    return chunks


def chunk_text(text: str, source: SourceFile) -> list[Chunk]:
    """Dispatch by lang: Markdown gets section chunking, the rest sliding windows."""
    if source.lang == "markdown":
        return chunk_markdown(text, source)
    return chunk_plain(text, source)
