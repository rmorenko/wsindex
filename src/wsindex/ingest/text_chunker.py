"""Text chunking for doc files: Markdown by headers, everything else by window.

Second stage of the indexing pipeline (ARCH §4). Chunk text is always a
verbatim slice of the file — `text` corresponds exactly to lines
`start_line..end_line` — so a search hit can point back at the real location.
"""

import itertools

from wsindex.model import Chunk, Kind

WINDOW_LINES = 40
OVERLAP_LINES = 10


def chunk_plain(
    text: str,
    *,
    repo: str,
    path: str,
    lang: str,
    kind: Kind,
    window: int = WINDOW_LINES,
    overlap: int = OVERLAP_LINES,
) -> list[Chunk]:
    """Split text into windows of `window` lines, each sharing `overlap` lines.

    Overlap keeps passages that straddle a window boundary retrievable from at
    least one chunk. Whitespace-only windows are dropped.
    """
    if overlap >= window:
        raise ValueError("Overlap must be less than window")
    lines = text.splitlines()
    step = window - overlap
    result: list[Chunk] = []
    for i in range(0, len(lines), step):
        window_lines = lines[i : i + window]
        window_text = "\n".join(window_lines)
        if not window_text.strip():
            continue
        chunk = Chunk(
            repo=repo,
            path=path,
            lang=lang,
            start_line=i + 1,
            end_line=min(i + window, len(lines)),
            text=window_text,
            kind=kind,
            symbol=None,
            node_type=None,
        )
        result.append(chunk)
        if i + window >= len(lines):
            break
    return result


def chunk_markdown(
    text: str,
    *,
    repo: str,
    path: str,
    lang: str,
    kind: Kind,
) -> list[Chunk]:
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
            Chunk(
                repo=repo,
                path=path,
                lang=lang,
                kind=Kind.DOC,
                start_line=start_line,
                end_line=end_line,
                symbol=symbol,
                node_type="section",
                text=section_text,
            )
        )
    return chunks


def chunk_text(text: str, *, repo: str, path: str, lang: str, kind: Kind) -> list[Chunk]:
    """Dispatch by lang: Markdown gets section chunking, the rest sliding windows."""
    if lang == "markdown":
        return chunk_markdown(text, repo=repo, path=path, lang=lang, kind=kind)
    return chunk_plain(text, repo=repo, path=path, lang=lang, kind=kind)
