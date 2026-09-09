"""Turning the in-memory workspace into the text on disk.

One rule governs this module: **whatever a person is told to hand-edit
has to be a shape they can hand-edit.** `tomli_w` decides per entry by
line length, so a repo of just an id and a path comes out as
`repos = [{...}]` — and TOML forbids attaching `[[repos]]` or
`[repos.formats]` to a static array. The documented way to mark up a
repository was then a syntax error on the very file `wsindex add-repo`
had just written.

So the array of tables is written here rather than left to a heuristic,
while every value still goes through `tomli_w`: quoting a key like
`".sql"` is not something to reimplement.
"""

from __future__ import annotations

from typing import Any

import tomli_w

from wsindex.config.schema import Repository


def repo_entry(repo: Repository) -> dict[str, Any]:
    """A repo as the TOML document holds it.

    Empty values are left out rather than written as null: TOML has no
    null, and `tomli_w` refuses to write one.
    """
    entry: dict[str, Any] = {"id": repo.id, "path": repo.path}
    if repo.remote is not None:
        entry["remote"] = repo.remote
    if repo.source is not None:
        entry["source"] = repo.source.value
    if repo.urls:
        entry["urls"] = list(repo.urls)
    if repo.ignore:
        entry["ignore"] = list(repo.ignore)
    if repo.formats:
        entry["formats"] = {
            suffix: {"lang": lang, "kind": kind.value}
            for suffix, (lang, kind) in repo.formats.items()
        }
    return entry


def render(data: dict[str, Any]) -> str:
    """The document as TOML text, with repos as `[[repos]]` sections.

    Args:
        data: The parsed document, as `Config` holds it.

    Returns:
        The file contents.
    """
    parts = [tomli_w.dumps({key: value for key, value in data.items() if key != "repos"})]
    for repo in data.get("repos", []):
        flat = {key: value for key, value in repo.items() if not isinstance(value, dict)}
        nested = {key: value for key, value in repo.items() if isinstance(value, dict)}
        parts.append("[[repos]]\n" + tomli_w.dumps(flat))
        if nested:
            # Rendered under the `repos` name so the headers come out as
            # `[repos.formats...]`, which TOML attaches to the last
            # `[[repos]]` — the entry just written above.
            parts.append(tomli_w.dumps({"repos": nested}))
    return "\n".join(parts)
