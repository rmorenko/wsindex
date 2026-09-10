"""The admin page: repos, their state, and the two buttons worth having.

Minimal on purpose: the repo list with indexing status, adding a repo,
running a sync by hand, and the log of recent runs. Nothing else — every
addition here is a thing the API already does that would then exist
twice.

Server-rendered HTML in one string, no template engine and no JavaScript
build. A template engine buys reuse across many pages and there is one;
a front end buys interactivity that a page with two forms does not need.
Both would be a dependency and a directory to keep in step with a page
that fits on a screen.

The forms post to the API's own endpoints and redirect back, so the page
has no privileges the API does not — the same rule the whole server
keeps about the library.
"""

from __future__ import annotations

import contextlib
import html
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

from wsindex.config import Repository
from wsindex.server.scheduler import sync_and_index

_STYLE = """
body { font: 14px/1.5 -apple-system, system-ui, sans-serif; margin: 2rem auto; max-width: 60rem;
       color: #222; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.05rem; margin-top: 2rem; }
h3 { font-size: .9rem; margin: 1.2rem 0 .2rem; color: #555; font-weight: 600; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0; }
th, td { text-align: left; padding: .35rem .6rem; border-bottom: 1px solid #e5e5e5;
         vertical-align: top; }
th { font-weight: 600; color: #555; font-size: .85rem; text-transform: uppercase;
     letter-spacing: .03em; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9em; }
form { display: inline-block; margin: .25rem .5rem .25rem 0; }
input { font: inherit; padding: .3rem .4rem; border: 1px solid #ccc; border-radius: 3px; }
button { font: inherit; padding: .35rem .9rem; border: 1px solid #888; border-radius: 3px;
         background: #f6f6f6; cursor: pointer; }
button:hover { background: #eee; }
.busy { color: #a60; } .idle { color: #666; } .err { color: #a00; }
.note { color: #666; font-size: .9em; }
"""


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """A table, or a line saying there is nothing to put in one."""
    if not rows:
        return '<p class="note">nothing yet</p>'
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _run_row(entry: dict[str, Any]) -> list[str]:
    """One run of the log as table cells."""
    if "error" in entry:
        detail = f'<span class="err">{html.escape(str(entry["error"]))}</span>'
    else:
        parts = [
            f"files {entry.get('files', 0)}",
            f"written {entry.get('written', 0)}",
            f"deleted {entry.get('deleted', 0)}",
        ]
        synced = entry.get("synced") or {}
        if synced:
            parts.append(
                "; ".join(f"{html.escape(k)}: {html.escape(str(v))}" for k, v in synced.items())
            )
        detail = ", ".join(parts)
    return [
        html.escape(str(entry.get("at", ""))),
        html.escape(str(entry.get("kind", ""))),
        f"{entry.get('seconds', '')}",
        detail,
    ]


def _stats_panel(app: FastAPI) -> str:
    """What was asked of this server, in aggregate and only in aggregate.

    Step 39's wording asked for "analytics per user". There is no user to
    split by — the token belongs to the server, not to a person, and
    `stats.py` deliberately records no identity — and inventing one would
    mean per-user authentication *and* a log of other people's questions
    attributed to them, which is a privacy decision rather than a feature
    of a page. So: the same numbers `wsindex stats` prints, aggregated,
    with the page saying whose they are.

    Never raises. The stats database is a side note; the repo list and
    the sync button are the page, and a locked or corrupt log must cost a
    panel rather than the whole screen.
    """
    log = getattr(app.state.pipeline, "stats", None)
    if log is None:
        return (
            '<p class="note">Recording is off — <code>[stats] enabled = false</code>. '
            "Turn it on and this fills up as people search.</p>"
        )
    try:
        summary = log.summary(top=5)
    except Exception as exc:  # pragma: no cover - a locked or unreadable database
        return f'<p class="err">the search log could not be read: {html.escape(str(exc))}</p>'
    if not summary.searches:
        return '<p class="note">nothing recorded yet — run a search or two</p>'
    since = datetime.fromtimestamp(summary.since).strftime("%Y-%m-%d") if summary.since else "?"
    empty = f" · empty {summary.empty} ({summary.empty_rate:.0%})" if summary.empty else ""
    # `html.escape` on every query, and it is not decoration: these
    # strings came from whoever typed them, and this is the one place in
    # the project that puts them in a page.
    worst = _table(
        ["best score", "query"],
        [[f"{score:.3f}", _mono(query)] for query, score in summary.weakest],
    )
    most = _table(
        ["times", "query"],
        [[str(count), _mono(query)] for query, count in summary.common],
    )
    # Not the CLI's label, deliberately. `wsindex stats` says "per
    # command, model load included" and is right to: the model loads
    # lazily inside the first search of a fresh process, so a CLI search
    # reads 2.3 s. This server loaded it once at startup, so the same
    # field means the search itself — 8 ms, not seconds. One field, two
    # honest readings, and the interface that shows it has to say which.
    latency = (
        f"p50 {summary.p50_ms:.0f} ms · p95 {summary.p95_ms:.0f} ms (per search; the model "
        "is loaded once, at startup, so the first request after a restart is not in this)"
    )
    return f"""<p class="note">
 {summary.searches} search(es) since {html.escape(since)} · picked {summary.picks}{empty}<br>
 {latency}</p>
<p class="note">Everyone's questions on this server, together — nobody's
 separately, and no record of who asked. Switch it off with
 <code>[stats] enabled = false</code>; empty it with
 <code>wsindex stats --forget</code>.</p>
<h3>Answered worst</h3>
{worst}
<h3>Asked most</h3>
{most}"""


def _mono(text: str) -> str:
    """Someone's query, escaped, in the monospace span the page uses."""
    return f'<span class="mono">{html.escape(text)}</span>'


def render(app: FastAPI) -> str:
    """The whole page for the current state of the workspace."""
    config = app.state.config
    busy = app.state.writer.busy
    repos = [
        [
            f"<code>{html.escape(repo.id)}</code>",
            f'<span class="mono">{html.escape(repo.path)}</span>',
            html.escape(repo.remote or ("connector" if repo.is_snapshot else "local checkout")),
            str(len(repo.urls)) if repo.is_snapshot else "",
        ]
        for repo in config.repos
    ]
    runs = [_run_row(entry) for entry in app.state.runs.entries]
    state = '<span class="busy">indexing…</span>' if busy else '<span class="idle">idle</span>'
    interval = config.server_interval
    schedule = f"every {interval:.0f}s" if interval else "off (run it by hand)"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>wsindex — {html.escape(config.name)}</title><style>{_STYLE}</style></head>
<body>
<h1>wsindex <span class="note">— {html.escape(config.name)}</span></h1>
<p class="note">store <code>{html.escape(config.store_uri)}</code> ·
 backend {html.escape(config.backend.value)} ·
 re-rank {"on" if config.rank_enabled else "off"} ·
 schedule {html.escape(schedule)} · {state}</p>

<h2>Repositories</h2>
{_table(["id", "path", "source", "documents"], repos)}

<form method="post" action="admin/sync"><button type="submit">Sync and re-index</button></form>
<form method="post" action="admin/add-repo">
  <input name="repo_id" placeholder="id" required>
  <input name="path" placeholder="path" required size="30">
  <input name="remote" placeholder="remote (optional)" size="30">
  <button type="submit">Add repo</button>
</form>

<h2>Recent runs</h2>
{_table(["at", "kind", "seconds", "detail"], runs)}

<h2>What gets asked</h2>
{_stats_panel(app)}
</body></html>"""


def mount_admin(app: FastAPI, guarded: list[Any]) -> None:
    """Attach the page and its two forms.

    Args:
        app: Application to mount on.
        guarded: The auth dependency list every other endpoint uses,
            passed in rather than rebuilt — two definitions of "who may
            call this" is how one of them ends up wrong.
    """
    from wsindex.server.api import Busy

    @app.get("/admin", response_class=HTMLResponse, dependencies=guarded)
    def admin() -> Any:
        """The page."""
        return HTMLResponse(render(app))

    @app.post("/admin/sync", dependencies=guarded)
    def admin_sync() -> Any:
        """Run a sync now, then show the page again.

        A busy server redirects too, rather than showing an error: the
        run in progress is the answer, and it is on the page.
        """
        # Suppressed rather than reported: a run already in progress is
        # the answer, and the page shows it.
        with contextlib.suppress(Busy):
            sync_and_index(app)
        # 303: the browser must follow with GET, or a refresh would post
        # the form again — the oldest bug in server-rendered forms.
        return RedirectResponse("/admin", status_code=303)

    @app.post("/admin/add-repo", dependencies=guarded)
    def admin_add_repo(
        repo_id: str = Form(...),
        path: str = Form(...),
        remote: str = Form(""),
    ) -> Any:
        """Register a repo in the workspace config, then redirect back.

        Writes through `Config`, exactly as `wsindex add-repo` does, so
        the file the CLI reads next is the file this wrote — including
        the array-of-tables shape that keeps it hand-editable.
        """
        config = app.state.config
        location = config.location
        if location is None:
            raise HTTPException(status_code=400, detail="this server has no config file to write")
        try:
            config.add_repo(Repository(id=repo_id, path=path, remote=remote or None))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        config.save(location.path)
        return RedirectResponse("/admin", status_code=303)


__all__ = ["mount_admin", "render"]
