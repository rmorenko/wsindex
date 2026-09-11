"""Commands that describe the workspace: create it, add to it, look at it."""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from wsindex.cli.composition import build_pipeline, config_or_default, require_config_file
from wsindex.config import Backend, Config, LinksBackend, Provider, Repository, RepoSource
from wsindex.ingest import SKIP_REASONS
from wsindex.ingest.git_state import STATE_FILE, IndexState
from wsindex.paths import user_config_file, workspace_config_path
from wsindex.stats import SearchLog


def init(
    name: str,
    backend: Annotated[Backend, typer.Option(help="Vector store backend")] = Backend.LOCAL,
    provider: Annotated[
        Provider, typer.Option(help="Embeddings provider")
    ] = Provider.SENTENCE_TRANSFORMERS,
    user: Annotated[
        bool,
        typer.Option("--user", help="Install into $XDG_CONFIG_HOME/wsindex/ instead of the CWD"),
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing config")] = False,
) -> None:
    """Create a wsindex config.

    Two modes: workspace (default) writes `./wsindex.toml`, so the tool
    lives with the project (like `git init`); `--user` writes
    `$XDG_CONFIG_HOME/wsindex/config.toml`, for `pipx install` and other
    from-anywhere invocations. Both refuse to overwrite unless `--force`
    is given.

    The workspace is fully offline once the embedding model is
    downloaded, and its index lands next to the config. Point
    `[store] uri` at s3://... for shared storage instead (credentials
    come from the AWS_* env).
    """
    target = user_config_file() if user else workspace_config_path()
    if target.exists() and not force:
        typer.echo(
            f"error: config already exists at {target} (use --force to overwrite)",
            err=True,
        )
        raise typer.Exit(code=1)
    target.parent.mkdir(parents=True, exist_ok=True)
    config = Config.default(name, backend=backend, provider=provider)
    config.save(target)
    typer.echo(f"created {target}: workspace '{name}', backend '{backend.value}'")


def add_repo(
    repo_id: str,
    path: str,
    remote: Annotated[
        str | None,
        typer.Option("--remote", help="Clone url; `wsindex sync` keeps <path> up to date from it"),
    ] = None,
    source: Annotated[
        RepoSource | None,
        typer.Option("--source", help="`connector` for a snapshot repo built from --url documents"),
    ] = None,
    url: Annotated[
        list[str] | None,
        typer.Option("--url", help="Document to materialize (repeat); needs --source connector"),
    ] = None,
    ignore: Annotated[
        list[str] | None,
        typer.Option("--ignore", help="Path glob this repo excludes (repeat)"),
    ] = None,
) -> None:
    """Register a repository; its id becomes the dataset name.

    With `--remote` the working copy becomes the workspace's business:
    `wsindex sync` clones it if `path` does not exist yet and
    fast-forwards it afterwards. Without one, `path` is a checkout you
    maintain yourself and sync leaves it alone.

    With `--source connector` the repo is a snapshot instead: sync
    fetches each `--url` through the configured connectors, writes it as
    markdown and commits. The two are mutually exclusive — a working copy
    has one owner.

    `--ignore` takes a path glob this repo excludes, on top of what the
    walker prunes everywhere. The other per-repo key, `formats`, is a
    suffix table and lives in `wsindex.toml` rather than on a command
    line.
    """
    config = config_or_default()
    location = require_config_file(config)
    try:
        config.add_repo(
            Repository(
                id=repo_id,
                path=path,
                remote=remote,
                source=source,
                urls=tuple(url or ()),
                ignore=tuple(ignore or ()),
            )
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    config.save(location.path)
    if source is RepoSource.CONNECTOR:
        suffix = f" (snapshot of {len(url or ())} document(s))"
    else:
        suffix = f" (remote: {remote})" if remote else ""
    typer.echo(f"added repo '{repo_id}' -> {path}{suffix}")
    if remote is None and source is None and not Path(path).expanduser().is_dir():
        # Said now, while the typo is still on screen. It used to surface
        # at the next `index` as `missing repos:`, possibly days later
        # and from a different command. A warning rather than a refusal:
        # with a remote the path is *supposed* not to exist yet, which is
        # why the two flags above exclude themselves from the check.
        typer.echo(
            f"warning: {path} does not exist — index will skip it; "
            "pass --remote to have sync clone it there",
            err=True,
        )


def status() -> None:
    """Show the workspace: name, backend, registered repos, index state."""
    config = config_or_default()
    location = config.location
    if location is None:
        typer.echo("config: none — showing built-in defaults")
    else:
        typer.echo(f"config: {location.path} ({location.mode.value})")
    typer.echo(f"workspace: {config.name}")
    typer.echo(f"backend: {config.backend.value}")
    typer.echo(f"links: {config.links_backend.value}")
    # `location is not None` first: `store_uri` needs an index dir, and a
    # config built from defaults has none. Status is the command people
    # run *because* something is wrong, so it must survive that.
    if (
        location is not None
        and config.links_backend is LinksBackend.SQLITE
        and config.store_uri.startswith("s3://")
    ):
        # The vectors are shared and the links are not. Said out loud
        # rather than left to be discovered, because `refs` and `why`
        # then answer from this machine's links only — a partial answer
        # that looks whole, which is what review 6 was about.
        typer.echo(
            "  note: the index is shared but links are local to this machine; "
            'set [links] backend = "postgres" to share them too',
            err=True,
        )
    if not config.repos:
        typer.echo("repos: none — add one with `wsindex add-repo <id> <path>`")
        return
    typer.echo("repos:")
    for repo in config.repos:
        typer.echo(f"  {repo.id} -> {repo.path}{_indexed_at(config, repo.id)}")


def _indexed_at(config: Config, repo_id: str) -> str:
    """What the index holds for one repo, as a suffix for the status line.

    `status` used to recite the config back — every line of it already
    visible in `wsindex.toml`. Nothing said whether the thing the tool
    exists for had ever run. Both facts are already on disk: the commit
    in `state.json`, the file's mtime for when.

    Never raises. Status is the command people run *because* something
    is wrong, so it must survive an index that is missing or damaged and
    say which.
    """
    state_file = config.index_dir / STATE_FILE
    try:
        state = IndexState.load(config.index_dir)
        commit = state.commits.get(repo_id)
        if commit is None:
            return "  (not indexed)"
        when = datetime.fromtimestamp(state_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        return f"  (indexed {when} at {commit[:7]})"
    except OSError:
        return "  (index state unreadable)"


def explain(
    path: Annotated[str, typer.Argument(help="File to ask about, as you would type it")],
) -> None:
    """Say whether a file is indexed, and if not, which rule left it out.

    The question this tool gets asked most: "why does search not find my
    file?" Six rules can exclude one, and until this existed every one of
    them looked identical from outside — the file was simply absent. Each
    answer points at a different fix, so guessing between them is the
    difference between editing `formats`, moving the file, and looking at
    permissions.

    The path may be absolute or relative to the current directory; the
    repo it belongs to is worked out from the config.
    """
    require_config_file(config_or_default())
    report = build_pipeline().describe(Path(path))
    if report is None:
        typer.echo(
            f"error: {Path(path).expanduser().resolve()} is not inside any configured repo — "
            "`wsindex status` lists them",
            err=True,
        )
        raise typer.Exit(code=1)
    where = f"{report.repo}/{report.rel_path}"
    if report.skipped is not None:
        typer.echo(f"{where}: not indexed — {SKIP_REASONS[report.skipped]}")
        return
    typer.echo(f"{where}: indexed as {report.lang} ({report.kind.value if report.kind else ''})")
    if report.parsed_cleanly:
        typer.echo(f"  {report.chunks} chunk(s), {report.symbols} with a symbol")
    else:
        typer.echo(
            f"  {report.chunks} chunk(s), but the {report.lang} grammar reported errors — "
            "the parts it could not read are indexed as text, not definitions"
        )


def stats(
    forget: Annotated[
        bool, typer.Option("--forget", help="Delete everything recorded, then say how much")
    ] = False,
    top: Annotated[int, typer.Option("--top", help="How many queries to name")] = 5,
) -> None:
    """What this machine has asked, and how often it got nothing.

    The quality loop: real questions beat invented ones for deciding
    whether re-ranking earns its keep or a hybrid index would. The
    acceptance criteria are ten queries somebody made up; this is
    however many the tool was actually asked.

    **Strictly local.** Nothing here leaves the machine, nothing is
    aggregated anywhere, and a test asserts a search opens no sockets.
    The file lives in the same 0700 directory as the index. Turn it off
    with `[stats] enabled = false`, empty it with `--forget`.
    """
    config = config_or_default()
    require_config_file(config)
    with SearchLog(config.index_dir) as log:
        if forget:
            typer.echo(f"forgot {log.forget()} recorded search(es)")
            return
        summary = log.summary(top=top)
    if not config.stats_enabled:
        typer.echo("note: recording is off ([stats] enabled = false)", err=True)
    if not summary.searches:
        typer.echo("nothing recorded yet — run a search or two")
        return
    when = datetime.fromtimestamp(summary.since).strftime("%Y-%m-%d") if summary.since else "?"
    typer.echo(f"{summary.searches} search(es) since {when}, picked {summary.picks}")
    # Labelled, because the number is honest and reads wrong without it:
    # a CLI search is a fresh process, so this is mostly the model load.
    # `poe bench` measures the search alone.
    typer.echo(
        f"waited: p50 {summary.p50_ms / 1000:.1f}s  p95 {summary.p95_ms / 1000:.1f}s "
        "(per command, model load included)"
    )
    if summary.empty:
        # Rare: semantic search answers something unless a filter
        # excluded everything. Worth naming when it does happen.
        typer.echo(f"empty: {summary.empty} ({summary.empty_rate:.0%})")
    if summary.weakest:
        # The questions the corpus could not really answer — invisible
        # from anywhere else, and the ones worth reading. Ranked rather
        # than thresholded: the gap between an answered query and an
        # unanswerable one belongs to the model, not to this file.
        typer.echo("answered worst:")
        for query, score in summary.weakest:
            typer.echo(f"  {score:.3f}  {query!r}")
    if summary.common:
        typer.echo("asked most:")
        for query, count in summary.common:
            typer.echo(f"  {count:>4}x  {query!r}")


def domains(
    repo: Annotated[
        str | None, typer.Option("--repo", help="Which repo to read; the only one by default")
    ] = None,
    prefix: Annotated[
        str | None,
        typer.Option(
            "--prefix",
            help="Only paths under this; derived from the repo's own layout by default",
        ),
    ] = None,
) -> None:
    """What this repository is made of, and where it crosses its own lines.

    A different question from search, for a different reader: not "where
    is X" but "what are the parts, and what is tangled". Two signals, and
    the value is where they disagree.

    **Meaning**, from the vectors already in the index: a file's subject
    is the average of its chunks. **Change**, from the history already
    indexed: files that keep moving in the same commit are coupled
    whether or not anything imports anything.

    Read `agreement` first. It says how much of the layout the meaning
    recovers, against the baseline of saying nothing. Well above it and
    the exceptions below are worth your time; near it and they are not,
    because nothing was found.

    A *stranger* is a file whose nearest neighbours are all outside its
    own package — its subject lives somewhere other than its directory.
    Sometimes deliberate, sometimes a module filed by when it runs rather
    than by what it is about.

    A *coupled pair* crosses a package boundary and keeps changing
    together. High similarity means the coupling is honest. **Low
    similarity is the one to read**: something binds two files that are
    not about the same thing.
    """
    from wsindex.domains import analyse

    config = config_or_default()
    require_config_file(config)
    pipeline = build_pipeline()
    chosen = repo or (config.repos[0].id if len(config.repos) == 1 else None)
    if chosen is None:
        typer.echo("error: this workspace holds several repos — name one with --repo", err=True)
        raise typer.Exit(code=1)
    try:
        found = analyse(pipeline, repo=chosen, prefix=prefix)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if found.files < 6:
        # Three different problems used to share one sentence, and the
        # sentence named the one that was usually not it: across the
        # field trial 151 of 193 runs were told to "Index first" with a
        # complete index beside them. What was wrong was the prefix,
        # which defaulted to this project's own layout.
        where = f"under {found.prefix!r}" if found.prefix else "in this repo"
        if found.files:
            typer.echo(f"{found.files} source file(s) {where} — too few to have domains.")
        elif prefix is not None:
            # The prefix was typed, so the prefix is the suspect. Saying
            # what the repo would have chosen turns this into one step.
            derived = analyse(pipeline, repo=chosen).prefix
            hint = f"; this repo keeps its code under {derived!r}" if derived else ""
            typer.echo(f"no indexed code under {prefix!r}{hint} — drop --prefix to derive it")
        else:
            typer.echo(
                f"no indexed code in {chosen} — `wsindex index` if it is new, "
                "or `wsindex explain <path>` to see which rule left its files out"
            )
        return
    where = f" under {found.prefix!r}" if found.prefix else ""
    typer.echo(f"{found.files} files in {len(found.packages)} packages{where}")
    typer.echo("  " + "  ".join(f"{name} {count}" for name, count in found.packages.items()))
    typer.echo(
        f"agreement {found.agreement:.0%} (meaning recovers the layout; "
        f"{found.baseline:.0%} would be chance)"
    )
    if found.strangers:
        typer.echo("\nfiled away from their subject:")
        for stranger in found.strangers:
            typer.echo(f"  {stranger.path}  [{stranger.package}]  {stranger.similarity:.3f}")
            typer.echo(f"      near {', '.join(stranger.neighbours[:3])}")
    if found.coupled:
        typer.echo(f"\ncoupled across packages ({len(found.coupled)}), most-changed first:")
        for pair in found.coupled[:10]:
            typer.echo(
                f"  {pair.commits:3} commits  similarity {pair.similarity:+.3f}   "
                f"{pair.left} + {pair.right}"
            )


def dupes(
    repo: Annotated[
        str | None, typer.Option("--repo", help="Which repo to read; the only one by default")
    ] = None,
    minimum: Annotated[
        float,
        typer.Option("--min", help="Shared shingles a pair must reach", min=0.1, max=1.0),
    ] = 0.45,
    limit: Annotated[int, typer.Option("--limit", help="How many groups to print")] = 15,
) -> None:
    """The same code in two places, grouped by where it lives.

    Found by what the code is *made of* — runs of five identifiers,
    hashed and compared — not by what it means. Vectors were measured
    against this and lost; see `wsindex.dupes` for the numbers.

    **The grouping is the point.** On a real codebase most duplication is
    a vendored library checked in twice or a tree of generated classes:
    thousands of pairs that are one decision somebody made once. Those
    collapse to a line each, marked `wholesale`, and what is left
    underneath is the copy-paste a person can act on — a form's `save.php`
    copied into the next form.

    Lower `--min` to see more and worse: below 0.45 the report fills with
    generated code that is duplicated by construction.
    """
    from wsindex.dupes import find

    config = config_or_default()
    require_config_file(config)
    pipeline = build_pipeline()
    chosen = repo or (config.repos[0].id if len(config.repos) == 1 else None)
    if chosen is None:
        typer.echo("error: this workspace holds several repos — name one with --repo", err=True)
        raise typer.Exit(code=1)
    try:
        found = find(pipeline, repo=chosen, minimum=minimum)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    pairs = sum(len(group.pairs) for group in found.between)
    typer.echo(
        f"{found.compared} of {found.chunks} code chunks were long enough to fingerprint; "
        f"{pairs} duplicate pair(s) in {len(found.between)} place(s)"
    )
    if not found.between:
        typer.echo("nothing above the threshold — try a lower --min")
        return
    for group in found.between[:limit]:
        where = (
            f"within {group.left}" if group.within else f"{group.left}\n            {group.right}"
        )
        tag = "wholesale" if group.wholesale else "  copied "
        typer.echo(f"\n{tag} {len(group.pairs):5} pair(s)  {where}")
        for pair in group.pairs[:3]:
            typer.echo(
                f"    {pair.overlap:.2f}  {pair.left}:{pair.left_lines[0]}"
                f"  <->  {pair.right}:{pair.right_lines[0]}"
            )
