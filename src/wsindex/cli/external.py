"""Fetching one document from outside the repositories."""

import typer

from wsindex.cli.composition import config_or_default, require_config_file
from wsindex.connectors import route


def fetch(url: str) -> None:
    """Fetch one external document through the configured connectors.

    The way to check a `[[connectors]]` entry does what you meant, and
    the seam `wsindex.snapshot` materializes through. A pointed pull: this brings
    back the document at the url and nothing around it.
    """
    config = config_or_default()
    require_config_file(config)
    connector = route(url, config.connectors)
    if connector is None:
        typer.echo(
            f"no connector claims {url} — add a [[connectors]] entry whose url_pattern matches it",
            err=True,
        )
        raise typer.Exit(code=1)
    # No `except ConnectorError` here: it is a RuntimeError, and
    # `wsindex.cli.run` turns those into the same message and the same
    # exit code for every command. A second handler saying the same thing
    # is one more place for the two to disagree.
    document = connector.fetch(url)
    typer.echo(f"{document.url}")
    if document.title:
        typer.echo(f"title: {document.title}")
    for key, value in sorted(document.metadata.items()):
        typer.echo(f"{key}: {value}")
    typer.echo("")
    typer.echo(document.text)
