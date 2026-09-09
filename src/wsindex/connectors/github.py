"""GitHub issues and pull requests, through the REST API.

First after the generic fallback because the workspaces this was built
for live on GitHub, so it can be checked against something real rather
than described. The API was asked before this was written, and three
answers are baked in below.

**An issue and a pull request are one endpoint.** `/issues/{n}` answers
for both — a PR is an issue with extra fields — so there is no url shape
to disambiguate and no second code path. `github.com/o/r/pull/7` and
`github.com/o/r/issues/7` route to the same request.

**The body is already markdown.** Nothing is converted; the text stored
is the text the author wrote.

**A private repository answers 404, not 403.** Indistinguishable from a
document that does not exist, by policy rather than by accident — which
is why `DocumentNotFound` says "not there, or not visible" and why the
connector refuses to run anonymously when a token was configured. An
unauthenticated caller also gets 60 requests an hour, which is another
reason to notice a missing token early rather than at the 61st fetch.
"""

from __future__ import annotations

import json
import re

from wsindex.connectors import Connector, ConnectorError, Document, DocumentNotFound
from wsindex.connectors.http import _get

API = "https://api.github.com"

_ISSUE_URL = re.compile(
    r"^https?://(?:www\.)?github\.com/([\w.\-]+)/([\w.\-]+)/(?:issues|pull)/(\d+)/?(?:[#?].*)?$"
)
"""A human issue or PR url. `pull` and `issues` both accepted, and both
answered by the same API endpoint."""


class GitHubConnector(Connector):
    """Fetches one issue or pull request as a document."""

    def matches(self, url: str) -> bool:
        """True only for an issue or PR url.

        Narrower than the config's pattern on purpose: someone routing
        `https://github.com/myorg/*` here means "our GitHub", not "every
        page GitHub serves". A repository front page is not something
        this connector can turn into a document, and saying so lets the
        router fall through to a generic entry.
        """
        return _ISSUE_URL.match(url) is not None

    def fetch(self, url: str) -> Document:
        """Fetch one issue or pull request.

        Args:
            url: A `github.com/<owner>/<repo>/issues|pull/<number>` url.

        Returns:
            The document; `text` is the body as written, `metadata`
            carries author, state and dates.

        Raises:
            DocumentNotFound: No such issue — or none this token sees.
            ConnectorError: The url is not one this connector handles,
                the API could not be reached, or it answered something
                unusable.
        """
        match = _ISSUE_URL.match(url)
        if match is None:
            raise ConnectorError(f"not a GitHub issue or pull request url: {url}")
        owner, repo, number = match.groups()

        headers = {
            "User-Agent": "wsindex",
            # Pinning the API version keeps a future default from
            # changing the field names this parses.
            "X-GitHub-Api-Version": "2022-11-28",
            "Accept": "application/vnd.github+json",
        }
        token = self.spec.token()
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"

        try:
            body, _ = _get(f"{API}/repos/{owner}/{repo}/issues/{number}", headers)
        except DocumentNotFound as exc:
            # Re-raised against the url the user asked about: they typed
            # a github.com link and should not have to recognise an
            # api.github.com one to understand the answer.
            raise DocumentNotFound(f"{url} is not there, or not visible") from exc
        try:
            data = json.loads(body)
        except ValueError as exc:
            raise ConnectorError(f"GitHub answered something that is not JSON for {url}") from exc

        metadata = {
            "source": "github",
            "repository": f"{owner}/{repo}",
            "number": number,
            # A PR carries a `pull_request` key; an issue does not. That
            # is the only thing distinguishing them at this endpoint.
            "type": "pull_request" if "pull_request" in data else "issue",
        }
        for key, field in (
            ("state", "state"),
            ("created", "created_at"),
            ("updated", "updated_at"),
        ):
            value = data.get(field)
            if value:
                metadata[key] = str(value)
        author = (data.get("user") or {}).get("login")
        if author:
            metadata["author"] = str(author)

        return Document(
            # The API's own `html_url`, not the url asked for: a `pull/7`
            # request and an `issues/7` request name the same document,
            # and a link should say which one GitHub considers canonical.
            url=str(data.get("html_url") or url),
            title=str(data.get("title") or ""),
            text=str(data.get("body") or ""),
            metadata=metadata,
        )
