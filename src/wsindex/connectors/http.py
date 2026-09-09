"""The fallback connector: fetch a url, hand back readable text.

Anything already written in text needs no source-specific knowledge — a
raw markdown file arrives usable, and `probes/step29a` confirmed it:
`README.md` from a raw.githubusercontent url is markdown with nothing to
convert. HTML is the case that needs work, and it needs only enough to
stop being markup.

How much is enough
------------------
Measured on a real documentation page: 83 KB of HTML becomes 6.9 KB of
text, 8% of the original, with no script or style content leaking
through. That is the whole justification for doing it with the standard
library instead of taking a dependency — the remaining 92% is markup and
JavaScript, and no parser is needed to be sure of that.

What it does not do is produce *markdown*: headings and links flatten,
and navigation chrome ("Skip to content", the nav menu) comes through as
text. Turning a source's own format into markdown belongs to
materialization (step 29b), where there is a file to write and a
converter per source to pick.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from html.parser import HTMLParser

from wsindex.connectors import Connector, ConnectorError, Document, DocumentNotFound

TIMEOUT = 20.0
"""Seconds one request may take. A document fetch is interactive — the
user is waiting — so this is short enough to fail rather than hang."""

MAX_BYTES = 8 * 1024 * 1024
"""Refuse a response larger than this. A connector fetches documents; a
response this size is a download, and reading it into memory to index it
is not what anyone asked for."""

_INVISIBLE = frozenset({"script", "style", "noscript", "svg", "head", "template"})
"""Elements whose *content* is not text a reader sees. Dropping them is
what takes a documentation page from 83 KB to 7 KB."""

_BREAKS = frozenset({"p", "div", "li", "br", "tr", "h1", "h2", "h3", "h4", "h5", "h6"})
"""Elements that end a line. Without them the page arrives as one
paragraph and every line number in it means nothing."""


class _TextExtractor(HTMLParser):
    """Collect the text an HTML page shows, and its title."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._hidden = 0
        self._in_title = False
        self.title: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _INVISIBLE:
            # A counter, not a flag: `<svg>` inside `<head>` would
            # otherwise un-hide the rest of the head on its close tag.
            self._hidden += 1
        if tag == "title":
            self._in_title = True
        if tag in _BREAKS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _INVISIBLE and self._hidden:
            self._hidden -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title and self.title is None:
            self.title = data.strip()
        if not self._hidden:
            self._parts.append(data)

    def text(self) -> str:
        """The collected text, with runs of whitespace and blank lines cut."""
        joined = re.sub(r"[ \t]+", " ", "".join(self._parts))
        lines = [line.strip() for line in joined.splitlines()]
        return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> tuple[str, str]:
    """Readable text and title from an HTML document.

    Args:
        html: The page source.

    Returns:
        `(text, title)`; title is empty when the page has none.
    """
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text(), parser.title or ""


class GenericHttpConnector(Connector):
    """Fetches any url and returns whatever text it can make of it."""

    def matches(self, url: str) -> bool:
        """True for http and https. The point of a fallback is breadth."""
        return url.startswith(("http://", "https://"))

    def fetch(self, url: str) -> Document:
        """Fetch a url and decode it into text.

        The content type decides: markdown and plain text arrive usable,
        HTML is stripped to what a reader would see, and anything else is
        refused rather than indexed as noise — a PDF read as UTF-8 is
        worse than an error, because it succeeds.

        Args:
            url: The document's url.

        Returns:
            The document; `metadata` carries the content type actually
            served, which is the only thing this source knows about it.

        Raises:
            DocumentNotFound: The server answered 404 or 410.
            ConnectorError: Anything else — unreachable, refused, too
                large, or a content type there is no text in.
        """
        headers = {"User-Agent": "wsindex", "Accept": "text/*, */*;q=0.5"}
        token = self.spec.token()
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        body, content_type = _get(url, headers)
        base_type = content_type.split(";")[0].strip().lower()
        charset = _charset(content_type)
        if base_type in ("text/html", "application/xhtml+xml"):
            text, title = html_to_text(body.decode(charset, errors="replace"))
        elif base_type.startswith("text/") or base_type in (
            "application/json",
            "application/xml",
        ):
            text, title = body.decode(charset, errors="replace"), ""
        else:
            raise ConnectorError(f"nothing to read at {url}: content type {base_type!r}")
        return Document(
            url=url,
            title=title,
            text=text,
            metadata={"content_type": base_type},
        )


def _charset(content_type: str) -> str:
    for part in content_type.split(";")[1:]:
        key, _, value = part.partition("=")
        if key.strip().lower() == "charset":
            return value.strip().strip('"') or "utf-8"
    return "utf-8"


def _get(url: str, headers: dict[str, str]) -> tuple[bytes, str]:
    """One GET, with the failures a caller has to tell apart."""
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > MAX_BYTES:
                raise ConnectorError(f"{url} is {int(declared):,} bytes, larger than the limit")
            # Read one byte past the limit: a server that sends no
            # Content-Length would otherwise slip through the check above.
            body = response.read(MAX_BYTES + 1)
            if len(body) > MAX_BYTES:
                raise ConnectorError(f"{url} is larger than the {MAX_BYTES:,} byte limit")
            return body, response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 410):
            raise DocumentNotFound(f"{url} is not there, or not visible") from exc
        raise ConnectorError(f"{url} answered {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ConnectorError(f"{url} could not be reached — {exc.reason}") from exc
