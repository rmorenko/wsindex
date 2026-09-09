"""A Notion connector for wsindex, as a worked example of the plugin seam.

Everything wsindex needs is in `pyproject.toml`:

    [project.entry-points."wsindex.connectors"]
    notion = "wsindex_connector_notion:NotionConnector"

`notion` is then a `type` any config can name. Nothing is imported by
wsindex until it is; nothing in wsindex has to know this package exists.

Why Notion is the example
-------------------------
Because its API returns neither markdown nor text. A page is a *tree of
blocks* — paragraph, heading_2, bulleted_list_item, code, to_do — each
carrying an array of rich-text runs with their own annotations and
links. Both built-in connectors get their text handed to them: GitHub
answers with markdown the author wrote, and generic-http gets HTML that
only has to stop being markup. Neither exercises the case where the
document has to be *rebuilt*, which is the case most real sources are.

So this is the honest test of the contract: a connector's job is to
return a `Document`, and how far it is from what the source said is the
connector's problem alone.

What is verified, and what is not
---------------------------------
`probes/step29v` in the wsindex repo asked api.notion.com what it says to
a caller with no token, and two answers are baked in below:

- **Notion distinguishes "not authorized" from "not there".** It answers
  401 `unauthorized` with two different messages — "Authorization header
  must use the format" when the header is missing, "API token is
  invalid." when the token is wrong — and reserves 404 for a page that
  is missing or not shared with the integration. That is the opposite of
  GitHub, which answers 404 for a private repository and makes the two
  indistinguishable. So this connector maps 401 to a `ConnectorError`
  that names the token variable, and only 404 to `DocumentNotFound`.
- **The page id must come from the url's path, never the whole url.** A
  database view url ends in `?v=<32 hex>`, and the obvious "find 32 hex
  characters anywhere" would return the *view* id — a request for the
  wrong object, and one that looks like it worked.

Not verified: everything that needs a token. The block shapes below
follow Notion's published API, and the tests drive them from recorded
payloads of that shape, but no live workspace has confirmed them. That
boundary is stated rather than papered over.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from wsindex.connectors import Connector, ConnectorError, Document, DocumentNotFound

API = "https://api.notion.com/v1"

NOTION_VERSION = "2022-06-28"
"""API version this speaks. Notion requires the header on every request
and keeps old versions working, so pinning it is what stops a future
default from renaming the fields parsed below."""

TIMEOUT = 20.0
MAX_DEPTH = 4
"""How deep nested blocks are followed. Notion nests without limit —
toggles inside list items inside callouts — and each level is another
request per block with children. Four covers ordinary documents; deeper
than that, a page is a wiki of its own and the boundary "one pointed
pull, not a crawl" is worth more than the last paragraph."""

_HOSTS = ("notion.so", "www.notion.so")
_ID = re.compile(
    r"[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
"""A page id, dashed or not. Notion accepts either form in the API and
puts the undashed one in urls."""

_PREFIX = {
    "heading_1": "# ",
    "heading_2": "## ",
    "heading_3": "### ",
    "bulleted_list_item": "- ",
    "numbered_list_item": "1. ",
    "quote": "> ",
    "callout": "> ",
    "toggle": "- ",
}
"""Block type -> what markdown puts in front of its text. Numbered items
are all `1.` on purpose: markdown renumbers them, and the alternative is
tracking a counter per nesting level for no gain in either rendering or
search."""


def page_id(url: str) -> str | None:
    """The page id in a Notion url, or None if there is none.

    Reads the *path* only. A database view url carries a second 32-hex id
    in `?v=`, and taking the last id in the whole url returns that one —
    a request for the wrong object that looks like it worked.

    Args:
        url: A notion.so or *.notion.site url.

    Returns:
        The id without dashes, lowercased, or None.
    """
    path = urllib.parse.urlsplit(url).path
    found = _ID.findall(path)
    # The last one: a url is `/<workspace>/<Title-slug><id>`, and a
    # workspace name can be hex too.
    return found[-1].replace("-", "").lower() if found else None


class NotionConnector(Connector):
    """Fetches one Notion page as markdown."""

    def matches(self, url: str) -> bool:
        """True for a Notion url that names a page.

        Narrower than the config's pattern, like every connector's
        `matches`: a workspace's search url or a database view lives on
        the same host and is not a document this can fetch.
        """
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
        if host not in _HOSTS and not host.endswith(".notion.site"):
            return False
        return page_id(url) is not None

    def fetch(self, url: str) -> Document:
        """Fetch one page: its title, its blocks, and what Notion knows.

        Args:
            url: The page url.

        Returns:
            The document; `text` is markdown rebuilt from the block tree.

        Raises:
            DocumentNotFound: No such page — or one the integration has
                not been given access to. In Notion those are genuinely
                the same thing: sharing is per page.
            ConnectorError: The url names no page, the token is missing
                or rejected, or the API answered something unusable.
        """
        identifier = page_id(url)
        if identifier is None:
            raise ConnectorError(f"not a Notion page url: {url}")
        page = self._get(f"{API}/pages/{identifier}", url)
        blocks = self._children(identifier, url, depth=0)
        metadata = {"source": "notion", "id": identifier}
        for key, field in (("created", "created_time"), ("updated", "last_edited_time")):
            value = page.get(field)
            if value:
                metadata[key] = str(value)
        return Document(
            # Notion's own url, which carries the workspace and the
            # current title slug — the one that keeps working when the
            # page is renamed is the id inside it, and both are here.
            url=str(page.get("url") or url),
            title=_title(page),
            text="\n\n".join(blocks),
            metadata=metadata,
        )

    def _children(self, block_id: str, url: str, *, depth: int) -> list[str]:
        """Every child block of `block_id`, as markdown paragraphs.

        Paginated: Notion returns at most 100 children per request and a
        `next_cursor` for the rest, so a long page is several requests
        and dropping the loop would silently truncate it.
        """
        out: list[str] = []
        cursor: str | None = None
        while True:
            page = f"&start_cursor={urllib.parse.quote(cursor, safe='')}" if cursor else ""
            query = f"?page_size=100{page}"
            payload = self._get(f"{API}/blocks/{block_id}/children{query}", url)
            for block in payload.get("results", []):
                out.extend(self._render(block, url, depth=depth))
            if not payload.get("has_more"):
                return out
            cursor = payload.get("next_cursor")
            if not cursor:
                # `has_more` without a cursor cannot be paged further;
                # returning what is in hand beats looping forever.
                return out

    def _render(self, block: dict[str, Any], url: str, *, depth: int) -> list[str]:
        """One block, plus its children, as markdown lines."""
        kind = str(block.get("type", ""))
        body = block.get(kind) or {}
        if kind == "divider":
            return ["---"]
        text = _rich_text(body.get("rich_text", []))
        if kind == "code":
            language = str(body.get("language") or "")
            rendered = [f"```{language}\n{text}\n```"]
        elif kind == "to_do":
            box = "x" if body.get("checked") else " "
            rendered = [f"- [{box}] {text}"]
        elif kind == "child_page":
            # A link, not a fetch: a child page is another document, and
            # following it would make this a crawler.
            rendered = [f"- {body.get('title') or 'Untitled'} (child page)"]
        elif text:
            rendered = [f"{_PREFIX.get(kind, '')}{text}"]
        else:
            # An unsupported block — an embed, a synced block, a file.
            # Nothing to index, and inventing a placeholder would put
            # words in the document that nobody wrote.
            rendered = []
        if block.get("has_children") and depth < MAX_DEPTH:
            nested = self._children(str(block["id"]), url, depth=depth + 1)
            rendered += [_indent(line) for line in nested]
        return rendered

    def _get(self, endpoint: str, url: str) -> dict[str, Any]:
        """One authenticated GET, with Notion's failures mapped to ours."""
        token = self.spec.token()
        if token is None:
            raise ConnectorError(
                f"the Notion connector for {self.spec.url_pattern!r} needs a token — "
                'add token_env = "NOTION_TOKEN" to its [[connectors]] entry'
            )
        request = urllib.request.Request(
            endpoint,
            headers={
                "Authorization": f"Bearer {token}",
                "Notion-Version": NOTION_VERSION,
                "User-Agent": "wsindex-connector-notion",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raise _translate(exc, url, self.spec.token_env) from exc
        except urllib.error.URLError as exc:
            raise ConnectorError(f"{url} could not be reached — {exc.reason}") from exc
        try:
            parsed = json.loads(body)
        except ValueError as exc:
            raise ConnectorError(f"Notion answered something that is not JSON for {url}") from exc
        if not isinstance(parsed, dict):
            raise ConnectorError(f"Notion answered a {type(parsed).__name__} for {url}")
        return parsed


def _translate(exc: urllib.error.HTTPError, url: str, token_env: str | None) -> ConnectorError:
    """Notion's error into ours, keeping the distinction it makes.

    The probe's finding: 401 means the token, 404 means the page. Folding
    them together — as a GitHub connector has to, because GitHub does —
    would send someone hunting for a page that was there all along.
    """
    if exc.code == 404:
        return DocumentNotFound(f"{url} is not there, or not shared with the integration")
    if exc.code == 401:
        name = f"${token_env}" if token_env else "the configured token"
        return ConnectorError(f"Notion rejected {name} — it is invalid or lacks access to {url}")
    if exc.code == 429:
        # Notion rate-limits at roughly three requests a second, and a
        # nested page is many requests. Saying so beats "answered 429".
        return ConnectorError(f"Notion is rate-limiting this workspace; retry later ({url})")
    return ConnectorError(f"{url} answered {exc.code}")


def _title(page: dict[str, Any]) -> str:
    """The page's title, wherever the schema put it.

    A page's properties are user-defined and their *names* differ per
    database — "Name", "Title", "Задача". Exactly one has `type ==
    "title"`, so that is what identifies it, not its name.
    """
    for prop in (page.get("properties") or {}).values():
        if isinstance(prop, dict) and prop.get("type") == "title":
            return _rich_text(prop.get("title", []))
    return ""


def _rich_text(runs: list[dict[str, Any]]) -> str:
    """A rich-text array as markdown.

    Notion splits a sentence into runs wherever formatting changes, so
    "**bold** word" arrives as two objects. Annotations are applied
    innermost-first — code before bold before italic — because
    `**`code`**` renders and ``**code**`` does not.
    """
    out: list[str] = []
    for run in runs:
        text = str(run.get("plain_text", ""))
        if not text:
            continue
        marks = run.get("annotations") or {}
        if marks.get("code"):
            text = f"`{text}`"
        if marks.get("bold"):
            text = f"**{text}**"
        if marks.get("italic"):
            text = f"*{text}*"
        if marks.get("strikethrough"):
            text = f"~~{text}~~"
        href = run.get("href")
        if href:
            text = f"[{text}]({href})"
        out.append(text)
    return "".join(out)


def _indent(line: str) -> str:
    """Indent one rendered line by one nesting level."""
    return "\n".join(f"  {part}" for part in line.splitlines())


__all__ = ["NOTION_VERSION", "NotionConnector", "page_id"]
