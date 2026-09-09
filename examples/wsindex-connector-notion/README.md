# wsindex-connector-notion

Notion support for [wsindex](../../README.md), and the worked example
every connector plugin can be copied from.

Install it, point a `[[connectors]]` entry at your workspace, and Notion
pages become documents wsindex can snapshot and search. No change to
wsindex itself:

```bash
uv pip install -e examples/wsindex-connector-notion
export NOTION_TOKEN=secret_...
uv run wsindex fetch https://www.notion.so/myorg/Engineering-Handbook-1f2e3d4c5b6a7890abcdef1234567890
```

```toml
[[connectors]]
type = "notion"
url_pattern = "https://www.notion.so/myorg/*"
token_env = "NOTION_TOKEN"
```

**Why Notion?** Because its API returns neither markdown nor text. A page
is a *tree of blocks* — `heading_2`, `bulleted_list_item`, `to_do`,
`code` — each holding an array of rich-text runs with their own
annotations and links. Both built-in connectors have their text handed to
them: GitHub answers with markdown the author wrote, generic-http gets
HTML that only has to stop being markup. Neither exercises the case where
the document has to be **rebuilt**, which is what most real sources need.
An example that only re-proved the easy case would not test the seam.

## What a connector plugin is

Two things, and the second is the only one that takes thought.

**1. An entry point.** This is the entire integration surface. The name
on the left is the `type` a config entry asks for:

```toml
# in your plugin's pyproject.toml
[project.entry-points."wsindex.connectors"]
notion = "wsindex_connector_notion:NotionConnector"
```

Nothing scans a filesystem. The build backend copies that line into the
installed distribution's metadata, and wsindex reads it back with
`importlib.metadata` — so your connector becomes available by being
*installed*, and wsindex keeps no list of plugins to update.

A plugin may not take a name that is already registered; `github` and
`generic-http` belong to the box. Trying gets a `ConnectorLoadWarning`
and a skip, because a config that says `type = "github"` was written
against the built-in and must keep meaning it.

**2. A `Connector` subclass**, which is the whole contract:

```python
class NotionConnector(Connector):
    def matches(self, url: str) -> bool: ...
    def fetch(self, url: str) -> Document: ...
```

`matches` and `fetch` are separate so a url can be routed without
anything being requested over the network. Make `matches` **narrower than
the config's pattern**: the pattern says which urls the user routed here,
`matches` says which ones you can actually turn into a document. This one
answers for a page url and not for a database view on the same host, and
saying no lets the router fall through to a generic entry instead of
failing.

Everything else is yours, including the HTTP. wsindex's own transport is
private on purpose — a plugin that reached into it would break on an
internal change. This package depends on nothing but the standard
library.

## Failures worth getting right

`fetch` may raise `DocumentNotFound` (there is nothing there) or
`ConnectorError` (anything else). Which is which is a judgement about the
*source*, not about the HTTP status, and it is worth making carefully —
it decides where the user goes looking.

Notion makes it easy, and it is worth contrasting with GitHub:

|            | 404                                                | 401                             |
| ---------- | -------------------------------------------------- | ------------------------------- |
| **GitHub** | no such issue **or** a private repo you cannot see | not used for this               |
| **Notion** | no such page, or not shared with the integration   | the token is missing or invalid |

So the GitHub connector must say "not there, **or** not visible" for a
404 — the API refuses to distinguish, by policy. Notion does
distinguish, so this connector keeps the distinction: a 401 becomes a
`ConnectorError` naming the token variable, and only a 404 becomes
`DocumentNotFound`. Folding them together would send someone hunting for
a page that was there all along.

A missing token is caught before the request, for the same reason: an
anonymous call returns 401, and a user reading "invalid token" when they
never set one would be debugging the wrong thing.

## The url is not a document id

Notion page urls end in a 32-character hex id:

```
https://www.notion.so/myorg/Engineering-Handbook-1f2e3d4c5b6a7890abcdef1234567890
```

The obvious implementation searches the whole url for 32 hex characters.
It is wrong, and the way it is wrong is worth knowing before you write
the same line for another source — a **database view** url carries a
second id in the query:

```
https://www.notion.so/myorg/Roadmap-abc?v=1234567890abcdef1234567890abcdef
                                          ^ the view, not the page
```

That request succeeds against the wrong object. Read the **path** only.

## What is verified here, and what is not

`probes/step29v` in the wsindex repo asked api.notion.com what it says to
a caller with no credentials. That is where the 401/404 split above comes
from, and it is checked: no headers gives "Authorization header must use
the format", a bogus token gives "API token is invalid.".

Everything past authentication — the block shapes, the pagination
envelope, the property schema — follows Notion's published API and is
driven in the tests from payloads of that shape. **No live workspace has
confirmed them.** That line is where this example stops being evidence
and starts being a template, and it is drawn here rather than left for
someone to discover.

```bash
uv run poe example-plugin   # installs both examples and runs their tests
```

## Bounds it keeps

- **One pointed pull.** A child page is rendered as a link, never
  followed. Following it would make this a crawler, and the cost of a
  source would stop being proportional to what the repository mentions.
- **Nesting stops at `MAX_DEPTH = 4`.** Notion nests without limit and
  every level is another request per block; four covers ordinary
  documents.
- **Pagination is followed to the end.** Notion returns at most 100
  blocks per request, so dropping the loop truncates every long page
  silently — a failure that looks like success.
