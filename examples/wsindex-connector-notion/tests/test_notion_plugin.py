"""Tests for the Notion connector, driven from recorded-shape payloads.

No network. What these can honestly check is everything between the
API's answer and the `Document`: the block tree becoming markdown, the
url becoming an id, and each of Notion's failures becoming the right one
of ours. What they cannot check is that Notion really answers this way —
that needs a workspace and a token, and the package README says so.

The one thing that *was* checked against the live API is what
`probes/step29v` could ask without credentials, and both of its findings
have a test here: the 401/404 split, and the database-view url.
"""

import urllib.error

import pytest
from wsindex.connectors import ConnectorError, ConnectorSpec, DocumentNotFound

from wsindex_connector_notion import NotionConnector, page_id

SPEC = ConnectorSpec(
    type="notion",
    url_pattern="https://www.notion.so/myorg/*",
    token_env="NOTION_TEST_TOKEN",
)
PAGE_URL = "https://www.notion.so/myorg/Handbook-1f2e3d4c5b6a7890abcdef1234567890"
PAGE_ID = "1f2e3d4c5b6a7890abcdef1234567890"


def rich(text: str, **annotations: object) -> dict[str, object]:
    return {"plain_text": text, "annotations": annotations}


def block(kind: str, text: str = "", **extra: object) -> dict[str, object]:
    body: dict[str, object] = {"rich_text": [rich(text)] if text else []}
    body.update(extra)
    return {"id": f"block-{kind}", "type": kind, kind: body, "has_children": False}


PAGE = {
    "url": PAGE_URL,
    "created_time": "2024-01-05T10:00:00.000Z",
    "last_edited_time": "2024-03-01T12:00:00.000Z",
    "properties": {
        # A user-defined name: the title is found by `type`, not by key.
        "Название": {"type": "title", "title": [rich("Engineering Handbook")]},
        "Owner": {"type": "people", "people": []},
    },
}


@pytest.fixture(autouse=True)
def token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTION_TEST_TOKEN", "secret_test")


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, object]]:
    """Answer `_get` from a dict keyed by the part of the endpoint that matters."""
    answers: dict[str, dict[str, object]] = {}

    def fake_get(self: NotionConnector, endpoint: str, url: str) -> dict[str, object]:
        key = "children" if "/children" in endpoint else "page"
        return answers.get(key, {})

    monkeypatch.setattr(NotionConnector, "_get", fake_get)
    return answers


# --- what the probe found ------------------------------------------------


def test_a_database_view_id_is_not_mistaken_for_a_page() -> None:
    # The measured trap: `?v=<32 hex>` is the *view* id. Reading the whole
    # url would fetch the wrong object and look like it worked.
    view = "https://www.notion.so/myorg/Roadmap-abc?v=1234567890abcdef1234567890abcdef"
    assert page_id(view) is None
    assert not NotionConnector(SPEC).matches(view)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (PAGE_URL, PAGE_ID),
        ("https://notion.so/1f2e3d4c5b6a7890abcdef1234567890", PAGE_ID),
        ("https://myorg.notion.site/Public-1f2e3d4c5b6a7890abcdef1234567890", PAGE_ID),
        # Dashed form, which is what the API itself hands back.
        ("https://www.notion.so/1f2e3d4c-5b6a-7890-abcd-ef1234567890", PAGE_ID),
        ("https://www.notion.so/myorg", None),
    ],
)
def test_the_id_comes_out_of_the_path(url: str, expected: str | None) -> None:
    assert page_id(url) == expected


def test_a_rejected_token_is_not_a_missing_page(monkeypatch: pytest.MonkeyPatch) -> None:
    # Notion's own distinction, and the reason this connector keeps it:
    # folding 401 into "not found" sends someone hunting for a page that
    # was there all along.
    def raise_401(request: object, timeout: float | None = None) -> None:
        raise urllib.error.HTTPError(PAGE_URL, 401, "unauthorized", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", raise_401)

    with pytest.raises(ConnectorError, match=r"\$NOTION_TEST_TOKEN") as caught:
        NotionConnector(SPEC).fetch(PAGE_URL)
    assert not isinstance(caught.value, DocumentNotFound)


def test_a_page_that_is_not_shared_is_a_missing_document(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_404(request: object, timeout: float | None = None) -> None:
        raise urllib.error.HTTPError(PAGE_URL, 404, "not found", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", raise_404)

    with pytest.raises(DocumentNotFound, match="not shared"):
        NotionConnector(SPEC).fetch(PAGE_URL)


def test_rate_limiting_says_what_it_is(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_429(request: object, timeout: float | None = None) -> None:
        raise urllib.error.HTTPError(PAGE_URL, 429, "too many", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", raise_429)

    with pytest.raises(ConnectorError, match="rate-limiting"):
        NotionConnector(SPEC).fetch(PAGE_URL)


def test_a_missing_token_is_named_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOTION_TEST_TOKEN")

    with pytest.raises(ConnectorError, match="NOTION_TEST_TOKEN"):
        NotionConnector(SPEC).fetch(PAGE_URL)


# --- blocks into markdown ------------------------------------------------


def test_a_page_becomes_markdown(api: dict[str, dict[str, object]]) -> None:
    api["page"] = PAGE
    api["children"] = {
        "results": [
            block("heading_1", "Overview"),
            block("paragraph", "We ship on Fridays."),
            block("bulleted_list_item", "Write it down"),
            block("to_do", "Review the draft", checked=True),
            block("code", "uv run wsindex index", language="bash"),
            block("divider"),
        ],
        "has_more": False,
    }

    document = NotionConnector(SPEC).fetch(PAGE_URL)

    assert document.title == "Engineering Handbook"
    assert document.text == (
        "# Overview\n\n"
        "We ship on Fridays.\n\n"
        "- Write it down\n\n"
        "- [x] Review the draft\n\n"
        "```bash\nuv run wsindex index\n```\n\n"
        "---"
    )
    assert document.metadata["updated"] == "2024-03-01T12:00:00.000Z"


def test_formatting_survives_the_trip(api: dict[str, dict[str, object]]) -> None:
    # Notion splits a sentence wherever formatting changes, so this is
    # three runs, not one string with markers in it.
    api["page"] = PAGE
    api["children"] = {
        "results": [
            {
                "id": "b1",
                "type": "paragraph",
                "has_children": False,
                "paragraph": {
                    "rich_text": [
                        rich("Run "),
                        rich("wsindex sync", code=True),
                        rich(" nightly", bold=True),
                        {"plain_text": "docs", "annotations": {}, "href": "https://x.invalid"},
                    ]
                },
            }
        ],
        "has_more": False,
    }

    assert NotionConnector(SPEC).fetch(PAGE_URL).text == (
        "Run `wsindex sync`** nightly**[docs](https://x.invalid)"
    )


def test_an_unsupported_block_contributes_nothing(api: dict[str, dict[str, object]]) -> None:
    # An embed or a file has no text. Inventing a placeholder would put
    # words in the document that nobody wrote.
    api["page"] = PAGE
    api["children"] = {
        "results": [block("paragraph", "Kept."), block("embed"), block("unsupported")],
        "has_more": False,
    }

    assert NotionConnector(SPEC).fetch(PAGE_URL).text == "Kept."


def test_a_child_page_is_linked_not_followed(api: dict[str, dict[str, object]]) -> None:
    # The boundary the whole connector layer keeps: one pointed pull.
    api["page"] = PAGE
    api["children"] = {
        "results": [{"id": "c", "type": "child_page", "child_page": {"title": "Onboarding"}}],
        "has_more": False,
    }

    assert "Onboarding (child page)" in NotionConnector(SPEC).fetch(PAGE_URL).text


def test_a_page_with_no_title_property_does_not_break(api: dict[str, dict[str, object]]) -> None:
    api["page"] = {"url": PAGE_URL, "properties": {"Owner": {"type": "people", "people": []}}}
    api["children"] = {"results": [block("paragraph", "Body.")], "has_more": False}

    assert NotionConnector(SPEC).fetch(PAGE_URL).title == ""


# --- the shapes that need more than one request --------------------------


def test_a_long_page_is_paged_to_the_end(monkeypatch: pytest.MonkeyPatch) -> None:
    # Dropping the pagination loop would silently truncate every page
    # over a hundred blocks — the failure that looks like success.
    pages = [
        {"results": [block("paragraph", "First")], "has_more": True, "next_cursor": "c1"},
        {"results": [block("paragraph", "Second")], "has_more": False},
    ]
    asked: list[str] = []

    def fake_get(self: NotionConnector, endpoint: str, url: str) -> dict[str, object]:
        asked.append(endpoint)
        if "/children" not in endpoint:
            return PAGE
        return pages[len([a for a in asked if "/children" in a]) - 1]

    monkeypatch.setattr(NotionConnector, "_get", fake_get)

    assert NotionConnector(SPEC).fetch(PAGE_URL).text == "First\n\nSecond"
    assert "start_cursor=c1" in asked[-1]


def test_has_more_without_a_cursor_does_not_loop_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(self: NotionConnector, endpoint: str, url: str) -> dict[str, object]:
        if "/children" not in endpoint:
            return PAGE
        return {"results": [block("paragraph", "One")], "has_more": True}

    monkeypatch.setattr(NotionConnector, "_get", fake_get)

    assert NotionConnector(SPEC).fetch(PAGE_URL).text == "One"


def test_nested_blocks_are_indented_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    # Notion nests without limit; this stops at MAX_DEPTH so one document
    # cannot turn into an unbounded crawl.
    def fake_get(self: NotionConnector, endpoint: str, url: str) -> dict[str, object]:
        if "/children" not in endpoint:
            return PAGE
        nested = block("bulleted_list_item", "Level")
        nested["has_children"] = True
        return {"results": [nested], "has_more": False}

    monkeypatch.setattr(NotionConnector, "_get", fake_get)

    text = NotionConnector(SPEC).fetch(PAGE_URL).text
    assert text.startswith("- Level\n\n  - Level")
    # Five levels rendered: the top one plus MAX_DEPTH of recursion.
    assert text.count("- Level") == 5
