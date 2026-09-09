"""Step 29a: the connector contract, the router, and the config.

A connector is a pointed pull — fetch the document at this url, nothing
around it. What is tested here is the seam: who claims a url, what a
token does and does not do, and what each source's answer becomes.

The network is stubbed everywhere but one case. `probes/step29a` did the
real asking, before any of this was written, and its findings are what
these tests pin; a suite that re-asked GitHub on every run would be slow,
rate-limited and green only when someone's wifi is up. The one live case
is marked `slow` and deselected by default.
"""

import json
from pathlib import Path

import pytest

from wsindex.config import Config
from wsindex.connectors import (
    SHIPPED,
    ConnectorError,
    ConnectorSpec,
    DocumentNotFound,
    GenericHttpConnector,
    GitHubConnector,
    route,
)
from wsindex.connectors.http import html_to_text

GITHUB = ConnectorSpec(type="github", url_pattern="https://github.com/org/*")
ANY_HTTP = ConnectorSpec(type="generic-http", url_pattern="https://*")


# --- routing -------------------------------------------------------------


def test_the_first_matching_entry_wins() -> None:
    # File order is routing order: someone writing a specific entry above
    # a catch-all expects the specific one to answer.
    connector = route("https://github.com/org/repo/issues/1", [GITHUB, ANY_HTTP])
    assert isinstance(connector, GitHubConnector)


def test_a_connector_that_does_not_understand_the_url_is_passed_over() -> None:
    # The config pattern says "our GitHub"; the connector says which urls
    # it can actually turn into a document. A repository front page is
    # not one, so the catch-all answers instead.
    connector = route("https://github.com/org/repo", [GITHUB, ANY_HTTP])
    assert isinstance(connector, GenericHttpConnector)


def test_an_unclaimed_url_routes_nowhere() -> None:
    assert route("ftp://example.invalid/x", [GITHUB, ANY_HTTP]) is None


def test_an_unknown_type_is_skipped_not_fatal() -> None:
    # The user may have a plugin installed on another machine (step 29c).
    # One unusable entry must not disable the rest.
    unknown = ConnectorSpec(type="not-installed", url_pattern="https://*")
    assert isinstance(route("https://example.invalid/a", [unknown, ANY_HTTP]), GenericHttpConnector)


def test_no_connectors_configured_routes_nowhere() -> None:
    assert route("https://example.invalid/a", []) is None


def test_the_builtins_are_the_two_that_were_probed() -> None:
    # The plan names five out of the box. Only these two could be probed
    # against something real without an instance and a token, and the
    # stage's own rule is probes before code.
    assert {"generic-http", "github"} == SHIPPED


# --- tokens: named in the config, valued in the environment --------------


def test_a_connector_without_a_token_env_needs_none() -> None:
    assert ANY_HTTP.token() is None


def test_a_token_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WSINDEX_TEST_TOKEN", "s3cret")
    spec = ConnectorSpec(type="github", url_pattern="*", token_env="WSINDEX_TEST_TOKEN")
    assert spec.token() == "s3cret"


def test_a_named_but_unset_token_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Falling back to an anonymous request is the worse failure: GitHub
    # then answers 404, which reads as "no such document" and sends the
    # user looking in entirely the wrong place.
    monkeypatch.delenv("WSINDEX_TEST_TOKEN", raising=False)
    spec = ConnectorSpec(type="github", url_pattern="*", token_env="WSINDEX_TEST_TOKEN")
    with pytest.raises(ConnectorError, match="WSINDEX_TEST_TOKEN"):
        spec.token()


# --- github: what the probe found, pinned --------------------------------


def stub_github(monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]) -> list[str]:
    """Answer the API from a dict; record which urls were requested."""
    asked: list[str] = []

    def fake_get(url: str, headers: dict[str, str]) -> tuple[bytes, str]:
        asked.append(url)
        return json.dumps(payload).encode(), "application/json"

    monkeypatch.setattr("wsindex.connectors.github._get", fake_get)
    return asked


ISSUE: dict[str, object] = {
    "html_url": "https://github.com/org/repo/issues/7",
    "title": "Add basic CI",
    "body": "Run tests, formatting, linting.",
    "state": "closed",
    "created_at": "2023-10-05T15:23:57Z",
    "updated_at": "2024-01-01T00:00:00Z",
    "user": {"login": "someone"},
}


def test_an_issue_becomes_a_document(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_github(monkeypatch, ISSUE)
    document = GitHubConnector(GITHUB).fetch("https://github.com/org/repo/issues/7")
    assert document.title == "Add basic CI"
    # The body is already markdown; nothing is converted.
    assert document.text == "Run tests, formatting, linting."
    assert document.metadata["author"] == "someone"
    assert document.metadata["state"] == "closed"
    assert document.metadata["type"] == "issue"


def test_a_pull_request_uses_the_same_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    # The probe's finding: `/issues/{n}` answers for both, so there is no
    # second code path and no url shape to disambiguate.
    asked = stub_github(monkeypatch, {**ISSUE, "pull_request": {"url": "..."}})
    document = GitHubConnector(GITHUB).fetch("https://github.com/org/repo/pull/7")
    assert asked == ["https://api.github.com/repos/org/repo/issues/7"]
    assert document.metadata["type"] == "pull_request"


def test_the_canonical_url_comes_from_the_api(monkeypatch: pytest.MonkeyPatch) -> None:
    # A `pull/7` request and an `issues/7` request name one document; the
    # link should say which one GitHub considers canonical.
    stub_github(monkeypatch, ISSUE)
    document = GitHubConnector(GITHUB).fetch("https://github.com/org/repo/pull/7")
    assert document.url == "https://github.com/org/repo/issues/7"


def test_github_claims_only_issue_and_pull_urls() -> None:
    connector = GitHubConnector(GITHUB)
    assert connector.matches("https://github.com/org/repo/issues/7")
    assert connector.matches("https://github.com/org/repo/pull/7")
    assert not connector.matches("https://github.com/org/repo")
    assert not connector.matches("https://github.com/org/repo/blob/main/README.md")


def test_a_missing_issue_names_the_url_the_user_asked_about(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(url: str, headers: dict[str, str]) -> tuple[bytes, str]:
        raise DocumentNotFound(f"{url} is not there, or not visible")

    monkeypatch.setattr("wsindex.connectors.github._get", fake_get)
    with pytest.raises(DocumentNotFound, match=r"github\.com/org/repo/issues/9"):
        GitHubConnector(GITHUB).fetch("https://github.com/org/repo/issues/9")


def test_non_json_from_github_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "wsindex.connectors.github._get",
        lambda url, headers: (b"<html>maintenance</html>", "text/html"),
    )
    with pytest.raises(ConnectorError, match="not JSON"):
        GitHubConnector(GITHUB).fetch("https://github.com/org/repo/issues/7")


def test_fetching_a_url_github_does_not_handle_is_an_error() -> None:
    with pytest.raises(ConnectorError, match="not a GitHub issue"):
        GitHubConnector(GITHUB).fetch("https://github.com/org/repo")


# --- generic http: the content type decides ------------------------------


def stub_http(monkeypatch: pytest.MonkeyPatch, body: bytes, content_type: str) -> None:
    monkeypatch.setattr("wsindex.connectors.http._get", lambda url, headers: (body, content_type))


def test_markdown_arrives_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_http(monkeypatch, b"# Title\n\nBody.\n", "text/plain; charset=utf-8")
    document = GenericHttpConnector(ANY_HTTP).fetch("https://example.invalid/README.md")
    assert document.text == "# Title\n\nBody.\n"
    assert document.metadata["content_type"] == "text/plain"


def test_html_is_stripped_to_what_a_reader_sees(monkeypatch: pytest.MonkeyPatch) -> None:
    page = (
        b"<html><head><title>Guide</title><script>var x=1;</script></head>"
        b"<body><p>Hello</p></body></html>"
    )
    stub_http(monkeypatch, page, "text/html; charset=utf-8")
    document = GenericHttpConnector(ANY_HTTP).fetch("https://example.invalid/guide")
    assert document.title == "Guide"
    assert document.text == "Hello"
    assert "var x" not in document.text


def test_a_format_with_no_text_in_it_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # A PDF read as UTF-8 is worse than an error, because it succeeds.
    stub_http(monkeypatch, b"%PDF-1.4 ...", "application/pdf")
    with pytest.raises(ConnectorError, match="nothing to read"):
        GenericHttpConnector(ANY_HTTP).fetch("https://example.invalid/paper.pdf")


def test_generic_http_claims_any_http_url() -> None:
    connector = GenericHttpConnector(ANY_HTTP)
    assert connector.matches("https://example.invalid/a")
    assert connector.matches("http://example.invalid/a")
    assert not connector.matches("ftp://example.invalid/a")


# --- the html stripper on its own ----------------------------------------


def test_invisible_elements_do_not_leak() -> None:
    text, _ = html_to_text("<style>.a{}</style><noscript>x</noscript><p>kept</p>")
    assert text == "kept"


def test_nested_invisible_elements_close_correctly() -> None:
    # A counter, not a flag: `<svg>` inside `<head>` would otherwise
    # un-hide the rest of the head when it closes.
    text, _ = html_to_text("<head><svg><title>t</title></svg><meta></head><body>seen</body>")
    assert "seen" in text


def test_block_elements_become_line_breaks() -> None:
    # Without them the page arrives as one paragraph and every line
    # number in it means nothing.
    text, _ = html_to_text("<li>one</li><li>two</li>")
    assert text.splitlines() == ["one", "two"]


def test_a_page_without_a_title_reports_none() -> None:
    _, title = html_to_text("<p>body</p>")
    assert title == ""


# --- the config side -----------------------------------------------------


def test_connectors_default_to_empty() -> None:
    assert Config.default("demo").connectors == []


def test_connectors_are_read_in_file_order(tmp_path: Path) -> None:
    path = tmp_path / "wsindex.toml"
    path.write_text(
        Config.default("demo").save(path).read_text()
        + '\n[[connectors]]\ntype = "github"\nurl_pattern = "https://github.com/org/*"\n'
        + 'token_env = "GH_TOKEN"\n'
        + '\n[[connectors]]\ntype = "generic-http"\nurl_pattern = "https://*"\n',
        encoding="utf-8",
    )
    Config.reset()
    specs = Config(path).connectors
    assert [spec.type for spec in specs] == ["github", "generic-http"]
    assert specs[0].token_env == "GH_TOKEN"
    assert specs[1].token_env is None


def test_an_incomplete_entry_is_skipped(tmp_path: Path) -> None:
    # An entry with no pattern routes nothing; refusing to load the whole
    # config over it would take the workspace down for a typo in a
    # section nothing else depends on.
    path = tmp_path / "wsindex.toml"
    path.write_text(
        Config.default("demo").save(path).read_text()
        + '\n[[connectors]]\ntype = "github"\n'
        + '\n[[connectors]]\ntype = "generic-http"\nurl_pattern = "https://*"\n',
        encoding="utf-8",
    )
    Config.reset()
    assert [spec.type for spec in Config(path).connectors] == ["generic-http"]


def test_init_does_not_write_an_empty_connectors_array(tmp_path: Path) -> None:
    # TOML does not let a static array become an array of tables, so
    # `connectors = []` in a fresh config would break the one documented
    # way to add an entry: appending a `[[connectors]]` block.
    path = Config.default("demo").save(tmp_path / "wsindex.toml")
    assert "connectors" not in path.read_text()


# --- one live case, deselected by default --------------------------------


@pytest.mark.slow
def test_github_really_answers() -> None:
    document = GitHubConnector(
        ConnectorSpec(type="github", url_pattern="https://github.com/astral-sh/*")
    ).fetch("https://github.com/astral-sh/uv/issues/1")
    assert document.title
    assert document.metadata["repository"] == "astral-sh/uv"


# --- _get: the failures a caller has to tell apart ------------------------


class _FakeResponse:
    def __init__(self, body: bytes, headers: dict[str, str]) -> None:
        self._body = body
        self.headers = headers

    def read(self, size: int | None = None) -> bytes:
        return self._body if size is None else self._body[:size]

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def serve(monkeypatch: pytest.MonkeyPatch, body: bytes, headers: dict[str, str]) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: _FakeResponse(body, headers)
    )


def test_get_returns_the_body_and_content_type(monkeypatch: pytest.MonkeyPatch) -> None:
    from wsindex.connectors.http import _get

    serve(monkeypatch, b"hello", {"Content-Type": "text/plain"})
    assert _get("https://example.invalid/a", {}) == (b"hello", "text/plain")


def test_a_404_is_a_missing_document(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    from wsindex.connectors.http import _get

    def raise_404(request: object, timeout: float | None = None) -> None:
        raise urllib.error.HTTPError("https://x", 404, "Not Found", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", raise_404)
    with pytest.raises(DocumentNotFound):
        _get("https://example.invalid/gone", {})


def test_another_status_is_a_plain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    from wsindex.connectors.http import _get

    def raise_500(request: object, timeout: float | None = None) -> None:
        raise urllib.error.HTTPError("https://x", 500, "Server Error", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", raise_500)
    with pytest.raises(ConnectorError, match="answered 500"):
        _get("https://example.invalid/a", {})


def test_an_unreachable_host_is_a_plain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    from wsindex.connectors.http import _get

    def unreachable(request: object, timeout: float | None = None) -> None:
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", unreachable)
    with pytest.raises(ConnectorError, match="could not be reached"):
        _get("https://example.invalid/a", {})


def test_a_declared_oversize_response_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from wsindex.connectors.http import MAX_BYTES, _get

    serve(monkeypatch, b"x", {"Content-Length": str(MAX_BYTES + 1), "Content-Type": "text/plain"})
    with pytest.raises(ConnectorError, match="larger than the limit"):
        _get("https://example.invalid/big", {})


def test_an_undeclared_oversize_response_is_refused_too(monkeypatch: pytest.MonkeyPatch) -> None:
    # A server that sends no Content-Length would slip past the header
    # check, so the read is bounded at one byte past the limit.
    from wsindex.connectors.http import MAX_BYTES, _get

    serve(monkeypatch, b"x" * (MAX_BYTES + 1), {"Content-Type": "text/plain"})
    with pytest.raises(ConnectorError, match="larger than"):
        _get("https://example.invalid/big", {})


def test_a_token_reaches_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    # The one thing a token must do: appear as a header. Checked here
    # rather than against GitHub, since a wrong header is a 404 there and
    # 404 is also what a missing document looks like.
    monkeypatch.setenv("WSINDEX_TEST_TOKEN", "s3cret")
    seen: dict[str, str] = {}

    def capture(url: str, headers: dict[str, str]) -> tuple[bytes, str]:
        seen.update(headers)
        return json.dumps(ISSUE).encode(), "application/json"

    monkeypatch.setattr("wsindex.connectors.github._get", capture)
    spec = ConnectorSpec(
        type="github", url_pattern="https://github.com/org/*", token_env="WSINDEX_TEST_TOKEN"
    )
    GitHubConnector(spec).fetch("https://github.com/org/repo/issues/7")
    assert seen["Authorization"] == "Bearer s3cret"


def test_the_generic_connector_sends_its_token_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WSINDEX_TEST_TOKEN", "s3cret")
    seen: dict[str, str] = {}

    def capture(url: str, headers: dict[str, str]) -> tuple[bytes, str]:
        seen.update(headers)
        return b"# Notes", "text/markdown"

    monkeypatch.setattr("wsindex.connectors.http._get", capture)
    spec = ConnectorSpec(
        type="generic-http",
        url_pattern="https://intranet.invalid/*",
        token_env="WSINDEX_TEST_TOKEN",
    )
    document = GenericHttpConnector(spec).fetch("https://intranet.invalid/page")
    assert seen["Authorization"] == "Bearer s3cret"
    assert document.text == "# Notes"
