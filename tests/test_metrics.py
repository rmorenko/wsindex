"""The metrics, checked against the format's own parser rather than a regex.

Emitting Prometheus' exposition format by hand is only defensible if
"valid" is something the tests establish. So they do: every assertion
about the body goes through `prometheus_client.parser`, the same code a
Prometheus server runs — a dev dependency for exactly this, never a
runtime one.

The rest is about what must *not* be in there. No query text, no path
from a caller: a metric label is the easiest place in a server to leak
what people searched for and the easiest place to let a stranger grow
memory without bound.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from prometheus_client.parser import text_string_to_metric_families

from wsindex.server.metrics import LATENCY_BUCKETS, Metrics, route_of


@pytest.fixture
def metrics() -> Metrics:
    return Metrics()


def parse(body: str) -> dict[str, Any]:
    """Every family in the body, by name, as the official parser sees it.

    "As the parser sees it" is load-bearing: a counter family written
    `wsindex_index_runs_total` is named `wsindex_index_runs`, because the
    `_total` belongs to the sample rather than the family. The keys here
    are therefore the stripped names, and the samples keep the suffix.
    """
    return {family.name: family for family in text_string_to_metric_families(body)}


def samples(body: str, name: str) -> dict[tuple[tuple[str, str], ...], float]:
    """Every sample called `name`, keyed by its sorted labels."""
    found = {}
    for family in text_string_to_metric_families(body):
        for sample in family.samples:
            if sample.name == name:
                found[tuple(sorted(sample.labels.items()))] = sample.value
    return found


def rendered(metrics: Metrics, **state: Any) -> str:
    return metrics.render(**{"version": "0.1.0", "indexing": False, "repos": 2, **state})


# --- the format itself ----------------------------------------------------


def test_an_empty_server_still_emits_valid_exposition(metrics: Metrics) -> None:
    # Before the first request, which is when a scraper first arrives.
    body = rendered(metrics)

    assert "wsindex_build_info" in parse(body)


def test_every_family_declares_its_type(metrics: Metrics) -> None:
    metrics.observed(route="/search", method="GET", status=200, seconds=0.01)
    metrics.indexed("ok", seconds=3.0, written=10, deleted=1)

    families = parse(rendered(metrics))

    assert families["wsindex_http_requests"].type == "counter"
    assert families["wsindex_http_request_seconds"].type == "histogram"
    assert families["wsindex_indexing"].type == "gauge"


def test_a_family_with_no_samples_still_announces_itself(metrics: Metrics) -> None:
    # A counter that appears only on its first increment is a counter a
    # scraper cannot tell from a restart.
    body = rendered(metrics)

    assert "wsindex_index_runs" in parse(body)
    assert samples(body, "wsindex_index_runs_total") == {}


def test_buckets_are_cumulative_and_end_at_inf(metrics: Metrics) -> None:
    for seconds in (0.001, 0.02, 0.4, 30.0):
        metrics.observed(route="/search", method="GET", status=200, seconds=seconds)

    buckets = samples(rendered(metrics), "wsindex_http_request_seconds_bucket")
    by_bound = {
        float(dict(labels)["le"]): value
        for labels, value in buckets.items()
        if "le" in dict(labels)
    }

    assert by_bound[0.005] == 1
    assert by_bound[0.025] == 2
    assert by_bound[0.5] == 3
    assert by_bound[math.inf] == 4
    assert sorted(by_bound.items()) == sorted(by_bound.items(), key=lambda pair: pair[0])


def test_a_value_exactly_on_a_bound_falls_inside_it(metrics: Metrics) -> None:
    # `le` means less than *or equal*. Off by one here is off by one in
    # every dashboard built on it.
    metrics.observed(route="/search", method="GET", status=200, seconds=0.5)

    buckets = samples(rendered(metrics), "wsindex_http_request_seconds_bucket")

    assert buckets[(("le", "0.5"), ("method", "GET"), ("route", "/search"))] == 1
    assert buckets[(("le", "0.25"), ("method", "GET"), ("route", "/search"))] == 0


def test_sum_and_count_agree_with_what_was_observed(metrics: Metrics) -> None:
    for seconds in (0.1, 0.2, 0.3):
        metrics.observed(route="/search", method="GET", status=200, seconds=seconds)

    body = rendered(metrics)
    labels = (("method", "GET"), ("route", "/search"))

    assert samples(body, "wsindex_http_request_seconds_count")[labels] == 3
    assert samples(body, "wsindex_http_request_seconds_sum")[labels] == pytest.approx(0.6)


def test_the_slo_has_a_bucket_of_its_own() -> None:
    # 500 ms is step 39's SLO, so the compliance ratio is a division of
    # two scraped series rather than a recording rule somebody has to
    # remember to write.
    assert 0.5 in LATENCY_BUCKETS


def test_a_label_value_with_a_quote_in_it_does_not_break_the_body(metrics: Metrics) -> None:
    metrics.observed(route='/odd"path\\', method="GET", status=200, seconds=0.01)

    found = samples(rendered(metrics), "wsindex_http_requests_total")

    assert (("method", "GET"), ("route", '/odd"path\\'), ("status", "200")) in found


# --- what it counts -------------------------------------------------------


def test_requests_are_counted_by_route_method_and_status(metrics: Metrics) -> None:
    metrics.observed(route="/search", method="GET", status=200, seconds=0.01)
    metrics.observed(route="/search", method="GET", status=200, seconds=0.01)
    metrics.observed(route="/search", method="GET", status=401, seconds=0.01)

    found = samples(rendered(metrics), "wsindex_http_requests_total")

    assert found[(("method", "GET"), ("route", "/search"), ("status", "200"))] == 2
    assert found[(("method", "GET"), ("route", "/search"), ("status", "401"))] == 1


def test_a_refused_run_is_not_a_failed_one(metrics: Metrics) -> None:
    # The one-writer rule working looks identical to an outage on a graph
    # that puts both under `error`.
    metrics.indexed("ok", seconds=2.0)
    metrics.indexed("busy")
    metrics.indexed("error")

    found = samples(rendered(metrics), "wsindex_index_runs_total")

    assert found[(("outcome", "ok"),)] == 1
    assert found[(("outcome", "busy"),)] == 1
    assert found[(("outcome", "error"),)] == 1


def test_only_a_completed_run_contributes_work_and_time(metrics: Metrics) -> None:
    metrics.indexed("ok", seconds=2.0, written=10, deleted=3)
    metrics.indexed("busy", seconds=99.0, written=99, deleted=99)

    body = rendered(metrics)

    assert samples(body, "wsindex_chunks_written_total")[()] == 10
    assert samples(body, "wsindex_chunks_deleted_total")[()] == 3
    assert samples(body, "wsindex_index_seconds_count")[()] == 1


def test_a_server_that_never_indexed_says_so_with_a_zero(metrics: Metrics) -> None:
    body = rendered(metrics)

    assert samples(body, "wsindex_last_index_success_timestamp_seconds")[()] == 0.0


def test_the_live_gauges_come_from_the_caller(metrics: Metrics) -> None:
    body = rendered(metrics, indexing=True, repos=7)

    assert samples(body, "wsindex_indexing")[()] == 1.0
    assert samples(body, "wsindex_repos")[()] == 7.0


def test_the_version_is_a_label_on_a_constant(metrics: Metrics) -> None:
    found = samples(rendered(metrics, version="9.9.9"), "wsindex_build_info")

    assert found[(("version", "9.9.9"),)] == 1.0


# --- cardinality, which is the way a metrics endpoint breaks a server -----


def test_an_unmatched_request_is_labelled_other_not_by_its_path() -> None:
    # Otherwise anyone with a URL bar can add a series per request until
    # the process runs out of memory.
    assert route_of({"path": "/../../etc/passwd"}) == "other"
    assert route_of({"route": object()}) == "other"


def test_a_matched_request_is_labelled_by_its_template() -> None:
    class Route:
        path = "/repos/{repo_id}"

    assert route_of({"route": Route(), "path": "/repos/wsindex"}) == "/repos/{repo_id}"


# The endpoint itself — that it is served, guarded, and fed by real
# traffic — is tested in `test_server.py`, where the application fixtures
# already are. What is here is the format and the counting.
