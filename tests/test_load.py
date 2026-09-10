"""The load harness's judgement, which is the part a test can hold.

Not the scenarios: those need a server, a corpus and a minute, and what
they measure is the product rather than this file. What is tested here is
what the harness *decides* — what counts as a failure, what the SLO
covers, and whether it can tell the server's account of a run from its
own. Each of those was got wrong at least once while writing it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from load import (
    FAILURES_ALLOWED,
    P95_BUDGET_MS,
    P99_BUDGET_MS,
    Result,
    Sample,
    cross_check,
    judge,
)


def searching(*latencies: float, scenario: str = "search") -> Result:
    return Result(
        scenario=scenario,
        samples=[Sample(ms=ms, status=200, hits=5) for ms in latencies],
        seconds=1.0,
    )


def verdicts(result: Result) -> dict[str, bool]:
    return {verdict.rule: verdict.ok for verdict in judge([result])}


# --- what counts as a failure ---------------------------------------------


def test_a_transport_error_is_a_failure() -> None:
    assert Sample(ms=1.0, status=0, error="ConnectError").failed


def test_a_server_error_is_a_failure() -> None:
    assert Sample(ms=1.0, status=503).failed


def test_a_refusal_is_not_a_failure() -> None:
    # 409 is the one-writer rule answering, and counting it as an outage
    # would make ADR-10 look like a bug on every dashboard.
    assert not Sample(ms=1.0, status=409).failed


def test_a_client_error_is_not_a_failure_either() -> None:
    # A 400 means the request was wrong; the server answered correctly.
    assert not Sample(ms=1.0, status=400).failed


def test_one_failed_request_fails_the_run() -> None:
    result = searching(10.0)
    result.samples.append(Sample(ms=5.0, status=500))

    assert verdicts(result)["search: no failed requests"] is (FAILURES_ALLOWED >= 1)


# --- the budgets ----------------------------------------------------------


def test_a_run_inside_the_budget_passes() -> None:
    assert all(verdicts(searching(*[10.0] * 100)).values())


def test_a_p95_over_budget_fails() -> None:
    # Ninety fast requests and ten slow ones: the mean would pass and the
    # p95 must not.
    slow = [P95_BUDGET_MS + 100] * 10
    result = searching(*([10.0] * 90 + slow))

    judged = verdicts(result)

    assert judged[f"search: p95 < {P95_BUDGET_MS:.0f} ms"] is False


def test_a_tail_hidden_under_a_good_p95_still_fails_the_p99() -> None:
    # The reason the p99 rule exists: one request in a hundred stalling
    # for two seconds leaves the p95 untouched.
    result = searching(*([10.0] * 98 + [P99_BUDGET_MS + 500] * 2))

    judged = verdicts(result)

    assert judged[f"search: p95 < {P95_BUDGET_MS:.0f} ms"] is True
    assert judged[f"search: p99 < {P99_BUDGET_MS:.0f} ms"] is False


def test_the_budget_applies_to_a_busy_server_too() -> None:
    """The same number while indexing, which is a decision not an oversight.

    A person waiting on a search does not care that the server is busy
    writing. A kinder budget for the busy case would be a budget for the
    server's comfort rather than theirs — and it is the case that
    actually fails, so a separate number would have been a way of not
    finding out.
    """
    busy = searching(*[P95_BUDGET_MS + 1] * 100, scenario="search-during-index")

    assert verdicts(busy)[f"search-during-index: p95 < {P95_BUDGET_MS:.0f} ms"] is False


def test_percentiles_are_nearest_rank() -> None:
    result = searching(10.0, 20.0, 30.0, 40.0, 100.0)

    assert result.percentile(0.50) == 30.0
    assert result.percentile(0.95) == 100.0


def test_a_scenario_that_measured_nothing_reports_zero_rather_than_crashing() -> None:
    assert Result(scenario="search").percentile(0.95) == 0.0


# --- the one-writer rule --------------------------------------------------


def contending(*statuses: int) -> Result:
    return Result(
        scenario="index-contention",
        samples=[Sample(ms=10.0, status=status) for status in statuses],
        seconds=1.0,
    )


RULE = "one-writer rule: exactly one concurrent run is accepted"


def test_one_accepted_and_the_rest_refused_is_the_rule_holding() -> None:
    assert verdicts(contending(200, 409, 409, 409))[RULE] is True


def test_two_accepted_runs_break_it() -> None:
    # Two indexing runs at once is the failure ADR-10 exists to prevent.
    assert verdicts(contending(200, 200, 409, 409))[RULE] is False


def test_a_refusal_that_is_neither_breaks_it() -> None:
    # A 500 under contention is not the lock working, whatever the count.
    assert verdicts(contending(200, 409, 500, 409))[RULE] is False


# --- the cross-check against the server's own metrics ---------------------


def test_two_readings_that_agree_say_so() -> None:
    note = cross_check(5.0, 105.0, [searching(*[10.0] * 100)])

    assert "they agree" in note


def test_a_disagreement_is_named() -> None:
    note = cross_check(0.0, 99.0, [searching(*[10.0] * 100)])

    assert "disagree" in note


def test_it_is_a_difference_of_two_readings_not_a_total() -> None:
    """The warm-up bug, kept as a test.

    The first version compared the server's lifetime count against the
    scenarios' samples and reported a disagreement of exactly one: the
    warm-up search, which the client makes and does not count. The
    metrics were right and the check was wrong.
    """
    warmed = 1.0

    note = cross_check(warmed, warmed + 100, [searching(*[10.0] * 100)])

    assert "they agree" in note


def test_no_parser_is_a_skip_not_a_verdict() -> None:
    assert "skipped" in cross_check(None, None, [searching(10.0)])


def test_only_search_scenarios_are_counted_against_the_server() -> None:
    # `index-contention` makes no searches; including it would make the
    # two readings disagree by however many runs were refused.
    note = cross_check(0.0, 3.0, [searching(1.0, 2.0, 3.0), contending(200, 409)])

    assert "they agree" in note


@pytest.mark.parametrize("scenario", ["search", "search-during-sync", "search-during-index"])
def test_every_search_scenario_is_held_to_the_budget(scenario: str) -> None:
    judged = verdicts(searching(*[10.0] * 100, scenario=scenario))

    assert f"{scenario}: p95 < {P95_BUDGET_MS:.0f} ms" in judged
