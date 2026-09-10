"""The benchmark harness's judgement, which is the part that was hard.

Not the scenarios — those need a corpus and minutes, and what they
measure is the product, not this file. What is tested here is what the
harness *decides*: when a change counts as a regression, when a number
is too small to argue about, and when two runs are not comparable at
all. Every one of those was got wrong at least once while writing it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from bench import FLOOR, REGRESSION, Measurement, busy, compare, environment, peak_mb

HERE = {"platform": "Darwin arm64", "python": "3.12.12", "embedder": "real"}


def saved(seconds: float, *, scenario: str = "index-cold", **env: str) -> dict[str, Any]:
    """A baseline holding one scenario, from this machine unless told otherwise."""
    return {
        "environment": {**HERE, **env},
        "runs": [{"scenario": scenario, "seconds": seconds, "peak_mb": 100.0, "detail": {}}],
    }


def measured(seconds: float, *, scenario: str = "index-cold") -> list[Measurement]:
    return [Measurement(scenario=scenario, seconds=seconds, peak_mb=100.0)]


def test_a_run_within_the_noise_says_nothing() -> None:
    # Two full runs of the harness differ by at most 3%, measured. A
    # threshold that fires there reports the laptop.
    assert compare(measured(10.2), saved(10.0), HERE) == []


def test_a_real_slowdown_is_named() -> None:
    lines = compare(measured(15.0), saved(10.0), HERE)

    assert len(lines) == 1
    assert "50% slower" in lines[0]
    assert "index-cold" in lines[0]


def test_a_speed_up_is_reported_too() -> None:
    # An unexplained one is as much a signal as a slowdown: review 4
    # chased a 130x speed-up and found a measurement error.
    lines = compare(measured(5.0), saved(10.0), HERE)

    assert "50% faster" in lines[0]


def test_a_change_too_small_to_matter_is_ignored() -> None:
    # `compact` runs in 0.02 s, where doubling is 20 milliseconds. It was
    # crying loudest of anything before the floor went in.
    before, after = 0.02, 0.02 + FLOOR / 2

    assert (after - before) / before > REGRESSION, "the relative change is large"
    now = measured(after, scenario="compact")
    then = saved(before, scenario="compact")

    assert compare(now, then, HERE) == []


def test_a_scenario_the_baseline_never_had_is_skipped() -> None:
    assert compare(measured(10.0, scenario="brand-new"), saved(10.0), HERE) == []


@pytest.mark.parametrize("key", ["platform", "python", "embedder"])
def test_a_baseline_from_another_machine_says_so_first(key: str) -> None:
    # Numbers from another machine are not a baseline, they are a
    # different question — and the warning has to lead, or the table
    # under it reads as a verdict.
    lines = compare(measured(10.0), saved(10.0, **{key: "something else"}), HERE)

    assert lines
    assert "not from here" in lines[0]
    assert key in lines[0]


def test_the_same_machine_is_not_warned_about() -> None:
    assert compare(measured(10.0), saved(10.0), HERE) == []


def test_peak_memory_is_in_megabytes_on_this_platform() -> None:
    # `ru_maxrss` is bytes on macOS and kilobytes on Linux, which is how
    # 400 MB silently becomes 400 GB in a report. A pytest process is
    # somewhere between ten megabytes and a few hundred on either.
    assert 5 < peak_mb() < 5000


# --- whether the machine was ours to measure on ---------------------------


def test_a_quiet_machine_is_not_complained_about(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bench.load", lambda: (0.7, 14))

    assert busy() is None


def test_a_machine_someone_else_is_using_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    # The case this exists for, with the numbers it actually had: another
    # program on ten of fourteen cores, and a benchmark reporting a 30%
    # regression that was entirely somebody else's build.
    monkeypatch.setattr("bench.load", lambda: (7.93, 14))

    said = busy()

    assert said is not None
    assert "7.93" in said and "14 cores" in said and "57%" in said


def test_an_ordinary_working_laptop_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The calibration, as a test, because the first threshold failed it.

    Sampled over ninety idle seconds with a browser and an IDE open, this
    machine's load average sits at 2.18-3.10 on fourteen cores. A guard
    that fires there is a guard everybody disables.
    """
    for average in (2.18, 3.10):
        monkeypatch.setattr("bench.load", lambda a=average: (a, 14))
        assert busy() is None, f"load {average} is this laptop doing nothing"


def test_the_threshold_is_a_share_of_the_machine_not_a_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The same load average means different things on a laptop and on a
    # build server, so the line is drawn per core.
    monkeypatch.setattr("bench.load", lambda: (4.0, 4))
    assert busy() is not None

    monkeypatch.setattr("bench.load", lambda: (4.0, 64))
    assert busy() is None


def test_the_environment_block_records_the_load(monkeypatch: pytest.MonkeyPatch) -> None:
    # A number without its machine is a rumour, and the machine's state
    # turns out to be part of the machine.
    monkeypatch.setattr("bench.load", lambda: (7.93, 14))

    assert environment()["load"] == "7.93 on 14 cores"
