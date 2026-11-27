from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from redline.core import (
    Attack,
    AttackCategory,
    Case,
    Outcome,
    RunResult,
    TargetResponse,
    ToolCallMatch,
)
from redline.runner import (
    AttemptEstimate,
    Pricing,
    RunConfig,
    RunReport,
    Trial,
    estimate_run,
    estimate_tokens,
    measured_seconds_per_attempt,
)

PLAIN = Attack(id="d-01", name="x", category=AttackCategory.DIRECT_INJECTION, prompt="Hi")
TOOL = Attack(
    id="ea-01",
    name="x",
    category=AttackCategory.EXCESSIVE_AGENCY,
    prompt="Email them",
    success_tool_call=ToolCallMatch(tool="send_email"),
)


class FakeFactory:
    """Only what estimating needs. Every attempt costs the same fixed amount."""

    name = "fake"
    model = "m"

    def __init__(self, *, has_tools: bool = True, description: dict[str, Any] | None = None):
        self.has_tools = has_tools
        self._description = description or {"temperature": 0.7, "phrases": ("a", "b")}

    def describe(self) -> Mapping[str, Any]:
        return self._description

    def preflight(self) -> None:
        pass

    def make_trial(self, attack: Case, seed: int) -> Trial:
        raise AssertionError("estimating must not build or send anything")

    def estimate_attempt(self, case: Case) -> AttemptEstimate:
        return AttemptEstimate(requests=2, input_tokens=1000, output_tokens=500)


def report_with(latencies_ms: list[float | None], target: dict[str, Any]) -> RunReport:
    results = tuple(
        RunResult(
            attack_id="d-01",
            attempt=index,
            target="fake",
            model="m",
            config_hash="h",
            seed=index,
            outcome=Outcome.ERROR if ms is None else Outcome.RESISTED,
            response=None if ms is None else TargetResponse(text="no", model="m", latency_ms=ms),
            error="boom" if ms is None else None,
        )
        for index, ms in enumerate(latencies_ms)
    )
    now = datetime.now(UTC)
    return RunReport(
        results=results,
        config_hash="h",
        cases_hash="c",
        target=target,
        started_at=now,
        finished_at=now,
    )


@pytest.mark.parametrize(("text", "tokens"), [("", 0), ("abc", 1), ("abcd", 2), ("a" * 300, 100)])
def test_tokens_are_guessed_from_length_and_rounded_up(text: str, tokens: int) -> None:
    assert estimate_tokens(text) == tokens


def test_price_is_per_million_tokens_with_separate_input_and_output_rates() -> None:
    pricing = Pricing(input_per_mtok=0.30, output_per_mtok=2.50)
    assert pricing.cost_usd(1_000_000, 0) == pytest.approx(0.30)
    assert pricing.cost_usd(2_000_000, 1_000_000) == pytest.approx(3.10)
    assert Pricing.free().cost_usd(10**9, 10**9) == 0


@pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan")])
def test_nonsense_prices_are_rejected(bad: float) -> None:
    with pytest.raises(ValidationError):
        Pricing(input_per_mtok=bad, output_per_mtok=0)


def test_totals_cover_every_case_and_repeat() -> None:
    estimate = estimate_run(
        [PLAIN, TOOL],
        FakeFactory(),
        RunConfig(repeats=3),
        Pricing(input_per_mtok=1, output_per_mtok=2),
    )
    assert estimate.attempts == 6
    assert estimate.requests == 12
    assert (estimate.input_tokens, estimate.output_tokens) == (6000, 3000)
    assert estimate.cost_usd == pytest.approx(0.012)
    assert estimate.seconds is None


def test_attempts_the_runner_would_skip_cost_nothing() -> None:
    estimate = estimate_run(
        [PLAIN, TOOL], FakeFactory(has_tools=False), RunConfig(), Pricing.free()
    )
    assert (estimate.attempts, estimate.requests) == (1, 2)


def test_time_is_per_attempt_and_not_divided_by_concurrency() -> None:
    estimate = estimate_run(
        [PLAIN],
        FakeFactory(),
        RunConfig(repeats=4, max_concurrency=4),
        Pricing.free(),
        seconds_per_attempt=2.5,
    )
    assert estimate.seconds == 10


@pytest.mark.parametrize(
    ("parallel", "concurrency", "rpm", "seconds"),
    [
        (4, 4, None, 10),  # 8 attempts x 5s, four at a time
        (8, 2, None, 20),  # the run only sends two at a time
        (1, 4, None, 40),  # a queueing server gains nothing
        (4, 4, 6, 160),  # 16 requests at 6 a minute take longer than the work
        (1, 1, 600, 40),  # a generous limit changes nothing
    ],
)
def test_time_allows_for_parallel_servers_and_rate_limits(
    parallel: int, concurrency: int, rpm: float | None, seconds: float
) -> None:
    estimate = estimate_run(
        [PLAIN],
        FakeFactory(),
        RunConfig(repeats=8, max_concurrency=concurrency),
        Pricing.free(),
        seconds_per_attempt=5,
        parallel_requests=parallel,
        requests_per_minute=rpm,
    )
    assert estimate.seconds == seconds


@pytest.mark.parametrize(("parallel", "rpm"), [(0, None), (1, 0.0), (1, -5.0)])
def test_nonsense_speed_settings_are_rejected(parallel: int, rpm: float | None) -> None:
    with pytest.raises(ValueError, match="must be"):
        estimate_run(
            [PLAIN],
            FakeFactory(),
            RunConfig(),
            Pricing.free(),
            parallel_requests=parallel,
            requests_per_minute=rpm,
        )


def test_measured_time_averages_the_attempts_that_got_a_response() -> None:
    # The saved report has a list where the live description has a tuple.
    report = report_with([1000, 3000, None], {"phrases": ["a", "b"], "temperature": 0.7})
    assert measured_seconds_per_attempt(report, FakeFactory()) == 2.0


@pytest.mark.parametrize(
    ("latencies", "target"),
    [
        ([1000.0], {"temperature": 0.0, "phrases": ["a", "b"]}),  # different settings
        ([None], {"temperature": 0.7, "phrases": ["a", "b"]}),  # nothing was measured
    ],
)
def test_no_measured_time_when_it_would_mislead(
    latencies: list[float | None], target: dict[str, Any]
) -> None:
    assert measured_seconds_per_attempt(report_with(latencies, target), FakeFactory()) is None
