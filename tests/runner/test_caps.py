"""The runner with a request or cost cap."""

from collections.abc import Mapping
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from redline.core import (
    Attack,
    AttackCategory,
    Case,
    Outcome,
    RunResult,
    TargetResponse,
    Usage,
)
from redline.detectors import RegexDetector
from redline.metrics import summarize
from redline.runner import (
    AttemptEstimate,
    BudgetExceeded,
    Pricing,
    RunConfig,
    Trial,
    config_hash,
    run_attacks,
)

PRICING = Pricing(input_per_mtok=1, output_per_mtok=2)


def attack(attack_id: str) -> Attack:
    return Attack(
        id=attack_id, name="A", category=AttackCategory.DIRECT_INJECTION, prompt=attack_id
    )


A, B, C = attack("a"), attack("b"), attack("c")


class CappableFactory:
    """Each attempt could take 2 requests and really takes 1. `failures` are raised first."""

    name = "fake"
    model = "m"
    has_tools = False

    def __init__(self, failures: list[Exception] | None = None) -> None:
        self.failures = failures or []
        self.sent: list[str] = []

    def describe(self) -> Mapping[str, Any]:
        return {"target": self.name}

    def preflight(self) -> None:
        pass

    def estimate_attempt(self, case: Case) -> AttemptEstimate:
        return AttemptEstimate(requests=2, input_tokens=1000, output_tokens=500)

    def make_trial(self, attack: Case, seed: int) -> Trial:
        return Trial(target=self, message=attack.prompt, detectors=(RegexDetector("l", ["X"]),))

    async def send(self, prompt: str) -> TargetResponse:
        self.sent.append(prompt)
        if self.failures:
            raise self.failures.pop(0)
        usage = Usage(requests=1, input_tokens=100, output_tokens=50)
        return TargetResponse(text="no", model="m", latency_ms=1, usage=usage)


async def no_sleep(seconds: float) -> None:
    return None


def outcomes(results: tuple[RunResult, ...]) -> list[tuple[str, int, Outcome]]:
    return [(r.attack_id, r.attempt, r.outcome) for r in results]


async def test_stops_at_the_request_cap_and_marks_the_rest_not_run() -> None:
    factory = CappableFactory()
    # Room for: 1 spent + 1 spent + 2 reserved = 4. A fourth try would need 5.
    report = await run_attacks(
        [A, B],
        factory,
        RunConfig(repeats=2, max_requests=4),
        cases_hash="c",
        allow_over_estimate=True,
    )

    # First attempts of every attack go first, so the partial run covers both.
    assert factory.sent == ["a", "b", "a"]
    # Results still come out in attack order, then attempt order.
    assert outcomes(report.results) == [
        ("a", 0, Outcome.RESISTED),
        ("a", 1, Outcome.RESISTED),
        ("b", 0, Outcome.RESISTED),
        ("b", 1, Outcome.NOT_RUN),
    ]


async def test_not_run_is_left_out_of_the_rate_but_widens_the_bounds() -> None:
    report = await run_attacks(
        [A, B],
        CappableFactory(),
        RunConfig(max_requests=2),
        cases_hash="c",
        allow_over_estimate=True,
    )
    overall = summarize(report.results, [A, B]).overall

    assert (overall.decided, overall.attempts) == (1, 2)
    assert overall.asr == 0
    assert (overall.lowest, overall.highest) == (0, 0.5)


async def test_stops_at_the_cost_cap() -> None:
    factory = CappableFactory()
    # Worst case per try is $0.002; each really costs $0.0002.
    report = await run_attacks(
        [A, B, C],
        factory,
        RunConfig(max_cost_usd=0.0022),
        cases_hash="c",
        pricing=PRICING,
        allow_over_estimate=True,
    )
    assert factory.sent == ["a", "b"]
    assert report.results[2].outcome is Outcome.NOT_RUN


@pytest.mark.parametrize(
    ("config", "message"),
    [(RunConfig(max_requests=3), "4 requests"), (RunConfig(max_cost_usd=0.001), r"\$0.0040")],
)
async def test_refuses_up_front_when_the_worst_case_is_over(
    config: RunConfig, message: str
) -> None:
    factory = CappableFactory()
    with pytest.raises(BudgetExceeded, match=message):
        await run_attacks([A, B], factory, config, cases_hash="c", pricing=PRICING)
    assert factory.sent == []


async def test_a_worst_case_inside_the_caps_runs_without_the_override() -> None:
    config = RunConfig(max_requests=4, max_cost_usd=0.004)
    report = await run_attacks([A, B], CappableFactory(), config, cases_hash="c", pricing=PRICING)
    assert {r.outcome for r in report.results} == {Outcome.RESISTED}


async def test_a_retry_that_does_not_fit_leaves_an_error_not_a_second_send() -> None:
    factory = CappableFactory([httpx.ConnectError("down")])
    # The failed try is charged its worst case (2), so the retry would need 4.
    report = await run_attacks(
        [A],
        factory,
        RunConfig(max_requests=3),
        cases_hash="c",
        sleep=no_sleep,
        allow_over_estimate=True,
    )
    (result,) = report.results
    assert factory.sent == ["a"]
    assert result.outcome is Outcome.ERROR
    assert result.error == "ConnectError: down (retry stopped by the cap)"
    assert result.retries == 0


async def test_a_retry_that_fits_is_sent() -> None:
    factory = CappableFactory([httpx.ConnectError("down")])
    report = await run_attacks(
        [A],
        factory,
        RunConfig(max_requests=4),
        cases_hash="c",
        sleep=no_sleep,
        allow_over_estimate=True,
    )
    (result,) = report.results
    assert (result.outcome, result.retries) == (Outcome.RESISTED, 1)


async def test_concurrent_attempts_cannot_overshoot_the_cap() -> None:
    factory = CappableFactory()
    report = await run_attacks(
        [A, B, C],
        factory,
        RunConfig(max_requests=4, max_concurrency=3),
        cases_hash="c",
        allow_over_estimate=True,
    )
    assert len(factory.sent) <= 4
    assert len(report.results) == 3


async def test_a_cost_cap_needs_pricing() -> None:
    with pytest.raises(ValueError, match="needs pricing"):
        await run_attacks([A], CappableFactory(), RunConfig(max_cost_usd=1), cases_hash="c")


async def test_a_cap_needs_a_factory_that_can_estimate() -> None:
    class Blind:
        """Everything a TrialFactory needs, but no estimate_attempt."""

        name = "blind"
        model = "m"
        has_tools = False

        def describe(self) -> Mapping[str, Any]:
            return {}

        def preflight(self) -> None:
            pass

        def make_trial(self, attack: Case, seed: int) -> Trial:
            raise AssertionError("nothing should be built")

    with pytest.raises(TypeError, match="estimate_attempt"):
        await run_attacks([A], Blind(), RunConfig(max_requests=10), cases_hash="c")


def test_caps_do_not_change_the_config_hash() -> None:
    plain = config_hash({}, "c", [A], RunConfig())
    capped = config_hash({}, "c", [A], RunConfig(max_requests=1, max_cost_usd=0))
    assert plain == capped


@pytest.mark.parametrize("bad", [{"max_requests": 0}, {"max_cost_usd": -1.0}])
def test_nonsense_caps_are_rejected(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RunConfig(**bad)
