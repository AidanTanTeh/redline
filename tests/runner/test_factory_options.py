"""The optional things a factory can tell the runner about itself."""

from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from redline.core import Attack, AttackCategory, Case, Outcome, TargetResponse, ToolCallMatch
from redline.detectors import RegexDetector
from redline.runner import (
    AttemptEstimate,
    Pricing,
    RunConfig,
    Trial,
    estimate_run,
    is_rate_limit,
    never_sent,
    run_attacks,
)

PLAIN = Attack(id="d-01", name="x", category=AttackCategory.DIRECT_INJECTION, prompt="Hi")


def tool_attack(attack_id: str, tool: str) -> Attack:
    return Attack(
        id=attack_id,
        name="x",
        category=AttackCategory.EXCESSIVE_AGENCY,
        prompt=attack_id,
        success_tool_call=ToolCallMatch(tool=tool),
    )


def status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://localhost/a")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("bad", request=request, response=response)


class Factory:
    name = "fake"
    model = "m"
    has_tools = True

    def __init__(self, failures: int = 0, error: Exception | None = None) -> None:
        self.failures = failures
        self.error = error or httpx.ConnectError("down")
        self.sent: list[str] = []

    def describe(self) -> Mapping[str, Any]:
        return {"target": self.name}

    def preflight(self) -> None:
        pass

    def make_trial(self, attack: Case, seed: int) -> Trial:
        return Trial(target=self, message=attack.prompt, detectors=(RegexDetector("l", ["X"]),))

    async def send(self, prompt: str) -> TargetResponse:
        self.sent.append(prompt)
        if self.failures:
            self.failures -= 1
            raise self.error
        return TargetResponse(text="no", model="m", latency_ms=1)


class RequestsOnly(Factory):
    def estimate_attempt(self, case: Case) -> AttemptEstimate:
        return AttemptEstimate(requests=3, input_tokens=0, output_tokens=0, tokens_known=False)


async def no_sleep(seconds: float) -> None:
    return None


async def test_cases_the_factory_does_not_support_are_skipped_and_cost_nothing() -> None:
    class OnlySearch(RequestsOnly):
        def supports(self, case: Case) -> bool:
            return case.id != "ea-email"

    cases = [PLAIN, tool_attack("ea-email", "send_email")]
    factory = OnlySearch()

    report = await run_attacks(cases, factory, RunConfig(max_requests=3), cases_hash="c")

    assert [r.outcome for r in report.results] == [Outcome.RESISTED, Outcome.SKIPPED]
    assert factory.sent == ["Hi"]
    assert estimate_run(cases, factory, RunConfig(), Pricing.free()).requests == 3


@pytest.mark.parametrize(("safe", "sends"), [(False, 1), (True, 2)])
async def test_an_unsafe_factory_is_never_retried(safe: bool, sends: int) -> None:
    factory = Factory(failures=1)
    factory.safe_to_retry = safe  # type: ignore[attr-defined]

    report = await run_attacks([PLAIN], factory, RunConfig(), cases_hash="c", sleep=no_sleep)

    assert len(factory.sent) == sends
    assert report.results[0].outcome is (Outcome.RESISTED if safe else Outcome.ERROR)


async def test_a_factory_that_says_nothing_is_retried_as_before() -> None:
    factory = Factory(failures=1)
    await run_attacks([PLAIN], factory, RunConfig(), cases_hash="c", sleep=no_sleep)
    assert len(factory.sent) == 2


class Judging(Factory):
    safe_to_retry = False

    def can_retry(self, error: Exception) -> bool:
        return never_sent(error)


@pytest.mark.parametrize(
    ("error", "sends"),
    [
        (httpx.ConnectError("down"), 2),
        (httpx.ConnectTimeout("slow"), 2),
        (httpx.PoolTimeout("busy"), 2),
        (httpx.ReadTimeout("sent, no answer"), 1),
        (httpx.ReadError("cut"), 1),
        (httpx.WriteError("half sent"), 1),
        (TimeoutError(), 1),
        (status_error(429), 1),
        (status_error(500), 1),
    ],
)
async def test_an_unsafe_factory_can_allow_just_the_harmless_failures(
    error: Exception, sends: int
) -> None:
    factory = Judging(failures=1, error=error)
    await run_attacks([PLAIN], factory, RunConfig(), cases_hash="c", sleep=no_sleep)
    assert len(factory.sent) == sends


async def test_allowing_an_error_does_not_make_a_permanent_one_retryable() -> None:
    class Anything(Judging):
        def can_retry(self, error: Exception) -> bool:
            return True

    factory = Anything(failures=1, error=status_error(401))
    await run_attacks([PLAIN], factory, RunConfig(), cases_hash="c", sleep=no_sleep)
    assert len(factory.sent) == 1


@pytest.mark.parametrize(("status", "limited"), [(429, True), (500, False)])
def test_only_429_is_a_rate_limit(status: int, limited: bool) -> None:
    assert is_rate_limit(status_error(status)) is limited
    assert not is_rate_limit(httpx.ConnectError("down"))


async def test_a_cost_cap_is_refused_when_tokens_are_unknown() -> None:
    factory = RequestsOnly()
    config = RunConfig(max_cost_usd=1)
    with pytest.raises(ValueError, match="token use is unknown"):
        await run_attacks([PLAIN], factory, config, cases_hash="c", pricing=Pricing.free())
    assert factory.sent == []


def test_an_estimate_says_when_tokens_are_unknown() -> None:
    estimate = estimate_run([PLAIN], RequestsOnly(), RunConfig(), Pricing.free())
    assert (estimate.requests, estimate.tokens_known) == (3, False)
