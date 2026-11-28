import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from typing import Any

import httpx
import pytest

from redline.core import (
    Attack,
    AttackCategory,
    Case,
    DetectionResult,
    Outcome,
    RunResult,
    TargetResponse,
    ToolCallMatch,
)
from redline.detectors import Detector, RegexDetector
from redline.runner import RunConfig, RunReport, Trial, run_attacks

Send = Callable[[str], Awaitable[TargetResponse]]


def attack(attack_id: str = "a-1", **extra: Any) -> Attack:
    return Attack(
        id=attack_id,
        name="A",
        category=AttackCategory.DIRECT_INJECTION,
        prompt=attack_id,
        **extra,
    )


def reply(text: str = "no", finish_reason: str = "stop") -> TargetResponse:
    return TargetResponse(text=text, model="m", latency_ms=1, finish_reason=finish_reason)


async def refuse(prompt: str) -> TargetResponse:
    return reply()


class FakeTarget:
    name = "fake"

    def __init__(self, send: Send) -> None:
        self._send = send

    async def send(self, prompt: str) -> TargetResponse:
        return await self._send(prompt)


class FakeFactory:
    """Runs `send` as the target. A reply containing "SECRET" counts as a leak."""

    name = "fake"
    model = "m"

    def __init__(
        self,
        send: Send = refuse,
        *,
        has_tools: bool = False,
        delivered: bool = True,
        detectors: tuple[Detector, ...] = (RegexDetector("leak", ["SECRET"]),),
    ) -> None:
        self.send = send
        self.has_tools = has_tools
        self.delivered = delivered
        self.detectors = detectors
        self.seeds: list[int] = []
        self.preflighted = False

    def describe(self) -> Mapping[str, Any]:
        return {"target": self.name}

    def preflight(self) -> None:
        self.preflighted = True

    def make_trial(self, attack: Case, seed: int) -> Trial:
        self.seeds.append(seed)
        return Trial(
            target=FakeTarget(self.send),
            message=attack.prompt,
            detectors=self.detectors,
            delivered=lambda response: self.delivered,
        )


async def no_sleep(seconds: float) -> None:
    return None


async def run(
    factory: FakeFactory,
    *attacks: Attack,
    config: RunConfig | None = None,
    on_result: Callable[[RunResult], None] | None = None,
) -> RunReport:
    return await run_attacks(
        attacks or (attack(),),
        factory,
        config or RunConfig(),
        cases_hash="corpus",
        on_result=on_result,
        sleep=no_sleep,
    )


async def only_result(factory: FakeFactory, config: RunConfig | None = None) -> RunResult:
    (result,) = (await run(factory, config=config)).results
    return result


# --- outcomes ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "finish_reason", "delivered", "expected"),
    [
        ("no", "stop", True, Outcome.RESISTED),
        ("the SECRET", "stop", True, Outcome.SUCCEEDED),
        ("the SECRET", "length", False, Outcome.SUCCEEDED),  # a leak beats everything
        ("", "length", True, Outcome.NO_ANSWER),
        ("", "max_tool_rounds", False, Outcome.NO_ANSWER),  # beats not delivered
        ("", "content_filter", True, Outcome.NO_ANSWER),  # blocked by the provider
        ("no", "stop", False, Outcome.NOT_DELIVERED),
    ],
)
async def test_outcome_of_an_answered_attempt(
    text: str, finish_reason: str, delivered: bool, expected: Outcome
) -> None:
    async def send(prompt: str) -> TargetResponse:
        return reply(text, finish_reason)

    result = await only_result(FakeFactory(send, delivered=delivered))
    assert (result.outcome, result.delivered) == (expected, delivered)
    assert result.response is not None


async def test_tool_attack_is_skipped_on_a_target_without_tools() -> None:
    factory = FakeFactory(has_tools=False)
    report = await run(factory, attack(success_tool_call=ToolCallMatch(tool="send_email")))
    assert report.results[0].outcome is Outcome.SKIPPED
    assert factory.seeds == []  # the target was never even built


async def test_tool_attack_runs_on_a_target_with_tools() -> None:
    wanted = ToolCallMatch(tool="send_email")
    report = await run(FakeFactory(has_tools=True), attack(success_tool_call=wanted))
    assert report.results[0].outcome is Outcome.RESISTED


# --- errors and retries -----------------------------------------------------


async def test_one_failing_attack_does_not_stop_the_others() -> None:
    async def send(prompt: str) -> TargetResponse:
        if prompt == "bad":
            raise ValueError("Unexpected response shape")
        return reply()

    report = await run(FakeFactory(send), attack("a-1"), attack("bad"), attack("a-3"))
    assert [r.outcome for r in report.results] == [
        Outcome.RESISTED,
        Outcome.ERROR,
        Outcome.RESISTED,
    ]
    assert report.results[1].error == "ValueError: Unexpected response shape"


async def test_transient_failure_is_retried_and_counted() -> None:
    calls = 0

    async def send(prompt: str) -> TargetResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("slow")
        return reply()

    result = await only_result(FakeFactory(send))
    assert (result.outcome, result.retries) == (Outcome.RESISTED, 1)


async def test_slow_attempt_times_out_and_becomes_an_error() -> None:
    async def send(prompt: str) -> TargetResponse:
        await asyncio.sleep(10)
        return reply()

    config = RunConfig(attempt_timeout_s=0.01, max_retries=1)
    result = await only_result(FakeFactory(send), config)
    assert (result.outcome, result.retries) == (Outcome.ERROR, 1)
    assert result.error is not None and result.error.startswith("TimeoutError")


async def test_broken_detector_is_an_error_and_does_not_call_the_model_again() -> None:
    calls = 0

    async def send(prompt: str) -> TargetResponse:
        nonlocal calls
        calls += 1
        return reply()

    class Broken:
        name = "broken"

        async def detect(self, attack: Case, response: TargetResponse) -> Any:
            raise httpx.ReadTimeout("judge is slow")

    result = await only_result(FakeFactory(send, detectors=(Broken(),)))
    assert (result.outcome, calls) == (Outcome.ERROR, 1)
    # The reply is evidence we already paid for: keep it so it can be judged again later.
    assert result.response is not None
    assert result.error == "judging failed: detector broken: ReadTimeout: judge is slow"
    (failed,) = result.detections
    assert (failed.detector, failed.triggered, failed.confidence) == ("broken", False, 0)


class Fixed:
    """A detector with a set verdict, or one that always crashes (`triggered=None`)."""

    def __init__(self, name: str, triggered: bool | None) -> None:
        self.name = name
        self._triggered = triggered

    async def detect(self, attack: Case, response: TargetResponse) -> tuple[DetectionResult, ...]:
        if self._triggered is None:
            raise ValueError("boom")
        return (DetectionResult(detector=self.name, triggered=self._triggered),)


@pytest.mark.parametrize(("seen", "expected"), [(True, Outcome.SUCCEEDED), (False, Outcome.ERROR)])
async def test_a_crashed_detector_keeps_the_other_verdicts(seen: bool, expected: Outcome) -> None:
    # The crash comes first, to show the detector after it still runs.
    detectors = (Fixed("judge", None), Fixed("canary", seen))
    result = await only_result(FakeFactory(detectors=detectors))

    assert result.outcome is expected
    assert [(d.detector, d.triggered) for d in result.detections] == [
        ("judge", False),
        ("canary", seen),
    ]
    assert result.detections[0].error == "ValueError: boom"
    assert (result.error is None) == (expected is Outcome.SUCCEEDED)


@pytest.mark.parametrize(("seen", "expected"), [(True, Outcome.SUCCEEDED), (False, Outcome.ERROR)])
async def test_a_crashed_delivery_check_keeps_the_reply(seen: bool, expected: Outcome) -> None:
    def crash(response: TargetResponse) -> bool:
        raise ValueError("boom")

    class BrokenDelivery(FakeFactory):
        def make_trial(self, attack: Case, seed: int) -> Trial:
            return replace(super().make_trial(attack, seed), delivered=crash)

    result = await only_result(BrokenDelivery(detectors=(Fixed("canary", seen),)))

    assert result.outcome is expected
    assert result.response is not None and len(result.detections) == 1
    assert (result.delivered, result.delivery_error) == (None, "ValueError: boom")
    assert (result.error is None) == (expected is Outcome.SUCCEEDED)


async def test_trial_without_detectors_is_an_error_not_resisted() -> None:
    result = await only_result(FakeFactory(detectors=()))
    assert result.outcome is Outcome.ERROR


# --- the run as a whole -----------------------------------------------------


async def test_repeats_get_their_own_seed_and_results_keep_their_order() -> None:
    factory = FakeFactory()
    config = RunConfig(repeats=2, base_seed=10)
    report = await run(factory, attack("a-1"), attack("a-2"), config=config)
    assert [(r.attack_id, r.attempt, r.seed) for r in report.results] == [
        ("a-1", 0, 10),
        ("a-1", 1, 11),
        ("a-2", 0, 10),
        ("a-2", 1, 11),
    ]
    assert {r.config_hash for r in report.results} == {report.config_hash}
    assert report.target == {"target": "fake"}
    assert factory.preflighted


async def test_on_result_is_called_once_per_attempt() -> None:
    seen: list[RunResult] = []
    report = await run(FakeFactory(), config=RunConfig(repeats=3), on_result=seen.append)
    assert seen == list(report.results)


async def test_failing_preflight_stops_before_any_attempt() -> None:
    class NoKey(FakeFactory):
        def preflight(self) -> None:
            raise RuntimeError("Environment variable X is not set")

    factory = NoKey()
    with pytest.raises(RuntimeError, match="not set"):
        await run(factory)
    assert factory.seeds == []


async def test_duplicate_attack_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        await run(FakeFactory(), attack("a-1"), attack("a-1"))


@pytest.mark.parametrize("limit", [1, 3])
async def test_never_more_attempts_in_flight_than_the_limit(limit: int) -> None:
    in_flight = 0
    peak = 0

    async def send(prompt: str) -> TargetResponse:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return reply()

    config = RunConfig(repeats=8, max_concurrency=limit)
    await run(FakeFactory(send), config=config)
    assert peak == limit
