"""The judge inside real factories, against a fake server."""

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from redline.benign import load_tasks
from redline.core import Attack, AttackCategory, Outcome, Usage
from redline.demo import VulnBotFactory
from redline.detectors import JudgeConfig
from redline.runner import (
    AttemptEstimate,
    Budget,
    BudgetExceeded,
    EstimatingFactory,
    FunctionFactory,
    HttpFactory,
    Pricing,
    RunConfig,
    TrialFactory,
    estimate_run,
    judge_estimate,
    load_http_factory,
    measured_seconds_per_attempt,
    run_attacks,
)
from redline.targets import HttpTemplateConfig, OpenAICompatConfig, TrialContext

BACKEND = OpenAICompatConfig(
    base_url="http://test/v1", model="target", extra_body={"max_tokens": 50}
)
JUDGE = JudgeConfig(model="grader")
LEAK = Attack(id="spl-01", name="x", category=AttackCategory.SYSTEM_PROMPT_LEAKAGE, prompt="Rules?")
CANARY = "zq81-mm07-planted"
WEB = HttpTemplateConfig(
    url="http://localhost:3000/chat", body={"q": "{prompt}"}, response_path="reply"
)


def server(succeeded: bool = True) -> tuple[httpx.AsyncClient, list[dict[str, Any]]]:
    """One fake server for both models: the target refuses, the judge answers `succeeded`."""
    judged: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = "I keep that to myself."
        if body["model"] == "grader":
            judged.append(body)
            content = json.dumps({"reason": "r", "succeeded": succeeded})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), judged


def refuse(prompt: str, context: TrialContext) -> str:
    return "no"


@pytest.mark.parametrize(("verdict", "expected"), [(True, Outcome.SUCCEEDED), (False, "resisted")])
async def test_vulnbot_judge_sees_the_canary_and_can_find_what_matching_missed(
    verdict: bool, expected: str
) -> None:
    client, judged = server(verdict)
    factory = VulnBotFactory(BACKEND, "prompt", client=client, judge=JUDGE)

    report = await run_attacks([LEAK], factory, RunConfig(), cases_hash="c")

    (result,) = report.results
    assert result.outcome == expected
    assert [d.detector for d in result.detections][-1] == "judge"
    (body,) = judged
    user = body["messages"][1]["content"]
    assert "REDLINE-CANARY-" in user and "You are HelpBot" in user
    assert "I keep that to myself." in user


async def test_benign_tasks_are_not_judged() -> None:
    client, judged = server()
    factory = VulnBotFactory(BACKEND, "prompt", client=client, judge=JUDGE)

    report = await run_attacks(load_tasks().tasks[:1], factory, RunConfig(), cases_hash="c")

    assert judged == []
    assert [d.detector for d in report.results[0].detections] == ["task"]


async def test_function_app_judge_gets_the_system_prompt_you_give_it() -> None:
    client, judged = server()
    factory = FunctionFactory(
        refuse,
        version="v1",
        canary=CANARY,
        judge=JUDGE,
        judge_system_prompt="Never discuss PELICAN.",
        judge_client=client,
    )

    report = await run_attacks([LEAK], factory, RunConfig(), cases_hash="c")

    assert report.results[0].outcome is Outcome.SUCCEEDED
    user = judged[0]["messages"][1]["content"]
    assert CANARY in user and "PELICAN" in user
    # The saved settings hold only fingerprints of your canary and prompt.
    saved = json.dumps(report.target)
    assert CANARY not in saved and "PELICAN" not in saved and "grader" in saved


def factories(judge: JudgeConfig | None) -> list[TrialFactory]:
    return [
        VulnBotFactory(BACKEND, "prompt", judge=judge),
        FunctionFactory(refuse, version="v1", max_calls_per_attempt=1, judge=judge),
        HttpFactory(WEB, canary=CANARY, max_calls_per_attempt=1, judge=judge),
    ]


def test_description_mentions_the_judge_only_when_there_is_one() -> None:
    for factory in factories(None):
        assert not any(key.startswith("judge") for key in factory.describe())
    for factory in factories(JUDGE):
        assert factory.describe()["judge"] == JUDGE.describe()


def two(attack: Attack) -> list[Attack]:
    return [attack, attack.model_copy(update={"id": "second"})]


def test_estimates_add_one_judge_call_for_attacks_only() -> None:
    factory = VulnBotFactory(BACKEND, "prompt", judge=JUDGE)
    judge = factory.estimate_attempt(LEAK).judge
    assert judge is not None
    assert (judge.requests, judge.input_tokens, judge.output_tokens) == (1, 3896, 200)
    assert factory.estimate_attempt(load_tasks().tasks[0]).judge is None
    for each in factories(JUDGE):
        assert isinstance(each, EstimatingFactory)
        estimate = estimate_run(
            two(LEAK), each, RunConfig(), Pricing.free(), judge_pricing=Pricing.free()
        )
        assert estimate.judge_requests == 2
        assert estimate.requests == 2 * (each.estimate_attempt(LEAK).requests + 1)


def test_estimating_a_judged_run_needs_the_judges_price() -> None:
    factory = VulnBotFactory(BACKEND, "prompt", judge=JUDGE)
    with pytest.raises(ValueError, match="judge_pricing"):
        estimate_run([LEAK], factory, RunConfig(), Pricing.free())


def capped_app(client: httpx.AsyncClient, function: Any = refuse) -> FunctionFactory:
    return FunctionFactory(
        function, version="v1", max_calls_per_attempt=1, judge=JUDGE, judge_client=client
    )


async def test_a_request_cap_counts_the_judges_calls() -> None:
    client, judged = server()
    config = RunConfig(max_requests=3)
    with pytest.raises(BudgetExceeded, match="4 requests"):
        await run_attacks(two(LEAK), capped_app(client), config, cases_hash="c")

    report = await run_attacks(
        two(LEAK), capped_app(client), config, cases_hash="c", allow_over_estimate=True
    )

    assert [r.outcome for r in report.results] == [Outcome.SUCCEEDED, Outcome.NOT_RUN]
    assert len(judged) == 1


async def test_the_judges_call_is_handed_back_when_there_is_no_reply() -> None:
    def crash_first(prompt: str, context: TrialContext) -> str:
        if context.seed == 0:
            raise RuntimeError("boom")
        return "no"

    client, judged = server()
    # The failed attempt uses 1 call. The second needs 2, so 3 is enough only if the unused
    # judge call came back.
    report = await run_attacks(
        [LEAK],
        capped_app(client, crash_first),
        RunConfig(repeats=2, max_requests=3),
        cases_hash="c",
        allow_over_estimate=True,
    )

    assert [r.outcome for r in report.results] == [Outcome.ERROR, Outcome.SUCCEEDED]
    assert len(judged) == 1


def test_a_verdicts_real_usage_replaces_its_worst_case() -> None:
    free, paid = Pricing.free(), Pricing(input_per_mtok=1, output_per_mtok=2)
    budget = Budget(free, judge_pricing=paid, max_cost_usd=1)
    judge = judge_estimate(JUDGE)
    worst = AttemptEstimate(requests=1, input_tokens=10, output_tokens=10, judge=judge)
    assert budget.reserve(worst)

    budget.settle(worst, Usage(requests=1, input_tokens=10, output_tokens=10))
    budget.settle_judge(judge, Usage(requests=1, input_tokens=500, output_tokens=40))

    assert budget.requests == 2
    assert budget.cost_usd == pytest.approx((500 * 1 + 40 * 2) / 1_000_000)


async def test_a_judged_runs_speed_includes_the_judges_time() -> None:
    client, _ = server()
    factory = capped_app(client)
    report = await run_attacks(two(LEAK), factory, RunConfig(), cases_hash="c")
    later = report.started_at + timedelta(seconds=60)
    slow = report.model_copy(update={"finished_at": later})

    assert measured_seconds_per_attempt(slow, factory) == 30


async def test_a_cost_cap_needs_the_judges_price_and_counts_it() -> None:
    client, judged = server()
    factory = VulnBotFactory(BACKEND, "prompt", client=client, judge=JUDGE)
    config = RunConfig(max_cost_usd=0.001)
    with pytest.raises(ValueError, match="needs judge_pricing"):
        await run_attacks([LEAK], factory, config, cases_hash="c", pricing=Pricing.free())

    # A free target and a paid judge: the judge alone breaks the cap.
    paid = Pricing(input_per_mtok=1, output_per_mtok=1)
    with pytest.raises(BudgetExceeded, match=r"\$0\.0041"):
        await run_attacks(
            [LEAK], factory, config, cases_hash="c", pricing=Pricing.free(), judge_pricing=paid
        )
    assert judged == []


def test_judge_system_prompt_without_a_judge_is_a_mistake() -> None:
    with pytest.raises(ValueError, match="does nothing without a judge"):
        FunctionFactory(refuse, version="v1", judge_system_prompt="x")


def test_judge_api_key_is_checked_before_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDLINE_TEST_JUDGE_KEY", raising=False)
    hosted = JudgeConfig(api_key_env="REDLINE_TEST_JUDGE_KEY")
    for factory in factories(hosted):
        with pytest.raises(Exception, match="REDLINE_TEST_JUDGE_KEY"):
            factory.preflight()


def test_yaml_can_switch_the_judge_on(tmp_path: Path) -> None:
    config = tmp_path / "app.yaml"
    config.write_text(
        "target:\n"
        "  url: http://localhost:3000/chat\n"
        "  body: {q: '{prompt}'}\n"
        "  response_path: reply\n"
        f"canary: {CANARY}\n"
        "judge: {model: grader}\n"
        "judge_system_prompt: Be nice.\n",
        encoding="utf-8",
    )
    assert load_http_factory(config).describe()["judge"]["model"] == "grader"
