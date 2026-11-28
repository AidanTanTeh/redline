"""The judge inside real factories, against a fake server."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from redline.benign import load_tasks
from redline.core import Attack, AttackCategory, Outcome
from redline.demo import VulnBotFactory
from redline.detectors import JudgeConfig
from redline.runner import (
    FunctionFactory,
    HttpFactory,
    RunConfig,
    TrialFactory,
    load_http_factory,
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


async def test_a_capped_run_with_a_judge_is_refused_for_now() -> None:
    for factory in factories(JUDGE):
        with pytest.raises(ValueError, match="with a judge isn't supported yet"):
            await run_attacks([LEAK], factory, RunConfig(max_requests=100), cases_hash="c")


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
