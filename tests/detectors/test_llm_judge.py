import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from redline.core import Attack, AttackCategory, DetectionResult, TargetResponse, Usage
from redline.detectors import Detector, JudgeConfig, JudgeContext, LlmJudge

ATTACK = Attack(id="d-01", name="x", category=AttackCategory.DIRECT_INJECTION, prompt="Hi")
USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
PLANTED = "s3cret"  # a fake planted secret


def reply(text: str = "No.", reasoning: str | None = None) -> TargetResponse:
    return TargetResponse(text=text, reasoning=reasoning, model="m", latency_ms=1)


def fake_judge(
    content: str, finish_reason: str = "stop", config: JudgeConfig | None = None
) -> tuple[LlmJudge, list[dict[str, Any]]]:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        choice = {"message": {"content": content}, "finish_reason": finish_reason}
        return httpx.Response(200, json={"choices": [choice], "usage": USAGE})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    judge = LlmJudge(config or JudgeConfig(), JudgeContext(secret=PLANTED), client=client)
    return judge, bodies


async def verdict_of(judge: LlmJudge, response: TargetResponse | None = None) -> DetectionResult:
    (result,) = await judge.detect(ATTACK, response or reply())
    return result


@pytest.mark.parametrize("succeeded", [True, False])
async def test_returns_the_judges_verdict_reason_and_usage(succeeded: bool) -> None:
    judge, _ = fake_judge(json.dumps({"reason": "because", "succeeded": succeeded}))
    detector: Detector = judge

    (result,) = await detector.detect(ATTACK, reply())

    assert (result.detector, result.triggered, result.evidence) == ("judge", succeeded, "because")
    assert result.error is None
    assert result.usage == Usage(requests=1, input_tokens=100, output_tokens=20)


async def test_sends_fixed_settings_and_never_the_targets_thinking() -> None:
    # A provider option must not be able to remove the cap or the reply shape.
    config = JudgeConfig(extra_body={"reasoning_effort": "none", "max_tokens": 9999})
    judge, (bodies) = fake_judge('{"reason": "x", "succeeded": false}', config=config)

    await judge.detect(ATTACK, reply("visible", reasoning="HIDDEN THOUGHT"))

    (body,) = bodies
    assert (body["model"], body["temperature"], body["seed"]) == ("gemma4:12b-it-qat", 0, 0)
    assert (body["max_tokens"], body["reasoning_effort"]) == (200, "none")
    assert body["response_format"]["type"] == "json_schema"
    sent = json.dumps(body["messages"])
    assert "visible" in sent and PLANTED in sent and "HIDDEN THOUGHT" not in sent


@pytest.mark.parametrize(
    ("content", "finish_reason", "expected"),
    [
        ("I think it worked", "stop", "unreadable verdict"),
        ('{"reason": "The', "length", "cut off at the token cap"),
    ],
)
async def test_unusable_verdict_is_a_failed_detection_that_keeps_usage(
    content: str, finish_reason: str, expected: str
) -> None:
    judge, _ = fake_judge(content, finish_reason)
    result = await verdict_of(judge)

    assert (result.triggered, result.confidence) == (False, 0)
    assert result.error is not None and expected in result.error
    assert result.usage is not None


@pytest.mark.parametrize("text", ["x" * 5000, "密" * 1300])  # the second is 3900 bytes
async def test_too_long_to_fit_is_refused_without_calling_the_model(text: str) -> None:
    judge, bodies = fake_judge('{"reason": "x", "succeeded": false}')
    result = await verdict_of(judge, reply(text))

    assert result.error is not None and "too long" in result.error
    assert bodies == [] and result.usage is None


async def test_http_errors_are_raised_for_the_runner_to_record() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    judge = LlmJudge(JudgeConfig(), client=client)
    with pytest.raises(httpx.HTTPStatusError):
        await judge.detect(ATTACK, reply())


def test_description_covers_what_changes_a_verdict_and_nothing_secret() -> None:
    base = JudgeConfig(model_digest="abc").describe()
    json.dumps(base)  # must be hashable
    assert base["prompt"] and base["model_digest"] == "abc" and base["temperature"] == 0
    assert "api_key_env" not in base and "timeout_s" not in base
    assert JudgeConfig(model_digest="abc", timeout_s=5, api_key_env="KEY").describe() == base
    for changed in (
        JudgeConfig(model_digest="abc", model="other"),
        JudgeConfig(model_digest="abc", seed=1),
        JudgeConfig(model_digest="abc", extra_body={}),
        JudgeConfig(model_digest="abc", max_output_tokens=300),
    ):
        assert changed.describe() != base


def test_context_must_leave_room_for_input() -> None:
    with pytest.raises(ValidationError, match="too small"):
        JudgeConfig(context_tokens=200, max_output_tokens=200)
