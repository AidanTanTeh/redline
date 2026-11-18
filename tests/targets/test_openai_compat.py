import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from redline.targets import OpenAICompatConfig, OpenAICompatTarget, Target

Handler = Callable[[httpx.Request], httpx.Response]


def reply(
    message: dict[str, Any],
    model: str | None = "server-model",
    finish_reason: str | None = None,
) -> httpx.Response:
    """A fake /chat/completions reply wrapping one assistant message."""
    choice: dict[str, Any] = {"message": message}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    data: dict[str, Any] = {"choices": [choice]}
    if model is not None:
        data["model"] = model
    return httpx.Response(200, json=data)


def make_target(handler: Handler, **config: Any) -> OpenAICompatTarget:
    cfg = OpenAICompatConfig(base_url="http://test/v1/", model="m", **config)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatTarget(cfg, client)


def recording_handler(message: dict[str, Any]) -> tuple[Handler, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return reply(message)

    return handler, seen


# --- request building -------------------------------------------------------


async def test_posts_to_chat_completions_with_expected_body() -> None:
    handler, seen = recording_handler({"content": "hi"})
    target = make_target(handler, system_prompt="be nice", temperature=0.3)

    await target.send("hello")

    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == "http://test/v1/chat/completions"  # no double slash
    body = json.loads(request.content)
    assert body["model"] == "m"
    assert body["temperature"] == 0.3
    assert body["messages"] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello"},
    ]


async def test_no_system_message_when_not_configured() -> None:
    handler, seen = recording_handler({"content": "hi"})
    await make_target(handler).send("hello")
    assert json.loads(seen[0].content)["messages"] == [{"role": "user", "content": "hello"}]


async def test_seed_is_sent_only_when_set() -> None:
    handler, seen = recording_handler({"content": "hi"})
    await make_target(handler).send("a")
    await make_target(handler, seed=42).send("b")
    assert "seed" not in json.loads(seen[0].content)
    assert json.loads(seen[1].content)["seed"] == 42


async def test_extra_body_is_merged_and_wins_on_clash() -> None:
    handler, seen = recording_handler({"content": "hi"})
    target = make_target(handler, extra_body={"think": False, "temperature": 0.9})
    await target.send("a")
    body = json.loads(seen[0].content)
    assert body["think"] is False
    assert body["temperature"] == 0.9


# --- auth -------------------------------------------------------------------


async def test_no_auth_header_without_api_key_env() -> None:
    handler, seen = recording_handler({"content": "hi"})
    await make_target(handler).send("a")
    assert "authorization" not in seen[0].headers


async def test_auth_header_comes_from_env_and_value_never_enters_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REDLINE_TEST_KEY", "fake-value-123")
    handler, seen = recording_handler({"content": "hi"})
    cfg = OpenAICompatConfig(base_url="http://test/v1", model="m", api_key_env="REDLINE_TEST_KEY")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    target = OpenAICompatTarget(cfg, client)

    response = await target.send("a")

    assert seen[0].headers["authorization"] == "Bearer fake-value-123"
    # Only the variable NAME is stored anywhere we might print or save.
    assert "fake-value-123" not in cfg.model_dump_json()
    assert "fake-value-123" not in response.model_dump_json()


@pytest.mark.parametrize("value", [None, ""])
async def test_missing_or_empty_env_var_raises_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("REDLINE_TEST_KEY", raising=False)
    else:
        monkeypatch.setenv("REDLINE_TEST_KEY", value)
    handler, seen = recording_handler({"content": "hi"})
    target = make_target(handler, api_key_env="REDLINE_TEST_KEY")

    with pytest.raises(RuntimeError, match="REDLINE_TEST_KEY"):
        await target.send("a")
    assert seen == []  # failed before any request went out


# --- response parsing -------------------------------------------------------


async def test_plain_response() -> None:
    handler, _ = recording_handler({"content": "Hello!"})
    response = await make_target(handler).send("hi")
    assert response.text == "Hello!"
    assert response.reasoning is None
    assert response.model == "server-model"
    assert response.latency_ms >= 0
    assert response.finish_reason is None  # the fake server didn't send one
    assert response.tool_calls == ()


@pytest.mark.parametrize("reason", ["stop", "length"])
async def test_finish_reason_is_recorded(reason: str) -> None:
    target = make_target(lambda _: reply({"content": "hi"}, finish_reason=reason))
    assert (await target.send("a")).finish_reason == reason


async def test_cut_off_thinking_is_distinguishable_from_a_real_empty_answer() -> None:
    # Thinking used up the token cap: no answer yet, but finish_reason says why.
    message = {"content": "", "reasoning": "hmm, let me think..."}
    target = make_target(lambda _: reply(message, finish_reason="length"))
    response = await target.send("a")
    assert response.text == ""
    assert response.finish_reason == "length"


async def test_model_falls_back_to_configured_when_server_omits_it() -> None:
    target = make_target(lambda _: reply({"content": "hi"}, model=None))
    assert (await target.send("a")).model == "m"


async def test_think_tags_are_moved_into_reasoning() -> None:
    handler, _ = recording_handler({"content": "<think>plan\nstep 2</think>\n\nThe answer."})
    response = await make_target(handler).send("a")
    assert response.text == "The answer."
    assert response.reasoning == "plan\nstep 2"


async def test_unclosed_think_tag_does_not_leak_into_text() -> None:
    handler, _ = recording_handler({"content": "Partial.<think>cut off mid-thou"})
    response = await make_target(handler).send("a")
    assert response.text == "Partial."
    assert response.reasoning == "cut off mid-thou"


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
async def test_separate_reasoning_field_is_picked_up(field: str) -> None:
    handler, _ = recording_handler({"content": "Final.", field: "thought about it"})
    response = await make_target(handler).send("a")
    assert response.text == "Final."
    assert response.reasoning == "thought about it"


async def test_null_content_with_tool_call() -> None:
    message = {
        "content": None,
        "tool_calls": [
            {"function": {"name": "send_email", "arguments": '{"to": "a@b.c", "body": "x"}'}}
        ],
    }
    handler, _ = recording_handler(message)
    response = await make_target(handler).send("a")
    assert response.text == ""
    assert response.tool_calls[0].name == "send_email"
    assert response.tool_calls[0].arguments == {"to": "a@b.c", "body": "x"}


async def test_malformed_tool_arguments_are_kept_raw_not_crashed() -> None:
    message = {"content": "", "tool_calls": [{"function": {"name": "f", "arguments": "{oops"}}]}
    handler, _ = recording_handler(message)
    response = await make_target(handler).send("a")
    assert response.tool_calls[0].arguments == {"_raw": "{oops"}


# --- errors -----------------------------------------------------------------


async def test_http_error_status_is_raised_not_swallowed() -> None:
    target = make_target(lambda _: httpx.Response(429, json={"error": "rate limited"}))
    with pytest.raises(httpx.HTTPStatusError):
        await target.send("a")


async def test_unexpected_response_shape_raises_value_error() -> None:
    target = make_target(lambda _: httpx.Response(200, json={"choices": []}))
    with pytest.raises(ValueError, match="Unexpected response shape"):
        await target.send("a")


# --- protocol ---------------------------------------------------------------


def test_adapter_exposes_name_and_satisfies_target() -> None:
    cfg = OpenAICompatConfig(base_url="http://test/v1", model="qwen3:8b")
    target: Target = OpenAICompatTarget(cfg)  # mypy checks this line, see below
    assert target.name == "openai-compat/qwen3:8b"
