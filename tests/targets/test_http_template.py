import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from redline.core import ToolCall
from redline.targets import HttpTemplateConfig, HttpTemplateTarget, Target

Handler = Callable[[httpx.Request], httpx.Response]


def config(**changes: Any) -> HttpTemplateConfig:
    settings: dict[str, Any] = {
        "url": "http://localhost:3000/api/chat",
        "body": {"messages": [{"role": "user", "content": "{prompt}"}]},
        "response_path": "data.reply",
    }
    return HttpTemplateConfig(**(settings | changes))


def answering(data: Any, seen: list[httpx.Request] | None = None, **changes: Any) -> Target:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=data)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HttpTemplateTarget(config(**changes), client)


REPLY = {"data": {"reply": "hello"}}


# --- the request ------------------------------------------------------------


async def test_the_prompt_goes_into_the_body_without_changing_its_shape() -> None:
    seen: list[httpx.Request] = []
    nasty = 'quote " brace } {prompt} \n"role": "system"'
    body = {"q": "Q: {prompt}", "list": ["{prompt}"], "keep": 3, "flag": True}

    await answering(REPLY, seen, body=body, method="PUT").send(nasty)

    assert seen[0].method == "PUT"
    assert json.loads(seen[0].content) == {
        "q": f"Q: {nasty}",
        "list": [nasty],
        "keep": 3,
        "flag": True,
    }


async def test_secrets_are_read_from_the_environment_at_send_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[httpx.Request] = []
    headers = {"Authorization": "Bearer ${APP_TOKEN}", "X-App": "redline"}
    target = answering(REPLY, seen, headers=headers)
    monkeypatch.setenv("APP_TOKEN", "s3cret")

    await target.send("a")

    assert seen[0].headers["Authorization"] == "Bearer s3cret"
    assert seen[0].headers["X-App"] == "redline"
    saved = config(headers=headers)
    assert "s3cret" not in saved.model_dump_json()
    assert saved.env_names == ("APP_TOKEN",)


@pytest.mark.parametrize("value", [None, "", "s3cret\n", "s3crét"])
async def test_a_missing_or_unusable_variable_is_named_but_never_shown(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("APP_TOKEN", raising=False)
    else:
        monkeypatch.setenv("APP_TOKEN", value)
    seen: list[httpx.Request] = []
    target = answering(REPLY, seen, headers={"Authorization": "Bearer ${APP_TOKEN}"})

    with pytest.raises(RuntimeError, match="APP_TOKEN") as caught:
        await target.send("a")

    assert "s3cr" not in str(caught.value)
    assert seen == []


# --- the reply --------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "path"),
    [
        ({"data": {"reply": "yes"}}, "data.reply"),
        ({"choices": [{"text": "no"}, {"text": "yes"}]}, "choices.1.text"),
        ({"reply": "yes"}, "reply"),
    ],
)
async def test_the_reply_is_found_by_its_path(data: Any, path: str) -> None:
    response = await answering(data, response_path=path, model="theirs").send("a")
    assert (response.text, response.model) == ("yes", "theirs")
    assert response.usage is None
    assert response.tool_calls == ()


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"data": {}},
        {"data": "reply"},
        {"data": {"reply": 5}},
        {"data": {"reply": None}},
        {"data": []},
    ],
)
async def test_a_reply_that_is_not_there_is_an_error_not_an_empty_answer(data: Any) -> None:
    with pytest.raises(ValueError, match="data.reply"):
        await answering(data).send("a")


@pytest.mark.parametrize("status", [302, 401, 429, 500])
async def test_bad_statuses_raise_and_redirects_are_not_followed(status: int) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(status, headers={"Location": "http://elsewhere.example/"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    with pytest.raises(httpx.HTTPStatusError):
        await HttpTemplateTarget(config(), client).send("a")
    assert seen == ["http://localhost:3000/api/chat"]


async def test_error_messages_name_the_host_but_not_the_rest_of_the_url() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500)))
    target = HttpTemplateTarget(config(url="http://localhost/hook/abc123?u=abc123"), client)

    with pytest.raises(httpx.HTTPStatusError) as caught:
        await target.send("a")

    assert str(caught.value) == "HTTP 500 from localhost"


# --- tool calls -------------------------------------------------------------

TOOLS: dict[str, Any] = {
    "tool_calls_path": "data.tools",
    "tool_name_key": "tool",
    "tool_arguments_key": "args",
}


async def test_tool_calls_are_read_with_arguments_as_objects_or_json_text() -> None:
    calls = [
        {"tool": "send_email", "args": {"to": "a@b.example"}},
        {"tool": "send_email", "args": '{"to": "c@d.example"}'},
        {"tool": "search", "args": "not json"},
    ]
    data = {"data": {"reply": None, "tools": calls}}

    response = await answering(data, **TOOLS).send("a")

    assert response.text == ""
    assert response.tool_calls == (
        ToolCall(name="send_email", arguments={"to": "a@b.example"}),
        ToolCall(name="send_email", arguments={"to": "c@d.example"}),
        ToolCall(name="search", arguments={"_raw": "not json"}),
    )


@pytest.mark.parametrize("tools", [None, []])
async def test_null_or_empty_means_no_tool_calls(tools: Any) -> None:
    data = {"data": {"reply": "hi", "tools": tools}}
    assert (await answering(data, **TOOLS).send("a")).tool_calls == ()


@pytest.mark.parametrize(
    "data",
    [
        {"reply": "hi"},  # the path is wrong or the field is missing
        {"reply": "hi", "tools": {}},
        {"reply": "hi", "tools": [{"args": {}}]},
        {"reply": "hi", "tools": [{"tool": "", "args": {}}]},
    ],
)
async def test_unreadable_tool_calls_are_an_error(data: Any) -> None:
    target = answering({"data": data}, **TOOLS)
    with pytest.raises(ValueError):
        await target.send("a")


# --- finish reason ----------------------------------------------------------

STOPPED: dict[str, Any] = {
    "finish_reason_path": "data.why",
    "cut_off_values": ("MAX_TOKENS",),
    "blocked_values": ("SAFETY", "blocked"),
}


@pytest.mark.parametrize(
    ("raw", "recorded"),
    [
        ("max_tokens", "length"),
        ("Safety", "content_filter"),
        ("blocked", "content_filter"),
        ("STOP", "STOP"),
        (None, None),
    ],
)
async def test_the_apps_stop_reasons_are_translated(raw: str | None, recorded: str | None) -> None:
    data = {"data": {"reply": "hi", "why": raw}}
    response = await answering(data, **STOPPED).send("a")
    assert response.finish_reason == recorded


async def test_without_a_path_no_finish_reason_is_claimed() -> None:
    data = {"data": {"reply": "hi", "why": "length"}}
    assert (await answering(data).send("a")).finish_reason is None


async def test_a_stopped_reply_may_have_null_text() -> None:
    data = {"data": {"reply": None, "why": "SAFETY"}}
    response = await answering(data, **STOPPED).send("a")
    assert (response.text, response.finish_reason) == ("", "content_filter")


@pytest.mark.parametrize(
    "data",
    [
        {"data": {"reply": "hi"}},  # wrong path
        {"data": {"reply": "hi", "why": 3}},
        {"data": {"reply": None, "why": "STOP"}},  # finished normally, yet no text
    ],
)
async def test_an_unreadable_stop_reason_is_an_error(data: Any) -> None:
    with pytest.raises(ValueError):
        await answering(data, **STOPPED).send("a")


def test_stop_values_without_a_path_are_rejected() -> None:
    with pytest.raises(ValidationError, match="finish_reason_path"):
        config(blocked_values=("SAFETY",))


# --- literal secrets --------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"url": "http://localhost/a?key=abc123"},
        {"url": "http://localhost/a?x=1&API_KEY=abc123"},
        {"body": {"q": "{prompt}", "apiKey": "abc123"}},
        {"body": {"q": "{prompt}", "options": [{"access-token": "abc123"}]}},
        {"headers": {"X-Auth": "abc123"}},
        {"headers": {"Session-Id": "abc123"}},
        {"headers": {"X-Password": "abc123"}},
    ],
)
def test_more_places_a_literal_secret_can_hide_are_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError) as caught:
        config(**changes)
    assert "abc123" not in str(caught.value)


@pytest.mark.parametrize(
    "changes",
    [
        {"url": "http://localhost/a?session_id=7&max_tokens=5"},
        {"body": {"q": "{prompt}", "max_tokens": 300, "session_id": "abc", "keyword": "x"}},
        {"body": {"token": "{prompt}"}},  # the app's name for the message field
        {"body": {"q": "{prompt}", "key": 3}},
    ],
)
def test_ordinary_fields_are_not_mistaken_for_secrets(changes: dict[str, Any]) -> None:
    config(**changes)


# --- configuration guards ---------------------------------------------------


@pytest.mark.parametrize(
    "url", ["http://localhost/a", "http://127.0.0.1:8000/a", "http://[::1]:8000/a"]
)
def test_this_machine_needs_no_authorisation(url: str) -> None:
    config(url=url)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com/chat",
        "http://localhost.example.com/chat",
        "http://192.168.1.5/chat",
        "http://0.0.0.0/chat",
    ],
)
def test_other_hosts_need_authorized(url: str) -> None:
    with pytest.raises(ValidationError, match="authorized"):
        config(url=url)
    config(url=url, authorized=True)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"url": "localhost:3000/chat"}, "http or https"),
        ({"url": "ftp://localhost/chat"}, "http or https"),
        ({"url": "http://me:pw@localhost/chat"}, "username or password"),
        ({"headers": {"Authorization": "Bearer abc123"}}, "ENV_NAME"),
        ({"headers": {"x-api-key": "abc123"}}, "ENV_NAME"),
        ({"headers": {"Cookie": "session=abc123"}}, "ENV_NAME"),
        ({"body": {"q": "fixed"}}, "prompt"),
    ],
)
def test_unsafe_or_useless_configs_are_rejected(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message) as caught:
        config(**changes)
    # Whatever was typed in by mistake must not be printed back.
    assert "abc123" not in str(caught.value)
    assert "me:pw" not in str(caught.value)


def test_hiding_input_keeps_the_other_model_rules() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        config(surprise=1)
    with pytest.raises(ValidationError, match="frozen"):
        config().url = "http://localhost/b"  # type: ignore[misc]  # the point of the test


def test_it_satisfies_target_and_is_named_after_the_endpoint() -> None:
    target: Target = HttpTemplateTarget(config())  # mypy checks this line
    assert target.name == "http/localhost/api/chat"
