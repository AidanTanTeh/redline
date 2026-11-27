import httpx
import pytest
from pydantic import ValidationError

from redline.core import TargetResponse
from redline.targets import (
    HttpTemplateConfig,
    HttpTemplateTarget,
    OpenAICompatConfig,
    OpenAICompatTarget,
)
from redline.targets.secrets import read_secret, redact

# Fake values only. "abc123" is the part that must never show up in an error.
BAD = [
    "fake-abc123\n",
    "fake\nabc123",
    "fake-abc123\r",
    " fake-abc123",
    "fake-abc123 ",
    "fake\tabc123",
    "fake-abc123é",
]


def test_a_normal_secret_is_returned_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_KEY", "fake-abc123 with inner spaces=/+")
    assert read_secret("FAKE_KEY") == "fake-abc123 with inner spaces=/+"


@pytest.mark.parametrize("value", BAD)
async def test_a_malformed_api_key_is_never_sent_or_shown(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("FAKE_KEY", value)
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    config = OpenAICompatConfig(base_url="http://test/v1", model="m", api_key_env="FAKE_KEY")

    with pytest.raises(RuntimeError, match="FAKE_KEY") as caught:
        await OpenAICompatTarget(config, client).send("hi")

    assert "abc123" not in str(caught.value)
    assert sent == []


KEY = "fake-key-abc12345"


def echoing_client(data: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=data))
    )


async def test_an_openai_server_echoing_the_key_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_KEY", KEY)
    arguments = f'{{"body": "{KEY}", "{KEY}": 1}}'
    message = {
        "content": f"your key is {KEY}, again {KEY}",
        "reasoning": KEY,
        "tool_calls": [{"function": {"name": "send_email", "arguments": arguments}}],
    }
    data = {"model": KEY, "choices": [{"message": message, "finish_reason": KEY}]}
    config = OpenAICompatConfig(base_url="http://test/v1", model="m", api_key_env="FAKE_KEY")

    response = await OpenAICompatTarget(config, echoing_client(data)).send("hi")

    assert KEY not in response.model_dump_json()
    assert response.text == "your key is [redacted], again [redacted]"
    assert response.tool_calls[0].arguments == {"body": "[redacted]", "[redacted]": 1}


async def test_a_web_app_echoing_a_header_secret_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_KEY", KEY)
    config = HttpTemplateConfig(
        url="http://localhost/a",
        headers={"Authorization": "Bearer ${FAKE_KEY}"},
        body={"q": "{prompt}"},
        response_path="reply",
    )
    client = echoing_client({"reply": f"Bearer {KEY}"})

    response = await HttpTemplateTarget(config, client).send("hi")

    assert response.text == "Bearer [redacted]"


def test_very_short_values_are_left_alone() -> None:
    reply = TargetResponse(text="the cat sat", model="m", latency_ms=1)
    assert redact(reply, ["", "cat"]) == reply


@pytest.mark.parametrize("header", ["Bearer ${K} ", " ${K}", "Bearer ${K}\n"])
def test_a_header_template_that_would_fail_at_send_time_is_rejected(header: str) -> None:
    with pytest.raises(ValidationError, match="header Authorization"):
        HttpTemplateConfig(
            url="http://localhost/a",
            headers={"Authorization": header},
            body={"q": "{prompt}"},
            response_path="r",
        )
