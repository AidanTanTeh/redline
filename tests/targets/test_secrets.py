import httpx
import pytest
from pydantic import ValidationError

from redline.targets import HttpTemplateConfig, OpenAICompatConfig, OpenAICompatTarget
from redline.targets.secrets import read_secret

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


@pytest.mark.parametrize("header", ["Bearer ${K} ", " ${K}", "Bearer ${K}\n"])
def test_a_header_template_that_would_fail_at_send_time_is_rejected(header: str) -> None:
    with pytest.raises(ValidationError, match="header Authorization"):
        HttpTemplateConfig(
            url="http://localhost/a",
            headers={"Authorization": header},
            body={"q": "{prompt}"},
            response_path="r",
        )
