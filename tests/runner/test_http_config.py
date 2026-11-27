import traceback
from pathlib import Path

import httpx
import pytest

from redline.core import Attack, AttackCategory, Outcome
from redline.runner import ConfigError, RunConfig, load_http_factory, run_attacks

EXAMPLE = Path(__file__).parents[2] / "examples" / "http_target.yaml"
MINIMAL = """
target:
  url: http://localhost:3000/chat
  body: {q: "{prompt}"}
  response_path: reply
canary: planted-secret-42
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "target.yaml"
    path.write_text(text, encoding="utf-8")
    return path


async def test_the_example_file_loads_and_runs_against_a_matching_app() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        data = {"reply": "demo-canary-7f3k9q2m", "tool_calls": [], "finish_reason": "stop"}
        return httpx.Response(200, json={"data": data})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    factory = load_http_factory(EXAMPLE, client=client)
    attack = Attack(id="d", name="x", category=AttackCategory.DIRECT_INJECTION, prompt="Hi")

    report = await run_attacks([attack], factory, RunConfig(), cases_hash="c")

    assert factory.has_tools
    assert report.results[0].outcome is Outcome.SUCCEEDED


def test_a_minimal_file_loads(tmp_path: Path) -> None:
    factory = load_http_factory(write(tmp_path, MINIMAL))
    assert factory.name == "http/localhost"
    assert not factory.has_tools


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "whole file"),
        ("- a\n- b\n", "whole file"),
        (MINIMAL + "surprise: 1\n", "surprise"),
        (MINIMAL.replace("canary: planted-secret-42", ""), "canary"),
        (MINIMAL.replace("planted-secret-42", "short"), "at least"),
        (MINIMAL.replace("localhost:3000", "app.example"), "authorized"),
        (MINIMAL + "tool_names: [send_email]\n", "tool_calls_path"),
        (MINIMAL + "tool_names: send_email\n", "tool_names"),
        (MINIMAL + "requests_per_minute: 0\n", "requests_per_minute"),
        (MINIMAL + "canary: another-secret-99\n", "appears twice"),
        (MINIMAL.replace("  response_path: reply", "  url: x\n  response_path: reply"), "twice"),
        (MINIMAL.replace('{q: "{prompt}"}', '{q: "{prompt}", day: 2026-01-02}'), "quotes"),
        # YAML reads a bare `on` as the boolean true, so it can't be a field name.
        (MINIMAL.replace('{q: "{prompt}"}', '{q: "{prompt}", on: 1}'), "valid string"),
    ],
)
def test_mistakes_are_reported(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_http_factory(write(tmp_path, text))


@pytest.mark.parametrize(
    "text",
    [
        # Broken YAML on the very line holding the secret.
        MINIMAL + 'extra: "abc123secret\n',
        MINIMAL + "extra: [abc123secret\n",
        # Valid YAML, rejected settings.
        MINIMAL.replace(
            "  response_path: reply",
            "  headers: {Authorization: Bearer abc123secret}\n  response_path: reply",
        ),
        MINIMAL.replace('{q: "{prompt}"}', '{q: "{prompt}", apiKey: abc123secret}'),
        MINIMAL + "max_calls_per_attempt: abc123secret\n",
    ],
)
def test_error_messages_and_tracebacks_never_quote_a_secret(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError) as caught:
        load_http_factory(write(tmp_path, text))
    shown = "".join(traceback.format_exception(caught.value))
    assert "abc123secret" not in shown


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="can't read"):
        load_http_factory(tmp_path / "nope.yaml")


def test_yaml_cannot_build_python_objects(tmp_path: Path) -> None:
    text = MINIMAL + "extra: !!python/object/apply:os.getcwd []\n"
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_http_factory(write(tmp_path, text))
