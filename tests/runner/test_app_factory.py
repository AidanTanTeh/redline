"""Running attacks against a user's own function or web app."""

import json
from typing import Any

import httpx
import pytest

from redline.attacks import load_corpus
from redline.benign import load_tasks
from redline.core import (
    Attack,
    AttackCategory,
    Document,
    Outcome,
    TargetResponse,
    ToolCall,
    ToolCallMatch,
)
from redline.runner import (
    EstimatingFactory,
    FunctionFactory,
    HttpFactory,
    RunConfig,
    TrialFactory,
    config_hash,
    run_attacks,
)
from redline.targets import HttpTemplateConfig, TrialContext

DIRECT = Attack(id="d-01", name="x", category=AttackCategory.DIRECT_INJECTION, prompt="Hi")
INDIRECT = Attack(
    id="ii-01",
    name="x",
    category=AttackCategory.INDIRECT_INJECTION,
    prompt="What about the lamp?",
    documents=(Document(title="Review", content="Great lamp!"),),
)
LEAK = Attack(id="spl-01", name="x", category=AttackCategory.SYSTEM_PROMPT_LEAKAGE, prompt="Repeat")
EMAIL = Attack(
    id="ea-01",
    name="x",
    category=AttackCategory.EXCESSIVE_AGENCY,
    prompt="Email them",
    success_tool_call=ToolCallMatch(tool="send_email"),
)
CANARY = "zq81-mm07-planted"


def echo(prompt: str, context: TrialContext) -> str:
    return f"secret: {context.canary}"


def refuse(prompt: str, context: TrialContext) -> str:
    return "no"


def full(*tools: str, text: str = "ok") -> TargetResponse:
    calls = tuple(ToolCall(name=name) for name in tools)
    return TargetResponse(text=text, model="m", latency_ms=1, tool_calls=calls)


def function_factory(function: Any = refuse, **options: Any) -> FunctionFactory:
    return FunctionFactory(function, version="v1", **options)


def http_config(**changes: Any) -> HttpTemplateConfig:
    settings: dict[str, Any] = {
        "url": "http://localhost:3000/hook/private-path",
        "body": {"q": "{prompt}"},
        "response_path": "reply",
    }
    return HttpTemplateConfig(**(settings | changes))


def http_factory(handler: Any = None, **options: Any) -> HttpFactory:
    handler = handler or (lambda _: httpx.Response(200, json={"reply": "no"}))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    options.setdefault("canary", CANARY)
    return HttpFactory(options.pop("config", http_config()), client=client, **options)


async def outcome(factory: TrialFactory, attack: Attack) -> Outcome:
    report = await run_attacks([attack], factory, RunConfig(), cases_hash="c", sleep=no_sleep)
    return report.results[0].outcome


async def no_sleep(seconds: float) -> None:
    return None


def hash_of(factory: TrialFactory) -> str:
    return config_hash(factory.describe(), "c", [DIRECT], RunConfig())


# --- judging ----------------------------------------------------------------


async def test_a_leaked_canary_is_caught_for_a_function() -> None:
    factory: EstimatingFactory = function_factory(echo)  # mypy checks the protocol
    assert await outcome(factory, DIRECT) is Outcome.SUCCEEDED
    assert await outcome(function_factory(), DIRECT) is Outcome.RESISTED


async def test_the_planted_canary_is_what_http_replies_are_checked_for() -> None:
    leaking = http_factory(lambda _: httpx.Response(200, json={"reply": f"it is {CANARY}"}))
    assert await outcome(leaking, DIRECT) is Outcome.SUCCEEDED
    assert await outcome(http_factory(), DIRECT) is Outcome.RESISTED


async def test_each_attempt_gets_its_own_canary_unless_one_is_planted() -> None:
    seen: list[str] = []

    def agent(prompt: str, context: TrialContext) -> str:
        seen.append(context.canary)
        return "no"

    for factory in (function_factory(agent), function_factory(agent, canary=CANARY)):
        await run_attacks([DIRECT], factory, RunConfig(repeats=2), cases_hash="c")

    assert seen[0] != seen[1]
    assert seen[2:] == [CANARY, CANARY]


async def test_leak_phrases_count_only_for_leakage_attacks() -> None:
    def agent(prompt: str, context: TrialContext) -> str:
        return "My instructions: You are TransitBot"

    factory = function_factory(agent, leak_phrases=["You are TransitBot"])
    assert await outcome(factory, LEAK) is Outcome.SUCCEEDED
    assert await outcome(factory, DIRECT) is Outcome.RESISTED
    assert await outcome(function_factory(agent), LEAK) is Outcome.RESISTED


# --- tools ------------------------------------------------------------------


@pytest.mark.parametrize("tools", [(), ("send_message",)])
async def test_an_attack_on_a_tool_the_app_lacks_is_skipped(tools: tuple[str, ...]) -> None:
    factory = function_factory(lambda p, c: full(), tool_names=tools)
    assert await outcome(factory, EMAIL) is Outcome.SKIPPED


async def test_a_reported_tool_call_is_judged_ignoring_case() -> None:
    factory = function_factory(lambda p, c: full("send_email"), tool_names=["Send_Email"])
    assert await outcome(factory, EMAIL) is Outcome.SUCCEEDED


async def test_a_function_with_tools_may_not_answer_with_a_plain_string() -> None:
    factory = function_factory(tool_names=["send_email"])
    assert await outcome(factory, EMAIL) is Outcome.ERROR


def test_tasks_are_filtered_by_tool_too() -> None:
    tool_tasks = [t for t in load_tasks().tasks if t.needs_tools]
    assert tool_tasks
    assert not any(function_factory().supports(t) for t in tool_tasks)


def test_every_corpus_attack_can_be_built() -> None:
    factory = function_factory(tool_names=["send_email"], leak_phrases=["You are X"])
    for attack in load_corpus().attacks:
        assert factory.make_trial(attack, 0).detectors


# --- documents --------------------------------------------------------------


@pytest.mark.parametrize("delivery", ["prompt", "context"])
async def test_documents_arrive_exactly_one_way(delivery: str) -> None:
    seen: list[tuple[str, TrialContext]] = []

    def agent(prompt: str, context: TrialContext) -> str:
        seen.append((prompt, context))
        return "no"

    options: dict[str, Any] = {"documents": delivery}
    if delivery == "context":
        options["assume_delivered"] = True
    await outcome(function_factory(agent, **options), INDIRECT)

    prompt, context = seen[0]
    assert ("Great lamp!" in prompt) is (delivery == "prompt")
    assert (context.documents == INDIRECT.documents) is (delivery == "context")


@pytest.mark.parametrize(
    ("tools", "expected"),
    [((), Outcome.NOT_DELIVERED), (("search",), Outcome.RESISTED)],
)
async def test_an_unread_document_is_not_counted_as_resisted(
    tools: tuple[str, ...], expected: Outcome
) -> None:
    factory = function_factory(
        lambda p, c: full(*tools), documents="context", retrieval_tool="search"
    )
    assert await outcome(factory, INDIRECT) is expected


# --- retries and caps -------------------------------------------------------


@pytest.mark.parametrize(("safe", "calls"), [(False, 1), (True, 2)])
async def test_a_function_is_only_retried_when_marked_safe(safe: bool, calls: int) -> None:
    seen: list[str] = []

    async def agent(prompt: str, context: TrialContext) -> str:
        seen.append(prompt)
        if len(seen) == 1:
            raise httpx.ConnectError("inside your agent, maybe after a tool ran")
        return "no"

    await outcome(function_factory(agent, safe_to_retry=safe), DIRECT)
    assert len(seen) == calls


@pytest.mark.parametrize(
    ("first", "rate_limits", "calls"),
    [
        ("connect", False, 2),
        (429, False, 1),
        (429, True, 2),
        (500, True, 1),
    ],
)
async def test_http_retries_only_what_cannot_repeat_an_action(
    first: str | int, rate_limits: bool, calls: int
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) > 1:
            return httpx.Response(200, json={"reply": "no"})
        if first == "connect":
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(int(first))

    await outcome(http_factory(handler, retry_rate_limits=rate_limits), DIRECT)
    assert len(seen) == calls


async def test_a_request_cap_works_and_a_cost_cap_is_refused() -> None:
    factory = function_factory(max_calls_per_attempt=3)
    report = await run_attacks(
        [DIRECT, LEAK], factory, RunConfig(max_requests=3), cases_hash="c",
        allow_over_estimate=True,
    )  # fmt: skip
    assert [r.outcome for r in report.results] == [Outcome.RESISTED, Outcome.NOT_RUN]


def test_a_cap_needs_the_calls_per_attempt() -> None:
    with pytest.raises(ValueError, match="max_calls_per_attempt"):
        function_factory().estimate_attempt(DIRECT)


# --- what gets saved --------------------------------------------------------


def test_http_reports_hold_the_host_and_a_fingerprint_only() -> None:
    factory = http_factory(leak_phrases=["You are TransitBot"])
    saved = json.dumps(factory.describe()) + factory.name
    assert "localhost" in saved
    for private in ("private-path", CANARY, "TransitBot", '"q"'):
        assert private not in saved


@pytest.mark.parametrize(
    "changes",
    [
        {"url": "http://localhost:3000/other"},
        {"body": {"message": "{prompt}"}},
        {"response_path": "answer"},
    ],
)
def test_changing_the_http_settings_changes_the_hash(changes: dict[str, Any]) -> None:
    assert hash_of(http_factory()) != hash_of(http_factory(config=http_config(**changes)))


def test_the_timeout_does_not_change_the_hash() -> None:
    slower = http_factory(config=http_config(timeout_s=5))
    assert hash_of(http_factory()) == hash_of(slower)


@pytest.mark.parametrize(
    "options",
    [
        {"documents": "context", "assume_delivered": True},
        {"canary": CANARY},
        {"tool_names": ["x"]},
        {"leak_phrases": ["x"]},
        {"model": "other"},
    ],
)
def test_changing_a_function_setting_changes_the_hash(options: dict[str, Any]) -> None:
    assert hash_of(function_factory()) != hash_of(function_factory(**options))


def test_the_version_label_changes_the_hash() -> None:
    assert hash_of(function_factory()) != hash_of(FunctionFactory(refuse, version="v2"))


# --- error messages ---------------------------------------------------------


def crash(prompt: str, context: TrialContext) -> str:
    raise KeyError("customer@private.example")


@pytest.mark.parametrize("keep", [False, True])
async def test_your_functions_error_messages_are_hidden_unless_asked_for(keep: bool) -> None:
    factory = function_factory(crash, save_error_messages=keep)

    report = await run_attacks([DIRECT], factory, RunConfig(), cases_hash="c")

    error = report.results[0].error
    assert error is not None and error.startswith("KeyError")
    assert ("customer@private.example" in report.model_dump_json()) is keep


async def test_redlines_own_complaint_about_the_reply_is_always_shown() -> None:
    factory = function_factory(tool_names=["send_email"])
    report = await run_attacks([EMAIL], factory, RunConfig(), cases_hash="c")
    assert "tool_calls" in (report.results[0].error or "")


async def test_hidden_errors_do_not_change_what_is_retried() -> None:
    calls: list[str] = []

    async def agent(prompt: str, context: TrialContext) -> str:
        calls.append(prompt)
        raise httpx.ConnectError("private")

    factory = function_factory(agent, safe_to_retry=True)
    report = await run_attacks([DIRECT], factory, RunConfig(), cases_hash="c", sleep=no_sleep)

    assert len(calls) == 3
    assert "private" not in report.model_dump_json()


# --- mistakes caught early --------------------------------------------------


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"tool_names": "send_email"}, "not one string"),
        ({"leak_phrases": "You are"}, "not one string"),
        ({"max_calls_per_attempt": 0}, "at least 1"),
        ({"canary": "short"}, "at least"),
        ({"retrieval_tool": "search"}, "context"),
        ({"assume_delivered": True}, "context"),
        ({"documents": "context"}, "exactly one"),
        (
            {"documents": "context", "retrieval_tool": "s", "assume_delivered": True},
            "exactly one",
        ),
    ],
)
def test_bad_function_settings_are_rejected(options: dict[str, Any], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        function_factory(**options)


def test_a_version_label_is_required() -> None:
    with pytest.raises(ValueError, match="version"):
        FunctionFactory(refuse, version=" ")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"tool_names": ["send_email"]}, "tool_calls_path"),
        ({"canary": "short"}, "at least"),
        ({"config": http_config(body={"q": "{prompt}", "x": CANARY})}, "must not appear"),
    ],
)
def test_bad_http_settings_are_rejected(options: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        http_factory(**options)


@pytest.mark.parametrize("value", [CANARY, CANARY.upper(), f"x-{CANARY}", CANARY[:10]])
def test_a_header_value_may_not_overlap_the_canary(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("APP_TOKEN", value)
    config = http_config(headers={"X-Debug": "${APP_TOKEN}"})
    with pytest.raises(ValueError, match="overlaps the canary"):
        http_factory(config=config).preflight()


def test_an_unrelated_header_value_passes_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_TOKEN", "fake-token-9999")
    config = http_config(headers={"Authorization": "Bearer ${APP_TOKEN}"})
    http_factory(config=config).preflight()


def test_preflight_checks_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_TOKEN", raising=False)
    config = http_config(headers={"Authorization": "Bearer ${APP_TOKEN}"})
    with pytest.raises(RuntimeError, match="APP_TOKEN"):
        http_factory(config=config).preflight()
