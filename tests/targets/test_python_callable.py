import asyncio
import threading
from typing import Any

import pytest

from redline.core import Document, TargetResponse, ToolCall
from redline.targets import PythonCallableTarget, Target, TrialContext

CONTEXT = TrialContext(canary="SECRET-1234", documents=(Document(title="t", content="c"),), seed=7)


def target(function: Any, **options: Any) -> PythonCallableTarget:
    return PythonCallableTarget(function, CONTEXT, name="my-agent", **options)


async def test_an_async_function_gets_the_prompt_and_context() -> None:
    seen: list[tuple[str, TrialContext]] = []

    async def agent(prompt: str, context: TrialContext) -> str:
        seen.append((prompt, context))
        return f"the secret is {context.canary}"

    response = await target(agent, model="my-model").send("hello")

    assert seen == [("hello", CONTEXT)]
    assert response.text == "the secret is SECRET-1234"
    assert response.model == "my-model"
    assert response.latency_ms >= 0
    assert response.usage is None  # unknown, so a budget charges the worst case


async def test_a_normal_function_runs_off_the_event_loop_thread() -> None:
    threads: list[int] = []

    def agent(prompt: str, context: TrialContext) -> str:
        threads.append(threading.get_ident())
        return "ok"

    assert (await target(agent).send("hi")).text == "ok"
    assert threads != [threading.get_ident()]


async def test_an_object_with_an_async_call_is_awaited() -> None:
    class Agent:
        async def __call__(self, prompt: str, context: TrialContext) -> str:
            return prompt.upper()

    assert (await target(Agent()).send("hi")).text == "HI"


async def test_a_full_response_is_passed_through_with_its_tool_calls() -> None:
    full = TargetResponse(
        text="done",
        model="theirs",
        latency_ms=12,
        tool_calls=(ToolCall(name="send_email", arguments={"to": "a@b.example"}),),
    )

    async def agent(prompt: str, context: TrialContext) -> TargetResponse:
        return full

    assert await target(agent).send("hi") is full


@pytest.mark.parametrize("bad", [None, 42, {"text": "hi"}, b"hi"])
async def test_any_other_return_value_is_an_error(bad: object) -> None:
    def agent(prompt: str, context: TrialContext) -> Any:
        return bad

    with pytest.raises(TypeError, match="must return str or TargetResponse"):
        await target(agent).send("hi")


async def test_a_string_is_rejected_when_tool_calls_must_be_reported() -> None:
    def agent(prompt: str, context: TrialContext) -> str:
        return "I sent it"

    with pytest.raises(TypeError, match="tool_calls"):
        await target(agent, require_full_response=True).send("hi")


async def test_the_functions_own_errors_are_not_swallowed() -> None:
    def agent(prompt: str, context: TrialContext) -> str:
        raise KeyError("boom")

    with pytest.raises(KeyError, match="boom"):
        await target(agent).send("hi")


async def test_a_timeout_cancels_an_async_function() -> None:
    cancelled = False

    async def agent(prompt: str, context: TrialContext) -> str:
        nonlocal cancelled
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled = True
            raise
        return "late"

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await target(agent).send("hi")
    assert cancelled


def test_it_satisfies_target() -> None:
    async def agent(prompt: str, context: TrialContext) -> str:
        return ""

    checked: Target = target(agent)  # mypy checks this line
    assert checked.name == "my-agent"
