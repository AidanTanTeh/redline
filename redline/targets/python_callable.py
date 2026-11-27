"""Adapter for testing your own Python code, such as an agent, as a Target."""

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable

from redline.core import Document, RedlineModel, TargetResponse


class TrialContext(RedlineModel):
    """What your function is given for one attempt, besides the prompt."""

    # Plant this secret where a real one would live (system prompt, database row, tool result).
    # If you don't, a leak can never be detected and every leak attack looks "resisted".
    canary: str
    # An indirect attack's poisoned documents. Feed them to your retriever or tool.
    documents: tuple[Document, ...] = ()
    seed: int


# Return plain text, or a full TargetResponse to also report tool calls, usage and so on.
Reply = str | TargetResponse
AgentFunction = Callable[[str, TrialContext], Reply | Awaitable[Reply]]


class AgentReplyError(TypeError):
    """Redline's own complaint about what a function returned. Its message holds no data from
    the app, so it is safe to save."""


class PythonCallableTarget:
    """A Target that calls your function. Build a new one per attempt (the context differs).

    Run agents with MOCKED tools: attacks try to make them send, delete or buy things.
    An `async def` function is stopped by the attempt timeout. A normal function runs in a
    thread, which Python cannot stop: after a timeout it keeps running in the background.
    """

    def __init__(
        self,
        function: AgentFunction,
        context: TrialContext,
        *,
        name: str,
        model: str = "unknown",
        require_full_response: bool = False,
    ) -> None:
        # True when tool calls are being judged: a plain string can't report them, so every
        # tool attack would silently look "resisted".
        self._require_full_response = require_full_response
        self._function = function
        self._context = context
        self._name = name
        # Recorded on string replies; a returned TargetResponse says its own.
        self._model = model

    @property
    def name(self) -> str:
        return self._name

    async def send(self, prompt: str) -> TargetResponse:
        start = time.perf_counter()
        if inspect.iscoroutinefunction(self._function):
            reply = await self._function(prompt, self._context)
        else:
            # In a thread, so a slow function doesn't freeze every other attempt.
            reply = await asyncio.to_thread(self._function, prompt, self._context)
            # E.g. an object with `async def __call__`, which the check above doesn't spot.
            if inspect.isawaitable(reply):
                reply = await reply
        latency_ms = (time.perf_counter() - start) * 1000

        if isinstance(reply, TargetResponse):
            return reply
        if isinstance(reply, str):
            if self._require_full_response:
                raise AgentReplyError(
                    "return a TargetResponse with its tool_calls (empty if none): a string "
                    "can't show whether a tool was called"
                )
            return TargetResponse(text=reply, model=self._model, latency_ms=latency_ms)
        # Turning None or a dict into text would be judged as a real reply.
        raise AgentReplyError(f"the function must return str or TargetResponse, not {type(reply)}")
