"""Adapter for any server that speaks the OpenAI chat-completions API (Ollama, Gemini, ...)."""

import json
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from pydantic import Field

from redline.core import RedlineModel, TargetResponse, ToolCall, Usage
from redline.targets.limiter import RateLimiter
from redline.targets.secrets import read_secret, redact

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class OpenAICompatConfig(RedlineModel):
    """Everything needed to talk to one OpenAI-compatible endpoint."""

    # Base URL up to and including the version prefix, e.g. "http://localhost:11434/v1".
    # The adapter appends "/chat/completions".
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    system_prompt: str | None = None
    # The NAME of the env var holding the API key, never the key itself.
    # None means no auth header (local Ollama).
    api_key_env: str | None = Field(default=None, min_length=1)
    # 0 keeps runs as repeatable as the server allows.
    temperature: float = Field(default=0.0, ge=0, le=2)
    timeout_s: float = Field(default=60.0, gt=0)
    seed: int | None = None
    # Extra JSON merged into the request body, for provider-specific options
    # (e.g. Ollama's "think"). Treat as read-only, like ToolCall.arguments.
    extra_body: dict[str, Any] = Field(default_factory=dict)


def _split_reasoning(content: str, reasoning_field: str | None) -> tuple[str, str | None]:
    """Separate the final answer from thinking text.

    Servers differ: some put thinking in a separate message field, others leave
    <think>...</think> tags inside the content. Handle both. Returns (text, reasoning).
    """
    parts: list[str] = [reasoning_field] if reasoning_field else []
    parts.extend(block.strip() for block in _THINK_BLOCK.findall(content))
    text = _THINK_BLOCK.sub("", content)
    # A reply cut off mid-thought has an opening tag but no closing one.
    if "<think>" in text:
        text, _, unfinished = text.partition("<think>")
        parts.append(unfinished.strip())
    reasoning = "\n".join(p for p in parts if p)
    return text.strip(), reasoning or None


def _call_id(raw: dict[str, Any], index: int) -> str:
    # Servers normally send an id. If one doesn't, invent a stable one so the conversation
    # history stays valid (a tool result has to name the call it answers).
    return str(raw.get("id") or f"call_{index}")


def _assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """The assistant's turn, rebuilt cleanly so it can be appended to the conversation history.

    Only role/content/tool_calls are kept (not extras like `reasoning`), and each call's
    `arguments` stays the raw JSON string the model produced, exactly as the API expects it back.
    """
    result: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
    raw_calls = message.get("tool_calls") or []
    if raw_calls:
        result["tool_calls"] = [
            {
                "id": _call_id(raw, index),
                "type": "function",
                "function": {
                    "name": raw["function"]["name"],
                    "arguments": raw["function"].get("arguments") or "{}",
                },
            }
            for index, raw in enumerate(raw_calls)
        ]
    return result


def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for index, raw in enumerate(raw_calls):
        function = raw["function"]
        # The API sends arguments as a JSON *string*. Models sometimes emit broken JSON,
        # so keep the raw text instead of crashing the whole run.
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {"_raw": function.get("arguments")}
        if not isinstance(arguments, dict):
            arguments = {"_raw": arguments}
        calls.append(ToolCall(name=function["name"], arguments=arguments, id=_call_id(raw, index)))
    return tuple(calls)


def _parse_usage(raw: object) -> Usage:
    """Token counts for one request. Anything missing or malformed counts as unknown."""

    def count(key: str) -> int | None:
        value = raw.get(key) if isinstance(raw, dict) else None
        # type() rather than isinstance(): in Python, True is also an int.
        return value if type(value) is int and value >= 0 else None

    prompt, completion, total = (
        count("prompt_tokens"),
        count("completion_tokens"),
        count("total_tokens"),
    )
    output = completion
    # Some servers (reported for Gemini) leave thinking tokens out of completion_tokens and
    # only include them in the total. Take the larger figure so billed tokens aren't missed.
    if prompt is not None and total is not None and total >= prompt:
        output = max(total - prompt, completion or 0)
    return Usage(requests=1, input_tokens=prompt, output_tokens=output)


class ChatTurn(RedlineModel):
    """One model turn: what it said, plus the message to append to the history."""

    response: TargetResponse
    # {"role": "assistant", "content": ..., "tool_calls": [...]}. Append it, then append one
    # {"role": "tool", "tool_call_id": ..., "content": ...} per call, then call chat() again.
    assistant_message: dict[str, Any]


class OpenAICompatTarget:
    """A Target backed by an OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        config: OpenAICompatConfig,
        client: httpx.AsyncClient | None = None,
        *,
        limiter: RateLimiter | None = None,
    ) -> None:
        self._config = config
        # Optional shared client (tests inject a fake transport; the runner can share one pool).
        self._client = client
        # Optional, and shared between targets that use the same quota.
        self._limiter = limiter

    @property
    def name(self) -> str:
        return f"openai-compat/{self._config.model}"

    def _headers(self) -> dict[str, str]:
        env_name = self._config.api_key_env
        if env_name is None:
            return {}
        return {"Authorization": f"Bearer {read_secret(env_name)}"}

    def start_messages(self, prompt: str) -> list[dict[str, Any]]:
        """The opening of a conversation: the configured system prompt (if any), then the user."""
        messages: list[dict[str, Any]] = []
        if self._config.system_prompt is not None:
            messages.append({"role": "system", "content": self._config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _body(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        cfg = self._config
        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": list(messages),
            "temperature": cfg.temperature,
        }
        if tools:
            body["tools"] = list(tools)
        if cfg.seed is not None:
            body["seed"] = cfg.seed
        return body | cfg.extra_body

    async def send(self, prompt: str) -> TargetResponse:
        """One prompt in, one response out (a single model turn, no tools)."""
        turn = await self.chat(self.start_messages(prompt))
        return turn.response

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
    ) -> ChatTurn:
        """One model turn over a full conversation, optionally offering tools.

        `messages` is sent exactly as given (the caller owns the history, including any system
        message). `tools` uses the OpenAI schema: {"type": "function", "function": {...}}.
        """
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        headers = self._headers()
        body = self._body(messages, tools)
        timeout = self._config.timeout_s

        if self._limiter is not None:
            await self._limiter.wait()  # before the timer: waiting isn't model latency
        start = time.perf_counter()
        if self._client is not None:
            response = await self._client.post(url, json=body, headers=headers, timeout=timeout)
        else:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=body, headers=headers, timeout=timeout)
        latency_ms = (time.perf_counter() - start) * 1000

        response.raise_for_status()
        data = response.json()
        try:
            choice = data["choices"][0]
            message = choice["message"]
            text, reasoning = _split_reasoning(
                message.get("content") or "",
                message.get("reasoning") or message.get("reasoning_content"),
            )
            tool_calls = _parse_tool_calls(message.get("tool_calls") or [])
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ValueError("Unexpected response shape from /chat/completions") from exc

        target_response = TargetResponse(
            text=text,
            reasoning=reasoning,
            model=data.get("model") or self._config.model,
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason"),
            tool_calls=tool_calls,
            usage=_parse_usage(data.get("usage")),
        )
        sent_key = headers.get("Authorization", "").removeprefix("Bearer ")
        target_response = redact(target_response, [sent_key])
        return ChatTurn(response=target_response, assistant_message=_assistant_message(message))
