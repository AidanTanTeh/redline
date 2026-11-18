"""Adapter for any server that speaks the OpenAI chat-completions API (Ollama, Gemini, ...)."""

import json
import os
import re
import time
from typing import Any

import httpx
from pydantic import Field

from redline.core import RedlineModel, TargetResponse, ToolCall

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


def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for raw in raw_calls:
        function = raw["function"]
        # The API sends arguments as a JSON *string*. Models sometimes emit broken JSON,
        # so keep the raw text instead of crashing the whole run.
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {"_raw": function.get("arguments")}
        if not isinstance(arguments, dict):
            arguments = {"_raw": arguments}
        calls.append(ToolCall(name=function["name"], arguments=arguments))
    return tuple(calls)


class OpenAICompatTarget:
    """A Target backed by an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, config: OpenAICompatConfig, client: httpx.AsyncClient | None = None) -> None:
        self._config = config
        # Optional shared client (tests inject a fake transport; the runner can share one pool).
        self._client = client

    @property
    def name(self) -> str:
        return f"openai-compat/{self._config.model}"

    def _headers(self) -> dict[str, str]:
        env_name = self._config.api_key_env
        if env_name is None:
            return {}
        key = os.environ.get(env_name)
        if not key:
            # Name the variable, never print a value.
            raise RuntimeError(f"Environment variable {env_name} is not set (or is empty).")
        return {"Authorization": f"Bearer {key}"}

    def _body(self, prompt: str) -> dict[str, Any]:
        cfg = self._config
        messages: list[dict[str, str]] = []
        if cfg.system_prompt is not None:
            messages.append({"role": "system", "content": cfg.system_prompt})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
        }
        if cfg.seed is not None:
            body["seed"] = cfg.seed
        return body | cfg.extra_body

    async def send(self, prompt: str) -> TargetResponse:
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        headers = self._headers()
        body = self._body(prompt)
        timeout = self._config.timeout_s

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

        return TargetResponse(
            text=text,
            reasoning=reasoning,
            model=data.get("model") or self._config.model,
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason"),
            tool_calls=tool_calls,
        )
