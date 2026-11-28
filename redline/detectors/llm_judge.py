"""The LLM judge: a detector that asks a second model whether an attack worked.

It catches what exact matching can't, such as a paraphrased leak. It is a model, so it can be
wrong or be manipulated: measure its agreement with human labels before quoting its verdicts.
"""

from typing import Any

import httpx
from pydantic import Field, model_validator

from redline.core import Case, DetectionResult, RedlineModel, TargetResponse
from redline.detectors.judge_prompt import (
    RESPONSE_FORMAT,
    JudgeContext,
    JudgeError,
    build_messages,
    parse_verdict,
    prompt_fingerprint,
)
from redline.targets import OpenAICompatConfig, OpenAICompatTarget, RateLimiter

# Tokens the server adds around the messages (role markers and the like).
_TEMPLATE_TOKENS = 64
_MAX_EVIDENCE = 500


class JudgeConfig(RedlineModel):
    """Which model judges, and how. The defaults are for a local Ollama model."""

    model: str = Field(default="gemma4:12b-it-qat", min_length=1)
    base_url: str = Field(default="http://localhost:11434/v1", min_length=1)
    # The NAME of the env var holding the API key (hosted judges only), never the key.
    api_key_env: str | None = None
    # The exact model build, e.g. Ollama's ID: a tag can be updated to point at a new one.
    # Recorded as given; nothing here checks it against the server.
    model_digest: str | None = None
    seed: int = 0
    timeout_s: float = Field(default=180.0, gt=0)
    # Provider options. The default turns thinking off on Ollama.
    extra_body: dict[str, Any] = Field(default_factory=lambda: {"reasoning_effort": "none"})
    # The server's context window. Ollama silently drops the START of a longer prompt, which
    # is where the rule is, so raise this only if the server's window was really raised.
    context_tokens: int = Field(default=4096, gt=0)
    max_output_tokens: int = Field(default=200, gt=0)

    @model_validator(mode="after")
    def _leaves_room_for_input(self) -> "JudgeConfig":
        if self.max_input_bytes <= 0:
            raise ValueError("context_tokens is too small for max_output_tokens")
        return self

    @property
    def max_input_bytes(self) -> int:
        """The longest prompt that is certain to fit: a token is at least one byte."""
        return self.context_tokens - self.max_output_tokens - _TEMPLATE_TOKENS

    def backend(self) -> OpenAICompatConfig:
        return OpenAICompatConfig(
            base_url=self.base_url,
            model=self.model,
            api_key_env=self.api_key_env,
            temperature=0.0,
            timeout_s=self.timeout_s,
            seed=self.seed,
            # Ours last, so provider options can't remove the cap or the reply shape.
            extra_body=self.extra_body
            | {"max_tokens": self.max_output_tokens, "response_format": RESPONSE_FORMAT},
        )

    def describe(self) -> dict[str, Any]:
        """Every setting that can change a verdict. Goes into the config hash."""
        return {
            # Left out: api_key_env and timeout_s, which don't change what the model sees.
            **self.model_dump(mode="json", exclude={"api_key_env", "timeout_s"}),
            "temperature": 0,
            "prompt": prompt_fingerprint(),
        }


class LlmJudge:
    """A `Detector` backed by a model. Built per attempt, because the secret differs."""

    name = "judge"

    def __init__(
        self,
        config: JudgeConfig,
        context: JudgeContext | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self._config = config
        self._context = context or JudgeContext()
        self._target = OpenAICompatTarget(config.backend(), client, limiter=limiter)

    def _failed(self, reason: str, response: TargetResponse | None = None) -> DetectionResult:
        return DetectionResult(
            detector=self.name,
            triggered=False,
            confidence=0,
            error=f"JudgeError: {reason}",
            usage=None if response is None else response.usage,
        )

    async def detect(self, attack: Case, response: TargetResponse) -> tuple[DetectionResult, ...]:
        """One verdict on the reply and its tool calls (never on the model's thinking).

        An unusable verdict comes back as a failed detection, which keeps its token usage.
        Network and HTTP errors are raised.
        """
        messages = build_messages(attack, response, self._context)
        size = sum(len(message["content"].encode()) for message in messages)
        limit = self._config.max_input_bytes
        if size > limit:
            return (self._failed(f"prompt too long to judge: {size} bytes, limit {limit}"),)

        answer = (await self._target.chat(messages)).response
        try:
            verdict = parse_verdict(answer.text)
        except JudgeError as error:
            cut_off = " (cut off at the token cap)" if answer.finish_reason == "length" else ""
            return (self._failed(f"{error}{cut_off}", answer),)
        return (
            DetectionResult(
                detector=self.name,
                triggered=verdict.succeeded,
                evidence=verdict.reason[:_MAX_EVIDENCE],
                usage=answer.usage,
            ),
        )
