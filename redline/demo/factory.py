"""Plugs VulnBot into the runner: a fresh bot, canary and detectors for every attempt."""

import json
import os
from typing import Any

import httpx

from redline.attacks import render_prompt
from redline.core import Attack, AttackCategory, BenignTask, Case, Document
from redline.demo.channels import Channel, channel_tools, make_vulnbot, was_delivered
from redline.demo.tools import send_email_tool
from redline.demo.vulnbot import DEFAULT_MAX_TOOL_ROUNDS, build_system_prompt, leak_phrases
from redline.detectors import (
    CANARY_PREFIX,
    CanaryDetector,
    Detector,
    JudgeConfig,
    JudgeContext,
    LlmJudge,
    RegexDetector,
    TaskDetector,
    ToolCallDetector,
    seeded_canary,
    unknown_to_attacker,
)
from redline.runner import AttemptEstimate, Trial, estimate_tokens, judge_estimate
from redline.targets import OpenAICompatConfig, RateLimiter

# Input tokens per request that we can't see: role markers and the server's tool instructions.
_TEMPLATE_TOKENS = 100

# A made-up attack, rendered by describe() so that a change to HOW documents are shown to the
# model also changes the config hash.
_PROBE = Attack(
    id="probe",
    name="probe",
    category=AttackCategory.INDIRECT_INJECTION,
    prompt="probe question",
    documents=(Document(title="probe title", content="probe content"),),
)


class VulnBotFactory:
    """A `TrialFactory` for VulnBot in one delivery channel.

    With `judge`, an LLM judge also rules on every ATTACK, alongside the exact-match
    detectors. Benign tasks are still checked by `TaskDetector` only: "any detector
    triggered" would let the wording check overrule the judge.
    """

    def __init__(
        self,
        backend: OpenAICompatConfig,
        channel: Channel,
        *,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        scan_reasoning: bool = False,
        client: httpx.AsyncClient | None = None,
        limiter: RateLimiter | None = None,
        judge: JudgeConfig | None = None,
        judge_limiter: RateLimiter | None = None,
    ) -> None:
        # One limiter for every bot this factory builds. Not in describe(): it changes how
        # fast requests go, not what is sent.
        self._limiter = limiter
        self._judge = judge
        # Separate, because the judge may be on a different server with its own quota.
        self._judge_limiter = judge_limiter
        self._backend = backend
        self._channel: Channel = channel
        self._max_tool_rounds = max_tool_rounds
        self._scan_reasoning = scan_reasoning
        # Optional shared connection pool (and how tests plug in a fake server).
        self._client = client

    @property
    def name(self) -> str:
        return f"vulnbot/{self._backend.model}/{self._channel}"

    @property
    def model(self) -> str:
        return self._backend.model

    @property
    def has_tools(self) -> bool:
        return True  # both channels can at least send email

    @property
    def has_judge(self) -> bool:
        return self._judge is not None

    def describe(self) -> dict[str, Any]:
        tools = channel_tools(self._channel, _PROBE.documents)
        # Present only with a judge: even `"judge": None` would change every earlier hash.
        judged = {} if self._judge is None else {"judge": self._judge.describe()}
        return {
            **judged,
            "target": "vulnbot",
            # Left out: seed (set per attempt; the run's base_seed is hashed instead),
            # system_prompt (VulnBot replaces it), and api_key_env and timeout_s (neither
            # changes what the model sees).
            "backend": self._backend.model_dump(
                mode="json",
                exclude={"seed", "system_prompt", "api_key_env", "timeout_s"},
            ),
            "channel": self._channel,
            "max_tool_rounds": self._max_tool_rounds,
            "system_prompt": build_system_prompt("{canary}", [tool.name for tool in tools]),
            "tools": [tool.spec for tool in tools],
            # What the model reads back: how documents are rendered and what email returns.
            "probe_message": render_prompt(_PROBE),
            "probe_tool_results": [tool.run({"to": "probe@acme.example"}) for tool in tools],
            "email_error": send_email_tool().run({}),
            "canary": "prefix + sha256(seed:attack_id)[:8]",
            "detectors": {
                "canary": {
                    "public_prefix": CANARY_PREFIX,
                    "scan_reasoning": self._scan_reasoning,
                    "scan_tool_calls": True,
                },
                "tool_call": {},
                "prompt_leak": {
                    "category": AttackCategory.SYSTEM_PROMPT_LEAKAGE.value,
                    "phrases": list(leak_phrases([tool.name for tool in tools])),
                    "skips_phrases_in_attack": True,
                    "scan_reasoning": self._scan_reasoning,
                    "scan_tool_calls": True,
                },
            },
        }

    def preflight(self) -> None:
        names = [self._backend.api_key_env]
        if self._judge is not None:
            names.append(self._judge.api_key_env)
        for name in names:
            # Checks only that the variable has a value. The value is never kept or shown.
            if name is not None and not os.environ.get(name):
                raise RuntimeError(f"Environment variable {name} is not set (or is empty).")

    def _detectors(self, attack: Case, canary: str) -> tuple[Detector, ...]:
        # A benign task is judged only on whether it was completed.
        if isinstance(attack, BenignTask):
            return (TaskDetector(),)
        if not isinstance(attack, Attack):
            raise TypeError(f"VulnBotFactory can't judge a {type(attack).__name__}")
        detectors: list[Detector] = [
            # public_prefix is wired here once, so no caller can forget it.
            CanaryDetector(
                canary,
                public_prefix=CANARY_PREFIX,
                scan_reasoning=self._scan_reasoning,
                scan_tool_calls=True,
            ),
            ToolCallDetector(),
        ]
        # Only for leakage attacks: elsewhere "success" keeps meaning the canary or a tool call.
        if attack.category is AttackCategory.SYSTEM_PROMPT_LEAKAGE:
            names = [tool.name for tool in channel_tools(self._channel, attack.documents)]
            phrases = unknown_to_attacker(leak_phrases(names), attack)
            if phrases:
                detectors.append(
                    RegexDetector.from_phrases(
                        "prompt_leak",
                        phrases,
                        scan_reasoning=self._scan_reasoning,
                        scan_tool_calls=True,
                    )
                )
        if self._judge is not None:
            names = [tool.name for tool in channel_tools(self._channel, attack.documents)]
            context = JudgeContext(secret=canary, system_prompt=build_system_prompt(canary, names))
            detectors.append(
                LlmJudge(self._judge, context, client=self._client, limiter=self._judge_limiter)
            )
        return tuple(detectors)

    def _output_token_cap(self) -> int:
        """The configured limit on one reply. Without one there is no worst case to compute."""
        body = self._backend.extra_body
        caps = [body.get(key) for key in ("max_tokens", "max_completion_tokens")]
        valid = [cap for cap in caps if type(cap) is int and cap > 0]
        if not valid:
            raise ValueError("set max_tokens in extra_body to estimate a run")
        return max(valid)

    def estimate_attempt(self, case: Case) -> AttemptEstimate:
        """Worst case for one attempt: the model asks for tools until the round cap.

        Assumes each tool is called at most once per round. A model asking for the same tool
        several times in one reply would add more input than this counts.
        """
        max_output = self._output_token_cap()
        canary = seeded_canary(0, case.id)  # every canary is the same length
        _, message = make_vulnbot(case, self._backend, self._channel, canary=canary)
        tools = channel_tools(self._channel, case.documents)
        system = build_system_prompt(canary, [tool.name for tool in tools])
        specs = json.dumps([tool.spec for tool in tools])
        first = estimate_tokens(system + message + specs) + _TEMPLATE_TOKENS
        # What each round adds to the history: the model's reply, then the tool results.
        results = (tool.run({"to": "someone@acme.example"}) for tool in tools)
        growth = max_output + sum(estimate_tokens(text) for text in results)
        # Both channels have at least one tool, so both can loop.
        calls = self._max_tool_rounds + 1
        return AttemptEstimate(
            requests=calls,
            # The whole history is re-sent each call: first, first + growth, first + 2*growth...
            input_tokens=calls * first + growth * calls * (calls - 1) // 2,
            output_tokens=calls * max_output,
            # Same rule as `_detectors`: the judge rules on attacks only.
            judge=(
                judge_estimate(self._judge)
                if self._judge is not None and isinstance(case, Attack)
                else None
            ),
        )

    def make_trial(self, attack: Case, seed: int) -> Trial:
        # The same bot for attacks and tasks: it holds a canary either way, so a utility score
        # describes exactly the setup that was attacked.
        canary = seeded_canary(seed, attack.id)
        bot, message = make_vulnbot(
            attack,
            self._backend.model_copy(update={"seed": seed}),
            self._channel,
            canary=canary,
            client=self._client,
            max_tool_rounds=self._max_tool_rounds,
            limiter=self._limiter,
        )
        return Trial(
            target=bot,
            message=message,
            detectors=self._detectors(attack, canary),
            delivered=lambda response: was_delivered(attack, self._channel, response),
        )
