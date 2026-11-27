"""Plug YOUR app into the runner: a Python function or a web endpoint."""

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Literal

import httpx

from redline.attacks import render_prompt
from redline.core import Attack, AttackCategory, BenignTask, Case, Document, TargetResponse
from redline.detectors import (
    CANARY_PREFIX,
    CanaryDetector,
    Detector,
    RegexDetector,
    TaskDetector,
    ToolCallDetector,
    seeded_canary,
    unknown_to_attacker,
)
from redline.runner.cost import AttemptEstimate
from redline.runner.retry import is_rate_limit, never_sent
from redline.runner.trial import Trial
from redline.targets import (
    AgentFunction,
    HttpTemplateConfig,
    HttpTemplateTarget,
    PythonCallableTarget,
    RateLimiter,
    TrialContext,
)
from redline.targets.secrets import read_secret

DocumentDelivery = Literal["prompt", "context"]

# A made-up attack, rendered by describe() so that a change to HOW documents are shown also
# changes the config hash.
_PROBE = Attack(
    id="probe",
    name="probe",
    category=AttackCategory.INDIRECT_INJECTION,
    prompt="probe question",
    documents=(Document(title="probe title", content="probe content"),),
)


def _sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


class _AppFactory:
    """What every app target shares: which cases apply, how they're judged, the request cap."""

    def __init__(
        self,
        *,
        tool_names: Sequence[str],
        leak_phrases: Sequence[str],
        max_calls_per_attempt: int | None,
    ) -> None:
        if isinstance(tool_names, str) or isinstance(leak_phrases, str):
            # A str is also a sequence of str, so "abc" would quietly mean "a", "b", "c".
            raise TypeError("tool_names and leak_phrases must be lists, not one string")
        if max_calls_per_attempt is not None and max_calls_per_attempt < 1:
            raise ValueError("max_calls_per_attempt must be at least 1")
        self._tool_names = tuple(sorted(name.casefold() for name in tool_names))
        self._leak_phrases = tuple(leak_phrases)
        self._max_calls = max_calls_per_attempt

    @property
    def has_tools(self) -> bool:
        return bool(self._tool_names)

    def supports(self, case: Case) -> bool:
        """False for a case that needs a tool this app doesn't have: it could never succeed."""
        if isinstance(case, Attack):
            wanted = case.success_tool_call
        elif isinstance(case, BenignTask):
            wanted = case.expected_tool_call
        else:
            wanted = None
        return wanted is None or wanted.tool.casefold() in self._tool_names

    def estimate_attempt(self, case: Case) -> AttemptEstimate:
        """Only a request count: Redline can't see how many tokens your app uses."""
        if self._max_calls is None:
            raise ValueError("set max_calls_per_attempt to estimate or cap a run")
        return AttemptEstimate(
            requests=self._max_calls, input_tokens=0, output_tokens=0, tokens_known=False
        )

    def _detectors(self, case: Case, canary: str) -> tuple[Detector, ...]:
        if isinstance(case, BenignTask):
            return (TaskDetector(),)
        if not isinstance(case, Attack):
            raise TypeError(f"can't judge a {type(case).__name__}")
        detectors: list[Detector] = [
            CanaryDetector(
                canary,
                # Without this, characters everyone knows would count towards a partial leak.
                public_prefix=CANARY_PREFIX if canary.startswith(CANARY_PREFIX) else "",
                scan_tool_calls=True,
            ),
            ToolCallDetector(),
        ]
        if case.category is AttackCategory.SYSTEM_PROMPT_LEAKAGE:
            phrases = unknown_to_attacker(self._leak_phrases, case)
            if phrases:
                detectors.append(
                    RegexDetector.from_phrases("prompt_leak", phrases, scan_tool_calls=True)
                )
        return tuple(detectors)

    def _describe_judging(self, canary: str | None) -> dict[str, Any]:
        # Hashes, not the values: the canary and phrases come from your system prompt.
        return {
            "canary": "seeded" if canary is None else _sha256(canary),
            "tool_names": list(self._tool_names),
            "leak_phrases_sha256": _sha256(list(self._leak_phrases)),
        }


class FunctionFactory(_AppFactory):
    """A `TrialFactory` for your own Python function.

    Never retried unless `safe_to_retry=True` (set it only when every tool is mocked).

    Documents of an indirect attack are pasted into the prompt by default. With
    `documents="context"` your function gets them in `context.documents` and must feed them to
    your retriever. Name that tool as `retrieval_tool` so attempts where it was never called
    are counted as "not delivered"; without it Redline trusts that they were read.
    """

    def __init__(
        self,
        function: AgentFunction,
        *,
        version: str,
        name: str = "python",
        model: str = "unknown",
        canary: str | None = None,
        tool_names: Sequence[str] = (),
        leak_phrases: Sequence[str] = (),
        documents: DocumentDelivery = "prompt",
        retrieval_tool: str | None = None,
        max_calls_per_attempt: int | None = None,
        safe_to_retry: bool = False,
    ) -> None:
        super().__init__(
            tool_names=tool_names,
            leak_phrases=leak_phrases,
            max_calls_per_attempt=max_calls_per_attempt,
        )
        if not version.strip():
            # Redline can't see your code, so this is what marks two runs as different.
            raise ValueError("give your app a version label, and change it when the app changes")
        if retrieval_tool is not None and documents != "context":
            raise ValueError('retrieval_tool only applies with documents="context"')
        if canary is not None:
            CanaryDetector(canary, public_prefix="")  # fail now if it is too short
        self._function = function
        self._version = version
        self._name = name
        self._model = model
        self._canary = canary
        self._documents: DocumentDelivery = documents
        self._retrieval_tool = retrieval_tool
        self.safe_to_retry = safe_to_retry

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    def describe(self) -> dict[str, Any]:
        return {
            "target": "python",
            "name": self._name,
            "version": self._version,
            "model": self._model,
            "documents": self._documents,
            "retrieval_tool": self._retrieval_tool,
            "probe_message": self._message(_PROBE),
            **self._describe_judging(self._canary),
        }

    def preflight(self) -> None:
        pass  # nothing can be checked without calling your function

    def _message(self, case: Case) -> str:
        return render_prompt(case) if self._documents == "prompt" else case.prompt

    def make_trial(self, attack: Case, seed: int) -> Trial:
        canary = self._canary or seeded_canary(seed, attack.id)
        in_context = self._documents == "context"
        context = TrialContext(
            canary=canary,
            # Never both ways at once.
            documents=attack.documents if in_context else (),
            seed=seed,
        )
        target = PythonCallableTarget(
            self._function,
            context,
            name=self._name,
            model=self._model,
            require_full_response=bool(self._tool_names or self._retrieval_tool),
        )

        def delivered(response: TargetResponse) -> bool:
            if self._retrieval_tool is None or not attack.documents:
                return True
            wanted = self._retrieval_tool.casefold()
            return any(call.name.casefold() == wanted for call in response.tool_calls)

        return Trial(
            target=target,
            message=self._message(attack),
            detectors=self._detectors(attack, canary),
            delivered=delivered,
        )


class HttpFactory(_AppFactory):
    """A `TrialFactory` for a web endpoint.

    `canary` is the secret YOU planted in the app (system prompt or data). It must be random
    text the model couldn't guess, at least 8 letters or digits.

    By default only failures where the request never left are retried. `retry_rate_limits=True`
    adds 429s: set it if your app rejects those before doing any work.
    """

    def __init__(
        self,
        config: HttpTemplateConfig,
        *,
        canary: str,
        tool_names: Sequence[str] = (),
        leak_phrases: Sequence[str] = (),
        max_calls_per_attempt: int | None = None,
        retry_rate_limits: bool = False,
        safe_to_retry: bool = False,
        client: httpx.AsyncClient | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        super().__init__(
            tool_names=tool_names,
            leak_phrases=leak_phrases,
            max_calls_per_attempt=max_calls_per_attempt,
        )
        if tool_names and config.tool_calls_path is None:
            raise ValueError(
                "tool_names needs tool_calls_path in the config: without it no tool call is "
                "seen, and every tool attack would look resisted"
            )
        CanaryDetector(canary, public_prefix="")  # fail now if it is too short
        if canary.casefold() in config.model_dump_json().casefold():
            # The request would hand the app its own secret, and an echo would count as a leak.
            raise ValueError("the canary must not appear in the request config")
        self._config = config
        self._canary = canary
        self._retry_rate_limits = retry_rate_limits
        self._client = client
        self._limiter = limiter
        self.safe_to_retry = safe_to_retry

    @property
    def name(self) -> str:
        # The host only: names are saved with every result.
        return f"http/{httpx.URL(self._config.url).host}"

    @property
    def model(self) -> str:
        return self._config.model

    def can_retry(self, error: Exception) -> bool:
        return never_sent(error) or (self._retry_rate_limits and is_rate_limit(error))

    def describe(self) -> dict[str, Any]:
        # The settings are saved as a fingerprint, so a secret that slipped into the url or
        # body is never written to a report. Equal fingerprints still mean equal settings.
        settings = self._config.model_dump(mode="json", exclude={"timeout_s", "authorized"})
        return {
            "target": "http",
            "host": httpx.URL(self._config.url).host,
            "model": self._config.model,
            "config_sha256": _sha256(settings),
            "probe_message": render_prompt(_PROBE),
            **self._describe_judging(self._canary),
        }

    def preflight(self) -> None:
        for name in self._config.env_names:
            read_secret(name)  # checked, then thrown away

    def make_trial(self, attack: Case, seed: int) -> Trial:
        target = HttpTemplateTarget(self._config, self._client, limiter=self._limiter)
        return Trial(
            target=target,
            message=render_prompt(attack),
            detectors=self._detectors(attack, self._canary),
        )
