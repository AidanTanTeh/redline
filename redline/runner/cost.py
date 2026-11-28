"""Estimate, before anything is sent, how big and expensive a run could get.

These are deliberately high estimates, not guarantees: token counts are guessed from text length,
and retries are not included. After a run, the real figure comes from each response's `usage`.
"""

import json
import math
from collections.abc import Sequence
from typing import Protocol, Self, runtime_checkable

from pydantic import Field

from redline.core import Case, RedlineModel
from redline.detectors import JudgeConfig
from redline.runner.report import RunReport
from redline.runner.trial import RunConfig, TrialFactory, can_run

# English averages about 4 characters per token. Dividing by 3 guesses high on purpose.
CHARS_PER_TOKEN = 3


def estimate_tokens(text: str) -> int:
    """A rough, high-side token count that needs no model-specific tokenizer."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


class Pricing(RedlineModel):
    """US dollars per million tokens. The caller supplies it: prices change too often to bundle."""

    input_per_mtok: float = Field(ge=0, allow_inf_nan=False)
    output_per_mtok: float = Field(ge=0, allow_inf_nan=False)

    @classmethod
    def free(cls) -> Self:
        """For local models such as Ollama."""
        return cls(input_per_mtok=0, output_per_mtok=0)

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        total = input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        return total / 1_000_000


class AttemptEstimate(RedlineModel):
    """The most one attempt is expected to use, over all its model calls."""

    requests: int = Field(ge=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    # False when only the number of requests is known. The token figures are then meaningless
    # zeros, and a cost cap must be refused.
    tokens_known: bool = True
    # The LLM judge's calls, if one rules on this case. Kept apart from the fields above (which
    # are the target's) because the judge has its own price and is not retried.
    judge: "AttemptEstimate | None" = None


def judge_estimate(judge: JudgeConfig) -> AttemptEstimate:
    """Worst case for one verdict: a single call that fills the judge's context window."""
    return AttemptEstimate(
        requests=1,
        # The judge refuses to send anything that might not fit, so this is a true ceiling.
        input_tokens=judge.context_tokens - judge.max_output_tokens,
        output_tokens=judge.max_output_tokens,
    )


@runtime_checkable  # so the runner can ask `isinstance` before enforcing a cap
class EstimatingFactory(TrialFactory, Protocol):
    """A factory that can also size an attempt. Optional: running doesn't need it."""

    def estimate_attempt(self, case: Case) -> AttemptEstimate:
        """Worst case for one attempt, worked out without calling the model."""
        ...


class RunEstimate(RedlineModel):
    """Worst-case totals for a whole run, without retries."""

    # Attempts that will actually be sent (skipped ones cost nothing).
    attempts: int = Field(ge=0)
    # Every model call, the judge's included. A request cap is compared with this.
    requests: int = Field(ge=0)
    # The target's tokens; the judge's are listed separately below.
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    judge_requests: int = Field(default=0, ge=0)
    judge_input_tokens: int = Field(default=0, ge=0)
    judge_output_tokens: int = Field(default=0, ge=0)
    # Target and judge together.
    cost_usd: float = Field(ge=0)
    # False: don't show the token or cost figures, they are not known.
    tokens_known: bool = True
    # None when there is no earlier measurement to base it on.
    seconds: float | None = Field(default=None, ge=0)


def estimate_run(
    cases: Sequence[Case],
    factory: EstimatingFactory,
    config: RunConfig,
    pricing: Pricing,
    *,
    seconds_per_attempt: float | None = None,
    parallel_requests: int = 1,
    requests_per_minute: float | None = None,
    judge_pricing: Pricing | None = None,
) -> RunEstimate:
    """Add up the worst case for every attempt `run_attacks` would send with these settings.

    `parallel_requests` is how many requests the SERVER really works on at once. It defaults to
    1 because a server that queues (local Ollama) gets no faster from `max_concurrency`; it
    never counts for more than `max_concurrency`. With a rate limit, the time is at least what
    the worst-case number of requests needs to get through it (the target's requests only: a
    judge has its own limit).

    `judge_pricing` is required when a judge is involved: it is never assumed to be free.
    """
    if parallel_requests < 1:
        raise ValueError("parallel_requests must be at least 1")
    if requests_per_minute is not None and not requests_per_minute > 0:
        raise ValueError("requests_per_minute must be positive")
    sent = [case for case in cases if can_run(factory, case)]
    per_case = [factory.estimate_attempt(case) for case in sent]
    attempts = len(sent) * config.repeats
    requests = sum(e.requests for e in per_case) * config.repeats
    input_tokens = sum(e.input_tokens for e in per_case) * config.repeats
    output_tokens = sum(e.output_tokens for e in per_case) * config.repeats
    judged = [e.judge for e in per_case if e.judge is not None]
    if judged and judge_pricing is None:
        raise ValueError("pass judge_pricing (Pricing.free() for a local judge)")
    judge_requests = sum(e.requests for e in judged) * config.repeats
    judge_input = sum(e.input_tokens for e in judged) * config.repeats
    judge_output = sum(e.output_tokens for e in judged) * config.repeats
    judge_cost = 0.0 if judge_pricing is None else judge_pricing.cost_usd(judge_input, judge_output)
    seconds = None
    if seconds_per_attempt is not None:
        seconds = attempts * seconds_per_attempt / min(parallel_requests, config.max_concurrency)
        if requests_per_minute is not None:
            seconds = max(seconds, requests * 60 / requests_per_minute)
    return RunEstimate(
        attempts=attempts,
        requests=requests + judge_requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        judge_requests=judge_requests,
        judge_input_tokens=judge_input,
        judge_output_tokens=judge_output,
        cost_usd=pricing.cost_usd(input_tokens, output_tokens) + judge_cost,
        tokens_known=all(e.tokens_known for e in per_case),
        seconds=seconds,
    )


def measured_seconds_per_attempt(report: RunReport, factory: TrialFactory) -> float | None:
    """Average time per attempt in an earlier run of the same target setup.

    The target's reply time, except for a judged run: judge time isn't recorded per call, so
    its whole duration is divided by the answered attempts.
    None if that run used different settings (its speed would mislead) or got no responses.
    """

    def canonical(description: object) -> str:
        # Compared as JSON because a saved report has lists where a live description has tuples.
        return json.dumps(description, sort_keys=True)

    if canonical(report.target) != canonical(dict(factory.describe())):
        return None
    times = [r.response.latency_ms / 1000 for r in report.results if r.response is not None]
    if not times:
        return None
    if "judge" in report.target:
        return (report.finished_at - report.started_at).total_seconds() / len(times)
    return sum(times) / len(times)
