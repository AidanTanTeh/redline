"""Add up what a finished run really used, as far as it was recorded."""

from collections.abc import Sequence

from redline.core import Outcome, RedlineModel, RunResult
from redline.detectors import LlmJudge

# Never sent, so they used nothing and nothing is missing for them.
_NEVER_SENT = (Outcome.SKIPPED, Outcome.NOT_RUN)


class UsageTotal(RedlineModel):
    """Recorded usage for a run. If anything is unrecorded, the real figures are higher."""

    # Sums of what was recorded for the TARGET.
    requests: int
    input_tokens: int
    output_tokens: int
    # Attempts that were sent; those whose usage is missing or partly unknown.
    attempts: int
    unrecorded_attempts: int
    # Failed tries before an answer. Their usage is never recorded.
    failed_tries: int
    # The LLM judge's recorded usage, kept apart: it has its own model and price.
    judge_requests: int
    judge_input_tokens: int
    judge_output_tokens: int
    # Verdicts with missing or partly unknown usage (e.g. the judge call failed).
    unrecorded_verdicts: int

    @property
    def complete(self) -> bool:
        return not (self.unrecorded_attempts or self.failed_tries or self.unrecorded_verdicts)

    @property
    def judged(self) -> bool:
        return bool(self.judge_requests or self.unrecorded_verdicts)


def total_usage(results: Sequence[RunResult]) -> UsageTotal:
    requests = input_tokens = output_tokens = unrecorded = failed_tries = 0
    sent = [r for r in results if r.outcome not in _NEVER_SENT]
    for result in sent:
        usage = result.response.usage if result.response is not None else None
        if result.response is not None:
            failed_tries += result.retries
        if usage is None:
            unrecorded += 1
            continue
        requests += usage.requests
        if usage.input_tokens is None or usage.output_tokens is None:
            unrecorded += 1
        input_tokens += usage.input_tokens or 0
        output_tokens += usage.output_tokens or 0

    judge_requests = judge_input = judge_output = unrecorded_verdicts = 0
    verdicts = [d for r in sent for d in r.detections if d.detector == LlmJudge.name]
    for verdict in verdicts:
        spent = verdict.usage
        if spent is None:
            unrecorded_verdicts += 1
            continue
        judge_requests += spent.requests
        if spent.input_tokens is None or spent.output_tokens is None:
            unrecorded_verdicts += 1
        judge_input += spent.input_tokens or 0
        judge_output += spent.output_tokens or 0

    return UsageTotal(
        judge_requests=judge_requests,
        judge_input_tokens=judge_input,
        judge_output_tokens=judge_output,
        unrecorded_verdicts=unrecorded_verdicts,
        requests=requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        attempts=len(sent),
        unrecorded_attempts=unrecorded,
        failed_tries=failed_tries,
    )
