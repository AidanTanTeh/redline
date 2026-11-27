from redline.core import Outcome, RunResult, TargetResponse, Usage
from redline.metrics import total_usage


def answered(usage: Usage | None, retries: int = 0) -> RunResult:
    return RunResult(
        attack_id="a",
        attempt=0,
        target="t",
        outcome=Outcome.RESISTED,
        response=TargetResponse(text="no", model="m", latency_ms=1, usage=usage),
        retries=retries,
        model="m",
        config_hash="h",
    )


def unanswered(outcome: Outcome, retries: int = 0) -> RunResult:
    return RunResult(
        attack_id="a",
        attempt=0,
        target="t",
        outcome=outcome,
        error="boom" if outcome is Outcome.ERROR else None,
        retries=retries,
        model="m",
        config_hash="h",
    )


def test_recorded_usage_is_added_up() -> None:
    total = total_usage(
        [
            answered(Usage(requests=1, input_tokens=100, output_tokens=10)),
            answered(Usage(requests=3, input_tokens=900, output_tokens=40)),
        ]
    )
    assert (total.requests, total.input_tokens, total.output_tokens) == (4, 1000, 50)
    assert total.attempts == 2
    assert total.complete


def test_attempts_never_sent_use_nothing_and_leave_nothing_missing() -> None:
    total = total_usage([unanswered(Outcome.SKIPPED), unanswered(Outcome.NOT_RUN)])
    assert (total.attempts, total.requests) == (0, 0)
    assert total.complete


def test_missing_and_partial_usage_is_flagged_but_known_parts_are_kept() -> None:
    total = total_usage(
        [
            answered(None),  # e.g. a result saved before usage was recorded
            answered(Usage(requests=2, input_tokens=50)),
            unanswered(Outcome.ERROR),
        ]
    )
    assert (total.requests, total.input_tokens, total.output_tokens) == (2, 50, 0)
    assert (total.attempts, total.unrecorded_attempts) == (3, 3)
    assert not total.complete


def test_a_retried_attempt_is_not_complete() -> None:
    usage = Usage(requests=1, input_tokens=1, output_tokens=1)
    total = total_usage([answered(usage, retries=2)])
    assert total.failed_tries == 2
    assert not total.complete
