import pytest

from redline.core import (
    Attack,
    AttackCategory,
    BenignTask,
    DetectionResult,
    Outcome,
    RunResult,
    TargetResponse,
    TaskKind,
)
from redline.metrics import UtilitySummary, summarize, summarize_utility, wilson_interval

OK, NO = Outcome.SUCCEEDED, Outcome.RESISTED


def task(task_id: str, kind: TaskKind = TaskKind.ANSWER) -> BenignTask:
    return BenignTask(id=task_id, name="x", prompt="p", kind=kind, expected_text=(("yes",),))


def result(task_id: str, attempt: int, outcome: Outcome, config_hash: str = "h") -> RunResult:
    """A valid RunResult for any outcome (the model checks the fields agree)."""
    answered = outcome not in (Outcome.ERROR, Outcome.SKIPPED, Outcome.NOT_RUN)
    done = DetectionResult(detector="task", triggered=True)
    return RunResult(
        attack_id=task_id,
        attempt=attempt,
        target="t",
        outcome=outcome,
        response=TargetResponse(text="x", model="m", latency_ms=1) if answered else None,
        detections=(done,) if outcome is OK else (),
        error="boom" if outcome is Outcome.ERROR else None,
        model="m",
        config_hash=config_hash,
    )


def results(task_id: str, *outcomes: Outcome) -> list[RunResult]:
    return [result(task_id, n, outcome) for n, outcome in enumerate(outcomes)]


def test_a_pass_without_reading_the_document_does_not_count() -> None:
    guessed = result("a", 0, OK).model_copy(update={"delivered": False})
    read = result("a", 1, OK).model_copy(update={"delivered": True})

    overall = summarize_utility([guessed, read], [task("a")]).overall

    assert overall.utility == 0.5
    assert overall.counts[OK] == 1
    assert overall.counts[Outcome.NOT_DELIVERED] == 1
    assert guessed.outcome is OK  # the saved result keeps what the checker said


def test_every_way_of_not_helping_the_user_counts_as_a_failure() -> None:
    run = results("a", OK, NO, Outcome.NOT_DELIVERED, Outcome.NO_ANSWER)

    summary = summarize_utility(run, [task("a")])

    assert summary.overall.scored == 4
    assert summary.overall.utility == 0.25


def test_errors_and_skips_are_left_out_but_still_counted() -> None:
    run = results("a", OK, Outcome.ERROR, Outcome.SKIPPED)

    overall = summarize_utility(run, [task("a")]).overall

    assert (overall.attempts, overall.scored) == (3, 1)
    assert overall.utility == 1.0
    assert overall.counts[Outcome.ERROR] == 1
    assert overall.counts[Outcome.SKIPPED] == 1


def test_attempts_a_cap_stopped_are_not_failures() -> None:
    overall = summarize_utility(results("a", OK, Outcome.NOT_RUN), [task("a")]).overall

    assert (overall.attempts, overall.scored) == (2, 1)
    assert overall.utility == 1.0


def test_each_task_weighs_the_same_however_many_attempts_it_has() -> None:
    run = results("a", OK, OK, OK) + results("b", NO)

    summary = summarize_utility(run, [task("a"), task("b")])

    assert summary.overall.utility == 0.5  # not 3/4


def test_interval_uses_the_number_of_tasks_so_repeats_do_not_narrow_it() -> None:
    tasks = [task("a"), task("b")]
    once = summarize_utility(results("a", OK) + results("b", NO), tasks)
    thrice = summarize_utility(results("a", OK, OK, OK) + results("b", NO, NO, NO), tasks)

    assert once.overall.interval == wilson_interval(0.5, 2)
    assert thrice.overall.interval == once.overall.interval


def test_failed_at_least_once_differs_from_the_average_rate() -> None:
    # Each task fails one attempt in three: mostly helpful, but neither is reliable.
    run = results("a", OK, OK, NO) + results("b", OK, OK, NO)

    overall = summarize_utility(run, [task("a"), task("b")]).overall

    assert overall.utility == pytest.approx(2 / 3)
    assert overall.tasks_failed == 2
    assert overall.failed_interval == wilson_interval(1.0, 2)


def test_nothing_scored_means_no_rate_not_zero() -> None:
    overall = summarize_utility(results("a", Outcome.ERROR), [task("a")]).overall

    assert overall.utility is None
    assert overall.interval is None
    assert overall.failed_interval is None
    assert (overall.tasks, overall.tasks_measured, overall.tasks_failed) == (1, 0, 0)


def test_kinds_are_separate_and_ordered() -> None:
    tasks = [task("a", TaskKind.SENSITIVE_SOUNDING), task("b", TaskKind.ANSWER)]
    run = results("a", NO) + results("b", OK)

    kinds = summarize_utility(run, tasks).kinds

    assert [(k.kind, k.stats.utility) for k in kinds] == [
        (TaskKind.ANSWER, 1.0),
        (TaskKind.SENSITIVE_SOUNDING, 0.0),
    ]


def test_summary_round_trips_through_json() -> None:
    summary = summarize_utility(results("a", OK, NO), [task("a")])

    assert UtilitySummary.model_validate_json(summary.model_dump_json()) == summary


@pytest.mark.parametrize(
    ("run", "message"),
    [
        ([], "no results"),
        ([result("a", 0, OK), result("a", 1, OK, config_hash="other")], "different setups"),
        ([result("a", 0, OK), result("a", 0, NO)], "more than once"),
        ([result("zzz", 0, OK)], "no task definition"),
    ],
)
def test_refuses_to_mix_things_that_are_not_comparable(run: list[RunResult], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_utility(run, [task("a")])


def test_a_task_run_cannot_be_scored_as_attacks() -> None:
    attacks = [Attack(id="di-x", name="x", category=AttackCategory.DIRECT_INJECTION, prompt="p")]

    with pytest.raises(ValueError, match="no attack definition"):
        summarize(results("a", OK), attacks)
