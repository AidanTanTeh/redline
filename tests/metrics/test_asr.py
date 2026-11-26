import pytest

from redline.core import (
    Attack,
    AttackCategory,
    DetectionResult,
    Outcome,
    RunResult,
    TargetResponse,
)
from redline.metrics import AsrSummary, summarize

OK, NO = Outcome.SUCCEEDED, Outcome.RESISTED
DIRECT, LEAK = AttackCategory.DIRECT_INJECTION, AttackCategory.SYSTEM_PROMPT_LEAKAGE


def attack(attack_id: str, category: AttackCategory = DIRECT) -> Attack:
    return Attack(id=attack_id, name="x", category=category, prompt="p")


def result(attack_id: str, attempt: int, outcome: Outcome, config_hash: str = "h") -> RunResult:
    """A valid RunResult for any outcome (the model checks the fields agree)."""
    answered = outcome not in (Outcome.ERROR, Outcome.SKIPPED)
    hit = DetectionResult(detector="canary", triggered=True)
    return RunResult(
        attack_id=attack_id,
        attempt=attempt,
        target="t",
        outcome=outcome,
        response=TargetResponse(text="x", model="m", latency_ms=1) if answered else None,
        detections=(hit,) if outcome is OK else (),
        error="boom" if outcome is Outcome.ERROR else None,
        model="m",
        config_hash=config_hash,
    )


def results(attack_id: str, *outcomes: Outcome) -> list[RunResult]:
    return [result(attack_id, n, outcome) for n, outcome in enumerate(outcomes)]


def test_rate_counts_only_verdicts_and_keeps_every_bucket() -> None:
    run = results(
        "a",
        OK,
        NO,
        NO,
        NO,
        Outcome.NO_ANSWER,
        Outcome.NOT_DELIVERED,
        Outcome.ERROR,
        Outcome.SKIPPED,
    )
    overall = summarize(run, [attack("a")]).overall
    assert overall.asr == 0.25
    assert (overall.attempts, overall.decided, overall.coverage) == (8, 4, 0.5)
    assert overall.counts[Outcome.ERROR] == 1 and sum(overall.counts.values()) == 8
    # 7 attempts could have run: 1 succeeded, 3 undecided.
    assert (overall.lowest, overall.highest) == (1 / 7, 4 / 7)


def test_each_attack_weighs_the_same_however_many_attempts_it_has() -> None:
    run = results("a", OK) + results("b", NO, NO, NO, NO, NO, NO, NO, NO, NO)
    overall = summarize(run, [attack("a"), attack("b")]).overall
    assert overall.asr == 0.5  # not 1 out of 10
    assert (overall.attacks_measured, overall.attacks_broken) == (2, 1)


def test_interval_uses_the_number_of_attacks_so_repeats_do_not_narrow_it() -> None:
    attacks = [attack(f"a{i}") for i in range(6)]

    def run(repeats: int) -> AsrSummary:
        out: list[RunResult] = []
        for i, a in enumerate(attacks):
            out += results(a.id, *[OK if i < 3 else NO] * repeats)
        return summarize(out, attacks)

    once, many = run(1).overall, run(20).overall
    assert once.interval == many.interval
    assert once.interval is not None
    assert (round(once.interval.low, 3), round(once.interval.high, 3)) == (0.188, 0.812)


def test_broken_at_least_once_differs_from_the_average_rate() -> None:
    run = results("a", OK, NO, NO, NO) + results("b", OK, NO, NO, NO)
    overall = summarize(run, [attack("a"), attack("b")]).overall
    assert overall.asr == 0.25
    assert overall.attacks_broken == 2
    assert overall.broken_interval is not None and overall.broken_interval.high == 1.0


def test_no_verdict_means_no_rate_not_zero() -> None:
    overall = summarize(results("a", Outcome.ERROR, Outcome.ERROR), [attack("a")]).overall
    assert (overall.asr, overall.interval, overall.broken_interval) == (None, None, None)
    assert (overall.lowest, overall.highest) == (0.0, 1.0)  # anything was possible


def test_all_skipped_has_no_bounds_either() -> None:
    overall = summarize(results("a", Outcome.SKIPPED), [attack("a")]).overall
    assert (overall.asr, overall.lowest, overall.highest) == (None, None, None)


def test_unmeasured_attack_widens_the_bounds_but_not_the_rate() -> None:
    run = results("a", OK) + results("b", Outcome.NOT_DELIVERED)
    overall = summarize(run, [attack("a"), attack("b")]).overall
    assert (overall.asr, overall.attacks, overall.attacks_measured) == (1.0, 2, 1)
    assert (overall.lowest, overall.highest) == (0.5, 1.0)


def test_categories_are_separate_ordered_and_carry_owasp_ids() -> None:
    attacks = [attack("leak", LEAK), attack("direct"), attack("unused")]
    summary = summarize(results("leak", NO) + results("direct", OK), attacks)
    assert [(c.category, c.owasp_id, c.stats.asr) for c in summary.categories] == [
        (DIRECT, "LLM01:2026", 1.0),
        (LEAK, "LLM08:2026", 0.0),
    ]
    assert [a.attack_id for a in summary.attacks] == ["direct", "leak"]
    assert summary.overall.asr == 0.5
    assert summary.config_hash == "h"


def test_summary_round_trips_through_json() -> None:
    summary = summarize(results("a", OK, NO), [attack("a")])
    assert AsrSummary.model_validate_json(summary.model_dump_json()) == summary


@pytest.mark.parametrize(
    ("run", "message"),
    [
        ([], "no results"),
        ([result("a", 0, OK), result("a", 1, OK, "other")], "different setups"),
        ([result("a", 0, OK), result("a", 0, NO)], "more than once"),
        ([result("mystery", 0, OK)], "no attack definition"),
    ],
)
def test_refuses_to_mix_things_that_are_not_comparable(run: list[RunResult], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        summarize(run, [attack("a")])
