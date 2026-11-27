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
from redline.metrics import run_warnings


def attack(attack_id: str, category: AttackCategory) -> Attack:
    return Attack(id=attack_id, name="x", category=category, prompt="p")


DIRECT = attack("d", AttackCategory.DIRECT_INJECTION)
LEAK = attack("l", AttackCategory.SYSTEM_PROMPT_LEAKAGE)


def result(
    attack_id: str = "d",
    *,
    reason: str | None = "stop",
    detectors: tuple[tuple[str, bool], ...] = (("canary", False),),
) -> RunResult:
    detections = tuple(DetectionResult(detector=n, triggered=hit) for n, hit in detectors)
    leaked = any(hit for _, hit in detectors)
    return RunResult(
        attack_id=attack_id,
        attempt=0,
        target="t",
        outcome=Outcome.SUCCEEDED if leaked else Outcome.RESISTED,
        response=TargetResponse(text="x", model="m", latency_ms=1, finish_reason=reason),
        detections=detections,
        model="m",
        config_hash="h",
    )


LEAKED = (("canary", True),)


def test_a_clean_vulnbot_run_has_no_warnings() -> None:
    assert run_warnings([result(reason="STOP")], [DIRECT], external_app=False) == ()


def test_missing_finish_reasons_are_counted_among_resisted_replies_only() -> None:
    results = [result(reason=None), result(), result(reason=None, detectors=LEAKED)]
    (warning,) = run_warnings(results, [DIRECT], external_app=False)
    assert warning.startswith("1 of 2 resisted replies had no finish reason")


def test_unrecognised_finish_reasons_are_listed() -> None:
    results = [result(reason="SAFETY"), result(reason="SAFETY"), result(reason="end_turn")]
    (warning,) = run_warnings(results, [DIRECT], external_app=False)
    assert "'SAFETY' x2" in warning
    assert "end_turn" not in warning


def test_an_app_whose_canary_never_appears_is_flagged() -> None:
    (warning,) = run_warnings([result()], [DIRECT], external_app=True)
    assert "canary never appeared" in warning


def test_one_canary_leak_shows_it_was_planted() -> None:
    results = [result(), result(detectors=LEAKED)]
    assert run_warnings(results, [DIRECT], external_app=True) == ()


def test_leakage_attacks_without_phrase_checks_are_flagged_per_attack() -> None:
    results = [result("l"), result("l")]
    (warning,) = run_warnings(results, [LEAK], external_app=False)
    assert warning.startswith("1 system prompt leakage attack(s)")


def test_phrase_checked_leakage_attacks_are_not_flagged() -> None:
    checked = result("l", detectors=(("canary", False), ("prompt_leak", False)))
    assert run_warnings([checked], [LEAK], external_app=False) == ()


def test_benign_tasks_and_empty_runs_produce_nothing() -> None:
    task = BenignTask(id="d", name="x", prompt="p", kind=TaskKind.ANSWER, expected_text=(("yes",),))
    assert run_warnings([result(reason=None)], [task], external_app=True) == ()
    assert run_warnings([], [DIRECT], external_app=True) == ()
