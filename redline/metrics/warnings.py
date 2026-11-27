"""Point out ways a run's results could make a target look safer than it is.

These are things Redline can't fix or decide on its own, so a person has to check them.
"""

from collections import Counter
from collections.abc import Sequence

from redline.core import Attack, AttackCategory, Case, Outcome, RunResult

# Reasons that mean the model finished normally (compared ignoring case).
_NORMAL_REASONS = frozenset({"stop", "tool_calls", "function_call", "end_turn"})
# Categories judged by looking for the canary.
_CANARY_CATEGORIES = (
    AttackCategory.DIRECT_INJECTION,
    AttackCategory.INDIRECT_INJECTION,
    AttackCategory.SYSTEM_PROMPT_LEAKAGE,
    AttackCategory.SENSITIVE_INFO_DISCLOSURE,
)


def run_warnings(
    results: Sequence[RunResult], cases: Sequence[Case], *, external_app: bool
) -> tuple[str, ...]:
    """Warnings about the attacks in a run. Benign tasks produce none.

    `external_app` is True for your own function or web app, where Redline can't check that the
    canary was planted. VulnBot plants its own.
    """
    attacks = {case.id: case for case in cases if isinstance(case, Attack)}
    answered = [r for r in results if r.response is not None and r.attack_id in attacks]
    warnings: list[str] = []

    # Only attempts that were counted as holding up can be wrong because of a stop reason.
    resisted = [r for r in answered if r.outcome is Outcome.RESISTED]
    reasons = [r.response.finish_reason for r in resisted if r.response is not None]
    missing = sum(reason is None for reason in reasons)
    if missing:
        warnings.append(
            f"{missing} of {len(resisted)} resisted replies had no finish reason, so a "
            "cut-off or blocked reply among them can't be told apart from a real refusal."
        )
    unusual = Counter(
        reason for reason in reasons if reason and reason.casefold() not in _NORMAL_REASONS
    )
    if unusual:
        listed = ", ".join(f"{reason!r} x{count}" for reason, count in sorted(unusual.items()))
        warnings.append(
            f"unrecognised finish reasons were counted as resisted: {listed}. If any means "
            "cut off or blocked, add it to cut_off_values or blocked_values."
        )

    canary_judged = [r for r in answered if attacks[r.attack_id].category in _CANARY_CATEGORIES]
    seen = any(d.detector == "canary" and d.triggered for r in answered for d in r.detections)
    if external_app and canary_judged and not seen:
        warnings.append(
            "the canary never appeared in any reply. Either the app held up, or the canary "
            "isn't planted in it: check that before trusting these results."
        )

    # An attack counts as covered if any of its attempts ran with the phrase detector.
    leakage = {
        r.attack_id
        for r in answered
        if attacks[r.attack_id].category is AttackCategory.SYSTEM_PROMPT_LEAKAGE
    }
    with_phrases = {
        r.attack_id for r in answered if any(d.detector == "prompt_leak" for d in r.detections)
    }
    uncovered = len(leakage - with_phrases)
    if uncovered:
        warnings.append(
            f"{uncovered} system prompt leakage attack(s) were judged by the canary only: a "
            "leak that quotes the prompt without the canary is missed. Set leak_phrases."
        )
    return tuple(warnings)
