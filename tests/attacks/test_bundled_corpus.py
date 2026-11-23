from collections import Counter

from redline.attacks import load_corpus
from redline.core import AttackCategory

# Categories 2.1 ships. Indirect injection arrives in 2.3, excessive agency in 2.4.
EXPECTED_NOW = {
    AttackCategory.DIRECT_INJECTION,
    AttackCategory.SYSTEM_PROMPT_LEAKAGE,
    AttackCategory.SENSITIVE_INFO_DISCLOSURE,
}


def test_bundled_corpus_covers_each_current_category() -> None:
    counts = Counter(a.category for a in load_corpus().attacks)
    for category in EXPECTED_NOW:
        assert counts[category] >= 1, f"no attacks for {category}"


def test_bundled_ids_are_unique_and_prompts_are_not() -> None:
    attacks = load_corpus().attacks
    assert len({a.id for a in attacks}) == len(attacks)
    # Two attacks with the same prompt would be double-counted in ASR.
    assert len({a.prompt for a in attacks}) == len(attacks)


def test_each_file_only_holds_its_own_category() -> None:
    # Keeps files tidy: spl-* ids are system prompt leakage, sid-* sensitive info, di-* direct.
    prefixes = {
        "di-": AttackCategory.DIRECT_INJECTION,
        "spl-": AttackCategory.SYSTEM_PROMPT_LEAKAGE,
        "sid-": AttackCategory.SENSITIVE_INFO_DISCLOSURE,
    }
    for attack in load_corpus().attacks:
        matches = [c for p, c in prefixes.items() if attack.id.startswith(p)]
        assert matches == [attack.category], f"{attack.id} has the wrong id prefix or category"
