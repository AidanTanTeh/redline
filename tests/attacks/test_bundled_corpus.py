import re
from collections import Counter

from redline.attacks import load_corpus, render_prompt
from redline.core import AttackCategory
from redline.demo import CANARY_PREFIX

# Categories shipped so far. Excessive agency arrives in 2.4.
EXPECTED_NOW = {
    AttackCategory.DIRECT_INJECTION,
    AttackCategory.INDIRECT_INJECTION,
    AttackCategory.SYSTEM_PROMPT_LEAKAGE,
    AttackCategory.SENSITIVE_INFO_DISCLOSURE,
}


def test_bundled_corpus_covers_each_current_category() -> None:
    counts = Counter(a.category for a in load_corpus().attacks)
    for category in EXPECTED_NOW:
        assert counts[category] >= 1, f"no attacks for {category}"


def test_bundled_ids_are_unique_and_sent_texts_are_not() -> None:
    attacks = load_corpus().attacks
    assert len({a.id for a in attacks}) == len(attacks)
    # Two attacks that send the same text would be double-counted in ASR. Compare what is
    # actually sent, not `prompt`: indirect attacks may share an innocent question and differ
    # only in their documents.
    assert len({render_prompt(a) for a in attacks}) == len(attacks)


def test_each_file_only_holds_its_own_category() -> None:
    # Keeps files tidy: spl-* ids are system prompt leakage, sid-* sensitive info, di-* direct.
    prefixes = {
        "di-": AttackCategory.DIRECT_INJECTION,
        "ii-": AttackCategory.INDIRECT_INJECTION,
        "spl-": AttackCategory.SYSTEM_PROMPT_LEAKAGE,
        "sid-": AttackCategory.SENSITIVE_INFO_DISCLOSURE,
    }
    for attack in load_corpus().attacks:
        matches = [c for p, c in prefixes.items() if attack.id.startswith(p)]
        assert matches == [attack.category], f"{attack.id} has the wrong id prefix or category"


def test_every_indirect_attack_has_an_innocent_question_and_a_document() -> None:
    indirect = [a for a in load_corpus().attacks if a.category is AttackCategory.INDIRECT_INJECTION]
    assert indirect
    for attack in indirect:
        assert attack.documents, attack.id
        assert "password" not in attack.prompt.lower(), f"{attack.id}: question must be innocent"


def test_no_attack_text_contains_the_canary_prefix() -> None:
    # If a payload contained the canary text, a bot that merely quoted the document would
    # look like it leaked. The corpus must never know the secret.
    for attack in load_corpus().attacks:
        texts = [attack.prompt, *(d.content for d in attack.documents)]
        assert not any(CANARY_PREFIX.lower() in t.lower() for t in texts), attack.id


def test_any_url_in_the_corpus_uses_the_reserved_example_domain() -> None:
    # Safety: payloads must never point at a real server (RFC 2606 reserves `.example`).
    host = re.compile(r"https?://([^/\s)?#]+)")
    for attack in load_corpus().attacks:
        texts = [attack.prompt, *(d.content for d in attack.documents)]
        for text in texts:
            for found in host.findall(text):
                assert found.endswith(".example"), f"{attack.id}: real-looking host {found}"
