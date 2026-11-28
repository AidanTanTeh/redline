import re
from collections import Counter

from redline.attacks import bundled_mutators, load_corpus, make_variants, render_prompt
from redline.attacks.protect import protected_pattern
from redline.core import Attack, AttackCategory
from redline.demo import CANARY_PREFIX, search_docs_tool, send_email_tool


def every_variant() -> tuple[Attack, ...]:
    """Every mutator that ships, applied to every corpus attack it can handle."""
    return make_variants(load_corpus().attacks, bundled_mutators())


def everything_sent() -> tuple[Attack, ...]:
    return (*load_corpus().attacks, *every_variant())


def everything_an_attack_says(attack: Attack) -> list[str]:
    """Every piece of text in an attack that could contain a URL or address."""
    wanted = attack.success_tool_call.arguments.values() if attack.success_tool_call else ()
    return [attack.prompt, *(d.content for d in attack.documents), *wanted]


def test_bundled_corpus_covers_every_category() -> None:
    counts = Counter(a.category for a in load_corpus().attacks)
    for category in AttackCategory:
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
        "ea-": AttackCategory.EXCESSIVE_AGENCY,
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
    for attack in everything_sent():
        texts = [attack.prompt, *(d.content for d in attack.documents)]
        assert not any(CANARY_PREFIX.lower() in t.lower() for t in texts), attack.id


def test_any_url_or_email_address_uses_the_reserved_example_domain() -> None:
    # Safety: payloads must never point at a real server or mailbox (RFC 2606 reserves
    # `.example`). Covers URLs, email addresses, and the addresses attacks wait for.
    url_host = re.compile(r"https?://([^/\s)?#]+)")
    email_host = re.compile(r"[\w.+-]+@([\w.-]+)")
    for attack in everything_sent():
        for text in everything_an_attack_says(attack):
            for found in [*url_host.findall(text), *email_host.findall(text)]:
                host = found.rstrip(".")
                assert host.endswith(".example"), f"{attack.id}: real-looking host {found}"


def test_every_variant_keeps_what_must_stay_readable() -> None:
    # Addresses, links, tool names and placeholders. base64 is the one exception: it encodes
    # everything, and the target has to decode them itself.
    variants = every_variant()
    assert len({v.id for v in variants}) == len(variants)
    for variant in variants:
        assert variant.variant_of is not None
        if variant.variant_of.mutator == "base64":
            continue
        base = variant.variant_of.base
        sent = "\n".join([variant.prompt, *(d.content for d in variant.documents)])
        for text in [base.prompt, *(d.content for d in base.documents)]:
            for match in protected_pattern(base).finditer(text):
                # Payload splitting quotes its pieces, so a quote mark arrives escaped.
                kept = match.group()
                assert kept in sent or kept.replace('"', '\\"') in sent, (variant.id, kept)


def test_every_agency_attack_names_a_tool_vulnbot_really_has() -> None:
    # A typo like `send_mail` would make the attack impossible to win and quietly lower ASR.
    tools = {send_email_tool().name, search_docs_tool(()).name}
    agency = [a for a in load_corpus().attacks if a.category is AttackCategory.EXCESSIVE_AGENCY]
    assert agency
    for attack in agency:
        assert attack.success_tool_call is not None, attack.id
        assert attack.success_tool_call.tool in tools, attack.id
        assert attack.needs_tools, attack.id


def test_each_success_address_is_unique_so_one_attack_cannot_trigger_another() -> None:
    wanted = [
        value
        for a in load_corpus().attacks
        if a.success_tool_call
        for value in a.success_tool_call.arguments.values()
    ]
    assert len(set(wanted)) == len(wanted)
