import re

import pytest

from redline.core import Attack, AttackCategory, TargetResponse
from redline.detectors import Detector, RegexDetector

ATTACK = Attack(id="a-01", name="x", category=AttackCategory.SYSTEM_PROMPT_LEAKAGE, prompt="p")
PROMPT_LINE = "You are HelpBot, a friendly customer-support assistant for Acme Corp."


def response(text: str, reasoning: str | None = None) -> TargetResponse:
    return TargetResponse(text=text, reasoning=reasoning, model="m", latency_ms=1.0)


def test_satisfies_the_detector_protocol() -> None:
    detector: Detector = RegexDetector("regex-test", [r"\bfoo\b"])
    assert detector.name == "regex-test"


async def test_matching_pattern_triggers_with_evidence() -> None:
    detector = RegexDetector("aws-key", [r"AKIA[0-9A-Z]{16}"])
    (result,) = await detector.detect(ATTACK, response("key: AKIAIOSFODNN7EXAMPLE ok"))
    assert result.triggered
    assert result.field == "text"
    assert result.detector == "aws-key"
    assert result.evidence is not None
    assert "AKIAIOSFODNN7EXAMPLE" in result.evidence


async def test_any_one_pattern_is_enough_and_none_means_no() -> None:
    detector = RegexDetector("multi", [r"alpha", r"beta"])
    assert (await detector.detect(ATTACK, response("only beta here")))[0].triggered
    assert not (await detector.detect(ATTACK, response("gamma")))[0].triggered


async def test_case_insensitive_by_default_and_configurable() -> None:
    assert (await RegexDetector("d", ["secret"]).detect(ATTACK, response("SECRET")))[0].triggered
    strict = RegexDetector("d", ["secret"], flags=re.NOFLAG)
    assert not (await strict.detect(ATTACK, response("SECRET")))[0].triggered


async def test_custom_confidence_is_reported_only_on_a_match() -> None:
    detector = RegexDetector("weak", ["foo"], confidence=0.5)
    hit, miss = (
        (await detector.detect(ATTACK, response("foo")))[0],
        (await detector.detect(ATTACK, response("bar")))[0],
    )
    assert hit.confidence == 0.5
    assert miss.confidence == 1.0


async def test_long_matches_are_cut_in_the_evidence() -> None:
    detector = RegexDetector("long", [r"x+"])
    (result,) = await detector.detect(ATTACK, response("x" * 500))
    assert result.evidence is not None
    assert len(result.evidence) < 200


async def test_reasoning_is_ignored_by_default_and_reported_separately_when_scanned() -> None:
    off = RegexDetector("d", ["leak"])
    assert [r.field for r in await off.detect(ATTACK, response("no", reasoning="leak"))] == ["text"]
    on = RegexDetector("d", ["leak"], scan_reasoning=True)
    results = await on.detect(ATTACK, response("no", reasoning="a leak"))
    assert [(r.field, r.triggered) for r in results] == [("text", False), ("reasoning", True)]


# --- from_phrases --------------------------------------------------------------------------


async def test_phrase_matches_even_when_rewrapped_or_recased() -> None:
    detector = RegexDetector.from_phrases("prompt-leak", [PROMPT_LINE])
    wrapped = "You are HelpBot,\na friendly   customer-support\nassistant for ACME Corp."
    assert (await detector.detect(ATTACK, response(wrapped)))[0].triggered


async def test_phrases_are_literal_not_regex() -> None:
    detector = RegexDetector.from_phrases("literal", ["a.b (c)"])
    assert (await detector.detect(ATTACK, response("see a.b (c) here")))[0].triggered
    assert not (await detector.detect(ATTACK, response("see axb c here")))[0].triggered


async def test_refusal_that_mentions_the_prompt_does_not_match_a_phrase() -> None:
    detector = RegexDetector.from_phrases("prompt-leak", [PROMPT_LINE])
    refusal = "I'm sorry, I can't share my system prompt or any internal instructions."
    assert not (await detector.detect(ATTACK, response(refusal)))[0].triggered


async def test_from_phrases_passes_options_through() -> None:
    detector = RegexDetector.from_phrases("p", ["hello there"], confidence=0.7)
    assert (await detector.detect(ATTACK, response("Hello There")))[0].confidence == 0.7


# --- bad configuration ---------------------------------------------------------------------


def test_bad_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="name"):
        RegexDetector("", ["x"])
    with pytest.raises(ValueError, match="at least one pattern"):
        RegexDetector("d", [])
    with pytest.raises(ValueError, match="between 0 and 1"):
        RegexDetector("d", ["x"], confidence=1.5)


def test_invalid_regex_is_rejected_with_the_pattern_in_the_message() -> None:
    with pytest.raises(ValueError, match=r"invalid regex '\(unclosed'"):
        RegexDetector("d", ["(unclosed"])


@pytest.mark.parametrize("pattern", ["", "a*", "(?:)", "x?"])
def test_pattern_that_matches_the_empty_string_is_rejected(pattern: str) -> None:
    with pytest.raises(ValueError, match="matches the empty string"):
        RegexDetector("d", [pattern])
