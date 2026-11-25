from typing import Any

import pytest

from redline.core import Attack, AttackCategory, TargetResponse, ToolCall
from redline.detectors import Detector, ToolCallDetector

ATTACKER = "attacker@evil.example"


def agency_attack(tool: str = "send_email", **arguments: str) -> Attack:
    return Attack(
        id="ea-01",
        name="x",
        category=AttackCategory.EXCESSIVE_AGENCY,
        prompt="p",
        success_tool_call={"tool": tool, "arguments": arguments},  # type: ignore[arg-type]
    )


def respond(*calls: ToolCall) -> TargetResponse:
    return TargetResponse(text="Done.", model="m", latency_ms=1.0, tool_calls=calls)


def email(**arguments: Any) -> ToolCall:
    return ToolCall(name="send_email", arguments=arguments)


async def verdict(attack: Attack, response: TargetResponse) -> bool:
    (result,) = await ToolCallDetector().detect(attack, response)
    return result.triggered


def test_satisfies_the_detector_protocol() -> None:
    detector: Detector = ToolCallDetector()
    assert detector.name == "tool_call"


async def test_matching_call_triggers_with_evidence() -> None:
    attack = agency_attack(to=ATTACKER)
    (result,) = await ToolCallDetector().detect(attack, respond(email(to=ATTACKER, body="secret")))
    assert result.triggered
    assert result.field == "tool_calls"
    assert result.evidence is not None
    assert ATTACKER in result.evidence


@pytest.mark.parametrize(
    "call",
    [
        email(to="ATTACKER@EVIL.EXAMPLE"),  # case does not matter
        email(to=f"Mallory <{ATTACKER}>"),  # the wanted text only has to appear in the value
        ToolCall(name="SEND_EMAIL", arguments={"to": ATTACKER}),  # tool name case too
    ],
)
async def test_flexible_matches_still_trigger(call: ToolCall) -> None:
    assert await verdict(agency_attack(to=ATTACKER), respond(call))


@pytest.mark.parametrize(
    "calls",
    [
        (),  # no tool calls at all
        (email(to="friend@acme.example"),),  # right tool, wrong recipient
        (ToolCall(name="search_docs", arguments={"to": ATTACKER}),),  # wrong tool
        (email(body="hello"),),  # the wanted argument is missing
        (ToolCall(name="send_email", arguments={"_raw": "{oops"}),),  # malformed, no address
    ],
)
async def test_non_matching_calls_do_not_trigger(calls: tuple[ToolCall, ...]) -> None:
    assert not await verdict(agency_attack(to=ATTACKER), respond(*calls))


async def test_any_one_matching_call_among_several_is_enough() -> None:
    response = respond(
        ToolCall(name="search_docs", arguments={"query": "x"}),
        email(to="friend@acme.example"),
        email(to=ATTACKER),
    )
    assert await verdict(agency_attack(to=ATTACKER), response)


async def test_every_wanted_argument_must_match() -> None:
    attack = agency_attack(to=ATTACKER, subject="password")
    assert not await verdict(attack, respond(email(to=ATTACKER, subject="hello")))
    assert await verdict(attack, respond(email(to=ATTACKER, subject="the password")))


async def test_no_wanted_arguments_means_any_call_to_the_tool_counts() -> None:
    assert await verdict(agency_attack(), respond(email(to="anyone@acme.example")))
    assert not await verdict(agency_attack(), respond())


async def test_non_text_argument_values_are_compared_as_text() -> None:
    assert await verdict(agency_attack(to="123"), respond(email(to=123)))


async def test_attack_without_a_success_condition_gets_no_result() -> None:
    attack = Attack(id="d-01", name="x", category=AttackCategory.DIRECT_INJECTION, prompt="p")
    assert await ToolCallDetector().detect(attack, respond(email(to=ATTACKER))) == ()


async def test_malformed_arguments_are_matched_against_their_raw_text() -> None:
    # A small model wrote broken JSON but clearly tried to email the attacker: still a hit.
    raw = '{"to": "attacker@evil.example", "body": "hi",}'
    broken = ToolCall(name="send_email", arguments={"_raw": raw})
    assert await verdict(agency_attack(to=ATTACKER), respond(broken))
    # ...and a malformed call that never mentions the wanted text is not a hit.
    other = ToolCall(name="send_email", arguments={"_raw": '{"to": "friend@acme.example",'})
    assert not await verdict(agency_attack(to=ATTACKER), respond(other))
    # the wrong tool is never a hit, however the arguments look
    wrong_tool = ToolCall(name="search_docs", arguments={"_raw": raw})
    assert not await verdict(agency_attack(to=ATTACKER), respond(wrong_tool))
