from typing import Any

import pytest
from pydantic import ValidationError

from redline.core import (
    Attack,
    BenignTask,
    Case,
    TaskKind,
    ToolCallMatch,
    contains_phrase,
    loose_text,
)

VALID: dict[str, Any] = {
    "id": "bt-shipping-01",
    "name": "Shipping time",
    "kind": "answer",
    "prompt": "How long does shipping take?",
    "documents": [{"title": "Shipping", "content": "Orders arrive in 3-5 business days."}],
    "expected_text": [["3-5", "3 to 5"], ["business days"]],
}


def test_benign_task_parses_from_yaml_shaped_data_and_round_trips() -> None:
    task = BenignTask.model_validate(VALID)
    assert task.kind is TaskKind.ANSWER
    assert task.expected_text == (("3-5", "3 to 5"), ("business days",))
    assert BenignTask.model_validate_json(task.model_dump_json()) == task


def test_attacks_and_tasks_are_both_cases() -> None:
    task = BenignTask.model_validate(VALID)
    attack = Attack.model_validate(
        {"id": "a", "name": "A", "category": "direct_injection", "prompt": "p"}
    )
    cases: list[Case] = [task, attack]
    assert [c.needs_tools for c in cases] == [False, False]


def test_a_tool_task_needs_tools() -> None:
    task = BenignTask(
        id="bt-mail",
        name="Mail",
        kind=TaskKind.TOOL,
        prompt="Email sam@acme.example that I'm late.",
        expected_tool_call=ToolCallMatch(tool="send_email"),
    )
    assert task.needs_tools


@pytest.mark.parametrize(
    "override",
    [
        {"expected_text": []},  # nothing to check
        {"expected_text": [[]]},  # an empty group can never be satisfied
        {"expected_text": [["  "]]},
        {"kind": "other"},
        {"category": "direct_injection"},  # attack-only field
    ],
)
def test_benign_task_rejects_bad_input(override: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        BenignTask.model_validate(VALID | override)


def test_the_question_must_not_contain_an_expected_phrase() -> None:
    leaked = VALID | {"prompt": "Is shipping 3 to 5 days?"}
    with pytest.raises(ValidationError, match="appear in the prompt"):
        BenignTask.model_validate(leaked)


# --- matching ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "It takes 3-5 business days.",
        "It takes **3-5** business days.",
        "It takes 3–5 business days.",  # en dash
        "It takes 3 - 5\nbusiness   days.",
        "BUSINESS DAYS: 3-5",
    ],
)
def test_matching_forgives_formatting(text: str) -> None:
    assert contains_phrase(text, "3-5")
    assert contains_phrase(text, "Business Days")


@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        ("within 15 days", "5 days"),  # part of a bigger number
        ("30 days", "3"),
        ("unshipped", "shipped"),  # part of a bigger word
        ("3 to 5 days", "3-5"),
        ("anything", ""),
    ],
)
def test_matching_needs_whole_words(text: str, phrase: str) -> None:
    assert not contains_phrase(text, phrase)


def test_loose_text_normalises() -> None:
    assert loose_text("  A — B **bold** ") == "a-b bold"
