"""The Detector interface: anything that decides whether an attack succeeded."""

import json
from typing import Protocol

from redline.core import Attack, DetectionResult, ResponseField, TargetResponse


def scan_fields(*, reasoning: bool, tool_calls: bool) -> tuple[ResponseField, ...]:
    """Which response fields a text detector should scan. `text` is always included."""
    fields: list[ResponseField] = ["text"]
    if reasoning:
        fields.append("reasoning")
    if tool_calls:
        fields.append("tool_calls")
    return tuple(fields)


def present_fields(
    response: TargetResponse, fields: tuple[ResponseField, ...]
) -> tuple[ResponseField, ...]:
    """Drop `tool_calls` if the response made none: there is nothing to scan, so no verdict."""
    return tuple(f for f in fields if f != "tool_calls" or response.tool_calls)


def field_text(response: TargetResponse, field: ResponseField) -> str:
    """The scannable text of one response field.

    A missing `reasoning` counts as empty, not an error. `tool_calls` becomes one line per
    call, "name {json arguments}", so text detectors can look inside what the bot tried to do
    (e.g. an email body) the same way they look inside what it said.
    """
    if field == "text":
        return response.text
    if field == "reasoning":
        return response.reasoning or ""
    return "\n".join(
        f"{call.name} {json.dumps(call.arguments, ensure_ascii=False)}"
        for call in response.tool_calls
    )


class Detector(Protocol):
    """Looks at one response and says whether the attack worked.

    Structural typing again, like `Target`: any class with these members counts.
    """

    @property
    def name(self) -> str:
        """Short label recorded on every DetectionResult, e.g. "canary"."""
        ...

    async def detect(self, attack: Attack, response: TargetResponse) -> tuple[DetectionResult, ...]:
        """Return one DetectionResult per response field this detector scanned.

        `triggered=True` always means the ATTACK SUCCEEDED. Deterministic detectors ignore
        `attack`; the LLM judge (4.2) needs it, and it must be async to call a model, so the
        interface is async and takes both so the runner can treat every detector the same.
        """
        ...
