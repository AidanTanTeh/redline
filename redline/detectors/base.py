"""The Detector interface: anything that decides whether an attack succeeded."""

from typing import Protocol

from redline.core import Attack, DetectionResult, ResponseField, TargetResponse


def field_text(response: TargetResponse, field: ResponseField) -> str:
    """The text of one response field. A missing `reasoning` counts as empty, not an error."""
    return response.text if field == "text" else (response.reasoning or "")


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
