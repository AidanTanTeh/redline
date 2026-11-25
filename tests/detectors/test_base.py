from redline.core import Attack, AttackCategory, DetectionResult, TargetResponse, ToolCall
from redline.detectors import Detector
from redline.detectors.base import field_text


class AlwaysTriggers:
    """Not inheriting from Detector: a class counts if it has the right members."""

    @property
    def name(self) -> str:
        return "always"

    async def detect(self, attack: Attack, response: TargetResponse) -> tuple[DetectionResult, ...]:
        return (DetectionResult(detector=self.name, triggered=True, field="text"),)


async def test_a_class_with_the_right_shape_is_a_detector() -> None:
    detector: Detector = AlwaysTriggers()  # mypy checks this line (see PROGRESS tooling note)
    attack = Attack(id="a-01", name="x", category=AttackCategory.DIRECT_INJECTION, prompt="p")
    response = TargetResponse(text="hi", model="m", latency_ms=1.0)
    (result,) = await detector.detect(attack, response)
    assert result.triggered
    assert result.field == "text"


def test_field_text_covers_every_response_field() -> None:
    response = TargetResponse(
        text="hello",
        reasoning="thinking",
        model="m",
        latency_ms=1.0,
        tool_calls=(
            ToolCall(name="send_email", arguments={"to": "a@b.example", "body": "café"}),
            ToolCall(name="search_docs", arguments={}),
        ),
    )
    assert field_text(response, "text") == "hello"
    assert field_text(response, "reasoning") == "thinking"
    # one line per call; non-ASCII stays readable instead of becoming \u escapes
    assert field_text(response, "tool_calls") == (
        'send_email {"to": "a@b.example", "body": "café"}\nsearch_docs {}'
    )


def test_field_text_is_empty_for_missing_reasoning_and_no_tool_calls() -> None:
    response = TargetResponse(text="hi", model="m", latency_ms=1.0)
    assert field_text(response, "reasoning") == ""
    assert field_text(response, "tool_calls") == ""
