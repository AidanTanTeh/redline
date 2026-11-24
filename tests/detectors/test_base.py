from redline.core import Attack, AttackCategory, DetectionResult, TargetResponse
from redline.detectors import Detector


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
