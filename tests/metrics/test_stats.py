import pytest

from redline.metrics import Interval, wilson_interval


def bounds(interval: Interval) -> tuple[float, float]:
    return round(interval.low, 4), round(interval.high, 4)


@pytest.mark.parametrize(
    ("successes", "n", "expected"),
    [
        (0, 10, (0.0, 0.2775)),  # published value
        (5, 10, (0.2366, 0.7634)),
        (10, 10, (0.7225, 1.0)),  # mirror image of 0 out of 10
        (3, 6, (0.1876, 0.8124)),  # a typical category: 6 attacks, half succeed
    ],
)
def test_known_values(successes: int, n: int, expected: tuple[float, float]) -> None:
    assert bounds(wilson_interval(successes / n, n)) == expected


@pytest.mark.parametrize("proportion", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("n", [1, 7, 1000])
def test_stays_in_range_contains_the_rate_and_is_never_zero_width(
    proportion: float, n: int
) -> None:
    interval = wilson_interval(proportion, n)
    assert 0 <= interval.low <= proportion + 1e-12
    assert proportion - 1e-12 <= interval.high <= 1
    assert interval.high > interval.low


def test_more_samples_narrow_it_and_more_confidence_widens_it() -> None:
    def width(n: int, confidence: float = 0.95) -> float:
        interval = wilson_interval(0.5, n, confidence)
        return interval.high - interval.low

    assert width(6) > width(30) > width(300)
    assert width(30, 0.99) > width(30, 0.95) > width(30, 0.80)


@pytest.mark.parametrize(
    ("proportion", "n", "confidence"),
    [(-0.1, 5, 0.95), (1.1, 5, 0.95), (0.5, 0, 0.95), (0.5, 5, 0.0), (0.5, 5, 1.0)],
)
def test_rejects_impossible_input(proportion: float, n: int, confidence: float) -> None:
    with pytest.raises(ValueError, match="must be"):
        wilson_interval(proportion, n, confidence)
