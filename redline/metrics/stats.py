"""How uncertain is a rate? Confidence intervals for proportions."""

import math
from statistics import NormalDist

from pydantic import Field

from redline.core import RedlineModel


class Interval(RedlineModel):
    """A range a rate plausibly lies in, e.g. 0.19 to 0.81 at 95% confidence."""

    low: float = Field(ge=0, le=1)
    high: float = Field(ge=0, le=1)
    confidence: float = Field(gt=0, lt=1)


def wilson_interval(proportion: float, n: int, confidence: float = 0.95) -> Interval:
    """The Wilson score interval for a `proportion` observed over `n` independent samples.

    Chosen over the textbook "p plus or minus an error" interval, which breaks with small
    samples: at 0 out of 10 it says exactly 0% to 0%, and it can go below 0 or above 1.
    Wilson stays inside 0-1 and still admits doubt at 0% and 100%.

    The maths assumes `n` independent yes/no samples. The caller must pick an honest `n`.
    """
    if not 0 <= proportion <= 1:
        raise ValueError("proportion must be between 0 and 1")
    if n < 1:
        raise ValueError("n must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1 (exclusive)")

    # How many standard deviations wide: about 1.96 for 95% confidence.
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    shrink = 1 + z**2 / n
    # The middle is pulled from the observed value towards 50%; less so as n grows.
    middle = (proportion + z**2 / (2 * n)) / shrink
    half_width = z * math.sqrt(proportion * (1 - proportion) / n + z**2 / (4 * n**2)) / shrink
    # min/max only tidy up floating-point noise such as -0.0000000000000001.
    return Interval(
        low=max(0.0, middle - half_width),
        high=min(1.0, middle + half_width),
        confidence=confidence,
    )
