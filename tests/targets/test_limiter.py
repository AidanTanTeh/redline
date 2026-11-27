import asyncio

import httpx
import pytest

from redline.targets import OpenAICompatConfig, OpenAICompatTarget, RateLimiter


class FakeTime:
    """A clock that only moves when someone sleeps, so the tests take no real time."""

    def __init__(self) -> None:
        self.now = 100.0

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def limiter(rpm: float, time: FakeTime) -> RateLimiter:
    return RateLimiter(rpm, clock=time.clock, sleep=time.sleep)


async def test_the_first_request_goes_at_once_and_the_rest_are_spaced() -> None:
    time = FakeTime()
    limit = limiter(30, time)
    started = []
    for _ in range(3):
        await limit.wait()
        started.append(time.now)
    assert started == [100, 102, 104]


async def test_a_caller_that_was_already_slow_does_not_wait() -> None:
    time = FakeTime()
    limit = limiter(30, time)
    await limit.wait()
    time.now += 10  # the model took a while
    await limit.wait()
    assert time.now == 110


async def test_idle_time_is_not_saved_up_for_a_burst() -> None:
    time = FakeTime()
    limit = limiter(30, time)
    await limit.wait()
    time.now += 60
    await limit.wait()
    await limit.wait()
    assert time.now == 162


async def test_callers_waiting_together_get_different_slots() -> None:
    asked: list[float] = []

    async def sleep(seconds: float) -> None:
        asked.append(seconds)
        await asyncio.sleep(0)  # let the others run while this one is "asleep"

    limit = RateLimiter(30, clock=lambda: 0.0, sleep=sleep)
    await asyncio.gather(limit.wait(), limit.wait(), limit.wait())
    assert asked == [2, 4]


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan")])
def test_nonsense_rates_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        RateLimiter(bad)


async def test_targets_sharing_a_limiter_share_the_quota_and_waiting_is_not_latency() -> None:
    time = FakeTime()
    limit = limiter(1, time)  # one a minute

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    config = OpenAICompatConfig(base_url="http://test/v1", model="m")
    first, second = (OpenAICompatTarget(config, client, limiter=limit) for _ in range(2))

    await first.send("a")
    response = await second.send("b")

    assert time.now == 160
    assert response.latency_ms < 5000
