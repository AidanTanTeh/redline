import asyncio
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from redline.runner import AttemptFailed, with_retries
from redline.runner.retry import backoff_seconds, is_retryable, retry_after_seconds

REQUEST = httpx.Request("POST", "http://test.example/v1/chat/completions")


def status_error(status: int, headers: dict[str, str] | None = None) -> httpx.HTTPStatusError:
    response = httpx.Response(status, headers=headers, request=REQUEST)
    return httpx.HTTPStatusError(f"HTTP {status}", request=REQUEST, response=response)


class Flaky:
    """Raises each given error in turn, then returns "ok"."""

    def __init__(self, *errors: Exception) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


class FakeSleep:
    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


# --- what is worth retrying -------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError(),  # asyncio.timeout around a whole attempt
        httpx.ReadTimeout("slow"),
        httpx.ConnectError("refused"),
        httpx.RemoteProtocolError("server hung up"),
        status_error(429),
        status_error(500),
        status_error(503),
    ],
)
def test_transient_errors_are_retryable(error: Exception) -> None:
    assert is_retryable(error)


@pytest.mark.parametrize(
    "error",
    [
        status_error(400),
        status_error(401),  # bad key: retrying can't fix it
        status_error(404),  # e.g. a model that doesn't exist
        httpx.UnsupportedProtocol("ftp://"),
        httpx.LocalProtocolError("bad header"),
        ValueError("Unexpected response shape"),
        RuntimeError("missing env var"),
    ],
)
def test_permanent_errors_are_not_retryable(error: Exception) -> None:
    assert not is_retryable(error)


# --- how long to wait -------------------------------------------------------


def test_backoff_doubles_is_jittered_and_capped() -> None:
    assert [backoff_seconds(n, 1.0) for n in range(3)] == [1.0, 2.0, 4.0]
    assert backoff_seconds(2, 0.0) == 2.0  # jitter can halve it, never more
    assert backoff_seconds(20, 1.0) == 60.0


def test_retry_after_reads_seconds_and_dates() -> None:
    assert retry_after_seconds(status_error(429, {"Retry-After": "7"})) == 7.0
    later = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    wait = retry_after_seconds(status_error(503, {"Retry-After": later}))
    assert wait is not None and 25 < wait <= 30
    past = format_datetime(datetime.now(UTC) - timedelta(days=1), usegmt=True)
    assert retry_after_seconds(status_error(429, {"Retry-After": past})) == 0.0


@pytest.mark.parametrize("value", ["soon", "-5", "nan", "inf", ""])
def test_unreadable_retry_after_is_ignored(value: str) -> None:
    assert retry_after_seconds(status_error(429, {"Retry-After": value})) is None


def test_retry_after_is_absent_for_other_errors() -> None:
    assert retry_after_seconds(status_error(429)) is None
    assert retry_after_seconds(httpx.ReadTimeout("slow")) is None


# --- the retry loop ---------------------------------------------------------


async def test_success_first_time_needs_no_retry() -> None:
    sleep = FakeSleep()
    assert await with_retries(Flaky(), max_retries=2, sleep=sleep) == ("ok", 0)
    assert sleep.waits == []


async def test_recovers_after_transient_errors_with_growing_waits() -> None:
    sleep = FakeSleep()
    op = Flaky(httpx.ReadTimeout("slow"), status_error(500))
    result = await with_retries(op, max_retries=2, sleep=sleep, jitter=lambda: 1.0)
    assert result == ("ok", 2)
    assert sleep.waits == [1.0, 2.0]


async def test_server_retry_after_beats_our_backoff() -> None:
    sleep = FakeSleep()
    op = Flaky(status_error(429, {"Retry-After": "12"}))
    await with_retries(op, max_retries=1, sleep=sleep)
    assert sleep.waits == [12.0]


async def test_gives_up_when_retries_run_out() -> None:
    op = Flaky(*[status_error(503)] * 5)
    with pytest.raises(AttemptFailed) as failed:
        await with_retries(op, max_retries=2, sleep=FakeSleep())
    assert (op.calls, failed.value.retries) == (3, 2)
    assert "HTTPStatusError" in str(failed.value)


async def test_permanent_error_fails_immediately() -> None:
    op = Flaky(status_error(401))
    sleep = FakeSleep()
    with pytest.raises(AttemptFailed) as failed:
        await with_retries(op, max_retries=5, sleep=sleep)
    assert (op.calls, failed.value.retries, sleep.waits) == (1, 0, [])


async def test_a_very_long_retry_after_gives_up_instead_of_retrying_early() -> None:
    op = Flaky(status_error(429, {"Retry-After": "3600"}))
    sleep = FakeSleep()
    with pytest.raises(AttemptFailed):
        await with_retries(op, max_retries=5, sleep=sleep)
    assert (op.calls, sleep.waits) == (1, [])


async def test_cancellation_is_never_swallowed_or_retried() -> None:
    calls = 0

    async def cancelled() -> str:
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await with_retries(cancelled, max_retries=3, sleep=FakeSleep())
    assert calls == 1
