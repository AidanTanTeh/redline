"""Retry transient failures (timeouts, dropped connections, 429, 5xx) with backoff."""

import asyncio
import math
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

BASE_DELAY_S = 1.0
# The longest single wait. A server asking for more than this ends the retries.
MAX_DELAY_S = 60.0

# Failures where the same request might work next time. TimeoutError is the builtin raised by
# asyncio.timeout. Other TransportErrors (bad URL scheme, malformed request, proxy refusal) are
# left out: they would fail the same way every time.
_TRANSIENT = (
    TimeoutError,
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


class AttemptFailed(Exception):
    """The operation failed for good. `cause` is the last error, `retries` how many we made."""

    def __init__(self, cause: Exception, retries: int) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause
        self.retries = retries


def is_retryable(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        # 429 = rate limited, 5xx = the server's fault. Other 4xx mean OUR request is wrong.
        return status == 429 or status >= 500
    return isinstance(error, _TRANSIENT)


def retry_after_seconds(error: Exception) -> float | None:
    """How long the server asked us to wait, from the Retry-After header, if it said.

    The header is either a number of seconds ("120") or a date. Anything unreadable is None.
    """
    if not isinstance(error, httpx.HTTPStatusError):
        return None
    value = error.response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except ValueError:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        # A date in the past means "go ahead now".
        return max(0.0, (when - datetime.now(UTC)).total_seconds())
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def backoff_seconds(retry_number: int, jitter: float) -> float:
    """Exponential backoff: about 1s, 2s, 4s, ... capped. `jitter` is a number from 0 to 1.

    Jitter spreads the wait between half and all of the full delay, so that many requests which
    failed together don't all come back at the same instant.
    """
    full = min(MAX_DELAY_S, BASE_DELAY_S * 2.0**retry_number)
    return full * (0.5 + jitter / 2)


async def with_retries[T](
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    jitter: Callable[[], float] = random.random,
) -> tuple[T, int]:
    """Run `operation`, retrying transient failures. Returns (result, retries used).

    `operation` is a function that STARTS a fresh attempt each time it is called, because a
    coroutine can only be awaited once. Raises AttemptFailed on a permanent error or when the
    retries run out. Cancellation is not caught (it is not an Exception), so Ctrl-C and
    TaskGroup shutdown still work.
    """
    retries = 0
    while True:
        try:
            return await operation(), retries
        except Exception as error:
            if retries >= max_retries or not is_retryable(error):
                raise AttemptFailed(error, retries) from error
            wait = retry_after_seconds(error)
            if wait is None:
                wait = backoff_seconds(retries, jitter())
            elif wait > MAX_DELAY_S:
                # E.g. a daily quota. Retrying earlier than the server asked would just fail.
                raise AttemptFailed(error, retries) from error
            await sleep(wait)
            retries += 1
