"""Space out requests so a provider's requests-per-minute limit isn't hit."""

import asyncio
import math
import time
from collections.abc import Awaitable, Callable


class RateLimiter:
    """Lets requests start evenly spaced: at 30 per minute, one every 2 seconds.

    Share ONE instance between all the targets that use the same quota. No bursts are allowed,
    which stays within a limit however the server measures its minute. Limits per day or per
    token are not handled.
    """

    def __init__(
        self,
        requests_per_minute: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not math.isfinite(requests_per_minute) or requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be a positive number")
        self._interval = 60 / requests_per_minute
        self._clock = clock
        self._sleep = sleep
        # The earliest time the next request may start. -inf: the first one never waits.
        self._next = -math.inf

    async def wait(self) -> None:
        """Return when it is this caller's turn to send."""
        now = self._clock()
        start = max(now, self._next)
        # Claim the slot BEFORE sleeping. There is no await between the two lines above and
        # this one, so two callers can't claim the same slot and no lock is needed.
        self._next = start + self._interval
        if start > now:
            await self._sleep(start - now)
