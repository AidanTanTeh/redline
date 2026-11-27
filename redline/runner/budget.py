"""Keeps a run inside its caps while it is running.

Each try is reserved at its worst case BEFORE it is sent, then replaced by what it really used.
So a cap can't be overshot, as long as no try uses more than its estimate.
"""

from redline.core import Usage
from redline.runner.cost import AttemptEstimate, Pricing

# Absorbs floating-point noise, so spending exactly the cap is allowed.
_COST_TOLERANCE_USD = 1e-9


class BudgetExceeded(ValueError):
    """The whole run could go over a cap, so nothing was sent."""


class Budget:
    """Requests and money for one run. None means that thing has no cap.

    A plain class, not a Pydantic model: it is a running tally that changes. There is no lock
    because asyncio only switches tasks at an `await`, and nothing here awaits.
    """

    def __init__(
        self,
        pricing: Pricing,
        *,
        max_requests: int | None = None,
        max_cost_usd: float | None = None,
    ) -> None:
        self._pricing = pricing
        self._max_requests = max_requests
        self._max_cost_usd = max_cost_usd
        # Spent: finished tries. Reserved: tries in flight, counted at their worst case.
        self._requests = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._reserved_requests = 0
        self._reserved_input = 0
        self._reserved_output = 0
        self._closed = False

    @property
    def requests(self) -> int:
        """Model calls spent by finished tries (unknown ones counted at the worst case)."""
        return self._requests

    @property
    def cost_usd(self) -> float:
        """Money spent by finished tries, on the same cautious basis."""
        return self._pricing.cost_usd(self._input_tokens, self._output_tokens)

    @property
    def closed(self) -> bool:
        """True once something didn't fit. Nothing more may start after that."""
        return self._closed

    def reserve(self, worst: AttemptEstimate) -> bool:
        """Set aside one try's worst case. False means it may not be sent.

        The first refusal closes the budget. Otherwise smaller cases later in the list would
        still run, and the results would lean towards whatever happens to be cheap.
        """
        if self._closed:
            return False
        requests = self._requests + self._reserved_requests + worst.requests
        cost = self._pricing.cost_usd(
            self._input_tokens + self._reserved_input + worst.input_tokens,
            self._output_tokens + self._reserved_output + worst.output_tokens,
        )
        too_many = self._max_requests is not None and requests > self._max_requests
        too_dear = (
            self._max_cost_usd is not None and cost > self._max_cost_usd + _COST_TOLERANCE_USD
        )
        if too_many or too_dear:
            self._closed = True
            return False
        self._reserved_requests += worst.requests
        self._reserved_input += worst.input_tokens
        self._reserved_output += worst.output_tokens
        return True

    def settle(self, worst: AttemptEstimate, usage: Usage | None) -> None:
        """Replace a reservation with what the try used. Call once for every successful reserve.

        Anything unknown (a failed call, a target that reports no tokens) is charged at the
        worst case: the server may well have done the work and billed it.
        """
        self._reserved_requests -= worst.requests
        self._reserved_input -= worst.input_tokens
        self._reserved_output -= worst.output_tokens
        if usage is None:
            self._requests += worst.requests
            self._input_tokens += worst.input_tokens
            self._output_tokens += worst.output_tokens
            return
        self._requests += usage.requests
        self._input_tokens += (
            worst.input_tokens if usage.input_tokens is None else usage.input_tokens
        )
        self._output_tokens += (
            worst.output_tokens if usage.output_tokens is None else usage.output_tokens
        )
