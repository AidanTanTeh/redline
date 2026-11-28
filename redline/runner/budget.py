"""Keeps a run inside its caps while it is running.

Each call is reserved at its worst case BEFORE it is sent, then replaced by what it really used.
So a cap can't be overshot, as long as no call uses more than its estimate.
"""

from dataclasses import dataclass

from redline.core import Usage
from redline.runner.cost import AttemptEstimate, Pricing

# Absorbs floating-point noise, so spending exactly the cap is allowed.
_COST_TOLERANCE_USD = 1e-9


class BudgetExceeded(ValueError):
    """The whole run could go over a cap, so nothing was sent."""


@dataclass
class _Tally:
    """A running count for one price list (the target's or the judge's)."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, estimate: AttemptEstimate, sign: int = 1) -> None:
        self.requests += sign * estimate.requests
        self.input_tokens += sign * estimate.input_tokens
        self.output_tokens += sign * estimate.output_tokens

    def add_used(self, worst: AttemptEstimate, usage: Usage | None) -> None:
        """Count what a call used. Anything unknown (a failed call, a server that reports no
        tokens) is charged at the worst case: the work may well have been done and billed."""
        if usage is None:
            self.add(worst)
            return
        self.requests += usage.requests
        self.input_tokens += (
            worst.input_tokens if usage.input_tokens is None else usage.input_tokens
        )
        self.output_tokens += (
            worst.output_tokens if usage.output_tokens is None else usage.output_tokens
        )


class Budget:
    """Requests and money for one run. None means that thing has no cap.

    A plain class, not a Pydantic model: it is a running tally that changes. There is no lock
    because asyncio only switches tasks at an `await`, and nothing here awaits.

    Requests are one shared pool for target and judge. Money uses each one's own price.
    """

    def __init__(
        self,
        pricing: Pricing,
        *,
        judge_pricing: Pricing | None = None,
        max_requests: int | None = None,
        max_cost_usd: float | None = None,
    ) -> None:
        self._pricing = pricing
        self._judge_pricing = judge_pricing
        self._max_requests = max_requests
        self._max_cost_usd = max_cost_usd
        # Spent: finished calls. Reserved: calls in flight, counted at their worst case.
        self._spent = _Tally()
        self._reserved = _Tally()
        self._judge_spent = _Tally()
        self._judge_reserved = _Tally()
        self._closed = False

    @property
    def requests(self) -> int:
        """Model calls spent by finished calls (unknown ones counted at the worst case)."""
        return self._spent.requests + self._judge_spent.requests

    @property
    def cost_usd(self) -> float:
        """Money spent by finished calls, on the same cautious basis."""
        return self._cost((self._spent,), (self._judge_spent,))

    @property
    def closed(self) -> bool:
        """True once something didn't fit. Nothing more may start after that."""
        return self._closed

    def _cost(self, target: tuple[_Tally, ...], judge: tuple[_Tally, ...]) -> float:
        total = self._pricing.cost_usd(
            sum(t.input_tokens for t in target), sum(t.output_tokens for t in target)
        )
        judge_input = sum(t.input_tokens for t in judge)
        judge_output = sum(t.output_tokens for t in judge)
        if judge_input or judge_output:
            if self._judge_pricing is not None:
                total += self._judge_pricing.cost_usd(judge_input, judge_output)
            elif self._max_cost_usd is not None:
                raise ValueError("a cost cap with a judge needs judge_pricing")
        return total

    def reserve(self, worst: AttemptEstimate) -> bool:
        """Set aside one try's worst case, plus `worst.judge` if given. False means it may not
        be sent, and then nothing is set aside.

        The first refusal closes the budget. Otherwise smaller cases later in the list would
        still run, and the results would lean towards whatever happens to be cheap.
        """
        if self._closed:
            return False
        wanted, judge_wanted = _Tally(), _Tally()
        wanted.add(worst)
        if worst.judge is not None:
            judge_wanted.add(worst.judge)
        tallies = (self._spent, self._reserved, wanted)
        judge_tallies = (self._judge_spent, self._judge_reserved, judge_wanted)
        requests = sum(t.requests for t in (*tallies, *judge_tallies))
        cost = self._cost(tallies, judge_tallies)
        too_many = self._max_requests is not None and requests > self._max_requests
        too_dear = (
            self._max_cost_usd is not None and cost > self._max_cost_usd + _COST_TOLERANCE_USD
        )
        if too_many or too_dear:
            self._closed = True
            return False
        self._reserved.add(worst)
        if worst.judge is not None:
            self._judge_reserved.add(worst.judge)
        return True

    def settle(self, worst: AttemptEstimate, usage: Usage | None) -> None:
        """Replace a target reservation with what the try used. Call once for every successful
        reserve. The judge part, if any, is settled or released separately."""
        self._reserved.add(worst, -1)
        self._spent.add_used(worst, usage)

    def settle_judge(self, worst: AttemptEstimate, usage: Usage | None) -> None:
        """Replace a judge reservation with what the verdict used."""
        self._judge_reserved.add(worst, -1)
        self._judge_spent.add_used(worst, usage)

    def release_judge(self, worst: AttemptEstimate) -> None:
        """Hand back a judge reservation when there was no reply, so it was never called."""
        self._judge_reserved.add(worst, -1)
