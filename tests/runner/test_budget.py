import pytest

from redline.core import Usage
from redline.runner import AttemptEstimate, Budget, Pricing

WORST = AttemptEstimate(requests=5, input_tokens=1000, output_tokens=500)
# $1 per million in, $2 per million out: one worst-case try costs $0.002.
PRICING = Pricing(input_per_mtok=1, output_per_mtok=2)


def used(requests: int = 1, input_tokens: int = 100, output_tokens: int = 50) -> Usage:
    return Usage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens)


def test_without_caps_everything_fits() -> None:
    budget = Budget(PRICING)
    assert all(budget.reserve(WORST) for _ in range(1000))
    assert not budget.closed


@pytest.mark.parametrize(("cap", "fits"), [(4, False), (5, True)])
def test_a_try_must_fit_at_its_worst_case(cap: int, fits: bool) -> None:
    assert Budget(PRICING, max_requests=cap).reserve(WORST) is fits


def test_real_usage_frees_the_rest_of_the_reservation() -> None:
    budget = Budget(PRICING, max_requests=7)
    assert budget.reserve(WORST)
    budget.settle(WORST, used(requests=2))
    assert budget.requests == 2
    assert budget.reserve(WORST)  # 2 spent + 5 worst = 7


def test_tries_in_flight_count_against_the_cap() -> None:
    budget = Budget(PRICING, max_requests=9)
    assert budget.reserve(WORST)
    assert not budget.reserve(WORST)  # 10 > 9 while the first is unfinished


def test_after_one_refusal_nothing_else_starts() -> None:
    budget = Budget(PRICING, max_requests=6)
    assert budget.reserve(WORST)
    assert not budget.reserve(WORST)
    assert budget.closed
    cheap = AttemptEstimate(requests=1, input_tokens=0, output_tokens=0)
    assert not budget.reserve(cheap)


def test_cost_cap_allows_spending_it_exactly_and_no_more() -> None:
    # 0.1 + 0.2 style float noise must not refuse the last try that fits.
    budget = Budget(PRICING, max_cost_usd=0.006)
    for _ in range(3):
        assert budget.reserve(WORST)
        budget.settle(WORST, used(input_tokens=1000, output_tokens=500))
    assert budget.cost_usd == pytest.approx(0.006)
    assert not budget.reserve(WORST)


def test_spending_never_passes_the_cap() -> None:
    budget = Budget(PRICING, max_requests=23, max_cost_usd=0.0105)
    while budget.reserve(WORST):
        budget.settle(WORST, None)  # every try as bad as it can be
    assert budget.requests <= 23
    assert budget.cost_usd <= 0.0105


def test_a_free_model_is_never_stopped_by_a_zero_cost_cap() -> None:
    assert Budget(Pricing.free(), max_cost_usd=0).reserve(WORST)


@pytest.mark.parametrize(
    ("usage", "requests", "cost"),
    [
        (None, 5, 0.002),  # nothing known: all worst case
        (Usage(requests=2), 2, 0.002),  # calls known, tokens not
        (Usage(requests=1, input_tokens=100), 1, 0.0011),  # output unknown
        (Usage(requests=1, input_tokens=100, output_tokens=0), 1, 0.0001),  # real zero
    ],
)
def test_unknown_figures_are_charged_at_the_worst_case(
    usage: Usage | None, requests: int, cost: float
) -> None:
    budget = Budget(PRICING)
    assert budget.reserve(WORST)
    budget.settle(WORST, usage)
    assert budget.requests == requests
    assert budget.cost_usd == pytest.approx(cost)
