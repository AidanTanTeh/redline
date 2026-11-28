"""Run attacks against a target setup and label what happened to each attempt.

Library code: it prints nothing and writes no files. Use `on_result` for progress or saving.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime

from redline.core import Case, DetectionResult, Outcome, RunResult, TargetResponse, Usage
from redline.detectors import Detector
from redline.runner.budget import Budget, BudgetExceeded
from redline.runner.cost import AttemptEstimate, EstimatingFactory, Pricing, estimate_run
from redline.runner.report import RunReport
from redline.runner.retry import AttemptFailed, with_retries
from redline.runner.trial import (
    RunConfig,
    TrialFactory,
    can_run,
    config_hash,
    may_retry,
)
from redline.targets import AgentReplyError

# Finish reasons meaning the target never gave a usable answer. "content_filter" is the
# OpenAI-style value for a reply the provider blocked: that isn't evidence the target itself
# resisted. Not yet seen live, and unconfirmed for Gemini.
_NO_ANSWER_REASONS = frozenset({"length", "max_tool_rounds", "content_filter"})


class _OverBudget(Exception):
    """A try that the budget would not allow to be sent."""


def classify(
    response: TargetResponse, detections: Sequence[DetectionResult], delivered: bool | None
) -> Outcome:
    """Label an attempt that got a response. The order of these checks is the rule.

    `delivered=None` means the delivery check crashed, so nobody knows.
    """
    if any(d.triggered for d in detections):
        return Outcome.SUCCEEDED  # a leak is a leak, even from a cut-off reply
    if delivered is None or any(d.error is not None for d in detections):
        return Outcome.ERROR  # a crashed check leaves nothing to conclude from
    if response.finish_reason in _NO_ANSWER_REASONS:
        return Outcome.NO_ANSWER
    if not delivered:
        return Outcome.NOT_DELIVERED
    return Outcome.RESISTED


async def run_attacks(
    attacks: Sequence[Case],
    factory: TrialFactory,
    config: RunConfig,
    *,
    cases_hash: str,
    on_result: Callable[[RunResult], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    pricing: Pricing | None = None,
    allow_over_estimate: bool = False,
    judge_pricing: Pricing | None = None,
) -> RunReport:
    """Run every attack `config.repeats` times and return all the results.

    A failing attempt becomes an ERROR result and the run carries on. `on_result` is called as
    each attempt finishes; if it raises, the run stops (losing results silently is worse).
    `sleep` is only the wait between retries, replaceable in tests.

    With `config.max_requests` or `config.max_cost_usd` set (a cost cap also needs `pricing`):
    the run is refused with BudgetExceeded if its worst case is over a cap, unless
    `allow_over_estimate` is True. Then it runs until the next try might not fit, and every
    attempt after that is recorded as NOT_RUN. A judge's calls count towards both caps; with
    a cost cap its price (`judge_pricing`) must be given too.
    """
    ids = [attack.id for attack in attacks]
    if len(set(ids)) != len(ids):
        # Results are identified by (attack_id, attempt), so duplicates would collide.
        raise ValueError("attack ids must be unique within a run")
    factory.preflight()
    description = dict(factory.describe())
    run_hash = config_hash(description, cases_hash, attacks, config)
    budget, worst_cases = _make_budget(
        attacks, factory, config, pricing, judge_pricing, allow_over_estimate
    )
    started_at = datetime.now(UTC)
    # At most this many attempts are in flight at once.
    slots = asyncio.Semaphore(config.max_concurrency)

    async def limited(attack: Case, attempt: int) -> RunResult:
        async with slots:
            worst = worst_cases.get(attack.id)
            result = await _run_attempt(
                attack, attempt, factory, config, run_hash, sleep, budget, worst
            )
            if on_result is not None:
                on_result(result)
            return result

    order = [(a, attempt) for a in range(len(attacks)) for attempt in range(config.repeats)]
    if budget is not None:
        # Every attack's first attempt, then every second one: a run stopped by its cap then
        # covers all the attacks thinly instead of only the first few.
        order.sort(key=lambda pair: pair[1])
    async with asyncio.TaskGroup() as group:
        tasks = {
            (a, attempt): group.create_task(limited(attacks[a], attempt)) for a, attempt in order
        }

    return RunReport(
        results=tuple(tasks[key].result() for key in sorted(tasks)),
        config_hash=run_hash,
        cases_hash=cases_hash,
        target=description,
        started_at=started_at,
        finished_at=datetime.now(UTC),
    )


def _make_budget(
    attacks: Sequence[Case],
    factory: TrialFactory,
    config: RunConfig,
    pricing: Pricing | None,
    judge_pricing: Pricing | None,
    allow_over_estimate: bool,
) -> tuple[Budget | None, dict[str, AttemptEstimate]]:
    """The run's budget and each case's worst case, or (None, {}) when nothing is capped."""
    if config.max_requests is None and config.max_cost_usd is None:
        return None, {}
    if config.max_cost_usd is not None and pricing is None:
        raise ValueError("max_cost_usd needs pricing (use Pricing.free() for a local model)")
    if not isinstance(factory, EstimatingFactory):
        raise TypeError("a capped run needs a factory with estimate_attempt()")
    prices = pricing or Pricing.free()  # only a request cap: money doesn't matter
    worst_cases = {
        case.id: factory.estimate_attempt(case) for case in attacks if can_run(factory, case)
    }
    judged = any(w.judge is not None for w in worst_cases.values())
    if judged and judge_pricing is None:
        if config.max_cost_usd is not None:
            raise ValueError(
                "a cost cap with a judge needs judge_pricing (Pricing.free() for a local one)"
            )
        judge_pricing = Pricing.free()
    if not allow_over_estimate:
        estimate = estimate_run(attacks, factory, config, prices, judge_pricing=judge_pricing)
        over = []
        if config.max_requests is not None and estimate.requests > config.max_requests:
            over.append(f"{estimate.requests} requests (cap {config.max_requests})")
        if config.max_cost_usd is not None and estimate.cost_usd > config.max_cost_usd:
            over.append(f"${estimate.cost_usd:.4f} (cap ${config.max_cost_usd})")
        if over:
            raise BudgetExceeded(
                f"worst case is {' and '.join(over)}. Run fewer cases or repeats, raise the "
                "cap, or pass allow_over_estimate=True to run until the cap is reached."
            )
    if config.max_cost_usd is not None and not all(w.tokens_known for w in worst_cases.values()):
        raise ValueError(
            "this target's token use is unknown, so a cost cap can't be enforced. "
            "Use max_requests instead."
        )
    budget = Budget(
        prices,
        judge_pricing=judge_pricing,
        max_requests=config.max_requests,
        max_cost_usd=config.max_cost_usd,
    )
    return budget, worst_cases


def _error_text(factory: TrialFactory, error: Exception | None) -> str:
    """What gets saved about a failure. A factory with `save_error_messages = False` runs
    code whose messages may hold private data, so only the error type is kept."""
    name = type(error).__name__
    if getattr(factory, "save_error_messages", True) or isinstance(error, AgentReplyError):
        return f"{name}: {error}"
    return f"{name} (message hidden; set save_error_messages=True to see it)"


async def _detect(
    factory: TrialFactory,
    detectors: Sequence[Detector],
    attack: Case,
    response: TargetResponse,
) -> tuple[DetectionResult, ...]:
    """Run every detector. One that raises gets a result carrying its error instead of a
    verdict, and the others still run."""
    results: list[DetectionResult] = []
    for detector in detectors:
        try:
            results.extend(await detector.detect(attack, response))
        except Exception as error:
            results.append(
                DetectionResult(
                    detector=detector.name,
                    triggered=False,
                    confidence=0,
                    error=_error_text(factory, error),
                )
            )
    return tuple(results)


def _verdict_usage(detections: Sequence[DetectionResult]) -> Usage | None:
    """What the model-calling detectors used, or None if none of them reported it."""
    total: Usage | None = None
    for detection in detections:
        if detection.usage is not None:
            total = detection.usage if total is None else total + detection.usage
    return total


async def _run_attempt(
    attack: Case,
    attempt: int,
    factory: TrialFactory,
    config: RunConfig,
    run_hash: str,
    sleep: Callable[[float], Awaitable[None]],
    budget: Budget | None,
    worst: AttemptEstimate | None,
) -> RunResult:
    """One attempt, start to finish. Never raises for an ordinary failure."""
    seed = config.seed_for(attempt)

    def record(
        outcome: Outcome,
        *,
        response: TargetResponse | None = None,
        detections: tuple[DetectionResult, ...] = (),
        delivered: bool | None = None,
        delivery_error: str | None = None,
        error: str | None = None,
        retries: int = 0,
    ) -> RunResult:
        return RunResult(
            attack_id=attack.id,
            attempt=attempt,
            target=factory.name,
            model=factory.model,
            config_hash=run_hash,
            seed=seed,
            outcome=outcome,
            response=response,
            detections=detections,
            delivered=delivered,
            delivery_error=delivery_error,
            error=error,
            retries=retries,
        )

    if not can_run(factory, attack):
        return record(Outcome.SKIPPED)

    retries = 0
    tries_sent = 0
    last_error: Exception | None = None
    # The judge rules once per attempt, so its call is reserved with the first try only, and
    # handed back if no reply ever arrives.
    judge_worst = None if worst is None else worst.judge
    judge_reserved = False
    try:
        trial = factory.make_trial(attack, seed)
        if not trial.detectors:
            # With nothing judging the reply, every attempt would look "resisted".
            raise ValueError("trial has no detectors")

        async def send() -> TargetResponse:
            nonlocal tries_sent, last_error, judge_reserved
            # Every try is paid for separately, so each retry has to fit in the budget too.
            if budget is not None and worst is not None:
                wanted = worst if not tries_sent else worst.model_copy(update={"judge": None})
                if not budget.reserve(wanted):
                    raise _OverBudget
                judge_reserved = judge_reserved or wanted.judge is not None
            tries_sent += 1
            response = None
            try:
                async with asyncio.timeout(config.attempt_timeout_s):
                    response = await trial.target.send(trial.message)
                return response
            except Exception as error:
                last_error = error
                raise
            finally:
                # Also runs on failure or cancellation, where the usage is unknown.
                if budget is not None and worst is not None:
                    budget.settle(worst, None if response is None else response.usage)

        response, retries = await with_retries(
            send,
            max_retries=config.max_retries,
            sleep=sleep,
            allow=lambda error: may_retry(factory, error),
        )
        # Outside the retries: a detector bug must not make us call the model again.
        detections = await _detect(factory, trial.detectors, attack, response)
        if budget is not None and judge_worst is not None and judge_reserved:
            budget.settle_judge(judge_worst, _verdict_usage(detections))
            judge_reserved = False
        delivered: bool | None = None
        delivery_error: str | None = None
        try:
            delivered = trial.delivered(response)
        except Exception as error:
            delivery_error = _error_text(factory, error)
    except AttemptFailed as failure:
        if isinstance(failure.cause, _OverBudget):
            if not tries_sent:
                return record(Outcome.NOT_RUN)
            # It was tried and failed; the cap only stopped the retry.
            return record(
                Outcome.ERROR,
                error=f"{_error_text(factory, last_error)} (retry stopped by the cap)",
                retries=tries_sent - 1,
            )
        return record(
            Outcome.ERROR, error=_error_text(factory, failure.cause), retries=failure.retries
        )
    except Exception as error:
        # Caught here on purpose: an escaped exception would make the TaskGroup cancel every
        # other attempt in the run.
        return record(Outcome.ERROR, error=_error_text(factory, error), retries=retries)
    finally:
        if budget is not None and judge_worst is not None and judge_reserved:
            budget.release_judge(judge_worst)

    outcome = classify(response, detections, delivered)
    failures = [f"detector {d.detector}: {d.error}" for d in detections if d.error is not None]
    if delivery_error is not None:
        failures.append(f"delivery check: {delivery_error}")
    return record(
        outcome,
        response=response,
        detections=detections,
        delivered=delivered,
        delivery_error=delivery_error,
        # A success keeps its failures on the detection or `delivery_error`; `error` is for
        # ERROR only.
        error=f"judging failed: {'; '.join(failures)}" if outcome is Outcome.ERROR else None,
        retries=retries,
    )
