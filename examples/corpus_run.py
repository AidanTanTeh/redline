"""Run corpus attacks, or benign tasks, against VulnBot through the runner and save every result.

Only attacks VulnBot, our own demo target, on local Ollama. Costs $0 (no API key).
Run from the repo root:
    uv run python examples/corpus_run.py                          # 4 attacks x 2 repeats
    uv run python examples/corpus_run.py --repeats 3 di-override-01 ii-email-01
    uv run python examples/corpus_run.py --channel tool ii-review-note-01
    uv run python examples/corpus_run.py --all --repeats 3      # every attack (slow)
    uv run python examples/corpus_run.py --benign --repeats 3   # every benign task
    uv run python examples/corpus_run.py --all --dry-run        # show the estimate only
    uv run python examples/corpus_run.py --max-requests 12 --partial --rpm 20
    uv run python examples/corpus_run.py --controls --repeats 3 # can it read each disguise?
    uv run python examples/corpus_run.py --mutators all --repeats 3

--mutators runs each chosen attack plain AND disguised, so the two can be compared. Without
ids or --all it uses every attack with no documents. Names: base64, leetspeak, homoglyph,
payload-split, translate-zh, or "all", separated by commas.

Every run first prints a worst-case estimate. --price-as shows (and lets --max-cost cap) what
the run would cost at a hosted model's prices; requests still go only to local Ollama.
A run over a cap is refused unless --partial, which runs until the cap is reached.

With --benign, "succeeded" means the reply passed the task check. Each finished attempt is appended
to runs/<time>.jsonl straight away, so stopping a long run keeps what it already did. The
printed summary only counts outcomes; use examples/summarize_run.py for rates.
"""

import argparse
import asyncio
import sys
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx

from redline.attacks import Mutator, bundled_mutators, load_corpus, make_variants
from redline.benign import control_mutators, control_tasks, controls_hash, load_tasks
from redline.core import Case, Outcome, RunResult
from redline.demo import VulnBotFactory
from redline.metrics import total_usage
from redline.runner import (
    Pricing,
    RunConfig,
    RunEstimate,
    RunReport,
    estimate_run,
    measured_seconds_per_attempt,
    run_attacks,
)
from redline.targets import OpenAICompatConfig, RateLimiter

DEFAULT_ATTACK_IDS = ["di-override-01", "spl-repeat-above-01", "sid-polite-01", "ii-review-note-01"]
RUNS_DIR = Path("runs")
# Ask for --yes above this worst-case cost.
CONFIRM_ABOVE_USD = 0.50
PRICES = {
    "local": Pricing.free(),
    # Paid tier, read 2026-11-27 from https://aistudio.google.com/docs/pricing. Check before use.
    "gemini-3.5-flash-lite": Pricing(input_per_mtok=0.30, output_per_mtok=2.50),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attacks or benign tasks on local VulnBot.")
    parser.add_argument("ids", nargs="*", help="attack ids, or task ids with --benign")
    parser.add_argument("--all", action="store_true", help="run every corpus attack")
    parser.add_argument(
        "--benign", action="store_true", help="run benign tasks (all of them unless ids are given)"
    )
    parser.add_argument("--mutators", help='disguises to compare, e.g. "base64,homoglyph" or "all"')
    parser.add_argument(
        "--controls",
        action="store_true",
        help="run the readability controls (narrow them with --mutators)",
    )
    parser.add_argument("--channel", choices=["prompt", "tool"], default="prompt")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=0)
    # Above 0 so that repeats can differ. At 0 they are nearly the same sample counted twice.
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-requests", type=int, help="cap on model calls for the run")
    parser.add_argument("--max-cost", type=float, help="cap in US dollars (see --price-as)")
    parser.add_argument("--rpm", type=float, help="limit requests per minute")
    parser.add_argument("--price-as", choices=list(PRICES), default="local")
    parser.add_argument(
        "--partial", action="store_true", help="start even if the worst case is over a cap"
    )
    parser.add_argument(
        "--yes", action="store_true", help=f"allow a worst case over ${CONFIRM_ABOVE_USD:.2f}"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the estimate and stop")
    return parser.parse_args()


def describe_result(result: RunResult) -> str:
    detail = ""
    if result.response is not None:
        detail = f"{result.response.latency_ms / 1000:5.1f}s"
        hits = [f"{d.detector}:{d.field}" for d in result.detections if d.triggered]
        if hits:
            detail += f"  triggered: {', '.join(hits)}"
    # Not `elif`: an attempt can have a reply AND an error (a detector crashed).
    if result.error is not None:
        detail = f"{detail}  {result.error[:100]}".strip()
    return (
        f"{result.attack_id:34} attempt {result.attempt} (seed {result.seed})  "
        f"{result.outcome:13} {detail}"
    )


def previous_speed(factory: VulnBotFactory, cases_hash: str) -> float | None:
    """Seconds per attempt from the most useful saved run with exactly these bot settings.

    Prefers a run of the same case set (tasks answer faster than attacks), then the one with
    the most attempts (a tiny run is mostly model loading), then the newest.
    """
    found: list[tuple[bool, int, str, float]] = []
    for path in RUNS_DIR.glob("*.report.json"):
        try:
            report = RunReport.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            continue  # an unreadable report only means one fewer measurement
        seconds = measured_seconds_per_attempt(report, factory)
        if seconds is not None:
            same_set = report.cases_hash == cases_hash
            found.append((same_set, len(report.results), path.name, seconds))
    return max(found)[3] if found else None


def print_estimate(estimate: RunEstimate, price_name: str) -> None:
    if estimate.tokens_known:
        size = (
            f"{estimate.input_tokens:,} input + {estimate.output_tokens:,} output tokens, "
            f"${estimate.cost_usd:.4f} at {price_name} prices"
        )
    else:
        size = "tokens and cost unknown"
    print(
        f"worst case (a ceiling guess, usually far above real use): {estimate.requests} model "
        f"calls, {size} (retries not included)"
    )
    if estimate.seconds is None:
        print("time: unknown (no earlier run with these settings)")
    else:
        print(f"time: about {estimate.seconds / 60:.1f} min, based on an earlier run")


def print_usage(report: RunReport, pricing: Pricing, price_name: str) -> None:
    used = total_usage(report.results)
    cost = pricing.cost_usd(used.input_tokens, used.output_tokens)
    print(
        f"used{'' if used.complete else ' AT LEAST'}: {used.requests} model calls, "
        f"{used.input_tokens:,} input + {used.output_tokens:,} output tokens, ${cost:.4f} at "
        f"{price_name} prices"
    )
    if not used.complete:
        print(
            f"  not recorded: {used.unrecorded_attempts} of {used.attempts} attempts and "
            f"{used.failed_tries} failed tries"
        )


async def run(
    cases: list[Case],
    backend: OpenAICompatConfig,
    args: argparse.Namespace,
    config: RunConfig,
    cases_hash: str,
    on_result: Callable[[RunResult], None],
) -> RunReport:
    limiter = None if args.rpm is None else RateLimiter(args.rpm)
    # One shared connection pool for the whole run.
    async with httpx.AsyncClient() as client:
        return await run_attacks(
            cases,
            VulnBotFactory(backend, args.channel, client=client, limiter=limiter),
            config,
            cases_hash=cases_hash,
            on_result=on_result,
            pricing=PRICES[args.price_as],
            allow_over_estimate=args.partial,
        )


def choose_mutators(names: str, available: tuple[Mutator, ...]) -> list[Mutator]:
    by_name = {mutator.name: mutator for mutator in available}
    wanted = list(by_name) if names == "all" else names.split(",")
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        sys.exit(f"unknown mutator(s) {unknown}. Choose from: all, {', '.join(by_name)}")
    return [by_name[name] for name in wanted]


def select_cases(args: argparse.Namespace) -> tuple[list[Case], str, str]:
    """What to run, what to call one of them, and the fingerprint of where they came from."""
    if args.controls:
        if args.benign or args.all or args.ids:
            sys.exit("--controls picks its own tasks: only --mutators can narrow it")
        chosen = (
            None if args.mutators is None else choose_mutators(args.mutators, control_mutators())
        )
        return list(control_tasks(chosen)), "task", controls_hash()
    if args.benign and args.mutators:
        sys.exit("--mutators disguises attacks, not benign tasks")
    if args.all and args.ids:
        sys.exit("give either --all or ids, not both")

    by_id: dict[str, Case]
    if args.benign:
        task_set = load_tasks()
        by_id = {task.id: task for task in task_set.tasks}
        noun, cases_hash = "task", task_set.content_hash
        # Tasks default to the whole set, so utility isn't quietly based on a few of them.
        default_ids = list(by_id)
    else:
        corpus = load_corpus()
        by_id = {attack.id: attack for attack in corpus.attacks}
        noun, cases_hash = "attack", corpus.content_hash
        if args.all:
            default_ids = list(by_id)
        elif args.mutators:
            default_ids = [a.id for a in corpus.attacks if not a.documents]
        else:
            default_ids = DEFAULT_ATTACK_IDS
    ids = args.ids or default_ids
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        sys.exit(f"unknown {noun} id(s) {unknown}. Choose from: {', '.join(sorted(by_id))}")
    cases = [by_id[i] for i in ids]
    if args.mutators:
        mutators = choose_mutators(args.mutators, bundled_mutators())
        plain = [corpus_attack for corpus_attack in corpus.attacks if corpus_attack.id in ids]
        cases += make_variants(plain, mutators)
    return cases, noun, cases_hash


def main() -> None:
    args = parse_args()
    cases, noun, cases_hash = select_cases(args)

    backend = OpenAICompatConfig(
        base_url="http://localhost:11434/v1",
        model="qwen3:8b",
        temperature=args.temperature,
        timeout_s=180,
        # Thinking off, as in the other examples. It is saved in the report's target settings.
        extra_body={"reasoning_effort": "none", "max_tokens": 300},
    )
    config = RunConfig(
        repeats=args.repeats,
        base_seed=args.base_seed,
        max_requests=args.max_requests,
        max_cost_usd=args.max_cost,
    )
    total = len(cases) * config.repeats

    pricing = PRICES[args.price_as]
    factory = VulnBotFactory(backend, args.channel)
    estimate = estimate_run(
        cases,
        factory,
        config,
        pricing,
        seconds_per_attempt=previous_speed(factory, cases_hash),
        requests_per_minute=args.rpm,
    )
    print_estimate(estimate, args.price_as)
    if args.dry_run:
        return
    # run_attacks enforces the caps too. Checking here avoids leaving empty result files.
    over_requests = args.max_requests is not None and estimate.requests > args.max_requests
    over_cost = args.max_cost is not None and estimate.cost_usd > args.max_cost
    if (over_requests or over_cost) and not args.partial:
        sys.exit("refused: the worst case is over your cap. Run less, raise it, or use --partial")
    # A cost cap at or below the threshold already guarantees the spend stays small.
    small_cap = args.max_cost is not None and args.max_cost <= CONFIRM_ABOVE_USD
    if estimate.cost_usd > CONFIRM_ABOVE_USD and not small_cap and not args.yes:
        sys.exit(f"worst case is over ${CONFIRM_ABOVE_USD:.2f}. Add --yes to run it.")

    RUNS_DIR.mkdir(exist_ok=True)
    stem = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    results_path = RUNS_DIR / f"{stem}.jsonl"
    print(
        f"{len(cases)} {noun}s x {config.repeats} = {total} attempts, {args.channel} channel, "
        f"{backend.model} (thinking off, temperature {backend.temperature}). Local: $0."
    )

    done = 0
    with results_path.open("w", encoding="utf-8") as out:

        def save(result: RunResult) -> None:
            nonlocal done
            done += 1
            out.write(result.model_dump_json() + "\n")
            out.flush()
            print(f"[{done}/{total}] {describe_result(result)}", flush=True)

        report = asyncio.run(run(cases, backend, args, config, cases_hash, save))

    # The whole report (settings, hashes, times, results) goes in a second file.
    report_path = RUNS_DIR / f"{stem}.report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    seconds = (report.finished_at - report.started_at).total_seconds()
    counts = Counter(result.outcome for result in report.results)
    print(
        f"\nconfig hash {report.config_hash}, {noun} set {report.cases_hash[:12]}, {seconds:.0f}s"
    )
    # For tasks this is the checker's verdict; the utility figure is stricter (summarize_run.py).
    meaning = (
        "succeeded = reply passed the task check" if noun == "task" else "succeeded = attack worked"
    )
    print(f"outcomes (raw counts, not a rate; {meaning}):")
    for outcome in Outcome:
        print(f"  {outcome:13} {counts[outcome]}")
    print_usage(report, pricing, args.price_as)
    print(f"saved: {results_path} and {report_path}")


if __name__ == "__main__":
    main()
