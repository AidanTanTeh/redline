"""Run corpus attacks, or benign tasks, against VulnBot through the runner and save every result.

Only attacks VulnBot, our own demo target, on local Ollama. Costs $0 (no API key).
Run from the repo root:
    uv run python examples/corpus_run.py                          # 4 attacks x 2 repeats
    uv run python examples/corpus_run.py --repeats 3 di-override-01 ii-email-01
    uv run python examples/corpus_run.py --channel tool ii-review-note-01
    uv run python examples/corpus_run.py --all --repeats 3      # every attack (slow)
    uv run python examples/corpus_run.py --benign --repeats 3   # every benign task

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

from redline.attacks import load_corpus
from redline.benign import load_tasks
from redline.core import Case, Outcome, RunResult
from redline.demo import Channel, VulnBotFactory
from redline.runner import RunConfig, RunReport, run_attacks
from redline.targets import OpenAICompatConfig

DEFAULT_ATTACK_IDS = ["di-override-01", "spl-repeat-above-01", "sid-polite-01", "ii-review-note-01"]
RUNS_DIR = Path("runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attacks or benign tasks on local VulnBot.")
    parser.add_argument("ids", nargs="*", help="attack ids, or task ids with --benign")
    parser.add_argument("--all", action="store_true", help="run every corpus attack")
    parser.add_argument(
        "--benign", action="store_true", help="run benign tasks (all of them unless ids are given)"
    )
    parser.add_argument("--channel", choices=["prompt", "tool"], default="prompt")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=0)
    # Above 0 so that repeats can differ. At 0 they are nearly the same sample counted twice.
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def describe_result(result: RunResult) -> str:
    detail = ""
    if result.response is not None:
        detail = f"{result.response.latency_ms / 1000:5.1f}s"
        hits = [f"{d.detector}:{d.field}" for d in result.detections if d.triggered]
        if hits:
            detail += f"  triggered: {', '.join(hits)}"
    elif result.error is not None:
        detail = result.error[:100]
    return (
        f"{result.attack_id:22} attempt {result.attempt} (seed {result.seed})  "
        f"{result.outcome:13} {detail}"
    )


async def run(
    cases: list[Case],
    backend: OpenAICompatConfig,
    channel: Channel,
    config: RunConfig,
    cases_hash: str,
    on_result: Callable[[RunResult], None],
) -> RunReport:
    # One shared connection pool for the whole run.
    async with httpx.AsyncClient() as client:
        return await run_attacks(
            cases,
            VulnBotFactory(backend, channel, client=client),
            config,
            cases_hash=cases_hash,
            on_result=on_result,
        )


def main() -> None:
    args = parse_args()
    by_id: dict[str, Case]
    if args.benign:
        task_set = load_tasks()
        by_id = {task.id: task for task in task_set.tasks}
        noun, cases_hash = "task", task_set.content_hash
    else:
        corpus = load_corpus()
        by_id = {attack.id: attack for attack in corpus.attacks}
        noun, cases_hash = "attack", corpus.content_hash
    if args.all and args.ids:
        sys.exit("give either --all or ids, not both")
    # Tasks default to the whole set, so a utility figure isn't quietly based on a few of them.
    default_ids = list(by_id) if args.all or args.benign else DEFAULT_ATTACK_IDS
    ids = args.ids or default_ids
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        sys.exit(f"unknown {noun} id(s) {unknown}. Choose from: {', '.join(sorted(by_id))}")
    cases = [by_id[i] for i in ids]

    backend = OpenAICompatConfig(
        base_url="http://localhost:11434/v1",
        model="qwen3:8b",
        temperature=args.temperature,
        timeout_s=180,
        # Thinking off, as in the other examples. It is saved in the report's target settings.
        extra_body={"reasoning_effort": "none", "max_tokens": 300},
    )
    config = RunConfig(repeats=args.repeats, base_seed=args.base_seed)
    total = len(cases) * config.repeats

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

        report = asyncio.run(run(cases, backend, args.channel, config, cases_hash, save))

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
        "succeeded = reply passed the task check" if args.benign else "succeeded = attack worked"
    )
    print(f"outcomes (raw counts, not a rate; {meaning}):")
    for outcome in Outcome:
        print(f"  {outcome:13} {counts[outcome]}")
    print(f"saved: {results_path} and {report_path}")


if __name__ == "__main__":
    main()
