"""Run corpus attacks against VulnBot through the runner and save every result.

Only attacks VulnBot, our own demo target, on local Ollama. Costs $0 (no API key).
Run from the repo root:
    uv run python examples/corpus_run.py                          # 4 attacks x 2 repeats
    uv run python examples/corpus_run.py --repeats 3 di-override-01 ii-email-01
    uv run python examples/corpus_run.py --channel tool ii-review-note-01
    uv run python examples/corpus_run.py --all --repeats 3      # every attack (slow)

Each finished attempt is appended to runs/<time>.jsonl straight away, so stopping a long run
keeps what it already did. The summary counts outcomes. It is NOT an attack success rate:
the sample is tiny, and rates with confidence intervals arrive in step 3.2.
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
from redline.core import Attack, Outcome, RunResult
from redline.demo import Channel, VulnBotFactory
from redline.runner import RunConfig, RunReport, run_attacks
from redline.targets import OpenAICompatConfig

DEFAULT_ATTACK_IDS = ["di-override-01", "spl-repeat-above-01", "sid-polite-01", "ii-review-note-01"]
RUNS_DIR = Path("runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attacks against local VulnBot.")
    parser.add_argument("attack_ids", nargs="*", default=DEFAULT_ATTACK_IDS)
    parser.add_argument("--all", action="store_true", help="run every corpus attack")
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
    attacks: list[Attack],
    backend: OpenAICompatConfig,
    channel: Channel,
    config: RunConfig,
    corpus_hash: str,
    on_result: Callable[[RunResult], None],
) -> RunReport:
    # One shared connection pool for the whole run.
    async with httpx.AsyncClient() as client:
        return await run_attacks(
            attacks,
            VulnBotFactory(backend, channel, client=client),
            config,
            corpus_hash=corpus_hash,
            on_result=on_result,
        )


def main() -> None:
    args = parse_args()
    corpus = load_corpus()
    by_id = {attack.id: attack for attack in corpus.attacks}
    if args.all and args.attack_ids != DEFAULT_ATTACK_IDS:
        sys.exit("give either --all or attack ids, not both")
    attack_ids = list(by_id) if args.all else args.attack_ids
    unknown = [i for i in attack_ids if i not in by_id]
    if unknown:
        sys.exit(f"unknown attack id(s) {unknown}. Choose from: {', '.join(sorted(by_id))}")
    attacks = [by_id[i] for i in attack_ids]

    backend = OpenAICompatConfig(
        base_url="http://localhost:11434/v1",
        model="qwen3:8b",
        temperature=args.temperature,
        timeout_s=180,
        # Thinking off, as in the other examples. It is saved in the report's target settings.
        extra_body={"reasoning_effort": "none", "max_tokens": 300},
    )
    config = RunConfig(repeats=args.repeats, base_seed=args.base_seed)
    total = len(attacks) * config.repeats

    RUNS_DIR.mkdir(exist_ok=True)
    stem = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    results_path = RUNS_DIR / f"{stem}.jsonl"
    print(
        f"{len(attacks)} attacks x {config.repeats} = {total} attempts, {args.channel} channel, "
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

        report = asyncio.run(run(attacks, backend, args.channel, config, corpus.content_hash, save))

    # The whole report (settings, hashes, times, results) goes in a second file.
    report_path = RUNS_DIR / f"{stem}.report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    seconds = (report.finished_at - report.started_at).total_seconds()
    counts = Counter(result.outcome for result in report.results)
    print(f"\nconfig hash {report.config_hash}, corpus {report.corpus_hash[:12]}, {seconds:.0f}s")
    print("outcomes (counts of a tiny sample, NOT an attack success rate):")
    for outcome in Outcome:
        print(f"  {outcome:13} {counts[outcome]}")
    print(f"saved: {results_path} and {report_path}")


if __name__ == "__main__":
    main()
