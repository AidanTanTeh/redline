"""Print attack success rates, or utility for a benign-task run. Reads files only; calls no model.

Run from the repo root:
    uv run python examples/summarize_run.py                          # the newest report
    uv run python examples/summarize_run.py runs/<time>.report.json

The kind of run is worked out from the ids in its results.

Attack runs: ASR = succeeded / (succeeded + resisted), averaged over attacks. "if undecided"
is the range if the attempts with no verdict had all resisted or all succeeded.
Task runs: utility = completed / all attempts except errors and skips, averaged over tasks.
A document task the bot passed without reading the document counts as "notd", not "done".
"95% int." reflects how few attacks or tasks were tried. Never quote a rate without its counts.

An attack run and a task run describe the same bot only if their "bot settings" match.
"""

import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from redline.attacks import load_corpus
from redline.benign import load_tasks
from redline.core import Attack, BenignTask, Outcome
from redline.metrics import GroupStats, Interval, UtilityGroup, summarize, summarize_utility
from redline.runner import RunReport

RUNS_DIR = Path("runs")


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def span(low: float | None, high: float | None) -> str:
    return "n/a" if low is None or high is None else f"{low:.0%}-{high:.0%}"


def interval_span(interval: Interval | None) -> str:
    return "n/a" if interval is None else span(interval.low, interval.high)


def outcome_counts(counts: dict[Outcome, int]) -> str:
    return " ".join(f"{counts[outcome]:>4}" for outcome in Outcome)


def outcome_headings(first: str) -> str:
    """Column names in `Outcome` order. `first` names SUCCEEDED, which differs by run kind."""
    names = (first, "resi", "noan", "notd", "skip", "err")
    return " ".join(f"{name:>4}" for name in names)


def attack_row(label: str, stats: GroupStats) -> str:
    broken = f"{stats.attacks_broken}/{stats.attacks_measured}"
    return (
        f"{label:40} {percent(stats.asr):>4}  {interval_span(stats.interval):>10}  "
        f"{span(stats.lowest, stats.highest):>12}  {broken:>6}  {outcome_counts(stats.counts)}"
    )


def task_row(label: str, stats: UtilityGroup) -> str:
    failed = f"{stats.tasks_failed}/{stats.tasks_measured}"
    return (
        f"{label:28} {percent(stats.utility):>7}  {interval_span(stats.interval):>10}  "
        f"{failed:>6}  {outcome_counts(stats.counts)}"
    )


def settings_fingerprint(report: RunReport) -> str:
    """A short hash of the bot's settings only (not the cases, repeats or seeds)."""
    canonical = json.dumps(report.target, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def warn_if_changed(report: RunReport, current_hash: str, what: str) -> None:
    if report.cases_hash != current_hash:
        print(
            f"WARNING: the {what} changed since this run, so the grouping below comes from the "
            "CURRENT files and may not match what was run."
        )


def print_asr(report: RunReport, attacks: Sequence[Attack]) -> None:
    summary = summarize(report.results, attacks)
    print(
        f"\n{'group':40} {'ASR':>4}  {'95% int.':>10}  {'if undecided':>12}  {'broken':>6}  "
        + outcome_headings("succ")
    )
    for category in summary.categories:
        print(attack_row(f"{category.category} ({category.owasp_id})", category.stats))
    print(attack_row("overall (depends on the attack mix)", summary.overall))

    print("\nper attack (succeeded / verdicts, of all attempts):")
    for attack in summary.attacks:
        print(
            f"  {attack.attack_id:24} {attack.counts[Outcome.SUCCEEDED]}/{attack.decided} "
            f"of {attack.attempts}"
        )


def print_utility(report: RunReport, tasks: Sequence[BenignTask]) -> None:
    summary = summarize_utility(report.results, tasks)
    print(
        f"\n{'group':28} {'utility':>7}  {'95% int.':>10}  {'failed':>6}  "
        + outcome_headings("done")
    )
    for kind in summary.kinds:
        print(task_row(kind.kind, kind.stats))
    print(task_row("overall (depends on the mix)", summary.overall))

    print("\nper task (completed / scored, of all attempts):")
    for task in summary.tasks:
        print(
            f"  {task.task_id:28} {task.counts[Outcome.SUCCEEDED]}/{task.scored} of {task.attempts}"
        )


def newest_report() -> Path:
    reports = sorted(RUNS_DIR.glob("*.report.json"))  # names start with the time
    if not reports:
        sys.exit("no reports in runs/. Run examples/corpus_run.py first.")
    return reports[-1]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_report()
    report = RunReport.model_validate_json(path.read_text(encoding="utf-8"))

    backend = report.target.get("backend", {})
    print(f"report: {path}")
    print(
        f"model {backend.get('model')}, channel {report.target.get('channel')}, "
        f"temperature {backend.get('temperature')}, options {backend.get('extra_body')}"
    )
    print(f"started {report.started_at:%Y-%m-%d %H:%M} UTC, config hash {report.config_hash}")
    print(f"bot settings {settings_fingerprint(report)}")

    run_ids = {result.attack_id for result in report.results}
    if not run_ids:
        sys.exit("this report has no results")
    corpus = load_corpus()
    task_set = load_tasks()
    if run_ids <= {attack.id for attack in corpus.attacks}:
        warn_if_changed(report, corpus.content_hash, "attack corpus")
        print_asr(report, corpus.attacks)
    elif run_ids <= {task.id for task in task_set.tasks}:
        warn_if_changed(report, task_set.content_hash, "task set")
        print_utility(report, task_set.tasks)
    else:
        sys.exit("the result ids are neither all current attacks nor all current tasks")


if __name__ == "__main__":
    main()
