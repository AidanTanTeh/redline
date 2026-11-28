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

An attack run and a task run describe the same target only if their "target settings" match.
"WARNING" lines point out ways an attack run could make the target look safer than it is.

Runs with disguised attacks get a second table. Each row compares one disguise with the
plain versions of the SAME attacks ("pairs": both have a verdict). "up/down/same" counts
the pairs whose success rate rose, fell or stayed put. Small differences may be luck.
"""

import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from redline.attacks import bundled_mutators, load_corpus, make_variants
from redline.benign import control_tasks, controls_hash, load_tasks
from redline.core import Attack, BenignTask, Case, Outcome, RunResult
from redline.metrics import (
    GroupStats,
    Interval,
    UtilityGroup,
    compare_mutators,
    run_warnings,
    summarize,
    summarize_utility,
    total_usage,
)
from redline.runner import RunConfig, RunReport, config_hash

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
    names = (first, "resi", "noan", "notd", "skip", "err", "notr")
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


def warn_if_definitions_differ(report: RunReport, cases: Sequence[Case]) -> None:
    """Rebuild the run's config hash from today's definitions of the cases that ran."""
    first = report.results[0]
    if first.seed is None:
        return
    config = RunConfig(
        repeats=1 + max(result.attempt for result in report.results),
        base_seed=first.seed - first.attempt,
    )
    ran = {result.attack_id for result in report.results}
    current = [case for case in cases if case.id in ran]
    if config_hash(report.target, report.cases_hash, current, config) != report.config_hash:
        print(
            "WARNING: today's definitions of these cases (or translations, or mutators) don't "
            "match what was run, so the grouping below may be wrong."
        )


def print_mutators(results: Sequence[RunResult], attacks: Sequence[Attack]) -> None:
    comparisons = compare_mutators(results, attacks)
    if not comparisons:
        return
    print(
        f"\n{'disguise vs plain':16} {'tried':>5} {'pairs':>5}  {'plain':>5} {'disg.':>5}  "
        f"{'up/down/same':>12}  " + outcome_headings("succ")
    )
    for c in comparisons:
        changes = f"{c.higher}/{c.lower}/{c.same}"
        print(
            f"{c.mutator:16} {c.tried:>5} {c.pairs:>5}  {percent(c.plain.asr):>5} "
            f"{percent(c.disguised.asr):>5}  {changes:>12}  {outcome_counts(c.counts)}"
        )
    print("(outcome counts cover every disguised attempt, paired or not)")


def print_asr(results: Sequence[RunResult], attacks: Sequence[Attack]) -> None:
    summary = summarize(results, attacks)
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


def print_controls(results: Sequence[RunResult], tasks: Sequence[BenignTask]) -> None:
    """How often the target answered the easy questions, per disguise."""
    groups: dict[str, list[RunResult]] = {}
    for result in results:
        name = result.attack_id.partition("--")[2] or "plain"
        groups.setdefault(name, []).append(result)
    print(f"\n{'control':16} {'passed':>7}  " + outcome_headings("done"))
    for name, members in sorted(groups.items(), key=lambda item: item[0] != "plain"):
        counts = summarize_utility(members, tasks).overall.counts
        passed = f"{counts[Outcome.SUCCEEDED]}/{len(members)}"
        print(f"{name:16} {passed:>7}  {outcome_counts(counts)}")
    print("A disguise the target often fails here can't be judged by its attack results.")


def print_usage(report: RunReport) -> None:
    used = total_usage(report.results)
    if used.attempts and used.unrecorded_attempts == used.attempts and not used.requests:
        print("usage: not recorded")
        return
    print(
        f"usage{'' if used.complete else ' (AT LEAST; some is unrecorded)'}: "
        f"{used.requests} model calls, {used.input_tokens:,} input + "
        f"{used.output_tokens:,} output tokens"
    )
    if used.judged:
        print(
            f"judge usage: {used.judge_requests} model calls, {used.judge_input_tokens:,} "
            f"input + {used.judge_output_tokens:,} output tokens, "
            f"{used.unrecorded_verdicts} verdicts unrecorded"
        )


def target_line(target: dict[str, Any]) -> str:
    """One line saying what was tested. Each kind of target saves different settings."""
    if target.get("target") == "python":
        return (
            f"your function {target.get('name')!r}, version {target.get('version')!r}, "
            f"model {target.get('model')}, documents via {target.get('documents')}"
        )
    if target.get("target") == "http":
        return f"web app on {target.get('host')}, model {target.get('model')}"
    backend = target.get("backend", {})
    return (
        f"model {backend.get('model')}, channel {target.get('channel')}, "
        f"temperature {backend.get('temperature')}, options {backend.get('extra_body')}"
    )


def newest_report() -> Path:
    reports = sorted(RUNS_DIR.glob("*.report.json"))  # names start with the time
    if not reports:
        sys.exit("no reports in runs/. Run examples/corpus_run.py first.")
    return reports[-1]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_report()
    report = RunReport.model_validate_json(path.read_text(encoding="utf-8"))

    print(f"report: {path}")
    print(target_line(report.target))
    print(f"started {report.started_at:%Y-%m-%d %H:%M} UTC, config hash {report.config_hash}")
    print(f"target settings {settings_fingerprint(report)}")
    print_usage(report)

    run_ids = {result.attack_id for result in report.results}
    if not run_ids:
        sys.exit("this report has no results")
    not_run = sum(result.outcome is Outcome.NOT_RUN for result in report.results)
    if not_run:
        print(
            f"WARNING: INCOMPLETE RUN. A cap stopped {not_run} of {len(report.results)} "
            "attempts, so the rates below cover only part of what was asked for."
        )
    corpus = load_corpus()
    task_set = load_tasks()
    attacks = (*corpus.attacks, *make_variants(corpus.attacks, bundled_mutators()))
    if run_ids <= {attack.id for attack in attacks}:
        warn_if_changed(report, corpus.content_hash, "attack corpus")
        warn_if_definitions_differ(report, attacks)
        plain_ids = {attack.id for attack in corpus.attacks}
        plain = [result for result in report.results if result.attack_id in plain_ids]
        if plain:
            print_asr(plain, corpus.attacks)
        print_mutators(report.results, attacks)
        external = report.target.get("target") in ("python", "http")
        for warning in run_warnings(report.results, attacks, external_app=external):
            print(f"\nWARNING: {warning}")
    elif run_ids <= {task.id for task in task_set.tasks}:
        warn_if_changed(report, task_set.content_hash, "task set")
        warn_if_definitions_differ(report, task_set.tasks)
        print_utility(report, task_set.tasks)
    elif run_ids <= {task.id for task in control_tasks()}:
        warn_if_changed(report, controls_hash(), "controls")
        warn_if_definitions_differ(report, control_tasks())
        print_controls(report.results, control_tasks())
    else:
        sys.exit("the result ids are neither all current attacks nor all current tasks")


if __name__ == "__main__":
    main()
