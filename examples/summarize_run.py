"""Print attack success rates for a saved run. Reads files only; calls no model.

Run from the repo root:
    uv run python examples/summarize_run.py                          # the newest report
    uv run python examples/summarize_run.py runs/<time>.report.json

Columns: ASR = succeeded / (succeeded + resisted), averaged over attacks. "95% interval"
reflects how few attacks were tried. "if undecided" is the range if the attempts with no
verdict had all resisted or all succeeded. Never quote a rate without its counts.
"""

import sys
from pathlib import Path

from redline.attacks import load_corpus
from redline.core import Outcome
from redline.metrics import GroupStats, summarize
from redline.runner import RunReport

RUNS_DIR = Path("runs")


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def span(low: float | None, high: float | None) -> str:
    return "n/a" if low is None or high is None else f"{low:.0%}-{high:.0%}"


def row(label: str, stats: GroupStats) -> str:
    interval = stats.interval
    counts = " ".join(f"{stats.counts[outcome]:>4}" for outcome in Outcome)
    broken = f"{stats.attacks_broken}/{stats.attacks_measured}"
    return (
        f"{label:40} {percent(stats.asr):>4}  "
        f"{span(interval.low, interval.high) if interval else 'n/a':>10}  "
        f"{span(stats.lowest, stats.highest):>12}  {broken:>6}  {counts}"
    )


def newest_report() -> Path:
    reports = sorted(RUNS_DIR.glob("*.report.json"))  # names start with the time
    if not reports:
        sys.exit("no reports in runs/. Run examples/corpus_run.py first.")
    return reports[-1]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_report()
    report = RunReport.model_validate_json(path.read_text(encoding="utf-8"))
    corpus = load_corpus()
    summary = summarize(report.results, corpus.attacks)

    backend = report.target.get("backend", {})
    print(f"report: {path}")
    print(
        f"model {backend.get('model')}, channel {report.target.get('channel')}, "
        f"temperature {backend.get('temperature')}, options {backend.get('extra_body')}"
    )
    print(f"started {report.started_at:%Y-%m-%d %H:%M} UTC, config hash {report.config_hash}")
    if report.corpus_hash != corpus.content_hash:
        print(
            "WARNING: the corpus has changed since this run, so attack categories below come "
            "from the CURRENT corpus and may not match what was run."
        )

    names = ("succ", "resi", "noan", "notd", "skip", "err")
    header = (
        f"\n{'group':40} {'ASR':>4}  {'95% int.':>10}  {'if undecided':>12}  {'broken':>6}  "
        + " ".join(f"{name:>4}" for name in names)
    )
    print(header)
    for category in summary.categories:
        print(row(f"{category.category} ({category.owasp_id})", category.stats))
    print(row("overall (depends on the attack mix)", summary.overall))

    print("\nper attack (succeeded / verdicts, of all attempts):")
    for attack in summary.attacks:
        print(
            f"  {attack.attack_id:24} {attack.counts[Outcome.SUCCEEDED]}/{attack.decided} "
            f"of {attack.attempts}"
        )


if __name__ == "__main__":
    main()
