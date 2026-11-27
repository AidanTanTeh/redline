"""What one run produced. Separate from the runner so cost estimates can read old reports."""

from datetime import datetime
from typing import Any

from redline.core import RedlineModel, RunResult


class RunReport(RedlineModel):
    """Everything one run produced, plus what is needed to reproduce it."""

    # In attack order, then attempt order, however the attempts happened to finish.
    results: tuple[RunResult, ...]
    config_hash: str
    # Fingerprint of the files the cases (attacks or benign tasks) were loaded from.
    cases_hash: str
    # The factory's description of the target setup (model, temperature, channel, ...).
    target: dict[str, Any]
    started_at: datetime
    finished_at: datetime
