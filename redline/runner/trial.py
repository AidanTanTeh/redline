"""What the runner needs from a target setup, and the settings that identify a run."""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import Field

from redline.core import Case, RedlineModel, TargetResponse
from redline.detectors import Detector
from redline.targets import Target


def _always_delivered(response: TargetResponse) -> bool:
    return True


@dataclass(frozen=True)
class Trial:
    """Everything needed to run ONE attempt of ONE attack.

    A dataclass, not a Pydantic model: it holds live objects (a target, detectors, a function),
    not data that gets validated or saved.
    """

    target: Target
    # What to send to the target. Already rendered (documents pasted in, or just the question).
    message: str
    # Built for this trial, because some detectors need this trial's secret (the canary).
    detectors: tuple[Detector, ...]
    # Did the attack's documents reach the model? Targets with nothing to deliver keep the default.
    delivered: Callable[[TargetResponse], bool] = _always_delivered


class TrialFactory(Protocol):
    """A target setup the runner can attack. It builds one fresh Trial per attempt.

    The runner knows nothing about canaries, channels or VulnBot: all of that lives behind this
    interface, so the same runner can drive VulnBot, a Python agent or an HTTP app.
    """

    @property
    def name(self) -> str:
        """Label recorded on each result, e.g. "vulnbot/qwen3:8b/tool"."""
        ...

    @property
    def model(self) -> str:
        """The configured model name, recorded on each result."""
        ...

    @property
    def has_tools(self) -> bool:
        """False means attacks that succeed through a tool call can't be tested here."""
        ...

    def describe(self) -> Mapping[str, Any]:
        """Every setting that can change a result. JSON values only; it goes into the hash."""
        ...

    def preflight(self) -> None:
        """Raise now if the run can't work (e.g. a missing API key), not 20 minutes in."""
        ...

    def make_trial(self, attack: Case, seed: int) -> Trial:
        """A fresh target, message and detectors. The same attack and seed give the same prompt."""
        ...


class RunConfig(RedlineModel):
    """How a run is carried out."""

    repeats: int = Field(default=1, ge=1)
    base_seed: int = Field(default=0, ge=0)
    # 1 because local Ollama queues requests (measured in 3.1). Raise it for hosted targets.
    max_concurrency: int = Field(default=1, ge=1)
    # Extra tries after a transient failure (timeout, 429, 5xx).
    max_retries: int = Field(default=2, ge=0)
    # Time limit for one whole attempt, which may be several model calls with tools.
    attempt_timeout_s: float = Field(default=300.0, gt=0)
    # Caps on the whole run; None means no cap. They need a factory that can estimate attempts.
    max_requests: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    def seed_for(self, attempt: int) -> int:
        """Each repeat gets its own seed, so repeats differ but can still be reproduced."""
        return self.base_seed + attempt


def config_hash(
    description: Mapping[str, Any],
    cases_hash: str,
    attacks: Sequence[Case],
    config: RunConfig,
) -> str:
    """A short fingerprint of everything that changes what the model sees.

    Two runs with the same hash are comparable. Concurrency, retries, the timeout and the caps
    are left out: they change how fast, reliably or far a run goes, not what is sent.
    """
    payload = {
        "target": dict(description),
        # The key stays "corpus" although it can be a task set's hash: key names are hashed too,
        # so renaming it would change every config hash and unpair earlier runs.
        "corpus": cases_hash,
        # The attacks actually run: 4 of them is not comparable with all 31, and an attack built
        # in code isn't covered by the corpus hash. Sorted, so their order doesn't matter.
        "attacks": [a.model_dump(mode="json") for a in sorted(attacks, key=lambda a: a.id)],
        "repeats": config.repeats,
        "base_seed": config.base_seed,
    }
    # sort_keys makes key order irrelevant. A non-JSON value raises TypeError on purpose: turning
    # it into a string could bake a memory address in and give a different hash on every run.
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
