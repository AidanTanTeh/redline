"""Guard: today's code must still produce the config hashes of the saved baseline runs.

A failure means something that goes into the hash changed: how VulnBot describes itself, a
core model's saved fields, or the text of one of the baseline attacks or tasks. Old and new
results would then no longer count as comparable. If that is intended, record why and update
the expected value here.

New attacks or tasks don't affect this test: only the ids listed below are used, and the
corpus fingerprints are the ones the baselines were run with, not today's.
"""

from collections.abc import Sequence

import pytest

from redline.attacks import load_corpus
from redline.benign import load_tasks
from redline.core import Case
from redline.demo import Channel, VulnBotFactory
from redline.runner import RunConfig, config_hash
from redline.targets import OpenAICompatConfig

# The settings of the 2026-11-26 attack and 2026-11-27 task baselines.
BACKEND = OpenAICompatConfig(
    base_url="http://localhost:11434/v1",
    model="qwen3:8b",
    temperature=0.7,
    timeout_s=180,
    extra_body={"reasoning_effort": "none", "max_tokens": 300},
)
CONFIG = RunConfig(repeats=3, base_seed=0)

CORPUS_HASH = "de084682f4872f7539029ccc9f63f5042756e89068b7d0c57c8d052733a4cead"
TASKS_HASH = "abf5a1abdf1231743cbd59e4c5d6acd88f204caeb7cbfc4e362237683792b3fe"

ATTACK_IDS = [
    "di-authority-01",
    "di-devmode-01",
    "di-fiction-01",
    "di-language-01",
    "di-override-01",
    "di-prefill-01",
    "ea-direct-audit-01",
    "ea-exfil-body-01",
    "ea-manager-01",
    "ea-review-forward-01",
    "ea-silent-01",
    "ii-delimiter-escape-01",
    "ii-email-01",
    "ii-exfil-link-01",
    "ii-fake-policy-01",
    "ii-html-comment-01",
    "ii-review-note-01",
    "sid-admin-claim-01",
    "sid-base64-01",
    "sid-hint-01",
    "sid-list-secrets-01",
    "sid-polite-01",
    "sid-spelled-out-01",
    "sid-urgency-01",
    "spl-codeblock-01",
    "spl-completion-01",
    "spl-config-dump-01",
    "spl-first-line-01",
    "spl-repeat-above-01",
    "spl-summary-01",
    "spl-translate-01",
]

TASK_IDS = [
    "bt-admin-access-01",
    "bt-email-invoice-01",
    "bt-email-late-01",
    "bt-email-room-01",
    "bt-email-shipped-01",
    "bt-express-price-01",
    "bt-ignore-order-01",
    "bt-order-status-01",
    "bt-password-reset-01",
    "bt-read-email-room-01",
    "bt-read-email-tracking-01",
    "bt-return-window-01",
    "bt-shipping-time-01",
    "bt-support-hours-01",
    "bt-warranty-01",
]


def pick(cases: Sequence[Case], ids: list[str]) -> list[Case]:
    by_id = {case.id: case for case in cases}
    return [by_id[i] for i in ids]  # a removed or renamed case fails here with its id


@pytest.mark.parametrize(
    ("channel", "expected"), [("prompt", "951440fb2de2"), ("tool", "979607b402fb")]
)
def test_attack_baseline_hash_is_unchanged(channel: Channel, expected: str) -> None:
    attacks = pick(load_corpus().attacks, ATTACK_IDS)
    description = VulnBotFactory(BACKEND, channel).describe()

    assert config_hash(description, CORPUS_HASH, attacks, CONFIG) == expected


@pytest.mark.parametrize(
    ("channel", "expected"), [("prompt", "9580fbb3ee62"), ("tool", "158700ceed16")]
)
def test_task_baseline_hash_is_unchanged(channel: Channel, expected: str) -> None:
    tasks = pick(load_tasks().tasks, TASK_IDS)
    description = VulnBotFactory(BACKEND, channel).describe()

    assert config_hash(description, TASKS_HASH, tasks, CONFIG) == expected
