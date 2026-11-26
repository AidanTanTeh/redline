from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from redline.core import Attack, AttackCategory, TargetResponse
from redline.runner import RunConfig, Trial, TrialFactory, config_hash

DESCRIPTION: dict[str, Any] = {
    "target": "fake",
    "model": "m",
    "temperature": 0.0,
    "extra_body": {"reasoning_effort": "none", "max_tokens": 300},
}


class EchoTarget:
    name = "echo"

    async def send(self, prompt: str) -> TargetResponse:
        return TargetResponse(text=prompt, model="m", latency_ms=1)


class FakeFactory:
    name = "fake"
    model = "m"
    has_tools = False

    def describe(self) -> Mapping[str, Any]:
        return DESCRIPTION

    def preflight(self) -> None:
        return None

    def make_trial(self, attack: Attack, seed: int) -> Trial:
        return Trial(target=EchoTarget(), message=f"{attack.prompt} #{seed}", detectors=())


# --- Trial / TrialFactory ---------------------------------------------------


def test_a_plain_class_satisfies_the_factory_protocol() -> None:
    # The annotation is the real check: mypy fails here if FakeFactory is missing a member.
    factory: TrialFactory = FakeFactory()
    attack = Attack(
        id="a-1", name="A", category=AttackCategory.DIRECT_INJECTION, prompt="Say hello"
    )
    assert factory.make_trial(attack, 7).message == "Say hello #7"


def test_trial_counts_as_delivered_by_default() -> None:
    trial = Trial(target=EchoTarget(), message="hi", detectors=())
    assert trial.delivered(TargetResponse(text="", model="m", latency_ms=1)) is True


# --- RunConfig --------------------------------------------------------------


def test_run_config_defaults_to_one_attempt_one_at_a_time() -> None:
    config = RunConfig()
    assert (config.repeats, config.max_concurrency, config.max_retries) == (1, 1, 2)


@pytest.mark.parametrize(
    "bad",
    [
        {"repeats": 0},
        {"max_concurrency": 0},
        {"max_retries": -1},
        {"attempt_timeout_s": 0},
        {"base_seed": -1},
    ],
)
def test_run_config_rejects_impossible_settings(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RunConfig(**bad)


def test_each_attempt_gets_its_own_seed() -> None:
    config = RunConfig(base_seed=100, repeats=3)
    assert [config.seed_for(i) for i in range(config.repeats)] == [100, 101, 102]


# --- config_hash ------------------------------------------------------------


def make_attack(attack_id: str, prompt: str = "Say hello") -> Attack:
    return Attack(id=attack_id, name="A", category=AttackCategory.DIRECT_INJECTION, prompt=prompt)


ATTACKS = (make_attack("a-1"), make_attack("a-2"))


def hash_of(
    description: Mapping[str, Any] = DESCRIPTION,
    corpus: str = "corpus-a",
    config: RunConfig | None = None,
    attacks: tuple[Attack, ...] = ATTACKS,
) -> str:
    return config_hash(description, corpus, attacks, config or RunConfig())


def test_hash_is_short_and_ignores_key_order() -> None:
    reordered = {
        "extra_body": {"max_tokens": 300, "reasoning_effort": "none"},
        "temperature": 0.0,
        "model": "m",
        "target": "fake",
    }
    assert len(hash_of()) == 12
    assert hash_of(reordered) == hash_of()


@pytest.mark.parametrize(
    "changed",
    [
        {"temperature": 0.7},
        {"model": "other"},
        {"extra_body": {"reasoning_effort": "high", "max_tokens": 300}},
    ],
)
def test_hash_changes_when_a_target_setting_changes(changed: dict[str, Any]) -> None:
    assert hash_of(DESCRIPTION | changed) != hash_of()


def test_hash_changes_with_the_corpus_repeats_and_seed() -> None:
    assert hash_of(corpus="corpus-b") != hash_of()
    assert hash_of(config=RunConfig(repeats=2)) != hash_of()
    assert hash_of(config=RunConfig(base_seed=1)) != hash_of()


def test_hash_changes_with_which_attacks_run_but_not_their_order() -> None:
    assert hash_of(attacks=ATTACKS[:1]) != hash_of()
    assert hash_of(attacks=(ATTACKS[0], make_attack("a-2", "Say bye"))) != hash_of()
    assert hash_of(attacks=ATTACKS[::-1]) == hash_of()


def test_hash_ignores_how_the_run_is_scheduled() -> None:
    scheduled = RunConfig(max_concurrency=8, max_retries=5, attempt_timeout_s=10)
    assert hash_of(config=scheduled) == hash_of()


def test_hash_refuses_values_that_are_not_json() -> None:
    with pytest.raises(TypeError):
        hash_of({"tool": object()})
