import pytest

from redline.attacks import Base64Mutator, LeetspeakMutator, Mutator, build_variant
from redline.core import Attack, Outcome, RunResult
from redline.metrics import compare_mutators, summarize
from tests.metrics.test_asr import NO, OK, attack, results

A, B, C = attack("a"), attack("b"), attack("c")


def disguised(base_id: str, mutator: str) -> str:
    return f"{base_id}--{mutator}"


def variants(mutator: str) -> list[Attack]:
    return [build_variant(base, mutator, lambda text: text + "!") for base in (A, B, C)]


def test_rates_are_compared_over_the_same_attacks_only() -> None:
    run: list[RunResult] = [
        *results("a", OK, OK),
        *results("b", NO, NO),
        *results("c", OK, OK),
        *results(disguised("a", "x"), NO, NO),
        *results(disguised("b", "x"), OK, NO),
        # No verdict at all, so "c" must drop out of BOTH sides.
        *results(disguised("c", "x"), Outcome.NO_ANSWER, Outcome.NO_ANSWER),
    ]

    (comparison,) = compare_mutators(run, [A, B, C, *variants("x")])

    assert (comparison.mutator, comparison.tried, comparison.pairs) == ("x", 3, 2)
    assert comparison.plain.asr == 0.5
    assert comparison.disguised.asr == 0.25
    assert (comparison.higher, comparison.lower, comparison.same) == (1, 1, 0)
    # The dropped attack's attempts are still visible.
    assert comparison.counts[Outcome.NO_ANSWER] == 2


def test_each_mutator_is_reported_separately() -> None:
    run = [
        *results("a", NO),
        *results(disguised("a", "y"), OK),
        *results(disguised("a", "x"), NO),
    ]

    found = compare_mutators(run, [A, *variants("x"), *variants("y")])

    assert [(c.mutator, c.higher, c.same) for c in found] == [("x", 0, 1), ("y", 1, 0)]


def test_a_disguise_with_no_plain_results_has_no_pairs() -> None:
    run = results(disguised("a", "x"), OK)

    (comparison,) = compare_mutators(run, [A, *variants("x")])

    assert (comparison.tried, comparison.pairs, comparison.plain.asr) == (1, 0, None)


def test_a_plain_only_run_has_nothing_to_compare() -> None:
    assert compare_mutators(results("a", OK), [A]) == ()


def test_a_variant_made_from_a_different_version_of_the_attack_is_rejected() -> None:
    edited = A.model_copy(update={"prompt": "edited"})
    run = [*results("a", OK), *results(disguised("a", "x"), OK)]

    with pytest.raises(ValueError, match="different 'a'"):
        compare_mutators(run, [edited, *variants("x")])


@pytest.mark.parametrize("mutator", [Base64Mutator(), LeetspeakMutator()])
def test_summarize_refuses_to_mix_plain_and_disguised(mutator: Mutator) -> None:
    base = A.model_copy(update={"prompt": "password"})
    variant = mutator.mutate(base)
    run = [*results("a", OK), *results(variant.id, NO)]

    with pytest.raises(ValueError, match="mix"):
        summarize(run, [base, variant])


def test_summarize_accepts_one_disguise_on_its_own() -> None:
    run = [*results(disguised("a", "x"), OK), *results(disguised("b", "x"), NO)]

    assert summarize(run, variants("x")).overall.asr == 0.5
