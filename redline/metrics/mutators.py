"""Does a disguise change how often attacks succeed? Compared pair by pair."""

from collections.abc import Sequence

from redline.core import Attack, Outcome, RedlineModel, RunResult
from redline.metrics.asr import GroupStats, attack_stats, disguise, group_stats
from redline.metrics.checks import check_run


class MutatorComparison(RedlineModel):
    """One disguise against the same attacks undisguised."""

    mutator: str
    # Every attempt of this disguise, including attacks that couldn't be compared. Many
    # "no_answer" here means the rates below describe only the attacks that got through.
    counts: dict[Outcome, int]
    # Disguised attacks tried, and how many have a verdict both plain and disguised.
    tried: int
    pairs: int
    # Both over the paired attacks only.
    plain: GroupStats
    disguised: GroupStats
    # Paired attacks whose rate went up, down, or stayed the same when disguised.
    higher: int
    lower: int
    same: int


def compare_mutators(
    results: Sequence[RunResult], attacks: Sequence[Attack]
) -> tuple[MutatorComparison, ...]:
    """One comparison per disguise in the run, sorted by name.

    The plain attacks must be in the same run, so both sides used the same settings. An
    attack counts only if both versions reached a verdict: otherwise the two rates would
    cover different attacks. Raises ValueError for the same problems as `summarize`.
    """
    by_id = {a.id: a for a in attacks}
    check_run(results, by_id.keys(), "attack")
    stats = {s.attack_id: s for s in attack_stats(results, by_id)}
    mutators = sorted({m for i in stats if (m := disguise(by_id[i])) is not None})

    comparisons = []
    for mutator in mutators:
        variants = [s for i, s in stats.items() if disguise(by_id[i]) == mutator]
        paired = []
        for variant in variants:
            origin = by_id[variant.attack_id].variant_of
            assert origin is not None  # noqa: S101 - guaranteed by the mutator filter above
            base = origin.base
            if by_id.get(base.id, base) != base:
                raise ValueError(f"{variant.attack_id} was made from a different {base.id!r}")
            plain = stats.get(base.id)
            if plain is not None and plain.asr is not None and variant.asr is not None:
                paired.append((plain, variant, variant.asr - plain.asr))
        comparisons.append(
            MutatorComparison(
                mutator=mutator,
                counts={o: sum(v.counts[o] for v in variants) for o in Outcome},
                tried=len(variants),
                pairs=len(paired),
                plain=group_stats([plain for plain, _, _ in paired]),
                disguised=group_stats([variant for _, variant, _ in paired]),
                higher=sum(change > 0 for _, _, change in paired),
                lower=sum(change < 0 for _, _, change in paired),
                same=sum(change == 0 for _, _, change in paired),
            )
        )
    return tuple(comparisons)
