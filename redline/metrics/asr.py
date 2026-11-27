"""Attack success rate (ASR) per attack, per category and overall."""

from collections import Counter, defaultdict
from collections.abc import Sequence
from statistics import fmean

from redline.core import Attack, AttackCategory, Outcome, RedlineModel, RunResult
from redline.metrics.checks import check_run
from redline.metrics.stats import Interval, wilson_interval

# The only outcomes that say whether the target held up.
_DECIDED = (Outcome.SUCCEEDED, Outcome.RESISTED)


class AttackStats(RedlineModel):
    """All the attempts of one attack."""

    attack_id: str
    category: AttackCategory
    # Every outcome has an entry, including zeros.
    counts: dict[Outcome, int]

    @property
    def attempts(self) -> int:
        return sum(self.counts.values())

    @property
    def decided(self) -> int:
        return sum(self.counts[outcome] for outcome in _DECIDED)

    @property
    def asr(self) -> float | None:
        """succeeded / decided. None (not 0) when no attempt reached a verdict."""
        return self.counts[Outcome.SUCCEEDED] / self.decided if self.decided else None


class GroupStats(RedlineModel):
    """The rate for a group of attacks (one category, or everything)."""

    counts: dict[Outcome, int]
    attempts: int
    decided: int
    # Attacks in the group; those with at least one verdict; those that ever succeeded.
    attacks: int
    attacks_measured: int
    attacks_broken: int
    # The mean of the measured attacks' rates, so each attack weighs the same.
    asr: float | None
    # Uses n = attacks_measured, because repeats of one attack are not independent. Applying a
    # binomial interval to a mean of rates is an approximation, and a wide (cautious) one.
    interval: Interval | None
    # The interval for attacks_broken / attacks_measured, a plain yes/no count per attack.
    broken_interval: Interval | None
    # What the rate would be if every undecided attempt (not skipped) had resisted / succeeded.
    lowest: float | None
    highest: float | None

    @property
    def coverage(self) -> float:
        """The share of attempts that reached a verdict."""
        return self.decided / self.attempts


class CategoryStats(RedlineModel):
    category: AttackCategory
    owasp_id: str
    stats: GroupStats


class AsrSummary(RedlineModel):
    config_hash: str
    # Depends on the mix of attacks: a corpus with more easy attacks scores higher. Compare
    # categories, or runs over the same attacks, rather than quoting this on its own.
    overall: GroupStats
    categories: tuple[CategoryStats, ...]
    attacks: tuple[AttackStats, ...]


def _group(attacks: Sequence[AttackStats]) -> GroupStats:
    counts = {outcome: sum(a.counts[outcome] for a in attacks) for outcome in Outcome}
    rates = [a.asr for a in attacks if a.asr is not None]
    broken = sum(1 for a in attacks if a.counts[Outcome.SUCCEEDED])

    lows: list[float] = []
    highs: list[float] = []
    for a in attacks:
        # Skipped attempts could never have run on this target, so they bound nothing.
        possible = a.attempts - a.counts[Outcome.SKIPPED]
        if possible:
            succeeded = a.counts[Outcome.SUCCEEDED]
            lows.append(succeeded / possible)
            highs.append((succeeded + possible - a.decided) / possible)

    asr = fmean(rates) if rates else None
    return GroupStats(
        counts=counts,
        attempts=sum(counts.values()),
        decided=sum(a.decided for a in attacks),
        attacks=len(attacks),
        attacks_measured=len(rates),
        attacks_broken=broken,
        asr=asr,
        interval=wilson_interval(asr, len(rates)) if asr is not None else None,
        broken_interval=wilson_interval(broken / len(rates), len(rates)) if rates else None,
        lowest=fmean(lows) if lows else None,
        highest=fmean(highs) if highs else None,
    )


def summarize(results: Sequence[RunResult], attacks: Sequence[Attack]) -> AsrSummary:
    """Turn one run's results into rates. `attacks` supplies each attack's category.

    Raises ValueError rather than produce a number that mixes things that aren't comparable.
    Attacks with no results are ignored, so passing the whole corpus is fine.
    """
    by_id = {a.id: a for a in attacks}
    config_hash = check_run(results, by_id.keys(), "attack")

    outcomes: defaultdict[str, Counter[Outcome]] = defaultdict(Counter)
    for result in results:
        outcomes[result.attack_id][result.outcome] += 1
    per_attack = tuple(
        AttackStats(
            attack_id=attack_id,
            category=by_id[attack_id].category,
            counts={outcome: counted[outcome] for outcome in Outcome},
        )
        for attack_id, counted in sorted(outcomes.items())
    )

    categories = []
    for category in AttackCategory:
        members = [a for a in per_attack if a.category is category]
        if members:
            categories.append(
                CategoryStats(
                    category=category,
                    owasp_id=by_id[members[0].attack_id].owasp_id,
                    stats=_group(members),
                )
            )

    return AsrSummary(
        config_hash=config_hash,
        overall=_group(per_attack),
        categories=tuple(categories),
        attacks=per_attack,
    )
