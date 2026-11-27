"""Checks every summary runs before turning results into numbers."""

from collections import Counter
from collections.abc import Collection, Sequence

from redline.core import RunResult


def check_run(results: Sequence[RunResult], known_ids: Collection[str], noun: str) -> str:
    """Return the run's config hash, or raise ValueError if these results can't be one rate.

    `known_ids` are the cases the caller has definitions for, and `noun` ("attack", "task")
    names them in the error message.
    """
    if not results:
        raise ValueError("no results to summarize")
    hashes = {r.config_hash for r in results}
    if len(hashes) > 1:
        raise ValueError(f"results come from different setups: {sorted(hashes)}")
    keys = Counter((r.attack_id, r.attempt) for r in results)
    repeated = sorted(key for key, times in keys.items() if times > 1)
    if repeated:
        raise ValueError(f"the same ({noun}, attempt) appears more than once: {repeated}")
    unknown = sorted({r.attack_id for r in results} - set(known_ids))
    if unknown:
        raise ValueError(f"no {noun} definition for: {unknown}")
    return hashes.pop()
