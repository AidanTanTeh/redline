"""Forgiving text matching, shared by the benign-task model and its detector."""

import re

# Hyphen look-alikes models like to use: ‐ ‑ ‒ – — ―  and the minus sign.
_DASHES = re.compile("[‐-―−]")


def loose_text(value: str) -> str:
    """Ignore differences that don't change meaning: case, "**bold**" stars, which kind of
    dash was used, spaces around a dash, and runs of whitespace."""
    value = _DASHES.sub("-", value.casefold().replace("*", ""))
    value = re.sub(r"\s*-\s*", "-", value)
    return " ".join(value.split())


def contains_phrase(text: str, phrase: str) -> bool:
    """True if `phrase` appears in `text` as whole words, so "5 days" is not in "15 days"."""
    wanted = loose_text(phrase)
    if not wanted:
        return False  # an empty phrase would "appear" everywhere
    # (?<!\w) and (?!\w): no letter or digit may touch either end of the match.
    return re.search(rf"(?<!\w){re.escape(wanted)}(?!\w)", loose_text(text)) is not None
