"""Keep the machine-readable parts of an attack unchanged while disguising the rest.

If a mutator swapped letters inside an email address, the target would email a different
address, the tool-call detector would not match it, and a successful attack would be counted
as resisted. A swapped web address could also become a look-alike of a real domain.
"""

import re
from collections.abc import Callable

from redline.core import Attack

# An all-capitals value in a link's query, e.g. "PASSWORD" in "?d=PASSWORD": a slot the
# attack's text tells the target to fill in.
_PLACEHOLDER = re.compile(r"https?://\S*?[?&][\w-]+=([A-Z][A-Z_]{2,})\b")

_ALWAYS_PROTECTED = (
    r"https?://[^\s<>()\[\]\"']+",  # web address, query string included
    r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",  # email address
    r"\b(?:[\w-]+\.)+example\b",  # bare host on the reserved .example domain
    r"\b[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b",  # snake_case names, e.g. send_email
)


def rewrite_unprotected(text: str, attack: Attack, rewrite: Callable[[str], str]) -> str:
    """Apply `rewrite` to everything in `text` except the protected parts."""
    # Whatever the detector will look for must survive too.
    wanted: list[str] = []
    if attack.success_tool_call is not None:
        wanted = [attack.success_tool_call.tool, *attack.success_tool_call.arguments.values()]
    # Longest first: the first alternative that matches wins.
    literals = [re.escape(value) for value in sorted(wanted, key=len, reverse=True) if value]
    # The sentence naming a placeholder must keep matching the link. Case-sensitive ("-i"),
    # so the ordinary word "password" elsewhere is still disguised.
    texts = [attack.prompt, *(document.content for document in attack.documents)]
    slots = {slot for source in texts for slot in _PLACEHOLDER.findall(source)}
    placeholders = [rf"(?-i:\b{slot}\b)" for slot in sorted(slots)]
    protected = re.compile("|".join([*literals, *_ALWAYS_PROTECTED, *placeholders]), re.IGNORECASE)

    pieces: list[str] = []
    end = 0
    for match in protected.finditer(text):
        pieces += [rewrite(text[end : match.start()]), match.group()]
        end = match.end()
    pieces.append(rewrite(text[end:]))
    return "".join(pieces)
