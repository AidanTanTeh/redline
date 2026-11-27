"""Read secrets from the environment without letting them reach an error message."""

import os
from collections.abc import Sequence
from typing import Any

from redline.core import TargetResponse

REDACTED = "[redacted]"
# Shorter values aren't worth replacing: they would blank out ordinary words in replies.
MIN_REDACT_LENGTH = 8


def redact(response: TargetResponse, secrets: Sequence[str]) -> TargetResponse:
    """The response with any secret WE SENT blanked out, in case the server echoed it back.

    Replies are saved to disk. Only exact copies are caught, not encoded or partial ones.
    """
    # Longest first, so a secret that contains another is replaced whole.
    targets = sorted((s for s in secrets if len(s) >= MIN_REDACT_LENGTH), key=len, reverse=True)
    if not targets:
        return response

    def scrub(value: Any) -> Any:
        if isinstance(value, str):
            for secret in targets:
                value = value.replace(secret, REDACTED)
            return value
        if isinstance(value, dict):
            return {scrub(key): scrub(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [scrub(item) for item in value]
        return value

    return TargetResponse.model_validate(scrub(response.model_dump()))


def is_safe_header_text(value: str) -> bool:
    """Plain printable ASCII with no space at either end.

    The HTTP library rejects anything else, and QUOTES it in the error, which the runner would
    save with the results.
    """
    return value.isascii() and value.isprintable() and value == value.strip()


def read_secret(name: str) -> str:
    """The value of environment variable `name`, checked as safe to send in a header.

    Errors name the variable and never show any part of the value.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set (or is empty).")
    if not is_safe_header_text(value):
        raise RuntimeError(
            f"Environment variable {name} can't go in a header: remove any spaces at the "
            "ends, line breaks or unusual characters."
        )
    return value
