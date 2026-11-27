"""Read secrets from the environment without letting them reach an error message."""

import os


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
