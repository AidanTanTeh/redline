"""Read a folder of YAML files into validated cases, with a fingerprint of the files."""

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from redline.core.models import Case


class CorpusError(Exception):
    """A case folder is missing, malformed, or inconsistent. The message names the file."""


def _format_errors(exc: ValidationError) -> str:
    # Each error has a location like ("attacks", 2, "catgory") and a message. The location is
    # empty when the whole file is wrong (e.g. an empty file), so say that instead of ": ...".
    return "; ".join(
        f"{'.'.join(str(part) for part in err['loc']) or '(whole file)'}: {err['msg']}"
        for err in exc.errors()
    )


def read_yaml_dir[F: BaseModel, C: Case](
    directory: Path,
    file_model: type[F],
    cases_in: Callable[[F], Sequence[C]],
    noun: str,
) -> tuple[tuple[C, ...], str]:
    """Load every `*.yaml` file in `directory`. Returns (cases, sha256 of the files).

    `file_model` is the shape of one file, `cases_in` picks the cases out of it, and `noun`
    ("attack", "task") is used in error messages.

    Raises CorpusError for a missing directory, bad YAML, a file that doesn't match the model,
    duplicate ids, or no cases at all.
    """
    if not directory.is_dir():
        raise CorpusError(f"{noun} directory not found: {directory}")

    digest = hashlib.sha256()
    cases: list[C] = []
    seen_in: dict[str, str] = {}  # case id -> file it first appeared in

    for file in sorted(directory.glob("*.yaml")):
        raw = file.read_bytes()
        # Hash the name too, so renaming a file also counts as a change.
        digest.update(file.name.encode() + b"\0" + raw + b"\0")

        try:
            # safe_load only builds plain data (dict/list/str/...). Plain yaml.load can
            # construct arbitrary Python objects from a crafted file, which is code execution.
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise CorpusError(f"{file.name}: invalid YAML: {exc}") from exc

        try:
            parsed = file_model.model_validate(data)
        except ValidationError as exc:
            raise CorpusError(f"{file.name}: {_format_errors(exc)}") from exc

        for case in cases_in(parsed):
            if case.id in seen_in:
                raise CorpusError(
                    f"{file.name}: duplicate {noun} id {case.id!r} "
                    f"(first seen in {seen_in[case.id]})"
                )
            seen_in[case.id] = file.name
            cases.append(case)

    if not cases:
        raise CorpusError(f"no {noun}s found in {directory}")

    return tuple(cases), digest.hexdigest()
