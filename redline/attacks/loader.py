"""Load and validate the YAML attack corpus."""

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import ValidationError

from redline.core import Attack, RedlineModel

# Corpus shipped inside the package. A plain path works because the package is installed
# as real files (not zipped); the YAML files are checked for in the built wheel in chunk 5.
BUNDLED_CORPUS_DIR = Path(__file__).parent / "corpus"


class CorpusError(Exception):
    """The corpus is missing, malformed, or inconsistent. The message names the file."""


class _CorpusFile(RedlineModel):
    """The shape of one YAML file. Validating it does all the checking in one go."""

    # Literal[1] means "exactly the value 1", like `schemaVersion: 1` in a TS type.
    # Bump it (and migrate the files) if the format ever changes in a breaking way.
    schema_version: Literal[1]
    attacks: tuple[Attack, ...]


class Corpus(RedlineModel):
    """All loaded attacks plus a fingerprint of the files they came from."""

    attacks: tuple[Attack, ...]
    # sha256 of the YAML files' names + bytes. Any edit changes it, so results can say
    # exactly which corpus version they were produced with.
    content_hash: str


def _format_errors(exc: ValidationError) -> str:
    # Each error has a location like ("attacks", 2, "catgory") and a message. The location is
    # empty when the whole file is wrong (e.g. an empty file), so say that instead of ": ...".
    return "; ".join(
        f"{'.'.join(str(part) for part in err['loc']) or '(whole file)'}: {err['msg']}"
        for err in exc.errors()
    )


def load_corpus(path: Path | None = None) -> Corpus:
    """Load every `*.yaml` file in `path` (default: the bundled corpus).

    Raises CorpusError for a missing directory, bad YAML, a wrong schema_version, unknown or
    missing fields, duplicate attack ids, or an empty corpus.
    """
    corpus_dir = path if path is not None else BUNDLED_CORPUS_DIR
    if not corpus_dir.is_dir():
        raise CorpusError(f"corpus directory not found: {corpus_dir}")

    files = sorted(corpus_dir.glob("*.yaml"))
    digest = hashlib.sha256()
    attacks: list[Attack] = []
    seen_in: dict[str, str] = {}  # attack id -> file it first appeared in

    for file in files:
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
            parsed = _CorpusFile.model_validate(data)
        except ValidationError as exc:
            raise CorpusError(f"{file.name}: {_format_errors(exc)}") from exc

        for attack in parsed.attacks:
            if attack.id in seen_in:
                raise CorpusError(
                    f"{file.name}: duplicate attack id {attack.id!r} "
                    f"(first seen in {seen_in[attack.id]})"
                )
            seen_in[attack.id] = file.name
            attacks.append(attack)

    if not attacks:
        raise CorpusError(f"no attacks found in {corpus_dir}")

    return Corpus(attacks=tuple(attacks), content_hash=digest.hexdigest())
