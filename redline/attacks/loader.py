"""Load and validate the YAML attack corpus."""

from pathlib import Path
from typing import Literal

from redline.core import Attack, RedlineModel
from redline.core.yaml_files import CorpusError, read_yaml_dir

__all__ = ["BUNDLED_CORPUS_DIR", "Corpus", "CorpusError", "load_corpus"]

# Corpus shipped inside the package. A plain path works because the package is installed
# as real files (not zipped); the YAML files are checked for in the built wheel in chunk 5.
BUNDLED_CORPUS_DIR = Path(__file__).parent / "corpus"


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


def load_corpus(path: Path | None = None) -> Corpus:
    """Load every `*.yaml` file in `path` (default: the bundled corpus).

    Raises CorpusError for a missing directory, bad YAML, a wrong schema_version, unknown or
    missing fields, duplicate attack ids, or an empty corpus.
    """
    attacks, content_hash = read_yaml_dir(
        path if path is not None else BUNDLED_CORPUS_DIR,
        _CorpusFile,
        lambda file: file.attacks,
        "attack",
    )
    return Corpus(attacks=attacks, content_hash=content_hash)
