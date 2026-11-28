"""Translated variants, made from translations saved ahead of time.

Translating once and saving the text means every run sends exactly the same words, costs no
extra model calls, and the translations can be read and checked by a person.
"""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, ValidationError, model_validator

from redline.attacks.mutators import build_variant
from redline.attacks.protect import MARKER, protected_pattern
from redline.core import Attack, RedlineModel
from redline.core.yaml_files import CorpusError


class Translation(RedlineModel):
    """One attack's prompt in another language."""

    # The id of the attack that was translated.
    id: str = Field(min_length=1)
    # The English it was translated from, to detect that the attack changed since.
    source: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    # The translation turned back into English, so a person can judge its quality.
    back_translation: str = ""


class TranslationFile(RedlineModel):
    """One YAML file: all translations into one language."""

    schema_version: Literal[1]
    # Short language code, e.g. "zu" for Zulu. Becomes part of each variant's id.
    language: str = Field(pattern=r"^[a-z]{2,3}$")
    # Who or what translated it, e.g. a model name. Quote it next to any result.
    translated_by: str = Field(min_length=1)
    translations: tuple[Translation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _one_translation_per_attack(self) -> Self:
        ids = [entry.id for entry in self.translations]
        if len(set(ids)) != len(ids):
            raise ValueError("an attack is translated more than once")
        return self


def load_translation(path: Path) -> TranslationFile:
    """Read one translation file. Raises CorpusError naming the file."""
    try:
        return TranslationFile.model_validate(yaml.safe_load(path.read_bytes()))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise CorpusError(f"{path.name}: {exc}") from exc


class TranslationMutator:
    """Ask for the same thing in another language.

    Only for direct attacks that have a saved translation.
    """

    def __init__(self, file: TranslationFile) -> None:
        self._entries = {entry.id: entry for entry in file.translations}
        self._name = f"translate-{file.language}"

    @property
    def name(self) -> str:
        return self._name

    def applies_to(self, attack: Attack) -> bool:
        return not attack.documents and attack.id in self._entries

    def mutate(self, attack: Attack) -> Attack:
        if not self.applies_to(attack):
            raise ValueError(f"no {self.name} translation applies to {attack.id!r}")
        entry = self._entries[attack.id]
        if entry.source != attack.prompt:
            raise ValueError(
                f"{self.name}: the translation of {attack.id!r} is out of date "
                "(the attack's prompt changed). Translate it again."
            )
        found = (match.group() for match in protected_pattern(attack).finditer(attack.prompt))
        lost = [text for text in found if text not in entry.prompt]
        if lost:
            raise ValueError(f"{self.name}: the translation of {attack.id!r} lost {lost}")
        if MARKER.search(entry.prompt):
            raise ValueError(f"{self.name}: the translation of {attack.id!r} has a leftover marker")
        return build_variant(attack, self.name, lambda _: entry.prompt)
