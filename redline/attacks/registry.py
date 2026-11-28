"""The one list of mutators that ship with Redline."""

from pathlib import Path

from redline.attacks.mutators import (
    Base64Mutator,
    HomoglyphMutator,
    LeetspeakMutator,
    Mutator,
    PayloadSplitMutator,
)
from redline.attacks.translation import TranslationMutator, load_translation

BUNDLED_TRANSLATIONS_DIR = Path(__file__).parent / "translations"


def bundled_mutators() -> tuple[Mutator, ...]:
    """The text disguises, then one translation mutator per bundled language file."""
    files = sorted(BUNDLED_TRANSLATIONS_DIR.glob("*.yaml"))
    return (
        Base64Mutator(),
        LeetspeakMutator(),
        HomoglyphMutator(),
        PayloadSplitMutator(),
        *(TranslationMutator(load_translation(path)) for path in files),
    )
