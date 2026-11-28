"""Redline attacks: the versioned YAML attack corpus, its loader, and prompt rendering."""

from redline.attacks.loader import BUNDLED_CORPUS_DIR, Corpus, CorpusError, load_corpus
from redline.attacks.mutators import (
    Base64Mutator,
    HomoglyphMutator,
    LeetspeakMutator,
    Mutator,
    PayloadSplitMutator,
    build_variant,
    make_variants,
)
from redline.attacks.protect import mask_protected, unmask_protected
from redline.attacks.registry import BUNDLED_TRANSLATIONS_DIR, bundled_mutators
from redline.attacks.render import render_documents, render_prompt
from redline.attacks.translation import (
    Translation,
    TranslationFile,
    TranslationMutator,
    load_translation,
)

__all__ = [
    "BUNDLED_CORPUS_DIR",
    "BUNDLED_TRANSLATIONS_DIR",
    "Base64Mutator",
    "Corpus",
    "CorpusError",
    "HomoglyphMutator",
    "LeetspeakMutator",
    "Mutator",
    "PayloadSplitMutator",
    "Translation",
    "TranslationFile",
    "TranslationMutator",
    "build_variant",
    "bundled_mutators",
    "load_corpus",
    "load_translation",
    "make_variants",
    "mask_protected",
    "render_documents",
    "render_prompt",
    "unmask_protected",
]
