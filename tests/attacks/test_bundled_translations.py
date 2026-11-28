import re

from redline.attacks import (
    BUNDLED_TRANSLATIONS_DIR,
    TranslationMutator,
    bundled_mutators,
    load_corpus,
    load_translation,
    make_variants,
)


def test_the_chinese_file_applies_cleanly_to_the_current_corpus() -> None:
    # Fails if an attack's prompt is edited without translating it again.
    file = load_translation(BUNDLED_TRANSLATIONS_DIR / "zh.yaml")

    variants = make_variants(load_corpus().attacks, [TranslationMutator(file)])

    assert len(variants) == len(file.translations) == 16
    assert all(re.search(r"[一-鿿]", v.prompt) for v in variants)


def test_every_shipped_mutator_is_listed_once() -> None:
    names = [mutator.name for mutator in bundled_mutators()]

    assert names == ["base64", "leetspeak", "homoglyph", "payload-split", "translate-zh"]
