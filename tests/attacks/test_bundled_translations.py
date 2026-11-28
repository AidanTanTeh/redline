import re
from pathlib import Path

import redline.attacks
from redline.attacks import TranslationMutator, load_corpus, load_translation, make_variants

TRANSLATIONS = Path(redline.attacks.__file__).parent / "translations"


def test_the_chinese_file_applies_cleanly_to_the_current_corpus() -> None:
    # Fails if an attack's prompt is edited without translating it again.
    file = load_translation(TRANSLATIONS / "zh.yaml")

    variants = make_variants(load_corpus().attacks, [TranslationMutator(file)])

    assert len(variants) == len(file.translations) == 16
    assert all(re.search(r"[一-鿿]", v.prompt) for v in variants)
