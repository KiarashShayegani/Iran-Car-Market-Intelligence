"""Tests for Persian normalization (raw text always preserved by callers)."""

from icmi.retrieval.normalization import (
    normalize_for_key,
    normalize_for_matching,
    persian_to_western_digits,
)


def test_arabic_variants_map_to_persian():
    assert normalize_for_matching("كپشني يو)") == normalize_for_matching("کپشنی یو)")
    assert normalize_for_matching("كي") == normalize_for_matching("کی")
    assert normalize_for_matching("متابعة") == normalize_for_matching("متابعه")


def test_zwnj_removed_for_matching():
    assert normalize_for_matching("می\u200cشود") == "میشود"
    assert normalize_for_matching("تندر\u200cگو") == normalize_for_matching("تندرگو")


def test_digits_and_english_case():
    assert normalize_for_matching("پژو ۲۰۶ مدل 1398 V8") == "پژو 206 مدل 1398 v8"
    assert persian_to_western_digits("۱۴۰۴/٠٩/٠١") == "1404/09/01"


def test_tatweel_and_whitespace():
    assert normalize_for_matching("پژوـــان") == "پژوان"
    assert normalize_for_matching("  متعدد   فاصله  ") == "متعدد فاصله"


def test_none_and_empty_safe():
    assert normalize_for_matching(None) == ""
    assert normalize_for_matching("") == ""


def test_key_normalization_collapses_spacing_and_punctuation():
    assert normalize_for_key("پژو ۲۰۶") == normalize_for_key("پژو206")
    assert normalize_for_key("پژو — 206 (تیپ 5)") == normalize_for_key("پژو206تیپ5")
    # sanity: distinct words stay distinct
    assert normalize_for_key("MOtor") != normalize_for_key("موتور")
