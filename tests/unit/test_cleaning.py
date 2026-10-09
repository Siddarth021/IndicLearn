import unicodedata

from indiclearn.preprocessing.cleaning import clean_text

TELUGU = "కన్వల్యూషన్ అంటే ఏమిటి"


def test_nfc_normalization():
    decomposed = unicodedata.normalize("NFD", "é")
    assert clean_text(decomposed) == "é"
    assert unicodedata.is_normalized("NFC", clean_text(TELUGU))


def test_telugu_preserved_exactly():
    assert clean_text(TELUGU) == unicodedata.normalize("NFC", TELUGU)


def test_zwj_zwnj_preserved():
    s = "క\u200d\u200cష"
    assert "\u200d" in clean_text(s) and "\u200c" in clean_text(s)


def test_strips_nul_bom_soft_hyphen():
    assert clean_text("a\x00b\ufeffc\u00add") == "abcd"


def test_dehyphenate_and_toggle():
    assert clean_text("convo-\nlution") == "convolution"
    assert clean_text("convo-\nlution", join_hyphenated_lines=False) == "convo-\nlution"
    assert clean_text("LTI-\nSystem") == "LTI-\nSystem"  # uppercase continuation not joined


def test_whitespace_collapse():
    assert clean_text("a   b\t\tc\n\n\n\n\nd  \ne") == "a b c\n\nd\ne"


def test_empty():
    assert clean_text("") == ""
