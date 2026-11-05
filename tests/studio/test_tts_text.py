"""Offline voices: text must be cleaned for the English voice, and a Telugu reply must be readable by ONE voice."""
from indiclearn.studio import tts_text as T


def test_english_acronyms_numbers_symbols_are_speakable():
    out = T.english_for_tts("The CPU runs 3 processes; x >= 10 [p. 4]")
    assert "c p u" in out and "three" in out and "greater than or equal to ten" in out and "[" not in out and "4" not in out


def test_english_respells_hard_words_and_io():
    out = T.english_for_tts("Use a mutex for I/O")
    assert "mew tex" in out and "i o" in out


def test_native_script_has_no_latin_letters_left():
    for lang in ("te", "hi"):
        out = T.to_native_script("Process అంటే deadlock లో CPU scheduling, FIFO and 10 frames", lang)
        assert not any("a" <= c.lower() <= "z" for c in out), out


def test_native_script_uses_lecturer_words():
    assert "ప్రాసెస్" in T.to_native_script("process", "te") and "డెడ్‌లాక్" in T.to_native_script("deadlock", "te")


def test_formula_text_is_spoken_not_spelled_as_latex():
    out = T.english_for_tts(r"false when $p \vee q$ is not true")
    assert "\\" not in out and "$" not in out
