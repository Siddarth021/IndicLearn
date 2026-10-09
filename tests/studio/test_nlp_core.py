from indiclearn.studio.lang import detect_language, split_sentences, tokenize_lower
from indiclearn.studio.phonetic import phonetic_similarity


def test_detects_four_languages():
    assert detect_language("What is the Nyquist rate and why does it matter?").code == "en"
    assert detect_language("నైక్విస్ట్ రేటు అంటే ఏమిటి మరియు ఎందుకు ముఖ్యం").code == "te"
    assert detect_language("अलियासिंग क्या है और यह क्यों होता है").code == "hi"


def test_telugu_words_are_not_split_on_combining_marks():
    toks = tokenize_lower("సిగ్నల్ అంటే సమయంతో")
    assert "సిగ్నల్" in toks and "అంటే" in toks


def test_sentence_split_handles_danda():
    assert len(split_sentences("यह पहला वाक्य है। यह दूसरा वाक्य है।")) == 2


def test_cross_script_phonetic_match():
    assert phonetic_similarity("నైక్విస్ట్", "Nyquist") > 0.6
    assert phonetic_similarity("Convolution", "Aliasing") < 0.7
