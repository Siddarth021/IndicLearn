"""INDICLEARN_OFFLINE: one switch between "everything on this computer" and "online first, offline as backup"."""

from __future__ import annotations

import re

import pytest

from indiclearn.studio import brain as B


@pytest.fixture()
def keys(monkeypatch):
    monkeypatch.setenv("INDICLEARN_GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("INDICLEARN_SARVAM_API_KEY", "test-sarvam-key")
    monkeypatch.setenv("INDICLEARN_OLLAMA", "on")  # the shared fixtures switch it off; here it is the point
    monkeypatch.delenv("INDICLEARN_BRAIN", raising=False)


def _names(b) -> list[str]:
    return [p.name for p in b.providers]


def test_online_mode_tries_the_internet_first_then_falls_back(keys, monkeypatch):
    monkeypatch.setenv("INDICLEARN_OFFLINE", "false")
    names = _names(B.Brain.from_env(probe_ollama=False))
    assert names[0] == "gemini" and "sarvam" in names
    assert names[-1] == "ollama", "the offline engine must stay last, as the backup"


def test_offline_mode_keeps_every_request_on_this_computer(keys, monkeypatch):
    monkeypatch.setenv("INDICLEARN_OFFLINE", "true")
    names = _names(B.Brain.from_env(probe_ollama=False))
    assert names == ["ollama"], f"offline mode still reaches the internet: {names}"


def test_offline_mode_keeps_a_local_openai_server(keys, monkeypatch):
    """A model served on the student's own PC (LM Studio, llama.cpp) is local, so it is allowed."""
    monkeypatch.setenv("INDICLEARN_OFFLINE", "true")
    monkeypatch.setenv("INDICLEARN_OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("INDICLEARN_OPENAI_MODEL", "local-model")
    monkeypatch.setenv("INDICLEARN_OPENAI_LOCAL", "true")
    assert set(_names(B.Brain.from_env(probe_ollama=False))) == {"openai", "ollama"}


@pytest.mark.parametrize("value,expected", [("true", True), ("1", True), ("on", True), ("yes", True),
                                            ("false", False), ("0", False), ("", False), (None, False)])
def test_the_switch_reads_plain_words(monkeypatch, value, expected):
    monkeypatch.delenv("INDICLEARN_OFFLINE", raising=False)
    if value is not None:
        monkeypatch.setenv("INDICLEARN_OFFLINE", value)
    assert B.offline_mode() is expected


def test_switch_is_reported_to_the_app(client, auth, monkeypatch):
    monkeypatch.setenv("INDICLEARN_OFFLINE", "true")
    r = client.get("/api/ai/status", headers=auth).json()
    assert r["offline"] is True
    assert "te_style_translator" in r["models"], "the Telugu translator is listed among the local models"


# ---------------------------------------------------------------- offline Telugu, without the model installed
def test_lessons_stay_english_when_no_translator_is_installed(client, auth, monkeypatch):
    """No model folder and no internet: the student still gets the English lesson, never an error."""
    monkeypatch.setenv("INDICLEARN_OFFLINE", "true")
    from indiclearn.studio.courses import lessons

    real = lessons.lessons  # drop the Telugu that ships with the course, so nothing is cached
    monkeypatch.setattr(lessons, "lessons", lambda subject="os": {k: {**v, "te": None} for k, v in real(subject).items()})
    monkeypatch.setattr(lessons, "_local_te", lambda: None)
    r = client.get("/api/courses/os/lessons/paging?lang=te", headers=auth).json()
    assert r["lang"] == "en" and r["lesson"]["summary"]


def test_offline_telugu_uses_our_own_model(monkeypatch):
    """With the translator present, a lesson is rendered field by field on this computer."""
    from indiclearn.studio import neural
    from indiclearn.studio.courses import lessons

    monkeypatch.setattr(neural, "to_telugu", lambda text, tr=None: "ఇది ఒక తెలుగు వాక్యం, లోపల " + text[:12])
    lesson = {"title": "Paging", "summary": "A page is a fixed-size block of memory used by the operating system.",
              "sections": [{"heading": "Why paging", "body": "Paging removes external fragmentation completely."}],
              "key_terms": [{"term": "Frame", "meaning": "A fixed-size slot in physical memory."}],
              "example": {"title": "", "body": ""}, "exam_tips": [], "mistakes": [], "diagram": ""}
    out = lessons.translate_local(object(), lesson)
    assert out["summary"].startswith("ఇది ఒక తెలుగు")
    assert out["sections"][0]["body"].startswith("ఇది ఒక తెలుగు")
    assert out["key_terms"][0]["term"] == "Frame", "a key term keeps its English name"


def test_offline_telugu_gives_up_rather_than_showing_half_english(monkeypatch):
    from indiclearn.studio import neural
    from indiclearn.studio.courses import lessons

    monkeypatch.setattr(neural, "to_telugu", lambda text, tr=None: "")  # the model answers nothing
    lesson = {"title": "Paging", "summary": "A page is a fixed-size block of memory.", "sections": [],
              "key_terms": [], "example": {"title": "", "body": ""}, "exam_tips": [], "mistakes": [], "diagram": ""}
    assert lessons.translate_local(object(), lesson) is None


# ---------------------------------------------------------------- the translator's own text handling
@pytest.mark.parametrize("text", ["$a_{11}$", "`code`", "$\\sum_{i=1}^{n} i$"])
def test_formulas_and_code_are_never_rewritten(text, monkeypatch):
    from indiclearn.studio import neural

    class Keeps:
        def generate(self, prompt, max_new_tokens=None):
            marker = re.search(r"Z\d+Z", prompt)
            return f"Pivot అనేది matrix లోని {marker.group(0)} విలువ."

    class Drops:
        def generate(self, prompt, max_new_tokens=None):
            return "Pivot అనేది matrix లోని విలువ."  # the model threw the formula away

    english = f"The pivot is {text} in the matrix."
    assert text in neural.to_telugu(english, Keeps()), f"{text} was lost in translation"
    # rather than publish a sentence with the formula missing, the line stays in English
    assert neural.to_telugu(english, Drops()) == english


def test_a_repeated_or_invented_marker_is_refused():
    """Seq2seq models sometimes say a clause twice. Half a formula in a lesson is worse than English."""
    from indiclearn.studio import neural

    class Repeats:
        def generate(self, prompt, max_new_tokens=None):
            m = re.search(r"Z\d+Z", prompt).group(0)
            return f"{m} విలువ, అంటే {m.lower()} విలువ."  # the same marker twice, one in lower case

    class Invents:
        def generate(self, prompt, max_new_tokens=None):
            return "Z7Z విలువ."  # a marker that was never sent

    english = "The pivot is $a_{11}$ here."
    assert neural.to_telugu(english, Repeats()) == english
    assert neural.to_telugu(english, Invents()) == english


def test_bullets_and_blank_lines_survive(monkeypatch):
    from indiclearn.studio import neural

    class Fake:
        def generate(self, prompt, max_new_tokens=None):
            return "అనువాదం"

    out = neural.to_telugu("- first point\n\n- second point", Fake())
    assert out.split("\n")[0].startswith("- ") and out.count("- ") == 2 and "\n\n" in out


# ---------------------------------------------------------------- models saved by transformers 5 must load on 4.x
def test_tokenizer_config_saved_by_transformers_5_is_repaired(tmp_path):
    import json

    from indiclearn.studio import neural

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    cfg = {"eos_token": "</s>", "extra_special_tokens": ["<x1>", "<x2>"], "pad_token": "<pad>"}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    neural._fix_tokenizer_config(tmp_path)
    fixed = json.loads((tmp_path / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert "extra_special_tokens" not in fixed, "transformers 4.x cannot read a list here"
    assert fixed["additional_special_tokens"] == ["<x1>", "<x2>"] and fixed["eos_token"] == "</s>"


def test_empty_list_is_dropped_and_a_dict_is_left_alone(tmp_path):
    import json

    from indiclearn.studio import neural

    for value, expect_key in (([], False), ({"pad": "<pad>"}, True)):
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"extra_special_tokens": value}), encoding="utf-8")
        neural._fix_tokenizer_config(tmp_path)
        assert ("extra_special_tokens" in json.loads((tmp_path / "tokenizer_config.json").read_text(encoding="utf-8"))) is expect_key
    neural._fix_tokenizer_config(tmp_path / "missing")  # no file: no error
