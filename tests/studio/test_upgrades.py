"""Tests for the October 2026 upgrades: reply language, MCQ validation, streaming, quiz bank, multi-model Gemini."""

from __future__ import annotations

import random

from indiclearn.studio import brain as B
from indiclearn.studio import quizbank
from indiclearn.studio.db import Database
from indiclearn.studio.tutor import _asked_lang, _reply_lang, _understand_locally


def test_reply_language_follows_the_question_not_the_history():
    assert _reply_lang({"lang": "te"}, "summarise this paper", "en", True) == "en"  # English after a Telugu chat
    assert _reply_lang({"lang": "te", "romanized": True}, "stability enti", "en", True) == "te"  # Romanized -> Telugu script
    assert _reply_lang({}, "ఈ పేపర్ లో CPC ఎందుకు వాడారు?", "en", True) == "te"
    assert _reply_lang({"lang": "en"}, "summarise this paper in Telugu", "en", True) == "te"  # explicit request wins
    assert _reply_lang({"lang": "en"}, "anything", "hi", False) == "hi"  # language forced in the UI


def test_explicit_language_requests():
    assert _asked_lang("telugu lo cheppu") == "te"
    assert _asked_lang("hindi mein batao") == "hi"
    assert _asked_lang("explain in english please") == "en"
    assert _asked_lang("what is telugu") is None


def test_simple_questions_skip_the_extra_llm_call():
    assert _understand_locally("What is knowledge distillation?", [], "en")["intent"] == "explain"
    assert _understand_locally("quiz me on convolution please", [], "en")["intent"] == "quiz"
    assert _understand_locally("convolution ante enti?", [], "en") is None  # Romanized: needs the LLM
    assert _understand_locally("explain it again", [{"role": "user", "content": "x"}], "en") is None  # follow-up


def test_mcq_placeholders_and_wrong_keys_are_rejected():
    assert B.clean_mcq({"q": "Which is right here?", "options": ["A", "B", "C", "D"], "answer": 0}) is None
    assert B.clean_mcq({"q": "Which is right here?", "options": ["x1", "x2", "x3", "x4"], "answer": 7}) is None
    assert B.clean_mcq({"q": "Which is right here?", "options": ["same", "same", "x3", "x4"], "answer": 0}) is None
    q = B.clean_mcq({"q": "Which is right here?", "options": ["A) low pass", "B) high pass", "C) band pass", "D) notch"], "answer": "B"})
    assert q["options"][1] == "high pass" and q["answer"] == 1
    assert B.clean_mcq({"q": "Which is right here?", "options": ["low", "high", "band", "notch"], "answer": "band"})["answer"] == 2
    assert B.empty_if_placeholder("empty string") == ""


def test_partial_answer_reads_an_unfinished_json_stream():
    assert B.partial_answer('{"answer": "Hello **wo') == "Hello **wo"
    assert B.partial_answer('{"answer": "a\\nb \\u0c15 ", "x') == "a\nb క "
    assert B.partial_answer('{"reason') == ""


def test_gemini_free_tier_rotates_models(monkeypatch):
    monkeypatch.setenv("INDICLEARN_GEMINI_API_KEY", "k")
    monkeypatch.setenv("INDICLEARN_BRAIN", "gemini")
    monkeypatch.setenv("INDICLEARN_GEMINI_MODEL", "m-a, m-b")
    assert [p.model for p in B.Brain.from_env().providers] == ["m-a", "m-b"]


class _FakeBrain:
    available = True

    def first_is_local(self):
        return False

    def quiz(self, concepts, lang, per):
        qs = [{"concept_id": c["id"], "q": f"Why does {c['label']} matter?", "options": ["right", "w1", "w2", "w3"], "answer": 0,
               "why": "because", "page": 1} for c in concepts]
        return qs, "fake"

    def verify_mcqs(self, items):
        return [0 if "keep" in it["q"] else 2 for it in items]  # only the "keep" concept's key is confirmed


def test_quiz_bank_keeps_only_confirmed_answer_keys(tmp_path):
    db = Database(tmp_path / "t.sqlite3")
    db.run("INSERT INTO users(id,username) VALUES ('u','u')")
    db.run("INSERT INTO documents(id,user_id,title,status) VALUES ('d','u','t','ready')")
    nodes = [{"id": "c1", "type": "concept", "label": "keep me", "definition": "x" * 50, "parent": "root"},
             {"id": "c2", "type": "concept", "label": "drop me", "definition": "y" * 50, "parent": "root"}]
    assert quizbank.generate(db, _FakeBrain(), "d", nodes, None, "en") == 1
    qs = quizbank.pick(db, _FakeBrain(), "u", "d", nodes, None, None, 5, "en", {}, random.Random(1))
    assert len(qs) == 1 and qs[0]["node_id"] == "c1"
    assert qs[0]["options"][qs[0]["answer"]] == "right"  # options are shuffled, the key follows them


def test_quiz_explanations_never_point_at_shuffled_option_numbers():
    opts = ["sill", "range", "nugget", "lag"]
    assert B.fix_option_refs("Option 0 refers to the sill, and Option 1 to the range.", opts) == "“sill” refers to the sill, and “range” to the range."
    assert B.fix_option_refs("Option B is the range.", opts) == "“range” is the range."


def test_exam_questions_get_an_exam_answer_not_a_quiz():
    from indiclearn.studio.i18n import detect_intent
    assert detect_intent("what will come in exam from this unit?") == ["exam"]
    assert detect_intent("ee unit lo important topics enti? exam ki ela prepare avvali") == ["exam"]
    assert "quiz" in detect_intent("test me before my exam")
    assert _understand_locally("what will come in exam from this unit?", [], "en")["intent"] == "exam"


def test_voice_keyterms_are_cleaned_for_sarvam(monkeypatch):
    from indiclearn.studio import speech
    sent = []
    monkeypatch.setattr(speech, "available", lambda kind: True)
    monkeypatch.setattr(speech, "_stt_call", lambda fields, *a: sent.append(dict(fields)) or {"text": "ok", "lang": "en"})
    speech.stt(b"x", "a.webm", "audio/webm", ["Encoding, Decoding", "Kriging", "Kriging", "ab"])
    assert sent[0]["keyterms"] == '["Encoding Decoding", "Kriging"]'
