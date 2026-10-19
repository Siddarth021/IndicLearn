"""Phase-2 safety net: the parser fixes, and the trained models staying optional."""

from __future__ import annotations

import pytest

from indiclearn.studio import neural
from indiclearn.studio.lang import normalize
from indiclearn.studio.parse import _is_tabular, _repair_ligature_gaps, parse_file
from indiclearn.studio.quiz import _parse_generated


# --------------------------------------------------------------------------- parser fixes
def test_ligature_characters_are_expanded():
    assert normalize("Artiﬁcial Classiﬁcation eﬃcient") == "Artificial Classification efficient"


def test_ligature_gap_repaired_only_into_real_words():
    from indiclearn.studio.parse import Block

    blocks = [Block("para", "Arti cial neural networks and classi cation of the si gnal", 1),
              Block("para", "the cat sat on the mat", 1)]
    _repair_ligature_gaps(blocks)
    assert "Artificial" in blocks[0].text
    assert "classification" in blocks[0].text
    # "si gnal" has no fi/fl/ff completion, so it is left alone rather than mangled
    assert "si gnal" in blocks[0].text
    assert blocks[1].text == "the cat sat on the mat"  # ordinary prose untouched


def test_result_table_rows_are_detected():
    assert _is_tabular("1-CNN 58.66 56.92 55.87 69.5 71.2")
    assert not _is_tabular("The Nyquist rate is twice the highest frequency in the signal")


def test_uuid_filename_never_becomes_the_title(tmp_path):
    stored = tmp_path / "faee623705b84365.md"
    stored.write_text("Sampling converts a continuous signal into a sequence of numbers. " * 4,
                      encoding="utf-8")
    doc = parse_file(stored, "Signals and Systems notes.md")
    assert doc.title == "Signals and Systems notes"


def test_reference_entries_do_not_become_topics(tmp_path):
    md = tmp_path / "paper.md"
    md.write_text(
        "# A Study of Sampling\n\n"
        "## Introduction\n\nSampling is the process of measuring a signal at regular intervals. "
        "The Nyquist rate is twice the highest frequency present in the signal.\n\n"
        "## References\n\n"
        "[1] Smith, G.H., Van den Heever, D., International Conference on Learning, 2023.\n\n"
        "[2] Kumar, A., Rao, B., IEEE Transactions on Signal Processing, 2021.\n",
        encoding="utf-8")
    doc = parse_file(md, "paper.md")
    headings = " ".join(b.text for b in doc.blocks if b.kind == "heading").lower()
    assert "introduction" in headings
    assert "smith" not in headings and "ieee transactions" not in headings


# --------------------------------------------------------------------------- optional models
def test_every_loader_returns_none_when_nothing_is_installed(monkeypatch, tmp_path):
    monkeypatch.setenv("INDICLEARN_MODELS_DIR", str(tmp_path))
    neural.reset_cache()
    try:
        assert neural.load_concept_filter() is None
        assert neural.load_describer() is None
        assert neural.load_quizzer() is None
        assert neural.load_relations() is None
        assert neural.load_recogniser() is None
        assert neural.describe("Nyquist rate", "te") == ""
    finally:
        neural.reset_cache()


def test_switches_turn_a_model_off(monkeypatch, tmp_path):
    monkeypatch.setenv("INDICLEARN_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("INDICLEARN_USE_M1", "0")
    neural.reset_cache()
    try:
        assert neural.load_concept_filter() is None
        assert neural.status()["m1_concept_filter"] is False
    finally:
        neural.reset_cache()


def test_status_is_reported_without_torch():
    st = neural.status()
    assert set(st) >= {"models_dir", "m1_concept_filter", "m4_quiz", "asr", "device"}


def test_graph_is_unchanged_when_no_model_is_installed(sample_md):
    """The default pipeline must behave exactly as before phase 2."""
    from indiclearn.studio.kgraph import build_knowledge_graph

    kg = build_knowledge_graph(parse_file(sample_md, sample_md.name))
    assert kg.stats["concepts"] > 0
    assert all(n.get("concept_score") is None for n in kg.nodes)


# --------------------------------------------------------------------------- M4 output parsing
GOOD = ("Q: What does the Nyquist rate describe?\n"
        "A) Twice the highest frequency\nB) Half the bandwidth\n"
        "C) The noise floor\nD) The sampling jitter\nANSWER: A")


def test_generated_question_is_parsed():
    q, opts, ans = _parse_generated(GOOD)
    assert q.startswith("What does")
    assert len(opts) == 4 and opts[ans] == "Twice the highest frequency"


@pytest.mark.parametrize("bad", [
    "nonsense with no question at all",
    "Q: Too few options?\nA) one\nB) two\nANSWER: A",
    "Q: Repeated options?\nA) one\nB) one\nC) two\nD) three\nANSWER: A",   # duplicates give it away
    "Q: No answer line?\nA) one\nB) two\nC) three\nD) four",
])
def test_malformed_generation_is_rejected_so_rules_take_over(bad):
    assert _parse_generated(bad) is None


# --------------------------------------------------------------------------- API
def test_voice_endpoint_explains_itself_when_not_installed(client, auth):
    r = client.post("/api/asr", headers=auth, files={"file": ("q.webm", b"x" * 64, "audio/webm")})
    assert r.status_code == 503
    assert "download" in r.json()["detail"].lower()


def test_info_reports_which_trained_models_are_live(client, auth):
    body = client.get("/api/info", headers=auth).json()
    assert "trained" in body and "models_dir" in body["trained"]
