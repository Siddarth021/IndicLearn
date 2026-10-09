from indiclearn.studio import mastery, quiz
from indiclearn.studio.tutor import answer


def labels(g):
    return {n["label"].lower() for n in g.nodes}


def test_default_graph_has_hierarchy_definitions_and_flow(en_graph):
    ls = labels(en_graph)
    assert {"convolution", "aliasing", "nyquist rate", "z-transform"} <= ls
    kinds = {n["type"] for n in en_graph.nodes}
    assert {"root", "chapter"} <= kinds
    assert any(e["type"] == "prereq" for e in en_graph.edges)
    assert any(e["type"] == "contains" for e in en_graph.edges)
    alias = next(n for n in en_graph.nodes if n["label"] == "Aliasing")
    assert "aliasing" in alias["definition"].lower()
    ids = {n["id"] for n in en_graph.nodes}
    assert all(e["source"] in ids and e["target"] in ids for e in en_graph.edges)


def test_graph_is_tree_without_cycles(en_graph):
    par = {n["id"]: n["parent"] for n in en_graph.nodes}
    for nid in par:
        seen, cur = set(), nid
        while cur:
            assert cur not in seen
            seen.add(cur)
            cur = par.get(cur)


def test_cross_lingual_linking(en_index):
    for q, want in [("What is convolution?", "convolution"), ("అలియాసింగ్ అంటే ఏమిటి", "aliasing"), ("नाइक्विस्ट रेट क्या है", "nyquist rate")]:
        links = en_index.link_concepts(q)
        assert links and links[0]["node"]["label"].lower() == want, (q, links[:2])


def test_tutor_grounded_answer_cites_pages(en_graph, en_index):
    r = answer("What is aliasing?", "en", en_index, en_graph.nodes, en_graph.edges, en_graph.title, "en")
    assert r.grounded and r.citations and r.evidence and r.evidence[0][1] == "asked"
    assert "aliasing" in r.text.lower()


def test_tutor_refuses_to_guess_and_creates_doubt(en_graph, en_index):
    r = answer("who won the cricket world cup", "en", en_index, en_graph.nodes, en_graph.edges, en_graph.title, "en")
    assert not r.grounded and r.doubt and r.doubt["covered"] is False and not r.evidence


def test_tutor_replies_in_ui_language_and_detects_intents(en_graph, en_index):
    r = answer("అలియాసింగ్ అంటే ఏమిటి?", "te", en_index, en_graph.nodes, en_graph.edges, en_graph.title, "en")
    assert r.lang == "te" and "నోట్స్" in r.text
    assert answer("quiz me on sampling", "en", en_index, en_graph.nodes, en_graph.edges, "t", "en").action == "quiz"
    assert answer("मुझे क्विज़ दो", "hi", en_index, en_graph.nodes, en_graph.edges, "t", "en").action == "quiz"
    assert answer("what should I study next", "en", en_index, en_graph.nodes, en_graph.edges, "t", "en").action == "recommend"


def test_quiz_questions_are_well_formed(en_graph):
    qs = quiz.generate(en_graph.nodes, None, 8, "en", seed=1)
    assert len(qs) >= 5
    for q in qs:
        assert 0 <= q["answer"] < len(q["options"])
        assert len(set(q["options"])) == len(q["options"])
        if q["qtype"] == "cloze":
            assert "_____" in q["stem"]
        assert "answer" not in quiz.public(q)


def test_mastery_math_and_recommendations(tmp_path, en_graph):
    from indiclearn.studio.db import Database

    db = Database(tmp_path / "t.sqlite3")
    nid = next(n["id"] for n in en_graph.nodes if n["label"] == "Aliasing")
    mastery.record_evidence(db, "u", "d", nid, "asked")
    st = mastery.node_states(db, "u", "d")[nid]
    assert st["status"] == "doubt"  # asking alone is weak evidence
    mastery.record_evidence(db, "u", "d", nid, "quiz_correct")
    assert mastery.node_states(db, "u", "d")[nid]["status"] == "learning"  # one right answer is not mastery
    for _ in range(3):
        mastery.record_evidence(db, "u", "d", nid, "quiz_correct")
    assert mastery.node_states(db, "u", "d")[nid]["status"] == "mastered"
    mastery.record_evidence(db, "u", "d", nid, "quiz_incorrect")
    mastery.record_evidence(db, "u", "d", nid, "quiz_incorrect")
    assert mastery.node_states(db, "u", "d")[nid]["score"] < 0.8
    ov = mastery.personal_overlay(en_graph.nodes, en_graph.edges, mastery.node_states(db, "u", "d"), [])
    assert any(n["mastery"]["aggregate"] for n in ov["nodes"] if n["type"] == "chapter")
    recs = mastery.recommend(en_graph.nodes, en_graph.edges, mastery.node_states(db, "u", "d"), 5)
    assert recs and all("label" in r for r in recs)


def test_short_offtopic_question_is_not_glued_to_the_focused_topic(en_graph, en_index):
    focus = next(n["id"] for n in en_graph.nodes if n["label"] == "Aliasing")
    r = answer("who invented pizza", "en", en_index, en_graph.nodes, en_graph.edges, "t", "en", focus_node=focus)
    assert not r.grounded and r.doubt
    r = answer("give an example", "en", en_index, en_graph.nodes, en_graph.edges, "t", "en", focus_node=focus)
    assert r.grounded and r.focus_node == focus
