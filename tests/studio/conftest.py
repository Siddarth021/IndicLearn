from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"


@pytest.fixture(scope="session")
def en_graph():
    from indiclearn.studio.kgraph import build_knowledge_graph
    from indiclearn.studio.parse import parse_file

    return build_knowledge_graph(parse_file(SAMPLES / "signals_and_systems_notes.md", "n.md"))


@pytest.fixture(scope="session")
def en_index(en_graph):
    from indiclearn.studio.retrieval import DocIndex

    chunks = [{"id": c.id, "section_id": c.section_id, "page": c.page, "text": c.text} for c in en_graph.chunks]
    return DocIndex(chunks, en_graph.nodes, en_graph.language)
