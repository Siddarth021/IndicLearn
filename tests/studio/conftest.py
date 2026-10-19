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


@pytest.fixture()
def sample_md():
    return SAMPLES / "signals_and_systems_notes.md"


@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient

    from indiclearn.studio.app import create_app
    from indiclearn.studio.config import Settings

    s = Settings()
    s.data_dir = tmp_path
    s.max_upload_mb = 1
    return TestClient(create_app(s))


@pytest.fixture()
def auth(client):
    return {"Authorization": "Bearer " + client.post("/api/auth/guest", json={}).json()["token"]}


@pytest.fixture(autouse=True)
def _hermetic_models(monkeypatch, tmp_path_factory):
    """Tests must not depend on what happens to be installed: no local models, no Ollama, no keys."""
    empty = tmp_path_factory.mktemp("no_models")
    monkeypatch.setenv("INDICLEARN_MODELS_DIR", str(empty))
    monkeypatch.setenv("INDICLEARN_EMBEDDING_BACKEND", "none")
    monkeypatch.setenv("INDICLEARN_TRANSLATOR_BACKEND", "none")
    monkeypatch.setenv("INDICLEARN_OLLAMA", "off")
    for k in ("INDICLEARN_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "INDICLEARN_SARVAM_API_KEY", "SARVAM_API_KEY",
              "INDICLEARN_OPENAI_BASE_URL", "INDICLEARN_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from indiclearn.studio import neural

    neural.reset_cache()
    yield
    neural.reset_cache()
