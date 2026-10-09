from indiclearn.config.settings import load_settings


def test_defaults_when_file_missing(tmp_path):
    s = load_settings(tmp_path / "nope.yaml", environ={})
    assert s.chunking.chunk_size == 800


def test_yaml_and_env_override(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("chunking:\n  chunk_size: 500\n  chunk_overlap: 50\n", encoding="utf-8")
    assert load_settings(f, environ={}).chunking.chunk_size == 500
    s = load_settings(f, environ={"INDICLEARN_CHUNKING__CHUNK_SIZE": "300"})
    assert s.chunking.chunk_size == 300 and s.chunking.chunk_overlap == 50


def test_repo_config_loads():
    s = load_settings("configs/config.yaml", environ={})
    assert s.ingestion.supported_extensions and s.student_model.evidence_weights
