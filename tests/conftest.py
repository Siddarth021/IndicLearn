import pytest

from indiclearn.chunking import RecursiveCharChunker
from indiclearn.config.settings import Settings
from indiclearn.ingestion.service import IngestionService


@pytest.fixture()
def settings() -> Settings:
    return Settings()


@pytest.fixture()
def service(settings) -> IngestionService:
    c = settings.chunking
    return IngestionService(settings, RecursiveCharChunker(c.chunk_size, c.chunk_overlap))
