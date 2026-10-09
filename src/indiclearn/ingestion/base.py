"""Loader interface and shared errors."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol

from indiclearn.domain.documents import ExtractedPage


class ExtractionError(Exception):
    """Raised by a loader when a file cannot be read; recorded, never propagated to the user as a crash."""


class DocumentLoader(Protocol):
    extensions: ClassVar[tuple[str, ...]]

    def load(self, path: Path) -> list[ExtractedPage]: ...
