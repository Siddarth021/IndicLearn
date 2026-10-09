"""Language and script detection submodule (Phase 4)."""

from indiclearn.nlp.language_detection.base import LanguageDetector
from indiclearn.nlp.language_detection.detector import HeuristicLanguageDetector

__all__ = ["HeuristicLanguageDetector", "LanguageDetector"]
