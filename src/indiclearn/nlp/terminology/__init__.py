"""Technical terminology detection submodule (Phase 4)."""

from indiclearn.nlp.terminology.base import TechnicalTermDetector
from indiclearn.nlp.terminology.detector import GraphGroundedTermDetector

__all__ = ["GraphGroundedTermDetector", "TechnicalTermDetector"]
