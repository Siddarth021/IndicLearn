"""Retrieval-Augmented Generation (RAG) and Grounded Generation package (Phase 5)."""

from indiclearn.rag.citation import CitationParser, CitationValidator
from indiclearn.rag.claims import (
    Claim,
    ClaimAuditResult,
    ClaimExtractor,
    ClaimLevelAuditor,
    EntailmentVerifier,
    HeuristicLexicalOverlapAssessor,
    SentenceClaimExtractor,
)
from indiclearn.rag.context_builder import ContextBuilder
from indiclearn.rag.evidence_selector import EvidenceSelector
from indiclearn.rag.grounding import GroundingVerifier
from indiclearn.rag.models import (
    Citation,
    CitationSpan,
    ContextDocument,
    EvidenceItem,
    EvidenceSet,
    GenerationMetadata,
    GroundingCheck,
    GroundingReport,
    RAGRequest,
    RAGResponse,
)
from indiclearn.rag.pipeline import RAGPipeline
from indiclearn.rag.policies import (
    AMBIGUOUS_SCOPE_MESSAGES,
    INSUFFICIENT_EVIDENCE_MESSAGES,
    OUT_OF_SCOPE_MESSAGES,
    RAG_SYSTEM_PROMPT,
    build_user_prompt,
)
from indiclearn.rag.scope import (
    HeuristicScopeAssessor,
    ScopeAssessor,
    ScopeDecision,
)

__all__ = [
    "AMBIGUOUS_SCOPE_MESSAGES",
    "INSUFFICIENT_EVIDENCE_MESSAGES",
    "OUT_OF_SCOPE_MESSAGES",
    "RAG_SYSTEM_PROMPT",
    "Citation",
    "CitationParser",
    "CitationSpan",
    "CitationValidator",
    "Claim",
    "ClaimAuditResult",
    "ClaimExtractor",
    "ClaimLevelAuditor",
    "ContextBuilder",
    "ContextDocument",
    "EntailmentVerifier",
    "EvidenceItem",
    "EvidenceSelector",
    "EvidenceSet",
    "GenerationMetadata",
    "GroundingCheck",
    "GroundingReport",
    "GroundingVerifier",
    "HeuristicLexicalOverlapAssessor",
    "HeuristicScopeAssessor",
    "RAGPipeline",
    "RAGRequest",
    "RAGResponse",
    "ScopeAssessor",
    "ScopeDecision",
    "SentenceClaimExtractor",
    "build_user_prompt",
]
