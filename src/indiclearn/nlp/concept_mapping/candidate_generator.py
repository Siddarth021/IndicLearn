"""Candidate generation and scoring across exact, lexical, terminology, and semantic signals (Phase 4)."""

from __future__ import annotations

import re
import unicodedata

from indiclearn.domain.concepts import Concept
from indiclearn.embeddings.base import EmbeddingProvider
from indiclearn.nlp.models import ConceptCandidate, TechnicalTerm
from indiclearn.nlp.normalization.normalizer import UnifiedQueryNormalizer


class CandidateGenerator:
    """Generates and scores concept candidates from diverse linguistic and semantic signals."""

    def __init__(
        self,
        normalizer: UnifiedQueryNormalizer | None = None,
    ) -> None:
        self.normalizer = normalizer or UnifiedQueryNormalizer()

    def generate_exact_candidates(
        self,
        query: str,
        concepts: list[Concept],
    ) -> list[ConceptCandidate]:
        """Candidate generation via exact canonical name or exact alias match."""
        clean_q = unicodedata.normalize("NFC", query).strip().lower()
        # Remove trailing question mark or period
        clean_q = re.sub(r"[?!.,]+$", "", clean_q).strip()

        candidates: list[ConceptCandidate] = []
        for cpt in concepts:
            cname = cpt.canonical_name.strip().lower()
            if clean_q == cname:
                candidates.append(
                    ConceptCandidate(
                        concept_id=cpt.concept_id,
                        canonical_name=cpt.canonical_name,
                        score=1.0,
                        match_method="exact_canonical",
                        evidence_spans=[cpt.canonical_name],
                    )
                )
                continue

            for alias in cpt.aliases:
                if clean_q == alias.strip().lower():
                    candidates.append(
                        ConceptCandidate(
                            concept_id=cpt.concept_id,
                            canonical_name=cpt.canonical_name,
                            score=0.95,
                            match_method="exact_alias",
                            evidence_spans=[alias],
                        )
                    )
                    break
        return candidates

    def generate_terminology_candidates(
        self,
        terms: list[TechnicalTerm],
        concepts: list[Concept],
    ) -> list[ConceptCandidate]:
        """Candidate generation grounded in detected academic technical terms."""
        candidates: list[ConceptCandidate] = []
        concepts_by_id = {c.concept_id: c for c in concepts}
        concepts_by_name = {c.canonical_name.lower(): c for c in concepts}

        seen_ids: set[str] = set()
        for term in terms:
            cpt: Concept | None = None
            if term.concept_id and term.concept_id in concepts_by_id:
                cpt = concepts_by_id[term.concept_id]
            elif term.canonical_name.lower() in concepts_by_name:
                cpt = concepts_by_name[term.canonical_name.lower()]

            if cpt and cpt.concept_id not in seen_ids:
                seen_ids.add(cpt.concept_id)
                score = 0.95 if term.term_type == "canonical" else (0.90 if term.term_type == "acronym" else 0.85)
                candidates.append(
                    ConceptCandidate(
                        concept_id=cpt.concept_id,
                        canonical_name=cpt.canonical_name,
                        score=score,
                        match_method=f"terminology_{term.term_type}",
                        evidence_spans=[term.matched_text],
                    )
                )
        return candidates

    def generate_lexical_candidates(
        self,
        query: str,
        concepts: list[Concept],
    ) -> list[ConceptCandidate]:
        """Candidate generation via token overlap and normalized substring matching."""
        norm_res = self.normalizer.normalize(query)
        norm_q = norm_res.canonical_normalized_text.lower()
        q_tokens = set(re.findall(r"[\w\-]+", norm_q, flags=re.UNICODE))

        candidates: list[ConceptCandidate] = []
        for cpt in concepts:
            cname = cpt.canonical_name.lower()
            c_tokens = set(re.findall(r"[\w\-]+", cname, flags=re.UNICODE))

            # 1. Check substring inclusion
            if cname in norm_q or norm_q in cname:
                score = 0.80 if len(cname) > 3 else 0.60
                candidates.append(
                    ConceptCandidate(
                        concept_id=cpt.concept_id,
                        canonical_name=cpt.canonical_name,
                        score=score,
                        match_method="normalized_substring",
                        evidence_spans=[cpt.canonical_name],
                    )
                )
                continue

            # 2. Check alias substring inclusion (including Telugu script aliases)
            alias_matched = False
            for alias in cpt.aliases:
                low_a = alias.lower().strip()
                if low_a and (low_a in norm_q or low_a in query.lower()):
                    candidates.append(
                        ConceptCandidate(
                            concept_id=cpt.concept_id,
                            canonical_name=cpt.canonical_name,
                            score=0.80,
                            match_method="normalized_alias_substring",
                            evidence_spans=[alias],
                        )
                    )
                    alias_matched = True
                    break
            if alias_matched:
                continue

            # 3. Jaccard / token set overlap
            if q_tokens and c_tokens:
                intersection = q_tokens & c_tokens
                # Filter out pure stopwords from overlap
                meaningful_overlap = {t for t in intersection if t not in {"in", "of", "the", "a", "an", "is", "ante", "enti"}}
                if meaningful_overlap:
                    jaccard = len(meaningful_overlap) / len(c_tokens)
                    if jaccard >= 0.5:
                        candidates.append(
                            ConceptCandidate(
                                concept_id=cpt.concept_id,
                                canonical_name=cpt.canonical_name,
                                score=round(0.50 + jaccard * 0.35, 4),
                                match_method="token_overlap",
                                evidence_spans=list(meaningful_overlap),
                            )
                        )
        return candidates

    def generate_embedding_candidates(
        self,
        query: str,
        concepts: list[Concept],
        embedding_provider: EmbeddingProvider,
        concept_embeddings_cache: dict[str, list[float]] | None = None,
    ) -> list[ConceptCandidate]:
        """Candidate ranking via dense multilingual semantic embeddings."""
        if not concepts:
            return []

        # 1. Embed query
        query_vec = embedding_provider.embed_query(query).vector

        # 2. Embed or retrieve concept representations
        candidates: list[ConceptCandidate] = []
        for cpt in concepts:
            cid = cpt.concept_id
            if concept_embeddings_cache is not None and cid in concept_embeddings_cache:
                cpt_vec = concept_embeddings_cache[cid]
            else:
                # Build rich concept description combining name, aliases, description
                text_repr = f"{cpt.canonical_name}. "
                if cpt.aliases:
                    text_repr += f"Also known as: {', '.join(cpt.aliases)}. "
                if cpt.description:
                    text_repr += cpt.description
                cpt_vec = embedding_provider.embed_documents([text_repr])[0].vector
                if concept_embeddings_cache is not None:
                    concept_embeddings_cache[cid] = cpt_vec

            # Cosine similarity for unit vectors: dot product
            sim = sum(q * c for q, c in zip(query_vec, cpt_vec, strict=False))
            # Clip to [0.0, 1.0]
            sim_score = max(0.0, min(1.0, (sim + 1.0) / 2.0 if sim < 0 else sim))

            candidates.append(
                ConceptCandidate(
                    concept_id=cpt.concept_id,
                    canonical_name=cpt.canonical_name,
                    score=round(sim_score, 4),
                    match_method="semantic_embedding",
                    evidence_spans=[],
                    metadata={"raw_cosine": round(sim, 4)},
                )
            )

        # Sort descending by similarity
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates
