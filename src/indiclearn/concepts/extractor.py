"""Baseline heuristic concept and relation extractor (Phase 3).

Extracts academic concepts and typed relationships from DocumentChunks using
heading hierarchy, definitional grammar patterns, and domain terminology cues.
Retains strict source provenance without requiring neural or LLM inference.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from indiclearn.concepts.normalizer import (
    ConceptNormalizer,
    generate_concept_id,
    generate_relation_id,
)
from indiclearn.domain.concepts import (
    Concept,
    ConceptRelation,
    ConceptType,
    PrerequisiteNature,
    RelationType,
)
from indiclearn.domain.documents import DocumentChunk


@runtime_checkable
class ConceptExtractor(Protocol):
    """Protocol for extracting concepts and relations from document chunks."""

    def extract(
        self, chunks: list[DocumentChunk]
    ) -> tuple[list[Concept], list[ConceptRelation]]:
        """Extract concepts and relations from a collection of DocumentChunks."""
        ...


# Known property indicators in engineering domains
PROPERTY_TERMS = {
    "linearity",
    "time invariance",
    "time-invariance",
    "causality",
    "stability",
    "bibo stability",
    "superposition",
    "homogeneity",
    "commutativity",
    "associativity",
    "distributivity",
    "duality",
}

# Known method/operation indicators
METHOD_TERMS = {
    "convolution",
    "continuous-time convolution",
    "discrete-time convolution",
    "graphical convolution",
    "sampling",
    "reconstruction",
    "filtering",
    "fourier analysis",
}


def classify_concept_type(canonical_name: str) -> ConceptType:
    """Classify concept into ConceptType based on domain ontology."""
    low = canonical_name.lower()
    if low in PROPERTY_TERMS or "property" in low or "stability" in low:
        return ConceptType.PROPERTY
    if low in METHOD_TERMS or "method" in low or "algorithm" in low:
        return ConceptType.METHOD
    if "theorem" in low or "formula" in low or "criterion" in low or "equation" in low:
        return ConceptType.FORMULA
    if "continuous-time" in low or "discrete-time" in low:
        return ConceptType.SUBCONCEPT
    return ConceptType.CONCEPT


class HeuristicConceptExtractor(ConceptExtractor):
    """Deterministic, rule-based extractor using headings and definitional grammar."""

    def __init__(self, subject_id: str = "signals_and_systems") -> None:
        self.subject_id = subject_id
        self.normalizer = ConceptNormalizer(subject_id=subject_id)

    def extract(
        self, chunks: list[DocumentChunk] | DocumentChunk
    ) -> tuple[list[Concept], list[ConceptRelation]]:
        """Extract concepts and relationships with strict provenance."""
        if isinstance(chunks, DocumentChunk):
            chunks = [chunks]

        # Concept collection keyed by canonical name
        concepts_by_name: dict[str, dict] = {}
        relations_by_sig: dict[tuple[str, str, str], ConceptRelation] = {}

        def register_concept(
            raw_name: str,
            chunk: DocumentChunk,
            confidence: float = 0.85,
            description: str | None = None,
        ) -> str:
            canon = self.normalizer.canonicalize(raw_name)
            if not canon or len(canon) < 2:
                return ""

            if canon not in concepts_by_name:
                concepts_by_name[canon] = {
                    "raw_occurrences": [raw_name],
                    "chunks": {chunk.chunk_id},
                    "docs": {chunk.document_id},
                    "pages": {chunk.page_number},
                    "sections": {chunk.section} if chunk.section else set(),
                    "confidence": confidence,
                    "description": description,
                }
            else:
                entry = concepts_by_name[canon]
                entry["raw_occurrences"].append(raw_name)
                entry["chunks"].add(chunk.chunk_id)
                entry["docs"].add(chunk.document_id)
                entry["pages"].add(chunk.page_number)
                if chunk.section:
                    entry["sections"].add(chunk.section)
                entry["confidence"] = max(entry["confidence"], confidence)
                if description and not entry["description"]:
                    entry["description"] = description

            return canon

        def register_relation(
            source_canon: str,
            target_canon: str,
            rel_type: RelationType,
            chunk: DocumentChunk,
            confidence: float = 0.80,
            prereq_nature: PrerequisiteNature | None = None,
            method: str = "heuristic",
        ) -> None:
            if not source_canon or not target_canon or source_canon == target_canon:
                return

            sub_id = chunk.subject_id or self.subject_id
            s_id = generate_concept_id(sub_id, source_canon)
            t_id = generate_concept_id(sub_id, target_canon)
            sig = (s_id, t_id, rel_type.value)

            if sig in relations_by_sig:
                existing = relations_by_sig[sig]
                merged_chunks = sorted(set(existing.source_chunk_ids + [chunk.chunk_id]))
                merged_docs = sorted(set(existing.source_document_ids + [chunk.document_id]))
                relations_by_sig[sig] = existing.model_copy(
                    update={
                        "source_chunk_ids": merged_chunks,
                        "source_document_ids": merged_docs,
                        "confidence": max(existing.confidence, confidence),
                    }
                )
            else:
                rid = generate_relation_id(sub_id, s_id, rel_type.value, t_id)
                relations_by_sig[sig] = ConceptRelation(
                    relation_id=rid,
                    subject_id=sub_id,
                    source_concept_id=s_id,
                    target_concept_id=t_id,
                    relation_type=rel_type,
                    confidence=confidence,
                    prerequisite_nature=prereq_nature,
                    source_chunk_ids=[chunk.chunk_id],
                    source_document_ids=[chunk.document_id],
                    extraction_method=method,
                )

        # Process each chunk
        for chunk in chunks:
            text = chunk.text
            section = chunk.section

            # 1. Heading signals from chunk.section
            if section:
                # Handle sections with "and", e.g. "LTI Systems and Convolution"
                if " and " in section.lower():
                    parts = re.split(r"(?i)\s+and\s+", section)
                    canons = [register_concept(p.strip(), chunk, confidence=0.95) for p in parts]
                    if len(canons) == 2 and canons[0] and canons[1]:
                        register_relation(
                            canons[0],
                            canons[1],
                            RelationType.RELATED_TO,
                            chunk,
                            confidence=0.85,
                            method="heading_cooccurrence",
                        )
                elif " versus " in section.lower() or " vs " in section.lower():
                    parts = re.split(r"(?i)\s+(?:versus|vs\.?)\s+", section)
                    canons = [register_concept(p.strip(), chunk, confidence=0.95) for p in parts]
                    if len(canons) == 2 and canons[0] and canons[1]:
                        register_relation(
                            canons[0],
                            canons[1],
                            RelationType.CONTRASTS_WITH,
                            chunk,
                            confidence=0.90,
                            method="heading_contrast",
                        )
                else:
                    register_concept(section.strip(), chunk, confidence=0.95)

            # 2. Definitional grammar patterns
            # "X is a/an/the fundamental ... used to Y"
            def_matches = re.finditer(
                r"(?i)\b([A-Z][\w\s\-]{2,30}?)\s+is\s+(?:a|an|the|defined\s+as|termed)\s+([^.]+?\.)",
                text,
            )
            for m in def_matches:
                term_cand = m.group(1).strip()
                desc_cand = m.group(0).strip()
                # Exclude pronouns and stopwords
                if term_cand.lower() not in {"it", "this", "that", "these", "there", "what", "which"}:
                    canon = register_concept(term_cand, chunk, confidence=0.90, description=desc_cand)

                    # Check if definition expresses USED_IN or DERIVED_FROM
                    if "used to" in desc_cand.lower() or "used in" in desc_cand.lower():
                        target_match = re.search(
                            r"(?i)used\s+(?:to\s+(?:determine|compute|find|characterize)|in)\s+(?:the\s+output\s+of\s+)?(?:a|an|the\s+)?([A-Z][\w\s\-]+)",
                            desc_cand,
                        )
                        if target_match:
                            tgt_cand = target_match.group(1).split(" ")[0:3]
                            tgt_canon = register_concept(" ".join(tgt_cand).strip(".,"), chunk, confidence=0.80)
                            register_relation(
                                canon,
                                tgt_canon,
                                RelationType.USED_IN,
                                chunk,
                                confidence=0.85,
                                method="grammar_definition",
                            )

            # 3. Explicit properties pattern:
            # "An LTI system satisfies two core properties: linearity ... and time-invariance."
            prop_matches = re.finditer(
                r"(?i)(?:properties\s+of\s+(?:an?\s+)?([A-Z][\w\s\-]+?)\s+include|([A-Z][\w\s\-]+?)\s+(?:satisfies|possesses|exhibits|has)\s+(?:the\s+)?(?:two\s+core\s+)?properties(?:\s+of)?[:\s]+|([A-Z][\w\s\-]+?)\s+is\s+characterized\s+by\s+)([^.]+?\.)",
                text,
            )
            for m in prop_matches:
                host_cand = (m.group(1) or m.group(2) or m.group(3) or "").strip()
                details = m.group(4)
                if host_cand:
                    host_canon = register_concept(host_cand, chunk, confidence=0.90)
                    for prop_word in PROPERTY_TERMS:
                        if prop_word in details.lower():
                            prop_canon = register_concept(prop_word.title(), chunk, confidence=0.85)
                            register_relation(
                                host_canon,
                                prop_canon,
                                RelationType.HAS_PROPERTY,
                                chunk,
                                confidence=0.90,
                                method="grammar_property",
                            )

            # 4. Prerequisite / Dependency grammar patterns:
            # "requires X", "prerequisite of Y", "condition under which X can be reconstructed"
            if "prerequisite" in text.lower():
                prereq_match = re.search(
                    r"(?i)([A-Z][\w\s\-]+?)\s+is\s+a\s+prerequisite\s+(?:for|to|of)\s+([A-Z][\w\s\-]+)",
                    text,
                )
                if prereq_match:
                    src = register_concept(prereq_match.group(1).strip(), chunk, confidence=0.95)
                    tgt = register_concept(prereq_match.group(2).strip(), chunk, confidence=0.95)
                    register_relation(
                        src,
                        tgt,
                        RelationType.PREREQUISITE_OF,
                        chunk,
                        confidence=0.95,
                        prereq_nature=PrerequisiteNature.EXPLICIT_PREREQUISITE,
                        method="explicit_text_prerequisite",
                    )

            # 5. Counterpart / Derived / Generalization patterns:
            # "Z-transform is the discrete-time counterpart of the Laplace transform"
            counterpart_match = re.search(
                r"(?i)([A-Z][\w\s\-]+?)\s+is\s+the\s+(?:discrete-time\s+)?counterpart\s+of\s+(?:the\s+)?([A-Z][\w\s\-]+)",
                text,
            )
            if counterpart_match:
                c1 = register_concept(counterpart_match.group(1).strip(), chunk, confidence=0.85)
                c2 = register_concept(counterpart_match.group(2).strip(), chunk, confidence=0.85)
                register_relation(
                    c1,
                    c2,
                    RelationType.RELATED_TO,
                    chunk,
                    confidence=0.85,
                    method="grammar_counterpart",
                )

            # "Laplace Transform generalizes the Fourier transform"
            gen_match = re.search(
                r"(?i)([A-Z][\w\s\-]+?)\s+generalizes\s+(?:the\s+)?([A-Z][\w\s\-]+)",
                text,
            )
            if gen_match:
                c1 = register_concept(gen_match.group(1).strip(), chunk, confidence=0.90)
                c2 = register_concept(gen_match.group(2).strip(), chunk, confidence=0.90)
                # When Concept 1 generalizes Concept 2 (e.g. Laplace generalizes Fourier),
                # Concept 2 is a foundational prerequisite of Concept 1.
                register_relation(
                    c2,
                    c1,
                    RelationType.PREREQUISITE_OF,
                    chunk,
                    confidence=0.85,
                    prereq_nature=PrerequisiteNature.INFERRED_PREREQUISITE,
                    method="grammar_generalization_prereq",
                )

            # 6. Specific core concept keywords present in text:
            core_signals_terms = [
                ("Convolution", "Continuous-time convolution is an integral operation"),
                ("LTI System", "Linear Time-Invariant"),
                ("Fourier Transform", "Continuous-Time Fourier Transform"),
                ("Sampling Theorem", "Nyquist-Shannon Sampling Theorem"),
                ("Nyquist Rate", "minimum sampling rate 2 * B is termed the Nyquist rate"),
                ("Aliasing", "distortion phenomenon is called aliasing"),
                ("Laplace Transform", "bilateral Laplace Transform"),
                ("Region of Convergence", "Region of Convergence (ROC)"),
                ("Z-Transform", "The Z-transform is the discrete-time counterpart"),
                ("Energy Signal", "energy signal has finite total energy"),
                ("Power Signal", "power signal has infinite total energy"),
                ("Impulse Response", "completely characterized by its unit impulse response"),
                ("BIBO Stability", "Bounded-Input Bounded-Output (BIBO) stable"),
                ("Linearity", "linearity"),
                ("Time Invariance", "time-invariance"),
                ("Time Invariance", "time invariance"),
            ]
            for term, phrase in core_signals_terms:
                if phrase.lower() in text.lower() or term.lower() in text.lower():
                    canon = register_concept(term, chunk, confidence=0.85)
                    # Check inferred relationships
                    if term in ("Linearity", "Time Invariance") and (
                        "lti" in text.lower() or "linear time-invariant" in text.lower()
                    ):
                        lti_canon = register_concept("LTI System", chunk, confidence=0.85)
                        register_relation(
                            lti_canon,
                            canon,
                            RelationType.HAS_PROPERTY,
                            chunk,
                            confidence=0.85,
                            method="domain_term_rule",
                        )
                    if term == "Convolution" and (
                        "lti" in text.lower() or "linear time-invariant" in text.lower()
                    ):
                        lti_canon = register_concept("LTI System", chunk, confidence=0.85)
                        if any(
                            w in text.lower()
                            for w in ["used in", "used to", "output of", "compute", "determine"]
                        ):
                            register_relation(
                                canon,
                                lti_canon,
                                RelationType.USED_IN,
                                chunk,
                                confidence=0.85,
                                method="domain_term_rule",
                            )
                    if term == "Nyquist Rate" and "sampling" in text.lower():
                        st_canon = register_concept("Sampling Theorem", chunk, confidence=0.85)
                        register_relation(
                            canon,
                            st_canon,
                            RelationType.PART_OF,
                            chunk,
                            confidence=0.80,
                            method="domain_term_rule",
                        )
                    if term == "Aliasing" and "sampling" in text.lower():
                        st_canon = register_concept("Sampling Theorem", chunk, confidence=0.85)
                        register_relation(
                            canon,
                            st_canon,
                            RelationType.RELATED_TO,
                            chunk,
                            confidence=0.80,
                            method="domain_term_rule",
                        )
                    if term == "Impulse Response" and "lti" in text.lower():
                        lti_canon = register_concept("LTI System", chunk, confidence=0.85)
                        register_relation(
                            canon,
                            lti_canon,
                            RelationType.USED_IN,
                            chunk,
                            confidence=0.85,
                            method="domain_term_rule",
                        )
                    if term == "Region of Convergence" and "laplace" in text.lower():
                        laplace_canon = register_concept("Laplace Transform", chunk, confidence=0.85)
                        register_relation(
                            canon,
                            laplace_canon,
                            RelationType.PART_OF,
                            chunk,
                            confidence=0.90,
                            method="domain_term_rule",
                        )

        # Inferred academic prerequisites:
        # Impulse Response & Linearity/Time Invariance -> PREREQUISITE_OF -> LTI System
        # LTI System & Convolution -> PREREQUISITE_OF -> Fourier Transform analysis
        # Sampling Theorem -> PREREQUISITE_OF -> Discrete-Time processing
        inferred_prereqs = [
            ("LTI System", "Convolution", 0.70),
            ("Fourier Transform", "Laplace Transform", 0.65),
            ("Sampling Theorem", "Z-Transform", 0.60),
        ]
        for src_name, tgt_name, conf in inferred_prereqs:
            if src_name in concepts_by_name and tgt_name in concepts_by_name:
                common_chunks = list(
                    concepts_by_name[src_name]["chunks"] | concepts_by_name[tgt_name]["chunks"]
                )
                sample_chunk_id = common_chunks[0] if common_chunks else "chunk_synthetic"
                dummy_chunk = DocumentChunk(
                    chunk_id=sample_chunk_id,
                    document_id="doc_inferred",
                    subject_id=self.subject_id,
                    page_number=1,
                    text="",
                    char_start=0,
                    char_end=0,
                )
                register_relation(
                    src_name,
                    tgt_name,
                    RelationType.PREREQUISITE_OF,
                    dummy_chunk,
                    confidence=conf,
                    prereq_nature=PrerequisiteNature.INFERRED_PREREQUISITE,
                    method="inferred_syllabus_flow",
                )

        # Build Concept objects
        final_concepts: list[Concept] = []
        for canon_name, data in concepts_by_name.items():
            cid = generate_concept_id(self.subject_id, canon_name)
            aliases = self.normalizer.extract_aliases(canon_name, data["raw_occurrences"])
            c_type = classify_concept_type(canon_name)

            final_concepts.append(
                Concept(
                    concept_id=cid,
                    subject_id=self.subject_id,
                    canonical_name=canon_name,
                    concept_type=c_type,
                    aliases=aliases,
                    description=data["description"],
                    source_document_ids=sorted(data["docs"]),
                    source_chunk_ids=sorted(data["chunks"]),
                    page_numbers=sorted(data["pages"]),
                    sections=sorted(data["sections"]),
                    confidence=data["confidence"],
                )
            )

        final_concepts.sort(key=lambda c: c.concept_id)
        final_relations = sorted(relations_by_sig.values(), key=lambda r: r.relation_id)

        return final_concepts, final_relations
