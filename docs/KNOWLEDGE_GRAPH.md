# Academic Knowledge Graph Architecture and Ontology

Status: **IMPLEMENTED (Phase 3 Verified)**  
Last Updated: 2026-10-06  
Primary Backend: `NetworkXGraphStore` (`networkx.MultiDiGraph`)  
Protocol: `GraphStore` (`src/indiclearn/knowledge_graph/base.py`)  

---

## 1. Architectural Role and Scope

The **Academic Knowledge Graph** represents the objective, canonical structure of an academic subject (such as *Signals and Systems*). It models what concepts exist in the curriculum and how those concepts are inter-related.

```text
                     Academic Documents
                             ↓
                     Document Chunks
                             ↓
             ┌───────────────┴───────────────┐
             ↓                               ↓
       Vector Index                   Concept Graph
 (Semantic chunk similarity)       (Curriculum relationships)
             ↓                               ↓
       Dense/Hybrid RAG              Topological Traversal /
                                   Prerequisite Pre-requisition
             ↓                               ↓
             └───────────────┬───────────────┘
                             ↓
              Context-Aware Instruction / RAG
```

### Critical Architectural Separation

The Academic Knowledge Graph is strictly decoupled from the learner:

```text
CANONICAL ACADEMIC GRAPH                  STUDENT KNOWLEDGE MODEL
(Objective Subject Structure)             (Personal Learner State)
-----------------------------             -----------------------
- Concept definitions                     - Mastery scores
- Canonical names and aliases             - Confidence estimates
- Curriculum prerequisites                - Quiz performance history
- Provenance (chunk/doc links)            - Forgetting curves / spaced review
```

> [!IMPORTANT]
> The academic knowledge graph **never** stores student-specific attributes such as `knowledge_score`, `mastery`, `confidence_level`, or `student_id`. Those attributes belong exclusively to the `StudentKnowledgeState` entity (see [DATA_MODEL.md](file:///d:/Indic_NLP/docs/DATA_MODEL.md)).

---

## 2. Subject Isolation

Concepts and relationships are strictly isolated by subject using a mandatory `subject_id` (e.g., `signals_and_systems`). This prevents homonymous terms across disparate disciplines (e.g., "Signal" or "Register" in *Signals and Systems* vs. *Digital Logic*) from erroneously colliding in a shared ontology.

---

## 3. Concept Domain Entity and Identity

### Concept Schema

A canonical concept is modeled as:

```python
class Concept(BaseModel):
    concept_id: str                   # Deterministic ID: cpt_{subject_id}_{slug}
    subject_id: str                   # Scoping discipline identifier
    canonical_name: str               # Standardized title-cased name
    concept_type: ConceptType         # Categorization (concept, property, method, etc.)
    aliases: list[str]                # Multilingual surface aliases (EN, TE, Code-mixed)
    description: str | None           # Extracted definitional text
    source_document_ids: list[str]    # Origin documents
    source_chunk_ids: list[str]       # Exact chunk provenance
    page_numbers: list[int]           # Source page numbers
    sections: list[str]               # Source headings / section titles
    confidence: float                 # Heuristic extraction confidence (0.0 - 1.0)
    metadata: dict[str, Any]
    created_at: datetime
```

### Concept Types (`ConceptType`)

1. **`concept`**: Primary theoretical noun or paradigm (e.g., *LTI System*, *Continuous-Time Signal*).
2. **`subconcept`**: Specialized variant or subdivision (e.g., *Periodic Signal*, *Continuous-Time Convolution*).
3. **`property`**: Attribute, axiom, or invariant governing a concept (e.g., *Linearity*, *Time Invariance*, *BIBO Stability*).
4. **`method`**: Algorithmic procedure, operation, or technique (e.g., *Convolution*, *Filtering*, *Sampling*).
5. **`formula`**: Mathematical theorem, equation, or quantitative criterion (e.g., *Sampling Theorem*, *Nyquist Rate*).
6. **`example`**: Concrete illustration, circuit, or canonical problem.

### Deterministic Concept ID Strategy

Concept IDs are uniquely and deterministically generated from the subject ID and the canonical concept name:

$$\text{concept\_id} = \texttt{"cpt\_"} + \text{slugify}(\text{subject\_id}) + \texttt{"\_"} + \text{slugify}(\text{canonical\_name})$$

Examples:
- `subject="signals_and_systems"`, `canonical_name="LTI System"` $\rightarrow$ `cpt_signals_and_systems_lti_system`
- `subject="signals_and_systems"`, `canonical_name="Continuous-Time Fourier Transform"` $\rightarrow$ `cpt_signals_and_systems_continuous_time_fourier_transform`

This guarantees idempotency: processing multiple documents or chunks containing "Convolution" consistently maps to `cpt_signals_and_systems_convolution`, aggregating occurrences rather than fragmenting into duplicate IDs.

---

## 4. Concept Normalization and Alias Layer

Academic texts contain diverse surface variations, abbreviations, and multilingual expressions. The `ConceptNormalizer` (`src/indiclearn/concepts/normalizer.py`) handles:

1. **Acronym Expansion & Disambiguation**:
   - `LTI` / `LTI system` / `Linear Time-Invariant System` $\rightarrow$ `LTI System`
   - `ROC` $\rightarrow$ `Region of Convergence`
   - `CTFT` $\rightarrow$ `Continuous-Time Fourier Transform`
   - `BIBO` $\rightarrow$ `BIBO Stability`
2. **Title-Casing & Hyphen Preservation**:
   - Accurately capitalizes hyphenated technical compounds (`Continuous-Time Signal`, `Time-Invariance`).
3. **Prefix & Article Stripping**:
   - Strips leading articles (`the`, `a`, `an`) and meta-descriptors (`properties of`, `characteristics of`).
4. **Noise Filtering**:
   - Discards sentence fragments terminating in or beginning with stop prepositions/conjunctions (`a continuous or`, `in`, `and`).
5. **Multilingual Aliases**:
   - Retains aliases across English, Telugu script, and Romanized Telugu:
     - Canonical: `Convolution`
     - Aliases: `convolution operation`, `continuous-time convolution`, `కన్వల్యూషన్`, `convolution ante`

---

## 5. Controlled Relation Ontology

Relationships are directed edges between concepts with strict semantics:

```python
class RelationType(str, Enum):
    RELATED_TO = "RELATED_TO"
    PREREQUISITE_OF = "PREREQUISITE_OF"
    PART_OF = "PART_OF"
    SUBCONCEPT_OF = "SUBCONCEPT_OF"
    HAS_PROPERTY = "HAS_PROPERTY"
    USED_IN = "USED_IN"
    DERIVED_FROM = "DERIVED_FROM"
    CONTRASTS_WITH = "CONTRASTS_WITH"
```

### Semantics and Directionality

| Relation Type | Direction Semantics | Example | Meaning |
| :--- | :--- | :--- | :--- |
| **`PREREQUISITE_OF`** | $A \rightarrow B$ | `Linearity` $\rightarrow$ `LTI System` | Concept $A$ must be understood before Concept $B$. In-edges to $B$ are $B$'s prerequisites. |
| **`HAS_PROPERTY`** | $Host \rightarrow Property$ | `LTI System` $\rightarrow$ `Time Invariance` | The host concept possesses or satisfies the specified property. |
| **`USED_IN`** | $Tool \rightarrow Context$ | `Convolution` $\rightarrow$ `LTI System` | Operation/tool $A$ is applied within context or system $B$. |
| **`PART_OF`** | $Component \rightarrow Whole$ | `Nyquist Rate` $\rightarrow$ `Sampling Theorem` | Concept $A$ is a formal structural constituent or parameter of $B$. |
| **`SUBCONCEPT_OF`** | $Specific \rightarrow General$ | `Energy Signal` $\rightarrow$ `Signal` | $A$ is a subtype or hyponym of $B$. |
| **`DERIVED_FROM`** | $Target \rightarrow Source$ | `Fourier Transform` $\rightarrow$ `Fourier Series` | $A$ is formulated from or mathematically derived from $B$. |
| **`CONTRASTS_WITH`** | $A \leftrightarrow B$ | `Energy Signal` $\rightarrow$ `Power Signal` | $A$ and $B$ represent mutually exclusive or contrasting paradigms. |
| **`RELATED_TO`** | $A \rightarrow B$ | `Aliasing` $\rightarrow$ `Sampling Theorem` | General conceptual or contextual association not covered by stricter relations. |

### Prerequisite Distinction

Prerequisite edges explicitly record their epistemic nature via `PrerequisiteNature`:
- **`EXPLICIT_PREREQUISITE`**: Directly stated in syllabus, section introduction, or textbook definition (e.g., *"Understanding Fourier transform is a prerequisite for Laplace transform"*). Confidence $\ge 0.90$.
- **`INFERRED_PREREQUISITE`**: Inferred through structural dependency rules, curriculum flow, or generalization axioms (e.g., Laplace generalizes Fourier $\implies$ Fourier is an inferred prerequisite). Confidence $\approx 0.70 - 0.85$.

---

## 6. Source Provenance

Every node and edge maintains full traceability back to raw document chunks:
- **Concept Provenance**: `source_document_ids`, `source_chunk_ids`, `page_numbers`, `sections`.
- **Relation Provenance**: `source_chunk_ids`, `source_document_ids`, `extraction_method` (e.g., `heading_contrast`, `grammar_property`, `domain_term_rule`), `confidence`.

This guarantees that every graph element can answer: *"Which paragraph in the uploaded course material substantiates this relationship?"*

---

## 7. Storage Architecture: `GraphStore` Protocol

The storage layer is decoupled using the `GraphStore` Protocol:

```text
ConceptService
      ↓
GraphStore Protocol
      ↓
NetworkXGraphStore (MultiDiGraph)
      ↓ (Future Backends)
[Neo4jGraphStore / RDFStore]
```

### Protocol Operations (`src/indiclearn/knowledge_graph/base.py`)

- `add_concept(concept)` / `get_concept(id)` / `delete_concept(id)` / `list_concepts(subject_id)`
- `add_relation(relation)` / `get_relation(id)` / `get_relations(...)` / `delete_relation(id)`
- `get_neighbors(concept_id, direction="both"|"in"|"out")`
- `get_prerequisites(concept_id, transitive=False|True)`: Traversing backwards along `PREREQUISITE_OF` edges.
- `get_dependents(concept_id, transitive=False|True)`: Traversing forwards along `PREREQUISITE_OF` edges.
- `find_concept_by_alias(alias, subject_id)`
- `search_concepts(query, subject_id, limit=5)`
- `save(path)` / `load(path)`

### Persistence Format

Graphs are persisted as human-readable, deterministic JSON:

```json
{
  "format_version": 1,
  "subject_id": "signals_and_systems",
  "created_at": "2026-10-06T11:45:00Z",
  "node_count": 22,
  "edge_count": 18,
  "nodes": [...],
  "edges": [...]
}
```

Nodes and edges are sorted deterministically by ID during serialization to guarantee diff reproducibility across runs.

---

## 8. Graph Validation and Integrity

The validation subsystem (`src/indiclearn/knowledge_graph/validation.py`) enforces:
1. **Referential Integrity**: Every relation source and target must exist as a registered node in the graph.
2. **No Self-Loops**: Prohibits $A \rightarrow A$ edges.
3. **Duplicate Detection**: Flags duplicate relation triplets $(u, \text{type}, v)$.
4. **Provenance Auditing**: Flags concepts or relations missing chunk IDs (warning in normal mode, error in strict mode).
5. **Prerequisite DAG Verification**: Detects cycles in the `PREREQUISITE_OF` subgraph using `networkx.simple_cycles`. Cycles are flagged as warnings/concerns for curriculum review without destructively mutating the graph.

---

## 9. Baseline Extraction and Evaluation (EXP-003)

### Heuristic Baseline

Extraction is performed without external LLM dependencies via `HeuristicConceptExtractor`:
1. **Markdown Heading Hierarchies**: Extracts chapter and section titles, mapping `A versus B` to `CONTRASTS_WITH`.
2. **Definitional Grammar Patterns**: Recognizes phrases such as *"X is defined as ..."*, *"X is characterized by ..."*, and *"X is used to compute Y"*.
3. **Property Term Patterns**: Detects property lists following host concepts (e.g., *"LTI system satisfies ... Linearity and Time Invariance"*).
4. **Generalization Patterns**: Recognizes *"X generalizes Y"*, inferring prerequisite relationships.
5. **Core Academic Keyword Matching**: Recognizes canonical engineering terminology and maps domain relationships with provenance.

### EXP-003 Benchmark Results

Evaluated against the manual gold standard (`data/evaluation/concept_graph_gold.json`, 15 concepts, 11 relations across 10 academic chunks):

| Category | Precision | Recall | F1 Score | TP | FP | FN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Concept Extraction** | **0.6818** | **1.0000** | **0.8108** | 15 | 7 | 0 |
| **Relation Extraction** | **0.5556** | **0.9091** | **0.6897** | 10 | 8 | 1 |

*Note: All results represent empirical performance on the current controlled evaluation benchmark and should not be construed as unconstrained domain claims.*
