"""Prompt templates, multilingual formatting policies, and grounding guidelines (Phase 5)."""

from __future__ import annotations

# Standard system prompt enforcing evidence-only answering
RAG_SYSTEM_PROMPT = """You are IndicLearn's academic assistant. Your goal is to help engineering students understand course concepts accurately.

MANDATORY RULES:
1. Answer the student's question using ONLY the supplied academic evidence.
2. Every factual claim must reference the exact source identifier using square brackets (e.g. [S1] or [S1, S2]).
3. Do NOT invent facts or concepts that are absent from the provided evidence.
4. Do NOT cite sources that were not provided in the context (never invent unprovided source identifiers or unknown authors).
5. If the supplied evidence does not contain enough information to answer confidently, you MUST state that the available material is insufficient.
6. Match the requested language and tone while keeping technical terms precise.
"""

INSUFFICIENT_EVIDENCE_MESSAGES = {
    "en": "I could not find sufficient information in the uploaded academic materials to answer this question reliably.",
    "te": "అందించిన విద్యా విషయాలలో ఈ ప్రశ్నకు ఖచ్చితమైన సమాధానం ఇవ్వడానికి తగినంత సమాచారం దొరకలేదు.",
    "te-Latn": "Upload chesina academic material lo e prasnaku saripoyina information ledu.",
    "te-en": "Uploaded academic material lo deeniki sambhandhinchina sufficient information ledu.",
}

OUT_OF_SCOPE_MESSAGES = {
    "en": "This question appears to be outside the scope of the introductory Signals & Systems curriculum. The uploaded course materials do not cover this topic.",
    "te": "ఈ ప్రశ్న పరిచయ సిగ్నల్స్ & సిస్టమ్స్ పాఠ్యప్రణాళిక పరిధికి వెలుపల ఉన్నట్లుంది. అప్‌లోడ్ చేసిన మెటీరియల్‌లో ఈ అంశం లేదు.",
    "te-Latn": "E prasna introductory Signals & Systems curriculum scope lo ledu. Upload chesina material lo e topic cover avvaledu.",
    "te-en": "E question introductory Signals & Systems syllabus scope lo ledu. Uploaded material lo deeniki related content ledu.",
}

AMBIGUOUS_SCOPE_MESSAGES = {
    "en": "This question is too broad or ambiguous to answer reliably from the course materials. Please specify which Signals & Systems concept you would like to know about (e.g., ideal reconstruction filter vs. LTI system response).",
    "te": "ఈ ప్రశ్న చాలా సాధారణంగా లేదా అస్పష్టంగా ఉంది. దయచేసి మీరు ఏ నిర్దిష్ట సిగ్నల్స్ & సిస్టమ్స్ భావన గురించి తెలుసుకోవాలనుకుంటున్నారో పేర్కొనండి.",
    "te-Latn": "E prasna chala broad ga undi. Dayachesi specific Signals & Systems concept ni mention cheyandi.",
    "te-en": "E question chala general ga undi. Please specify the exact Signals & Systems concept you want to learn.",
}


def build_user_prompt(
    query: str,
    context_str: str,
    requested_language: str = "en",
) -> str:
    """Construct user prompt combining query, structured context, and language instruction."""
    lang_instructions = {
        "en": "Please provide a clear, factual answer in English with citations.",
        "te": "దయచేసి మూలాలను ఉదహరిస్తూ (citations) తెలుగులో స్పష్టమైన వివరణ ఇవ్వండి.",
        "te-Latn": "Please explain in conversational Romanized Telugu (Telugu in English alphabet) citing sources [S1], [S2].",
        "te-en": "Please explain in simple Telugu-English code-mixed style citing sources [S1], [S2].",
    }

    instruction = lang_instructions.get(
        requested_language, "Please provide a factual answer citing sources [S1], [S2]."
    )

    prompt = f"""{context_str}

STUDENT QUESTION:
{query}

INSTRUCTION:
{instruction}
Remember: Cite all supporting statements using [S1], [S2], etc. If the evidence above does not answer the question, state that the material is insufficient."""
    return prompt
