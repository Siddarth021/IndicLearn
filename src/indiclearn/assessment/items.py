"""Deterministic evaluation logic for assessment items (Phase 7A).

Provides robust, reproducible evaluation of student answers without relying
on non-deterministic LLMs or uncalibrated semantic graders.
"""

from __future__ import annotations

import unicodedata

from indiclearn.assessment.models import AssessmentItem, QuestionType


def _normalize_str(text: str) -> str:
    """Normalize text with NFC form, lowercasing, and whitespace collapse."""
    return " ".join(unicodedata.normalize("NFC", text).strip().lower().split())


def _extract_option_key(option_text: str) -> str:
    """Extract option key (e.g. 'A' from 'A) ...' or '(A) ...')."""
    clean = option_text.strip()
    if clean.startswith("(") and len(clean) > 2 and clean[2] == ")":
        return clean[1].upper()
    if len(clean) > 1 and clean[1] in {")", ".", ":"}:
        return clean[0].upper()
    return ""


def evaluate_response(item: AssessmentItem, selected_answer: str) -> tuple[bool, str]:
    """Deterministically evaluate a student's answer against an assessment item.

    Parameters
    ----------
    item : AssessmentItem
        The item being evaluated.
    selected_answer : str
        Student-provided answer (choice identifier or text).

    Returns
    -------
    tuple[bool, str]
        (is_correct, feedback_string)
    """
    clean_student = selected_answer.strip()
    clean_target = item.correct_answer.strip()

    if item.question_type == QuestionType.MCQ:
        # Match choice letters directly (e.g. "A" == "A")
        if clean_student.upper() == clean_target.upper():
            return True, f"Correct. {item.explanation}"

        # Match against option keys and option contents
        target_key = clean_target.upper()
        # Find which option corresponds to target_key
        target_option_text = ""
        for opt in item.options:
            key = _extract_option_key(opt)
            if key == target_key:
                target_option_text = opt
                break

        # Check if student gave full option string matching the target option
        if target_option_text and _normalize_str(clean_student) == _normalize_str(target_option_text):
            return True, f"Correct. {item.explanation}"

        # Check if student gave option body (e.g. "Impulse response" matching "A) Impulse response")
        if target_option_text:
            body = target_option_text
            extracted = _extract_option_key(target_option_text)
            if extracted and len(body) > 2:
                body = body[2:].lstrip(").: ")
            if _normalize_str(clean_student) == _normalize_str(body):
                return True, f"Correct. {item.explanation}"

        # Check if student gave a letter that matches some other option
        return False, f"Incorrect. Correct answer is '{item.correct_answer}'. {item.explanation}"

    elif item.question_type == QuestionType.TRUE_FALSE:
        true_set = {"true", "t", "1", "yes", "సత్యం", "నిజం"}
        false_set = {"false", "f", "0", "no", "అసత్యం", "తప్పు"}

        norm_student = _normalize_str(clean_student)
        norm_target = _normalize_str(clean_target)

        student_bool = norm_student in true_set if (norm_student in true_set or norm_student in false_set) else None
        target_bool = norm_target in true_set if (norm_target in true_set or norm_target in false_set) else None

        if student_bool is not None and target_bool is not None:
            is_correct = student_bool == target_bool
        else:
            is_correct = norm_student == norm_target

        if is_correct:
            return True, f"Correct. {item.explanation}"
        return False, f"Incorrect. Correct answer is '{item.correct_answer}'. {item.explanation}"

    elif item.question_type == QuestionType.SHORT_ANSWER:
        is_correct = _normalize_str(clean_student) == _normalize_str(clean_target)
        if is_correct:
            return True, f"Correct. {item.explanation}"
        return False, f"Incorrect. Correct answer is '{item.correct_answer}'. {item.explanation}"

    # Unsupported or custom question type fallback
    is_correct = _normalize_str(clean_student) == _normalize_str(clean_target)
    if is_correct:
        return True, f"Correct. {item.explanation}"
    return False, f"Incorrect. Correct answer is '{item.correct_answer}'. {item.explanation}"
