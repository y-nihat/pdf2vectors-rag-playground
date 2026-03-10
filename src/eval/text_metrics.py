"""Text normalization and similarity metrics for OCR/parsing evaluation."""

from __future__ import annotations

import re
from collections import Counter


def normalize_text(text: str) -> str:
    """Normalize text for robust parser-to-reference comparison."""
    lowered = text.lower().strip()

    # Drop lines that are only digits (often page numbers).
    lines = [line for line in lowered.splitlines() if not re.fullmatch(r"\s*\d+\s*", line)]

    # Collapse all whitespace to single spaces for stable comparisons.
    return re.sub(r"\s+", " ", "\n".join(lines)).strip()


def normalized_levenshtein_similarity(text_a: str, text_b: str) -> float:
    """Return character-level normalized Levenshtein similarity in [0.0, 1.0]."""
    if text_a == text_b:
        return 1.0

    len_a = len(text_a)
    len_b = len(text_b)
    if len_a == 0 and len_b == 0:
        return 1.0
    if len_a == 0 or len_b == 0:
        return 0.0

    if len_a < len_b:
        text_a, text_b = text_b, text_a
        len_a, len_b = len_b, len_a

    previous_row = list(range(len_b + 1))
    for i, char_a in enumerate(text_a, start=1):
        current_row = [i]
        for j, char_b in enumerate(text_b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (char_a != char_b)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row

    distance = previous_row[-1]
    return 1.0 - (distance / max(len_a, len_b))


def token_precision_recall_f1(predicted_text: str, reference_text: str) -> tuple[float, float, float]:
    """Compute token-level precision, recall, and F1 using multiset overlap."""
    predicted_tokens = predicted_text.split()
    reference_tokens = reference_text.split()

    if not predicted_tokens and not reference_tokens:
        return 1.0, 1.0, 1.0

    predicted_counter = Counter(predicted_tokens)
    reference_counter = Counter(reference_tokens)
    overlap = sum((predicted_counter & reference_counter).values())

    precision = overlap / len(predicted_tokens) if predicted_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    if precision + recall == 0.0:
        return precision, recall, 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def length_ratio(predicted_text: str, reference_text: str) -> float:
    """Return predicted/reference character length ratio."""
    ref_length = len(reference_text)
    if ref_length == 0:
        return 1.0 if len(predicted_text) == 0 else 0.0
    return len(predicted_text) / ref_length
