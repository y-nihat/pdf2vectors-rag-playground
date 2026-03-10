"""Load pixparse samples and build reference text for evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset


def iter_samples(limit: int, split: str = "train") -> Iterator[dict[str, Any]]:
    """Yield up to ``limit`` streamed samples from pixparse/pdfa-eng-wds."""
    if limit <= 0:
        return

    dataset = load_dataset("pixparse/pdfa-eng-wds", split=split, streaming=True)
    for index, sample in enumerate(dataset):
        if index >= limit:
            break
        yield sample


def parse_sample_json(sample_json: Any) -> dict[str, Any]:
    """Return sample JSON as a dictionary."""
    if isinstance(sample_json, dict):
        return sample_json

    if isinstance(sample_json, bytes):
        sample_json = sample_json.decode("utf-8")

    if isinstance(sample_json, str):
        parsed = json.loads(sample_json)
        if isinstance(parsed, dict):
            return parsed

    raise TypeError("Unsupported sample['json'] payload shape.")


def build_reference_text(sample_json: Any) -> str:
    """Build v1 ground-truth text by concatenating page lines in reading order."""
    parsed = parse_sample_json(sample_json)
    pages = parsed.get("pages")
    if not isinstance(pages, list):
        return ""

    page_texts: list[str] = []
    for page in pages:
        if not isinstance(page, dict):
            continue

        lines = page.get("lines")
        if isinstance(lines, dict):
            line_text = lines.get("text")
            if isinstance(line_text, list):
                cleaned_lines = [
                    line.strip()
                    for line in line_text
                    if isinstance(line, str) and line.strip()
                ]
                if cleaned_lines:
                    page_texts.append("\n".join(cleaned_lines))
                    continue

        # Fallback for samples where line text is missing but token text exists.
        words = page.get("words")
        if isinstance(words, dict):
            word_text = words.get("text")
            if isinstance(word_text, list):
                cleaned_words = [
                    word.strip()
                    for word in word_text
                    if isinstance(word, str) and word.strip()
                ]
                if cleaned_words:
                    page_texts.append(" ".join(cleaned_words))

    return "\n\n".join(page_texts).strip()
