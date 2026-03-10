"""Docling parsing adapter for PDF bytes."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from threading import Lock

from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream


# Keep one converter instance to preserve Docling pipeline/model initialization reuse.
# Docling's DocumentConverter (and downstream HybridChunker/tokenizer state) is not
# safe for concurrent access, so calls must be serialized under a process-local
# lock. `run_parser_comparison_eval.py` is currently sequential; if it is
# parallelized with threads later, this lock prevents races but throughput will
# still be single-conversion at a time.
SHARED_CONVERTER = DocumentConverter()
_CONVERTER_LOCK = Lock()


@dataclass(slots=True)
class ParsedDoclingResult:
    """Minimal Docling parse output used by evaluation."""

    key: str
    markdown: str


def parse_pdf_bytes(pdf_bytes: bytes, key: str) -> ParsedDoclingResult:
    """Parse PDF bytes with Docling and return markdown text."""
    doc_stream = DocumentStream(name=f"{key}.pdf", stream=BytesIO(pdf_bytes))
    # Never call convert concurrently on the shared converter.
    with _CONVERTER_LOCK:
        result = SHARED_CONVERTER.convert(doc_stream)
    return ParsedDoclingResult(key=key, markdown=result.document.export_to_markdown())


def parse_with_docling(pdf_bytes: bytes, key: str) -> str:
    """Parse PDF bytes with classic Docling and return markdown text."""
    return parse_pdf_bytes(pdf_bytes=pdf_bytes, key=key).markdown
