"""Docling parsing adapter for PDF bytes."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream


SHARED_CONVERTER = DocumentConverter()


@dataclass(slots=True)
class ParsedDoclingResult:
    """Minimal Docling parse output used by evaluation."""

    key: str
    markdown: str


def parse_pdf_bytes(pdf_bytes: bytes, key: str) -> ParsedDoclingResult:
    """Parse PDF bytes with Docling and return markdown text."""
    doc_stream = DocumentStream(name=f"{key}.pdf", stream=BytesIO(pdf_bytes))
    result = SHARED_CONVERTER.convert(doc_stream)
    return ParsedDoclingResult(key=key, markdown=result.document.export_to_markdown())


def parse_with_docling(pdf_bytes: bytes, key: str) -> str:
    """Parse PDF bytes with classic Docling and return markdown text."""
    return parse_pdf_bytes(pdf_bytes=pdf_bytes, key=key).markdown
