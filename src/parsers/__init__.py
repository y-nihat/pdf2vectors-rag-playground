"""Parser adapters used by evaluation scripts."""

from parsers.docling_parser import parse_with_docling
from parsers.granite_docling_parser import (
    parse_with_granite_docling,
    render_pdf_to_images,
)

__all__ = [
    "parse_with_docling",
    "parse_with_granite_docling",
    "render_pdf_to_images",
]
