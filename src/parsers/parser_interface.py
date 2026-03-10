"""Parser interface contracts for evaluation harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from PIL.Image import Image as PILImage


class RenderPdfToImagesFn(Protocol):
    """Function contract for converting PDF bytes to ordered page images."""

    def __call__(self, pdf_bytes: bytes, scale: float = 2.0) -> list[PILImage]:
        """Return rendered page images in page order."""


class ParsePdfFn(Protocol):
    """Function contract for parser adapters used by evaluation."""

    def __call__(self, pdf_bytes: bytes, key: str) -> str:
        """Return parser output text for a single PDF sample."""


@dataclass(slots=True)
class ParserFunctions:
    """Typed container for parser/evaluator wiring."""

    render_pdf_to_images: RenderPdfToImagesFn
    parse_with_docling: ParsePdfFn
    parse_with_granite_docling: ParsePdfFn


ParserName = str
ParserOutputMap = dict[ParserName, str]
ParserMap = dict[ParserName, Callable[[bytes, str], str]]
