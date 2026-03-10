"""Granite-Docling parser adapter for rendered PDF page images."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO

import pypdfium2 as pdfium
from PIL.Image import Image as PILImage

from docling.backend.image_backend import ImageDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmConvertOptions, VlmPipelineOptions
from docling.document_converter import DocumentConverter, FormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types.io import DocumentStream


def render_pdf_to_images(pdf_bytes: bytes, scale: float = 2.0) -> list[PILImage]:
    """Render PDF bytes into RGB PIL images in page order."""
    pdf_doc = pdfium.PdfDocument(pdf_bytes)
    try:
        page_count = len(pdf_doc)
        if page_count == 0:
            return []

        rendered_images: list[PILImage] = []
        render_scale = max(1, int(round(scale)))
        for page_index in range(page_count):
            page = pdf_doc[page_index]
            bitmap = page.render(scale=render_scale)
            image = bitmap.to_pil().convert("RGB")
            rendered_images.append(image)

            page_close = getattr(page, "close", None)
            if callable(page_close):
                page_close()

            bitmap_close = getattr(bitmap, "close", None)
            if callable(bitmap_close):
                bitmap_close()

        return rendered_images
    finally:
        pdf_doc_close = getattr(pdf_doc, "close", None)
        if callable(pdf_doc_close):
            pdf_doc_close()


@lru_cache(maxsize=1)
def _get_granite_docling_converter() -> DocumentConverter:
    """Create and cache a Docling VLM converter using the granite_docling preset."""
    vlm_convert_options = VlmConvertOptions.from_preset("granite_docling")
    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_convert_options,
        generate_page_images=True,
    )

    format_options = {
        InputFormat.IMAGE: FormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
            backend=ImageDocumentBackend,
        )
    }

    return DocumentConverter(
        allowed_formats=[InputFormat.IMAGE],
        format_options=format_options,
    )


def parse_with_granite_docling(pdf_bytes: bytes, key: str, images_scale: float = 2.0) -> str:
    """Parse PDF bytes with Granite-Docling by rendering and converting each page image."""
    page_images = render_pdf_to_images(pdf_bytes=pdf_bytes, scale=images_scale)
    if not page_images:
        return ""

    converter = _get_granite_docling_converter()
    page_markdown: list[str] = []

    for page_number, page_image in enumerate(page_images, start=1):
        buffer = BytesIO()
        page_image.save(buffer, format="PNG")
        buffer.seek(0)

        stream = DocumentStream(name=f"{key}_p{page_number}.png", stream=buffer)
        result = converter.convert(stream)
        markdown = result.document.export_to_markdown().strip()
        if markdown:
            page_markdown.append(markdown)

    return "\n\n".join(page_markdown).strip()
