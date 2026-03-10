# Parsers

Adapters for extracting markdown text from PDF bytes.

## Files

- `docling_parser.py`
  - Wraps classic Docling conversion from PDF bytes.
- `granite_docling_parser.py`
  - Renders PDF pages to RGB images and runs Granite-Docling VLM conversion.
- `parser_interface.py`
  - Typed protocol/dataclass contracts used by evaluation runners.

## Design Notes

- Granite converter creation is cached with `lru_cache(maxsize=1)` to reduce repeated model initialization overhead.
- `render_pdf_to_images` closes PDF/page/bitmap handles defensively.
