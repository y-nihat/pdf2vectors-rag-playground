"""Run a minimal Docling parse on one streamed pixparse/pdfa-eng-wds sample."""

from io import BytesIO
from pathlib import Path

from datasets import load_dataset
from docling_core.types.io import DocumentStream
from docling.document_converter import DocumentConverter


def main() -> None:
    """Load one PDF sample from pixparse/pdfa-eng-wds and export Docling outputs."""
    output_dir = Path("out/docling")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stream a single sample to avoid downloading the full dataset.
    dataset = load_dataset("pixparse/pdfa-eng-wds", split="train", streaming=True)
    sample = next(iter(dataset))

    pdf_bytes = sample["pdf"]
    key = sample["__key__"]

    # Persist the raw PDF for easier debugging and reproducibility.
    pdf_path = output_dir / f"{key}.pdf"
    pdf_path.write_bytes(pdf_bytes)
    print(f"Saved PDF to: {pdf_path}")

    converter = DocumentConverter()
    doc_stream = DocumentStream(name=f"{key}.pdf", stream=BytesIO(pdf_bytes))
    result = converter.convert(doc_stream)

    md_path = output_dir / f"{key}.md"
    json_path = output_dir / f"{key}.json"

    result.document.save_as_markdown(md_path)
    result.document.save_as_json(json_path)

    print(f"Saved markdown to: {md_path}")
    print(f"Saved JSON to: {json_path}")


if __name__ == "__main__":
    main()
