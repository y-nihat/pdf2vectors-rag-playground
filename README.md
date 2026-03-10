# pdf2vectors-rag-playground

Playground for parsing streamed PDFs from `pixparse/pdfa-eng-wds` and evaluating text extraction quality with Docling variants.

## Quickstart

```bash
uv venv .venv
source .venv/bin/activate
uv sync --extra dev
```

## Run

```bash
uv run docling-eval --limit 20
uv run parser-comparison-eval --limit 10 --granite-images-scale 2.0
```

## Project Layout

```text
src/
	data_sources/   # dataset streaming + reference text builders
	parsers/        # classic Docling + Granite-Docling parser adapters
	eval/           # metrics and evaluation runners
out/              # generated evaluation artifacts (gitignored)
```

## Module Docs

- [Source Overview](src/README.md)
- [Data Sources](src/data_sources/README.md)
- [Parsers](src/parsers/README.md)
- [Evaluation](src/eval/README.md)

## Notes

- Evaluation outputs are written under `out/` and intentionally ignored by Git.
- Entry points are configured in `pyproject.toml`:
	- `docling-eval`
	- `parser-comparison-eval`

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

You may use, modify, and share this project for non-commercial purposes with appropriate credit.
