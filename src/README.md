# Source Overview

This directory contains the runtime code for dataset loading, parser adapters, and evaluation scripts.

## Submodules

- [data_sources](data_sources/README.md): dataset streaming and reference-text construction.
- [parsers](parsers/README.md): Docling parser adapters and PDF-to-image rendering.
- [eval](eval/README.md): text metrics and evaluation runners.

## Boundaries

- `data_sources` does not depend on parser implementations.
- `parsers` does not depend on evaluation logic.
- `eval` orchestrates `data_sources` and `parsers` and writes reports.

## Usage

Run evaluations from the repository root:

```bash
uv run docling-eval --limit 20
uv run parser-comparison-eval --limit 10
```
