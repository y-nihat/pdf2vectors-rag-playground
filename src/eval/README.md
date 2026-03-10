# Evaluation

Runners and metrics for comparing parser outputs against dataset reference text.

## Files

- `text_metrics.py`
  - Normalization, normalized Levenshtein similarity, token precision/recall/F1, length ratio.
- `run_docling_eval.py`
  - Evaluates classic Docling parser against references.
- `run_parser_comparison_eval.py`
  - Compares classic Docling and Granite-Docling side by side.

## Outputs

Both runners write artifacts under `out/`, including:

- per-document score CSV files
- aggregate summary notes/CSV files
- outlier text files for manual inspection

## Commands

```bash
uv run docling-eval --limit 20
uv run parser-comparison-eval --limit 10 --granite-images-scale 2.0
```
