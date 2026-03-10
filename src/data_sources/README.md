# Data Sources

Utilities for reading streamed samples from `pixparse/pdfa-eng-wds` and building reference text for evaluation.

## Files

- `pixparse_loader.py`
  - `iter_samples(limit, split)`: yields streamed samples.
  - `parse_sample_json(sample_json)`: normalizes JSON payload shape.
  - `build_reference_text(sample_json)`: builds ground truth text from page lines.

## Design Notes

- Streaming is used to avoid downloading the full dataset.
- Reference text prefers `pages[*].lines.text`, with a fallback to `words.text`.
