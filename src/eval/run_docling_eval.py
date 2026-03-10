"""Run a small Docling text-extraction evaluation on pixparse/pdfa-eng-wds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from data_sources.pixparse_loader import (
    build_reference_text,
    iter_samples,
    parse_sample_json,
)
from eval.text_metrics import (
    length_ratio,
    normalize_text,
    normalized_levenshtein_similarity,
    token_precision_recall_f1,
)
from parsers.docling_parser import parse_pdf_bytes


def _preview_sample_json(sample_json: Any, max_chars: int = 400) -> str:
    """Return a compact preview of the sample JSON payload."""
    parsed = parse_sample_json(sample_json)
    preview = json.dumps(parsed, ensure_ascii=True)
    return preview[:max_chars]


def _write_outlier_texts(results: list[dict[str, Any]], out_dir: Path, count: int) -> None:
    """Persist a few worst-scoring samples for manual diff/error analysis."""
    if count <= 0:
        return

    outlier_dir = out_dir / "outliers"
    outlier_dir.mkdir(parents=True, exist_ok=True)

    worst_by_f1 = sorted(results, key=lambda row: row["token_f1"])[:count]
    for row in worst_by_f1:
        key = row["key"]
        (outlier_dir / f"{key}_ref.txt").write_text(row["reference_norm"], encoding="utf-8")
        (outlier_dir / f"{key}_docling.txt").write_text(row["docling_norm"], encoding="utf-8")


def run(limit: int, output_csv: Path, inspect_count: int, outlier_count: int) -> None:
    """Evaluate Docling markdown text vs dataset reference text on first ``limit`` docs."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for index, sample in enumerate(iter_samples(limit=limit), start=1):
        key = str(sample.get("__key__", f"sample_{index}"))

        if index <= inspect_count:
            parsed = parse_sample_json(sample["json"])
            pages = parsed.get("pages", [])
            print(f"[inspect] key={key}")
            print(f"[inspect] top-level keys={sorted(parsed.keys())}")
            print(f"[inspect] pages={len(pages) if isinstance(pages, list) else 0}")
            if isinstance(pages, list) and pages:
                page0 = pages[0]
                if isinstance(page0, dict):
                    line_text = page0.get("lines", {}).get("text", [])
                    if isinstance(line_text, list):
                        print(f"[inspect] first page line count={len(line_text)}")
                        print(f"[inspect] first lines={line_text[:5]}")
            print(f"[inspect] json preview={_preview_sample_json(sample['json'])}")

        try:
            reference_text = build_reference_text(sample["json"])
            docling_result = parse_pdf_bytes(sample["pdf"], key=key)
            docling_text = docling_result.markdown

            reference_norm = normalize_text(reference_text)
            docling_norm = normalize_text(docling_text)

            char_similarity = normalized_levenshtein_similarity(
                docling_norm,
                reference_norm,
            )
            precision, recall, f1 = token_precision_recall_f1(
                docling_norm,
                reference_norm,
            )
            ratio = length_ratio(docling_norm, reference_norm)

            row = {
                "key": key,
                "char_similarity": char_similarity,
                "token_precision": precision,
                "token_recall": recall,
                "token_f1": f1,
                "length_ratio": ratio,
                "ref_length": len(reference_norm),
                "docling_length": len(docling_norm),
                "reference_norm": reference_norm,
                "docling_norm": docling_norm,
            }
            rows.append(row)
            print(
                f"[{index}/{limit}] key={key} "
                f"char_sim={char_similarity:.4f} token_f1={f1:.4f} len_ratio={ratio:.4f}"
            )
        except Exception as exc:  # pragma: no cover - defensive logging for streamed data
            print(f"[{index}/{limit}] key={key} failed: {exc}")

    if not rows:
        raise RuntimeError("No successful samples to evaluate.")

    fieldnames = [
        "key",
        "char_similarity",
        "token_precision",
        "token_recall",
        "token_f1",
        "length_ratio",
        "ref_length",
        "docling_length",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})

    out_dir = output_csv.parent
    _write_outlier_texts(rows, out_dir=out_dir, count=outlier_count)

    summary = {
        "samples": len(rows),
        "mean_char_similarity": statistics.fmean(row["char_similarity"] for row in rows),
        "median_char_similarity": statistics.median(row["char_similarity"] for row in rows),
        "mean_token_f1": statistics.fmean(row["token_f1"] for row in rows),
        "median_token_f1": statistics.median(row["token_f1"] for row in rows),
        "mean_length_ratio": statistics.fmean(row["length_ratio"] for row in rows),
        "median_length_ratio": statistics.median(row["length_ratio"] for row in rows),
    }

    notes_path = out_dir / "docling_eval_notes.md"
    notes_path.write_text(
        "\n".join(
            [
                "# Docling Eval Notes",
                "",
                "## Ground Truth Mapping (v1)",
                "- `reference_text = build_reference_text(sample['json'])`",
                "- `sample['json']` contains `pages`; each page has `lines.text` in reading order.",
                "- v1 reference is page-wise concatenation of `pages[*].lines.text`.",
                "",
                "## Summary",
                f"- samples: {summary['samples']}",
                f"- mean_char_similarity: {summary['mean_char_similarity']:.6f}",
                f"- median_char_similarity: {summary['median_char_similarity']:.6f}",
                f"- mean_token_f1: {summary['mean_token_f1']:.6f}",
                f"- median_token_f1: {summary['median_token_f1']:.6f}",
                f"- mean_length_ratio: {summary['mean_length_ratio']:.6f}",
                f"- median_length_ratio: {summary['median_length_ratio']:.6f}",
                "",
                "## Outliers",
                f"- Saved lowest-F1 pairs under `{(out_dir / 'outliers').as_posix()}`.",
            ]
        ),
        encoding="utf-8",
    )

    print("\n=== Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print(f"CSV written to: {output_csv}")
    print(f"Notes written to: {notes_path}")


def main() -> None:
    """Parse CLI args and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Docling parse quality on pixparse/pdfa-eng-wds.",
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of streamed docs.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("out/docling_eval/docling_eval_scores.csv"),
        help="Path to write per-document scores.",
    )
    parser.add_argument(
        "--inspect-count",
        type=int,
        default=2,
        help="Number of initial samples to print JSON inspection details for.",
    )
    parser.add_argument(
        "--outlier-count",
        type=int,
        default=5,
        help="Number of lowest-F1 samples to persist for manual diffing.",
    )
    args = parser.parse_args()

    run(
        limit=args.limit,
        output_csv=args.output_csv,
        inspect_count=args.inspect_count,
        outlier_count=args.outlier_count,
    )


if __name__ == "__main__":
    main()
