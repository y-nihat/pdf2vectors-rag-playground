"""Compare classic Docling vs Granite-Docling on pixparse/pdfa-eng-wds."""

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
from parsers.docling_parser import parse_with_docling
from parsers.granite_docling_parser import (
    parse_with_granite_docling,
    render_pdf_to_images,
)
from parsers.parser_interface import ParserFunctions


def _preview_sample_json(sample_json: Any, max_chars: int = 400) -> str:
    """Return a short preview of JSON payload for schema inspection logs."""
    parsed = parse_sample_json(sample_json)
    return json.dumps(parsed, ensure_ascii=True)[:max_chars]


def _compute_metrics(predicted_text: str, reference_text: str) -> tuple[float, float, float, float, float]:
    """Compute char similarity, token precision, token recall, token F1, and length ratio."""
    char_similarity = normalized_levenshtein_similarity(predicted_text, reference_text)
    precision, recall, f1 = token_precision_recall_f1(predicted_text, reference_text)
    ratio = length_ratio(predicted_text, reference_text)
    return char_similarity, precision, recall, f1, ratio


def _write_delta_outliers(rows: list[dict[str, Any]], out_dir: Path, count: int) -> None:
    """Write text triplets for the most divergent parser outcomes."""
    if count <= 0:
        return

    granite_better = sorted(
        (row for row in rows if row["delta_token_f1"] > 0),
        key=lambda row: row["delta_token_f1"],
        reverse=True,
    )[:count]
    docling_better = sorted(
        (row for row in rows if row["delta_token_f1"] < 0),
        key=lambda row: row["delta_token_f1"],
    )[:count]

    for bucket_name, bucket_rows in (
        ("granite_better", granite_better),
        ("docling_better", docling_better),
    ):
        bucket_dir = out_dir / "outliers" / bucket_name
        bucket_dir.mkdir(parents=True, exist_ok=True)
        for row in bucket_rows:
            key = row["key"]
            (bucket_dir / f"{key}_reference.txt").write_text(
                row["reference_norm"],
                encoding="utf-8",
            )
            (bucket_dir / f"{key}_docling_classic.txt").write_text(
                row["docling_norm"],
                encoding="utf-8",
            )
            (bucket_dir / f"{key}_granite_docling.txt").write_text(
                row["granite_norm"],
                encoding="utf-8",
            )


def _aggregate_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    """Aggregate parser-specific metrics for summary reporting."""
    if not rows:
        return {
            "mean_char_similarity": float("nan"),
            "median_char_similarity": float("nan"),
            "mean_token_f1": float("nan"),
            "median_token_f1": float("nan"),
            "mean_length_ratio": float("nan"),
            "median_length_ratio": float("nan"),
        }

    return {
        "mean_char_similarity": statistics.fmean(row[f"{prefix}_char_similarity"] for row in rows),
        "median_char_similarity": statistics.median(row[f"{prefix}_char_similarity"] for row in rows),
        "mean_token_f1": statistics.fmean(row[f"{prefix}_token_f1"] for row in rows),
        "median_token_f1": statistics.median(row[f"{prefix}_token_f1"] for row in rows),
        "mean_length_ratio": statistics.fmean(row[f"{prefix}_length_ratio"] for row in rows),
        "median_length_ratio": statistics.median(row[f"{prefix}_length_ratio"] for row in rows),
    }


def run(
    limit: int,
    output_csv: Path,
    inspect_count: int,
    outlier_count: int,
    granite_images_scale: float,
) -> None:
    """Run parser-comparison evaluation across streamed dataset samples."""
    parser_fns = ParserFunctions(
        render_pdf_to_images=render_pdf_to_images,
        parse_with_docling=parse_with_docling,
        parse_with_granite_docling=lambda pdf_bytes, key: parse_with_granite_docling(
            pdf_bytes=pdf_bytes,
            key=key,
            images_scale=granite_images_scale,
        ),
    )

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
            print(f"[inspect] json preview={_preview_sample_json(sample['json'])}")

        try:
            reference_text = build_reference_text(sample["json"])
            reference_norm = normalize_text(reference_text)

            docling_text = parser_fns.parse_with_docling(sample["pdf"], key)
            granite_text = parser_fns.parse_with_granite_docling(sample["pdf"], key)

            docling_norm = normalize_text(docling_text)
            granite_norm = normalize_text(granite_text)

            (
                doc_char,
                doc_precision,
                doc_recall,
                doc_f1,
                doc_ratio,
            ) = _compute_metrics(docling_norm, reference_norm)
            (
                gra_char,
                gra_precision,
                gra_recall,
                gra_f1,
                gra_ratio,
            ) = _compute_metrics(granite_norm, reference_norm)

            row = {
                "key": key,
                "reference_norm": reference_norm,
                "docling_norm": docling_norm,
                "granite_norm": granite_norm,
                "docling_char_similarity": doc_char,
                "docling_token_precision": doc_precision,
                "docling_token_recall": doc_recall,
                "docling_token_f1": doc_f1,
                "docling_length_ratio": doc_ratio,
                "granite_char_similarity": gra_char,
                "granite_token_precision": gra_precision,
                "granite_token_recall": gra_recall,
                "granite_token_f1": gra_f1,
                "granite_length_ratio": gra_ratio,
                "delta_token_f1": gra_f1 - doc_f1,
            }
            rows.append(row)

            print(
                f"[{index}/{limit}] key={key} "
                f"doc_f1={doc_f1:.4f} gra_f1={gra_f1:.4f} delta={row['delta_token_f1']:+.4f}"
            )
        except Exception as exc:  # pragma: no cover - external model/runtime behavior
            print(f"[{index}/{limit}] key={key} failed: {exc}")

    if not rows:
        raise RuntimeError("No successful parser-comparison results were produced.")

    fieldnames = [
        "key",
        "docling_char_similarity",
        "docling_token_precision",
        "docling_token_recall",
        "docling_token_f1",
        "docling_length_ratio",
        "granite_char_similarity",
        "granite_token_precision",
        "granite_token_recall",
        "granite_token_f1",
        "granite_length_ratio",
        "delta_token_f1",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})

    out_dir = output_csv.parent
    _write_delta_outliers(rows=rows, out_dir=out_dir, count=outlier_count)

    docling_summary = _aggregate_metrics(rows, prefix="docling")
    granite_summary = _aggregate_metrics(rows, prefix="granite")

    summary_csv = out_dir / "parser_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "parser",
                "samples",
                "mean_char_similarity",
                "median_char_similarity",
                "mean_token_f1",
                "median_token_f1",
                "mean_length_ratio",
                "median_length_ratio",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "parser": "docling_classic",
                "samples": len(rows),
                **docling_summary,
            }
        )
        writer.writerow(
            {
                "parser": "granite_docling",
                "samples": len(rows),
                **granite_summary,
            }
        )

    notes_path = out_dir / "comparison_eval_notes.md"
    notes_path.write_text(
        "\n".join(
            [
                "# Parser Comparison Notes",
                "",
                "## Ground Truth Mapping (v1)",
                "- `reference_text = build_reference_text(sample['json'])`",
                "- `sample['json']` contains `pages`; each page has `lines.text` in reading order.",
                "",
                "## Summary",
                f"- samples: {len(rows)}",
                f"- docling_classic.mean_token_f1: {docling_summary['mean_token_f1']:.6f}",
                f"- granite_docling.mean_token_f1: {granite_summary['mean_token_f1']:.6f}",
                f"- mean_delta_token_f1 (granite-classic): {statistics.fmean(row['delta_token_f1'] for row in rows):.6f}",
                "",
                "## Outlier Buckets",
                f"- granite better: `{(out_dir / 'outliers' / 'granite_better').as_posix()}`",
                f"- classic better: `{(out_dir / 'outliers' / 'docling_better').as_posix()}`",
            ]
        ),
        encoding="utf-8",
    )

    print("\n=== Docling Classic Summary ===")
    for key, value in docling_summary.items():
        print(f"{key}: {value:.6f}")

    print("\n=== Granite-Docling Summary ===")
    for key, value in granite_summary.items():
        print(f"{key}: {value:.6f}")

    mean_delta = statistics.fmean(row["delta_token_f1"] for row in rows)
    print(f"\nmean_delta_token_f1 (granite-classic): {mean_delta:.6f}")
    print(f"Per-doc CSV written to: {output_csv}")
    print(f"Parser summary CSV written to: {summary_csv}")
    print(f"Notes written to: {notes_path}")


def main() -> None:
    """Parse CLI arguments and run parser-comparison evaluation."""
    parser = argparse.ArgumentParser(
        description="Compare classic Docling and Granite-Docling on pixparse/pdfa-eng-wds.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of streamed docs.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("out/parser_comparison/per_doc_scores.csv"),
        help="Path to write per-document comparison scores.",
    )
    parser.add_argument(
        "--inspect-count",
        type=int,
        default=2,
        help="Number of initial samples for schema/preview logging.",
    )
    parser.add_argument(
        "--outlier-count",
        type=int,
        default=3,
        help="Number of examples to save per delta bucket.",
    )
    parser.add_argument(
        "--granite-images-scale",
        type=float,
        default=2.0,
        help="Page-render scale used for Granite-Docling image inputs.",
    )
    args = parser.parse_args()

    run(
        limit=args.limit,
        output_csv=args.output_csv,
        inspect_count=args.inspect_count,
        outlier_count=args.outlier_count,
        granite_images_scale=args.granite_images_scale,
    )


if __name__ == "__main__":
    main()
