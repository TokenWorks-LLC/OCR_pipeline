#!/usr/bin/env python3
"""Run preprocessing profile ablations for ensemble OCR pages."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
import sys
import time
import unicodedata
from typing import Any

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from production.ensemble_ocr import FortifiedOCREnsemble
from production.page_diagnostics import PageDiagnosticsAnalyzer
from production.preprocessing_profiles import PROFILE_AUTO, available_preprocessing_profiles


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("preprocessing_ablation")


def _edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            insert = dp[j] + 1
            delete = dp[j - 1] + 1
            replace = prev + cost
            prev = dp[j]
            dp[j] = min(insert, delete, replace)
    return dp[m]


def _char_error_rate(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(list(reference), list(hypothesis)) / len(reference)


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def _extract_diacritic_chars(text: str) -> list[str]:
    chars: list[str] = []
    for char in text:
        if not char.strip():
            continue
        decomposed = unicodedata.normalize("NFD", char)
        if any(unicodedata.combining(token) for token in decomposed[1:]):
            chars.append(char)
    return chars


def _counter_preservation_rate(reference_items: list[str], hypothesis_items: list[str]) -> float | None:
    if not reference_items:
        return None

    ref_counts: dict[str, int] = {}
    hyp_counts: dict[str, int] = {}
    for item in reference_items:
        ref_counts[item] = ref_counts.get(item, 0) + 1
    for item in hypothesis_items:
        hyp_counts[item] = hyp_counts.get(item, 0) + 1

    kept = 0
    for item, count in ref_counts.items():
        kept += min(count, hyp_counts.get(item, 0))
    return kept / sum(ref_counts.values())


def _detected_character_coverage(reference: str | None, hypothesis: str) -> float | None:
    if reference is None:
        stripped = [char for char in hypothesis if char.isalnum()]
        if not hypothesis:
            return 0.0
        return len(stripped) / max(len(hypothesis), 1)

    ref_chars = {char for char in reference if char.isalnum()}
    if not ref_chars:
        return None
    hyp_chars = {char for char in hypothesis if char.isalnum()}
    return len(ref_chars & hyp_chars) / len(ref_chars)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _parse_profile_list(raw: str | None) -> list[str]:
    if not raw:
        return list(available_preprocessing_profiles())

    parsed = [token.strip().lower() for token in raw.split(",") if token.strip()]
    if not parsed:
        return list(available_preprocessing_profiles())

    valid = set(available_preprocessing_profiles()) | {PROFILE_AUTO}
    normalized: list[str] = []
    for profile in parsed:
        if profile not in valid:
            raise ValueError(f"Unknown profile '{profile}'. Valid values: {sorted(valid)}")
        normalized.append(profile)
    return normalized


def _collect_pages(manifest: str | None, inputs: str | None, max_pages: int) -> list[tuple[str, int]]:
    pages: list[tuple[str, int]] = []

    if manifest:
        with open(manifest, "r", encoding="utf-8") as handle:
            for idx, raw in enumerate(handle):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if idx == 0 and line.lower().startswith("pdf_path"):
                    continue

                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                pdf_path = parts[0]
                try:
                    page_num = int(parts[1]) - 1
                except Exception:
                    continue
                pages.append((pdf_path, page_num))
                if max_pages > 0 and len(pages) >= max_pages:
                    return pages
        return pages

    if not inputs:
        return pages

    root = Path(inputs)
    pdf_files = sorted([path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"])
    for pdf_path in pdf_files:
        try:
            with fitz.open(str(pdf_path)) as doc:
                count = len(doc)
        except Exception:
            continue

        for page_num in range(count):
            pages.append((str(pdf_path), page_num))
            if max_pages > 0 and len(pages) >= max_pages:
                return pages
    return pages


def _row_get(row: dict[str, str], candidates: list[str]) -> str:
    for key in candidates:
        if key in row and row[key] is not None:
            value = str(row[key]).strip()
            if value:
                return value
    return ""


def _gold_key(pdf_name: str, page_num_1_based: int) -> tuple[str, str]:
    base = Path(pdf_name).name
    stem = Path(base).stem
    return (base.casefold(), f"{stem.casefold()}::{page_num_1_based}")


def _load_gold_map(gold_csv_path: str | None) -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}
    if not gold_csv_path:
        return mapping

    with open(gold_csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pdf_name = _row_get(row, ["pdf_name", "pdf", "pdf_path", "PDF LINK", "pdf link", "input_file"])
            page_raw = _row_get(row, ["page", "page_number", "PAGE", "pdf_page"])
            text = _row_get(
                row,
                ["ground_truth_text", "gold_text", "reference_text", "HANDTYPED", "handtyped", "text"],
            )
            if not pdf_name or not page_raw or not text:
                continue
            try:
                page_num = int(float(page_raw))
            except Exception:
                continue

            key = _gold_key(pdf_name, page_num)
            mapping[key] = text
    return mapping


def _lookup_gold_text(gold_map: dict[tuple[str, str], str], pdf_path: str, page_num_0_based: int) -> str | None:
    key = _gold_key(pdf_path, page_num_0_based + 1)
    return gold_map.get(key)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _format_optional_float(value: float | None) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def run_ablation(args: argparse.Namespace) -> int:
    profiles = _parse_profile_list(args.profiles)
    pages = _collect_pages(args.manifest, args.inputs, args.max_pages)
    if not pages:
        logger.error("No pages found to evaluate.")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir: Path | None = None
    if args.debug_artifacts:
        debug_dir = output_dir / "debug_artifacts"
        debug_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_analyzer = PageDiagnosticsAnalyzer.from_profile(args.profile, default_dpi=300)
    ensemble = FortifiedOCREnsemble(
        profile_path=args.profile,
        per_engine_timeout_s=(args.engine_timeout_ms / 1000.0) if args.engine_timeout_ms else None,
    )

    gold_map = _load_gold_map(args.gold_csv)

    per_page_csv = output_dir / "preprocessing_ablation_per_page.csv"
    summary_csv = output_dir / "preprocessing_ablation_summary.csv"
    by_page_type_csv = output_dir / "preprocessing_ablation_by_page_type.csv"
    outputs_jsonl = output_dir / "preprocessing_ablation_outputs.jsonl"

    fieldnames = [
        "pdf_path",
        "page_num",
        "profile_requested",
        "profile_applied",
        "page_type",
        "runtime_ms",
        "empty_output",
        "confidence",
        "char_count",
        "detected_character_coverage",
        "diacritic_preservation",
        "cer",
        "wer",
        "orientation_class",
        "rotation_angle",
        "detected_layout_type",
        "detected_column_count",
        "detected_region_count",
        "detected_has_footnotes",
        "detected_has_table_interruptions",
        "reading_order_confidence",
        "reading_order_source",
        "region_ocr_used",
        "estimated_skew_degrees",
        "contrast_score",
        "noise_score",
        "foreground_ratio",
        "layout_complexity_score",
        "recommended_preprocessing_profile",
    ]

    rows: list[dict[str, Any]] = []
    with per_page_csv.open("w", encoding="utf-8", newline="") as csv_handle, outputs_jsonl.open(
        "w", encoding="utf-8"
    ) as jsonl_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
        writer.writeheader()

        for index, (pdf_path, page_num) in enumerate(pages, start=1):
            diagnostics, _ = diagnostics_analyzer.inspect_page(
                pdf_path,
                page_num,
                language_hint=args.language_hint,
            )
            diag_meta = diagnostics.to_pipeline_metadata()
            page_type = str(diag_meta.get("recommended_ocr_strategy", "unknown") or "unknown")
            reference_text = _lookup_gold_text(gold_map, pdf_path, page_num)

            for profile in profiles:
                start = time.perf_counter()
                text, meta = ensemble.extract_page_text(
                    pdf_path,
                    page_num,
                    preprocessing_profile=profile,
                    diagnostics=diag_meta,
                    language_hint=args.language_hint,
                    debug_artifacts_dir=str(debug_dir) if debug_dir else None,
                    debug_artifact_prefix=f"{Path(pdf_path).stem}_p{page_num + 1:04d}_{profile}",
                )
                runtime_ms = (time.perf_counter() - start) * 1000.0

                applied_profile = str(meta.get("preprocessing_profile", profile) or profile)
                confidence = _to_float(meta.get("confidence", 0.0))
                char_count = len(text or "")
                empty_output = not bool((text or "").strip())
                coverage = _detected_character_coverage(reference_text, text or "")
                diacritic_preservation = _counter_preservation_rate(
                    _extract_diacritic_chars(reference_text or ""),
                    _extract_diacritic_chars(text or ""),
                )

                cer = _char_error_rate(reference_text, text) if reference_text is not None else None
                wer = _word_error_rate(reference_text, text) if reference_text is not None else None

                row: dict[str, Any] = {
                    "pdf_path": pdf_path,
                    "page_num": page_num + 1,
                    "profile_requested": profile,
                    "profile_applied": applied_profile,
                    "page_type": page_type,
                    "runtime_ms": f"{runtime_ms:.4f}",
                    "empty_output": "true" if empty_output else "false",
                    "confidence": _format_optional_float(confidence),
                    "char_count": str(char_count),
                    "detected_character_coverage": _format_optional_float(coverage),
                    "diacritic_preservation": _format_optional_float(diacritic_preservation),
                    "cer": _format_optional_float(cer),
                    "wer": _format_optional_float(wer),
                    "orientation_class": str(meta.get("orientation_class", "") or ""),
                    "rotation_angle": _format_optional_float(_to_float(meta.get("rotation_angle", 0.0))),
                    "detected_layout_type": str(meta.get("detected_layout_type", "") or ""),
                    "detected_column_count": str(_to_int(meta.get("detected_column_count", 0))),
                    "detected_region_count": str(_to_int(meta.get("detected_region_count", 0))),
                    "detected_has_footnotes": "true" if bool(meta.get("detected_has_footnotes", False)) else "false",
                    "detected_has_table_interruptions": "true" if bool(meta.get("detected_has_table_interruptions", False)) else "false",
                    "reading_order_confidence": _format_optional_float(_to_float(meta.get("reading_order_confidence", 0.0))),
                    "reading_order_source": str(meta.get("reading_order_source", "") or ""),
                    "region_ocr_used": "true" if bool(meta.get("region_ocr_used", False)) else "false",
                    "estimated_skew_degrees": _format_optional_float(_to_float(diag_meta.get("estimated_skew_degrees", 0.0))),
                    "contrast_score": _format_optional_float(_to_float(diag_meta.get("contrast_score", 0.0))),
                    "noise_score": _format_optional_float(_to_float(diag_meta.get("noise_score", 0.0))),
                    "foreground_ratio": _format_optional_float(_to_float(diag_meta.get("foreground_ratio", 0.0))),
                    "layout_complexity_score": _format_optional_float(_to_float(diag_meta.get("layout_complexity_score", 0.0))),
                    "recommended_preprocessing_profile": str(diag_meta.get("recommended_preprocessing_profile", "") or ""),
                }
                writer.writerow(row)
                rows.append(row)

                jsonl_payload = {
                    "pdf_path": pdf_path,
                    "page_num": page_num + 1,
                    "profile_requested": profile,
                    "profile_applied": applied_profile,
                    "page_type": page_type,
                    "runtime_ms": runtime_ms,
                    "ocr_text": text,
                    "diagnostics": diag_meta,
                    "ocr_meta": meta,
                    "ground_truth_text": reference_text,
                    "cer": cer,
                    "wer": wer,
                    "detected_character_coverage": coverage,
                    "diacritic_preservation": diacritic_preservation,
                }
                jsonl_handle.write(json.dumps(jsonl_payload, ensure_ascii=False) + "\n")

            if index % 10 == 0 or index == len(pages):
                logger.info("Processed %s/%s pages", index, len(pages))

    _write_summaries(rows, summary_csv, by_page_type_csv)

    logger.info("Ablation complete")
    logger.info("Per-page metrics: %s", per_page_csv)
    logger.info("Summary metrics: %s", summary_csv)
    logger.info("Page-type summary: %s", by_page_type_csv)
    logger.info("Detailed outputs: %s", outputs_jsonl)
    return 0


def _write_summaries(rows: list[dict[str, Any]], summary_csv: Path, by_page_type_csv: Path) -> None:
    profile_groups: dict[str, list[dict[str, Any]]] = {}
    profile_type_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for row in rows:
        profile = str(row.get("profile_applied", "") or "")
        page_type = str(row.get("page_type", "unknown") or "unknown")
        profile_groups.setdefault(profile, []).append(row)
        profile_type_groups.setdefault((profile, page_type), []).append(row)

    summary_fields = [
        "profile",
        "samples",
        "empty_rate",
        "mean_runtime_ms",
        "mean_confidence",
        "mean_char_count",
        "mean_cer",
        "mean_wer",
        "mean_detected_character_coverage",
        "mean_diacritic_preservation",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for profile in sorted(profile_groups):
            rows_for_profile = profile_groups[profile]
            writer.writerow(_aggregate_rows(profile, None, rows_for_profile))

    by_page_type_fields = ["page_type", *summary_fields]
    with by_page_type_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=by_page_type_fields)
        writer.writeheader()
        for profile, page_type in sorted(profile_type_groups):
            rows_for_group = profile_type_groups[(profile, page_type)]
            aggregated = _aggregate_rows(profile, page_type, rows_for_group)
            writer.writerow(aggregated)


def _aggregate_rows(profile: str, page_type: str | None, rows: list[dict[str, Any]]) -> dict[str, Any]:
    runtime_vals = [_to_float(row.get("runtime_ms", "0")) for row in rows]
    confidence_vals = [_to_float(row.get("confidence", "0")) for row in rows if str(row.get("confidence", "")).strip()]
    char_counts = [_to_float(row.get("char_count", "0")) for row in rows]

    cer_vals = [_to_float(row.get("cer", "0")) for row in rows if str(row.get("cer", "")).strip()]
    wer_vals = [_to_float(row.get("wer", "0")) for row in rows if str(row.get("wer", "")).strip()]
    coverage_vals = [
        _to_float(row.get("detected_character_coverage", "0"))
        for row in rows
        if str(row.get("detected_character_coverage", "")).strip()
    ]
    diacritic_vals = [
        _to_float(row.get("diacritic_preservation", "0"))
        for row in rows
        if str(row.get("diacritic_preservation", "")).strip()
    ]

    empty_count = sum(1 for row in rows if str(row.get("empty_output", "")).strip().lower() == "true")

    aggregated: dict[str, Any] = {
        "profile": profile,
        "samples": str(len(rows)),
        "empty_rate": _format_optional_float(_safe_rate(empty_count, len(rows))),
        "mean_runtime_ms": _format_optional_float(_safe_mean(runtime_vals)),
        "mean_confidence": _format_optional_float(_safe_mean(confidence_vals)),
        "mean_char_count": _format_optional_float(_safe_mean(char_counts)),
        "mean_cer": _format_optional_float(_safe_mean(cer_vals)),
        "mean_wer": _format_optional_float(_safe_mean(wer_vals)),
        "mean_detected_character_coverage": _format_optional_float(_safe_mean(coverage_vals)),
        "mean_diacritic_preservation": _format_optional_float(_safe_mean(diacritic_vals)),
    }
    if page_type is not None:
        aggregated["page_type"] = page_type
    return aggregated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OCR preprocessing profile ablation")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=str, help="Manifest TSV with pdf_path and page columns")
    source.add_argument("--inputs", type=str, help="Root directory to recursively scan for PDFs")

    parser.add_argument("--output-dir", type=str, required=True, help="Directory for ablation outputs")
    parser.add_argument("--profile", type=str, default="profiles/akkadian_strict.json", help="Pipeline profile JSON")
    parser.add_argument(
        "--profiles",
        type=str,
        default="",
        help=(
            "Comma-separated preprocessing profiles to test; defaults to all named profiles. "
            f"Supports {PROFILE_AUTO} for diagnostics-driven per-page selection."
        ),
    )
    parser.add_argument("--gold-csv", type=str, default="", help="Optional GT CSV for CER/WER and preservation metrics")
    parser.add_argument("--language-hint", type=str, default="unknown", help="Optional language/domain hint")
    parser.add_argument("--engine-timeout-ms", type=int, default=0, help="Per-engine timeout in milliseconds")
    parser.add_argument("--max-pages", type=int, default=0, help="Optional cap on number of pages to process")
    parser.add_argument("--debug-artifacts", action="store_true", help="Write profile debug images and OCR text files")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run_ablation(args)
    except Exception as exc:
        logger.exception("Ablation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
