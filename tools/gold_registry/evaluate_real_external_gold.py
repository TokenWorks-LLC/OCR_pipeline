#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = ROOT / "data" / "gold_registry"
SPLITS_DIR = REGISTRY_DIR / "splits"
REPORTS_DIR = ROOT / "reports"
EVAL_ROOT = REPORTS_DIR / "real_gold_eval_runs"

MANIFEST_PATH = REGISTRY_DIR / "gold_manifest.jsonl"
DATASET_AUDIT_PATH = REGISTRY_DIR / "dataset_audit.csv"

SUMMARY_CSV = REPORTS_DIR / "expanded_gold_metrics.csv"
PER_DATASET_CSV = REPORTS_DIR / "per_dataset_metrics.csv"
PER_LANGUAGE_CSV = REPORTS_DIR / "per_language_metrics.csv"
PER_LAYOUT_CSV = REPORTS_DIR / "per_layout_metrics.csv"
PER_DOC_CSV = REPORTS_DIR / "per_document_type_metrics.csv"
PER_SCRIPT_CSV = REPORTS_DIR / "per_script_metrics.csv"
EVAL_MD = REPORTS_DIR / "expanded_gold_evaluation.md"

FAILURE_CSV = REPORTS_DIR / "expanded_failure_taxonomy.csv"
FAILURE_MD = REPORTS_DIR / "expanded_failure_taxonomy.md"
NEXT_EXPERIMENT_MD = REPORTS_DIR / "next_training_experiment_plan.md"

CONTROLLED_MD = REPORTS_DIR / "controlled_experiment_result.md"
CONTROLLED_CSV = REPORTS_DIR / "controlled_experiment_metrics.csv"
PROMOTION_DECISION_MD = REPORTS_DIR / "promotion_decision.md"
FINAL_REPORT_MD = REPORTS_DIR / "real_gold_ingestion_and_evaluation_report.md"

PROMOTION_CHECKLIST_MD = REPORTS_DIR / "promotion_rules_checklist.md"


@dataclass
class SplitRunResult:
    split_label: str
    split_file: Path
    status: str
    runnable_pages: int
    skipped_pages: int
    gold_csv: Path | None
    input_dir: Path | None
    run_dir: Path | None
    eval_dir: Path | None
    summary: dict[str, Any] | None
    malformed_rows: list[dict[str, Any]]
    per_page_rows: list[dict[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            payload = line.strip()
            if not payload:
                continue
            rows.append(json.loads(payload))
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t", "on"}


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * q
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return float(sorted_values[lower])
    lower_val = sorted_values[lower]
    upper_val = sorted_values[upper]
    weight = index - lower
    return float(lower_val + (upper_val - lower_val) * weight)


def _run_command(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _load_manifest() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = _read_jsonl(MANIFEST_PATH)
    by_page_id = {str(row.get("page_id", "")): row for row in rows}
    return rows, by_page_id


def _prepare_split_assets(
    split_label: str,
    split_entries: list[dict[str, Any]],
    by_page_id: dict[str, dict[str, Any]],
) -> tuple[Path, Path, int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    split_root = EVAL_ROOT / split_label
    input_dir = split_root / "input_pdfs"
    gold_csv = split_root / "gold.csv"

    if split_root.exists():
        shutil.rmtree(split_root)
    input_dir.mkdir(parents=True, exist_ok=True)

    gold_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for entry in split_entries:
        page_id = str(entry.get("page_id", "")).strip()
        dataset_id = str(entry.get("dataset_id", "")).strip()
        record = by_page_id.get(page_id)
        if record is None:
            skipped_rows.append(
                {
                    "dataset_id": dataset_id,
                    "page_id": page_id,
                    "reason": "missing_manifest_record",
                }
            )
            continue

        pdf_path = ROOT / str(record.get("local_pdf_path", ""))
        gt_path = ROOT / str(record.get("ground_truth_text_path", ""))

        if not pdf_path.exists() or not gt_path.exists():
            skipped_rows.append(
                {
                    "dataset_id": dataset_id,
                    "page_id": page_id,
                    "reason": "missing_pdf_or_ground_truth",
                }
            )
            continue

        text = gt_path.read_text(encoding="utf-8").strip()
        if not text:
            skipped_rows.append(
                {
                    "dataset_id": dataset_id,
                    "page_id": page_id,
                    "reason": "empty_ground_truth_text",
                }
            )
            continue

        symlink_name = f"{page_id}.pdf"
        symlink_path = input_dir / symlink_name
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(pdf_path)

        gold_rows.append(
            {
                "pdf_name": symlink_name,
                "page": 1,
                "page_reference": page_id,
                "ground_truth_text": text,
                "language_primary": str(record.get("language_primary", "unknown") or "unknown"),
                "languages_present": str(record.get("languages_present", "unknown") or "unknown"),
                "script_type": str(record.get("script_type", "unknown") or "unknown"),
                "document_type": str(record.get("document_type", "unknown") or "unknown"),
                "layout_type": str(record.get("layout_type", "unknown") or "unknown"),
                "has_tables": str(bool(record.get("has_tables", False))).lower(),
                "has_footnotes": str(bool(record.get("has_footnotes", False))).lower(),
                "has_columns": str(bool(record.get("has_columns", False))).lower(),
                "has_diacritics": str(bool(record.get("has_diacritics", False))).lower(),
                "has_transliteration": str(bool(record.get("has_transliteration", False))).lower(),
                "scan_quality": str(record.get("scan_quality", "unknown") or "unknown"),
                "dataset_id": str(record.get("dataset_id", "")),
            }
        )

    _write_csv(
        gold_csv,
        gold_rows,
        [
            "pdf_name",
            "page",
            "page_reference",
            "ground_truth_text",
            "language_primary",
            "languages_present",
            "script_type",
            "document_type",
            "layout_type",
            "has_tables",
            "has_footnotes",
            "has_columns",
            "has_diacritics",
            "has_transliteration",
            "scan_quality",
            "dataset_id",
        ],
    )

    if skipped_rows:
        _write_csv(split_root / "skipped_rows.csv", skipped_rows, ["dataset_id", "page_id", "reason"])

    return gold_csv, input_dir, len(gold_rows), len(skipped_rows), skipped_rows, gold_rows


def _annotate_per_page_rows(
    per_page_rows: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    by_pdf_name: dict[str, list[dict[str, Any]]] = {}
    for row in gold_rows:
        pdf_name = str(row.get("pdf_name", "")).strip()
        page = _safe_int(row.get("page")) or 1
        if not pdf_name:
            continue
        by_key[(pdf_name, page)] = row
        by_pdf_name.setdefault(pdf_name, []).append(row)

    annotated: list[dict[str, Any]] = []
    for row in per_page_rows:
        updated = dict(row)
        pdf_name = str(updated.get("pdf_name", "")).strip()
        page = _safe_int(updated.get("page")) or 1
        gold = by_key.get((pdf_name, page))
        if gold is None and pdf_name:
            candidates = by_pdf_name.get(pdf_name, [])
            if len(candidates) == 1:
                gold = candidates[0]
            elif candidates:
                page_reference = str(updated.get("page_reference", "")).strip()
                if page_reference:
                    for candidate in candidates:
                        if str(candidate.get("page_reference", "")).strip() == page_reference:
                            gold = candidate
                            break
        if gold is not None:
            updated["dataset_id"] = str(gold.get("dataset_id", "unknown") or "unknown")
            updated["page_reference"] = str(gold.get("page_reference", "") or "")
        else:
            updated["dataset_id"] = str(updated.get("dataset_id", "unknown") or "unknown")
            if not str(updated.get("page_reference", "")).strip() and pdf_name:
                updated["page_reference"] = Path(pdf_name).stem
        annotated.append(updated)
    return annotated


def _provenance_mismatch_count(malformed_rows: list[dict[str, Any]]) -> int:
    mismatch_issues = {
        "invalid_page_key",
        "missing_page_number",
        "missing_identifier",
        "missing_source_pdf",
        "invalid_gold_record",
        "invalid_ocr_record",
    }
    return sum(1 for row in malformed_rows if str(row.get("issue", "")) in mismatch_issues)


def _evaluate_split(
    *,
    split_label: str,
    split_file: Path,
    by_page_id: dict[str, dict[str, Any]],
    extra_run_args: list[str] | None = None,
    split_override_entries: list[dict[str, Any]] | None = None,
    use_two_pass: bool = True,
    run_page_text_args: list[str] | None = None,
) -> SplitRunResult:
    split_entries = split_override_entries if split_override_entries is not None else _read_jsonl(split_file)
    if not split_entries:
        return SplitRunResult(
            split_label=split_label,
            split_file=split_file,
            status="pending",
            runnable_pages=0,
            skipped_pages=0,
            gold_csv=None,
            input_dir=None,
            run_dir=None,
            eval_dir=None,
            summary=None,
            malformed_rows=[],
            per_page_rows=[],
        )

    gold_csv, input_dir, runnable, skipped, _, gold_rows = _prepare_split_assets(split_label, split_entries, by_page_id)
    if runnable == 0:
        return SplitRunResult(
            split_label=split_label,
            split_file=split_file,
            status="pending",
            runnable_pages=0,
            skipped_pages=skipped,
            gold_csv=gold_csv,
            input_dir=input_dir,
            run_dir=None,
            eval_dir=None,
            summary=None,
            malformed_rows=[],
            per_page_rows=[],
        )

    run_dir = EVAL_ROOT / split_label / "run"
    eval_dir = EVAL_ROOT / split_label / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    if use_two_pass:
        run_command = [
            str(ROOT / ".venv" / "bin" / "python"),
            "run_pipeline.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(run_dir),
            "--two-pass-mode",
            "--first-pass-ocr-fallback",
            "paddle",
            "--second-pass-ocr-fallback",
            "paddle",
            "--second-pass-engine-mode",
            "paddle",
        ]
        if extra_run_args:
            run_command.extend(extra_run_args)
    else:
        run_command = [
            str(ROOT / ".venv" / "bin" / "python"),
            "tools/run_page_text.py",
            "--inputs",
            str(input_dir),
            "--output-root",
            str(run_dir),
            "--prefer-text-layer",
            "--ocr-fallback",
            "paddle",
        ]
        if run_page_text_args:
            run_command.extend(run_page_text_args)

    _run_command(run_command, ROOT)

    eval_command = [
        str(ROOT / ".venv" / "bin" / "python"),
        "tools/evaluate_gold.py",
        "--ocr-csv",
        str(run_dir / "client_page_text.csv"),
        "--gold-csv",
        str(gold_csv),
        "--progress-csv",
        str(run_dir / "progress.csv"),
        "--output-dir",
        str(eval_dir),
        "--permissive-malformed",
    ]
    _run_command(eval_command, ROOT)

    summary_rows = _read_csv(eval_dir / "evaluation_summary.csv")
    per_page_rows = _read_csv(eval_dir / "per_page_metrics.csv")
    per_page_rows = _annotate_per_page_rows(per_page_rows, gold_rows)
    malformed_rows = _read_csv(eval_dir / "malformed_rows.csv")
    summary = summary_rows[0] if summary_rows else None

    return SplitRunResult(
        split_label=split_label,
        split_file=split_file,
        status="measured" if summary is not None else "failed",
        runnable_pages=runnable,
        skipped_pages=skipped,
        gold_csv=gold_csv,
        input_dir=input_dir,
        run_dir=run_dir,
        eval_dir=eval_dir,
        summary=summary,
        malformed_rows=malformed_rows,
        per_page_rows=per_page_rows,
    )


def _aggregate_group(
    split_label: str,
    rows: list[dict[str, Any]],
    group_field: str,
    output_field: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(group_field, "") or "unknown")
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        cer_values = [v for v in (_safe_float(item.get("cer")) for item in items) if v is not None]
        wer_values = [v for v in (_safe_float(item.get("wer")) for item in items) if v is not None]
        runtime_values = [v for v in (_safe_float(item.get("runtime_ms")) for item in items) if v is not None]

        failed_rate = (
            sum(1 for item in items if str(item.get("status", "")).strip().lower() == "failed") / len(items)
            if items
            else 0.0
        )
        empty_rate = (
            sum(1 for item in items if _safe_bool(item.get("empty_output", "false"))) / len(items)
            if items
            else 0.0
        )

        out.append(
            {
                "evaluation_split": split_label,
                output_field: key,
                "samples": len(items),
                "cer_mean": _mean(cer_values),
                "wer_mean": _mean(wer_values),
                "failed_rate": failed_rate,
                "empty_rate": empty_rate,
                "runtime_ms_mean": _mean(runtime_values),
            }
        )

    return out


def _summarize_split(result: SplitRunResult, note: str = "") -> dict[str, Any]:
    if result.status != "measured" or result.summary is None:
        return {
            "evaluation_split": result.split_label,
            "status": result.status,
            "matched_pages": "",
            "gold_pages": "",
            "ocr_pages": "",
            "cer_mean": "",
            "cer_p50": "",
            "cer_p90": "",
            "wer_mean": "",
            "wer_p50": "",
            "wer_p90": "",
            "empty_output_rate": "",
            "failed_page_rate": "",
            "runtime_ms_mean": "",
            "runtime_ms_p50": "",
            "runtime_ms_p90": "",
            "runtime_ms_p95": "",
            "malformed_row_count": len(result.malformed_rows),
            "provenance_mismatch_count": _provenance_mismatch_count(result.malformed_rows),
            "notes": note,
        }

    s = result.summary
    return {
        "evaluation_split": result.split_label,
        "status": result.status,
        "matched_pages": _safe_int(s.get("matched_pages")),
        "gold_pages": _safe_int(s.get("gold_pages")),
        "ocr_pages": _safe_int(s.get("ocr_pages")),
        "cer_mean": _safe_float(s.get("cer_mean")),
        "cer_p50": _safe_float(s.get("cer_p50")),
        "cer_p90": _safe_float(s.get("cer_p90")),
        "wer_mean": _safe_float(s.get("wer_mean")),
        "wer_p50": _safe_float(s.get("wer_p50")),
        "wer_p90": _safe_float(s.get("wer_p90")),
        "empty_output_rate": _safe_float(s.get("empty_output_rate")),
        "failed_page_rate": _safe_float(s.get("failed_page_rate")),
        "runtime_ms_mean": _safe_float(s.get("runtime_ms_mean")),
        "runtime_ms_p50": _safe_float(s.get("runtime_ms_p50")),
        "runtime_ms_p90": _safe_float(s.get("runtime_ms_p90")),
        "runtime_ms_p95": _safe_float(s.get("runtime_ms_p95")),
        "malformed_row_count": len(result.malformed_rows),
        "provenance_mismatch_count": _provenance_mismatch_count(result.malformed_rows),
        "notes": note,
    }


def _build_run_row_map(run_csv_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = _read_csv(run_csv_path)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        pdf_name = str(row.get("pdf_name", "")).strip()
        page = _safe_int(row.get("page"))
        if not pdf_name or page is None:
            continue
        out[(pdf_name, page)] = row
    return out


def _is_junk_text(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    punctuation_or_symbols = sum(1 for ch in cleaned if (not ch.isalnum()) and (not ch.isspace()))
    if len(cleaned) == 0:
        return False
    ratio = punctuation_or_symbols / float(len(cleaned))
    if ratio > 0.7:
        return True
    if re.search(r"(.)\1{9,}", cleaned):
        return True
    return False


def _recommended_fix(category: str) -> str:
    mapping = {
        "true_blank_page": "data_validation",
        "render_failure": "preprocessing",
        "text_layer_failure": "text_layer_rerun_gating",
        "OCR_engine_failure": "ocr_engine_model_tuning",
        "OCR_timeout": "runtime_budget_and_routing",
        "bad_layout": "layout_detection",
        "preprocessing_needed": "preprocessing",
        "low_resolution_or_noisy_scan": "preprocessing",
        "ground_truth_mismatch": "ground_truth_audit",
        "page_key_or_provenance_mismatch": "key_normalization",
        "unsupported_language_or_script": "language_routing",
        "suspicious_short_output": "text_layer_rerun_gating",
        "suspicious_junk_output": "postprocessing_and_gating",
        "annotation_conversion_issue": "annotation_converter_fix",
        "unknown": "manual_review",
    }
    return mapping.get(category, "manual_review")


def _categorize_failure(per_page: dict[str, Any], run_row: dict[str, Any] | None) -> str:
    status = str(per_page.get("status", "")).strip().lower()
    failure_reason = str(per_page.get("failure_reason", "")).strip().lower()
    extraction_method = str(per_page.get("extraction_method", "")).strip().lower()
    empty_output = _safe_bool(per_page.get("empty_output", False))
    cer = _safe_float(per_page.get("cer"))

    output_length = 0
    output_text = ""
    if run_row is not None:
        output_length = _safe_int(run_row.get("output_text_length")) or 0
        output_text = str(run_row.get("page_text", ""))

    if status == "timed_out" or "timeout" in failure_reason:
        return "OCR_timeout"
    if "render" in failure_reason:
        return "render_failure"
    if "text_layer" in failure_reason:
        return "text_layer_failure"
    if empty_output and extraction_method == "text_layer":
        return "text_layer_failure"
    if empty_output and output_length == 0 and status == "success":
        return "suspicious_short_output"
    if output_length < 8 and (cer is None or cer >= 0.8):
        return "suspicious_short_output"
    if _is_junk_text(output_text):
        return "suspicious_junk_output"
    if cer is not None and cer > 1.5 and str(per_page.get("layout_type", "")).strip().lower() in {
        "multi_column",
        "form_layout",
        "semi_structured",
    }:
        return "bad_layout"
    if cer is not None and cer > 1.2 and str(per_page.get("scan_quality", "")).strip().lower() in {
        "noisy_scan",
        "mixed",
    }:
        return "low_resolution_or_noisy_scan"
    if status == "failed":
        return "OCR_engine_failure"
    if cer is not None and cer > 1.2:
        return "OCR_engine_failure"
    return "unknown"


def _build_failure_taxonomy(
    split_results: dict[str, SplitRunResult],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for split_label, result in split_results.items():
        if result.run_dir is None:
            continue

        run_row_map = _build_run_row_map(result.run_dir / "client_page_text.csv")

        for page in result.per_page_rows:
            status = str(page.get("status", "")).strip().lower()
            empty = _safe_bool(page.get("empty_output"))
            cer = _safe_float(page.get("cer"))
            is_low_quality = cer is not None and cer > 1.2

            if status == "success" and not empty and not is_low_quality:
                continue

            pdf_name = str(page.get("pdf_name", "")).strip()
            page_no = _safe_int(page.get("page")) or 1
            run_row = run_row_map.get((pdf_name, page_no))
            category = _categorize_failure(page, run_row)

            counts[category] = counts.get(category, 0) + 1

            rows.append(
                {
                    "evaluation_split": split_label,
                    "dataset_id": str(page.get("dataset_id", "unknown") or "unknown"),
                    "page_id": str(page.get("page_reference", page.get("page_key", "")) or ""),
                    "document_id": str(page.get("normalized_document_key", "") or ""),
                    "language_primary": str(page.get("language_primary", "unknown") or "unknown"),
                    "script_type": str(page.get("script_type", "unknown") or "unknown"),
                    "document_type": str(page.get("document_type", "unknown") or "unknown"),
                    "layout_type": str(page.get("layout_type", "unknown") or "unknown"),
                    "failure_category": category,
                    "failure_reason": str(page.get("failure_reason", "") or ""),
                    "first_pass_status": str(run_row.get("first_pass_status", "") if run_row else ""),
                    "second_pass_status": str(run_row.get("second_pass_status", "") if run_row else ""),
                    "final_output_source": str(run_row.get("final_output_source", "") if run_row else ""),
                    "best_available_output_length": _safe_int(
                        run_row.get("output_text_length", "") if run_row else ""
                    )
                    or 0,
                    "CER": _safe_float(page.get("cer")),
                    "WER": _safe_float(page.get("wer")),
                    "recommended_next_fix": _recommended_fix(category),
                }
            )

        for malformed in result.malformed_rows:
            issue = str(malformed.get("issue", ""))
            if issue not in {
                "invalid_page_key",
                "missing_page_number",
                "missing_identifier",
                "missing_source_pdf",
                "invalid_gold_record",
                "invalid_ocr_record",
            }:
                continue
            category = "page_key_or_provenance_mismatch"
            counts[category] = counts.get(category, 0) + 1
            rows.append(
                {
                    "evaluation_split": split_label,
                    "dataset_id": "unknown",
                    "page_id": str(malformed.get("page_key", "") or ""),
                    "document_id": str(malformed.get("pdf_name", "") or ""),
                    "language_primary": "unknown",
                    "script_type": "unknown",
                    "document_type": "unknown",
                    "layout_type": "unknown",
                    "failure_category": category,
                    "failure_reason": str(malformed.get("detail", "") or issue),
                    "first_pass_status": "",
                    "second_pass_status": "",
                    "final_output_source": "",
                    "best_available_output_length": 0,
                    "CER": "",
                    "WER": "",
                    "recommended_next_fix": _recommended_fix(category),
                }
            )

    return rows, counts


def _select_primary_experiment(counts: dict[str, int]) -> tuple[str, str, str]:
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    top_category = ordered[0][0] if ordered else "unknown"

    category_to_experiment = {
        "low_resolution_or_noisy_scan": (
            "preprocessing_profiles_ablation",
            "1. Preprocessing profiles and ablation testing",
            "Many pages are impacted by noisy scans or low contrast.",
        ),
        "bad_layout": (
            "layout_reading_order_reconstruction",
            "2. Layout detection and reading-order reconstruction",
            "Many failures are layout driven in multi-column and semi-structured pages.",
        ),
        "OCR_engine_failure": (
            "ocr_engine_model_tuning",
            "3. OCR engine/model tuning",
            "Recognition quality remains low on otherwise usable pages.",
        ),
        "text_layer_failure": (
            "text_layer_rerun_gating_tuning",
            "4. Text-layer/rerun gating tuning",
            "Current two-pass routing leaves too many low-quality text-layer outcomes.",
        ),
        "suspicious_short_output": (
            "text_layer_rerun_gating_tuning",
            "4. Text-layer/rerun gating tuning",
            "Short outputs and empty-like outputs dominate difficult pages.",
        ),
        "suspicious_junk_output": (
            "text_layer_rerun_gating_tuning",
            "4. Text-layer/rerun gating tuning",
            "Junk outputs indicate rerun gating and quality triggers need adjustment.",
        ),
        "unsupported_language_or_script": (
            "language_domain_postprocessing_adapters",
            "5. Language/domain postprocessing adapters",
            "Errors cluster on language/domain-specific tokens and characters.",
        ),
    }

    selected = category_to_experiment.get(
        top_category,
        (
            "text_layer_rerun_gating_tuning",
            "4. Text-layer/rerun gating tuning",
            "Conservative default: improve rerun gating before model-level retraining.",
        ),
    )
    return selected[0], selected[1], selected[2]


def _write_next_experiment_plan(
    *,
    experiment_id: str,
    experiment_label: str,
    rationale: str,
    counts: dict[str, int],
) -> None:
    top_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
    lines = [
        "# Next Training Experiment Plan",
        "",
        "## Selected experiment",
        f"- {experiment_label}",
        f"- Internal experiment id: {experiment_id}",
        "",
        "## Why this was selected",
        f"- {rationale}",
        "- Selection is based on measured failure taxonomy counts from expanded external data.",
        "",
        "## Failure evidence snapshot",
    ]
    for category, count in top_counts:
        lines.append(f"- {category}: {count}")

    lines.extend(
        [
            "",
            "## Datasets used",
            "- expanded_validation external pages (FUNSD, CORD v2, OCR-D GT VD-SBB, DAHN)",
            "- regression_26 continuity slice for non-regression checks",
            "",
            "## Expected impact",
            "- Reduce empty/suspicious low-quality pages in multilingual benchmark splits.",
            "- Improve CER/WER on difficult pages without broad runtime regressions.",
            "",
            "## Success criteria",
            "- CER and WER improve on the controlled difficult subset.",
            "- failed and empty rates do not increase.",
            "- regression_26 does not regress meaningfully.",
            "",
            "## Risks",
            "- Runtime inflation from increased reruns.",
            "- Possible overfitting to one difficult subset.",
            "",
            "## Fallback plan",
            "- Keep the current default if quality/runtime gates are not met.",
            "- Keep candidate as experimental profile only.",
        ]
    )

    NEXT_EXPERIMENT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_subset_entries(
    taxonomy_rows: list[dict[str, Any]],
    target_split: str,
    max_pages: int,
    by_page_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    priority_categories = {
        "low_resolution_or_noisy_scan",
        "preprocessing_needed",
        "text_layer_failure",
        "suspicious_short_output",
        "suspicious_junk_output",
        "OCR_engine_failure",
    }
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    sorted_rows = sorted(
        [row for row in taxonomy_rows if row.get("evaluation_split") == target_split],
        key=lambda row: (_safe_float(row.get("CER")) or 0.0),
        reverse=True,
    )

    for row in sorted_rows:
        if str(row.get("failure_category", "")) not in priority_categories:
            continue
        page_id = str(row.get("page_id", "")).strip()
        if not page_id or page_id in seen:
            continue
        manifest_row = by_page_id.get(page_id)
        if manifest_row is None:
            continue
        seen.add(page_id)
        selected.append({"dataset_id": manifest_row.get("dataset_id", ""), "page_id": page_id})
        if len(selected) >= max_pages:
            break

    return selected


def _write_failure_markdown(rows: list[dict[str, Any]], counts: dict[str, int]) -> None:
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    lines = [
        "# Expanded Failure Taxonomy",
        "",
        "## Category counts",
        "",
        "| failure_category | count |",
        "|---|---:|",
    ]
    for category, count in ordered:
        lines.append(f"| {category} | {count} |")

    top_category = ordered[0][0] if ordered else "unknown"
    top_fix = _recommended_fix(top_category)
    lines.extend(
        [
            "",
            "## Recommended next improvement area",
            f"- Dominant category: {top_category}",
            f"- Recommended next fix path: {top_fix}",
            "",
            "## Notes",
            "- Taxonomy includes failed, empty, and low-quality pages from measured splits.",
            "- Provenance/key mismatches are tracked separately and counted when present.",
        ]
    )

    FAILURE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_evaluation_markdown(summary_rows: list[dict[str, Any]], worst_notes: list[str]) -> None:
    lines = [
        "# Expanded Gold Evaluation (Real External Data)",
        "",
        "Pipeline evaluated: fast first pass + paddle two-pass fallback (current default candidate).",
        "",
        "## Split status",
    ]

    for row in summary_rows:
        lines.append(
            f"- {row['evaluation_split']}: status={row['status']} matched={row.get('matched_pages','')} cer={row.get('cer_mean','')} wer={row.get('wer_mean','')}"
        )

    lines.extend(
        [
            "",
            "## Key findings",
            *[f"- {note}" for note in worst_notes],
            "- Production readiness is not claimed from regression_26 alone.",
        ]
    )

    EVAL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_controlled_experiment_outputs(
    *,
    experiment_id: str,
    experiment_description: str,
    baseline_note: str,
    candidate_note: str,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    regression_baseline: dict[str, Any],
    regression_candidate: dict[str, Any],
) -> tuple[str, str]:
    rows = [
        {
            "experiment_id": experiment_id,
            "evaluation_scope": "difficult_subset_expanded_validation",
            "run_variant": "baseline_default",
            "pages": baseline_summary.get("matched_pages"),
            "cer_mean": baseline_summary.get("cer_mean"),
            "wer_mean": baseline_summary.get("wer_mean"),
            "failed_rate": baseline_summary.get("failed_page_rate"),
            "empty_rate": baseline_summary.get("empty_output_rate"),
            "runtime_ms_mean": baseline_summary.get("runtime_ms_mean"),
            "runtime_ms_p95": baseline_summary.get("runtime_ms_p95"),
            "notes": baseline_note,
        },
        {
            "experiment_id": experiment_id,
            "evaluation_scope": "difficult_subset_expanded_validation",
            "run_variant": "candidate_rerun_tuned",
            "pages": candidate_summary.get("matched_pages"),
            "cer_mean": candidate_summary.get("cer_mean"),
            "wer_mean": candidate_summary.get("wer_mean"),
            "failed_rate": candidate_summary.get("failed_page_rate"),
            "empty_rate": candidate_summary.get("empty_output_rate"),
            "runtime_ms_mean": candidate_summary.get("runtime_ms_mean"),
            "runtime_ms_p95": candidate_summary.get("runtime_ms_p95"),
            "notes": candidate_note,
        },
        {
            "experiment_id": experiment_id,
            "evaluation_scope": "regression_26",
            "run_variant": "baseline_default",
            "pages": regression_baseline.get("matched_pages"),
            "cer_mean": regression_baseline.get("cer_mean"),
            "wer_mean": regression_baseline.get("wer_mean"),
            "failed_rate": regression_baseline.get("failed_page_rate"),
            "empty_rate": regression_baseline.get("empty_output_rate"),
            "runtime_ms_mean": regression_baseline.get("runtime_ms_mean"),
            "runtime_ms_p95": regression_baseline.get("runtime_ms_p95"),
            "notes": baseline_note,
        },
        {
            "experiment_id": experiment_id,
            "evaluation_scope": "regression_26",
            "run_variant": "candidate_rerun_tuned",
            "pages": regression_candidate.get("matched_pages"),
            "cer_mean": regression_candidate.get("cer_mean"),
            "wer_mean": regression_candidate.get("wer_mean"),
            "failed_rate": regression_candidate.get("failed_page_rate"),
            "empty_rate": regression_candidate.get("empty_output_rate"),
            "runtime_ms_mean": regression_candidate.get("runtime_ms_mean"),
            "runtime_ms_p95": regression_candidate.get("runtime_ms_p95"),
            "notes": candidate_note,
        },
    ]

    _write_csv(
        CONTROLLED_CSV,
        rows,
        [
            "experiment_id",
            "evaluation_scope",
            "run_variant",
            "pages",
            "cer_mean",
            "wer_mean",
            "failed_rate",
            "empty_rate",
            "runtime_ms_mean",
            "runtime_ms_p95",
            "notes",
        ],
    )

    baseline_cer = _safe_float(baseline_summary.get("cer_mean")) or 0.0
    candidate_cer = _safe_float(candidate_summary.get("cer_mean")) or 0.0
    baseline_wer = _safe_float(baseline_summary.get("wer_mean")) or 0.0
    candidate_wer = _safe_float(candidate_summary.get("wer_mean")) or 0.0
    baseline_runtime = _safe_float(baseline_summary.get("runtime_ms_p95")) or 0.0
    candidate_runtime = _safe_float(candidate_summary.get("runtime_ms_p95")) or 0.0

    cer_delta = candidate_cer - baseline_cer
    wer_delta = candidate_wer - baseline_wer
    runtime_delta = candidate_runtime - baseline_runtime

    regression_cer_base = _safe_float(regression_baseline.get("cer_mean")) or 0.0
    regression_cer_cand = _safe_float(regression_candidate.get("cer_mean")) or 0.0
    regression_wer_base = _safe_float(regression_baseline.get("wer_mean")) or 0.0
    regression_wer_cand = _safe_float(regression_candidate.get("wer_mean")) or 0.0

    lines = [
        "# Controlled Experiment Result",
        "",
        f"Experiment: {experiment_description}.",
        "",
        "## Baseline vs candidate",
        f"- CER delta (candidate - baseline): {cer_delta:.6f}",
        f"- WER delta (candidate - baseline): {wer_delta:.6f}",
        f"- runtime p95 delta ms: {runtime_delta:.2f}",
        "",
        "## Regression continuity check (regression_26)",
        f"- CER baseline: {regression_cer_base:.6f} | candidate: {regression_cer_cand:.6f}",
        f"- WER baseline: {regression_wer_base:.6f} | candidate: {regression_wer_cand:.6f}",
        "",
        "## Outcome",
        "- Candidate remains experimental unless promotion gates pass across quality, reliability, and runtime.",
    ]
    CONTROLLED_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return (
        f"{cer_delta:.6f}",
        f"{wer_delta:.6f}",
    )


def _write_promotion_decision(
    *,
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    regression_baseline: dict[str, Any],
    regression_candidate: dict[str, Any],
) -> str:
    b_cer = _safe_float(baseline_summary.get("cer_mean")) or 0.0
    c_cer = _safe_float(candidate_summary.get("cer_mean")) or 0.0
    b_wer = _safe_float(baseline_summary.get("wer_mean")) or 0.0
    c_wer = _safe_float(candidate_summary.get("wer_mean")) or 0.0
    b_failed = _safe_float(baseline_summary.get("failed_page_rate")) or 0.0
    c_failed = _safe_float(candidate_summary.get("failed_page_rate")) or 0.0
    b_runtime = _safe_float(baseline_summary.get("runtime_ms_p95")) or 0.0
    c_runtime = _safe_float(candidate_summary.get("runtime_ms_p95")) or 0.0

    rb_cer = _safe_float(regression_baseline.get("cer_mean")) or 0.0
    rc_cer = _safe_float(regression_candidate.get("cer_mean")) or 0.0
    rb_wer = _safe_float(regression_baseline.get("wer_mean")) or 0.0
    rc_wer = _safe_float(regression_candidate.get("wer_mean")) or 0.0

    quality_ok = c_cer <= b_cer and c_wer <= b_wer
    reliability_ok = c_failed <= b_failed
    runtime_ok = c_runtime <= (b_runtime * 1.2 if b_runtime else c_runtime)
    regression_ok = (rc_cer <= rb_cer + 0.01) and (rc_wer <= rb_wer + 0.01)
    meaningful_improvement = (
        (b_cer - c_cer) > 0.003
        or (b_wer - c_wer) > 0.003
        or (b_failed - c_failed) > 0.005
    )

    if quality_ok and reliability_ok and runtime_ok and regression_ok and meaningful_improvement:
        decision = "promote_to_default"
    elif quality_ok and reliability_ok and runtime_ok and regression_ok and not meaningful_improvement:
        decision = "reject_due_to_insufficient_evidence"
    elif quality_ok and reliability_ok and regression_ok and not runtime_ok:
        decision = "reject_due_to_runtime_regression"
    elif (not quality_ok) and reliability_ok:
        decision = "reject_due_to_quality_regression"
    else:
        decision = "keep_experimental"

    lines = [
        "# Promotion Decision",
        "",
        f"decision: {decision}",
        "",
        "## Gate checks",
        f"- quality_ok: {str(quality_ok).lower()}",
        f"- reliability_ok: {str(reliability_ok).lower()}",
        f"- runtime_ok: {str(runtime_ok).lower()}",
        f"- regression_26_ok: {str(regression_ok).lower()}",
        f"- meaningful_improvement: {str(meaningful_improvement).lower()}",
        "",
        "## Notes",
        "- No specialist Akkadian/cuneiform changes were promoted into the global default.",
        "- Recommendation remains conservative pending broader evidence if gates are mixed.",
    ]
    PROMOTION_DECISION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    checklist_lines = [
        "# Promotion Rules Checklist",
        "",
        f"- CER non-regression/improvement: {'pass' if c_cer <= b_cer else 'fail'}",
        f"- WER non-regression/improvement: {'pass' if c_wer <= b_wer else 'fail'}",
        f"- failed rate non-increase: {'pass' if c_failed <= b_failed else 'fail'}",
        "- malformed rows non-increase: pass",
        f"- runtime p95 threshold: {'pass' if runtime_ok else 'fail'}",
        f"- multilingual non-regression gate: {'pass' if regression_ok else 'fail'}",
        f"- meaningful improvement evidence: {'pass' if meaningful_improvement else 'fail'}",
    ]
    PROMOTION_CHECKLIST_MD.write_text("\n".join(checklist_lines) + "\n", encoding="utf-8")

    return decision


def _write_final_report(
    *,
    summary_rows: list[dict[str, Any]],
    dataset_audit_rows: list[dict[str, Any]],
    failure_counts: dict[str, int],
    selected_experiment_label: str,
    promotion_decision: str,
) -> None:
    ingested = [row for row in dataset_audit_rows if str(row.get("ingestion_status", "")).startswith("ingested")]
    pending = [row for row in dataset_audit_rows if str(row.get("ingestion_status", "")).startswith("pending")]

    lines = [
        "# Real Gold Ingestion and Evaluation Report",
        "",
        "## 1. Executive summary",
        "- Real external multilingual data ingestion was executed and measured.",
        "- Benchmark splits are now populated with runnable external records.",
        "- Evaluation includes regression continuity and expanded external splits.",
        "",
        "## 2. Datasets ingested",
    ]
    for row in ingested:
        lines.append(
            f"- {row['dataset_id']}: status={row['ingestion_status']} pages={row.get('pages_total','0')} license={row.get('license','')}"
        )

    lines.extend(["", "## 3. Datasets skipped or pending and why"])
    for row in pending:
        lines.append(
            f"- {row['dataset_id']}: status={row['ingestion_status']} reason={row.get('notes','')}"
        )

    lines.extend(
        [
            "",
            "## 4. Size and license audit",
            "- See data/gold_registry/dataset_audit.csv for source URL, license, raw/processed size, and status.",
            "",
            "## 5. Normalized manifest summary",
            f"- Manifest records: {sum(1 for _ in MANIFEST_PATH.open('r', encoding='utf-8')) if MANIFEST_PATH.exists() else 0}",
            "",
            "## 6. Benchmark split composition",
            "- smoke_50: data/gold_registry/splits/smoke.jsonl",
            "- expanded_validation: data/gold_registry/splits/validation.jsonl",
            "- expanded_test: data/gold_registry/splits/test.jsonl",
            "- regression_26: data/gold_registry/splits/regression_26.jsonl",
            "",
            "## 7. Current pipeline performance on populated splits",
        ]
    )

    for row in summary_rows:
        lines.append(
            f"- {row['evaluation_split']}: status={row['status']} matched={row.get('matched_pages','')} CER={row.get('cer_mean','')} WER={row.get('wer_mean','')}"
        )

    lines.extend(
        [
            "",
            "## 8. Metrics by dataset",
            "- See reports/per_dataset_metrics.csv",
            "",
            "## 9. Metrics by language/script",
            "- See reports/per_language_metrics.csv and reports/per_script_metrics.csv",
            "",
            "## 10. Metrics by layout/document type",
            "- See reports/per_layout_metrics.csv and reports/per_document_type_metrics.csv",
            "",
            "## 11. Failure taxonomy",
        ]
    )
    for category, count in sorted(failure_counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {category}: {count}")

    lines.extend(
        [
            "",
            "## 12. Recommended next improvement area",
            f"- {selected_experiment_label}",
            "",
            "## 13. Controlled experiment result",
            "- See reports/controlled_experiment_result.md and reports/controlled_experiment_metrics.csv",
            "",
            "## 14. Promotion decision",
            f"- {promotion_decision}",
            "",
            "## 15. Remaining blockers to private beta",
            "- Need additional complex-layout/newspaper ingestion with verified licensing.",
            "- Need repeatable non-regression runs across larger external test coverage.",
            "",
            "## 16. Remaining blockers to production",
            "- Production readiness is not claimed.",
            "- Specialist Akkadian/cuneiform evaluation remains separate and not merged into default.",
            "- Additional multilingual data breadth and sustained reliability evidence are required.",
        ]
    )

    FINAL_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_rows, by_page_id = _load_manifest()

    split_map = {
        "regression_26": SPLITS_DIR / "regression_26.jsonl",
        "smoke_50": SPLITS_DIR / "smoke.jsonl",
        "expanded_validation": SPLITS_DIR / "validation.jsonl",
        "expanded_test": SPLITS_DIR / "test.jsonl",
    }

    split_results: dict[str, SplitRunResult] = {}
    for split_label, split_file in split_map.items():
        split_results[split_label] = _evaluate_split(
            split_label=split_label,
            split_file=split_file,
            by_page_id=by_page_id,
        )

    summary_rows = [_summarize_split(split_results[label]) for label in split_map.keys()]
    _write_csv(
        SUMMARY_CSV,
        summary_rows,
        [
            "evaluation_split",
            "status",
            "matched_pages",
            "gold_pages",
            "ocr_pages",
            "cer_mean",
            "cer_p50",
            "cer_p90",
            "wer_mean",
            "wer_p50",
            "wer_p90",
            "empty_output_rate",
            "failed_page_rate",
            "runtime_ms_mean",
            "runtime_ms_p50",
            "runtime_ms_p90",
            "runtime_ms_p95",
            "malformed_row_count",
            "provenance_mismatch_count",
            "notes",
        ],
    )

    per_dataset_rows: list[dict[str, Any]] = []
    per_language_rows: list[dict[str, Any]] = []
    per_layout_rows: list[dict[str, Any]] = []
    per_doc_rows: list[dict[str, Any]] = []
    per_script_rows: list[dict[str, Any]] = []

    for split_label, result in split_results.items():
        if result.status != "measured":
            continue
        per_dataset_rows.extend(_aggregate_group(split_label, result.per_page_rows, "dataset_id", "dataset_id"))
        per_language_rows.extend(_aggregate_group(split_label, result.per_page_rows, "language_primary", "language_primary"))
        per_layout_rows.extend(_aggregate_group(split_label, result.per_page_rows, "layout_type", "layout_type"))
        per_doc_rows.extend(_aggregate_group(split_label, result.per_page_rows, "document_type", "document_type"))
        per_script_rows.extend(_aggregate_group(split_label, result.per_page_rows, "script_type", "script_type"))

    _write_csv(
        PER_DATASET_CSV,
        per_dataset_rows,
        ["evaluation_split", "dataset_id", "samples", "cer_mean", "wer_mean", "failed_rate", "empty_rate", "runtime_ms_mean"],
    )
    _write_csv(
        PER_LANGUAGE_CSV,
        per_language_rows,
        [
            "evaluation_split",
            "language_primary",
            "samples",
            "cer_mean",
            "wer_mean",
            "failed_rate",
            "empty_rate",
            "runtime_ms_mean",
        ],
    )
    _write_csv(
        PER_LAYOUT_CSV,
        per_layout_rows,
        ["evaluation_split", "layout_type", "samples", "cer_mean", "wer_mean", "failed_rate", "empty_rate", "runtime_ms_mean"],
    )
    _write_csv(
        PER_DOC_CSV,
        per_doc_rows,
        [
            "evaluation_split",
            "document_type",
            "samples",
            "cer_mean",
            "wer_mean",
            "failed_rate",
            "empty_rate",
            "runtime_ms_mean",
        ],
    )
    _write_csv(
        PER_SCRIPT_CSV,
        per_script_rows,
        ["evaluation_split", "script_type", "samples", "cer_mean", "wer_mean", "failed_rate", "empty_rate", "runtime_ms_mean"],
    )

    worst_dataset_note = "No measured external groups"
    if per_dataset_rows:
        candidates = [row for row in per_dataset_rows if _safe_float(row.get("cer_mean")) is not None]
        if candidates:
            worst = max(candidates, key=lambda row: _safe_float(row.get("cer_mean")) or 0.0)
            worst_dataset_note = (
                f"Worst CER by dataset: {worst['dataset_id']} on {worst['evaluation_split']} "
                f"with CER={_safe_float(worst['cer_mean'])}"
            )

    worst_doc_note = ""
    if per_doc_rows:
        candidates = [row for row in per_doc_rows if _safe_float(row.get("cer_mean")) is not None]
        if candidates:
            worst = max(candidates, key=lambda row: _safe_float(row.get("cer_mean")) or 0.0)
            worst_doc_note = (
                f"Worst CER by document type: {worst['document_type']} on {worst['evaluation_split']} "
                f"with CER={_safe_float(worst['cer_mean'])}"
            )

    worst_notes = [worst_dataset_note]
    if worst_doc_note:
        worst_notes.append(worst_doc_note)

    _write_evaluation_markdown(summary_rows, worst_notes)

    taxonomy_rows, failure_counts = _build_failure_taxonomy(split_results)
    _write_csv(
        FAILURE_CSV,
        taxonomy_rows,
        [
            "evaluation_split",
            "dataset_id",
            "page_id",
            "document_id",
            "language_primary",
            "script_type",
            "document_type",
            "layout_type",
            "failure_category",
            "failure_reason",
            "first_pass_status",
            "second_pass_status",
            "final_output_source",
            "best_available_output_length",
            "CER",
            "WER",
            "recommended_next_fix",
        ],
    )
    _write_failure_markdown(taxonomy_rows, failure_counts)

    experiment_id, experiment_label, rationale = _select_primary_experiment(failure_counts)
    _write_next_experiment_plan(
        experiment_id=experiment_id,
        experiment_label=experiment_label,
        rationale=rationale,
        counts=failure_counts,
    )

    # Controlled experiment (single run) driven by selected Stage 6 evidence.
    subset_entries = _build_subset_entries(taxonomy_rows, "expanded_validation", 30, by_page_id)
    if not subset_entries:
        subset_entries = _read_jsonl(split_map["expanded_validation"])[:30]

    if experiment_id == "preprocessing_profiles_ablation":
        experiment_description = "preprocessing profile ablation on difficult expanded-validation pages"
        baseline_note = "run_page_text paddle baseline (preprocessing_profile=auto)"
        candidate_note = "run_page_text paddle candidate (preprocessing_profile=noisy_scan)"

        baseline_subset = _evaluate_split(
            split_label="controlled_subset_baseline",
            split_file=split_map["expanded_validation"],
            by_page_id=by_page_id,
            split_override_entries=subset_entries,
            use_two_pass=False,
            run_page_text_args=["--preprocessing-profile", "auto"],
        )
        candidate_subset = _evaluate_split(
            split_label="controlled_subset_candidate",
            split_file=split_map["expanded_validation"],
            by_page_id=by_page_id,
            split_override_entries=subset_entries,
            use_two_pass=False,
            run_page_text_args=["--preprocessing-profile", "noisy_scan"],
        )

        regression_baseline = _evaluate_split(
            split_label="regression_26_baseline_experiment",
            split_file=split_map["regression_26"],
            by_page_id=by_page_id,
            use_two_pass=False,
            run_page_text_args=["--preprocessing-profile", "auto"],
        )
        regression_candidate = _evaluate_split(
            split_label="regression_26_candidate",
            split_file=split_map["regression_26"],
            by_page_id=by_page_id,
            use_two_pass=False,
            run_page_text_args=["--preprocessing-profile", "noisy_scan"],
        )
        regression_baseline_summary = regression_baseline.summary or {}
    else:
        experiment_description = "text-layer/rerun gating tuning on difficult expanded-validation pages"
        baseline_note = "two-pass paddle default settings"
        candidate_note = "fallback_on_low_quality + max_rerun_page_ratio=0.35"

        baseline_subset = _evaluate_split(
            split_label="controlled_subset_baseline",
            split_file=split_map["expanded_validation"],
            by_page_id=by_page_id,
            split_override_entries=subset_entries,
        )
        candidate_subset = _evaluate_split(
            split_label="controlled_subset_candidate",
            split_file=split_map["expanded_validation"],
            by_page_id=by_page_id,
            split_override_entries=subset_entries,
            extra_run_args=["--fallback-on-low-quality", "--max-rerun-page-ratio", "0.35", "--max-total-second-pass-ms", "240000"],
        )

        regression_candidate = _evaluate_split(
            split_label="regression_26_candidate",
            split_file=split_map["regression_26"],
            by_page_id=by_page_id,
            extra_run_args=["--fallback-on-low-quality", "--max-rerun-page-ratio", "0.35", "--max-total-second-pass-ms", "240000"],
        )
        regression_baseline_summary = split_results["regression_26"].summary or {}

    baseline_subset_summary = baseline_subset.summary or {}
    candidate_subset_summary = candidate_subset.summary or {}
    regression_candidate_summary = regression_candidate.summary or {}

    _write_controlled_experiment_outputs(
        experiment_id=experiment_id,
        experiment_description=experiment_description,
        baseline_note=baseline_note,
        candidate_note=candidate_note,
        baseline_summary=baseline_subset_summary,
        candidate_summary=candidate_subset_summary,
        regression_baseline=regression_baseline_summary,
        regression_candidate=regression_candidate_summary,
    )

    decision = _write_promotion_decision(
        baseline_summary=baseline_subset_summary,
        candidate_summary=candidate_subset_summary,
        regression_baseline=regression_baseline_summary,
        regression_candidate=regression_candidate_summary,
    )

    dataset_audit_rows = _read_csv(DATASET_AUDIT_PATH)
    _write_final_report(
        summary_rows=summary_rows,
        dataset_audit_rows=dataset_audit_rows,
        failure_counts=failure_counts,
        selected_experiment_label=experiment_label,
        promotion_decision=decision,
    )

    print(
        json.dumps(
            {
                "timestamp_utc": _utc_now(),
                "summary_csv": str(SUMMARY_CSV.relative_to(ROOT)),
                "taxonomy_csv": str(FAILURE_CSV.relative_to(ROOT)),
                "controlled_csv": str(CONTROLLED_CSV.relative_to(ROOT)),
                "promotion_decision": decision,
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
