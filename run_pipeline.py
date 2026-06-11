#!/usr/bin/env python3
"""Compatibility entrypoint for OCR pipeline execution.

This script preserves the long-standing `run_pipeline.py` command while routing
execution to the maintained page-text pipeline at `tools/run_page_text.py`.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RUN_PAGE_TEXT_PATH = ROOT / "tools" / "run_page_text.py"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_default_output_root(config: dict[str, Any]) -> str:
    return (
        config.get("output", {}).get("output_directory")
        or config.get("paths", {}).get("output_dir")
        or "reports/output"
    )


def _resolve_default_inputs(config: dict[str, Any]) -> str:
    return (
        config.get("input", {}).get("input_directory")
        or config.get("paths", {}).get("input_dir")
        or "data/input_pdfs"
    )


def _resolve_engine(config: dict[str, Any], explicit_engine: str | None) -> str:
    if explicit_engine:
        return explicit_engine

    configured = config.get("ocr", {}).get("engine") if isinstance(config, dict) else None
    if configured:
        return str(configured)

    return "ensemble"


def _resolve_force_ocr(config: dict[str, Any], explicit_force_ocr: bool) -> bool:
    if explicit_force_ocr:
        return True

    configured = config.get("ocr", {}).get("force_ocr_on_text_layer") if isinstance(config, dict) else None
    return bool(configured)


def _validate_only(config_path: Path | None, input_dir: str | None, input_file: str | None) -> int:
    errors: list[str] = []

    if not RUN_PAGE_TEXT_PATH.exists():
        errors.append(f"Missing runner: {RUN_PAGE_TEXT_PATH}")

    if config_path is not None and not config_path.exists():
        errors.append(f"Config file not found: {config_path}")

    if input_dir and not Path(input_dir).exists():
        errors.append(f"Input directory not found: {input_dir}")

    if input_file and not Path(input_file).exists():
        errors.append(f"Input file not found: {input_file}")

    if errors:
        print("Validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("Validation passed.")
    if config_path is not None:
        print(f"Config: {config_path}")
    print(f"Runner: {RUN_PAGE_TEXT_PATH}")
    return 0


def _build_manifest_for_single_pdf(pdf_path: Path) -> str:
    pages = [1]
    try:
        import fitz  # type: ignore

        with fitz.open(str(pdf_path)) as doc:
            page_count = len(doc)
        pages = list(range(1, max(page_count, 1) + 1))
    except Exception:
        # Fall back to first page when page counting dependencies are unavailable.
        pages = [1]

    fd, manifest_path = tempfile.mkstemp(prefix="ocr_single_pdf_", suffix=".tsv")
    os.close(fd)
    manifest_file = Path(manifest_path)
    with manifest_file.open("w", encoding="utf-8") as fh:
        for page in pages:
            fh.write(f"{pdf_path}\t{page}\n")
    return manifest_path


PASS_PROVENANCE_FIELDS = [
    "pass_number",
    "first_pass_status",
    "second_pass_status",
    "final_status",
    "final_output_source",
    "final_selection_reason",
    "fallback_reason",
    "fallback_rejected_reason",
    "fallback_engine",
    "first_pass_quality_score",
    "second_pass_quality_score",
    "fallback_improved_quality_estimate",
    "first_pass_runtime_ms",
    "second_pass_runtime_ms",
    "total_page_runtime_ms",
]


OPTIMIZATION_AUDIT_FIELDS = [
    "optimization_enabled",
    "optimization_name",
    "cache_hit",
    "cache_miss",
    "skipped_stage_reason",
    "worker_count",
    "timeout_policy",
    "performance_trace",
]


DEFAULT_MAX_RERUN_PAGE_RATIO = 0.20
DEFAULT_MAX_TOTAL_SECOND_PASS_MS = 120000
DEFAULT_MAX_SECOND_PASS_MS_PER_PAGE = 30000


CANDIDATE_CLASS_PRIORITY = {
    "critical_failed": 90,
    "empty_output": 85,
    "malformed_metadata": 75,
    "suspicious_junk_text": 65,
    "suspicious_short_text": 55,
    "low_confidence": 45,
    "adapter_suspicious": 35,
    "normal": 10,
}


def _candidate_reason_severity(reason: str) -> int:
    token = str(reason or "").strip().lower()
    if token == "empty_output":
        return 100
    if token == "malformed_metadata":
        return 95
    if token == "status:failed":
        return 95
    if token == "status:timed_out":
        return 90
    if token == "status:partial_success":
        return 80
    if token == "render_failure":
        return 80
    if token == "text_layer_rejected":
        return 75
    if token == "text_layer_quality_low":
        return 70
    if token == "broken_unicode":
        return 70
    if token == "mostly_symbols":
        return 65
    if token == "quality_class:failed_ocr":
        return 65
    if token == "quality_class:weak_ocr":
        return 50
    if token == "low_confidence":
        return 35
    if token == "adapter_suspicious":
        return 35
    if token == "low_text_length":
        return 30
    if token == "text_shorter_than_text_layer":
        return 30
    if token == "low_character_diversity":
        return 25
    if token == "repeated_character_run":
        return 25
    if token == "mostly_punctuation_or_symbols":
        return 25
    return 10


def _severity_for_candidate(candidate: dict[str, Any]) -> int:
    reasons = candidate.get("reasons") or []
    if not isinstance(reasons, list):
        return 0
    score = sum(_candidate_reason_severity(str(reason)) for reason in reasons)
    candidate_class = str(candidate.get("rerun_candidate_class", "normal") or "normal").strip().lower()
    score += int(CANDIDATE_CLASS_PRIORITY.get(candidate_class, CANDIDATE_CLASS_PRIORITY["normal"]))
    if "empty_output" in reasons and any(str(reason).startswith("status:") for reason in reasons):
        score += 20
    return int(score)


def _candidate_class_for_reasons(reasons: list[str]) -> str:
    tokens = {str(reason or "").strip().lower() for reason in reasons}
    if any(token in {"status:failed", "status:timed_out", "status:partial_success", "render_failure"} for token in tokens):
        return "critical_failed"
    if "empty_output" in tokens:
        return "empty_output"
    if "malformed_metadata" in tokens:
        return "malformed_metadata"
    if any(
        token
        in {
            "broken_unicode",
            "repeated_character_run",
            "mostly_punctuation_or_symbols",
            "mostly_symbols",
            "text_layer_rejected",
            "text_layer_quality_low",
        }
        for token in tokens
    ):
        return "suspicious_junk_text"
    if any(token in {"low_text_length", "text_shorter_than_text_layer"} for token in tokens):
        return "suspicious_short_text"
    if "low_confidence" in tokens:
        return "low_confidence"
    if "adapter_suspicious" in tokens:
        return "adapter_suspicious"
    return "normal"


def _resolve_second_pass_engine_mode(mode: str) -> tuple[str, str]:
    requested = str(mode or "auto").strip().lower()
    if requested == "paddle":
        return "paddle", "paddle"
    if requested == "ensemble":
        return "ensemble", "ensemble"
    if requested in {"kraken", "doctr"}:
        # run_page_text exposes backend-specific routing through ensemble mode.
        return "ensemble", requested
    return "paddle", "auto"


def _slugify_identifier(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return slug.strip("_") or "item"


def _chunk_candidates(candidates: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    size = max(1, int(chunk_size or 1))
    return [candidates[index : index + size] for index in range(0, len(candidates), size)]


def _apply_rerun_budget_limits(
    candidates: list[dict[str, Any]],
    *,
    total_pages: int,
    max_rerun_pages: int | None,
    max_rerun_page_ratio: float | None,
    fallback_budget_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    ranked = []
    for item in candidates:
        enriched = dict(item)
        candidate_score = _safe_int(enriched.get("rerun_candidate_score"))
        if candidate_score is None:
            candidate_score = _severity_for_candidate(enriched)
        enriched["rerun_candidate_score"] = int(candidate_score)
        enriched["severity_score"] = int(candidate_score)
        ranked.append(enriched)

    ranked.sort(
        key=lambda item: (
            -int(item.get("rerun_candidate_score", 0)),
            -int(CANDIDATE_CLASS_PRIORITY.get(str(item.get("rerun_candidate_class", "normal") or "normal"), 0)),
            str(item.get("pdf_name", "")),
            int(item.get("page", 0)),
        )
    )

    for index, item in enumerate(ranked, start=1):
        item["rerun_candidate_rank"] = index

    cap = len(ranked)
    if max_rerun_pages is not None and max_rerun_pages > 0:
        cap = min(cap, int(max_rerun_pages))

    if max_rerun_page_ratio is not None and max_rerun_page_ratio > 0:
        ratio_cap = int(math.floor(float(total_pages) * float(max_rerun_page_ratio)))
        ratio_cap = max(1 if ranked else 0, ratio_cap)
        cap = min(cap, ratio_cap)

    if str(fallback_budget_mode).strip().lower() == "permissive" and ranked and cap <= 0:
        cap = 1

    selected = ranked[:cap]
    skipped = ranked[cap:]
    meta = {
        "rerun_candidates_total": len(ranked),
        "rerun_attempted": len(selected),
        "rerun_skipped_budget": len(skipped),
        "rerun_budget_cap": cap,
    }
    return selected, skipped, meta


def _extract_page_row(rows: list[dict[str, str]], key: tuple[str, int]) -> dict[str, str] | None:
    for row in rows:
        row_key = _row_key(row)
        if row_key == key:
            return row
    return rows[0] if rows else None


def _run_budgeted_second_pass(
    *,
    candidates: list[dict[str, Any]],
    output_root: Path,
    ocr_fallback_mode: str,
    engine_mode_label: str,
    profile: str | None,
    status_bar: bool,
    max_second_pass_ms_per_page: int,
    max_total_second_pass_ms: int,
    fallback_budget_mode: str,
    enable_backend_warm_reuse: bool = False,
    backend_warm_batch_size: int = 8,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    pass2_rows: list[dict[str, str]] = []
    pass2_progress_rows: list[dict[str, str]] = []
    timed_out_keys: set[tuple[str, int]] = set()
    budget_skipped_keys: set[tuple[str, int]] = set()

    pass2_pages_dir = output_root / "pass2_fallback_pages"
    pass2_pages_dir.mkdir(parents=True, exist_ok=True)

    strict_mode = str(fallback_budget_mode).strip().lower() == "strict"
    total_second_pass_ms = 0
    second_pass_budget_exhausted = False

    warm_batch_size = max(1, int(backend_warm_batch_size or 1)) if enable_backend_warm_reuse else 1
    candidate_chunks = _chunk_candidates(candidates, warm_batch_size)

    optimization_enabled = bool(enable_backend_warm_reuse and warm_batch_size > 1 and len(candidates) > 1)
    if not enable_backend_warm_reuse:
        optimization_skipped_reason = "optimization_flag_disabled"
    elif warm_batch_size <= 1:
        optimization_skipped_reason = "batch_size_leq_1"
    elif len(candidates) <= 1:
        optimization_skipped_reason = "insufficient_rerun_candidates"
    else:
        optimization_skipped_reason = ""

    second_pass_invocation_count = 0
    second_pass_invocation_wall_ms: list[int] = []

    for batch_index, chunk in enumerate(candidate_chunks):
        if max_total_second_pass_ms > 0 and total_second_pass_ms >= max_total_second_pass_ms and strict_mode:
            second_pass_budget_exhausted = True
            for remaining in candidate_chunks[batch_index:]:
                for pending in remaining:
                    budget_skipped_keys.add((str(pending.get("pdf_name", "")), int(pending.get("page", 0))))
            break

        if len(chunk) == 1:
            candidate = chunk[0]
            slug = _slugify_identifier(f"{candidate.get('pdf_name', '')}_{candidate.get('page', '')}_{batch_index}")
        else:
            first_name = str(chunk[0].get("pdf_name", "") or "batch")
            slug = _slugify_identifier(f"batch_{batch_index}_{first_name}")

        page_dir = pass2_pages_dir / slug
        page_dir.mkdir(parents=True, exist_ok=True)

        fd, manifest_path = tempfile.mkstemp(prefix=f"ocr_rerun_{slug}_", suffix=".tsv")
        os.close(fd)
        manifest_file = Path(manifest_path)
        manifest_lines = [f"{item.get('pdf_path', '')}\t{int(item.get('page', 1))}" for item in chunk]
        manifest_file.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

        page_args = [
            "--manifest",
            str(manifest_file),
            "--output-root",
            str(page_dir),
            "--prefer-text-layer",
            "--ocr-fallback",
            ocr_fallback_mode,
            "--force-ocr",
        ]
        if max_second_pass_ms_per_page > 0:
            page_args.extend(["--page-timeout-ms", str(max_second_pass_ms_per_page)])
        if status_bar:
            page_args.append("--status-bar")
        if profile:
            page_args.extend(["--profile", profile])

        started = time.perf_counter()
        try:
            _call_run_page_text(page_args)
        finally:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            total_second_pass_ms += elapsed_ms
            second_pass_invocation_count += 1
            second_pass_invocation_wall_ms.append(elapsed_ms)
            manifest_file.unlink(missing_ok=True)

        page_rows = _read_csv_rows(page_dir / "client_page_text.csv", encoding="utf-8-sig")
        page_rows_by_key: dict[tuple[str, int], dict[str, str]] = {}
        for row in page_rows:
            row_key = _row_key(row)
            if row_key is not None and row_key not in page_rows_by_key:
                page_rows_by_key[row_key] = row

        page_progress_rows = _read_csv_rows(page_dir / "progress.csv")
        page_progress_by_key: dict[tuple[str, int], dict[str, str]] = {}
        for row in page_progress_rows:
            row_key = _row_key(row)
            if row_key is not None and row_key not in page_progress_by_key:
                page_progress_by_key[row_key] = row

        for candidate in chunk:
            key = (str(candidate.get("pdf_name", "")), int(candidate.get("page", 0)))
            chosen_row = page_rows_by_key.get(key)
            if chosen_row is None:
                chosen_row = {
                    "pdf_name": key[0],
                    "page": str(key[1]),
                    "status": "failed",
                    "failure_reason": "missing_second_pass_output",
                    "page_text": "",
                    "extraction_method": "failed",
                    "runtime_ms": str(elapsed_ms),
                }

            if _safe_float(chosen_row.get("runtime_ms")) is None:
                chosen_row["runtime_ms"] = str(elapsed_ms)

            chosen_row["second_pass_engine_mode"] = engine_mode_label
            chosen_row["second_pass_engine_resolved"] = ocr_fallback_mode
            chosen_row["second_pass_wall_ms"] = str(elapsed_ms)

            status = str(chosen_row.get("status", "") or "").strip().lower()
            failure_reason = str(chosen_row.get("failure_reason", "") or "").strip().lower()
            if status == "timed_out" or "timeout" in failure_reason:
                timed_out_keys.add(key)

            pass2_rows.append(chosen_row)

            page_progress_row = page_progress_by_key.get(key)
            if page_progress_row is not None:
                pass2_progress_rows.append(page_progress_row)

        if max_total_second_pass_ms > 0 and total_second_pass_ms >= max_total_second_pass_ms and strict_mode:
            second_pass_budget_exhausted = True
            for remaining in candidate_chunks[batch_index + 1 :]:
                for pending in remaining:
                    budget_skipped_keys.add((str(pending.get("pdf_name", "")), int(pending.get("page", 0))))
            break

    stats = {
        "rerun_attempted": len(pass2_rows),
        "rerun_timed_out": len(timed_out_keys),
        "second_pass_budget_exhausted": second_pass_budget_exhausted,
        "second_pass_total_runtime_ms": total_second_pass_ms,
        "second_pass_invocation_count": second_pass_invocation_count,
        "second_pass_invocation_wall_ms": second_pass_invocation_wall_ms,
        "estimated_backend_initialization_count": second_pass_invocation_count,
        "optimization_enabled": optimization_enabled,
        "optimization_name": "backend_model_warm_reuse" if optimization_enabled else "",
        "optimization_skipped_reason": optimization_skipped_reason,
        "worker_count": 1,
        "timeout_policy": str(fallback_budget_mode or "strict"),
        "timed_out_keys": timed_out_keys,
        "budget_skipped_keys": budget_skipped_keys,
    }
    return pass2_rows, pass2_progress_rows, stats


def _write_second_pass_artifacts(pass2_dir: Path, rows: list[dict[str, str]], progress_rows: list[dict[str, str]]) -> None:
    pass2_dir.mkdir(parents=True, exist_ok=True)
    pass2_csv = pass2_dir / "client_page_text.csv"
    pass2_json = pass2_dir / "client_page_text.json"
    pass2_progress = pass2_dir / "progress.csv"

    _write_csv_rows(
        pass2_csv,
        [dict(row) for row in rows],
        [
            "pdf_name",
            "page",
            "status",
            "failure_reason",
            "extraction_method",
            "runtime_ms",
            "page_text",
            "second_pass_engine_mode",
            "second_pass_engine_resolved",
            "second_pass_wall_ms",
        ],
        encoding="utf-8-sig",
    )
    pass2_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if not progress_rows:
        progress_rows = [
            {
                "pdf_name": row.get("pdf_name", ""),
                "page": row.get("page", ""),
                "status": row.get("status", ""),
                "failure_reason": row.get("failure_reason", ""),
                "extraction_method": row.get("extraction_method", ""),
                "ms": row.get("runtime_ms", ""),
                "second_pass_engine_mode": row.get("second_pass_engine_mode", ""),
                "second_pass_engine_resolved": row.get("second_pass_engine_resolved", ""),
            }
            for row in rows
        ]

    _write_csv_rows(
        pass2_progress,
        [dict(row) for row in progress_rows],
        [
            "pdf_name",
            "page",
            "status",
            "failure_reason",
            "extraction_method",
            "ms",
            "second_pass_engine_mode",
            "second_pass_engine_resolved",
        ],
        encoding="utf-8",
    )


def _estimate_junk_score(text: str, low_text_threshold: int) -> float:
    stripped = (text or "").strip()
    if not stripped:
        return 100.0

    reasons = _is_suspicious_text(stripped, low_text_threshold=low_text_threshold)
    score = 0.0
    weights = {
        "empty_output": 100.0,
        "low_text_length": 25.0,
        "low_character_diversity": 20.0,
        "repeated_character_run": 20.0,
        "mostly_punctuation_or_symbols": 20.0,
        "broken_unicode": 30.0,
    }
    for reason in reasons:
        score += float(weights.get(reason, 10.0))
    return score


def _selection_score(row: dict[str, Any], low_text_threshold: int) -> float:
    status = str(row.get("status", "") or "").strip().lower()
    text = str(row.get("page_text", "") or "")
    confidence = _safe_float(row.get("confidence"))
    quality_score = _safe_float(row.get("page_quality_score"))
    if quality_score is None:
        quality_score = _safe_float(row.get("quality_score"))

    score = float(_status_rank(status) * 10)
    if text.strip():
        score += 1.0
    score += min(len(text.strip()) / 500.0, 1.0)
    if confidence is not None:
        score += max(0.0, min(confidence, 1.0))
    if quality_score is not None:
        score += max(0.0, min(quality_score, 1.0))
    score -= min(_estimate_junk_score(text, low_text_threshold=low_text_threshold) / 50.0, 4.0)
    return score


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
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
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t", "on"}


def _status_rank(status: str) -> int:
    lowered = str(status or "").strip().lower()
    if lowered == "success":
        return 4
    if lowered == "partial_success":
        return 3
    if lowered == "timed_out":
        return 2
    if lowered == "failed":
        return 1
    return 0


def _row_key(row: dict[str, Any]) -> tuple[str, int] | None:
    pdf_name = str(row.get("pdf_name", "") or "").strip()
    page = _safe_int(row.get("page"))
    if not pdf_name or page is None:
        return None
    return (pdf_name, page)


def _read_csv_rows(path: Path, encoding: str = "utf-8") -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding=encoding, newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv_rows(path: Path, rows: list[dict[str, Any]], preferred_first: list[str], encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    observed: set[str] = set()
    for row in rows:
        observed.update(row.keys())

    fieldnames: list[str] = []
    for key in preferred_first:
        if key in observed:
            fieldnames.append(key)
            observed.remove(key)
    fieldnames.extend(sorted(observed))

    with path.open("w", encoding=encoding, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _resolve_pdf_path_for_row(row: dict[str, str], input_dir: str | None) -> Path | None:
    candidates = [
        str(row.get("input_pdf_path", "") or "").strip(),
        str(row.get("input_file", "") or "").strip(),
        str(row.get("pdf_name", "") or "").strip(),
    ]

    for raw in candidates:
        if not raw:
            continue
        candidate = Path(raw)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if not candidate.is_absolute():
            root_candidate = (ROOT / candidate).resolve()
            if root_candidate.exists():
                return root_candidate
            if input_dir:
                input_candidate = (Path(input_dir) / candidate.name).resolve()
                if input_candidate.exists():
                    return input_candidate
    return None


def _is_suspicious_text(text: str, low_text_threshold: int) -> list[str]:
    reasons: list[str] = []
    stripped = (text or "").strip()
    if not stripped:
        return ["empty_output"]

    if len(stripped) < max(1, int(low_text_threshold)):
        reasons.append("low_text_length")

    unique_ratio = len(set(stripped)) / float(max(len(stripped), 1))
    if unique_ratio < 0.08:
        reasons.append("low_character_diversity")

    if re.search(r"(.)\1{9,}", stripped):
        reasons.append("repeated_character_run")

    punctuation_or_symbols = sum(1 for ch in stripped if (not ch.isalnum()) and (not ch.isspace()))
    punct_ratio = punctuation_or_symbols / float(max(len(stripped), 1))
    if punct_ratio > 0.7:
        reasons.append("mostly_punctuation_or_symbols")

    replacement_chars = stripped.count("\ufffd")
    if replacement_chars > 0:
        reasons.append("broken_unicode")

    return reasons


def _collect_rerun_candidates(
    rows: list[dict[str, str]],
    *,
    input_dir: str | None,
    rerun_failed_pages: bool,
    fallback_on_empty: bool,
    fallback_on_low_quality: bool,
    low_text_threshold: int,
    low_confidence_threshold: float,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[str]]]:
    candidates: list[dict[str, Any]] = []
    reason_map: dict[tuple[str, int], list[str]] = {}

    for row in rows:
        key = _row_key(row)
        if key is None:
            continue

        reasons: list[str] = []
        status = str(row.get("status", "") or "").strip().lower()
        text = str(row.get("page_text", "") or "")
        failure_reason = str(row.get("failure_reason", "") or "").strip().lower()

        if rerun_failed_pages and status in {"failed", "timed_out", "partial_success"}:
            reasons.append(f"status:{status}")

        page_rendered = _safe_bool(row.get("page_image_rendered", True))
        render_failure_reason = str(row.get("render_failure_reason", "") or "").strip()
        if (not page_rendered) or render_failure_reason:
            reasons.append("render_failure")

        if fallback_on_empty and not text.strip():
            reasons.append("empty_output")

        if fallback_on_low_quality:
            reasons.extend(_is_suspicious_text(text, low_text_threshold=low_text_threshold))

            text_layer_char_count = _safe_int(row.get("text_layer_char_count"))
            extracted_length = len(text.strip())
            if text_layer_char_count is not None and text_layer_char_count >= (low_text_threshold * 2):
                if extracted_length < max(4, int(text_layer_char_count * 0.30)):
                    reasons.append("text_shorter_than_text_layer")

            text_layer_accepted = _safe_bool(row.get("text_layer_accepted"))
            text_layer_rejected_reason = str(row.get("text_layer_rejected_reason", "") or "").strip().lower()
            text_layer_quality_score = _safe_float(row.get("text_layer_quality_score"))
            if not text_layer_accepted and text_layer_rejected_reason:
                reasons.append("text_layer_rejected")
            if text_layer_quality_score is not None and text_layer_quality_score < 0.55:
                reasons.append("text_layer_quality_low")

            quality_class = str(row.get("quality_class", "") or "").strip().lower()
            if quality_class in {"failed_ocr", "weak_ocr"}:
                reasons.append(f"quality_class:{quality_class}")

            confidence = _safe_float(row.get("confidence"))
            if confidence is not None and confidence < float(low_confidence_threshold):
                reasons.append("low_confidence")

            unknown_token_rate = _safe_float(row.get("unknown_token_rate"))
            adapter_used = str(row.get("adapter_used", "") or "").strip().lower()
            quality_reasons = str(row.get("quality_reasons", "") or "").strip().lower()
            if (
                (unknown_token_rate is not None and unknown_token_rate >= 0.45)
                or "unknown_token" in quality_reasons
                or (adapter_used and "fallback" in adapter_used)
            ):
                reasons.append("adapter_suspicious")

        missing_metadata_tokens: list[str] = []
        if not str(row.get("pdf_name", "") or "").strip():
            missing_metadata_tokens.append("pdf_name")
        if _safe_int(row.get("page")) is None:
            missing_metadata_tokens.append("page")
        if not str(row.get("document_id", "") or "").strip():
            missing_metadata_tokens.append("document_id")
        if not str(row.get("page_id", "") or "").strip():
            missing_metadata_tokens.append("page_id")
        if missing_metadata_tokens and (fallback_on_low_quality or reasons):
            reasons.append("malformed_metadata")

        if "timeout" in failure_reason and "status:timed_out" not in reasons:
            reasons.append("status:timed_out")

        if not reasons:
            continue

        pdf_path = _resolve_pdf_path_for_row(row, input_dir=input_dir)
        if pdf_path is None:
            continue

        page = _safe_int(row.get("page"))
        if page is None:
            continue

        dedup_reasons = sorted(set(reasons))
        candidate_class = _candidate_class_for_reasons(dedup_reasons)
        severity_score = _severity_for_candidate(
            {
                "reasons": dedup_reasons,
                "rerun_candidate_class": candidate_class,
            }
        )
        candidates.append(
            {
                "pdf_path": str(pdf_path),
                "pdf_name": key[0],
                "page": page,
                "page_key": f"{key[0]}::{page}",
                "reasons": dedup_reasons,
                "severity_score": severity_score,
                "rerun_candidate_class": candidate_class,
                "rerun_candidate_score": severity_score,
                "rerun_candidate_reason": "|".join(dedup_reasons),
                "rerun_candidate_rank": "",
                "first_pass_status": status,
                "first_pass_text_length": len(text.strip()),
            }
        )
        reason_map[key] = dedup_reasons

    candidates.sort(
        key=lambda item: (
            -int(item.get("rerun_candidate_score", 0)),
            -int(CANDIDATE_CLASS_PRIORITY.get(str(item.get("rerun_candidate_class", "normal") or "normal"), 0)),
            str(item.get("pdf_name", "")),
            int(item.get("page", 0)),
        )
    )
    for index, item in enumerate(candidates, start=1):
        item["rerun_candidate_rank"] = index
    return candidates, reason_map


def _write_rerun_manifest(manifest_path: Path, candidates: list[dict[str, Any]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        for item in candidates:
            fh.write(f"{item['pdf_path']}\t{int(item['page'])}\n")


def _merge_two_pass_rows(
    pass1_rows: list[dict[str, str]],
    pass2_rows: list[dict[str, str]],
    rerun_reasons: dict[tuple[str, int], list[str]],
    fallback_engine: str,
    *,
    second_pass_engine_mode: str = "paddle",
    second_pass_timeout_action: str = "keep_first_pass",
    low_text_threshold: int = 24,
    budget_skipped_keys: set[tuple[str, int]] | None = None,
    timed_out_keys: set[tuple[str, int]] | None = None,
    second_pass_budget_exhausted: bool = False,
    optimization_enabled: bool = False,
    optimization_name: str = "",
    optimization_skipped_reason: str = "",
    optimization_worker_count: int = 1,
    optimization_timeout_policy: str = "strict",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    budget_skipped = budget_skipped_keys or set()
    timed_out = timed_out_keys or set()
    allowed_keys = set(rerun_reasons.keys())
    pass2_by_key: dict[tuple[str, int], dict[str, str]] = {}
    for row in pass2_rows:
        key = _row_key(row)
        if key is not None and key in allowed_keys and key not in pass2_by_key:
            pass2_by_key[key] = row

    merged: list[dict[str, Any]] = []
    stats = {
        "pages_total": 0,
        "first_pass_success_pages": 0,
        "fallback_candidate_pages": 0,
        "fallback_rerun_pages": 0,
        "fallback_selected_pages": 0,
        "still_failing_after_fallback": 0,
        "rerun_candidates_total": len(allowed_keys),
        "rerun_attempted": 0,
        "rerun_skipped_budget": len(budget_skipped),
        "rerun_timed_out": len(timed_out & allowed_keys),
        "second_pass_budget_exhausted": bool(second_pass_budget_exhausted),
        "fallback_selection_count": 0,
        "second_pass_engine_mode": second_pass_engine_mode,
    }

    for first_row in pass1_rows:
        key = _row_key(first_row)
        if key is None:
            continue
        stats["pages_total"] += 1

        first_status = str(first_row.get("status", "") or "").strip().lower()
        if first_status == "success":
            stats["first_pass_success_pages"] += 1

        reasons = rerun_reasons.get(key, [])
        if reasons:
            stats["fallback_candidate_pages"] += 1

        second_row = pass2_by_key.get(key)
        if second_row is not None:
            stats["fallback_rerun_pages"] += 1
            stats["rerun_attempted"] += 1

        first_text = str(first_row.get("page_text", "") or "")
        first_runtime = _safe_float(first_row.get("runtime_ms"))
        first_score = _selection_score(first_row, low_text_threshold=low_text_threshold)
        first_suspicious = _is_suspicious_text(first_text, low_text_threshold=low_text_threshold)

        chosen = dict(first_row)
        second_status = ""
        second_runtime = None
        second_score: float | None = None
        final_output_source = "first_pass_fast"
        final_selection_reason = "first_pass_default"
        fallback_rejected_reason = ""
        choose_second = False
        force_mark_failed = False

        if second_row is not None:
            second_status = str(second_row.get("status", "") or "").strip().lower()
            second_text = str(second_row.get("page_text", "") or "")
            second_runtime = _safe_float(second_row.get("runtime_ms"))
            second_failure_reason = str(second_row.get("failure_reason", "") or "").strip().lower()
            second_timed_out = second_status == "timed_out" or key in timed_out or "timeout" in second_failure_reason
            second_suspicious = _is_suspicious_text(second_text, low_text_threshold=low_text_threshold)
            second_score = _selection_score(second_row, low_text_threshold=low_text_threshold)
            severe_suspicious = any(
                reason in {"broken_unicode", "repeated_character_run", "mostly_punctuation_or_symbols", "empty_output"}
                for reason in second_suspicious
            )
            first_failed_or_empty = first_status in {"failed", "timed_out", "partial_success"} or not first_text.strip()
            second_failed_or_empty = second_status in {"failed", "timed_out"} or not second_text.strip()

            if second_timed_out and second_pass_timeout_action == "keep_first_pass":
                final_output_source = "first_pass_kept_due_to_timeout"
                final_selection_reason = "second_pass_timeout_keep_first_pass"
                fallback_rejected_reason = "second_pass_timeout"
            elif second_timed_out and second_pass_timeout_action == "mark_failed":
                force_mark_failed = True
                final_output_source = "first_pass_kept_due_to_timeout"
                final_selection_reason = "second_pass_timeout_mark_failed"
                fallback_rejected_reason = "second_pass_timeout"
            elif second_timed_out and second_pass_timeout_action == "partial_result" and second_text.strip():
                choose_second = True
                final_selection_reason = "second_pass_timeout_partial_result"
            elif second_failed_or_empty:
                final_output_source = "first_pass_kept_due_to_failed_fallback"
                final_selection_reason = "second_pass_failed_or_empty"
                fallback_rejected_reason = "second_pass_failed_or_empty"
            elif severe_suspicious and not first_failed_or_empty:
                final_output_source = "first_pass_kept_due_to_suspicious_fallback"
                final_selection_reason = "suspicious_second_pass_output"
                fallback_rejected_reason = "suspicious_second_pass_output"
            else:
                if first_failed_or_empty and second_text.strip() and _status_rank(second_status) >= _status_rank("partial_success"):
                    choose_second = True
                    final_selection_reason = "first_pass_failed_second_pass_nonempty"
                elif _status_rank(second_status) > _status_rank(first_status):
                    choose_second = True
                    final_selection_reason = "better_second_pass_status"
                elif (
                    first_suspicious
                    and second_text.strip()
                    and not severe_suspicious
                    and second_score is not None
                    and second_score >= (first_score - 0.05)
                ):
                    choose_second = True
                    final_selection_reason = "first_pass_suspicious_second_pass_reasonable"
                elif second_score is not None and second_score > first_score + 0.35 and second_text.strip():
                    choose_second = True
                    final_selection_reason = "better_second_pass_quality_score"
                else:
                    final_selection_reason = "second_pass_not_better"
                    fallback_rejected_reason = "quality_not_improved"

            if choose_second:
                chosen = dict(second_row)
                normalized_engine = "ensemble" if str(fallback_engine).strip().lower().startswith("ensemble") else "paddle"
                final_output_source = f"second_pass_{normalized_engine}"
                fallback_rejected_reason = ""
                stats["fallback_selected_pages"] += 1
                stats["fallback_selection_count"] += 1

        if key in budget_skipped:
            final_output_source = "first_pass_kept_due_to_budget"
            final_selection_reason = "second_pass_skipped_due_to_budget"
            fallback_rejected_reason = "second_pass_skipped_due_to_budget"

        if second_row is None and key in allowed_keys and key not in budget_skipped:
            final_selection_reason = "second_pass_not_available"
            fallback_rejected_reason = "second_pass_not_available"

        if force_mark_failed:
            chosen = dict(first_row)
            chosen["status"] = "failed"
            chosen["failure_reason"] = "second_pass_timeout_mark_failed"
            chosen["page_text"] = ""

        chosen_status = str(chosen.get("status", "") or "").strip().lower()
        if chosen_status in {"failed", "timed_out"} or not str(chosen.get("page_text", "") or "").strip():
            stats["still_failing_after_fallback"] += 1

        chosen["pass_number"] = "2" if final_output_source.startswith("second_pass_") else "1"
        chosen["first_pass_status"] = first_status or "unknown"
        if key in budget_skipped:
            chosen["second_pass_status"] = "skipped_budget"
        else:
            chosen["second_pass_status"] = second_status or ("not_rerun" if not reasons else "not_improved")
        chosen["final_status"] = chosen_status or "unknown"
        chosen["final_output_source"] = final_output_source
        chosen["final_selection_reason"] = final_selection_reason
        chosen["fallback_reason"] = "|".join(reasons)
        chosen["fallback_rejected_reason"] = fallback_rejected_reason
        chosen["fallback_engine"] = fallback_engine if reasons else ""
        chosen["first_pass_quality_score"] = f"{first_score:.4f}".rstrip("0").rstrip(".")
        if second_score is None:
            chosen["second_pass_quality_score"] = ""
            chosen["fallback_improved_quality_estimate"] = ""
        else:
            chosen["second_pass_quality_score"] = f"{second_score:.4f}".rstrip("0").rstrip(".")
            delta = second_score - first_score
            chosen["fallback_improved_quality_estimate"] = f"{delta:.4f}".rstrip("0").rstrip(".")
        chosen["first_pass_runtime_ms"] = "" if first_runtime is None else f"{first_runtime:.3f}".rstrip("0").rstrip(".")
        chosen["second_pass_runtime_ms"] = "" if second_runtime is None else f"{second_runtime:.3f}".rstrip("0").rstrip(".")

        if first_runtime is None and second_runtime is None:
            chosen["total_page_runtime_ms"] = ""
        else:
            total_runtime = (first_runtime or 0.0) + (second_runtime or 0.0)
            chosen["total_page_runtime_ms"] = f"{total_runtime:.3f}".rstrip("0").rstrip(".")

        chosen["engine_used"] = str(chosen.get("extraction_method", "") or chosen.get("engine_used", "")).strip()
        chosen["output_text_length"] = str(len(str(chosen.get("page_text", "") or "")))

        chosen["optimization_enabled"] = "true" if optimization_enabled else "false"
        chosen["optimization_name"] = str(optimization_name or "")
        chosen["cache_hit"] = ""
        chosen["cache_miss"] = ""
        chosen["skipped_stage_reason"] = "" if optimization_enabled else str(optimization_skipped_reason or "")
        chosen["worker_count"] = str(max(1, int(optimization_worker_count or 1)))
        chosen["timeout_policy"] = str(optimization_timeout_policy or "strict")
        trace_payload = {
            "first_pass_runtime_ms": first_runtime,
            "second_pass_runtime_ms": second_runtime,
            "total_page_runtime_ms": chosen.get("total_page_runtime_ms", ""),
            "final_output_source": final_output_source,
            "final_selection_reason": final_selection_reason,
            "fallback_reason": "|".join(reasons),
            "optimization_enabled": bool(optimization_enabled),
            "optimization_name": str(optimization_name or ""),
            "timeout_policy": str(optimization_timeout_policy or "strict"),
        }
        chosen["performance_trace"] = json.dumps(trace_payload, ensure_ascii=True, sort_keys=True)

        merged.append(chosen)

    merged.sort(key=lambda item: (str(item.get("pdf_name", "")), _safe_int(item.get("page")) or 0))
    return merged, stats


def _write_two_pass_report(
    path: Path,
    *,
    stats: dict[str, Any],
    candidates: list[dict[str, Any]],
    max_rerun_pages: int,
    max_rerun_page_ratio: float,
    max_second_pass_ms_per_page: int,
    max_total_second_pass_ms: int,
    second_pass_engine_mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(),
        "max_rerun_pages": max_rerun_pages,
        "max_rerun_page_ratio": max_rerun_page_ratio,
        "max_second_pass_ms_per_page": max_second_pass_ms_per_page,
        "max_total_second_pass_ms": max_total_second_pass_ms,
        "second_pass_engine_mode": second_pass_engine_mode,
        "rerun_candidates_total": stats.get("rerun_candidates_total", 0),
        "rerun_attempted": stats.get("rerun_attempted", 0),
        "rerun_skipped_budget": stats.get("rerun_skipped_budget", 0),
        "rerun_timed_out": stats.get("rerun_timed_out", 0),
        "second_pass_budget_exhausted": bool(stats.get("second_pass_budget_exhausted", False)),
        "fallback_selection_count": stats.get("fallback_selection_count", 0),
        "stats": stats,
        "fallback_candidates": candidates,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _build_run_page_text_args(args: argparse.Namespace, cfg: dict[str, Any]) -> tuple[list[str], list[Path]]:
    temp_paths: list[Path] = []
    output_root = args.output_dir or _resolve_default_output_root(cfg)
    selected_engine = _resolve_engine(cfg, args.engine)
    force_ocr = _resolve_force_ocr(cfg, args.force_ocr)

    if args.input_file:
        input_path = Path(args.input_file).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        manifest_path = _build_manifest_for_single_pdf(input_path)
        temp_paths.append(Path(manifest_path))
        run_args = [
            "--manifest",
            manifest_path,
            "--output-root",
            output_root,
            "--prefer-text-layer",
        ]
    else:
        inputs_dir = args.input_dir or _resolve_default_inputs(cfg)
        run_args = [
            "--inputs",
            inputs_dir,
            "--output-root",
            output_root,
            "--prefer-text-layer",
        ]

    selected_engine_lower = str(selected_engine).lower()
    if selected_engine_lower.startswith("paddle"):
        run_args.extend(["--ocr-fallback", "paddle"])
    elif selected_engine_lower.startswith("ensemble") or selected_engine_lower.startswith("multi"):
        run_args.extend(["--ocr-fallback", "ensemble"])

    if force_ocr:
        run_args.append("--force-ocr")

    if args.status_bar:
        run_args.append("--status-bar")

    if args.profile:
        run_args.extend(["--profile", args.profile])

    if args.progress_csv:
        run_args.extend(["--progress-csv", args.progress_csv])

    return run_args, temp_paths


def _call_run_page_text(mapped_args: list[str]) -> int:
    spec = importlib.util.spec_from_file_location("_compat_run_page_text", RUN_PAGE_TEXT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner module from {RUN_PAGE_TEXT_PATH}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    main_fn = getattr(mod, "main", None)
    if main_fn is None:
        raise AttributeError("tools/run_page_text.py does not define main()")

    original_argv = sys.argv[:]
    try:
        sys.argv = [str(RUN_PAGE_TEXT_PATH)] + mapped_args
        try:
            result = main_fn()
        except SystemExit as exc:
            code = exc.code
            if code is None:
                return 0
            if isinstance(code, int):
                return int(code)
            return 1
    finally:
        sys.argv = original_argv

    return int(result) if isinstance(result, int) else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "OCR pipeline compatibility entrypoint. Supports legacy run_pipeline flags "
            "and routes execution to tools/run_page_text.py."
        )
    )
    parser.add_argument("-c", "--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--input-dir", help="Input directory containing PDFs")
    parser.add_argument("--input-file", help="Single PDF file path")
    parser.add_argument("--output-dir", help="Output root directory")
    parser.add_argument("--engine", help="OCR engine name (compatibility flag)")
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR even when text layer is present")
    parser.add_argument("--profile", help="Akkadian detection profile JSON")
    parser.add_argument("--progress-csv", help="Optional progress CSV output path")
    parser.add_argument("--status-bar", action="store_true", help="Display progress bar")
    parser.add_argument("--two-pass-mode", action="store_true", help="Enable two-pass adaptive OCR flow")
    parser.add_argument(
        "--rerun-failed-pages",
        action="store_true",
        default=False,
        help="In two-pass mode, rerun pages with failed/timed_out/partial_success status",
    )
    parser.add_argument(
        "--fallback-on-empty",
        action="store_true",
        default=False,
        help="In two-pass mode, rerun pages with empty extracted text",
    )
    parser.add_argument(
        "--fallback-on-low-quality",
        action="store_true",
        default=False,
        help="In two-pass mode, rerun pages with suspicious low-quality fast output",
    )
    parser.add_argument(
        "--low-text-threshold",
        type=int,
        default=24,
        help="Low-quality trigger threshold for minimum text length in two-pass mode",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.15,
        help="Low-quality trigger threshold for confidence in two-pass mode",
    )
    parser.add_argument(
        "--first-pass-ocr-fallback",
        choices=["none", "paddle", "ensemble"],
        default="none",
        help="OCR fallback engine for pass 1 in two-pass mode (default: none)",
    )
    parser.add_argument(
        "--second-pass-ocr-fallback",
        choices=["paddle", "ensemble"],
        default="ensemble",
        help="OCR fallback engine for pass 2 in two-pass mode",
    )
    parser.add_argument(
        "--max-rerun-pages",
        type=int,
        default=0,
        help="Hard cap on number of pages eligible for pass-2 rerun (0 uses ratio/default policy)",
    )
    parser.add_argument(
        "--max-rerun-page-ratio",
        type=float,
        default=DEFAULT_MAX_RERUN_PAGE_RATIO,
        help="Hard cap on rerun pages as a fraction of total pages",
    )
    parser.add_argument(
        "--max-second-pass-ms-per-page",
        type=int,
        default=DEFAULT_MAX_SECOND_PASS_MS_PER_PAGE,
        help="Per-page runtime budget for pass 2 in milliseconds",
    )
    parser.add_argument(
        "--max-total-second-pass-ms",
        type=int,
        default=DEFAULT_MAX_TOTAL_SECOND_PASS_MS,
        help="Total runtime budget for pass 2 in milliseconds",
    )
    parser.add_argument(
        "--second-pass-engine-mode",
        choices=["paddle", "ensemble", "kraken", "doctr", "auto"],
        default="auto",
        help="Requested second-pass engine mode (kraken/doctr route through ensemble fallback)",
    )
    parser.add_argument(
        "--second-pass-timeout-action",
        choices=["keep_first_pass", "mark_failed", "partial_result"],
        default="keep_first_pass",
        help="Merge behavior when second-pass rerun times out",
    )
    parser.add_argument(
        "--fallback-budget-mode",
        choices=["strict", "permissive"],
        default="strict",
        help="How rerun caps are enforced when budget limits are reached",
    )
    parser.add_argument(
        "--enable-backend-warm-reuse",
        action="store_true",
        default=False,
        help="Enable second-pass backend/model warm reuse by batching rerun pages",
    )
    parser.add_argument(
        "--backend-warm-batch-size",
        type=int,
        default=8,
        help="Batch size for warm-reuse rerun calls when --enable-backend-warm-reuse is enabled",
    )
    parser.add_argument("--validate-only", action="store_true", help="Validate environment and config")
    parser.add_argument("--dry-run", action="store_true", help="Print mapped command without executing")
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Extra flags forwarded directly to tools/run_page_text.py",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg_path = Path(args.config) if args.config else None
    cfg: dict[str, Any] = {}
    if cfg_path is not None and cfg_path.exists():
        cfg = _load_json(cfg_path)

    if args.validate_only:
        return _validate_only(cfg_path if cfg_path and args.config else None, args.input_dir, args.input_file)

    if args.two_pass_mode:
        two_pass_started = time.perf_counter()
        output_root = Path(args.output_dir or _resolve_default_output_root(cfg))
        output_root.mkdir(parents=True, exist_ok=True)
        pass1_dir = output_root / "pass1_fast"
        pass2_dir = output_root / "pass2_fallback"

        pass1_stage_ms = 0
        candidate_resolution_stage_ms = 0
        budget_selection_stage_ms = 0
        second_pass_stage_ms = 0
        merge_stage_ms = 0
        artifact_write_stage_ms = 0

        if args.input_file:
            input_path = Path(args.input_file).resolve()
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            manifest_path = _build_manifest_for_single_pdf(input_path)
            temp_paths: list[Path] = [Path(manifest_path)]
            base_input_args = ["--manifest", manifest_path]
            input_dir_for_resolution: str | None = str(input_path.parent)
        else:
            inputs_dir = args.input_dir or _resolve_default_inputs(cfg)
            temp_paths = []
            base_input_args = ["--inputs", inputs_dir]
            input_dir_for_resolution = inputs_dir

        pass1_args = [
            *base_input_args,
            "--output-root",
            str(pass1_dir),
            "--prefer-text-layer",
        ]
        if args.first_pass_ocr_fallback != "none":
            pass1_args.extend(["--ocr-fallback", args.first_pass_ocr_fallback])
            if args.force_ocr:
                pass1_args.append("--force-ocr")

        if args.status_bar:
            pass1_args.append("--status-bar")
        if args.profile:
            pass1_args.extend(["--profile", args.profile])

        if args.dry_run:
            requested_engine_mode = str(args.second_pass_engine_mode or "auto").strip().lower()
            if requested_engine_mode == "auto":
                requested_engine_mode = str(args.second_pass_ocr_fallback or "paddle").strip().lower()
            resolved_second_pass_fallback, _ = _resolve_second_pass_engine_mode(requested_engine_mode)
            print("Two-pass mapped commands:")
            print("pass1: python tools/run_page_text.py " + " ".join(pass1_args))
            print(
                "pass2: python tools/run_page_text.py --manifest <rerun_manifest.tsv> "
                f"--output-root {pass2_dir} --prefer-text-layer --ocr-fallback {resolved_second_pass_fallback} --force-ocr "
                f"--page-timeout-ms {max(0, int(args.max_second_pass_ms_per_page))}"
            )
            print(
                "budgets: "
                f"max_rerun_pages={int(args.max_rerun_pages)} "
                f"max_rerun_page_ratio={float(args.max_rerun_page_ratio)} "
                f"max_total_second_pass_ms={int(args.max_total_second_pass_ms)} "
                f"fallback_budget_mode={args.fallback_budget_mode} "
                f"enable_backend_warm_reuse={bool(args.enable_backend_warm_reuse)} "
                f"backend_warm_batch_size={int(args.backend_warm_batch_size)}"
            )
            return 0

        try:
            pass1_started = time.perf_counter()
            pass1_exit = _call_run_page_text(pass1_args)
            pass1_stage_ms = int((time.perf_counter() - pass1_started) * 1000)
            if pass1_exit != 0:
                return pass1_exit

            pass1_csv = pass1_dir / "client_page_text.csv"
            pass1_rows = _read_csv_rows(pass1_csv, encoding="utf-8-sig")
            if not pass1_rows:
                print(f"No first-pass rows found at {pass1_csv}")
                return 1

            rerun_failed_pages = bool(args.rerun_failed_pages)
            fallback_on_empty = bool(args.fallback_on_empty)
            fallback_on_low_quality = bool(args.fallback_on_low_quality)
            if not (rerun_failed_pages or fallback_on_empty or fallback_on_low_quality):
                rerun_failed_pages = True
                fallback_on_empty = True

            max_rerun_pages = max(0, int(args.max_rerun_pages or 0))
            max_rerun_page_ratio = max(0.0, float(args.max_rerun_page_ratio or 0.0))
            max_second_pass_ms_per_page = max(0, int(args.max_second_pass_ms_per_page or 0))
            max_total_second_pass_ms = max(0, int(args.max_total_second_pass_ms or 0))
            fallback_budget_mode = str(args.fallback_budget_mode or "strict").strip().lower()

            requested_engine_mode = str(args.second_pass_engine_mode or "auto").strip().lower()
            if requested_engine_mode == "auto":
                requested_engine_mode = str(args.second_pass_ocr_fallback or "paddle").strip().lower()
            resolved_second_pass_fallback, engine_mode_label = _resolve_second_pass_engine_mode(requested_engine_mode)

            candidate_resolution_started = time.perf_counter()
            candidates, reason_map = _collect_rerun_candidates(
                pass1_rows,
                input_dir=input_dir_for_resolution,
                rerun_failed_pages=rerun_failed_pages,
                fallback_on_empty=fallback_on_empty,
                fallback_on_low_quality=fallback_on_low_quality,
                low_text_threshold=max(1, int(args.low_text_threshold)),
                low_confidence_threshold=float(args.low_confidence_threshold),
            )
            candidate_resolution_stage_ms = int((time.perf_counter() - candidate_resolution_started) * 1000)

            budget_selection_started = time.perf_counter()
            selected_candidates, skipped_candidates, rerun_budget_meta = _apply_rerun_budget_limits(
                candidates,
                total_pages=len(pass1_rows),
                max_rerun_pages=max_rerun_pages if max_rerun_pages > 0 else None,
                max_rerun_page_ratio=max_rerun_page_ratio if max_rerun_page_ratio > 0 else None,
                fallback_budget_mode=fallback_budget_mode,
            )
            budget_selection_stage_ms = int((time.perf_counter() - budget_selection_started) * 1000)

            selected_keys: set[tuple[str, int]] = {
                (str(item.get("pdf_name", "")), int(item.get("page", 0)))
                for item in selected_candidates
            }
            skipped_keys: set[tuple[str, int]] = {
                (str(item.get("pdf_name", "")), int(item.get("page", 0)))
                for item in skipped_candidates
            }

            budget_skipped_keys: set[tuple[str, int]] = {
                (str(item.get("pdf_name", "")), int(item.get("page", 0)))
                for item in skipped_candidates
            }

            rerun_manifest = output_root / "rerun_manifest.tsv"
            candidates_csv = output_root / "fallback_candidates.csv"
            if selected_candidates:
                _write_rerun_manifest(rerun_manifest, selected_candidates)
            _write_csv_rows(
                candidates_csv,
                [
                    {
                        "pdf_path": item["pdf_path"],
                        "pdf_name": item["pdf_name"],
                        "page": item["page"],
                        "severity_score": item.get("severity_score", 0),
                        "rerun_candidate_score": item.get("rerun_candidate_score", item.get("severity_score", 0)),
                        "rerun_candidate_class": item.get("rerun_candidate_class", "normal"),
                        "rerun_candidate_reason": item.get("rerun_candidate_reason", ""),
                        "rerun_candidate_rank": item.get("rerun_candidate_rank", ""),
                        "selected_for_rerun": (
                            "true" if (str(item.get("pdf_name", "")), int(item.get("page", 0))) in selected_keys else "false"
                        ),
                        "skipped_due_to_budget": (
                            "true" if (str(item.get("pdf_name", "")), int(item.get("page", 0))) in skipped_keys else "false"
                        ),
                        "reasons": "|".join(item["reasons"]),
                    }
                    for item in candidates
                ],
                [
                    "pdf_path",
                    "pdf_name",
                    "page",
                    "rerun_candidate_score",
                    "rerun_candidate_class",
                    "rerun_candidate_reason",
                    "rerun_candidate_rank",
                    "severity_score",
                    "selected_for_rerun",
                    "skipped_due_to_budget",
                    "reasons",
                ],
                encoding="utf-8",
            )

            pass2_rows: list[dict[str, str]] = []
            pass2_progress_rows: list[dict[str, str]] = []
            timed_out_keys: set[tuple[str, int]] = set()
            second_pass_budget_exhausted = False
            second_pass_total_runtime_ms = 0
            pass2_stats: dict[str, Any] = {
                "timed_out_keys": set(),
                "budget_skipped_keys": set(),
                "second_pass_budget_exhausted": False,
                "second_pass_total_runtime_ms": 0,
                "second_pass_invocation_count": 0,
                "second_pass_invocation_wall_ms": [],
                "estimated_backend_initialization_count": 0,
                "optimization_enabled": False,
                "optimization_name": "",
                "optimization_skipped_reason": "no_rerun_candidates",
                "worker_count": 1,
                "timeout_policy": fallback_budget_mode,
            }

            if selected_candidates:
                second_pass_started = time.perf_counter()
                pass2_rows, pass2_progress_rows, pass2_stats = _run_budgeted_second_pass(
                    candidates=selected_candidates,
                    output_root=output_root,
                    ocr_fallback_mode=resolved_second_pass_fallback,
                    engine_mode_label=engine_mode_label,
                    profile=args.profile,
                    status_bar=bool(args.status_bar),
                    max_second_pass_ms_per_page=max_second_pass_ms_per_page,
                    max_total_second_pass_ms=max_total_second_pass_ms,
                    fallback_budget_mode=fallback_budget_mode,
                    enable_backend_warm_reuse=bool(args.enable_backend_warm_reuse),
                    backend_warm_batch_size=max(1, int(args.backend_warm_batch_size or 1)),
                )
                second_pass_stage_ms = int((time.perf_counter() - second_pass_started) * 1000)
                timed_out_keys = set(pass2_stats.get("timed_out_keys", set()))
                budget_skipped_keys.update(set(pass2_stats.get("budget_skipped_keys", set())))
                second_pass_budget_exhausted = bool(pass2_stats.get("second_pass_budget_exhausted", False))
                second_pass_total_runtime_ms = int(pass2_stats.get("second_pass_total_runtime_ms", 0))

            _write_second_pass_artifacts(pass2_dir, pass2_rows, pass2_progress_rows)

            merge_started = time.perf_counter()
            merged_rows, merge_stats = _merge_two_pass_rows(
                pass1_rows,
                pass2_rows,
                reason_map,
                fallback_engine=resolved_second_pass_fallback,
                second_pass_engine_mode=requested_engine_mode,
                second_pass_timeout_action=str(args.second_pass_timeout_action),
                low_text_threshold=max(1, int(args.low_text_threshold)),
                budget_skipped_keys=budget_skipped_keys,
                timed_out_keys=timed_out_keys,
                second_pass_budget_exhausted=second_pass_budget_exhausted,
                optimization_enabled=bool(pass2_stats.get("optimization_enabled", False)),
                optimization_name=str(pass2_stats.get("optimization_name", "") or ""),
                optimization_skipped_reason=str(pass2_stats.get("optimization_skipped_reason", "") or ""),
                optimization_worker_count=int(pass2_stats.get("worker_count", 1) or 1),
                optimization_timeout_policy=str(pass2_stats.get("timeout_policy", fallback_budget_mode) or fallback_budget_mode),
            )
            merge_stage_ms = int((time.perf_counter() - merge_started) * 1000)
            merge_stats["rerun_candidates_total"] = int(rerun_budget_meta.get("rerun_candidates_total", len(candidates)))
            merge_stats["rerun_attempted"] = len(pass2_rows)
            merge_stats["rerun_skipped_budget"] = len(budget_skipped_keys)
            merge_stats["rerun_timed_out"] = len(timed_out_keys)
            merge_stats["second_pass_budget_exhausted"] = second_pass_budget_exhausted
            merge_stats["fallback_selection_count"] = int(merge_stats.get("fallback_selection_count", 0))
            merge_stats["second_pass_engine_mode"] = requested_engine_mode
            merge_stats["second_pass_engine_resolved"] = resolved_second_pass_fallback
            merge_stats["second_pass_total_runtime_ms"] = second_pass_total_runtime_ms
            merge_stats["second_pass_invocation_count"] = int(pass2_stats.get("second_pass_invocation_count", 0))
            merge_stats["second_pass_invocation_wall_ms"] = list(pass2_stats.get("second_pass_invocation_wall_ms", []))
            merge_stats["estimated_backend_initialization_count"] = int(pass2_stats.get("estimated_backend_initialization_count", 0))
            merge_stats["optimization_enabled"] = bool(pass2_stats.get("optimization_enabled", False))
            merge_stats["optimization_name"] = str(pass2_stats.get("optimization_name", "") or "")
            merge_stats["optimization_skipped_reason"] = str(pass2_stats.get("optimization_skipped_reason", "") or "")
            merge_stats["worker_count"] = int(pass2_stats.get("worker_count", 1) or 1)
            merge_stats["timeout_policy"] = str(pass2_stats.get("timeout_policy", fallback_budget_mode) or fallback_budget_mode)
            merge_stats["requested_enable_backend_warm_reuse"] = bool(args.enable_backend_warm_reuse)
            merge_stats["requested_backend_warm_batch_size"] = max(1, int(args.backend_warm_batch_size or 1))
            merge_stats["pass1_stage_runtime_ms"] = pass1_stage_ms
            merge_stats["candidate_resolution_runtime_ms"] = candidate_resolution_stage_ms
            merge_stats["budget_selection_runtime_ms"] = budget_selection_stage_ms
            merge_stats["second_pass_stage_runtime_ms"] = second_pass_stage_ms
            merge_stats["merge_stage_runtime_ms"] = merge_stage_ms

            merged_csv = output_root / "client_page_text.csv"
            merged_json = output_root / "client_page_text.json"
            merged_progress = output_root / "progress.csv"
            report_json = output_root / "two_pass_report.json"

            artifact_write_started = time.perf_counter()
            _write_csv_rows(
                merged_csv,
                merged_rows,
                [
                    "pdf_name",
                    "page",
                    "status",
                    "failure_reason",
                    "extraction_method",
                    "engine_used",
                    "runtime_ms",
                    "output_text_length",
                    *PASS_PROVENANCE_FIELDS,
                    *OPTIMIZATION_AUDIT_FIELDS,
                    "page_text",
                ],
                encoding="utf-8-sig",
            )
            merged_json.write_text(json.dumps(merged_rows, ensure_ascii=False, indent=2), encoding="utf-8")

            progress_rows = []
            for row in merged_rows:
                progress_rows.append(
                    {
                        "pdf_name": row.get("pdf_name", ""),
                        "page": row.get("page", ""),
                        "ms": row.get("total_page_runtime_ms", row.get("runtime_ms", "")),
                        "used_text_layer": str(row.get("extraction_method", "")).strip() == "text_layer",
                        "has_akkadian": row.get("has_akkadian", "false"),
                        "status": row.get("status", ""),
                        "failure_reason": row.get("failure_reason", ""),
                        "extraction_method": row.get("extraction_method", ""),
                        "engine_statuses": row.get("engine_statuses", "{}"),
                        "pass_number": row.get("pass_number", ""),
                        "first_pass_status": row.get("first_pass_status", ""),
                        "second_pass_status": row.get("second_pass_status", ""),
                        "final_status": row.get("final_status", ""),
                        "final_output_source": row.get("final_output_source", ""),
                        "final_selection_reason": row.get("final_selection_reason", ""),
                        "fallback_reason": row.get("fallback_reason", ""),
                        "fallback_rejected_reason": row.get("fallback_rejected_reason", ""),
                        "fallback_engine": row.get("fallback_engine", ""),
                        "first_pass_quality_score": row.get("first_pass_quality_score", ""),
                        "second_pass_quality_score": row.get("second_pass_quality_score", ""),
                        "fallback_improved_quality_estimate": row.get("fallback_improved_quality_estimate", ""),
                        "first_pass_runtime_ms": row.get("first_pass_runtime_ms", ""),
                        "second_pass_runtime_ms": row.get("second_pass_runtime_ms", ""),
                        "total_page_runtime_ms": row.get("total_page_runtime_ms", ""),
                        "optimization_enabled": row.get("optimization_enabled", ""),
                        "optimization_name": row.get("optimization_name", ""),
                        "cache_hit": row.get("cache_hit", ""),
                        "cache_miss": row.get("cache_miss", ""),
                        "skipped_stage_reason": row.get("skipped_stage_reason", ""),
                        "worker_count": row.get("worker_count", ""),
                        "timeout_policy": row.get("timeout_policy", ""),
                        "performance_trace": row.get("performance_trace", ""),
                    }
                )

            _write_csv_rows(
                merged_progress,
                progress_rows,
                [
                    "pdf_name",
                    "page",
                    "ms",
                    "status",
                    "failure_reason",
                    "extraction_method",
                    *PASS_PROVENANCE_FIELDS,
                    *OPTIMIZATION_AUDIT_FIELDS,
                ],
                encoding="utf-8",
            )

            artifact_write_stage_ms = int((time.perf_counter() - artifact_write_started) * 1000)
            total_two_pass_runtime_ms = int((time.perf_counter() - two_pass_started) * 1000)
            merge_stats["artifact_write_runtime_ms"] = artifact_write_stage_ms
            merge_stats["total_two_pass_runtime_ms"] = total_two_pass_runtime_ms
            _write_two_pass_report(
                report_json,
                stats=merge_stats,
                candidates=candidates,
                max_rerun_pages=max_rerun_pages,
                max_rerun_page_ratio=max_rerun_page_ratio,
                max_second_pass_ms_per_page=max_second_pass_ms_per_page,
                max_total_second_pass_ms=max_total_second_pass_ms,
                second_pass_engine_mode=requested_engine_mode,
            )

            print("Two-pass workflow completed.")
            print(f"pass1_output={pass1_dir}")
            print(f"pass2_output={pass2_dir}")
            print(f"merged_output={merged_csv}")
            print(f"two_pass_report={report_json}")
            return 0
        finally:
            for tmp_path in temp_paths:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    mapped_args, temp_paths = _build_run_page_text_args(args, cfg)
    if args.passthrough:
        mapped_args.extend(args.passthrough)

    if args.dry_run:
        print("Mapped command:")
        print("python tools/run_page_text.py " + " ".join(mapped_args))
        return 0

    try:
        return _call_run_page_text(mapped_args)
    finally:
        for tmp_path in temp_paths:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
