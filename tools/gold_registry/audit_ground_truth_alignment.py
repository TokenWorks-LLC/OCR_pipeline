#!/usr/bin/env python3
from __future__ import annotations

import csv
import difflib
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from production.ensemble_ocr import FortifiedOCREnsemble
from production.preprocessing_profiles import (
    PREPROCESSING_PROFILES,
    PROFILE_AUTO,
    PROFILE_NOISY_SCAN,
    PreprocessingProfile,
)
from tools.gold_registry.converters.local_gold_converter import (
    LOCAL_GOLD_SUPPORTED_FORMATS,
    build_source_page_key,
    canonical_lookup_key,
    extract_local_gold_text,
)

REGISTRY_DIR = ROOT / "data" / "gold_registry"
SPLITS_DIR = REGISTRY_DIR / "splits"
MANIFEST_PATH = REGISTRY_DIR / "gold_manifest.jsonl"
LOCAL_GOLD_CSV_PATH = REGISTRY_DIR / "local_gold_for_eval.csv"
LOCAL_GOLD_TEXT_DIR = REGISTRY_DIR / "ground_truth_text" / "local_gold_pages"
LOCAL_GOLD_EXTRACTION_MANIFEST_PATH = ROOT / "data" / "gold_pages_only" / "extraction_manifest.csv"
LEGACY_GOLD_MANIFEST_PATH = ROOT / "data" / "gold" / "manifest_from_gold.txt"
REPORTS_DIR = ROOT / "reports"
EVAL_ROOT = REPORTS_DIR / "real_gold_eval_runs"
PROFILE_PATH = ROOT / "profiles" / "akkadian_strict.json"

OUT_AUDIT_CSV = REPORTS_DIR / "ground_truth_alignment_audit.csv"
OUT_AUDIT_MD = REPORTS_DIR / "ground_truth_alignment_audit.md"
OUT_QUEUE_CSV = REPORTS_DIR / "ground_truth_correction_queue.csv"
OUT_QUEUE_MD = REPORTS_DIR / "ground_truth_correction_queue.md"
OUT_AUTO_FIXES_CSV = REPORTS_DIR / "ground_truth_auto_fixes.csv"

OUT_VERIFIED_METRICS_CSV = REPORTS_DIR / "alignment_verified_metrics.csv"
OUT_VERIFIED_EVAL_MD = REPORTS_DIR / "alignment_verified_evaluation.md"
OUT_VERIFIED_PER_DATASET_CSV = REPORTS_DIR / "alignment_verified_per_dataset_metrics.csv"
OUT_VERIFIED_PER_LANGUAGE_CSV = REPORTS_DIR / "alignment_verified_per_language_metrics.csv"
OUT_VERIFIED_PER_LAYOUT_CSV = REPORTS_DIR / "alignment_verified_per_layout_metrics.csv"

OUT_PREPROC_VERIFIED_CSV = REPORTS_DIR / "preprocessing_verified_ablation_metrics.csv"
OUT_PREPROC_VERIFIED_MD = REPORTS_DIR / "preprocessing_verified_ablation_report.md"

OUT_VERIFIED_FAILURE_CSV = REPORTS_DIR / "alignment_verified_failure_taxonomy.csv"
OUT_VERIFIED_FAILURE_MD = REPORTS_DIR / "alignment_verified_failure_taxonomy.md"
OUT_DATA_QUALITY_FAILURES_CSV = REPORTS_DIR / "data_quality_failures.csv"
OUT_DATA_QUALITY_FAILURES_MD = REPORTS_DIR / "data_quality_failures.md"

OUT_REGRESSION_PATH_FIXES_CSV = REPORTS_DIR / "regression_26_path_resolution_fixes.csv"
OUT_REGRESSION_CONVERSION_FIXES_MD = REPORTS_DIR / "regression_26_annotation_conversion_fixes.md"

OUT_RENDER_DPI_MATRIX_CSV = REPORTS_DIR / "render_dpi_benchmark_matrix.csv"
OUT_RENDER_DPI_PER_PAGE_CSV = REPORTS_DIR / "render_dpi_per_page_metrics.csv"
OUT_RENDER_DPI_REPORT_MD = REPORTS_DIR / "render_dpi_experiment_report.md"
OUT_RENDER_DPI_PAGE_ANALYSIS_CSV = REPORTS_DIR / "render_dpi_page_level_analysis.csv"
OUT_RENDER_DPI_PAGE_ANALYSIS_MD = REPORTS_DIR / "render_dpi_page_level_analysis.md"
OUT_RENDER_DPI_PROMOTION_MD = REPORTS_DIR / "render_dpi_promotion_decision.md"
OUT_RENDER_DPI_STRATEGY_MD = REPORTS_DIR / "render_dpi_strategy_report.md"

OUT_NEXT_EXPERIMENT_MD = REPORTS_DIR / "next_experiment_after_alignment.md"
OUT_FINAL_REPORT_MD = REPORTS_DIR / "ground_truth_alignment_and_verified_evaluation_report.md"

VERIFIED_SPLIT_SMOKE = SPLITS_DIR / "alignment_verified_smoke.jsonl"
VERIFIED_SPLIT_VALIDATION = SPLITS_DIR / "alignment_verified_validation.jsonl"
VERIFIED_SPLIT_REGRESSION = SPLITS_DIR / "alignment_verified_regression.jsonl"

PADDLE_ONLY_PROFILE_PATH = REPORTS_DIR / "alignment_audit_artifacts" / "paddle_only_profile.json"

OLD_SUMMARY_CSV = REPORTS_DIR / "expanded_gold_metrics.csv"

SUPPORTED_ANNOTATION_FORMATS = {"JSON_BOXES", "PAGE_XML", "ALTO_XML"} | set(LOCAL_GOLD_SUPPORTED_FORMATS)

EVALUATED_SPLIT_SOURCES = [
    {
        "source_label": "smoke_50",
        "split_kind": "smoke",
        "split_file": SPLITS_DIR / "smoke.jsonl",
        "run_label": "smoke_50",
    },
    {
        "source_label": "expanded_validation",
        "split_kind": "validation",
        "split_file": SPLITS_DIR / "validation.jsonl",
        "run_label": "expanded_validation",
    },
    {
        "source_label": "expanded_test",
        "split_kind": "test",
        "split_file": SPLITS_DIR / "test.jsonl",
        "run_label": "expanded_test",
    },
    {
        "source_label": "regression_26",
        "split_kind": "regression",
        "split_file": SPLITS_DIR / "regression_26.jsonl",
        "run_label": "regression_26",
    },
]

ISSUES_PROVENANCE_MISMATCH = {
    "invalid_page_key",
    "missing_page_number",
    "missing_identifier",
    "missing_source_pdf",
    "invalid_gold_record",
    "invalid_ocr_record",
}


@dataclass
class VerifiedEvalResult:
    split_label: str
    split_kind: str
    status: str
    split_file: Path
    gold_csv: Path | None
    input_dir: Path | None
    run_dir: Path | None
    eval_dir: Path | None
    summary: dict[str, Any] | None
    per_page_rows: list[dict[str, Any]]
    per_dataset_rows: list[dict[str, Any]]
    per_language_rows: list[dict[str, Any]]
    per_layout_rows: list[dict[str, Any]]
    per_doc_rows: list[dict[str, Any]]
    malformed_rows: list[dict[str, Any]]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_command(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _safe_float(value: Any) -> float:
    try:
        text = str(value).strip()
        if not text:
            return 0.0
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if not text:
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t", "on"}


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_values[lo])
    weight = idx - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * weight)


def _format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _normalize_path(path_value: str) -> str:
    return str(path_value or "").strip().replace("\\", "/")


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", str(text or ""))


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_page_key(value: str) -> str:
    key = str(value or "").strip()
    key = key.replace("\\", "/")
    key = Path(key).stem
    key = re.sub(r"\.pdf$", "", key, flags=re.IGNORECASE)
    key = re.sub(r"_page_(\d+)_page_\1$", r"_page_\1", key)
    key = re.sub(r"__+", "_", key)
    return key.strip("_").lower()


def _normalize_document_key(value: str) -> str:
    key = str(value or "").strip()
    key = key.replace("\\", "/")
    key = Path(key).stem
    key = re.sub(r"\.pdf$", "", key, flags=re.IGNORECASE)
    key = re.sub(r"__+", "_", key)
    return key.strip("_").lower()


def _canonical_page_lookup_key(value: str) -> str:
    normalized = _normalize_page_key(value)
    if normalized:
        return canonical_lookup_key(normalized)
    return canonical_lookup_key(value)


def _rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _resolve_case_insensitive(base: Path, rel_path: str) -> Path | None:
    parts = [p for p in Path(rel_path).parts if p not in {".", ""}]
    cursor = base
    for part in parts:
        exact = cursor / part
        if exact.exists():
            cursor = exact
            continue
        try:
            children = list(cursor.iterdir())
        except Exception:
            return None
        matches = [child for child in children if child.name.lower() == part.lower()]
        if not matches:
            return None
        matches.sort(key=lambda c: c.name)
        cursor = matches[0]
    return cursor if cursor.exists() else None


def _resolve_repo_path(path_value: str, *, purpose: str) -> tuple[str, bool, str]:
    raw = _normalize_path(path_value)
    if not raw:
        return "", False, "empty"

    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add_candidate(candidate: str, mode: str) -> None:
        key = candidate.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append((candidate, mode))

    if Path(raw).is_absolute():
        abs_path = Path(raw)
        if abs_path.exists():
            return _rel_path(abs_path), True, "absolute_exact"
        try:
            rel = str(abs_path.relative_to(ROOT)).replace("\\", "/")
            add_candidate(rel, "absolute_rebased")
        except Exception:
            pass

    add_candidate(raw, "relative_exact")
    if raw.startswith("./"):
        add_candidate(raw[2:], "relative_dot")
    if not raw.lower().startswith("data/"):
        add_candidate(f"data/{raw}", "data_root_relative")

    ext_variants: list[str] = []
    if purpose in {"source", "pdf", "image"}:
        ext_variants = [".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"]
    elif purpose in {"text", "layout"}:
        ext_variants = [".txt", ".csv", ".tsv", ".json", ".jsonl", ".xml"]

    for candidate, mode in candidates:
        rel = candidate.replace("\\", "/")
        exact = ROOT / rel
        if exact.exists():
            return _rel_path(exact), True, mode

        ci = _resolve_case_insensitive(ROOT, rel)
        if ci is not None:
            return _rel_path(ci), True, f"{mode}_casefold"

        if ext_variants:
            stem = str(Path(rel).with_suffix(""))
            for ext in ext_variants:
                alt_rel = f"{stem}{ext}"
                alt_path = ROOT / alt_rel
                if alt_path.exists():
                    return _rel_path(alt_path), True, f"{mode}_ext_variant"
                ci_alt = _resolve_case_insensitive(ROOT, alt_rel)
                if ci_alt is not None:
                    return _rel_path(ci_alt), True, f"{mode}_ext_variant_casefold"

    return raw, False, "not_found"


_LOCAL_GOLD_FALLBACK_INDEX: dict[str, Any] | None = None


def _build_local_gold_fallback_index() -> dict[str, Any]:
    pdf_by_key: dict[str, list[Path]] = {}
    text_by_key: dict[str, list[Path]] = {}
    extraction_by_key: dict[str, list[dict[str, Any]]] = {}
    csv_by_key: dict[str, list[dict[str, Any]]] = {}
    legacy_by_key: dict[str, list[dict[str, Any]]] = {}

    gold_pages_only = ROOT / "data" / "gold_pages_only"
    if gold_pages_only.exists():
        for path in gold_pages_only.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}:
                continue
            key = _canonical_page_lookup_key(path.stem)
            pdf_by_key.setdefault(key, []).append(path)

    if LOCAL_GOLD_TEXT_DIR.exists():
        for path in LOCAL_GOLD_TEXT_DIR.iterdir():
            if not path.is_file() or path.suffix.lower() != ".txt":
                continue
            key = _canonical_page_lookup_key(path.stem)
            text_by_key.setdefault(key, []).append(path)

    if LOCAL_GOLD_EXTRACTION_MANIFEST_PATH.exists():
        for row in _read_csv(LOCAL_GOLD_EXTRACTION_MANIFEST_PATH):
            output_file = str(row.get("output_file", "")).strip()
            if not output_file:
                continue
            key = _canonical_page_lookup_key(Path(output_file).stem)
            source_pdf = str(row.get("source_pdf", "")).strip()
            page_num = _safe_int(row.get("page"), _extract_page_number(output_file))
            extraction_by_key.setdefault(key, []).append(
                {
                    "output_file": output_file,
                    "source_pdf": source_pdf,
                    "page": page_num,
                }
            )

    if LOCAL_GOLD_CSV_PATH.exists():
        for row in _read_csv(LOCAL_GOLD_CSV_PATH):
            pdf_name = str(row.get("pdf_name", "")).strip()
            if not pdf_name:
                continue
            page_num = _safe_int(row.get("page"), 1)
            key = build_source_page_key(pdf_name, page_num)
            csv_by_key.setdefault(key, []).append(row)

    if LEGACY_GOLD_MANIFEST_PATH.exists():
        with LEGACY_GOLD_MANIFEST_PATH.open("r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split("\t", 2)
                if len(parts) < 3:
                    continue
                source_file = parts[0].strip()
                page_num = _safe_int(parts[1], 1)
                key = build_source_page_key(source_file, page_num)
                legacy_by_key.setdefault(key, []).append(
                    {
                        "source_file": source_file,
                        "page": page_num,
                    }
                )

    for mapping in (pdf_by_key, text_by_key):
        for _, items in mapping.items():
            items.sort(key=lambda p: str(p))

    return {
        "pdf_by_key": pdf_by_key,
        "text_by_key": text_by_key,
        "extraction_by_key": extraction_by_key,
        "csv_by_key": csv_by_key,
        "legacy_by_key": legacy_by_key,
    }


def _get_local_gold_fallback_index() -> dict[str, Any]:
    global _LOCAL_GOLD_FALLBACK_INDEX
    if _LOCAL_GOLD_FALLBACK_INDEX is None:
        _LOCAL_GOLD_FALLBACK_INDEX = _build_local_gold_fallback_index()
    return _LOCAL_GOLD_FALLBACK_INDEX


def _build_synthetic_local_gold_record(page_id: str, split_kind: str) -> tuple[dict[str, Any] | None, str, list[dict[str, str]]]:
    index = _get_local_gold_fallback_index()
    key = _canonical_page_lookup_key(page_id)

    fix_rows: list[dict[str, str]] = []

    pdf_path = ""
    source_file = ""

    extraction_rows = index["extraction_by_key"].get(key, [])
    if extraction_rows:
        chosen = extraction_rows[0]
        output_file = str(chosen.get("output_file", "")).strip()
        if output_file:
            pdf_path = _normalize_path(str((Path("data/gold_pages_only") / output_file).as_posix()))
        source_pdf = str(chosen.get("source_pdf", "")).strip()
        if source_pdf:
            source_file = _normalize_path(str((Path("data/input_pdfs") / source_pdf).as_posix()))

    if not pdf_path:
        pdf_candidates = index["pdf_by_key"].get(key, [])
        if pdf_candidates:
            pdf_path = _rel_path(pdf_candidates[0])

    if not source_file:
        source_file = pdf_path

    gt_text_path = ""
    annotation_format = "PLAIN_TEXT"

    text_candidates = index["text_by_key"].get(key, [])
    if text_candidates:
        gt_text_path = _rel_path(text_candidates[0])
    elif index["csv_by_key"].get(key):
        gt_text_path = _rel_path(LOCAL_GOLD_CSV_PATH)
        annotation_format = "LOCAL_GOLD_CSV"
    elif index["legacy_by_key"].get(key):
        gt_text_path = _rel_path(LEGACY_GOLD_MANIFEST_PATH)
        annotation_format = "LOCAL_GOLD_MANIFEST_TSV"

    if not pdf_path and not gt_text_path:
        return None, "local_gold_fallback_unresolved", fix_rows

    if pdf_path:
        fix_rows.append(
            {
                "fix_type": "local_gold_path_recovery",
                "before": page_id,
                "after": pdf_path,
                "confidence": "medium",
                "reason": "Recovered local gold page asset by canonical key matching.",
                "applied": "true",
            }
        )
    if gt_text_path:
        fix_rows.append(
            {
                "fix_type": "local_gold_text_recovery",
                "before": page_id,
                "after": gt_text_path,
                "confidence": "medium",
                "reason": "Recovered local gold text source by canonical key matching.",
                "applied": "true",
            }
        )

    document_id = re.sub(r"_page_\d+$", "", page_id)
    record = {
        "dataset_id": "local_gold_pages",
        "page_id": page_id,
        "document_id": document_id,
        "source_dataset": "local_gold_pages",
        "source_file": source_file,
        "local_image_path": "",
        "local_pdf_path": pdf_path,
        "ground_truth_text_path": gt_text_path,
        "ground_truth_layout_path": "",
        "annotation_format": annotation_format,
        "split": split_kind,
        "language_primary": "unknown",
        "languages_present": "unknown",
        "script_type": "unknown",
        "document_type": "historical_book",
        "layout_type": "multi_column",
        "scan_quality": "unknown",
        "has_tables": False,
        "has_footnotes": False,
        "has_columns": True,
        "has_diacritics": True,
        "has_transliteration": False,
    }
    return record, "local_gold_fallback_recovered", fix_rows


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", str(text or ""), flags=re.UNICODE)


def _char_overlap_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    ca = Counter(a)
    cb = Counter(b)
    shared = sum(min(ca[k], cb[k]) for k in ca.keys() | cb.keys())
    denom = float(max(len(a), len(b)))
    return shared / denom if denom else 0.0


def _token_overlap_ratio(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    ca = Counter(ta)
    cb = Counter(tb)
    shared = sum(min(ca[k], cb[k]) for k in ca.keys() | cb.keys())
    denom = float(max(len(ta), len(tb)))
    return shared / denom if denom else 0.0


def _sequence_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _lcs_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    matcher = difflib.SequenceMatcher(None, a, b)
    total = sum(block.size for block in matcher.get_matching_blocks())
    denom = float(max(len(a), len(b)))
    return total / denom if denom else 0.0


def _edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    m = len(b)
    prev = list(range(m + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * m
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[m]


def _rough_edit_distance_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    if max(len(a), len(b)) > 1800:
        return max(0.0, 1.0 - _sequence_ratio(a, b))
    distance = _edit_distance(list(a), list(b))
    denom = float(max(len(a), len(b)))
    return distance / denom if denom else 0.0


def _cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(list(reference), list(hypothesis)) / len(reference)


def _wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def _is_mostly_markup(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    tag_count = len(re.findall(r"<[^>]+>", cleaned))
    markup_chars = sum(1 for ch in cleaned if ch in "<>[]{}=/")
    alpha_chars = sum(1 for ch in cleaned if ch.isalpha())
    ratio = markup_chars / float(len(cleaned))
    if tag_count >= 3 and alpha_chars < len(cleaned) * 0.5:
        return True
    return ratio > 0.33


def _is_mostly_junk(text: str) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False
    punct_or_symbols = sum(1 for ch in cleaned if (not ch.isalnum()) and (not ch.isspace()))
    ratio = punct_or_symbols / float(len(cleaned))
    if ratio > 0.75:
        return True
    if re.search(r"(.)\1{11,}", cleaned):
        return True
    if len(_tokenize(cleaned)) <= 1 and len(cleaned) > 32:
        return True
    return False


def _extract_page_number(text: str) -> int:
    matches = re.findall(r"_page_(\d+)", str(text or ""))
    if matches:
        return int(matches[-1])
    return 1


def _local_tag_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _extract_text_from_json(obj: Any) -> list[str]:
    out: list[str] = []
    keys = {
        "text",
        "value",
        "label",
        "content",
        "transcription",
        "line_text",
        "ocr",
        "word",
    }
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in keys and isinstance(value, str):
                txt = value.strip()
                if txt:
                    out.append(txt)
            out.extend(_extract_text_from_json(value))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_extract_text_from_json(item))
    return out


def _extract_layout_text(
    layout_path: Path,
    annotation_format: str,
    *,
    page_id: str = "",
    pdf_name: str = "",
    page_number: int = 1,
) -> tuple[str, str]:
    if not layout_path.exists():
        return "", "layout_path_missing"

    fmt = str(annotation_format or "").strip().upper()
    if not fmt:
        return "", "missing_annotation_format"

    try:
        if fmt in LOCAL_GOLD_SUPPORTED_FORMATS:
            text, warning = extract_local_gold_text(
                source_path=layout_path,
                annotation_format=fmt,
                page_id=page_id,
                pdf_name=pdf_name,
                page_number=page_number,
            )
            if text:
                return text, ""
            return "", warning or "local_gold_conversion_failed"

        if fmt == "JSON_BOXES":
            data = json.loads(layout_path.read_text(encoding="utf-8"))
            text_parts = _extract_text_from_json(data)
            if not text_parts:
                return "", "json_boxes_no_text_extracted"
            return "\n".join(text_parts), ""

        if fmt == "PAGE_XML":
            tree = ET.parse(layout_path)
            root = tree.getroot()
            lines: list[str] = []
            for elem in root.iter():
                local = _local_tag_name(elem.tag)
                if local == "Unicode" and elem.text:
                    txt = elem.text.strip()
                    if txt:
                        lines.append(txt)
            if not lines:
                return "", "page_xml_no_unicode_text"
            return "\n".join(lines), ""

        if fmt == "ALTO_XML":
            tree = ET.parse(layout_path)
            root = tree.getroot()
            lines: list[str] = []
            for line in root.iter():
                if _local_tag_name(line.tag) != "TextLine":
                    continue
                tokens: list[str] = []
                for child in line.iter():
                    if _local_tag_name(child.tag) == "String":
                        content = child.attrib.get("CONTENT", "").strip()
                        if content:
                            tokens.append(content)
                if tokens:
                    lines.append(" ".join(tokens))
            if not lines:
                return "", "alto_xml_no_line_text"
            return "\n".join(lines), ""

        return "", "unsupported_annotation_format"

    except Exception as exc:  # pragma: no cover - parsing robustness
        return "", f"annotation_parse_error:{exc.__class__.__name__}"


def _build_manifest_indexes(
    manifest_rows: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    by_page_id: dict[str, dict[str, Any]] = {}
    by_norm_key: dict[str, list[dict[str, Any]]] = {}
    by_pdf_name: dict[str, list[dict[str, Any]]] = {}
    by_page_reference: dict[str, list[dict[str, Any]]] = {}

    for row in manifest_rows:
        page_id = str(row.get("page_id", "")).strip()
        if not page_id:
            continue
        by_page_id[page_id] = row
        norm_key = _normalize_page_key(page_id)
        by_norm_key.setdefault(norm_key, []).append(row)

        local_pdf = Path(str(row.get("local_pdf_path", "")).strip()).name
        if local_pdf:
            by_pdf_name.setdefault(local_pdf, []).append(row)

        page_ref = str(row.get("page_id", "")).strip()
        by_page_reference.setdefault(page_ref, []).append(row)

    return by_page_id, by_norm_key, by_pdf_name, by_page_reference


def _load_run_row_maps(run_label: str) -> dict[str, Any]:
    run_csv = EVAL_ROOT / run_label / "run" / "client_page_text.csv"
    rows = _read_csv(run_csv)

    by_pdf_page: dict[tuple[str, int], dict[str, Any]] = {}
    by_page_reference: dict[str, dict[str, Any]] = {}
    by_pdf_stem: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        pdf_name = str(row.get("pdf_name", "")).strip()
        page_no = _safe_int(row.get("page")) or 1
        if pdf_name:
            by_pdf_page[(pdf_name, page_no)] = row
            by_pdf_stem.setdefault(Path(pdf_name).stem, []).append(row)
        page_ref = str(row.get("page_reference", "")).strip()
        if page_ref:
            by_page_reference[page_ref] = row

    return {
        "run_csv": run_csv,
        "rows": rows,
        "by_pdf_page": by_pdf_page,
        "by_page_reference": by_page_reference,
        "by_pdf_stem": by_pdf_stem,
    }


def _resolve_manifest_record(
    *,
    page_id: str,
    dataset_id: str,
    split_kind: str,
    by_page_id: dict[str, dict[str, Any]],
    by_norm_key: dict[str, list[dict[str, Any]]],
    by_pdf_name: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str, list[dict[str, Any]]]:
    auto_fixes: list[dict[str, Any]] = []
    record = by_page_id.get(page_id)
    if record is not None:
        return record, "page_id_exact", auto_fixes

    normalized_page_key = _normalize_page_key(page_id)
    candidates = by_norm_key.get(normalized_page_key, [])
    if dataset_id:
        dataset_matches = [c for c in candidates if str(c.get("dataset_id", "")) == dataset_id]
        if len(dataset_matches) == 1:
            auto_fixes.append(
                {
                    "fix_type": "page_key_normalization",
                    "before": page_id,
                    "after": str(dataset_matches[0].get("page_id", "")),
                    "confidence": "high",
                    "reason": "Unambiguous normalized page key match within dataset.",
                    "applied": "true",
                }
            )
            return dataset_matches[0], "normalized_page_key_dataset_match", auto_fixes

    if len(candidates) == 1:
        auto_fixes.append(
            {
                "fix_type": "page_key_normalization",
                "before": page_id,
                "after": str(candidates[0].get("page_id", "")),
                "confidence": "high",
                "reason": "Unambiguous normalized page key match.",
                "applied": "true",
            }
        )
        return candidates[0], "normalized_page_key_match", auto_fixes

    pdf_name = f"{page_id}.pdf"
    pdf_candidates = by_pdf_name.get(pdf_name, [])
    if dataset_id:
        dataset_matches = [c for c in pdf_candidates if str(c.get("dataset_id", "")) == dataset_id]
        if len(dataset_matches) == 1:
            auto_fixes.append(
                {
                    "fix_type": "pdf_name_page_reference_match",
                    "before": page_id,
                    "after": str(dataset_matches[0].get("page_id", "")),
                    "confidence": "high",
                    "reason": "Unambiguous pdf_name + dataset match.",
                    "applied": "true",
                }
            )
            return dataset_matches[0], "pdf_name_dataset_match", auto_fixes

    if len(pdf_candidates) == 1:
        auto_fixes.append(
            {
                "fix_type": "pdf_name_page_reference_match",
                "before": page_id,
                "after": str(pdf_candidates[0].get("page_id", "")),
                "confidence": "high",
                "reason": "Unambiguous pdf_name match.",
                "applied": "true",
            }
        )
        return pdf_candidates[0], "pdf_name_match", auto_fixes

    if len(candidates) > 1:
        auto_fixes.append(
            {
                "fix_type": "page_key_normalization",
                "before": page_id,
                "after": "",
                "confidence": "low",
                "reason": "Ambiguous normalized page key match; manual review required.",
                "applied": "false",
            }
        )

    if dataset_id == "local_gold_pages":
        synthetic, mode, fallback_fixes = _build_synthetic_local_gold_record(page_id, split_kind)
        auto_fixes.extend(fallback_fixes)
        if synthetic is not None:
            return synthetic, mode, auto_fixes

    return None, "unresolved", auto_fixes


def _find_run_row(
    *,
    page_id: str,
    page_reference: str,
    page_no: int,
    run_maps: dict[str, Any],
) -> dict[str, Any] | None:
    expected_pdf_name = f"{page_id}.pdf"
    row = run_maps["by_pdf_page"].get((expected_pdf_name, page_no))
    if row is not None:
        return row

    if page_reference:
        row = run_maps["by_page_reference"].get(page_reference)
        if row is not None:
            return row

    row = run_maps["by_page_reference"].get(page_id)
    if row is not None:
        return row

    stem_rows = run_maps["by_pdf_stem"].get(page_id, [])
    if len(stem_rows) == 1:
        return stem_rows[0]

    return None


def _alignment_score(
    *,
    unicode_similarity: float,
    lcs_ratio: float,
    char_overlap_ratio: float,
    token_overlap_ratio: float,
    rough_edit_distance_ratio: float,
    page_mismatch: bool,
    duplicate_warning: bool,
    source_exists: bool,
    ground_truth_exists: bool,
) -> float:
    base = (
        (unicode_similarity * 0.30)
        + (lcs_ratio * 0.25)
        + (char_overlap_ratio * 0.20)
        + (token_overlap_ratio * 0.15)
        + ((1.0 - rough_edit_distance_ratio) * 0.10)
    )
    if not source_exists:
        base -= 0.40
    if not ground_truth_exists:
        base -= 0.40
    if page_mismatch:
        base -= 0.25
    if duplicate_warning:
        base -= 0.08
    return max(0.0, min(1.0, base))


def _classify_alignment_status(
    *,
    source_file_exists: bool,
    ground_truth_exists: bool,
    unsupported_annotation_format: bool,
    gold_char_count: int,
    ocr_char_count: int,
    page_mismatch_indicator: bool,
    severe_text_mismatch: bool,
    line_order_mismatch: bool,
    layout_extraction_warning: bool,
    annotation_conversion_warning: bool,
    requires_manual_review: bool,
) -> str:
    if not source_file_exists:
        return "missing_source_file"
    if not ground_truth_exists:
        return "missing_ground_truth"
    if unsupported_annotation_format:
        return "unsupported_annotation_format"
    if gold_char_count < 5:
        return "suspicious_empty_or_too_short_gold"
    if ocr_char_count < 3:
        return "suspicious_empty_or_too_short_ocr"
    if page_mismatch_indicator:
        return "suspicious_page_mismatch"
    if annotation_conversion_warning:
        return "suspicious_annotation_conversion"
    if layout_extraction_warning:
        return "suspicious_layout_text_extraction"
    if line_order_mismatch:
        return "suspicious_line_order_mismatch"
    if severe_text_mismatch:
        return "suspicious_text_mismatch"
    if requires_manual_review:
        return "needs_manual_review"
    return "aligned"


def _status_severity(status: str) -> tuple[str, int]:
    mapping = {
        "missing_source_file": ("critical", 100),
        "missing_ground_truth": ("critical", 96),
        "unsupported_annotation_format": ("critical", 92),
        "suspicious_page_mismatch": ("high", 88),
        "suspicious_annotation_conversion": ("high", 83),
        "suspicious_layout_text_extraction": ("high", 80),
        "suspicious_line_order_mismatch": ("high", 76),
        "suspicious_text_mismatch": ("high", 73),
        "needs_manual_review": ("medium", 66),
        "suspicious_empty_or_too_short_gold": ("high", 75),
        "suspicious_empty_or_too_short_ocr": ("medium", 62),
        "safely_auto_fixed": ("low", 20),
        "aligned": ("low", 0),
    }
    return mapping.get(status, ("medium", 60))


def _suspected_root_cause(status: str, warnings: str) -> str:
    if status == "missing_source_file":
        return "source_asset_missing_or_path_drift"
    if status == "missing_ground_truth":
        return "ground_truth_missing"
    if status == "unsupported_annotation_format":
        return "converter_missing_for_annotation_format"
    if status == "suspicious_page_mismatch":
        return "page_reference_or_pdf_mapping_mismatch"
    if status == "suspicious_text_mismatch":
        return "gold_text_not_from_same_page"
    if status == "suspicious_line_order_mismatch":
        return "reading_order_or_line_pairing_mismatch"
    if status == "suspicious_layout_text_extraction":
        return "layout_extraction_or_region_flattening_issue"
    if status == "suspicious_annotation_conversion":
        return "annotation_conversion_or_parser_error"
    if status == "suspicious_empty_or_too_short_gold":
        return "empty_or_truncated_gold_text"
    if status == "suspicious_empty_or_too_short_ocr":
        return "ocr_empty_or_text_layer_not_usable"
    if status == "needs_manual_review":
        return "ambiguous_alignment_signals"
    if "duplicate" in warnings:
        return "duplicate_keys_or_duplicate_source_records"
    return "none"


def _recommended_fix(status: str) -> str:
    mapping = {
        "missing_source_file": "Fix source path mapping or remove record from scored benchmark until source files are restored.",
        "missing_ground_truth": "Regenerate or recover ground-truth text from authoritative annotation; keep excluded until available.",
        "unsupported_annotation_format": "Add/validate converter for this annotation format before using record for scoring.",
        "suspicious_page_mismatch": "Validate page_id, page_reference, pdf_name, and page-number linkage; remap only when unambiguous.",
        "suspicious_text_mismatch": "Check that gold text belongs to the same page/region as OCR output and not adjacent pages.",
        "suspicious_line_order_mismatch": "Audit reading-order reconstruction and line pairing from layout annotations.",
        "suspicious_layout_text_extraction": "Fix layout text flattening/conversion and verify region ordering.",
        "suspicious_annotation_conversion": "Review parser logs and converter assumptions for this dataset format.",
        "suspicious_empty_or_too_short_gold": "Inspect source annotation and regenerate text extraction output.",
        "suspicious_empty_or_too_short_ocr": "Rerun OCR for this page and verify rendering/text-layer fallback behavior.",
        "needs_manual_review": "Manual triage required; keep excluded from quality claims until resolved.",
        "safely_auto_fixed": "Use auto-normalized record and keep audit trail.",
        "aligned": "No correction needed.",
    }
    return mapping.get(status, "Manual review.")


def _build_audit_rows(
    manifest_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_page_id, by_norm_key, by_pdf_name, _ = _build_manifest_indexes(manifest_rows)

    all_rows: list[dict[str, Any]] = []
    auto_fixes: list[dict[str, Any]] = []

    for split_source in EVALUATED_SPLIT_SOURCES:
        split_label = split_source["source_label"]
        split_kind = split_source["split_kind"]
        split_entries = _read_jsonl(Path(split_source["split_file"]))
        run_maps = _load_run_row_maps(str(split_source["run_label"]))

        for entry in split_entries:
            page_id = str(entry.get("page_id", "")).strip()
            dataset_id = str(entry.get("dataset_id", "")).strip()
            if not page_id:
                continue

            record_id = f"{split_kind}:{page_id}"
            manifest_record, resolution_mode, fix_rows = _resolve_manifest_record(
                page_id=page_id,
                dataset_id=dataset_id,
                split_kind=split_kind,
                by_page_id=by_page_id,
                by_norm_key=by_norm_key,
                by_pdf_name=by_pdf_name,
            )
            for fix in fix_rows:
                auto_fixes.append(
                    {
                        "record_id": record_id,
                        "fix_type": fix["fix_type"],
                        "before": fix["before"],
                        "after": fix["after"],
                        "confidence": fix["confidence"],
                        "reason": fix["reason"],
                        "applied": fix["applied"],
                    }
                )

            resolved_page_id = str(manifest_record.get("page_id", "")).strip() if manifest_record else ""
            resolved_dataset_id = str(manifest_record.get("dataset_id", "")).strip() if manifest_record else dataset_id

            source_file_original = _normalize_path(str(manifest_record.get("source_file", "")) if manifest_record else "")
            local_image_path_original = _normalize_path(
                str(manifest_record.get("local_image_path", "")) if manifest_record else ""
            )
            local_pdf_path_original = _normalize_path(
                str(manifest_record.get("local_pdf_path", "")) if manifest_record else ""
            )
            gt_text_path_original = _normalize_path(
                str(manifest_record.get("ground_truth_text_path", "")) if manifest_record else ""
            )
            gt_layout_path_original = _normalize_path(
                str(manifest_record.get("ground_truth_layout_path", "")) if manifest_record else ""
            )
            annotation_format = str(manifest_record.get("annotation_format", "")) if manifest_record else ""
            source_dataset = str(manifest_record.get("source_dataset", "")) if manifest_record else ""
            document_id = str(manifest_record.get("document_id", "")) if manifest_record else ""

            source_file, source_file_path_exists, source_file_resolution_mode = _resolve_repo_path(
                source_file_original,
                purpose="source",
            )
            local_image_path, local_image_exists, local_image_resolution_mode = _resolve_repo_path(
                local_image_path_original,
                purpose="image",
            )
            local_pdf_path, local_pdf_exists, local_pdf_resolution_mode = _resolve_repo_path(
                local_pdf_path_original,
                purpose="pdf",
            )
            gt_text_path, gt_text_exists, gt_text_resolution_mode = _resolve_repo_path(
                gt_text_path_original,
                purpose="text",
            )
            gt_layout_path, gt_layout_exists, gt_layout_resolution_mode = _resolve_repo_path(
                gt_layout_path_original,
                purpose="layout",
            )

            path_resolution_events = [
                ("source_file", source_file_original, source_file, source_file_resolution_mode),
                ("local_image_path", local_image_path_original, local_image_path, local_image_resolution_mode),
                ("local_pdf_path", local_pdf_path_original, local_pdf_path, local_pdf_resolution_mode),
                ("ground_truth_text_path", gt_text_path_original, gt_text_path, gt_text_resolution_mode),
                ("ground_truth_layout_path", gt_layout_path_original, gt_layout_path, gt_layout_resolution_mode),
            ]
            for field_name, before_path, after_path, mode in path_resolution_events:
                if not before_path:
                    continue
                if before_path == after_path and mode in {"relative_exact", "absolute_exact"}:
                    continue
                if mode == "not_found":
                    continue
                auto_fixes.append(
                    {
                        "record_id": record_id,
                        "fix_type": f"path_resolution:{field_name}",
                        "before": before_path,
                        "after": after_path,
                        "confidence": "medium",
                        "reason": f"Resolved path using {mode}.",
                        "applied": "true",
                    }
                )

            source_file_abs = ROOT / source_file if source_file else Path("")
            local_image_abs = ROOT / local_image_path if local_image_path else Path("")
            local_pdf_abs = ROOT / local_pdf_path if local_pdf_path else Path("")
            gt_text_abs = ROOT / gt_text_path if gt_text_path else Path("")
            gt_layout_abs = ROOT / gt_layout_path if gt_layout_path else Path("")

            source_file_exists = bool(local_pdf_exists and local_pdf_abs.exists()) or bool(local_image_exists and local_image_abs.exists())

            page_reference = resolved_page_id or page_id
            page_no = _extract_page_number(page_reference)

            gold_text_raw = ""
            gold_text_warning = ""
            gold_text_source = ""

            annotation_fmt_upper = str(annotation_format or "").strip().upper()
            if gt_text_path and gt_text_abs.exists():
                if annotation_fmt_upper in LOCAL_GOLD_SUPPORTED_FORMATS:
                    gold_text_raw, gold_text_warning = extract_local_gold_text(
                        source_path=gt_text_abs,
                        annotation_format=annotation_fmt_upper,
                        page_id=page_reference,
                        pdf_name=f"{page_reference}.pdf",
                        page_number=page_no,
                    )
                    gold_text_source = "ground_truth_text_converted"
                else:
                    gold_text_raw = gt_text_abs.read_text(encoding="utf-8", errors="ignore")
                    gold_text_source = "ground_truth_text_raw"

            if (
                (not gold_text_raw)
                and resolved_dataset_id == "local_gold_pages"
                and LEGACY_GOLD_MANIFEST_PATH.exists()
            ):
                fallback_text, fallback_warning = extract_local_gold_text(
                    source_path=LEGACY_GOLD_MANIFEST_PATH,
                    annotation_format="LOCAL_GOLD_MANIFEST_TSV",
                    page_id=page_reference,
                    pdf_name=f"{page_reference}.pdf",
                    page_number=page_no,
                )
                if fallback_text:
                    gold_text_raw = fallback_text
                    gold_text_warning = ""
                    gold_text_source = "legacy_gold_manifest_fallback"
                    if not gt_text_path:
                        gt_text_path = _rel_path(LEGACY_GOLD_MANIFEST_PATH)
                        gt_text_abs = LEGACY_GOLD_MANIFEST_PATH
                        gt_text_exists = True
                    if not annotation_format:
                        annotation_format = "LOCAL_GOLD_MANIFEST_TSV"
                        annotation_fmt_upper = "LOCAL_GOLD_MANIFEST_TSV"
                elif not gold_text_warning:
                    gold_text_warning = fallback_warning

            ground_truth_exists = bool(_normalize_whitespace(gold_text_raw))

            gold_text_unicode = _normalize_unicode(gold_text_raw)
            if gold_text_unicode != gold_text_raw:
                auto_fixes.append(
                    {
                        "record_id": record_id,
                        "fix_type": "unicode_normalization_gold_text",
                        "before": gold_text_raw[:160],
                        "after": gold_text_unicode[:160],
                        "confidence": "high",
                        "reason": "NFC normalization for stable multilingual comparison.",
                        "applied": "true",
                    }
                )

            gold_text_effective = _normalize_whitespace(gold_text_unicode)
            if gold_text_effective != gold_text_unicode:
                auto_fixes.append(
                    {
                        "record_id": record_id,
                        "fix_type": "whitespace_normalization_gold_text",
                        "before": gold_text_unicode[:160],
                        "after": gold_text_effective[:160],
                        "confidence": "high",
                        "reason": "Whitespace normalization to avoid formatting-only mismatches.",
                        "applied": "true",
                    }
                )

            run_row = _find_run_row(
                page_id=page_id,
                page_reference=page_reference,
                page_no=page_no,
                run_maps=run_maps,
            )

            ocr_text_raw = str(run_row.get("page_text", "")) if run_row else ""
            ocr_text_effective = _normalize_whitespace(_normalize_unicode(ocr_text_raw))

            layout_text, layout_warning = ("", "")
            if gt_layout_path and gt_layout_abs.exists():
                layout_text, layout_warning = _extract_layout_text(
                    gt_layout_abs,
                    annotation_format,
                    page_id=page_reference,
                    pdf_name=f"{page_reference}.pdf",
                    page_number=page_no,
                )

            gold_char_count = len(gold_text_effective)
            ocr_char_count = len(ocr_text_effective)

            gold_is_empty = gold_char_count == 0
            ocr_is_empty = ocr_char_count == 0

            if gold_char_count and ocr_char_count:
                length_ratio = ocr_char_count / float(gold_char_count)
            else:
                length_ratio = 0.0

            char_overlap = _char_overlap_ratio(gold_text_effective, ocr_text_effective)
            token_overlap = _token_overlap_ratio(gold_text_effective, ocr_text_effective)
            unicode_similarity = _sequence_ratio(gold_text_effective, ocr_text_effective)
            lcs = _lcs_ratio(gold_text_effective, ocr_text_effective)
            rough_edit_ratio = _rough_edit_distance_ratio(gold_text_effective, ocr_text_effective)

            layout_text_effective = _normalize_whitespace(_normalize_unicode(layout_text))
            layout_similarity = _sequence_ratio(gold_text_effective, layout_text_effective)

            expected_pdf_name = f"{page_id}.pdf"
            observed_pdf_name = str(run_row.get("pdf_name", "")) if run_row else ""
            manifest_pdf_name = Path(local_pdf_path).name if local_pdf_path else ""

            page_ref_consistent = _normalize_page_key(page_reference) == _normalize_page_key(page_id)
            pdf_name_consistent = (not observed_pdf_name) or (
                Path(observed_pdf_name).name == expected_pdf_name
                or Path(observed_pdf_name).name == manifest_pdf_name
            )

            pid_page_no = _extract_page_number(page_id)
            obs_page_no = _safe_int(run_row.get("page", "")) if run_row else 0
            # Each normalized input PDF is a single page for the evaluator, so run-page should stay 1.
            filename_page_consistent = (obs_page_no == 0) or (obs_page_no == 1)

            manifest_split = str(manifest_record.get("split", "")) if manifest_record else ""
            split_entry_split = str(entry.get("split", split_kind))
            manifest_split_consistent = (not manifest_split) or (manifest_split == split_entry_split)

            severe_text_mismatch = (
                (gold_char_count >= 24)
                and (ocr_char_count >= 24)
                and unicode_similarity < 0.04
                and token_overlap < 0.05
                and char_overlap < 0.10
            )

            line_order_mismatch = False
            if gold_char_count > 40 and ocr_char_count > 40:
                if token_overlap > 0.65 and lcs < 0.32 and unicode_similarity < 0.40:
                    annotation_fmt_norm = str(annotation_format or "").strip().upper()
                    is_layout_source = annotation_fmt_norm in {"PAGE_XML", "ALTO_XML", "JSON_BOXES"}
                    has_strong_content_match = (
                        char_overlap >= 0.90
                        and token_overlap >= 0.80
                        and 0.60 <= length_ratio <= 1.60
                    )

                    # Plain/local text sources can legitimately differ in line breaks/order while still
                    # representing the same page; keep them alignment-eligible when overlap is very high.
                    if is_layout_source or not has_strong_content_match:
                        line_order_mismatch = True
                    else:
                        auto_fixes.append(
                            {
                                "record_id": record_id,
                                "fix_type": "line_order_warning_downgraded",
                                "field": "line_order_mismatch",
                                "before": "true",
                                "after": "false",
                                "confidence": "medium",
                                "reason": "High-overlap plain/local text row; ordering variance treated as non-blocking.",
                                "applied": "true",
                            }
                        )

            annotation_conversion_warning = bool(
                layout_warning.startswith("annotation_parse_error")
                or layout_warning == "unsupported_annotation_format"
                or gold_text_warning.startswith("local_gold_")
            )
            layout_extraction_warning = layout_warning in {
                "page_xml_no_unicode_text",
                "alto_xml_no_line_text",
            }

            page_mismatch_indicator = (
                (not page_ref_consistent or not pdf_name_consistent)
                and (unicode_similarity < 0.20 or token_overlap < 0.20)
            )

            unsupported_annotation_format = bool(
                annotation_format and str(annotation_format).strip().upper() not in SUPPORTED_ANNOTATION_FORMATS
            )

            warnings: list[str] = []
            if not page_ref_consistent:
                warnings.append("page_id_vs_page_reference_inconsistent")
            if not pdf_name_consistent:
                warnings.append("pdf_name_inconsistent")
            if not filename_page_consistent:
                warnings.append("filename_page_number_inconsistent")
            if not manifest_split_consistent:
                warnings.append("manifest_split_inconsistent")
            if layout_warning:
                warnings.append(layout_warning)
            if gold_text_warning:
                warnings.append(gold_text_warning)
            if _is_mostly_markup(gold_text_effective):
                warnings.append("gold_mostly_markup")
            if _is_mostly_junk(ocr_text_effective):
                warnings.append("ocr_mostly_junk")

            requires_manual_review = False
            if gold_char_count > 0 and ocr_char_count > 0 and unicode_similarity < 0.08 and token_overlap < 0.10:
                requires_manual_review = True

            status = _classify_alignment_status(
                source_file_exists=source_file_exists,
                ground_truth_exists=ground_truth_exists,
                unsupported_annotation_format=unsupported_annotation_format,
                gold_char_count=gold_char_count,
                ocr_char_count=ocr_char_count,
                page_mismatch_indicator=page_mismatch_indicator,
                severe_text_mismatch=severe_text_mismatch,
                line_order_mismatch=line_order_mismatch,
                layout_extraction_warning=layout_extraction_warning,
                annotation_conversion_warning=annotation_conversion_warning,
                requires_manual_review=requires_manual_review,
            )

            row = {
                "record_id": record_id,
                "source_split_label": split_label,
                "split_kind": split_kind,
                "dataset_id": resolved_dataset_id,
                "document_id": document_id,
                "page_id": page_id,
                "original_page_id": page_id,
                "resolved_page_id": resolved_page_id,
                "normalized_page_key": _normalize_page_key(page_id),
                "source_dataset": source_dataset,
                "source_file": source_file,
                "source_file_original": source_file_original,
                "source_file_resolution_mode": source_file_resolution_mode,
                "source_filename": Path(source_file).name if source_file else "",
                "pdf_name": observed_pdf_name or expected_pdf_name,
                "page_reference": page_reference,
                "page_index": max(pid_page_no - 1, 0),
                "local_image_path": local_image_path,
                "local_image_path_original": local_image_path_original,
                "local_image_resolution_mode": local_image_resolution_mode,
                "local_pdf_path": local_pdf_path,
                "local_pdf_path_original": local_pdf_path_original,
                "local_pdf_resolution_mode": local_pdf_resolution_mode,
                "ground_truth_text_path": gt_text_path,
                "ground_truth_text_path_original": gt_text_path_original,
                "ground_truth_text_resolution_mode": gt_text_resolution_mode,
                "ground_truth_layout_path": gt_layout_path,
                "ground_truth_layout_path_original": gt_layout_path_original,
                "ground_truth_layout_resolution_mode": gt_layout_resolution_mode,
                "annotation_format": annotation_format,
                "manifest_split": manifest_split,
                "split_entry_split": split_entry_split,
                "source_file_exists": "true" if source_file_exists else "false",
                "ground_truth_exists": "true" if ground_truth_exists else "false",
                "gold_char_count": gold_char_count,
                "ocr_char_count": ocr_char_count,
                "gold_ocr_length_ratio": _format_float(length_ratio),
                "char_overlap_ratio": _format_float(char_overlap),
                "token_overlap_ratio": _format_float(token_overlap),
                "unicode_similarity": _format_float(unicode_similarity),
                "lcs_ratio": _format_float(lcs),
                "rough_edit_distance_ratio": _format_float(rough_edit_ratio),
                "gold_is_empty": "true" if gold_is_empty else "false",
                "ocr_is_empty": "true" if ocr_is_empty else "false",
                "gold_mostly_markup": "true" if _is_mostly_markup(gold_text_effective) else "false",
                "ocr_mostly_junk": "true" if _is_mostly_junk(ocr_text_effective) else "false",
                "layout_text_char_count": len(layout_text_effective),
                "layout_text_similarity": _format_float(layout_similarity),
                "layout_line_order_warning": "true" if line_order_mismatch else "false",
                "page_reference_consistent": "true" if page_ref_consistent else "false",
                "pdf_name_consistent": "true" if pdf_name_consistent else "false",
                "filename_page_consistent": "true" if filename_page_consistent else "false",
                "manifest_split_consistent": "true" if manifest_split_consistent else "false",
                "duplicate_normalized_key_count": 0,
                "duplicate_ground_truth_path_count": 0,
                "duplicate_image_path_count": 0,
                "multi_record_same_source_image": "false",
                "duplicate_key_warning": "false",
                "annotation_conversion_warning": "true" if annotation_conversion_warning else "false",
                "alignment_status": status,
                "alignment_warnings": "|".join(warnings),
                "alignment_score": "0",
                "safe_to_use_for_scoring": "false",
                "requires_manual_review": "true" if requires_manual_review else "false",
                "auto_fix_applied": "true" if any(x["applied"] == "true" for x in fix_rows) else "false",
                "gold_text_source": gold_text_source,
                "gold_text_effective": gold_text_effective,
                "ocr_text_effective": ocr_text_effective,
                "run_resolution_mode": resolution_mode,
                "page_mismatch_indicator": "true" if page_mismatch_indicator else "false",
            }
            all_rows.append(row)

    norm_key_counts = Counter(_normalize_page_key(str(row.get("normalized_page_key", ""))) for row in all_rows)
    gt_path_counts = Counter(str(row.get("ground_truth_text_path", "")) for row in all_rows)
    image_path_counts = Counter(str(row.get("local_image_path", "")) for row in all_rows)

    for row in all_rows:
        norm_key = _normalize_page_key(str(row.get("normalized_page_key", "")))
        gt_path = str(row.get("ground_truth_text_path", ""))
        image_path = str(row.get("local_image_path", ""))

        duplicate_norm = norm_key_counts.get(norm_key, 0)
        duplicate_gt = gt_path_counts.get(gt_path, 0) if gt_path else 0
        duplicate_img = image_path_counts.get(image_path, 0) if image_path else 0

        row["duplicate_normalized_key_count"] = duplicate_norm
        row["duplicate_ground_truth_path_count"] = duplicate_gt
        row["duplicate_image_path_count"] = duplicate_img
        row["multi_record_same_source_image"] = "true" if duplicate_img > 1 else "false"

        duplicate_warning = duplicate_norm > 1 or duplicate_gt > 1 or duplicate_img > 1
        row["duplicate_key_warning"] = "true" if duplicate_warning else "false"

        warnings = str(row.get("alignment_warnings", "")).strip()
        if duplicate_warning:
            warnings = f"{warnings}|duplicate_key_warning" if warnings else "duplicate_key_warning"
            row["requires_manual_review"] = "true"

        row["alignment_warnings"] = warnings

        score = _alignment_score(
            unicode_similarity=_safe_float(row.get("unicode_similarity")),
            lcs_ratio=_safe_float(row.get("lcs_ratio")),
            char_overlap_ratio=_safe_float(row.get("char_overlap_ratio")),
            token_overlap_ratio=_safe_float(row.get("token_overlap_ratio")),
            rough_edit_distance_ratio=_safe_float(row.get("rough_edit_distance_ratio")),
            page_mismatch=_safe_bool(row.get("page_mismatch_indicator")),
            duplicate_warning=duplicate_warning,
            source_exists=_safe_bool(row.get("source_file_exists")),
            ground_truth_exists=_safe_bool(row.get("ground_truth_exists")),
        )

        status = str(row.get("alignment_status", ""))
        if status == "aligned" and _safe_bool(row.get("requires_manual_review")):
            status = "needs_manual_review"

        if status == "aligned" and _safe_bool(row.get("auto_fix_applied")):
            status = "safely_auto_fixed"

        row["alignment_status"] = status
        row["alignment_score"] = _format_float(score)

        safe_to_use = status in {"aligned", "safely_auto_fixed"} and not _safe_bool(row.get("requires_manual_review"))
        if status in {
            "missing_source_file",
            "missing_ground_truth",
            "suspicious_page_mismatch",
            "needs_manual_review",
            "unsupported_annotation_format",
        }:
            safe_to_use = False
        row["safe_to_use_for_scoring"] = "true" if safe_to_use else "false"

    return all_rows, auto_fixes


def _write_audit_markdown(audit_rows: list[dict[str, Any]]) -> None:
    status_counts = Counter(str(row.get("alignment_status", "")) for row in audit_rows)
    split_counts = Counter(str(row.get("split_kind", "")) for row in audit_rows)

    lines = [
        "# Ground-Truth Alignment Audit",
        "",
        "## Scope",
        f"- Audited records: {len(audit_rows)}",
        "- Every split record was audited; none were silently dropped.",
        "",
        "## Split coverage",
    ]
    for split_kind, count in sorted(split_counts.items()):
        lines.append(f"- {split_kind}: {count}")

    lines.extend([
        "",
        "## Alignment status counts",
        "",
        "| alignment_status | count |",
        "|---|---:|",
    ])

    for status, count in sorted(status_counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {status} | {count} |")

    lines.extend([
        "",
        "## Notes",
        "- Core checks are language-agnostic and Unicode-safe.",
        "- Annotation-format parsing is supported for JSON_BOXES, PAGE_XML, and ALTO_XML.",
        "- Low-confidence or ambiguous records are routed to manual review.",
    ])

    OUT_AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_correction_queue(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue_rows: list[dict[str, Any]] = []

    for row in audit_rows:
        status = str(row.get("alignment_status", ""))
        safe_to_use = _safe_bool(row.get("safe_to_use_for_scoring"))
        requires_manual = _safe_bool(row.get("requires_manual_review"))

        if status in {"aligned", "safely_auto_fixed"} and safe_to_use and not requires_manual:
            continue

        severity_label, severity_score = _status_severity(status)
        if _safe_bool(row.get("duplicate_key_warning")):
            severity_score += 8
            if severity_label == "low":
                severity_label = "medium"

        suspected = _suspected_root_cause(status, str(row.get("alignment_warnings", "")))
        recommended = _recommended_fix(status)

        queue_rows.append(
            {
                "record_id": str(row.get("record_id", "")),
                "source_split_label": str(row.get("source_split_label", "")),
                "split_kind": str(row.get("split_kind", "")),
                "dataset_id": str(row.get("dataset_id", "")),
                "document_id": str(row.get("document_id", "")),
                "page_id": str(row.get("page_id", "")),
                "normalized_page_key": str(row.get("normalized_page_key", "")),
                "source_dataset": str(row.get("source_dataset", "")),
                "local_image_path": str(row.get("local_image_path", "")),
                "local_pdf_path": str(row.get("local_pdf_path", "")),
                "ground_truth_text_path": str(row.get("ground_truth_text_path", "")),
                "ground_truth_layout_path": str(row.get("ground_truth_layout_path", "")),
                "alignment_status": status,
                "alignment_score": str(row.get("alignment_score", "0")),
                "severity": severity_label,
                "severity_score": severity_score,
                "suspected_root_cause": suspected,
                "recommended_fix": recommended,
                "safe_to_use_for_scoring": str(row.get("safe_to_use_for_scoring", "false")),
                "requires_manual_review": str(row.get("requires_manual_review", "false")),
            }
        )

    queue_rows.sort(key=lambda r: (int(r["severity_score"]), -_safe_float(r["alignment_score"])), reverse=True)
    return queue_rows


def _write_correction_queue_markdown(queue_rows: list[dict[str, Any]]) -> None:
    by_severity = Counter(str(row.get("severity", "")) for row in queue_rows)
    manual_count = sum(1 for row in queue_rows if _safe_bool(row.get("requires_manual_review")))
    auto_fixable_count = sum(1 for row in queue_rows if str(row.get("alignment_status", "")) == "safely_auto_fixed")

    lines = [
        "# Ground-Truth Correction Queue",
        "",
        "## Summary",
        f"- Queued records: {len(queue_rows)}",
        f"- Manual-review records: {manual_count}",
        f"- Auto-fixed records retained in queue for traceability: {auto_fixable_count}",
        "",
        "## Severity breakdown",
    ]
    for severity, count in sorted(by_severity.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {severity}: {count}")

    lines.extend(
        [
            "",
            "## Top queued items",
            "",
            "| dataset_id | page_id | alignment_status | severity | suspected_root_cause | recommended_fix |",
            "|---|---|---|---|---|---|",
        ]
    )

    for row in queue_rows[:50]:
        lines.append(
            "| {dataset_id} | {page_id} | {alignment_status} | {severity} | {suspected_root_cause} | {recommended_fix} |".format(
                dataset_id=row.get("dataset_id", ""),
                page_id=row.get("page_id", ""),
                alignment_status=row.get("alignment_status", ""),
                severity=row.get("severity", ""),
                suspected_root_cause=row.get("suspected_root_cause", ""),
                recommended_fix=row.get("recommended_fix", ""),
            )
        )

    OUT_QUEUE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_verified_splits(
    audit_rows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_split_page: dict[tuple[str, str], dict[str, Any]] = {}
    for row in audit_rows:
        split_kind = str(row.get("split_kind", ""))
        page_id = str(row.get("page_id", ""))
        if split_kind and page_id:
            by_split_page[(split_kind, page_id)] = row

    split_plan = {
        "smoke": (SPLITS_DIR / "smoke.jsonl", VERIFIED_SPLIT_SMOKE),
        "validation": (SPLITS_DIR / "validation.jsonl", VERIFIED_SPLIT_VALIDATION),
        "regression": (SPLITS_DIR / "regression_26.jsonl", VERIFIED_SPLIT_REGRESSION),
    }

    verified_entries: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {}

    excluded_statuses = {
        "missing_source_file",
        "missing_ground_truth",
        "suspicious_page_mismatch",
        "needs_manual_review",
        "unsupported_annotation_format",
    }

    for split_kind, (in_file, out_file) in split_plan.items():
        original_entries = _read_jsonl(in_file)
        keep_rows: list[dict[str, Any]] = []
        excluded = 0
        missing_audit = 0

        for entry in original_entries:
            page_id = str(entry.get("page_id", "")).strip()
            if not page_id:
                excluded += 1
                continue

            audit = by_split_page.get((split_kind, page_id))
            if audit is None:
                missing_audit += 1
                excluded += 1
                continue

            status = str(audit.get("alignment_status", ""))
            safe = _safe_bool(audit.get("safe_to_use_for_scoring"))
            manual = _safe_bool(audit.get("requires_manual_review"))

            if status in excluded_statuses or manual or not safe:
                excluded += 1
                continue

            if status not in {"aligned", "safely_auto_fixed"}:
                excluded += 1
                continue

            row = dict(entry)
            row["alignment_status"] = status
            row["normalized_page_key"] = str(audit.get("normalized_page_key", ""))
            row["resolved_page_id"] = str(audit.get("resolved_page_id", "")) or page_id
            row["safe_to_use_for_scoring"] = "true"
            keep_rows.append(row)

        _write_jsonl(out_file, keep_rows)

        verified_entries[split_kind] = keep_rows
        summary[split_kind] = {
            "original": len(original_entries),
            "verified": len(keep_rows),
            "excluded": excluded,
            "missing_audit": missing_audit,
            "output_file": str(out_file.relative_to(ROOT)),
        }

    return verified_entries, summary


def _prepare_split_assets(
    *,
    split_label: str,
    split_kind: str,
    split_entries: list[dict[str, Any]],
    manifest_by_page_id: dict[str, dict[str, Any]],
    audit_by_split_page: dict[tuple[str, str], dict[str, Any]],
) -> tuple[Path, Path, int, int, list[dict[str, Any]]]:
    split_root = EVAL_ROOT / split_label
    input_dir = split_root / "input_pdfs"
    gold_csv = split_root / "gold.csv"

    if split_root.exists():
        shutil.rmtree(split_root)
    input_dir.mkdir(parents=True, exist_ok=True)

    gold_rows: list[dict[str, Any]] = []
    skipped = 0

    for entry in split_entries:
        page_id = str(entry.get("page_id", "")).strip()
        if not page_id:
            skipped += 1
            continue

        resolved_page_id = str(entry.get("resolved_page_id", "")).strip() or page_id
        manifest = manifest_by_page_id.get(resolved_page_id)
        if manifest is None:
            skipped += 1
            continue

        pdf_path = ROOT / str(manifest.get("local_pdf_path", ""))
        gt_path = ROOT / str(manifest.get("ground_truth_text_path", ""))
        if not pdf_path.exists():
            skipped += 1
            continue

        audit_row = audit_by_split_page.get((split_kind, page_id), {})
        gold_text_effective = str(audit_row.get("gold_text_effective", "")).strip()
        if not gold_text_effective and gt_path.exists():
            raw_text = gt_path.read_text(encoding="utf-8", errors="ignore")
            gold_text_effective = _normalize_whitespace(_normalize_unicode(raw_text))

        if not gold_text_effective:
            skipped += 1
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
                "ground_truth_text": gold_text_effective,
                "language_primary": str(manifest.get("language_primary", "unknown") or "unknown"),
                "languages_present": str(manifest.get("languages_present", "unknown") or "unknown"),
                "script_type": str(manifest.get("script_type", "unknown") or "unknown"),
                "document_type": str(manifest.get("document_type", "unknown") or "unknown"),
                "layout_type": str(manifest.get("layout_type", "unknown") or "unknown"),
                "has_tables": str(bool(manifest.get("has_tables", False))).lower(),
                "has_footnotes": str(bool(manifest.get("has_footnotes", False))).lower(),
                "has_columns": str(bool(manifest.get("has_columns", False))).lower(),
                "has_diacritics": str(bool(manifest.get("has_diacritics", False))).lower(),
                "has_transliteration": str(bool(manifest.get("has_transliteration", False))).lower(),
                "scan_quality": str(manifest.get("scan_quality", "unknown") or "unknown"),
                "dataset_id": str(manifest.get("dataset_id", "")),
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

    return gold_csv, input_dir, len(gold_rows), skipped, gold_rows


def _provenance_mismatch_count(malformed_rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in malformed_rows if str(row.get("issue", "")) in ISSUES_PROVENANCE_MISMATCH)


def _evaluate_verified_split(
    *,
    split_label: str,
    split_kind: str,
    split_entries: list[dict[str, Any]],
    manifest_by_page_id: dict[str, dict[str, Any]],
    audit_by_split_page: dict[tuple[str, str], dict[str, Any]],
) -> VerifiedEvalResult:
    if not split_entries:
        return VerifiedEvalResult(
            split_label=split_label,
            split_kind=split_kind,
            status="pending",
            split_file=Path(""),
            gold_csv=None,
            input_dir=None,
            run_dir=None,
            eval_dir=None,
            summary=None,
            per_page_rows=[],
            per_dataset_rows=[],
            per_language_rows=[],
            per_layout_rows=[],
            per_doc_rows=[],
            malformed_rows=[],
        )

    split_file = {
        "smoke": VERIFIED_SPLIT_SMOKE,
        "validation": VERIFIED_SPLIT_VALIDATION,
        "regression": VERIFIED_SPLIT_REGRESSION,
    }[split_kind]

    gold_csv, input_dir, runnable, skipped, gold_rows = _prepare_split_assets(
        split_label=split_label,
        split_kind=split_kind,
        split_entries=split_entries,
        manifest_by_page_id=manifest_by_page_id,
        audit_by_split_page=audit_by_split_page,
    )

    if runnable == 0:
        return VerifiedEvalResult(
            split_label=split_label,
            split_kind=split_kind,
            status="pending",
            split_file=split_file,
            gold_csv=gold_csv,
            input_dir=input_dir,
            run_dir=None,
            eval_dir=None,
            summary={"matched_pages": 0, "gold_pages": 0, "ocr_pages": 0, "skipped": skipped},
            per_page_rows=[],
            per_dataset_rows=[],
            per_language_rows=[],
            per_layout_rows=[],
            per_doc_rows=[],
            malformed_rows=[],
        )

    run_dir = EVAL_ROOT / split_label / "run"
    eval_dir = EVAL_ROOT / split_label / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

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
    summary = summary_rows[0] if summary_rows else None

    per_page_rows = _read_csv(eval_dir / "per_page_metrics.csv")
    gold_row_map = {
        (str(row.get("pdf_name", "")).strip(), _safe_int(row.get("page")) or 1): row for row in gold_rows
    }

    for row in per_page_rows:
        pdf_name = str(row.get("pdf_name", "")).strip()
        page = _safe_int(row.get("page")) or 1
        gold_row = gold_row_map.get((pdf_name, page), {})
        page_ref = str(gold_row.get("page_reference", "")) or str(row.get("page_reference", ""))
        audit = audit_by_split_page.get((split_kind, page_ref), {})
        row["dataset_id"] = str(gold_row.get("dataset_id", row.get("dataset_id", "unknown")))
        row["page_reference"] = page_ref
        row["alignment_status"] = str(audit.get("alignment_status", "unknown"))

    per_dataset_rows = _read_csv(eval_dir / "per_dataset_metrics.csv")
    per_language_rows = _read_csv(eval_dir / "per_language_metrics.csv")
    per_layout_rows = _read_csv(eval_dir / "per_layout_metrics.csv")
    per_doc_rows = _read_csv(eval_dir / "per_document_type_metrics.csv")
    malformed_rows = _read_csv(eval_dir / "malformed_rows.csv")

    return VerifiedEvalResult(
        split_label=split_label,
        split_kind=split_kind,
        status="measured" if summary is not None else "failed",
        split_file=split_file,
        gold_csv=gold_csv,
        input_dir=input_dir,
        run_dir=run_dir,
        eval_dir=eval_dir,
        summary=summary,
        per_page_rows=per_page_rows,
        per_dataset_rows=per_dataset_rows,
        per_language_rows=per_language_rows,
        per_layout_rows=per_layout_rows,
        per_doc_rows=per_doc_rows,
        malformed_rows=malformed_rows,
    )


def _summarize_verified_result(result: VerifiedEvalResult) -> dict[str, Any]:
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
            "failed_rate": "",
            "empty_rate": "",
            "malformed_row_count": len(result.malformed_rows),
            "provenance_mismatch_count": _provenance_mismatch_count(result.malformed_rows),
            "runtime_ms_mean": "",
            "runtime_ms_median": "",
            "runtime_ms_p90": "",
            "runtime_ms_p95": "",
            "notes": "pending_or_failed",
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
        "failed_rate": _safe_float(s.get("failed_page_rate")),
        "empty_rate": _safe_float(s.get("empty_output_rate")),
        "malformed_row_count": len(result.malformed_rows),
        "provenance_mismatch_count": _provenance_mismatch_count(result.malformed_rows),
        "runtime_ms_mean": _safe_float(s.get("runtime_ms_mean")),
        "runtime_ms_median": _safe_float(s.get("runtime_ms_p50")),
        "runtime_ms_p90": _safe_float(s.get("runtime_ms_p90")),
        "runtime_ms_p95": _safe_float(s.get("runtime_ms_p95")),
        "notes": "alignment_verified",
    }


def _load_old_unverified_summary() -> dict[str, dict[str, Any]]:
    rows = _read_csv(OLD_SUMMARY_CSV)
    by_split: dict[str, dict[str, Any]] = {}
    for row in rows:
        split = str(row.get("evaluation_split", "")).strip()
        if split:
            by_split[split] = row
    return by_split


def _write_alignment_verified_evaluation_md(
    verified_rows: list[dict[str, Any]],
    verified_split_summary: dict[str, Any],
    old_summary: dict[str, dict[str, Any]],
) -> None:
    old_map = {
        "alignment_verified_smoke": "smoke_50",
        "alignment_verified_validation": "expanded_validation",
        "alignment_verified_regression": "regression_26",
    }

    lines = [
        "# Alignment-Verified Evaluation",
        "",
        "## Verified split composition",
        f"- smoke: original={verified_split_summary['smoke']['original']} verified={verified_split_summary['smoke']['verified']} excluded={verified_split_summary['smoke']['excluded']}",
        f"- validation: original={verified_split_summary['validation']['original']} verified={verified_split_summary['validation']['verified']} excluded={verified_split_summary['validation']['excluded']}",
        f"- regression: original={verified_split_summary['regression']['original']} verified={verified_split_summary['regression']['verified']} excluded={verified_split_summary['regression']['excluded']}",
        "",
        "## Metrics (verified only)",
        "",
        "| split | CER mean | CER p90 | WER mean | WER p90 | failed rate | empty rate | runtime p95 ms | malformed | provenance mismatches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in verified_rows:
        if str(row.get("status", "")) != "measured":
            continue
        lines.append(
            "| {split} | {cer_mean:.6f} | {cer_p90:.6f} | {wer_mean:.6f} | {wer_p90:.6f} | {failed:.6f} | {empty:.6f} | {rt:.2f} | {malformed} | {prov} |".format(
                split=row.get("evaluation_split", ""),
                cer_mean=_safe_float(row.get("cer_mean")),
                cer_p90=_safe_float(row.get("cer_p90")),
                wer_mean=_safe_float(row.get("wer_mean")),
                wer_p90=_safe_float(row.get("wer_p90")),
                failed=_safe_float(row.get("failed_rate")),
                empty=_safe_float(row.get("empty_rate")),
                rt=_safe_float(row.get("runtime_ms_p95")),
                malformed=_safe_int(row.get("malformed_row_count")),
                prov=_safe_int(row.get("provenance_mismatch_count")),
            )
        )

    lines.extend(
        [
            "",
            "## Old vs verified comparison",
            "",
            "| split | old CER mean | verified CER mean | old WER mean | verified WER mean | note |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )

    for row in verified_rows:
        split = str(row.get("evaluation_split", ""))
        old_split = old_map.get(split, "")
        old = old_summary.get(old_split, {})
        old_cer = _safe_float(old.get("cer_mean"))
        old_wer = _safe_float(old.get("wer_mean"))
        new_cer = _safe_float(row.get("cer_mean"))
        new_wer = _safe_float(row.get("wer_mean"))

        note = ""
        if old and row.get("status") == "measured":
            if abs(new_cer - old_cer) > 0.05 or abs(new_wer - old_wer) > 0.05:
                note = "difference indicates prior metrics were sensitive to alignment quality"
            else:
                note = "verified and prior metrics are close"
        else:
            note = "no direct old metric available"

        lines.append(
            f"| {split} | {old_cer:.6f} | {new_cer:.6f} | {old_wer:.6f} | {new_wer:.6f} | {note} |"
        )

    lines.extend(
        [
            "",
            "## Continuity-only note",
            "- Original unverified regression_26 remains available only for continuity tracking, not for quality claims.",
            "- Quality claims in this report use alignment-verified records only.",
        ]
    )

    OUT_VERIFIED_EVAL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _register_micro_profiles() -> None:
    additions: dict[str, PreprocessingProfile] = {
        "high_dpi_render": PreprocessingProfile(
            name="high_dpi_render",
            render_dpi=420,
            preprocessing_overrides={
                "enable_denoise": False,
                "enable_adaptive_threshold": False,
                "enable_morphology": False,
                "enable_background_normalization": False,
                "deskew": True,
                "contrast_factor": 1.2,
                "avoid_aggressive_binarization": True,
                "profile_primary_variant": "original",
                "profile_variant_order": ["original", "contrast", "sharpen"],
            },
            description="High-DPI render only with conservative preprocessing.",
        ),
        "contrast_normalization": PreprocessingProfile(
            name="contrast_normalization",
            render_dpi=320,
            preprocessing_overrides={
                "enable_denoise": False,
                "enable_adaptive_threshold": False,
                "enable_morphology": False,
                "enable_background_normalization": True,
                "deskew": True,
                "contrast_factor": 2.0,
                "profile_primary_variant": "contrast",
                "profile_variant_order": ["contrast", "autocontrast", "original"],
            },
            description="Contrast and background normalization profile.",
        ),
        "adaptive_binarization": PreprocessingProfile(
            name="adaptive_binarization",
            render_dpi=320,
            preprocessing_overrides={
                "enable_denoise": False,
                "enable_adaptive_threshold": True,
                "adaptive_block_size": 35,
                "adaptive_C": 9,
                "enable_morphology": False,
                "deskew": True,
                "contrast_factor": 1.4,
                "profile_primary_variant": "adaptive",
                "profile_variant_order": ["adaptive", "contrast", "original"],
            },
            description="Adaptive thresholding profile.",
        ),
        "denoise_light": PreprocessingProfile(
            name="denoise_light",
            render_dpi=320,
            preprocessing_overrides={
                "enable_denoise": True,
                "denoise_strength": 8,
                "enable_adaptive_threshold": False,
                "enable_morphology": False,
                "deskew": True,
                "contrast_factor": 1.3,
                "profile_primary_variant": "contrast",
                "profile_variant_order": ["contrast", "original"],
            },
            description="Light denoise profile.",
        ),
        "deskew_if_needed": PreprocessingProfile(
            name="deskew_if_needed",
            render_dpi=320,
            preprocessing_overrides={
                "enable_denoise": False,
                "enable_adaptive_threshold": False,
                "enable_morphology": False,
                "deskew": True,
                "deskew_min_abs_degrees": 0.15,
                "deskew_max_abs_degrees": 45.0,
                "contrast_factor": 1.15,
                "profile_primary_variant": "original",
                "profile_variant_order": ["original", "contrast"],
            },
            description="Deskew-focused profile.",
        ),
        "high_dpi_plus_contrast": PreprocessingProfile(
            name="high_dpi_plus_contrast",
            render_dpi=420,
            preprocessing_overrides={
                "enable_denoise": False,
                "enable_adaptive_threshold": False,
                "enable_morphology": False,
                "enable_background_normalization": True,
                "deskew": True,
                "contrast_factor": 2.2,
                "profile_primary_variant": "contrast",
                "profile_variant_order": ["contrast", "autocontrast", "original"],
            },
            description="High-DPI with contrast normalization.",
        ),
        "high_dpi_plus_binarization": PreprocessingProfile(
            name="high_dpi_plus_binarization",
            render_dpi=420,
            preprocessing_overrides={
                "enable_denoise": True,
                "denoise_strength": 10,
                "enable_adaptive_threshold": True,
                "adaptive_block_size": 39,
                "adaptive_C": 8,
                "enable_morphology": True,
                "morphology_kernel_size": 2,
                "enable_background_normalization": True,
                "deskew": True,
                "contrast_factor": 1.9,
                "profile_primary_variant": "adaptive",
                "profile_variant_order": ["adaptive", "morphology", "contrast", "original"],
            },
            description="High-DPI adaptive threshold profile.",
        ),
    }
    for key, profile in additions.items():
        PREPROCESSING_PROFILES[key] = profile


def _build_paddle_only_profile() -> Path:
    base = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))

    engines = dict(base.get("engines", {}))
    engines["enabled"] = ["paddle"]
    base["engines"] = engines

    routing = dict(base.get("routing", {}))
    routing["engine_priority"] = ["paddle"]
    routing["fast_ocr_engines"] = ["paddle"]
    routing["layout_first_engines"] = ["paddle"]
    routing["diacritic_sensitive_engines"] = ["paddle"]
    routing["max_engines_per_page"] = 1
    base["routing"] = routing

    PADDLE_ONLY_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PADDLE_ONLY_PROFILE_PATH.write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return PADDLE_ONLY_PROFILE_PATH


def _run_verified_preprocessing_ablation(
    *,
    eval_results: dict[str, VerifiedEvalResult],
    audit_by_split_page: dict[tuple[str, str], dict[str, Any]],
    manifest_by_page_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _register_micro_profiles()
    profile_path = _build_paddle_only_profile()
    ensemble = FortifiedOCREnsemble(profile_path=str(profile_path))

    candidate_pages: list[dict[str, Any]] = []
    validation_result = eval_results.get("alignment_verified_validation")
    if validation_result is None or validation_result.status != "measured":
        return [], {"reason": "validation_not_measured", "selected_pages": 0}

    for row in validation_result.per_page_rows:
        page_id = str(row.get("page_reference", "")).strip()
        if not page_id:
            continue
        audit = audit_by_split_page.get(("validation", page_id), {})
        if str(audit.get("alignment_status", "")) not in {"aligned", "safely_auto_fixed"}:
            continue

        cer = _safe_float(row.get("cer"))
        scan_quality = str(row.get("scan_quality", "")).strip().lower()
        layout_type = str(row.get("layout_type", "")).strip().lower()

        difficult = cer >= 1.0 or scan_quality in {"noisy_scan", "mixed"} or layout_type in {
            "multi_column",
            "form_layout",
            "semi_structured",
        }
        if not difficult:
            continue

        candidate_pages.append(
            {
                "page_id": page_id,
                "cer": cer,
                "audit": audit,
            }
        )

    candidate_pages.sort(key=lambda x: x["cer"], reverse=True)
    selected = candidate_pages[:8]

    profiles: list[tuple[str, str]] = [
        ("baseline_auto", PROFILE_AUTO),
        ("noisy_scan", PROFILE_NOISY_SCAN),
        ("contrast_normalization", "contrast_normalization"),
        ("high_dpi_render", "high_dpi_render"),
        ("adaptive_binarization", "adaptive_binarization"),
        ("denoise_light", "denoise_light"),
        ("deskew_if_needed", "deskew_if_needed"),
        ("high_dpi_plus_contrast", "high_dpi_plus_contrast"),
        ("high_dpi_plus_binarization", "high_dpi_plus_binarization"),
    ]

    rows: list[dict[str, Any]] = []

    for entry in selected:
        page_id = str(entry["page_id"])
        audit = entry["audit"]
        resolved_page_id = str(audit.get("resolved_page_id", "")).strip() or page_id
        manifest = manifest_by_page_id.get(resolved_page_id)
        if manifest is None:
            continue

        pdf_path = ROOT / str(manifest.get("local_pdf_path", ""))
        if not pdf_path.exists():
            continue

        gold_text = str(audit.get("gold_text_effective", ""))
        page_no = _extract_page_number(resolved_page_id)
        page_index = max(page_no - 1, 0)

        baseline_text = ""
        for profile_id, internal_profile in profiles:
            t0 = time.perf_counter()
            text, meta = ensemble.extract_page_text(
                str(pdf_path),
                page_index,
                preprocessing_profile=internal_profile,
                diagnostics={},
                language_hint=str(manifest.get("language_primary", "unknown")),
                script_hint=str(manifest.get("script_type", "unknown")),
                document_type=str(manifest.get("document_type", "unknown")),
                debug_artifacts_dir="",
                debug_artifact_prefix="",
            )
            runtime_ms = (time.perf_counter() - t0) * 1000.0

            if profile_id == "baseline_auto":
                baseline_text = text

            rows.append(
                {
                    "dataset_id": str(manifest.get("dataset_id", "")),
                    "document_id": str(manifest.get("document_id", "")),
                    "page_id": page_id,
                    "resolved_page_id": resolved_page_id,
                    "alignment_status": str(audit.get("alignment_status", "")),
                    "profile_id": profile_id,
                    "profile_internal": internal_profile,
                    "profile_applied": str(meta.get("preprocessing_profile", internal_profile)),
                    "CER": _format_float(_cer(gold_text, text)),
                    "WER": _format_float(_wer(gold_text, text)),
                    "runtime_ms": _format_float(runtime_ms),
                    "text_length": len(text.strip()),
                    "output_changed": "false"
                    if profile_id == "baseline_auto"
                    else ("true" if text.strip() != baseline_text.strip() else "false"),
                    "final_output_source": str(meta.get("final_output_source", "")),
                }
            )

    context = {
        "selected_pages": len(selected),
        "measured_rows": len(rows),
    }
    return rows, context


def _write_verified_preprocessing_report(rows: list[dict[str, Any]], context: dict[str, Any]) -> None:
    _write_csv(
        OUT_PREPROC_VERIFIED_CSV,
        rows,
        [
            "dataset_id",
            "document_id",
            "page_id",
            "resolved_page_id",
            "alignment_status",
            "profile_id",
            "profile_internal",
            "profile_applied",
            "CER",
            "WER",
            "runtime_ms",
            "text_length",
            "output_changed",
            "final_output_source",
        ],
    )

    if not rows:
        lines = [
            "# Preprocessing Ablation on Alignment-Verified Records",
            "",
            "- insufficient_evidence: true",
            "- reason: No alignment-verified difficult pages were available for ablation.",
        ]
        OUT_PREPROC_VERIFIED_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    by_profile: dict[str, list[dict[str, Any]]] = {}
    by_page: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_profile.setdefault(str(row.get("profile_id", "")), []).append(row)
        by_page.setdefault(str(row.get("page_id", "")), []).append(row)

    improved_pages = 0
    for page_id, page_rows in by_page.items():
        baseline = next((r for r in page_rows if str(r.get("profile_id")) == "baseline_auto"), None)
        if baseline is None:
            continue
        base_cer = _safe_float(baseline.get("CER"))
        best = min(_safe_float(r.get("CER")) for r in page_rows)
        if best < base_cer - 1e-12:
            improved_pages += 1

    lines = [
        "# Preprocessing Ablation on Alignment-Verified Records",
        "",
        f"- selected_difficult_pages: {context.get('selected_pages', 0)}",
        f"- measured_rows: {context.get('measured_rows', 0)}",
        f"- pages_with_any_CER_improvement: {improved_pages}",
        "",
        "## Profile summary",
        "",
        "| profile_id | samples | mean CER | mean WER | mean runtime ms |",
        "|---|---:|---:|---:|---:|",
    ]

    for profile, items in sorted(by_profile.items()):
        lines.append(
            "| {profile} | {samples} | {cer:.6f} | {wer:.6f} | {rt:.2f} |".format(
                profile=profile,
                samples=len(items),
                cer=_mean([_safe_float(i.get("CER")) for i in items]),
                wer=_mean([_safe_float(i.get("WER")) for i in items]),
                rt=_mean([_safe_float(i.get("runtime_ms")) for i in items]),
            )
        )

    if len(by_page) < 3:
        lines.extend(
            [
                "",
                "## Outcome",
                "- insufficient_evidence: true",
                "- Too few verified difficult pages were available; do not promote preprocessing.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Outcome",
                "- Preprocessing was judged only on alignment-verified records.",
                "- Do not promote preprocessing unless verified metrics improve consistently.",
            ]
        )

    OUT_PREPROC_VERIFIED_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _categorize_verified_failure(page_row: dict[str, Any], run_row: dict[str, Any] | None) -> str:
    status = str(page_row.get("status", "")).strip().lower()
    failure_reason = str(page_row.get("failure_reason", "")).strip().lower()
    empty = _safe_bool(page_row.get("empty_output"))
    cer = _safe_float(page_row.get("cer"))
    layout = str(page_row.get("layout_type", "")).strip().lower()
    scan_quality = str(page_row.get("scan_quality", "")).strip().lower()

    output_text = str(run_row.get("page_text", "")) if run_row else ""
    output_len = _safe_int(run_row.get("output_text_length", "")) if run_row else 0

    if status == "timed_out" or "timeout" in failure_reason:
        return "OCR_timeout"
    if "render" in failure_reason:
        return "render_failure"
    if "text_layer" in failure_reason:
        return "text_layer_failure"
    if empty:
        gt_len = _safe_int(page_row.get("ground_truth_text_length", ""))
        if gt_len == 0:
            return "true_blank_page"
        return "suspicious_short_output"
    if _is_mostly_junk(output_text):
        return "suspicious_junk_output"
    if output_len < 8 and cer >= 0.8:
        return "suspicious_short_output"
    if cer >= 1.4 and layout in {"multi_column", "form_layout", "semi_structured"}:
        return "bad_layout"
    if cer >= 1.2 and scan_quality in {"noisy_scan", "mixed"}:
        return "low_resolution_or_noisy_scan"
    if cer >= 1.0 and scan_quality in {"noisy_scan", "mixed"}:
        return "preprocessing_needed"
    if status == "failed":
        return "OCR_engine_failure"
    if cer >= 1.2:
        return "OCR_engine_failure"
    return "unknown"


def _build_alignment_verified_failure_taxonomy(
    eval_results: dict[str, VerifiedEvalResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for split_label, result in eval_results.items():
        if result.status != "measured" or result.run_dir is None:
            continue

        run_rows = _read_csv(result.run_dir / "client_page_text.csv")
        run_map: dict[tuple[str, int], dict[str, Any]] = {}
        for run in run_rows:
            pdf_name = str(run.get("pdf_name", "")).strip()
            page_no = _safe_int(run.get("page")) or 1
            if pdf_name:
                run_map[(pdf_name, page_no)] = run

        for page in result.per_page_rows:
            alignment_status = str(page.get("alignment_status", ""))
            if alignment_status not in {"aligned", "safely_auto_fixed"}:
                continue

            status = str(page.get("status", "")).strip().lower()
            empty = _safe_bool(page.get("empty_output"))
            cer = _safe_float(page.get("cer"))

            if status == "success" and not empty and cer < 1.0:
                continue

            pdf_name = str(page.get("pdf_name", "")).strip()
            page_no = _safe_int(page.get("page")) or 1
            run_row = run_map.get((pdf_name, page_no))
            category = _categorize_verified_failure(page, run_row)

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
                    "final_output_source": str(run_row.get("final_output_source", "") if run_row else ""),
                    "best_available_output_length": _safe_int(run_row.get("output_text_length", "") if run_row else ""),
                    "CER": _safe_float(page.get("cer")),
                    "WER": _safe_float(page.get("wer")),
                    "recommended_next_fix": _recommended_fix(category),
                }
            )

    return rows


def _write_alignment_verified_failure_markdown(rows: list[dict[str, Any]]) -> None:
    counts = Counter(str(row.get("failure_category", "")) for row in rows)

    lines = [
        "# Alignment-Verified Failure Taxonomy",
        "",
        f"- failure_rows: {len(rows)}",
        "",
        "## Category counts",
        "",
        "| failure_category | count |",
        "|---|---:|",
    ]

    for category, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {category} | {count} |")

    lines.extend(
        [
            "",
            "## Notes",
            "- Taxonomy is computed only on alignment-verified records.",
            "- Alignment-excluded records are tracked separately as data-quality failures.",
        ]
    )

    OUT_VERIFIED_FAILURE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_data_quality_failures(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in audit_rows:
        status = str(row.get("alignment_status", ""))
        safe = _safe_bool(row.get("safe_to_use_for_scoring"))
        if status in {"aligned", "safely_auto_fixed"} and safe:
            continue

        severity, severity_score = _status_severity(status)
        rows.append(
            {
                "record_id": str(row.get("record_id", "")),
                "source_split_label": str(row.get("source_split_label", "")),
                "split_kind": str(row.get("split_kind", "")),
                "dataset_id": str(row.get("dataset_id", "")),
                "document_id": str(row.get("document_id", "")),
                "page_id": str(row.get("page_id", "")),
                "alignment_status": status,
                "severity": severity,
                "severity_score": severity_score,
                "suspected_root_cause": _suspected_root_cause(status, str(row.get("alignment_warnings", ""))),
                "recommended_fix": _recommended_fix(status),
                "safe_to_use_for_scoring": str(row.get("safe_to_use_for_scoring", "false")),
                "requires_manual_review": str(row.get("requires_manual_review", "false")),
            }
        )

    rows.sort(key=lambda r: int(r["severity_score"]), reverse=True)
    return rows


def _write_data_quality_failures_markdown(rows: list[dict[str, Any]]) -> None:
    counts = Counter(str(row.get("alignment_status", "")) for row in rows)
    lines = [
        "# Data-Quality Failures",
        "",
        f"- records: {len(rows)}",
        "",
        "## Status counts",
    ]

    for status, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {status}: {count}")

    lines.extend(
        [
            "",
            "## Top items",
            "",
            "| dataset_id | page_id | alignment_status | severity | suspected_root_cause |",
            "|---|---|---|---|---|",
        ]
    )

    for row in rows[:50]:
        lines.append(
            f"| {row['dataset_id']} | {row['page_id']} | {row['alignment_status']} | {row['severity']} | {row['suspected_root_cause']} |"
        )

    OUT_DATA_QUALITY_FAILURES_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_regression_blocker_rows() -> list[dict[str, Any]]:
    blockers_path = REPORTS_DIR / "regression_26_alignment_blockers.csv"
    if blockers_path.exists():
        return _read_csv(blockers_path)
    return []


def _write_regression_path_resolution_fixes(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocker_rows = _load_regression_blocker_rows()
    missing_source_ids = {
        str(row.get("page_id", "")).strip()
        for row in blocker_rows
        if str(row.get("exact_exclusion_reason", "")).strip() == "missing_source_file"
    }

    if not missing_source_ids:
        missing_source_ids = {
            str(row.get("page_id", "")).strip()
            for row in audit_rows
            if str(row.get("split_kind", "")).strip() == "regression"
            and str(row.get("alignment_status", "")).strip() == "missing_source_file"
        }

    by_page = {
        str(row.get("page_id", "")).strip(): row
        for row in audit_rows
        if str(row.get("split_kind", "")).strip() == "regression"
    }

    rows: list[dict[str, Any]] = []
    for page_id in sorted(missing_source_ids):
        row = by_page.get(page_id, {})
        source_exists = _safe_bool(row.get("source_file_exists"))
        resolution_outcome = "resolved_existing_asset" if source_exists else "still_missing"
        rows.append(
            {
                "page_id": page_id,
                "original_source_file": str(row.get("source_file_original", "")),
                "resolved_source_file": str(row.get("source_file", "")),
                "source_file_resolution_mode": str(row.get("source_file_resolution_mode", "")),
                "original_local_pdf_path": str(row.get("local_pdf_path_original", "")),
                "resolved_local_pdf_path": str(row.get("local_pdf_path", "")),
                "local_pdf_resolution_mode": str(row.get("local_pdf_resolution_mode", "")),
                "original_local_image_path": str(row.get("local_image_path_original", "")),
                "resolved_local_image_path": str(row.get("local_image_path", "")),
                "local_image_resolution_mode": str(row.get("local_image_resolution_mode", "")),
                "source_file_exists_after_fix": "true" if source_exists else "false",
                "alignment_status_after_fix": str(row.get("alignment_status", "")),
                "resolution_outcome": resolution_outcome,
                "notes": "resolved via generalized path matching" if source_exists else "no source asset located",
            }
        )

    _write_csv(
        OUT_REGRESSION_PATH_FIXES_CSV,
        rows,
        [
            "page_id",
            "original_source_file",
            "resolved_source_file",
            "source_file_resolution_mode",
            "original_local_pdf_path",
            "resolved_local_pdf_path",
            "local_pdf_resolution_mode",
            "original_local_image_path",
            "resolved_local_image_path",
            "local_image_resolution_mode",
            "source_file_exists_after_fix",
            "alignment_status_after_fix",
            "resolution_outcome",
            "notes",
        ],
    )
    return rows


def _write_regression_annotation_conversion_fixes(audit_rows: list[dict[str, Any]]) -> None:
    blocker_rows = _load_regression_blocker_rows()
    unsupported_ids = {
        str(row.get("page_id", "")).strip()
        for row in blocker_rows
        if str(row.get("exact_exclusion_reason", "")).strip() == "unsupported_annotation_format"
    }

    by_page = {
        str(row.get("page_id", "")).strip(): row
        for row in audit_rows
        if str(row.get("split_kind", "")).strip() == "regression"
    }

    recovered = 0
    still_unsupported = 0
    moved_to_other_failure = 0
    details: list[str] = []

    for page_id in sorted(unsupported_ids):
        row = by_page.get(page_id, {})
        status = str(row.get("alignment_status", ""))
        fmt = str(row.get("annotation_format", ""))
        source = str(row.get("gold_text_source", ""))
        if status in {"aligned", "safely_auto_fixed"}:
            recovered += 1
        elif status == "unsupported_annotation_format":
            still_unsupported += 1
        else:
            moved_to_other_failure += 1
        details.append(f"- {page_id}: status={status or 'missing'} format={fmt or 'missing'} source={source or 'missing'}")

    lines = [
        "# Regression 26 Annotation Conversion Fixes",
        "",
        "## Converter updates",
        "- Added local gold converter support for plain text, CSV/TSV row formats, JSON record formats, and legacy manifest TSV rows.",
        "- Extraction is deterministic and page-targeted using page_id/pdf_name/page matching.",
        "- Converter does not modify source gold files and emits explicit conversion warnings on malformed/missing rows.",
        "",
        "## Regression unsupported-format recovery",
        f"- baseline_unsupported_records: {len(unsupported_ids)}",
        f"- recovered_after_converter: {recovered}",
        f"- still_unsupported: {still_unsupported}",
        f"- shifted_to_other_data_quality_failure: {moved_to_other_failure}",
        "",
        "## Per-record status",
    ]
    lines.extend(details or ["- none"])

    OUT_REGRESSION_CONVERSION_FIXES_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _choose_next_experiment(
    *,
    audited_count: int,
    data_quality_failures_count: int,
    verified_failure_rows: list[dict[str, Any]],
) -> tuple[int, str, str]:
    unverified_ratio = (data_quality_failures_count / audited_count) if audited_count else 1.0

    if unverified_ratio > 0.25:
        return (
            7,
            "Expand/fix gold data further before any OCR tuning",
            "A large share of records remains unverified; alignment/data quality is still the dominant blocker.",
        )

    counts = Counter(str(row.get("failure_category", "")) for row in verified_failure_rows)
    top_category = counts.most_common(1)[0][0] if counts else "unknown"

    if top_category == "bad_layout":
        return (
            2,
            "Switch to layout detection / reading-order reconstruction",
            "Verified failures are dominated by layout and ordering problems.",
        )

    if top_category in {"low_resolution_or_noisy_scan", "preprocessing_needed"}:
        noisy = counts.get("low_resolution_or_noisy_scan", 0)
        preproc = counts.get("preprocessing_needed", 0)
        if noisy >= preproc:
            return (
                1,
                "Continue preprocessing profiles and ablation testing",
                "Verified failures are mainly scan-quality driven.",
            )
        return (
            3,
            "Switch to render/DPI strategy",
            "Verified failures suggest rendering resolution is a stronger lever than profile tweaks.",
        )

    if top_category in {"text_layer_failure", "suspicious_short_output", "OCR_timeout"}:
        return (
            4,
            "Switch to text-layer/rerun gating",
            "Verified failures are dominated by fallback/gating behaviors and short outputs.",
        )

    if top_category in {"suspicious_junk_output", "unsupported_language_or_script"}:
        return (
            5,
            "Switch to postprocessing/language adapters",
            "Verified failures indicate weak token quality/language adaptation on otherwise processed pages.",
        )

    if top_category in {"OCR_engine_failure", "render_failure"}:
        return (
            6,
            "Switch to engine/model evaluation",
            "Verified failures remain high despite usable alignment and routing.",
        )

    return (
        7,
        "Expand/fix gold data further before any OCR tuning",
        "Evidence remains mixed; prioritize stronger data trust before model-level tuning.",
    )


def _write_next_experiment_md(
    *,
    option_number: int,
    option_label: str,
    rationale: str,
    audited_count: int,
    data_quality_failures_count: int,
    verified_failure_rows: list[dict[str, Any]],
) -> None:
    counts = Counter(str(row.get("failure_category", "")) for row in verified_failure_rows)

    lines = [
        "# Next Experiment After Alignment",
        "",
        "## Decision",
        f"- Option: {option_number}",
        f"- Recommendation: {option_label}",
        "",
        "## Evidence",
        f"- audited_records: {audited_count}",
        f"- data_quality_failures: {data_quality_failures_count}",
        f"- verified_failure_rows: {len(verified_failure_rows)}",
        f"- rationale: {rationale}",
        "",
        "## Verified failure category counts",
    ]

    for category, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {category}: {count}")

    lines.extend(
        [
            "",
            "## Guardrail",
            "- No model training or promotion should proceed while alignment reliability is the dominant blocker.",
        ]
    )

    OUT_NEXT_EXPERIMENT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_report(
    *,
    audit_rows: list[dict[str, Any]],
    auto_fix_rows: list[dict[str, Any]],
    correction_queue_rows: list[dict[str, Any]],
    verified_split_summary: dict[str, Any],
    verified_metrics_rows: list[dict[str, Any]],
    verified_failure_rows: list[dict[str, Any]],
    data_quality_failures_rows: list[dict[str, Any]],
    option_number: int,
    option_label: str,
    option_rationale: str,
) -> None:
    status_counts = Counter(str(row.get("alignment_status", "")) for row in audit_rows)
    failure_counts = Counter(str(row.get("failure_category", "")) for row in verified_failure_rows)

    lines = [
        "# Ground-Truth Alignment and Verified Evaluation Report",
        "",
        "## 1. Executive summary",
        "- Ground-truth alignment was audited before further OCR tuning.",
        "- Alignment-verified subsets were created and re-evaluated.",
        "- Preprocessing ablation was rerun only on alignment-verified difficult pages.",
        "",
        "## 2. Why alignment was audited",
        "- Prior diagnostics showed zero CER/WER movement despite input/output changes, indicating data-alignment risk.",
        "",
        "## 3. Alignment audit methodology",
        "- Compared manifest identifiers, page keys, paths, annotation conversion, and multilingual-safe text similarity signals.",
        "- Routed ambiguous records to manual review instead of blind scoring.",
        "",
        "## 4. Alignment status counts",
    ]

    for status, count in sorted(status_counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {status}: {count}")

    lines.extend(
        [
            "",
            "## 5. Common alignment problems found",
            "- See reports/ground_truth_alignment_audit.md and reports/ground_truth_correction_queue.md for detailed breakdown.",
            "",
            "## 6. Automatic fixes applied",
            f"- auto_fix_events: {len(auto_fix_rows)}",
            "- Safe fixes include Unicode/whitespace normalization and unambiguous key/path normalization.",
            "",
            "## 7. Manual correction queue summary",
            f"- queue_items: {len(correction_queue_rows)}",
            f"- manual_review_required: {sum(1 for r in correction_queue_rows if _safe_bool(r.get('requires_manual_review')))}",
            "",
            "## 8. Alignment-verified subset composition",
            f"- smoke verified: {verified_split_summary['smoke']['verified']} / {verified_split_summary['smoke']['original']}",
            f"- validation verified: {verified_split_summary['validation']['verified']} / {verified_split_summary['validation']['original']}",
            f"- regression verified: {verified_split_summary['regression']['verified']} / {verified_split_summary['regression']['original']}",
            "",
            "## 9. Evaluation on verified records",
        ]
    )

    for row in verified_metrics_rows:
        if str(row.get("status", "")) != "measured":
            continue
        lines.append(
            f"- {row['evaluation_split']}: CER={_safe_float(row['cer_mean']):.6f} WER={_safe_float(row['wer_mean']):.6f} failed={_safe_float(row['failed_rate']):.6f} empty={_safe_float(row['empty_rate']):.6f}"
        )

    lines.extend(
        [
            "",
            "## 10. Comparison to previous unverified metrics",
            "- Comparison is documented in reports/alignment_verified_evaluation.md.",
            "- Quality claims are made only on alignment-verified records.",
            "",
            "## 11. Preprocessing ablation on verified records",
            "- See reports/preprocessing_verified_ablation_report.md and reports/preprocessing_verified_ablation_metrics.csv.",
            "",
            "## 12. Updated failure taxonomy",
        ]
    )

    for category, count in sorted(failure_counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {category}: {count}")

    lines.extend(
        [
            "",
            "## 13. Data-quality failures vs OCR failures",
            f"- data_quality_failures: {len(data_quality_failures_rows)}",
            f"- verified_ocr_failure_rows: {len(verified_failure_rows)}",
            "",
            "## 14. Readiness impact",
            "- Readiness claims now depend on verified subsets and explicit exclusion logs.",
            "",
            "## 15. Recommended next experiment",
            f"- Option {option_number}: {option_label}",
            f"- Rationale: {option_rationale}",
            "",
            "## 16. Remaining blockers to private beta",
            "- Complete correction queue triage for high-severity alignment issues.",
            "- Expand verified difficult-page coverage for stronger experiment power.",
            "",
            "## 17. Remaining blockers to production",
            "- Keep multilingual benchmark alignment trust high over time with recurring audits.",
            "- Do not promote OCR/preprocessing changes from unverified records.",
        ]
    )

    OUT_FINAL_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_jsonl(MANIFEST_PATH)
    manifest_by_page_id = {str(row.get("page_id", "")): row for row in manifest_rows}

    # Stage 1 + Stage 2 + Stage 4
    audit_rows, auto_fix_rows = _build_audit_rows(manifest_rows)

    audit_fieldnames = [
        "record_id",
        "source_split_label",
        "split_kind",
        "dataset_id",
        "document_id",
        "page_id",
        "original_page_id",
        "resolved_page_id",
        "normalized_page_key",
        "source_dataset",
        "source_file",
        "source_file_original",
        "source_file_resolution_mode",
        "source_filename",
        "pdf_name",
        "page_reference",
        "page_index",
        "local_image_path",
        "local_image_path_original",
        "local_image_resolution_mode",
        "local_pdf_path",
        "local_pdf_path_original",
        "local_pdf_resolution_mode",
        "ground_truth_text_path",
        "ground_truth_text_path_original",
        "ground_truth_text_resolution_mode",
        "ground_truth_layout_path",
        "ground_truth_layout_path_original",
        "ground_truth_layout_resolution_mode",
        "annotation_format",
        "manifest_split",
        "split_entry_split",
        "source_file_exists",
        "ground_truth_exists",
        "gold_char_count",
        "ocr_char_count",
        "gold_ocr_length_ratio",
        "char_overlap_ratio",
        "token_overlap_ratio",
        "unicode_similarity",
        "lcs_ratio",
        "rough_edit_distance_ratio",
        "gold_is_empty",
        "ocr_is_empty",
        "gold_mostly_markup",
        "ocr_mostly_junk",
        "layout_text_char_count",
        "layout_text_similarity",
        "layout_line_order_warning",
        "page_reference_consistent",
        "pdf_name_consistent",
        "filename_page_consistent",
        "manifest_split_consistent",
        "duplicate_normalized_key_count",
        "duplicate_ground_truth_path_count",
        "duplicate_image_path_count",
        "multi_record_same_source_image",
        "duplicate_key_warning",
        "source_file_exists",
        "ground_truth_exists",
        "annotation_conversion_warning",
        "alignment_score",
        "alignment_status",
        "alignment_warnings",
        "safe_to_use_for_scoring",
        "requires_manual_review",
        "auto_fix_applied",
        "gold_text_source",
        "run_resolution_mode",
    ]

    # Deduplicate accidental repeated field names while preserving order.
    unique_fields: list[str] = []
    seen_fields: set[str] = set()
    for field in audit_fieldnames:
        if field in seen_fields:
            continue
        seen_fields.add(field)
        unique_fields.append(field)

    _write_csv(OUT_AUDIT_CSV, audit_rows, unique_fields)
    _write_audit_markdown(audit_rows)

    _write_csv(
        OUT_AUTO_FIXES_CSV,
        auto_fix_rows,
        ["record_id", "fix_type", "before", "after", "confidence", "reason", "applied"],
    )

    # Stage 3
    correction_queue_rows = _build_correction_queue(audit_rows)
    _write_csv(
        OUT_QUEUE_CSV,
        correction_queue_rows,
        [
            "record_id",
            "source_split_label",
            "split_kind",
            "dataset_id",
            "document_id",
            "page_id",
            "normalized_page_key",
            "source_dataset",
            "local_image_path",
            "local_pdf_path",
            "ground_truth_text_path",
            "ground_truth_layout_path",
            "alignment_status",
            "alignment_score",
            "severity",
            "severity_score",
            "suspected_root_cause",
            "recommended_fix",
            "safe_to_use_for_scoring",
            "requires_manual_review",
        ],
    )
    _write_correction_queue_markdown(correction_queue_rows)

    # Stage 5
    verified_entries, verified_split_summary = _build_verified_splits(audit_rows)

    audit_by_split_page: dict[tuple[str, str], dict[str, Any]] = {}
    for row in audit_rows:
        split_kind = str(row.get("split_kind", ""))
        page_id = str(row.get("page_id", ""))
        if split_kind and page_id:
            audit_by_split_page[(split_kind, page_id)] = row

    # Stage 6
    eval_results: dict[str, VerifiedEvalResult] = {}
    eval_plan = [
        ("alignment_verified_smoke", "smoke", verified_entries.get("smoke", [])),
        ("alignment_verified_validation", "validation", verified_entries.get("validation", [])),
        ("alignment_verified_regression", "regression", verified_entries.get("regression", [])),
    ]

    for split_label, split_kind, entries in eval_plan:
        eval_results[split_label] = _evaluate_verified_split(
            split_label=split_label,
            split_kind=split_kind,
            split_entries=entries,
            manifest_by_page_id=manifest_by_page_id,
            audit_by_split_page=audit_by_split_page,
        )

    verified_metric_rows = [_summarize_verified_result(result) for result in eval_results.values()]

    # Add continuity-only row from original unverified regression_26.
    old_summary = _load_old_unverified_summary()
    old_reg = old_summary.get("regression_26", {})
    if old_reg:
        verified_metric_rows.append(
            {
                "evaluation_split": "regression_26_continuity_unverified",
                "status": "continuity_only",
                "matched_pages": _safe_int(old_reg.get("matched_pages")),
                "gold_pages": _safe_int(old_reg.get("gold_pages")),
                "ocr_pages": _safe_int(old_reg.get("ocr_pages")),
                "cer_mean": _safe_float(old_reg.get("cer_mean")),
                "cer_p50": _safe_float(old_reg.get("cer_p50")),
                "cer_p90": _safe_float(old_reg.get("cer_p90")),
                "wer_mean": _safe_float(old_reg.get("wer_mean")),
                "wer_p50": _safe_float(old_reg.get("wer_p50")),
                "wer_p90": _safe_float(old_reg.get("wer_p90")),
                "failed_rate": _safe_float(old_reg.get("failed_page_rate")),
                "empty_rate": _safe_float(old_reg.get("empty_output_rate")),
                "malformed_row_count": _safe_int(old_reg.get("malformed_row_count")),
                "provenance_mismatch_count": _safe_int(old_reg.get("provenance_mismatch_count")),
                "runtime_ms_mean": _safe_float(old_reg.get("runtime_ms_mean")),
                "runtime_ms_median": _safe_float(old_reg.get("runtime_ms_p50")),
                "runtime_ms_p90": _safe_float(old_reg.get("runtime_ms_p90")),
                "runtime_ms_p95": _safe_float(old_reg.get("runtime_ms_p95")),
                "notes": "continuity_only_unverified",
            }
        )

    _write_csv(
        OUT_VERIFIED_METRICS_CSV,
        verified_metric_rows,
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
            "failed_rate",
            "empty_rate",
            "malformed_row_count",
            "provenance_mismatch_count",
            "runtime_ms_mean",
            "runtime_ms_median",
            "runtime_ms_p90",
            "runtime_ms_p95",
            "notes",
        ],
    )

    per_dataset_rows: list[dict[str, Any]] = []
    per_language_rows: list[dict[str, Any]] = []
    per_layout_rows: list[dict[str, Any]] = []

    for split_label, result in eval_results.items():
        for row in result.per_dataset_rows:
            per_dataset_rows.append({"evaluation_split": split_label, **row})
        for row in result.per_language_rows:
            per_language_rows.append({"evaluation_split": split_label, **row})
        for row in result.per_layout_rows:
            per_layout_rows.append({"evaluation_split": split_label, **row})

    if per_dataset_rows:
        _write_csv(OUT_VERIFIED_PER_DATASET_CSV, per_dataset_rows, list(per_dataset_rows[0].keys()))
    else:
        _write_csv(OUT_VERIFIED_PER_DATASET_CSV, [], ["evaluation_split"])

    if per_language_rows:
        _write_csv(OUT_VERIFIED_PER_LANGUAGE_CSV, per_language_rows, list(per_language_rows[0].keys()))
    else:
        _write_csv(OUT_VERIFIED_PER_LANGUAGE_CSV, [], ["evaluation_split"])

    if per_layout_rows:
        _write_csv(OUT_VERIFIED_PER_LAYOUT_CSV, per_layout_rows, list(per_layout_rows[0].keys()))
    else:
        _write_csv(OUT_VERIFIED_PER_LAYOUT_CSV, [], ["evaluation_split"])

    _write_alignment_verified_evaluation_md(verified_metric_rows, verified_split_summary, old_summary)

    # Stage 7
    verified_preproc_rows, preproc_context = _run_verified_preprocessing_ablation(
        eval_results=eval_results,
        audit_by_split_page=audit_by_split_page,
        manifest_by_page_id=manifest_by_page_id,
    )
    _write_verified_preprocessing_report(verified_preproc_rows, preproc_context)

    # Stage 8
    verified_failure_rows = _build_alignment_verified_failure_taxonomy(eval_results)
    _write_csv(
        OUT_VERIFIED_FAILURE_CSV,
        verified_failure_rows,
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
            "final_output_source",
            "best_available_output_length",
            "CER",
            "WER",
            "recommended_next_fix",
        ],
    )
    _write_alignment_verified_failure_markdown(verified_failure_rows)

    data_quality_failures_rows = _build_data_quality_failures(audit_rows)
    _write_csv(
        OUT_DATA_QUALITY_FAILURES_CSV,
        data_quality_failures_rows,
        [
            "record_id",
            "source_split_label",
            "split_kind",
            "dataset_id",
            "document_id",
            "page_id",
            "alignment_status",
            "severity",
            "severity_score",
            "suspected_root_cause",
            "recommended_fix",
            "safe_to_use_for_scoring",
            "requires_manual_review",
        ],
    )
    _write_data_quality_failures_markdown(data_quality_failures_rows)

    # Track A2 + A3 reporting
    _write_regression_path_resolution_fixes(audit_rows)
    _write_regression_annotation_conversion_fixes(audit_rows)

    # Stage 9
    option_number, option_label, option_rationale = _choose_next_experiment(
        audited_count=len(audit_rows),
        data_quality_failures_count=len(data_quality_failures_rows),
        verified_failure_rows=verified_failure_rows,
    )
    _write_next_experiment_md(
        option_number=option_number,
        option_label=option_label,
        rationale=option_rationale,
        audited_count=len(audit_rows),
        data_quality_failures_count=len(data_quality_failures_rows),
        verified_failure_rows=verified_failure_rows,
    )

    # Stage 10
    _write_final_report(
        audit_rows=audit_rows,
        auto_fix_rows=auto_fix_rows,
        correction_queue_rows=correction_queue_rows,
        verified_split_summary=verified_split_summary,
        verified_metrics_rows=verified_metric_rows,
        verified_failure_rows=verified_failure_rows,
        data_quality_failures_rows=data_quality_failures_rows,
        option_number=option_number,
        option_label=option_label,
        option_rationale=option_rationale,
    )

    print(
        json.dumps(
            {
                "ground_truth_alignment_audit_csv": str(OUT_AUDIT_CSV.relative_to(ROOT)),
                "ground_truth_correction_queue_csv": str(OUT_QUEUE_CSV.relative_to(ROOT)),
                "ground_truth_auto_fixes_csv": str(OUT_AUTO_FIXES_CSV.relative_to(ROOT)),
                "alignment_verified_smoke": str(VERIFIED_SPLIT_SMOKE.relative_to(ROOT)),
                "alignment_verified_validation": str(VERIFIED_SPLIT_VALIDATION.relative_to(ROOT)),
                "alignment_verified_regression": str(VERIFIED_SPLIT_REGRESSION.relative_to(ROOT)),
                "alignment_verified_metrics_csv": str(OUT_VERIFIED_METRICS_CSV.relative_to(ROOT)),
                "preprocessing_verified_ablation_csv": str(OUT_PREPROC_VERIFIED_CSV.relative_to(ROOT)),
                "alignment_verified_failure_taxonomy_csv": str(OUT_VERIFIED_FAILURE_CSV.relative_to(ROOT)),
                "data_quality_failures_csv": str(OUT_DATA_QUALITY_FAILURES_CSV.relative_to(ROOT)),
                "next_experiment_after_alignment_md": str(OUT_NEXT_EXPERIMENT_MD.relative_to(ROOT)),
                "final_report_md": str(OUT_FINAL_REPORT_MD.relative_to(ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
