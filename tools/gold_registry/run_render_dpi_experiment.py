#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from production.ensemble_ocr import FortifiedOCREnsemble
from production.preprocessing_profiles import PREPROCESSING_PROFILES, PROFILE_AUTO, PreprocessingProfile

REGISTRY_DIR = ROOT / "data" / "gold_registry"
SPLITS_DIR = REGISTRY_DIR / "splits"
REPORTS_DIR = ROOT / "reports"
EVAL_ROOT = REPORTS_DIR / "real_gold_eval_runs"
PROFILE_PATH = ROOT / "profiles" / "akkadian_strict.json"

AUDIT_CSV = REPORTS_DIR / "ground_truth_alignment_audit.csv"
VERIFIED_SPLIT_SMOKE = SPLITS_DIR / "alignment_verified_smoke.jsonl"
VERIFIED_SPLIT_VALIDATION = SPLITS_DIR / "alignment_verified_validation.jsonl"
VERIFIED_SPLIT_REGRESSION = SPLITS_DIR / "alignment_verified_regression.jsonl"
REGRESSION_SOURCE_SPLIT = SPLITS_DIR / "regression_26.jsonl"

OUT_RENDER_DPI_MATRIX_CSV = REPORTS_DIR / "render_dpi_benchmark_matrix.csv"
OUT_RENDER_DPI_PER_PAGE_CSV = REPORTS_DIR / "render_dpi_per_page_metrics.csv"
OUT_RENDER_DPI_REPORT_MD = REPORTS_DIR / "render_dpi_experiment_report.md"
OUT_RENDER_DPI_PAGE_ANALYSIS_CSV = REPORTS_DIR / "render_dpi_page_level_analysis.csv"
OUT_RENDER_DPI_PAGE_ANALYSIS_MD = REPORTS_DIR / "render_dpi_page_level_analysis.md"
OUT_RENDER_DPI_PROMOTION_MD = REPORTS_DIR / "render_dpi_promotion_decision.md"
OUT_RENDER_DPI_STRATEGY_MD = REPORTS_DIR / "render_dpi_strategy_report.md"

PADDLE_ONLY_PROFILE_PATH = REPORTS_DIR / "alignment_audit_artifacts" / "paddle_only_profile.json"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if not text:
            return default
        return int(float(text))
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on", "t"}


def _format_float(value: float) -> str:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * float(q)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_values[lo])
    weight = idx - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * weight)


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
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def _cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(list(reference), list(hypothesis)) / max(len(reference), 1)


def _wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / max(len(ref_words), 1)


def _extract_page_number_from_id(text: str) -> int:
    matches = re.findall(r"_page_(\d+)", str(text or ""))
    if matches:
        return int(matches[-1])
    return 1


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
    routing["layout_first_engines"] = ["paddle"]
    base["routing"] = routing

    PADDLE_ONLY_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PADDLE_ONLY_PROFILE_PATH.write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return PADDLE_ONLY_PROFILE_PATH


def _build_page_attempts(page_index: int) -> list[int]:
    # Never silently score page 0 when a different page was requested.
    return [max(page_index, 0)]


def _register_render_profiles() -> list[dict[str, Any]]:
    additions: dict[str, PreprocessingProfile] = {
        "render_300_dpi": PreprocessingProfile(
            name="render_300_dpi",
            render_dpi=300,
            preprocessing_overrides={
                "enable_denoise": False,
                "enable_adaptive_threshold": False,
                "enable_morphology": False,
                "deskew": True,
                "contrast_factor": 1.15,
                "profile_primary_variant": "original",
                "profile_variant_order": ["original", "contrast", "sharpen"],
            },
            description="Low-cost 300 DPI render with conservative preprocessing.",
        ),
        "render_400_dpi": PreprocessingProfile(
            name="render_400_dpi",
            render_dpi=400,
            preprocessing_overrides={
                "enable_denoise": False,
                "enable_adaptive_threshold": False,
                "enable_morphology": False,
                "deskew": True,
                "contrast_factor": 1.3,
                "preserve_diacritics": True,
                "profile_primary_variant": "contrast",
                "profile_variant_order": ["contrast", "original", "sharpen"],
            },
            description="Mid-range high DPI render balancing detail and runtime.",
        ),
        "render_600_dpi": PreprocessingProfile(
            name="render_600_dpi",
            render_dpi=600,
            preprocessing_overrides={
                "enable_denoise": False,
                "enable_adaptive_threshold": False,
                "enable_morphology": False,
                "deskew": True,
                "contrast_factor": 1.2,
                "preserve_diacritics": True,
                "avoid_aggressive_binarization": True,
                "profile_primary_variant": "original",
                "profile_variant_order": ["original", "contrast", "sharpen"],
            },
            description="Aggressive high DPI render for tiny text and marginal glyphs.",
        ),
    }
    for key, profile in additions.items():
        PREPROCESSING_PROFILES[key] = profile

    return [
        {
            "profile_id": "current_default_render",
            "internal_profile": PROFILE_AUTO,
            "render_dpi": "auto",
            "render_scale": "auto",
            "apply_mode": "fixed_all_pages",
            "max_runtime_budget_ms": 6000,
            "memory_risk": "low",
            "expected_benefit": "baseline for comparison",
            "expected_cost": "none",
        },
        {
            "profile_id": "render_300_dpi",
            "internal_profile": "render_300_dpi",
            "render_dpi": 300,
            "render_scale": _format_float(300 / 72.0),
            "apply_mode": "fixed_all_pages",
            "max_runtime_budget_ms": 6000,
            "memory_risk": "low",
            "expected_benefit": "stable baseline-like quality",
            "expected_cost": "minimal",
        },
        {
            "profile_id": "render_400_dpi",
            "internal_profile": "render_400_dpi",
            "render_dpi": 400,
            "render_scale": _format_float(400 / 72.0),
            "apply_mode": "fixed_all_pages",
            "max_runtime_budget_ms": 8000,
            "memory_risk": "moderate",
            "expected_benefit": "better small text legibility",
            "expected_cost": "higher latency",
        },
        {
            "profile_id": "render_600_dpi",
            "internal_profile": "render_600_dpi",
            "render_dpi": 600,
            "render_scale": _format_float(600 / 72.0),
            "apply_mode": "fixed_all_pages",
            "max_runtime_budget_ms": 11000,
            "memory_risk": "high",
            "expected_benefit": "best chance on tiny/low-quality text",
            "expected_cost": "high latency and memory",
        },
        {
            "profile_id": "adaptive_high_dpi_for_small_or_low_quality_pages",
            "internal_profile": "adaptive",
            "render_dpi": "300_or_600",
            "render_scale": "4.167_or_8.333",
            "apply_mode": "adaptive_subset",
            "max_runtime_budget_ms": 9000,
            "memory_risk": "moderate",
            "expected_benefit": "targeted quality lift on hard pages",
            "expected_cost": "moderate control overhead",
            "low_profile": "render_300_dpi",
            "high_profile": "render_600_dpi",
        },
        {
            "profile_id": "high_dpi_only_for_fallback_pages",
            "internal_profile": "fallback_only",
            "render_dpi": "default_or_600",
            "render_scale": "auto_or_8.333",
            "apply_mode": "fallback_only",
            "max_runtime_budget_ms": 10000,
            "memory_risk": "moderate",
            "expected_benefit": "rescue failed or empty pages",
            "expected_cost": "minimal on healthy pages",
            "high_profile": "render_600_dpi",
        },
    ]


def _load_validation_per_page_metrics() -> dict[str, dict[str, Any]]:
    per_page = EVAL_ROOT / "alignment_verified_validation" / "eval" / "per_page_metrics.csv"
    rows = _read_csv(per_page)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        page_id = str(row.get("page_reference", row.get("page_key", ""))).strip()
        if page_id:
            out[page_id] = row
    return out


def _select_pages_for_experiment(audit_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_entries = {
        "smoke": _read_jsonl(VERIFIED_SPLIT_SMOKE),
        "validation": _read_jsonl(VERIFIED_SPLIT_VALIDATION),
        "regression": _read_jsonl(VERIFIED_SPLIT_REGRESSION),
    }

    audit_by_split_page: dict[tuple[str, str], dict[str, Any]] = {}
    for row in audit_rows:
        split = str(row.get("split_kind", "")).strip()
        page_id = str(row.get("page_id", "")).strip()
        resolved = str(row.get("resolved_page_id", "")).strip()
        if split and page_id:
            audit_by_split_page[(split, page_id)] = row
        if split and resolved:
            audit_by_split_page[(split, resolved)] = row

    validation_metrics = _load_validation_per_page_metrics()
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    validation_candidates: list[dict[str, Any]] = []

    for split_kind, entries in split_entries.items():
        for entry in entries:
            if str(entry.get("status", "")).strip().lower() != "available":
                excluded.append({"split_kind": split_kind, "page_id": str(entry.get("page_id", "")), "reason": "split_status_not_available"})
                continue
            if not _safe_bool(entry.get("safe_to_use_for_scoring", True)):
                excluded.append({"split_kind": split_kind, "page_id": str(entry.get("page_id", "")), "reason": "not_safe_for_scoring"})
                continue

            page_id = str(entry.get("resolved_page_id") or entry.get("page_id") or "").strip()
            if not page_id:
                excluded.append({"split_kind": split_kind, "page_id": "", "reason": "missing_page_id"})
                continue

            audit = audit_by_split_page.get((split_kind, page_id))
            if audit is None:
                excluded.append({"split_kind": split_kind, "page_id": page_id, "reason": "missing_audit_row"})
                continue

            status = str(audit.get("alignment_status", "")).strip()
            if status not in {"aligned", "safely_auto_fixed"}:
                excluded.append({"split_kind": split_kind, "page_id": page_id, "reason": f"not_alignment_verified:{status}"})
                continue

            pdf_rel = str(audit.get("local_pdf_path", "")).strip()
            pdf_path = ROOT / pdf_rel if pdf_rel else Path()
            if not pdf_rel or not pdf_path.exists():
                excluded.append({"split_kind": split_kind, "page_id": page_id, "reason": "missing_local_pdf"})
                continue

            gold_text = str(audit.get("gold_text_effective", "")).strip()
            if not gold_text:
                gt_rel = str(audit.get("ground_truth_text_path", "")).strip()
                gt_path = ROOT / gt_rel if gt_rel else Path()
                if gt_rel and gt_path.exists():
                    gold_text = gt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not gold_text:
                excluded.append({"split_kind": split_kind, "page_id": page_id, "reason": "missing_gold_text"})
                continue

            page_index = _safe_int(audit.get("page_index"), -1)
            if page_index < 0:
                page_index = _extract_page_number_from_id(page_id) - 1
            page_index = max(page_index, 0)

            page_payload = {
                "split_kind": split_kind,
                "selection_group": split_kind,
                "page_id": page_id,
                "dataset_id": str(audit.get("dataset_id", "unknown") or "unknown"),
                "document_id": str(audit.get("document_id", "unknown") or "unknown"),
                "language_primary": str(audit.get("language_primary", "unknown") or "unknown"),
                "script_type": str(audit.get("script_type", "unknown") or "unknown"),
                "document_type": str(audit.get("document_type", "unknown") or "unknown"),
                "layout_type": str(audit.get("layout_type", "unknown") or "unknown"),
                "scan_quality": str(audit.get("scan_quality", "unknown") or "unknown"),
                "expected_difficulty": str(audit.get("expected_difficulty", "unknown") or "unknown"),
                "pdf_path": str(pdf_path),
                "page_index": page_index,
                "gold_text": gold_text,
            }

            if split_kind == "validation":
                eval_row = validation_metrics.get(page_id, {})
                cer = _safe_float(eval_row.get("cer"), 0.0)
                quality = str(page_payload["scan_quality"]).lower()
                layout = str(page_payload["layout_type"]).lower()
                difficult = (
                    cer >= 1.0
                    or quality in {"mixed", "noisy_scan", "low_quality", "noisy"}
                    or layout in {"multi_column", "form_layout", "semi_structured"}
                    or str(page_payload["expected_difficulty"]).lower() in {"hard", "very_hard"}
                )
                score = cer
                if quality in {"mixed", "noisy_scan", "low_quality", "noisy"}:
                    score += 0.35
                if layout in {"multi_column", "form_layout", "semi_structured"}:
                    score += 0.25
                if str(page_payload["expected_difficulty"]).lower() in {"hard", "very_hard"}:
                    score += 0.2
                page_payload["difficulty_score"] = score
                page_payload["difficult_candidate"] = difficult
                validation_candidates.append(page_payload)
            else:
                selected.append(page_payload)

    validation_total = len(validation_candidates)
    full_validation_threshold = 60
    difficult_subset_limit = 12
    full_validation_included = validation_total <= full_validation_threshold

    if full_validation_included:
        for row in validation_candidates:
            row["selection_group"] = "validation_full"
        selected.extend(validation_candidates)
    else:
        difficult = [row for row in validation_candidates if _safe_bool(row.get("difficult_candidate"))]
        if len(difficult) < difficult_subset_limit:
            remaining = [row for row in validation_candidates if row not in difficult]
            remaining.sort(key=lambda r: _safe_float(r.get("difficulty_score"), 0.0), reverse=True)
            difficult.extend(remaining[: max(0, difficult_subset_limit - len(difficult))])
        difficult.sort(key=lambda r: _safe_float(r.get("difficulty_score"), 0.0), reverse=True)
        chosen = difficult[:difficult_subset_limit]
        for row in chosen:
            row["selection_group"] = "validation_difficult_subset"
        selected.extend(chosen)

    context = {
        "selected_total": len(selected),
        "selected_smoke": sum(1 for row in selected if row.get("split_kind") == "smoke"),
        "selected_regression": sum(1 for row in selected if row.get("split_kind") == "regression"),
        "selected_validation": sum(1 for row in selected if row.get("split_kind") == "validation"),
        "selected_validation_difficult_subset": sum(
            1 for row in selected if row.get("selection_group") == "validation_difficult_subset"
        ),
        "validation_verified_total": validation_total,
        "full_validation_included": full_validation_included,
        "excluded_count": len(excluded),
        "excluded_examples": excluded[:30],
    }
    return selected, context


def _is_timeout_like(meta: dict[str, Any]) -> bool:
    reason = str(meta.get("failure_reason", "")).lower()
    if "timeout" in reason:
        return True
    engine_statuses = meta.get("engine_page_statuses")
    if isinstance(engine_statuses, dict):
        for value in engine_statuses.values():
            if isinstance(value, dict) and "timed_out" in str(value.get("status", "")).lower():
                return True
    return False


def _is_memory_warning(meta: dict[str, Any], runtime_ms: float, budget_ms: int) -> bool:
    warning = str(meta.get("warning", "")).lower()
    if "memory" in warning:
        return True
    return runtime_ms > max(float(budget_ms), 1.0) * 1.35


def _looks_difficult(page: dict[str, Any]) -> bool:
    quality = str(page.get("scan_quality", "")).lower()
    layout = str(page.get("layout_type", "")).lower()
    difficulty = str(page.get("expected_difficulty", "")).lower()
    if quality in {"mixed", "noisy_scan", "low_quality", "noisy"}:
        return True
    if layout in {"multi_column", "form_layout", "semi_structured"}:
        return True
    return difficulty in {"hard", "very_hard"}


def _run_render_experiment(
    pages: list[dict[str, Any]],
    profile_defs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profile_path = _build_paddle_only_profile()
    ensemble = FortifiedOCREnsemble(profile_path=str(profile_path))

    rows: list[dict[str, Any]] = []
    default_cache: dict[str, dict[str, Any]] = {}

    def run_extract(page: dict[str, Any], profile_name: str) -> tuple[str, dict[str, Any], float, int]:
        page_index = max(_safe_int(page.get("page_index"), 0), 0)
        attempts = _build_page_attempts(page_index)

        total_runtime = 0.0
        last_text = ""
        last_meta: dict[str, Any] = {}
        applied_index = page_index

        for idx in attempts:
            t0 = time.perf_counter()
            try:
                text, meta = ensemble.extract_page_text(
                    str(page["pdf_path"]),
                    idx,
                    preprocessing_profile=profile_name,
                    diagnostics={},
                    language_hint=str(page.get("language_primary", "unknown")),
                    script_hint=str(page.get("script_type", "unknown")),
                    document_type=str(page.get("document_type", "unknown")),
                    debug_artifacts_dir="",
                    debug_artifact_prefix="",
                )
            except Exception as exc:
                text = ""
                meta = {
                    "error": str(exc),
                    "failure_reason": "exception",
                    "final_output_source": "none",
                    "preprocessing_profile": profile_name,
                }
            runtime_ms = (time.perf_counter() - t0) * 1000.0
            total_runtime += runtime_ms
            last_text = text
            last_meta = meta
            applied_index = idx

            break

        return last_text, last_meta, total_runtime, applied_index

    for i, page in enumerate(pages, start=1):
        if i % 10 == 0:
            print(f"progress: processed {i}/{len(pages)} pages for default profile", file=sys.stderr)

        gold_text = str(page.get("gold_text", ""))

        base_cfg = profile_defs[0]
        base_text, base_meta, base_runtime_ms, applied_index = run_extract(page, str(base_cfg["internal_profile"]))
        base_timeout = _is_timeout_like(base_meta)
        base_empty = not str(base_text).strip()
        base_failed = bool(str(base_meta.get("error", "")).strip()) or (
            bool(str(base_meta.get("failure_reason", "")).strip()) and base_empty
        )

        base_row = {
            "profile_id": str(base_cfg["profile_id"]),
            "profile_internal": str(base_cfg["internal_profile"]),
            "render_dpi": str(base_cfg["render_dpi"]),
            "render_scale": str(base_cfg["render_scale"]),
            "apply_mode": str(base_cfg["apply_mode"]),
            "max_runtime_budget_ms": int(base_cfg["max_runtime_budget_ms"]),
            "memory_risk": str(base_cfg["memory_risk"]),
            "expected_benefit": str(base_cfg["expected_benefit"]),
            "expected_cost": str(base_cfg["expected_cost"]),
            "dataset_id": str(page.get("dataset_id", "unknown")),
            "document_id": str(page.get("document_id", "unknown")),
            "page_id": str(page.get("page_id", "")),
            "split_kind": str(page.get("split_kind", "")),
            "selection_group": str(page.get("selection_group", "")),
            "language_primary": str(page.get("language_primary", "unknown")),
            "document_type": str(page.get("document_type", "unknown")),
            "layout_type": str(page.get("layout_type", "unknown")),
            "scan_quality": str(page.get("scan_quality", "unknown")),
            "pdf_path": str(page.get("pdf_path", "")),
            "requested_page_index": _safe_int(page.get("page_index"), 0),
            "applied_page_index": applied_index,
            "gold_text_length": len(gold_text),
            "output_text_length": len(str(base_text).strip()),
            "CER": _cer(gold_text, str(base_text).strip()),
            "WER": _wer(gold_text, str(base_text).strip()),
            "runtime_ms": base_runtime_ms,
            "failed": "true" if base_failed else "false",
            "empty": "true" if base_empty else "false",
            "timeout_like": "true" if base_timeout else "false",
            "memory_warning": "true" if _is_memory_warning(base_meta, base_runtime_ms, int(base_cfg["max_runtime_budget_ms"])) else "false",
            "malformed_row": "false",
            "failure_reason": str(base_meta.get("failure_reason", "")),
            "final_output_source": str(base_meta.get("final_output_source", "")),
            "profile_applied": str(base_meta.get("preprocessing_profile", "")),
            "effective_render_dpi": _safe_int(base_meta.get("preprocessing_render_dpi"), 0),
            "baseline_reused": "false",
            "high_dpi_applied": "false",
            "notes": "",
        }
        rows.append(base_row)
        default_cache[str(page.get("page_id", ""))] = {
            "text": str(base_text),
            "meta": base_meta,
            "row": base_row,
        }

    for cfg in profile_defs[1:]:
        profile_id = str(cfg["profile_id"])
        print(f"running profile: {profile_id}", file=sys.stderr)
        for i, page in enumerate(pages, start=1):
            if i % 10 == 0:
                print(f"progress: {profile_id} {i}/{len(pages)}", file=sys.stderr)

            gold_text = str(page.get("gold_text", ""))
            page_id = str(page.get("page_id", ""))
            baseline = default_cache.get(page_id, {})
            baseline_row = baseline.get("row", {})
            baseline_text = str(baseline.get("text", ""))
            baseline_meta = dict(baseline.get("meta", {}))

            chosen_profile = str(cfg.get("internal_profile", PROFILE_AUTO))
            high_dpi_applied = False
            baseline_reused = False

            if profile_id == "adaptive_high_dpi_for_small_or_low_quality_pages":
                if _looks_difficult(page):
                    chosen_profile = str(cfg.get("high_profile", "render_600_dpi"))
                    high_dpi_applied = True
                else:
                    chosen_profile = str(cfg.get("low_profile", "render_300_dpi"))
            elif profile_id == "high_dpi_only_for_fallback_pages":
                baseline_bad = _safe_bool(baseline_row.get("failed")) or _safe_bool(baseline_row.get("empty")) or _safe_bool(
                    baseline_row.get("timeout_like")
                )
                if baseline_bad:
                    chosen_profile = str(cfg.get("high_profile", "render_600_dpi"))
                    high_dpi_applied = True
                else:
                    baseline_reused = True

            if baseline_reused:
                text = baseline_text
                meta = baseline_meta
                runtime_ms = 0.0
                applied_index = _safe_int(page.get("page_index"), 0)
                notes = "reused_default_output"
            else:
                text, meta, runtime_ms, applied_index = run_extract(page, chosen_profile)
                notes = ""

            timeout_like = _is_timeout_like(meta)
            empty = not str(text).strip()
            failed = bool(str(meta.get("error", "")).strip()) or (
                bool(str(meta.get("failure_reason", "")).strip()) and empty
            )

            rows.append(
                {
                    "profile_id": profile_id,
                    "profile_internal": str(cfg.get("internal_profile", "")),
                    "render_dpi": str(cfg.get("render_dpi", "")),
                    "render_scale": str(cfg.get("render_scale", "")),
                    "apply_mode": str(cfg.get("apply_mode", "")),
                    "max_runtime_budget_ms": int(cfg.get("max_runtime_budget_ms", 0)),
                    "memory_risk": str(cfg.get("memory_risk", "")),
                    "expected_benefit": str(cfg.get("expected_benefit", "")),
                    "expected_cost": str(cfg.get("expected_cost", "")),
                    "dataset_id": str(page.get("dataset_id", "unknown")),
                    "document_id": str(page.get("document_id", "unknown")),
                    "page_id": page_id,
                    "split_kind": str(page.get("split_kind", "")),
                    "selection_group": str(page.get("selection_group", "")),
                    "language_primary": str(page.get("language_primary", "unknown")),
                    "document_type": str(page.get("document_type", "unknown")),
                    "layout_type": str(page.get("layout_type", "unknown")),
                    "scan_quality": str(page.get("scan_quality", "unknown")),
                    "pdf_path": str(page.get("pdf_path", "")),
                    "requested_page_index": _safe_int(page.get("page_index"), 0),
                    "applied_page_index": applied_index,
                    "gold_text_length": len(gold_text),
                    "output_text_length": len(str(text).strip()),
                    "CER": _cer(gold_text, str(text).strip()),
                    "WER": _wer(gold_text, str(text).strip()),
                    "runtime_ms": runtime_ms,
                    "failed": "true" if failed else "false",
                    "empty": "true" if empty else "false",
                    "timeout_like": "true" if timeout_like else "false",
                    "memory_warning": "true"
                    if _is_memory_warning(meta, runtime_ms, int(cfg.get("max_runtime_budget_ms", 0)))
                    else "false",
                    "malformed_row": "false",
                    "failure_reason": str(meta.get("failure_reason", "")),
                    "final_output_source": str(meta.get("final_output_source", "")),
                    "profile_applied": str(meta.get("preprocessing_profile", "")),
                    "effective_render_dpi": _safe_int(meta.get("preprocessing_render_dpi"), 0),
                    "baseline_reused": "true" if baseline_reused else "false",
                    "high_dpi_applied": "true" if high_dpi_applied else "false",
                    "notes": notes,
                }
            )

    context = {
        "profile_count": len(profile_defs),
        "page_count": len(pages),
    }
    return rows, context


def _build_matrix(rows: list[dict[str, Any]], profile_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_profile.setdefault(str(row.get("profile_id", "")), []).append(row)

    default_rows = by_profile.get("current_default_render", [])
    default_by_page = {str(row.get("page_id", "")): row for row in default_rows}

    matrix_rows: list[dict[str, Any]] = []

    for profile in profile_defs:
        profile_id = str(profile.get("profile_id", ""))
        items = by_profile.get(profile_id, [])
        if not items:
            matrix_rows.append(
                {
                    "profile_id": profile_id,
                    "profile_internal": str(profile.get("internal_profile", "")),
                    "render_dpi": str(profile.get("render_dpi", "")),
                    "render_scale": str(profile.get("render_scale", "")),
                    "apply_mode": str(profile.get("apply_mode", "")),
                    "max_runtime_budget_ms": int(profile.get("max_runtime_budget_ms", 0)),
                    "memory_risk": str(profile.get("memory_risk", "")),
                    "expected_benefit": str(profile.get("expected_benefit", "")),
                    "expected_cost": str(profile.get("expected_cost", "")),
                    "page_count": 0,
                    "CER_mean": 0.0,
                    "CER_median": 0.0,
                    "CER_p90": 0.0,
                    "WER_mean": 0.0,
                    "WER_median": 0.0,
                    "WER_p90": 0.0,
                    "failed_rate": 0.0,
                    "empty_rate": 0.0,
                    "runtime_ms_mean": 0.0,
                    "runtime_ms_median": 0.0,
                    "runtime_ms_p90": 0.0,
                    "runtime_ms_p95": 0.0,
                    "timeout_like_count": 0,
                    "memory_warning_count": 0,
                    "malformed_row_count": 0,
                    "pages_improved_vs_default": 0,
                    "pages_worsened_vs_default": 0,
                    "regression_cer_mean": 0.0,
                    "smoke_cer_mean": 0.0,
                    "difficult_validation_cer_mean": 0.0,
                    "notes": "no measurements",
                }
            )
            continue

        cer_vals = [_safe_float(item.get("CER"), 0.0) for item in items]
        wer_vals = [_safe_float(item.get("WER"), 0.0) for item in items]
        rt_vals = [_safe_float(item.get("runtime_ms"), 0.0) for item in items]

        improved = 0
        worsened = 0
        if profile_id != "current_default_render":
            for item in items:
                page_id = str(item.get("page_id", ""))
                base = default_by_page.get(page_id)
                if base is None:
                    continue
                delta = _safe_float(item.get("CER"), 0.0) - _safe_float(base.get("CER"), 0.0)
                if delta < -1e-12:
                    improved += 1
                elif delta > 1e-12:
                    worsened += 1

        regression_vals = [
            _safe_float(item.get("CER"), 0.0)
            for item in items
            if str(item.get("split_kind", "")).strip() == "regression"
        ]
        smoke_vals = [
            _safe_float(item.get("CER"), 0.0)
            for item in items
            if str(item.get("split_kind", "")).strip() == "smoke"
        ]
        difficult_vals = [
            _safe_float(item.get("CER"), 0.0)
            for item in items
            if str(item.get("selection_group", "")).strip() == "validation_difficult_subset"
        ]

        matrix_rows.append(
            {
                "profile_id": profile_id,
                "profile_internal": str(profile.get("internal_profile", "")),
                "render_dpi": str(profile.get("render_dpi", "")),
                "render_scale": str(profile.get("render_scale", "")),
                "apply_mode": str(profile.get("apply_mode", "")),
                "max_runtime_budget_ms": int(profile.get("max_runtime_budget_ms", 0)),
                "memory_risk": str(profile.get("memory_risk", "")),
                "expected_benefit": str(profile.get("expected_benefit", "")),
                "expected_cost": str(profile.get("expected_cost", "")),
                "page_count": len(items),
                "CER_mean": _mean(cer_vals),
                "CER_median": statistics.median(cer_vals) if cer_vals else 0.0,
                "CER_p90": _quantile(cer_vals, 0.9),
                "WER_mean": _mean(wer_vals),
                "WER_median": statistics.median(wer_vals) if wer_vals else 0.0,
                "WER_p90": _quantile(wer_vals, 0.9),
                "failed_rate": _mean([1.0 if _safe_bool(item.get("failed")) else 0.0 for item in items]),
                "empty_rate": _mean([1.0 if _safe_bool(item.get("empty")) else 0.0 for item in items]),
                "runtime_ms_mean": _mean(rt_vals),
                "runtime_ms_median": statistics.median(rt_vals) if rt_vals else 0.0,
                "runtime_ms_p90": _quantile(rt_vals, 0.9),
                "runtime_ms_p95": _quantile(rt_vals, 0.95),
                "timeout_like_count": sum(1 for item in items if _safe_bool(item.get("timeout_like"))),
                "memory_warning_count": sum(1 for item in items if _safe_bool(item.get("memory_warning"))),
                "malformed_row_count": sum(1 for item in items if _safe_bool(item.get("malformed_row"))),
                "pages_improved_vs_default": improved,
                "pages_worsened_vs_default": worsened,
                "regression_cer_mean": _mean(regression_vals),
                "smoke_cer_mean": _mean(smoke_vals),
                "difficult_validation_cer_mean": _mean(difficult_vals),
                "notes": "",
            }
        )

    return matrix_rows


def _build_page_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    default_rows = {
        str(row.get("page_id", "")): row
        for row in rows
        if str(row.get("profile_id", "")) == "current_default_render"
    }

    analysis: list[dict[str, Any]] = []
    for row in rows:
        profile_id = str(row.get("profile_id", ""))
        if profile_id == "current_default_render":
            continue
        page_id = str(row.get("page_id", ""))
        base = default_rows.get(page_id)
        if base is None:
            continue

        delta_cer = _safe_float(row.get("CER"), 0.0) - _safe_float(base.get("CER"), 0.0)
        delta_wer = _safe_float(row.get("WER"), 0.0) - _safe_float(base.get("WER"), 0.0)
        delta_rt = _safe_float(row.get("runtime_ms"), 0.0) - _safe_float(base.get("runtime_ms"), 0.0)

        suspected_reason = "neutral_or_unclear"
        if _safe_bool(row.get("high_dpi_applied")) and delta_cer < -0.02:
            suspected_reason = "high_dpi_helped_small_or_low_quality_text"
        elif _safe_bool(row.get("high_dpi_applied")) and delta_cer > 0.02:
            suspected_reason = "high_dpi_added_noise_or_over_sharpening"
        elif _safe_bool(row.get("baseline_reused")):
            suspected_reason = "fallback_reused_default_output"
        elif str(row.get("layout_type", "")).strip().lower() in {"multi_column", "form_layout", "semi_structured"} and delta_cer < -0.01:
            suspected_reason = "layout_sensitive_page_improved"

        analysis.append(
            {
                "page_id": page_id,
                "split_kind": str(row.get("split_kind", "")),
                "selection_group": str(row.get("selection_group", "")),
                "dataset_id": str(row.get("dataset_id", "")),
                "document_id": str(row.get("document_id", "")),
                "layout_type": str(row.get("layout_type", "")),
                "profile_id": profile_id,
                "default_CER": _safe_float(base.get("CER"), 0.0),
                "candidate_CER": _safe_float(row.get("CER"), 0.0),
                "delta_CER": delta_cer,
                "default_WER": _safe_float(base.get("WER"), 0.0),
                "candidate_WER": _safe_float(row.get("WER"), 0.0),
                "delta_WER": delta_wer,
                "default_runtime_ms": _safe_float(base.get("runtime_ms"), 0.0),
                "candidate_runtime_ms": _safe_float(row.get("runtime_ms"), 0.0),
                "delta_runtime_ms": delta_rt,
                "high_dpi_applied": str(row.get("high_dpi_applied", "false")),
                "apply_mode": str(row.get("apply_mode", "")),
                "is_improved": "true" if delta_cer < -1e-12 else "false",
                "is_worsened": "true" if delta_cer > 1e-12 else "false",
                "suspected_reason": suspected_reason,
            }
        )

    return analysis


def _aggregate_breakdown(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        profile = str(row.get("profile_id", ""))
        bucket = str(row.get(key, "unknown") or "unknown")
        buckets.setdefault((profile, bucket), []).append(row)

    out: list[dict[str, Any]] = []
    for (profile, bucket), items in sorted(buckets.items()):
        out.append(
            {
                "profile_id": profile,
                key: bucket,
                "count": len(items),
                "CER_mean": _mean([_safe_float(i.get("CER"), 0.0) for i in items]),
                "WER_mean": _mean([_safe_float(i.get("WER"), 0.0) for i in items]),
                "runtime_ms_mean": _mean([_safe_float(i.get("runtime_ms"), 0.0) for i in items]),
                "failed_rate": _mean([1.0 if _safe_bool(i.get("failed")) else 0.0 for i in items]),
            }
        )
    return out


def _choose_promotion_decision(matrix_rows: list[dict[str, Any]]) -> tuple[str, str, list[dict[str, Any]]]:
    by_profile = {str(row.get("profile_id", "")): row for row in matrix_rows}
    default = by_profile.get("current_default_render")
    if default is None:
        return "reject_due_to_insufficient_evidence", "default profile measurements missing", []

    default_cer = _safe_float(default.get("CER_mean"), 0.0)
    default_wer = _safe_float(default.get("WER_mean"), 0.0)
    default_fail = _safe_float(default.get("failed_rate"), 0.0)
    default_empty = _safe_float(default.get("empty_rate"), 0.0)
    default_p95 = _safe_float(default.get("runtime_ms_p95"), 0.0)
    default_regression = _safe_float(default.get("regression_cer_mean"), 0.0)

    assessments: list[dict[str, Any]] = []
    for row in matrix_rows:
        profile_id = str(row.get("profile_id", ""))
        if profile_id == "current_default_render":
            continue

        cer = _safe_float(row.get("CER_mean"), 0.0)
        wer = _safe_float(row.get("WER_mean"), 0.0)
        fail = _safe_float(row.get("failed_rate"), 0.0)
        empty = _safe_float(row.get("empty_rate"), 0.0)
        p95 = _safe_float(row.get("runtime_ms_p95"), 0.0)
        reg = _safe_float(row.get("regression_cer_mean"), 0.0)

        quality_guard = cer <= (default_cer + 0.01) and wer <= (default_wer + 0.01)
        stability_guard = fail <= (default_fail + 0.01) and empty <= (default_empty + 0.01)
        runtime_guard = p95 <= max(default_p95 * 1.6, default_p95 + 1500.0)
        regression_guard = reg <= (default_regression + 0.01)
        improved_pages = _safe_int(row.get("pages_improved_vs_default"), 0)

        assessments.append(
            {
                "profile_id": profile_id,
                "quality_guard": quality_guard,
                "stability_guard": stability_guard,
                "runtime_guard": runtime_guard,
                "regression_guard": regression_guard,
                "improved_pages": improved_pages,
                "cer_delta": cer - default_cer,
                "runtime_p95_delta_ms": p95 - default_p95,
                "all_guards_pass": quality_guard and stability_guard and runtime_guard and regression_guard,
            }
        )

    promotable = [item for item in assessments if item["all_guards_pass"] and item["improved_pages"] > 0]
    promotable.sort(key=lambda item: (item["cer_delta"], -item["runtime_p95_delta_ms"]))

    if not promotable:
        if any(item["improved_pages"] > 0 for item in assessments):
            return "promote_as_experimental_profile", "quality improvements observed but production guardrails were not met", assessments
        return "reject_due_to_insufficient_evidence", "no candidate improved quality under guardrails", assessments

    best = promotable[0]
    if best["profile_id"] in {
        "adaptive_high_dpi_for_small_or_low_quality_pages",
        "high_dpi_only_for_fallback_pages",
    }:
        return "promote_adaptive_high_dpi_fallback_only", "adaptive/fallback profile improved quality without violating runtime stability constraints", assessments

    if best["cer_delta"] <= -0.01:
        return "promote_default_render_profile", "fixed render profile delivered robust quality gains with acceptable runtime", assessments

    return "promote_as_experimental_profile", "candidate passes guardrails but gains are modest; ship as opt-in experimental", assessments


def _write_markdown_reports(
    pages_context: dict[str, Any],
    experiment_context: dict[str, Any],
    profile_defs: list[dict[str, Any]],
    per_page_rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
    page_analysis_rows: list[dict[str, Any]],
    decision: str,
    decision_reason: str,
    decision_assessments: list[dict[str, Any]],
    regression_verified: int,
    regression_total: int,
    unrecoverable_note: str,
) -> None:
    lines = [
        "# Render DPI Experiment Report",
        "",
        "## Scope",
        f"- selected_pages_total: {pages_context.get('selected_total', 0)}",
        f"- selected_smoke: {pages_context.get('selected_smoke', 0)}",
        f"- selected_regression: {pages_context.get('selected_regression', 0)}",
        f"- selected_validation: {pages_context.get('selected_validation', 0)}",
        f"- selected_validation_difficult_subset: {pages_context.get('selected_validation_difficult_subset', 0)}",
        f"- full_validation_included: {str(bool(pages_context.get('full_validation_included', False))).lower()}",
        f"- excluded_pages: {pages_context.get('excluded_count', 0)}",
        f"- profile_count: {experiment_context.get('profile_count', 0)}",
        "",
        "## Profile Designs",
        "",
        "| profile_id | internal_profile | render_dpi | render_scale | apply_mode | max_runtime_budget_ms | memory_risk | expected_benefit | expected_cost |",
        "|---|---|---:|---:|---|---:|---|---|---|",
    ]
    for profile in profile_defs:
        lines.append(
            "| {profile_id} | {internal_profile} | {render_dpi} | {render_scale} | {apply_mode} | {max_runtime_budget_ms} | {memory_risk} | {expected_benefit} | {expected_cost} |".format(
                profile_id=profile.get("profile_id", ""),
                internal_profile=profile.get("internal_profile", ""),
                render_dpi=profile.get("render_dpi", ""),
                render_scale=profile.get("render_scale", ""),
                apply_mode=profile.get("apply_mode", ""),
                max_runtime_budget_ms=profile.get("max_runtime_budget_ms", ""),
                memory_risk=profile.get("memory_risk", ""),
                expected_benefit=profile.get("expected_benefit", ""),
                expected_cost=profile.get("expected_cost", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Benchmark Matrix",
            "",
            "| profile_id | CER_mean | WER_mean | failed_rate | empty_rate | runtime_ms_p95 | improved_pages | worsened_pages | regression_cer_mean | smoke_cer_mean | difficult_validation_cer_mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in matrix_rows:
        lines.append(
            "| {profile_id} | {cer} | {wer} | {failed} | {empty} | {p95} | {improved} | {worsened} | {reg} | {smoke} | {difficult} |".format(
                profile_id=row.get("profile_id", ""),
                cer=_format_float(_safe_float(row.get("CER_mean"), 0.0)),
                wer=_format_float(_safe_float(row.get("WER_mean"), 0.0)),
                failed=_format_float(_safe_float(row.get("failed_rate"), 0.0)),
                empty=_format_float(_safe_float(row.get("empty_rate"), 0.0)),
                p95=_format_float(_safe_float(row.get("runtime_ms_p95"), 0.0)),
                improved=_safe_int(row.get("pages_improved_vs_default"), 0),
                worsened=_safe_int(row.get("pages_worsened_vs_default"), 0),
                reg=_format_float(_safe_float(row.get("regression_cer_mean"), 0.0)),
                smoke=_format_float(_safe_float(row.get("smoke_cer_mean"), 0.0)),
                difficult=_format_float(_safe_float(row.get("difficult_validation_cer_mean"), 0.0)),
            )
        )

    dataset_breakdown = _aggregate_breakdown(rows=per_page_rows, key="dataset_id")
    language_breakdown = _aggregate_breakdown(rows=per_page_rows, key="language_primary")
    layout_breakdown = _aggregate_breakdown(rows=per_page_rows, key="layout_type")
    document_breakdown = _aggregate_breakdown(rows=per_page_rows, key="document_id")

    def _append_breakdown(section_title: str, key: str, rows: list[dict[str, Any]], max_rows: int) -> None:
        lines.extend(
            [
                "",
                f"## {section_title}",
                "",
                f"| profile_id | {key} | count | CER_mean | WER_mean | runtime_ms_mean | failed_rate |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        if not rows:
            lines.append("| n/a | n/a | 0 | 0 | 0 | 0 | 0 |")
            return
        ordered = sorted(
            rows,
            key=lambda r: (
                str(r.get("profile_id", "")),
                -_safe_int(r.get("count"), 0),
                str(r.get(key, "")),
            ),
        )
        for row in ordered[:max_rows]:
            lines.append(
                "| {profile_id} | {bucket} | {count} | {cer} | {wer} | {rt} | {failed} |".format(
                    profile_id=row.get("profile_id", ""),
                    bucket=row.get(key, ""),
                    count=_safe_int(row.get("count"), 0),
                    cer=_format_float(_safe_float(row.get("CER_mean"), 0.0)),
                    wer=_format_float(_safe_float(row.get("WER_mean"), 0.0)),
                    rt=_format_float(_safe_float(row.get("runtime_ms_mean"), 0.0)),
                    failed=_format_float(_safe_float(row.get("failed_rate"), 0.0)),
                )
            )

    _append_breakdown("Per-dataset Metrics", "dataset_id", dataset_breakdown, 80)
    _append_breakdown("Per-language Metrics", "language_primary", language_breakdown, 60)
    _append_breakdown("Per-layout Metrics", "layout_type", layout_breakdown, 80)
    _append_breakdown("Per-document Metrics (top rows)", "document_id", document_breakdown, 120)

    lines.extend(
        [
            "",
            "## Excluded Pages (sample)",
        ]
    )
    examples = pages_context.get("excluded_examples", [])
    if examples:
        for example in examples:
            lines.append(
                f"- split={example.get('split_kind', '')} page_id={example.get('page_id', '')} reason={example.get('reason', '')}"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Promotion Recommendation",
            f"- decision: {decision}",
            f"- rationale: {decision_reason}",
        ]
    )
    OUT_RENDER_DPI_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    by_profile = Counter(str(row.get("profile_id", "")) for row in page_analysis_rows)
    improved = Counter(str(row.get("profile_id", "")) for row in page_analysis_rows if _safe_bool(row.get("is_improved")))
    worsened = Counter(str(row.get("profile_id", "")) for row in page_analysis_rows if _safe_bool(row.get("is_worsened")))

    page_lines = [
        "# Render DPI Page Level Analysis",
        "",
        "## Per-profile impact counts",
        "",
        "| profile_id | compared_pages | improved_pages | worsened_pages |",
        "|---|---:|---:|---:|",
    ]
    for profile_id in sorted(by_profile.keys()):
        page_lines.append(
            f"| {profile_id} | {by_profile[profile_id]} | {improved[profile_id]} | {worsened[profile_id]} |"
        )

    best_improvements = sorted(page_analysis_rows, key=lambda r: _safe_float(r.get("delta_CER"), 0.0))[:20]
    worst_regressions = sorted(page_analysis_rows, key=lambda r: _safe_float(r.get("delta_CER"), 0.0), reverse=True)[:20]

    page_lines.extend(
        [
            "",
            "## Top CER Improvements",
        ]
    )
    for row in best_improvements:
        page_lines.append(
            f"- profile={row.get('profile_id', '')} page_id={row.get('page_id', '')} delta_CER={_format_float(_safe_float(row.get('delta_CER'), 0.0))} reason={row.get('suspected_reason', '')}"
        )

    page_lines.extend(
        [
            "",
            "## Top CER Regressions",
        ]
    )
    for row in worst_regressions:
        page_lines.append(
            f"- profile={row.get('profile_id', '')} page_id={row.get('page_id', '')} delta_CER={_format_float(_safe_float(row.get('delta_CER'), 0.0))} reason={row.get('suspected_reason', '')}"
        )

    OUT_RENDER_DPI_PAGE_ANALYSIS_MD.write_text("\n".join(page_lines) + "\n", encoding="utf-8")

    promo_lines = [
        "# Render DPI Promotion Decision",
        "",
        f"- decision: {decision}",
        f"- rationale: {decision_reason}",
        "",
        "## Guardrail Assessment",
        "",
        "| profile_id | quality_guard | stability_guard | runtime_guard | regression_guard | improved_pages | cer_delta | runtime_p95_delta_ms |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for item in decision_assessments:
        promo_lines.append(
            "| {profile_id} | {quality} | {stability} | {runtime} | {regression} | {improved} | {cer_delta} | {runtime_delta} |".format(
                profile_id=item.get("profile_id", ""),
                quality=str(bool(item.get("quality_guard", False))).lower(),
                stability=str(bool(item.get("stability_guard", False))).lower(),
                runtime=str(bool(item.get("runtime_guard", False))).lower(),
                regression=str(bool(item.get("regression_guard", False))).lower(),
                improved=_safe_int(item.get("improved_pages"), 0),
                cer_delta=_format_float(_safe_float(item.get("cer_delta"), 0.0)),
                runtime_delta=_format_float(_safe_float(item.get("runtime_p95_delta_ms"), 0.0)),
            )
        )
    OUT_RENDER_DPI_PROMOTION_MD.write_text("\n".join(promo_lines) + "\n", encoding="utf-8")

    strategy_lines = [
        "# Render DPI Strategy Report",
        "",
        "## 1. Objective",
        "- Restore alignment-verified regression coverage first, then run controlled render/DPI profiling on verified-only records.",
        "",
        "## 2. Hard Gate Status",
        f"- regression_verified: {regression_verified}/{regression_total}",
        f"- gate_note: {unrecoverable_note}",
        "",
        "## 3. Evaluated Page Set",
        f"- smoke_pages: {pages_context.get('selected_smoke', 0)}",
        f"- regression_pages: {pages_context.get('selected_regression', 0)}",
        f"- validation_pages: {pages_context.get('selected_validation', 0)}",
        f"- validation_difficult_subset_pages: {pages_context.get('selected_validation_difficult_subset', 0)}",
        "",
        "## 4. Compared Profiles",
        "- current_default_render",
        "- render_300_dpi",
        "- render_400_dpi",
        "- render_600_dpi",
        "- adaptive_high_dpi_for_small_or_low_quality_pages",
        "- high_dpi_only_for_fallback_pages",
        "",
        "## 5. Quality Summary",
    ]
    for row in matrix_rows:
        strategy_lines.append(
            f"- {row.get('profile_id', '')}: CER_mean={_format_float(_safe_float(row.get('CER_mean'), 0.0))}, WER_mean={_format_float(_safe_float(row.get('WER_mean'), 0.0))}"
        )

    strategy_lines.extend(
        [
            "",
            "## 6. Regression Stability",
        ]
    )
    for row in matrix_rows:
        strategy_lines.append(
            f"- {row.get('profile_id', '')}: regression_cer_mean={_format_float(_safe_float(row.get('regression_cer_mean'), 0.0))}"
        )

    strategy_lines.extend(
        [
            "",
            "## 7. Smoke Stability",
        ]
    )
    for row in matrix_rows:
        strategy_lines.append(
            f"- {row.get('profile_id', '')}: smoke_cer_mean={_format_float(_safe_float(row.get('smoke_cer_mean'), 0.0))}"
        )

    strategy_lines.extend(
        [
            "",
            "## 8. Difficult Validation Behavior",
        ]
    )
    for row in matrix_rows:
        strategy_lines.append(
            f"- {row.get('profile_id', '')}: difficult_validation_cer_mean={_format_float(_safe_float(row.get('difficult_validation_cer_mean'), 0.0))}"
        )

    reason_counts = Counter(str(row.get("suspected_reason", "")) for row in page_analysis_rows)
    strategy_lines.extend(
        [
            "",
            "## 9. Page-Level Patterns",
        ]
    )
    for reason, count in reason_counts.most_common(12):
        strategy_lines.append(f"- {reason}: {count}")

    strategy_lines.extend(
        [
            "",
            "## 10. Runtime Tail",
        ]
    )
    for row in matrix_rows:
        strategy_lines.append(
            f"- {row.get('profile_id', '')}: runtime_p95_ms={_format_float(_safe_float(row.get('runtime_ms_p95'), 0.0))}, timeout_like_count={_safe_int(row.get('timeout_like_count'), 0)}"
        )

    strategy_lines.extend(
        [
            "",
            "## 11. Risk Assessment",
            "- High-DPI fixed profiles carry memory/latency risk and should not be default unless clear quality gains persist across splits.",
            "- Adaptive and fallback profiles reduce unnecessary high-DPI usage on easy pages.",
            "",
            "## 12. Recommendation",
            f"- {decision}",
            f"- rationale: {decision_reason}",
            "",
            "## 13. Rollout Plan",
            "- Start with canary on alignment-verified pages only.",
            "- Monitor CER/WER drift and runtime p95 against this benchmark matrix.",
            "",
            "## 14. Rollback Triggers",
            "- rollback if CER_mean regresses by >0.01 or runtime_p95 grows by >60% vs baseline.",
            "- rollback if failed_rate or empty_rate increases by >0.01 absolute.",
            "",
            "## 15. Next Action",
            "- Keep OCR tuning frozen until regression verified coverage remains stable across reruns.",
        ]
    )
    OUT_RENDER_DPI_STRATEGY_MD.write_text("\n".join(strategy_lines) + "\n", encoding="utf-8")


def main() -> int:
    audit_rows = _read_csv(AUDIT_CSV)
    if not audit_rows:
        print("audit csv missing or empty", file=sys.stderr)
        return 1

    regression_verified = len(_read_jsonl(VERIFIED_SPLIT_REGRESSION))
    regression_total = len(_read_jsonl(REGRESSION_SOURCE_SPLIT))
    regression_excluded = [
        row
        for row in audit_rows
        if str(row.get("split_kind", "")).strip() == "regression"
        and str(row.get("alignment_status", "")).strip() not in {"aligned", "safely_auto_fixed"}
    ]
    unresolved_statuses = Counter(str(row.get("alignment_status", "")).strip() for row in regression_excluded)

    gate_open = regression_verified >= 20
    unrecoverable_only = set(unresolved_statuses.keys()) <= {"missing_source_file", "missing_ground_truth_file"}
    unrecoverable_note = ""
    if gate_open:
        unrecoverable_note = "regression coverage recovered to threshold"
    elif regression_verified > 0 and unrecoverable_only:
        gate_open = True
        unrecoverable_note = "below threshold but remaining exclusions are unrecoverable source/gold missing cases"
    else:
        unrecoverable_note = "gate blocked: regression coverage below threshold with unresolved recoverable issues"

    if not gate_open:
        for path in [
            OUT_RENDER_DPI_MATRIX_CSV,
            OUT_RENDER_DPI_PER_PAGE_CSV,
            OUT_RENDER_DPI_REPORT_MD,
            OUT_RENDER_DPI_PAGE_ANALYSIS_CSV,
            OUT_RENDER_DPI_PAGE_ANALYSIS_MD,
            OUT_RENDER_DPI_PROMOTION_MD,
            OUT_RENDER_DPI_STRATEGY_MD,
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
        OUT_RENDER_DPI_MATRIX_CSV.write_text("profile_id,page_count,notes\n,0,gate_blocked\n", encoding="utf-8")
        OUT_RENDER_DPI_PER_PAGE_CSV.write_text("profile_id,page_id,notes\n", encoding="utf-8")
        OUT_RENDER_DPI_PAGE_ANALYSIS_CSV.write_text("profile_id,page_id,notes\n", encoding="utf-8")
        text = (
            "# Render DPI Experiment Report\n\n"
            f"- gate_open: false\n"
            f"- regression_verified: {regression_verified}/{regression_total}\n"
            f"- reason: {unrecoverable_note}\n"
        )
        OUT_RENDER_DPI_REPORT_MD.write_text(text, encoding="utf-8")
        OUT_RENDER_DPI_PAGE_ANALYSIS_MD.write_text(text, encoding="utf-8")
        OUT_RENDER_DPI_PROMOTION_MD.write_text(text, encoding="utf-8")
        OUT_RENDER_DPI_STRATEGY_MD.write_text(text, encoding="utf-8")
        print(
            json.dumps(
                {
                    "gate_open": False,
                    "regression_verified": regression_verified,
                    "regression_total": regression_total,
                    "reason": unrecoverable_note,
                },
                indent=2,
            )
        )
        return 0

    profile_defs = _register_render_profiles()
    pages, pages_context = _select_pages_for_experiment(audit_rows)
    if not pages:
        print("no eligible pages for render experiment", file=sys.stderr)
        return 1

    per_page_rows, experiment_context = _run_render_experiment(pages, profile_defs)
    matrix_rows = _build_matrix(per_page_rows, profile_defs)
    page_analysis_rows = _build_page_analysis(per_page_rows)
    decision, decision_reason, decision_assessments = _choose_promotion_decision(matrix_rows)

    _write_csv(
        OUT_RENDER_DPI_PER_PAGE_CSV,
        per_page_rows,
        [
            "profile_id",
            "profile_internal",
            "render_dpi",
            "render_scale",
            "apply_mode",
            "max_runtime_budget_ms",
            "memory_risk",
            "expected_benefit",
            "expected_cost",
            "dataset_id",
            "document_id",
            "page_id",
            "split_kind",
            "selection_group",
            "language_primary",
            "document_type",
            "layout_type",
            "scan_quality",
            "pdf_path",
            "requested_page_index",
            "applied_page_index",
            "gold_text_length",
            "output_text_length",
            "CER",
            "WER",
            "runtime_ms",
            "failed",
            "empty",
            "timeout_like",
            "memory_warning",
            "malformed_row",
            "failure_reason",
            "final_output_source",
            "profile_applied",
            "effective_render_dpi",
            "baseline_reused",
            "high_dpi_applied",
            "notes",
        ],
    )

    _write_csv(
        OUT_RENDER_DPI_MATRIX_CSV,
        matrix_rows,
        [
            "profile_id",
            "profile_internal",
            "render_dpi",
            "render_scale",
            "apply_mode",
            "max_runtime_budget_ms",
            "memory_risk",
            "expected_benefit",
            "expected_cost",
            "page_count",
            "CER_mean",
            "CER_median",
            "CER_p90",
            "WER_mean",
            "WER_median",
            "WER_p90",
            "failed_rate",
            "empty_rate",
            "runtime_ms_mean",
            "runtime_ms_median",
            "runtime_ms_p90",
            "runtime_ms_p95",
            "timeout_like_count",
            "memory_warning_count",
            "malformed_row_count",
            "pages_improved_vs_default",
            "pages_worsened_vs_default",
            "regression_cer_mean",
            "smoke_cer_mean",
            "difficult_validation_cer_mean",
            "notes",
        ],
    )

    _write_csv(
        OUT_RENDER_DPI_PAGE_ANALYSIS_CSV,
        page_analysis_rows,
        [
            "page_id",
            "split_kind",
            "selection_group",
            "dataset_id",
            "document_id",
            "layout_type",
            "profile_id",
            "default_CER",
            "candidate_CER",
            "delta_CER",
            "default_WER",
            "candidate_WER",
            "delta_WER",
            "default_runtime_ms",
            "candidate_runtime_ms",
            "delta_runtime_ms",
            "high_dpi_applied",
            "apply_mode",
            "is_improved",
            "is_worsened",
            "suspected_reason",
        ],
    )

    _write_markdown_reports(
        pages_context=pages_context,
        experiment_context=experiment_context,
        profile_defs=profile_defs,
        per_page_rows=per_page_rows,
        matrix_rows=matrix_rows,
        page_analysis_rows=page_analysis_rows,
        decision=decision,
        decision_reason=decision_reason,
        decision_assessments=decision_assessments,
        regression_verified=regression_verified,
        regression_total=regression_total,
        unrecoverable_note=unrecoverable_note,
    )

    print(
        json.dumps(
            {
                "render_dpi_benchmark_matrix_csv": str(OUT_RENDER_DPI_MATRIX_CSV.relative_to(ROOT)),
                "render_dpi_per_page_metrics_csv": str(OUT_RENDER_DPI_PER_PAGE_CSV.relative_to(ROOT)),
                "render_dpi_experiment_report_md": str(OUT_RENDER_DPI_REPORT_MD.relative_to(ROOT)),
                "render_dpi_page_level_analysis_csv": str(OUT_RENDER_DPI_PAGE_ANALYSIS_CSV.relative_to(ROOT)),
                "render_dpi_page_level_analysis_md": str(OUT_RENDER_DPI_PAGE_ANALYSIS_MD.relative_to(ROOT)),
                "render_dpi_promotion_decision_md": str(OUT_RENDER_DPI_PROMOTION_MD.relative_to(ROOT)),
                "render_dpi_strategy_report_md": str(OUT_RENDER_DPI_STRATEGY_MD.relative_to(ROOT)),
                "selected_pages_total": pages_context.get("selected_total", 0),
                "regression_verified": regression_verified,
                "regression_total": regression_total,
                "decision": decision,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
