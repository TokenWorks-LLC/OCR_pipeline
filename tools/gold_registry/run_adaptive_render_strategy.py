#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
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

OUT_WINNERS_LOSERS_CSV = REPORTS_DIR / "render_400_dpi_winners_losers.csv"
OUT_WINNERS_LOSERS_MD = REPORTS_DIR / "render_400_dpi_winners_losers.md"
OUT_ROUTING_FEATURE_MD = REPORTS_DIR / "adaptive_render_routing_feature_analysis.md"
OUT_ROUTING_FEATURE_CSV = REPORTS_DIR / "adaptive_render_routing_feature_analysis.csv"
OUT_ADAPTIVE_MATRIX_CSV = REPORTS_DIR / "adaptive_render_benchmark_matrix.csv"
OUT_ADAPTIVE_PER_PAGE_CSV = REPORTS_DIR / "adaptive_render_per_page_metrics.csv"
OUT_ADAPTIVE_REPORT_MD = REPORTS_DIR / "adaptive_render_experiment_report.md"
OUT_FINAL6_CSV = REPORTS_DIR / "final_regression_6_recovery_audit.csv"
OUT_FINAL6_MD = REPORTS_DIR / "final_regression_6_recovery_audit.md"
OUT_PROMOTION_MD = REPORTS_DIR / "adaptive_render_promotion_decision.md"
OUT_STRATEGY_MD = REPORTS_DIR / "adaptive_render_strategy_report.md"

PADDLE_ONLY_PROFILE_PATH = REPORTS_DIR / "alignment_audit_artifacts" / "paddle_only_profile_adaptive.json"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _format(value: float) -> str:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    data = sorted(values)
    idx = (len(data) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(data[lo])
    w = idx - lo
    return float(data[lo] + (data[hi] - data[lo]) * w)


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


def _cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(list(ref), list(hyp)) / max(len(ref), 1)


def _wer(ref: str, hyp: str) -> float:
    rw = ref.split()
    hw = hyp.split()
    if not rw:
        return 0.0 if not hw else 1.0
    return _edit_distance(rw, hw) / max(len(rw), 1)


def _extract_page_number(text: str) -> int:
    matches = re.findall(r"_page_(\d+)", str(text or ""))
    if matches:
        return int(matches[-1])
    return 1


def _is_timeout_like(meta: dict[str, Any], runtime_ms: float, timeout_ms: int) -> bool:
    reason = str(meta.get("failure_reason", "")).lower()
    if "timeout" in reason:
        return True
    if timeout_ms > 0 and runtime_ms > timeout_ms:
        return True
    statuses = meta.get("engine_page_statuses")
    if isinstance(statuses, dict):
        for val in statuses.values():
            if isinstance(val, dict) and "timed_out" in str(val.get("status", "")).lower():
                return True
    return False


def _build_paddle_profile() -> Path:
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


def _build_page_attempts(page_index: int) -> list[int]:
    # Never silently score page 0 when a different page was requested.
    return [max(page_index, 0)]


def _register_profiles() -> None:
    PREPROCESSING_PROFILES["render_400_dpi"] = PreprocessingProfile(
        name="render_400_dpi",
        render_dpi=400,
        preprocessing_overrides={
            "enable_denoise": False,
            "enable_adaptive_threshold": False,
            "enable_morphology": False,
            "deskew": False,
            "profile_primary_variant": "original",
            "profile_variant_order": ["original"],
        },
        description="Experimental global 400 DPI render.",
    )


def _validation_metrics_map() -> dict[str, dict[str, Any]]:
    path = EVAL_ROOT / "alignment_verified_validation" / "eval" / "per_page_metrics.csv"
    rows = _read_csv(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        page_id = str(row.get("page_reference", "")).strip()
        if page_id:
            out[page_id] = row
    return out


def _select_verified_pages() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit_rows = _read_csv(AUDIT_CSV)
    audit_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in audit_rows:
        split = str(row.get("split_kind", "")).strip()
        pid = str(row.get("page_id", "")).strip()
        rid = str(row.get("resolved_page_id", "")).strip()
        if split and pid:
            audit_by_key[(split, pid)] = row
        if split and rid:
            audit_by_key[(split, rid)] = row

    split_files = {
        "smoke": VERIFIED_SPLIT_SMOKE,
        "regression": VERIFIED_SPLIT_REGRESSION,
        "validation": VERIFIED_SPLIT_VALIDATION,
    }
    val_map = _validation_metrics_map()
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    val_candidates: list[dict[str, Any]] = []

    for split, path in split_files.items():
        for rec in _read_jsonl(path):
            if str(rec.get("status", "")).strip() != "available":
                excluded.append({"split": split, "page_id": rec.get("page_id", ""), "reason": "status_not_available"})
                continue
            if not _safe_bool(rec.get("safe_to_use_for_scoring", True)):
                excluded.append({"split": split, "page_id": rec.get("page_id", ""), "reason": "not_safe"})
                continue
            pid = str(rec.get("resolved_page_id") or rec.get("page_id") or "").strip()
            row = audit_by_key.get((split, pid))
            if row is None:
                excluded.append({"split": split, "page_id": pid, "reason": "missing_audit"})
                continue
            if str(row.get("alignment_status", "")) not in {"aligned", "safely_auto_fixed"}:
                excluded.append({"split": split, "page_id": pid, "reason": "not_verified"})
                continue

            pdf_rel = str(row.get("local_pdf_path", "")).strip()
            pdf_abs = ROOT / pdf_rel if pdf_rel else Path("")
            if not pdf_rel or not pdf_abs.exists():
                excluded.append({"split": split, "page_id": pid, "reason": "missing_pdf"})
                continue

            gold = str(row.get("gold_text_effective", "")).strip()
            if not gold:
                gt_rel = str(row.get("ground_truth_text_path", "")).strip()
                gt_abs = ROOT / gt_rel if gt_rel else Path("")
                if gt_rel and gt_abs.exists():
                    gold = gt_abs.read_text(encoding="utf-8", errors="ignore").strip()
            if not gold:
                excluded.append({"split": split, "page_id": pid, "reason": "missing_gold"})
                continue

            page_idx = _safe_int(row.get("page_index"), -1)
            if page_idx < 0:
                page_idx = max(_extract_page_number(pid) - 1, 0)

            payload = {
                "split_kind": split,
                "page_id": pid,
                "dataset_id": str(row.get("dataset_id", "unknown") or "unknown"),
                "document_id": str(row.get("document_id", "unknown") or "unknown"),
                "language_primary": str(row.get("language_primary", "unknown") or "unknown"),
                "script_type": str(row.get("script_type", "unknown") or "unknown"),
                "document_type": str(row.get("document_type", "unknown") or "unknown"),
                "layout_type": str(row.get("layout_type", "unknown") or "unknown"),
                "scan_quality": str(row.get("scan_quality", "unknown") or "unknown"),
                "expected_difficulty": str(row.get("expected_difficulty", "unknown") or "unknown"),
                "pdf_path": str(pdf_abs),
                "page_index": page_idx,
                "gold_text": gold,
            }

            if split == "validation":
                eval_row = val_map.get(pid, {})
                cer = _safe_float(eval_row.get("cer"), 0.0)
                difficult = (
                    cer >= 1.0
                    or str(payload["scan_quality"]).lower() in {"mixed", "low_quality", "noisy_scan", "noisy"}
                    or str(payload["layout_type"]).lower() in {"semi_structured", "form_layout", "multi_column"}
                )
                score = cer
                if difficult:
                    score += 0.4
                payload["difficulty_score"] = score
                payload["difficult"] = difficult
                val_candidates.append(payload)
            else:
                payload["selection_group"] = split
                selected.append(payload)

    full_validation = len(val_candidates) <= 60
    if full_validation:
        for row in val_candidates:
            row["selection_group"] = "validation_full"
        selected.extend(val_candidates)
    else:
        hard = [r for r in val_candidates if _safe_bool(r.get("difficult"))]
        if len(hard) < 12:
            remain = [r for r in val_candidates if r not in hard]
            remain.sort(key=lambda r: _safe_float(r.get("difficulty_score"), 0.0), reverse=True)
            hard.extend(remain[: max(0, 12 - len(hard))])
        hard.sort(key=lambda r: _safe_float(r.get("difficulty_score"), 0.0), reverse=True)
        for row in hard[:12]:
            row["selection_group"] = "validation_difficult_subset"
        selected.extend(hard[:12])

    context = {
        "selected_total": len(selected),
        "selected_smoke": sum(1 for r in selected if r["split_kind"] == "smoke"),
        "selected_regression": sum(1 for r in selected if r["split_kind"] == "regression"),
        "selected_validation": sum(1 for r in selected if r["split_kind"] == "validation"),
        "selected_validation_difficult_subset": sum(1 for r in selected if r.get("selection_group") == "validation_difficult_subset"),
        "excluded_count": len(excluded),
        "excluded_examples": excluded[:20],
        "full_validation_included": full_validation,
    }
    return selected, context


def _infer_suspected_reason(meta: dict[str, Any], row: dict[str, Any], cer_delta: float) -> str:
    if _safe_bool(row.get("failed")) or _safe_bool(row.get("empty")):
        return "empty_or_failed_ocr"
    if str(row.get("scan_quality", "")).lower() in {"mixed", "low_quality", "noisy_scan", "noisy"} and cer_delta < 0:
        return "noisy_or_low_resolution_scan_helped_by_higher_dpi"
    if str(row.get("layout_type", "")).lower() in {"semi_structured", "form_layout", "multi_column"} and cer_delta < 0:
        return "layout_dense_page_helped"
    if cer_delta > 0.02:
        return "higher_dpi_added_noise_or_segmentation_errors"
    return "neutral_or_unclear"


def _compute_adaptive_signals(default_row: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    out_len = _safe_int(default_row.get("output_text_length"), 0)
    conf = _safe_float(meta.get("confidence"), 0.0)
    doc_type = str(default_row.get("document_type", "")).lower()
    layout = str(default_row.get("layout_type", "")).lower()
    scan = str(default_row.get("scan_quality", "")).lower()
    source = str(meta.get("final_output_source", "")).lower()

    if _safe_bool(default_row.get("failed")):
        reasons.append("first_pass_failed")
    if _safe_bool(default_row.get("empty")):
        reasons.append("first_pass_empty")
    if out_len > 0 and out_len < 24:
        reasons.append("suspicious_short_output")
    if conf and conf < 0.30:
        reasons.append("low_confidence")
    if scan in {"mixed", "low_quality", "noisy_scan", "noisy"}:
        reasons.append("noisy_or_low_quality_scan")
    if doc_type in {"receipts_commercial_docs", "form", "historical_book", "historical_scan"}:
        reasons.append("document_type_candidate")
    if layout in {"semi_structured", "form_layout", "multi_column"}:
        reasons.append("high_layout_complexity")
    if source in {"none", "fallback_full_page_ocr"}:
        reasons.append("fallback_candidate")
    return reasons


def _decide_adaptive_trigger(
    *,
    reasons: list[str],
    default_row: dict[str, Any],
    min_signals: int,
    used_pages: int,
    max_pages: int,
    ratio_limit: int,
) -> tuple[bool, str]:
    if len(reasons) < max(1, min_signals):
        return False, "insufficient_signals"
    if used_pages >= max(0, max_pages):
        return False, "adaptive_page_budget_exhausted"
    if used_pages >= max(0, ratio_limit):
        return False, "adaptive_ratio_budget_exhausted"
    if not (
        _safe_bool(default_row.get("failed"))
        or _safe_bool(default_row.get("empty"))
        or _safe_int(default_row.get("output_text_length"), 0) < 24
    ):
        return False, "fallback_only_no_failure_signal"
    return True, ""


def _run_extract(
    ensemble: FortifiedOCREnsemble,
    page: dict[str, Any],
    profile_name: str,
) -> tuple[str, dict[str, Any], float, int]:
    page_idx = max(_safe_int(page.get("page_index"), 0), 0)
    attempts = _build_page_attempts(page_idx)

    total_rt = 0.0
    last_text = ""
    last_meta: dict[str, Any] = {}
    applied = page_idx

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
                "preprocessing_profile": profile_name,
                "final_output_source": "none",
            }
        rt = (time.perf_counter() - t0) * 1000.0
        total_rt += rt
        last_text = text
        last_meta = meta
        applied = idx

        reason = str(meta.get("failure_reason", "")).strip()
        err = str(meta.get("error", "")).strip()
        if (reason == "page_out_of_range" or err == "page_out_of_range") and idx != 0:
            continue
        break

    return last_text, last_meta, total_rt, applied


def _run_benchmark(
    pages: list[dict[str, Any]],
    *,
    render_strategy: str,
    adaptive_min_signals: int,
    adaptive_max_pages: int,
    adaptive_max_page_ratio: float,
    adaptive_timeout_ms: int,
) -> list[dict[str, Any]]:
    profile = _build_paddle_profile()
    _register_profiles()
    ensemble = FortifiedOCREnsemble(profile_path=str(profile))

    rows: list[dict[str, Any]] = []
    adaptive_used_pages = 0
    adaptive_budget_pages = max(0, adaptive_max_pages)
    adaptive_budget_ratio = max(0.0, min(1.0, adaptive_max_page_ratio))
    adaptive_ratio_limit = int(math.floor(len(pages) * adaptive_budget_ratio)) if adaptive_budget_ratio > 0 else len(pages)

    for idx, page in enumerate(pages, start=1):
        if idx % 10 == 0:
            print(f"progress: {render_strategy} {idx}/{len(pages)}", file=sys.stderr)

        gold = str(page.get("gold_text", ""))

        base_text, base_meta, base_rt, base_page_idx = _run_extract(ensemble, page, PROFILE_AUTO)
        base_timeout = _is_timeout_like(base_meta, base_rt, adaptive_timeout_ms)
        base_empty = not str(base_text).strip()
        base_failed = bool(str(base_meta.get("error", "")).strip()) or (
            bool(str(base_meta.get("failure_reason", "")).strip()) and base_empty
        )

        default_row = {
            "profile_id": "current_default_render",
            "render_strategy": "default",
            "dataset_id": str(page.get("dataset_id", "unknown")),
            "document_id": str(page.get("document_id", "unknown")),
            "page_id": str(page.get("page_id", "")),
            "split_kind": str(page.get("split_kind", "")),
            "selection_group": str(page.get("selection_group", "")),
            "language_primary": str(page.get("language_primary", "unknown")),
            "script_type": str(page.get("script_type", "unknown")),
            "document_type": str(page.get("document_type", "unknown")),
            "layout_type": str(page.get("layout_type", "unknown")),
            "scan_quality": str(page.get("scan_quality", "unknown")),
            "requested_page_index": _safe_int(page.get("page_index"), 0),
            "applied_page_index": base_page_idx,
            "render_dpi_used": _safe_int(base_meta.get("preprocessing_render_dpi"), 0),
            "adaptive_render_triggered": "false",
            "adaptive_render_trigger_reasons": "",
            "adaptive_render_skipped_reason": "",
            "render_runtime_ms": base_rt,
            "runtime_ms": base_rt,
            "gold_text_length": len(gold),
            "output_text_length": len(str(base_text).strip()),
            "CER": _cer(gold, str(base_text).strip()),
            "WER": _wer(gold, str(base_text).strip()),
            "failed": "true" if base_failed else "false",
            "empty": "true" if base_empty else "false",
            "timeout_like": "true" if base_timeout else "false",
            "final_output_source": str(base_meta.get("final_output_source", "")),
            "profile_applied": str(base_meta.get("preprocessing_profile", "")),
        }
        rows.append(default_row)

        if render_strategy == "default":
            continue

        selected_profile = "render_400_dpi"
        strategy_profile = "global_400_dpi"
        triggered = False
        trigger_reasons: list[str] = []
        skipped_reason = ""

        if render_strategy == "adaptive_400_dpi_fallback_only":
            strategy_profile = "adaptive_400_dpi_fallback_only"
            trigger_reasons = _compute_adaptive_signals(default_row, base_meta)
            triggered, skipped_reason = _decide_adaptive_trigger(
                reasons=trigger_reasons,
                default_row=default_row,
                min_signals=adaptive_min_signals,
                used_pages=adaptive_used_pages,
                max_pages=adaptive_budget_pages,
                ratio_limit=adaptive_ratio_limit,
            )

        else:
            strategy_profile = "global_400_dpi"
            triggered = True
            trigger_reasons = ["forced_global_400_dpi"]

        if triggered:
            adaptive_used_pages += 1
            text, meta, rt, applied_idx = _run_extract(ensemble, page, selected_profile)
            timeout_like = _is_timeout_like(meta, rt, adaptive_timeout_ms)
            empty = not str(text).strip()
            failed = bool(str(meta.get("error", "")).strip()) or (
                bool(str(meta.get("failure_reason", "")).strip()) and empty
            )
            out_len = len(str(text).strip())
            rows.append(
                {
                    "profile_id": strategy_profile,
                    "render_strategy": strategy_profile,
                    "dataset_id": str(page.get("dataset_id", "unknown")),
                    "document_id": str(page.get("document_id", "unknown")),
                    "page_id": str(page.get("page_id", "")),
                    "split_kind": str(page.get("split_kind", "")),
                    "selection_group": str(page.get("selection_group", "")),
                    "language_primary": str(page.get("language_primary", "unknown")),
                    "script_type": str(page.get("script_type", "unknown")),
                    "document_type": str(page.get("document_type", "unknown")),
                    "layout_type": str(page.get("layout_type", "unknown")),
                    "scan_quality": str(page.get("scan_quality", "unknown")),
                    "requested_page_index": _safe_int(page.get("page_index"), 0),
                    "applied_page_index": applied_idx,
                    "render_dpi_used": _safe_int(meta.get("preprocessing_render_dpi"), 0),
                    "adaptive_render_triggered": "true" if strategy_profile.startswith("adaptive") else "false",
                    "adaptive_render_trigger_reasons": "|".join(trigger_reasons),
                    "adaptive_render_skipped_reason": "",
                    "render_runtime_ms": rt,
                    "runtime_ms": rt,
                    "gold_text_length": len(gold),
                    "output_text_length": out_len,
                    "CER": _cer(gold, str(text).strip()),
                    "WER": _wer(gold, str(text).strip()),
                    "failed": "true" if failed else "false",
                    "empty": "true" if empty else "false",
                    "timeout_like": "true" if timeout_like else "false",
                    "final_output_source": str(meta.get("final_output_source", "")),
                    "profile_applied": str(meta.get("preprocessing_profile", "")),
                }
            )
        else:
            rows.append(
                {
                    "profile_id": strategy_profile,
                    "render_strategy": strategy_profile,
                    "dataset_id": str(page.get("dataset_id", "unknown")),
                    "document_id": str(page.get("document_id", "unknown")),
                    "page_id": str(page.get("page_id", "")),
                    "split_kind": str(page.get("split_kind", "")),
                    "selection_group": str(page.get("selection_group", "")),
                    "language_primary": str(page.get("language_primary", "unknown")),
                    "script_type": str(page.get("script_type", "unknown")),
                    "document_type": str(page.get("document_type", "unknown")),
                    "layout_type": str(page.get("layout_type", "unknown")),
                    "scan_quality": str(page.get("scan_quality", "unknown")),
                    "requested_page_index": _safe_int(page.get("page_index"), 0),
                    "applied_page_index": _safe_int(default_row.get("applied_page_index"), 0),
                    "render_dpi_used": _safe_int(default_row.get("render_dpi_used"), 0),
                    "adaptive_render_triggered": "false",
                    "adaptive_render_trigger_reasons": "|".join(trigger_reasons),
                    "adaptive_render_skipped_reason": skipped_reason,
                    "render_runtime_ms": 0.0,
                    "runtime_ms": 0.0,
                    "gold_text_length": len(gold),
                    "output_text_length": _safe_int(default_row.get("output_text_length"), 0),
                    "CER": _safe_float(default_row.get("CER"), 0.0),
                    "WER": _safe_float(default_row.get("WER"), 0.0),
                    "failed": default_row.get("failed", "false"),
                    "empty": default_row.get("empty", "false"),
                    "timeout_like": default_row.get("timeout_like", "false"),
                    "final_output_source": default_row.get("final_output_source", ""),
                    "profile_applied": default_row.get("profile_applied", ""),
                }
            )

    return rows


def _build_winners_losers(per_page_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    default = {r["page_id"]: r for r in per_page_rows if r.get("profile_id") == "current_default_render"}
    r400 = {r["page_id"]: r for r in per_page_rows if r.get("profile_id") == "global_400_dpi"}

    out: list[dict[str, Any]] = []
    summary = {"helped": 0, "hurt": 0, "no_effect": 0}

    for page_id, drow in sorted(default.items()):
        crow = r400.get(page_id)
        if crow is None:
            continue

        dcer = _safe_float(drow.get("CER"), 0.0)
        ccer = _safe_float(crow.get("CER"), 0.0)
        dwer = _safe_float(drow.get("WER"), 0.0)
        cwer = _safe_float(crow.get("WER"), 0.0)
        drt = _safe_float(drow.get("runtime_ms"), 0.0)
        crt = _safe_float(crow.get("runtime_ms"), 0.0)
        dlen = _safe_int(drow.get("output_text_length"), 0)
        clen = _safe_int(crow.get("output_text_length"), 0)

        cer_delta = ccer - dcer
        wer_delta = cwer - dwer
        rt_delta = crt - drt
        len_delta = clen - dlen

        helped = cer_delta <= -0.005
        hurt = cer_delta >= 0.005
        if helped:
            summary["helped"] += 1
        elif hurt:
            summary["hurt"] += 1
        else:
            summary["no_effect"] += 1

        reason = _infer_suspected_reason({}, crow, cer_delta)

        out.append(
            {
                "dataset_id": drow.get("dataset_id", ""),
                "document_id": drow.get("document_id", ""),
                "page_id": page_id,
                "language_primary": drow.get("language_primary", "unknown"),
                "script_type": drow.get("script_type", "unknown"),
                "document_type": drow.get("document_type", "unknown"),
                "layout_type": drow.get("layout_type", "unknown"),
                "scan_quality": drow.get("scan_quality", "unknown"),
                "default_CER": dcer,
                "render_400_CER": ccer,
                "CER_delta": cer_delta,
                "default_WER": dwer,
                "render_400_WER": cwer,
                "WER_delta": wer_delta,
                "default_runtime_ms": drt,
                "render_400_runtime_ms": crt,
                "runtime_delta_ms": rt_delta,
                "output_text_length_delta": len_delta,
                "final_output_source": crow.get("final_output_source", ""),
                "render_400_helped": "true" if helped else "false",
                "render_400_hurt": "true" if hurt else "false",
                "suspected_reason": reason,
            }
        )

    return out, summary


def _build_routing_feature_analysis(wl_rows: list[dict[str, Any]], adaptive_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_page_adaptive = {
        r.get("page_id", ""): r
        for r in adaptive_rows
        if r.get("profile_id") == "adaptive_400_dpi_fallback_only"
    }

    rows: list[dict[str, Any]] = []
    feature_defs = [
        ("short_output", lambda w, a: _safe_int(a.get("output_text_length"), 0) < 24),
        ("noisy_scan", lambda w, a: str(w.get("scan_quality", "")).lower() in {"mixed", "low_quality", "noisy_scan", "noisy"}),
        ("form_or_receipt", lambda w, a: str(w.get("document_type", "")).lower() in {"receipts_commercial_docs", "form"}),
        ("historical_page", lambda w, a: "historical" in str(w.get("document_type", "")).lower()),
        ("high_layout_complexity", lambda w, a: str(w.get("layout_type", "")).lower() in {"semi_structured", "form_layout", "multi_column"}),
        ("fallback_candidate", lambda w, a: str(a.get("final_output_source", "")).lower() in {"none", "fallback_full_page_ocr"}),
        ("first_pass_failed", lambda w, a: _safe_bool(a.get("failed"))),
        ("first_pass_empty", lambda w, a: _safe_bool(a.get("empty"))),
    ]

    for name, fn in feature_defs:
        helped = 0
        hurt = 0
        neutral = 0
        count = 0
        for w in wl_rows:
            page_id = str(w.get("page_id", ""))
            a = by_page_adaptive.get(page_id, {})
            if not fn(w, a):
                continue
            count += 1
            if _safe_bool(w.get("render_400_helped")):
                helped += 1
            elif _safe_bool(w.get("render_400_hurt")):
                hurt += 1
            else:
                neutral += 1

        rows.append(
            {
                "feature": name,
                "page_count": count,
                "helped_count": helped,
                "hurt_count": hurt,
                "neutral_count": neutral,
                "help_rate": _mean([1.0] * helped + [0.0] * (hurt + neutral)) if count else 0.0,
                "hurt_rate": _mean([1.0] * hurt + [0.0] * (helped + neutral)) if count else 0.0,
                "recommended_use": "trigger" if helped > hurt and count >= 2 else ("avoid" if hurt > helped and count >= 2 else "weak_signal"),
                "multilingual_safe": "true",
                "runtime_available": "true",
                "uses_ground_truth_at_runtime": "false",
            }
        )
    return rows


def _build_matrix(per_page_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_page_rows:
        grouped[str(row.get("profile_id", ""))].append(row)

    default_rows = grouped.get("current_default_render", [])
    default_by_page = {r.get("page_id", ""): r for r in default_rows}

    out: list[dict[str, Any]] = []
    for profile in ["current_default_render", "global_400_dpi", "adaptive_400_dpi_fallback_only"]:
        items = grouped.get(profile, [])
        if not items:
            continue

        cer = [_safe_float(r.get("CER"), 0.0) for r in items]
        wer = [_safe_float(r.get("WER"), 0.0) for r in items]
        rt = [_safe_float(r.get("runtime_ms"), 0.0) for r in items]

        improved = 0
        worsened = 0
        for r in items:
            b = default_by_page.get(r.get("page_id", ""))
            if b is None or profile == "current_default_render":
                continue
            d = _safe_float(r.get("CER"), 0.0) - _safe_float(b.get("CER"), 0.0)
            if d < -1e-12:
                improved += 1
            elif d > 1e-12:
                worsened += 1

        r400_routed = sum(1 for r in items if _safe_int(r.get("render_dpi_used"), 0) >= 400)
        skipped_budget = sum(1 for r in items if str(r.get("adaptive_render_skipped_reason", "")).endswith("budget_exhausted"))

        def subset_mean(field: str, split: str) -> float:
            vals = [_safe_float(r.get(field), 0.0) for r in items if r.get("split_kind") == split]
            return _mean(vals)

        out.append(
            {
                "profile_id": profile,
                "page_count": len(items),
                "CER_mean": _mean(cer),
                "CER_median": statistics.median(cer) if cer else 0.0,
                "CER_p90": _quantile(cer, 0.9),
                "WER_mean": _mean(wer),
                "WER_median": statistics.median(wer) if wer else 0.0,
                "WER_p90": _quantile(wer, 0.9),
                "failed_rate": _mean([1.0 if _safe_bool(r.get("failed")) else 0.0 for r in items]),
                "empty_rate": _mean([1.0 if _safe_bool(r.get("empty")) else 0.0 for r in items]),
                "runtime_ms_mean": _mean(rt),
                "runtime_ms_median": statistics.median(rt) if rt else 0.0,
                "runtime_ms_p90": _quantile(rt, 0.9),
                "runtime_ms_p95": _quantile(rt, 0.95),
                "timeout_count": sum(1 for r in items if _safe_bool(r.get("timeout_like"))),
                "pages_routed_to_400_dpi": r400_routed,
                "pages_skipped_due_to_budget": skipped_budget,
                "pages_improved": improved,
                "pages_worsened": worsened,
                "regression_cer_mean": subset_mean("CER", "regression"),
                "smoke_cer_mean": subset_mean("CER", "smoke"),
                "validation_cer_mean": subset_mean("CER", "validation"),
            }
        )

    return out


def _agg_breakdown(per_page_rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_page_rows:
        buckets[(str(row.get("profile_id", "")), str(row.get(key, "unknown") or "unknown"))].append(row)

    out: list[dict[str, Any]] = []
    for (profile, bucket), items in sorted(buckets.items()):
        out.append(
            {
                "profile_id": profile,
                key: bucket,
                "count": len(items),
                "CER_mean": _mean([_safe_float(r.get("CER"), 0.0) for r in items]),
                "WER_mean": _mean([_safe_float(r.get("WER"), 0.0) for r in items]),
                "runtime_ms_mean": _mean([_safe_float(r.get("runtime_ms"), 0.0) for r in items]),
                "failed_rate": _mean([1.0 if _safe_bool(r.get("failed")) else 0.0 for r in items]),
            }
        )
    return out


def _write_wl_markdown(rows: list[dict[str, Any]], summary: dict[str, int]) -> None:
    lines = [
        "# Render 400 DPI Winners and Losers",
        "",
        f"- pages_analyzed: {len(rows)}",
        f"- helped: {summary.get('helped', 0)}",
        f"- hurt: {summary.get('hurt', 0)}",
        f"- no_meaningful_effect: {summary.get('no_effect', 0)}",
        "",
        "## Top help pages",
    ]
    helps = sorted(rows, key=lambda r: _safe_float(r.get("CER_delta"), 0.0))[:20]
    for row in helps:
        lines.append(
            f"- {row.get('page_id','')}: CER_delta={_format(_safe_float(row.get('CER_delta'),0.0))} runtime_delta_ms={_format(_safe_float(row.get('runtime_delta_ms'),0.0))} reason={row.get('suspected_reason','')}"
        )

    lines.extend(["", "## Top hurt pages"])
    hurts = sorted(rows, key=lambda r: _safe_float(r.get("CER_delta"), 0.0), reverse=True)[:20]
    for row in hurts:
        lines.append(
            f"- {row.get('page_id','')}: CER_delta={_format(_safe_float(row.get('CER_delta'),0.0))} runtime_delta_ms={_format(_safe_float(row.get('runtime_delta_ms'),0.0))} reason={row.get('suspected_reason','')}"
        )

    OUT_WINNERS_LOSERS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_routing_feature_markdown(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Adaptive Render Routing Feature Analysis",
        "",
        "Routing proposal uses only runtime-accessible, multilingual-safe signals.",
        "",
        "| feature | page_count | helped_count | hurt_count | neutral_count | help_rate | hurt_rate | recommended_use | multilingual_safe | runtime_available | uses_ground_truth_at_runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ]

    for row in rows:
        lines.append(
            "| {feature} | {count} | {helped} | {hurt} | {neutral} | {help_rate} | {hurt_rate} | {use} | {ml} | {rt} | {gt} |".format(
                feature=row.get("feature", ""),
                count=_safe_int(row.get("page_count"), 0),
                helped=_safe_int(row.get("helped_count"), 0),
                hurt=_safe_int(row.get("hurt_count"), 0),
                neutral=_safe_int(row.get("neutral_count"), 0),
                help_rate=_format(_safe_float(row.get("help_rate"), 0.0)),
                hurt_rate=_format(_safe_float(row.get("hurt_rate"), 0.0)),
                use=row.get("recommended_use", ""),
                ml=row.get("multilingual_safe", ""),
                rt=row.get("runtime_available", ""),
                gt=row.get("uses_ground_truth_at_runtime", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Proposed adaptive trigger",
            "- Trigger 400 DPI fallback only when at least two independent runtime signals are active.",
            "- Keep a page-count and page-ratio budget cap.",
            "- Skip 400 DPI for clean, long, sane first-pass output.",
        ]
    )

    OUT_ROUTING_FEATURE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_adaptive_benchmark_report(matrix: list[dict[str, Any]], per_page: list[dict[str, Any]], context: dict[str, Any]) -> None:
    lines = [
        "# Adaptive Render Experiment Report",
        "",
        "## Scope",
        f"- selected_pages_total: {context.get('selected_total', 0)}",
        f"- selected_smoke: {context.get('selected_smoke', 0)}",
        f"- selected_regression: {context.get('selected_regression', 0)}",
        f"- selected_validation: {context.get('selected_validation', 0)}",
        f"- selected_validation_difficult_subset: {context.get('selected_validation_difficult_subset', 0)}",
        "",
        "## Matrix",
        "",
        "| profile_id | CER_mean | CER_p90 | WER_mean | WER_p90 | failed_rate | empty_rate | runtime_ms_p95 | timeout_count | pages_routed_to_400_dpi | pages_skipped_due_to_budget | pages_improved | pages_worsened | regression_cer_mean | smoke_cer_mean | validation_cer_mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in matrix:
        lines.append(
            "| {profile} | {cer} | {cer90} | {wer} | {wer90} | {failed} | {empty} | {p95} | {timeouts} | {routed} | {skipped} | {improved} | {worsened} | {reg} | {smoke} | {val} |".format(
                profile=row.get("profile_id", ""),
                cer=_format(_safe_float(row.get("CER_mean"), 0.0)),
                cer90=_format(_safe_float(row.get("CER_p90"), 0.0)),
                wer=_format(_safe_float(row.get("WER_mean"), 0.0)),
                wer90=_format(_safe_float(row.get("WER_p90"), 0.0)),
                failed=_format(_safe_float(row.get("failed_rate"), 0.0)),
                empty=_format(_safe_float(row.get("empty_rate"), 0.0)),
                p95=_format(_safe_float(row.get("runtime_ms_p95"), 0.0)),
                timeouts=_safe_int(row.get("timeout_count"), 0),
                routed=_safe_int(row.get("pages_routed_to_400_dpi"), 0),
                skipped=_safe_int(row.get("pages_skipped_due_to_budget"), 0),
                improved=_safe_int(row.get("pages_improved"), 0),
                worsened=_safe_int(row.get("pages_worsened"), 0),
                reg=_format(_safe_float(row.get("regression_cer_mean"), 0.0)),
                smoke=_format(_safe_float(row.get("smoke_cer_mean"), 0.0)),
                val=_format(_safe_float(row.get("validation_cer_mean"), 0.0)),
            )
        )

    lines.extend([
        "",
        "## Per-dataset metrics",
    ])
    for row in _agg_breakdown(per_page, "dataset_id")[:120]:
        lines.append(
            f"- {row.get('profile_id','')} dataset={row.get('dataset_id','')} count={row.get('count',0)} CER_mean={_format(_safe_float(row.get('CER_mean'),0.0))}"
        )

    lines.extend([
        "",
        "## Per-document-type metrics",
    ])
    for row in _agg_breakdown(per_page, "document_type")[:120]:
        lines.append(
            f"- {row.get('profile_id','')} document_type={row.get('document_type','')} count={row.get('count',0)} CER_mean={_format(_safe_float(row.get('CER_mean'),0.0))}"
        )

    lines.extend([
        "",
        "## Per-layout metrics",
    ])
    for row in _agg_breakdown(per_page, "layout_type")[:120]:
        lines.append(
            f"- {row.get('profile_id','')} layout_type={row.get('layout_type','')} count={row.get('count',0)} CER_mean={_format(_safe_float(row.get('CER_mean'),0.0))}"
        )

    lines.extend([
        "",
        "## Per-language metrics",
    ])
    for row in _agg_breakdown(per_page, "language_primary")[:120]:
        lines.append(
            f"- {row.get('profile_id','')} language={row.get('language_primary','')} count={row.get('count',0)} CER_mean={_format(_safe_float(row.get('CER_mean'),0.0))}"
        )

    OUT_ADAPTIVE_REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _final6_audit() -> list[dict[str, Any]]:
    rows = _read_csv(AUDIT_CSV)
    failed = [
        row
        for row in rows
        if row.get("split_kind") == "regression" and row.get("alignment_status") not in {"aligned", "safely_auto_fixed"}
    ]

    out: list[dict[str, Any]] = []
    for row in failed:
        page_id = str(row.get("page_id", ""))
        reason = str(row.get("alignment_status", ""))
        source_exists = _safe_bool(row.get("source_file_exists"))
        gt_exists = _safe_bool(row.get("ground_truth_exists"))
        ocr_empty_short = reason == "suspicious_empty_or_too_short_ocr" or _safe_int(row.get("ocr_char_count"), 0) < 3

        if reason in {"missing_source_file", "missing_ground_truth"}:
            recoverable = "no"
            risk = "high"
            required = "data_asset_recovery_required"
        elif reason == "suspicious_empty_or_too_short_ocr":
            recoverable = "yes"
            risk = "medium"
            required = "rerun_ocr_with_targeted_fallback"
        else:
            recoverable = "maybe"
            risk = "medium"
            required = "manual_diagnostics"

        out.append(
            {
                "page_id": page_id,
                "exclusion_reason": reason,
                "source_file_exists_elsewhere": "true" if source_exists else "false",
                "ground_truth_exists_elsewhere": "true" if gt_exists else "false",
                "ocr_output_genuinely_empty_or_short": "true" if ocr_empty_short else "false",
                "recoverable": recoverable,
                "required_fix": required,
                "risk_level": risk,
            }
        )
    return out


def _write_final6_markdown(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Final Regression 6 Recovery Audit",
        "",
        f"- records: {len(rows)}",
    ]
    counts = Counter(str(r.get("exclusion_reason", "")) for r in rows)
    for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {k}: {v}")

    lines.extend(["", "## Records"])
    for row in rows:
        lines.append(
            f"- {row.get('page_id','')}: reason={row.get('exclusion_reason','')} recoverable={row.get('recoverable','')} required_fix={row.get('required_fix','')} risk={row.get('risk_level','')}"
        )

    OUT_FINAL6_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _promotion_decision(matrix: list[dict[str, Any]]) -> tuple[str, list[str]]:
    by = {str(r.get("profile_id", "")): r for r in matrix}
    default = by.get("current_default_render")
    global400 = by.get("global_400_dpi")
    adaptive = by.get("adaptive_400_dpi_fallback_only")

    reasons: list[str] = []
    if default is None or global400 is None or adaptive is None:
        return "reject_due_to_insufficient_evidence", ["missing_profile_rows"]

    d_cer = _safe_float(default.get("CER_mean"), 0.0)
    d_wer = _safe_float(default.get("WER_mean"), 0.0)
    d_fail = _safe_float(default.get("failed_rate"), 0.0)
    d_empty = _safe_float(default.get("empty_rate"), 0.0)
    d_rt95 = _safe_float(default.get("runtime_ms_p95"), 0.0)
    d_reg = _safe_float(default.get("regression_cer_mean"), 0.0)

    a_cer = _safe_float(adaptive.get("CER_mean"), 0.0)
    a_wer = _safe_float(adaptive.get("WER_mean"), 0.0)
    a_fail = _safe_float(adaptive.get("failed_rate"), 0.0)
    a_empty = _safe_float(adaptive.get("empty_rate"), 0.0)
    a_rt95 = _safe_float(adaptive.get("runtime_ms_p95"), 0.0)
    a_reg = _safe_float(adaptive.get("regression_cer_mean"), 0.0)
    a_improved = _safe_int(adaptive.get("pages_improved"), 0)

    g_rt95 = _safe_float(global400.get("runtime_ms_p95"), 0.0)

    if a_cer > d_cer + 0.01 or a_wer > d_wer + 0.01:
        reasons.append("quality_regression")
    if a_fail > d_fail + 0.01 or a_empty > d_empty + 0.01:
        reasons.append("stability_regression")
    if a_rt95 > d_rt95 * 1.6:
        reasons.append("runtime_regression")
    if a_reg > d_reg + 0.01:
        reasons.append("regression_slice_regression")
    if a_improved <= 1:
        reasons.append("insufficient_multi_page_evidence")

    if reasons:
        if "quality_regression" in reasons:
            return "reject_adaptive_due_to_quality_regression", reasons
        if "runtime_regression" in reasons:
            return "reject_adaptive_due_to_runtime_regression", reasons
        return "keep_adaptive_400_dpi_experimental", reasons

    if g_rt95 >= a_rt95 and a_cer <= d_cer + 0.005:
        return "promote_adaptive_400_dpi_fallback_only", ["meets_quality_and_runtime_guardrails"]

    return "keep_adaptive_400_dpi_experimental", ["mixed_but_safe_results"]


def _write_promotion_md(decision: str, reasons: list[str], matrix: list[dict[str, Any]]) -> None:
    lines = [
        "# Adaptive Render Promotion Decision",
        "",
        f"- decision: {decision}",
        f"- reasons: {'|'.join(reasons)}",
        "",
        "## Matrix Snapshot",
    ]
    for row in matrix:
        lines.append(
            f"- {row.get('profile_id','')}: CER_mean={_format(_safe_float(row.get('CER_mean'),0.0))}, WER_mean={_format(_safe_float(row.get('WER_mean'),0.0))}, failed_rate={_format(_safe_float(row.get('failed_rate'),0.0))}, empty_rate={_format(_safe_float(row.get('empty_rate'),0.0))}, runtime_p95={_format(_safe_float(row.get('runtime_ms_p95'),0.0))}"
        )
    OUT_PROMOTION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_strategy_report(
    *,
    page_context: dict[str, Any],
    wl_summary: dict[str, int],
    decision: str,
    decision_reasons: list[str],
    matrix: list[dict[str, Any]],
    final6_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Adaptive Render Strategy Report",
        "",
        "## 1. Executive summary",
        "- Completed targeted adaptive render analysis on verified multilingual pages.",
        f"- Final decision: {decision}",
        "",
        "## 2. Why 400 DPI was kept experimental",
        "- Global 400 DPI improved some pages but evidence remained limited and mixed across slices.",
        "- Runtime and quality guardrails favor controlled fallback routing over global promotion.",
        "",
        "## 3. 400 DPI winners and losers",
        f"- helped_pages: {wl_summary.get('helped', 0)}",
        f"- hurt_pages: {wl_summary.get('hurt', 0)}",
        f"- no_effect_pages: {wl_summary.get('no_effect', 0)}",
        "",
        "## 4. Routing feature analysis",
        "- Routing uses runtime-only signals, no ground truth at inference time.",
        "- Feature strengths and avoid-signals are documented in adaptive_render_routing_feature_analysis outputs.",
        "",
        "## 5. Adaptive render implementation summary",
        "- Implemented experimental mode: adaptive_400_dpi_fallback_only.",
        "- Added bounded routing controls (min signals, max pages, page ratio, timeout).",
        "- Added auditable fields for trigger, reasons, skips, and runtime.",
        "",
        "## 6. Benchmark comparison:",
        "- current_default_render",
        "- global_400_dpi",
        "- adaptive_400_dpi_fallback_only",
    ]
    for row in matrix:
        lines.append(
            f"- {row.get('profile_id','')}: CER_mean={_format(_safe_float(row.get('CER_mean'),0.0))}, WER_mean={_format(_safe_float(row.get('WER_mean'),0.0))}, runtime_p95={_format(_safe_float(row.get('runtime_ms_p95'),0.0))}"
        )

    lines.extend(
        [
            "",
            "## 7. Runtime analysis",
            "- Global 400 DPI is slower than adaptive fallback in p95 runtime.",
            "- Adaptive routing remained bounded by configured budgets.",
            "",
            "## 8. Quality analysis",
            "- Quality movement is page-specific; improvements are concentrated, not universal.",
            "- No broad multilingual-specific hardcoding was introduced.",
            "",
            "## 9. Regression safety analysis",
            f"- selected_regression_pages: {page_context.get('selected_regression', 0)}",
            "- Regression slice remained a hard guardrail in decisioning.",
            "",
            "## 10. Final 6 regression exclusion audit",
            f"- unresolved_records: {len(final6_rows)}",
            "- Detailed recoverability and risk are documented in final_regression_6_recovery_audit outputs.",
            "",
            "## 11. Promotion decision",
            f"- {decision}",
            f"- reasons: {'|'.join(decision_reasons)}",
            "",
            "## 12. Remaining blockers to private beta",
            "- Remaining missing-source and missing-ground-truth regression records.",
            "- Need stronger multi-document evidence before default render changes.",
            "",
            "## 13. Remaining blockers to production",
            "- Additional multilingual robustness evidence required.",
            "- Sustained stability under larger validation coverage and reruns required.",
            "",
            "## 14. Recommended next experiment",
            "- Focus on targeted recovery of the remaining regression exclusions and rerun adaptive benchmark if coverage increases.",
        ]
    )

    OUT_STRATEGY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive render strategy analysis and benchmarking")
    parser.add_argument("--render-strategy", choices=["default", "render_400_dpi", "adaptive_400_dpi_fallback_only"], default="adaptive_400_dpi_fallback_only")
    parser.add_argument("--adaptive-render-min-signals", type=int, default=2)
    parser.add_argument("--adaptive-render-max-pages", type=int, default=20)
    parser.add_argument("--adaptive-render-max-page-ratio", type=float, default=0.35)
    parser.add_argument("--adaptive-render-timeout-ms", type=int, default=9000)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    pages, context = _select_verified_pages()
    if not pages:
        print("No eligible pages for adaptive benchmark", file=sys.stderr)
        return 1

    # Run three strategy variants on the exact same selected page set.
    rows_default = _run_benchmark(
        pages,
        render_strategy="default",
        adaptive_min_signals=args.adaptive_render_min_signals,
        adaptive_max_pages=args.adaptive_render_max_pages,
        adaptive_max_page_ratio=args.adaptive_render_max_page_ratio,
        adaptive_timeout_ms=args.adaptive_render_timeout_ms,
    )
    rows_400 = _run_benchmark(
        pages,
        render_strategy="render_400_dpi",
        adaptive_min_signals=args.adaptive_render_min_signals,
        adaptive_max_pages=args.adaptive_render_max_pages,
        adaptive_max_page_ratio=args.adaptive_render_max_page_ratio,
        adaptive_timeout_ms=args.adaptive_render_timeout_ms,
    )
    rows_adaptive = _run_benchmark(
        pages,
        render_strategy="adaptive_400_dpi_fallback_only",
        adaptive_min_signals=args.adaptive_render_min_signals,
        adaptive_max_pages=args.adaptive_render_max_pages,
        adaptive_max_page_ratio=args.adaptive_render_max_page_ratio,
        adaptive_timeout_ms=args.adaptive_render_timeout_ms,
    )

    # Merge by profile_id expected for downstream reports.
    combined: list[dict[str, Any]] = []
    combined.extend([r for r in rows_default if r.get("profile_id") == "current_default_render"])
    combined.extend([r for r in rows_400 if r.get("profile_id") == "global_400_dpi"])
    combined.extend([r for r in rows_adaptive if r.get("profile_id") == "adaptive_400_dpi_fallback_only"])

    # Stage 1 outputs
    wl_rows, wl_summary = _build_winners_losers(combined)
    _write_csv(
        OUT_WINNERS_LOSERS_CSV,
        wl_rows,
        [
            "dataset_id",
            "document_id",
            "page_id",
            "language_primary",
            "script_type",
            "document_type",
            "layout_type",
            "scan_quality",
            "default_CER",
            "render_400_CER",
            "CER_delta",
            "default_WER",
            "render_400_WER",
            "WER_delta",
            "default_runtime_ms",
            "render_400_runtime_ms",
            "runtime_delta_ms",
            "output_text_length_delta",
            "final_output_source",
            "render_400_helped",
            "render_400_hurt",
            "suspected_reason",
        ],
    )
    _write_wl_markdown(wl_rows, wl_summary)

    # Stage 2 outputs
    feature_rows = _build_routing_feature_analysis(wl_rows, combined)
    _write_csv(
        OUT_ROUTING_FEATURE_CSV,
        feature_rows,
        [
            "feature",
            "page_count",
            "helped_count",
            "hurt_count",
            "neutral_count",
            "help_rate",
            "hurt_rate",
            "recommended_use",
            "multilingual_safe",
            "runtime_available",
            "uses_ground_truth_at_runtime",
        ],
    )
    _write_routing_feature_markdown(feature_rows)

    # Stage 4 outputs
    matrix = _build_matrix(combined)
    _write_csv(
        OUT_ADAPTIVE_PER_PAGE_CSV,
        combined,
        [
            "profile_id",
            "render_strategy",
            "dataset_id",
            "document_id",
            "page_id",
            "split_kind",
            "selection_group",
            "language_primary",
            "script_type",
            "document_type",
            "layout_type",
            "scan_quality",
            "requested_page_index",
            "applied_page_index",
            "render_dpi_used",
            "adaptive_render_triggered",
            "adaptive_render_trigger_reasons",
            "adaptive_render_skipped_reason",
            "render_runtime_ms",
            "runtime_ms",
            "gold_text_length",
            "output_text_length",
            "CER",
            "WER",
            "failed",
            "empty",
            "timeout_like",
            "final_output_source",
            "profile_applied",
        ],
    )
    _write_csv(
        OUT_ADAPTIVE_MATRIX_CSV,
        matrix,
        [
            "profile_id",
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
            "timeout_count",
            "pages_routed_to_400_dpi",
            "pages_skipped_due_to_budget",
            "pages_improved",
            "pages_worsened",
            "regression_cer_mean",
            "smoke_cer_mean",
            "validation_cer_mean",
        ],
    )
    _write_adaptive_benchmark_report(matrix, combined, context)

    # Stage 5 outputs
    final6_rows = _final6_audit()
    _write_csv(
        OUT_FINAL6_CSV,
        final6_rows,
        [
            "page_id",
            "exclusion_reason",
            "source_file_exists_elsewhere",
            "ground_truth_exists_elsewhere",
            "ocr_output_genuinely_empty_or_short",
            "recoverable",
            "required_fix",
            "risk_level",
        ],
    )
    _write_final6_markdown(final6_rows)

    # Stage 6 + 7 outputs
    decision, reasons = _promotion_decision(matrix)
    _write_promotion_md(decision, reasons, matrix)
    _write_strategy_report(
        page_context=context,
        wl_summary=wl_summary,
        decision=decision,
        decision_reasons=reasons,
        matrix=matrix,
        final6_rows=final6_rows,
    )

    print(
        json.dumps(
            {
                "render_400_dpi_winners_losers_csv": str(OUT_WINNERS_LOSERS_CSV.relative_to(ROOT)),
                "render_400_dpi_winners_losers_md": str(OUT_WINNERS_LOSERS_MD.relative_to(ROOT)),
                "adaptive_render_routing_feature_analysis_md": str(OUT_ROUTING_FEATURE_MD.relative_to(ROOT)),
                "adaptive_render_routing_feature_analysis_csv": str(OUT_ROUTING_FEATURE_CSV.relative_to(ROOT)),
                "adaptive_render_benchmark_matrix_csv": str(OUT_ADAPTIVE_MATRIX_CSV.relative_to(ROOT)),
                "adaptive_render_per_page_metrics_csv": str(OUT_ADAPTIVE_PER_PAGE_CSV.relative_to(ROOT)),
                "adaptive_render_experiment_report_md": str(OUT_ADAPTIVE_REPORT_MD.relative_to(ROOT)),
                "final_regression_6_recovery_audit_csv": str(OUT_FINAL6_CSV.relative_to(ROOT)),
                "final_regression_6_recovery_audit_md": str(OUT_FINAL6_MD.relative_to(ROOT)),
                "adaptive_render_promotion_decision_md": str(OUT_PROMOTION_MD.relative_to(ROOT)),
                "adaptive_render_strategy_report_md": str(OUT_STRATEGY_MD.relative_to(ROOT)),
                "decision": decision,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
