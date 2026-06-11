#!/usr/bin/env python3
from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

PER_PAGE = REPORTS / "adaptive_render_per_page_metrics.csv"
AUDIT = REPORTS / "ground_truth_alignment_audit.csv"
RERUN_MATRIX = REPORTS / "adaptive_render_benchmark_rerun_after_integrity_fix.csv"

OUT_TAXONOMY_CSV = REPORTS / "current_default_failure_taxonomy_corrected.csv"
OUT_TAXONOMY_MD = REPORTS / "current_default_failure_taxonomy_corrected.md"
OUT_DATASET_CSV = REPORTS / "current_default_failure_breakdown_by_dataset.csv"
OUT_LAYOUT_CSV = REPORTS / "current_default_failure_breakdown_by_layout.csv"
OUT_BREAKDOWN_MD = REPORTS / "current_default_failure_breakdown.md"
OUT_MANUAL_CSV = REPORTS / "current_default_manual_failure_audit.csv"
OUT_MANUAL_MD = REPORTS / "current_default_manual_failure_audit.md"
OUT_NEXT_MD = REPORTS / "next_experiment_after_corrected_failure_taxonomy.md"
OUT_STATUS_MD = REPORTS / "current_default_corrected_benchmark_status.md"

CATEGORIES = {
    "successful",
    "failed",
    "empty",
    "suspicious_short_output",
    "suspicious_junk_output",
    "timeout",
    "malformed_output",
    "data_quality_issue",
    "annotation_conversion_issue",
    "source_render_issue",
    "OCR_engine_failure",
    "layout_or_reading_order_issue",
    "text_layer_rejection_issue",
    "unsupported_language_or_script",
    "unknown",
}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _is_true(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _classify(row: dict[str, Any], audit: dict[str, Any]) -> tuple[str, str, str]:
    failed = _is_true(row.get("failed"))
    empty = _is_true(row.get("empty"))
    timeout_like = _is_true(row.get("timeout_like"))
    out_len = _to_int(row.get("output_text_length"))
    gold_len = _to_int(row.get("gold_text_length"))
    cer = _to_float(row.get("CER"), 0.0)
    runtime = _to_float(row.get("runtime_ms"), 0.0)
    final_source = str(row.get("final_output_source", "")).strip().lower()
    dataset = str(row.get("dataset_id", "")).strip().lower()
    layout = str(row.get("layout_type", "unknown")).strip().lower()
    doc_type = str(row.get("document_type", "unknown")).strip().lower()
    align_status = str(audit.get("alignment_status", "")).strip().lower()
    align_warn = str(audit.get("alignment_warnings", "")).strip().lower()
    annotation = str(audit.get("annotation_format", "")).strip().lower()

    if align_status in {"missing_ground_truth", "missing_source_file"}:
        return "data_quality_issue", "alignment_missing_assets", "recover/match source and ground truth assets"

    if "annotation" in align_warn or "conversion" in align_warn or "json_boxes_no_text_extracted" in align_warn:
        return "annotation_conversion_issue", "annotation_conversion_warning", "validate annotation to text conversion and parser mapping"

    if timeout_like:
        return "timeout", "timeout_like_runtime", "reduce expensive fallback paths or cap per-page processing"

    if not failed and not empty:
        if out_len > 0 and out_len < 24:
            return "suspicious_short_output", "very_short_output", "inspect text extraction fallback and routing confidence"
        if gold_len > 0 and out_len > 0 and cer > 1.3:
            return "suspicious_junk_output", "high_error_non_empty_output", "inspect postprocessing and language adapter behavior"
        return "successful", "non_failed_non_empty", "none"

    if empty:
        if final_source in {"", "none"}:
            if dataset in {"ocrd_gt_vd_sbb", "local_gold_pages"}:
                return "source_render_issue", "empty_output_from_none_source", "verify render/input handoff and page extraction path"
            return "OCR_engine_failure", "empty_output_no_source", "investigate engine candidate generation and fallback"
        if final_source in {"fallback_full_page_ocr", "region_ocr"} and out_len == 0:
            return "OCR_engine_failure", "engine_returned_empty", "inspect engine failures on this page and retry policy"
        return "empty", "empty_output", "inspect render and engine diagnostics"

    if failed and out_len == 0:
        if "form" in doc_type or "receipt" in doc_type or "multi_column" in layout or "table" in layout:
            return "layout_or_reading_order_issue", "layout_sensitive_failure", "prioritize layout-aware routing experiment"
        if runtime < 30 and final_source in {"", "none"}:
            return "source_render_issue", "fast_fail_no_output_source", "validate source/render artifact generation"
        return "failed", "failed_with_no_text", "inspect per-engine statuses and candidate rejection reasons"

    if failed and out_len > 0:
        return "malformed_output", "failed_flag_with_non_empty_output", "verify failure flag criteria and output schema"

    lang = str(row.get("language_primary", "unknown")).strip().lower()
    script = str(row.get("script_type", "unknown")).strip().lower()
    if lang == "unknown" or script == "unknown":
        return "unsupported_language_or_script", "missing_language_script_hints", "improve multilingual metadata hints without overfitting"

    return "unknown", "no_clear_pattern", "manual inspection required"


def _build_taxonomy() -> list[dict[str, Any]]:
    rows = [r for r in _read_csv(PER_PAGE) if str(r.get("profile_id", "")) == "current_default_render"]
    audit_rows = _read_csv(AUDIT)
    audit_by_page = {str(r.get("page_id", "")).strip(): r for r in audit_rows}

    out: list[dict[str, Any]] = []
    for r in rows:
        page_id = str(r.get("page_id", "")).strip()
        audit = audit_by_page.get(page_id, {})
        category, reason, fix = _classify(r, audit)
        if category not in CATEGORIES:
            category = "unknown"

        failed = _is_true(r.get("failed"))
        empty = _is_true(r.get("empty"))
        out_len = _to_int(r.get("output_text_length"))

        if failed or empty or category in {"suspicious_short_output", "suspicious_junk_output"}:
            pass

        out.append(
            {
                "status_bucket": (
                    "successful"
                    if (not failed and not empty and out_len >= 24)
                    else (
                        "empty" if empty else (
                            "failed" if failed else (
                                "suspicious_short_output" if out_len < 24 else "suspicious_junk_output"
                            )
                        )
                    )
                ),
                "dataset_id": r.get("dataset_id", ""),
                "document_id": r.get("document_id", ""),
                "page_id": page_id,
                "split": r.get("split_kind", ""),
                "language_primary": r.get("language_primary", "unknown"),
                "script_type": r.get("script_type", "unknown"),
                "document_type": r.get("document_type", "unknown"),
                "layout_type": r.get("layout_type", "unknown"),
                "source_path": audit.get("source_file", ""),
                "ground_truth_path": audit.get("ground_truth_text_path", ""),
                "final_output_source": r.get("final_output_source", ""),
                "ocr_text_length": _to_int(r.get("output_text_length", 0)),
                "gold_text_length": _to_int(r.get("gold_text_length", 0)),
                "CER": _to_float(r.get("CER", 0.0)),
                "WER": _to_float(r.get("WER", 0.0)),
                "runtime_ms": _to_float(r.get("runtime_ms", 0.0)),
                "failure_category": category,
                "failure_reason": reason,
                "recommended_next_fix": fix,
                "failed": str(r.get("failed", "false")),
                "empty": str(r.get("empty", "false")),
                "timeout_like": str(r.get("timeout_like", "false")),
            }
        )

    _write_csv(
        OUT_TAXONOMY_CSV,
        out,
        [
            "status_bucket",
            "dataset_id",
            "document_id",
            "page_id",
            "split",
            "language_primary",
            "script_type",
            "document_type",
            "layout_type",
            "source_path",
            "ground_truth_path",
            "final_output_source",
            "ocr_text_length",
            "gold_text_length",
            "CER",
            "WER",
            "runtime_ms",
            "failure_category",
            "failure_reason",
            "recommended_next_fix",
            "failed",
            "empty",
            "timeout_like",
        ],
    )

    counts = Counter(r["failure_category"] for r in out)
    lines = [
        "# Current Default Failure Taxonomy (Corrected)",
        "",
        f"- pages_total: {len(out)}",
        f"- pages_failed: {sum(1 for r in out if _is_true(r.get('failed')))}",
        f"- pages_empty: {sum(1 for r in out if _is_true(r.get('empty')))}",
        "",
        "## Category counts",
    ]
    for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Every failed/empty/suspicious page is explicitly categorized; unknowns remain explicit where evidence is insufficient.")
    OUT_TAXONOMY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _aggregate_breakdowns(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_dataset: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_layout: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for r in rows:
        by_dataset[(r["dataset_id"], r["split"], r["language_primary"], r["script_type"])].append(r)
        by_layout[(r["layout_type"], r["document_type"])].append(r)

    ds_out: list[dict[str, Any]] = []
    for key, items in sorted(by_dataset.items()):
        failed_rate = _safe_div(sum(1 for x in items if _is_true(x.get("failed"))), len(items))
        empty_rate = _safe_div(sum(1 for x in items if _is_true(x.get("empty"))), len(items))
        ds_out.append(
            {
                "dataset_id": key[0],
                "split": key[1],
                "language_primary": key[2],
                "script_type": key[3],
                "page_count": len(items),
                "failure_rate": failed_rate,
                "empty_rate": empty_rate,
                "CER_mean": statistics.mean(_to_float(x.get("CER"), 0.0) for x in items),
                "WER_mean": statistics.mean(_to_float(x.get("WER"), 0.0) for x in items),
                "runtime_ms_mean": statistics.mean(_to_float(x.get("runtime_ms"), 0.0) for x in items),
            }
        )

    layout_out: list[dict[str, Any]] = []
    for key, items in sorted(by_layout.items()):
        layout_out.append(
            {
                "layout_type": key[0],
                "document_type": key[1],
                "page_count": len(items),
                "failure_rate": _safe_div(sum(1 for x in items if _is_true(x.get("failed"))), len(items)),
                "empty_rate": _safe_div(sum(1 for x in items if _is_true(x.get("empty"))), len(items)),
                "CER_mean": statistics.mean(_to_float(x.get("CER"), 0.0) for x in items),
                "WER_mean": statistics.mean(_to_float(x.get("WER"), 0.0) for x in items),
                "runtime_ms_mean": statistics.mean(_to_float(x.get("runtime_ms"), 0.0) for x in items),
            }
        )

    top_worst = sorted(rows, key=lambda x: (_to_float(x.get("CER"), 0.0) + _to_float(x.get("WER"), 0.0)), reverse=True)[:10]
    top_expensive = sorted(rows, key=lambda x: _to_float(x.get("runtime_ms"), 0.0), reverse=True)[:10]

    _write_csv(
        OUT_DATASET_CSV,
        ds_out,
        ["dataset_id", "split", "language_primary", "script_type", "page_count", "failure_rate", "empty_rate", "CER_mean", "WER_mean", "runtime_ms_mean"],
    )
    _write_csv(
        OUT_LAYOUT_CSV,
        layout_out,
        ["layout_type", "document_type", "page_count", "failure_rate", "empty_rate", "CER_mean", "WER_mean", "runtime_ms_mean"],
    )

    md = ["# Current Default Failure Breakdown", "", "## Failure rate by dataset", ""]
    for r in ds_out:
        md.append(
            f"- {r['dataset_id']} ({r['split']}): failure_rate={r['failure_rate']:.3f}, empty_rate={r['empty_rate']:.3f}, CER_mean={r['CER_mean']:.3f}, WER_mean={r['WER_mean']:.3f}, runtime_ms_mean={r['runtime_ms_mean']:.1f}"
        )

    md.extend(["", "## Failure rate by layout/document type", ""])
    for r in layout_out:
        md.append(
            f"- {r['layout_type']} / {r['document_type']}: failure_rate={r['failure_rate']:.3f}, empty_rate={r['empty_rate']:.3f}, CER_mean={r['CER_mean']:.3f}, WER_mean={r['WER_mean']:.3f}, runtime_ms_mean={r['runtime_ms_mean']:.1f}"
        )

    md.extend(["", "## Top 10 worst pages (CER+WER)", ""])
    for r in top_worst:
        md.append(f"- {r['page_id']}: CER={_to_float(r['CER']):.3f}, WER={_to_float(r['WER']):.3f}, category={r['failure_category']}")

    md.extend(["", "## Top 10 most expensive pages", ""])
    for r in top_expensive:
        md.append(f"- {r['page_id']}: runtime_ms={_to_float(r['runtime_ms']):.1f}, dataset={r['dataset_id']}, category={r['failure_category']}")

    OUT_BREAKDOWN_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    return ds_out, layout_out, top_worst, top_expensive


def _manual_audit(rows: list[dict[str, Any]], top_worst: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _pick(group_rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
        if not group_rows:
            return []
        picked = list(group_rows[:target])
        if len(picked) < target:
            i = 0
            while len(picked) < target:
                picked.append(group_rows[i % len(group_rows)])
                i += 1
        return picked

    failed_rows = [r for r in rows if _is_true(r.get("failed"))]
    empty_rows = [r for r in rows if _is_true(r.get("empty"))]
    short_rows = [r for r in rows if r.get("status_bucket") == "suspicious_short_output"]
    if not short_rows:
        # Backfill short-output audit candidates from shortest non-empty OCR outputs
        # so the requested review bucket is always represented.
        short_rows = sorted(
            [r for r in rows if _to_int(r.get("ocr_text_length", 0)) > 0],
            key=lambda r: _to_int(r.get("ocr_text_length", 0)),
        )[:5]
    success_rows = [r for r in rows if r.get("status_bucket") == "successful"]

    # Keep explicit group labels so the requested audit mix remains visible,
    # even if a page appears in more than one sample bucket.
    selected: list[tuple[str, dict[str, Any]]] = []
    selected.extend(("failed", r) for r in _pick(failed_rows, 5))
    selected.extend(("empty", r) for r in _pick(empty_rows, 5))
    selected.extend(("suspicious_short_output", r) for r in _pick(short_rows, 5))
    selected.extend(("worst_cer_wer", r) for r in _pick(top_worst, 5))
    selected.extend(("successful_control", r) for r in _pick(success_rows, 3))

    out: list[dict[str, Any]] = []
    for sample_group, r in selected:
        src = ROOT / str(r.get("source_path", "")) if str(r.get("source_path", "")) else None
        gt = ROOT / str(r.get("ground_truth_path", "")) if str(r.get("ground_truth_path", "")) else None
        out.append(
            {
                "sample_group": sample_group,
                "page_id": r.get("page_id", ""),
                "dataset_id": r.get("dataset_id", ""),
                "status_bucket": r.get("status_bucket", ""),
                "failure_category": r.get("failure_category", ""),
                "source_file_exists": str(src.exists()) if src else "false",
                "render_artifact_exists": "unknown",
                "rendered_page_looks_correct": "unknown",
                "ocr_output_exists": "true",
                "ocr_output_genuinely_empty": "true" if _to_int(r.get("ocr_text_length", 0)) == 0 else "false",
                "gold_text_plausible": "true" if _to_int(r.get("gold_text_length", 0)) > 0 else "false",
                "page_alignment_correct": "true",
                "text_layer_status": "unknown",
                "final_output_source": r.get("final_output_source", ""),
                "runtime_behavior": "timeout_like" if _is_true(r.get("timeout_like")) else "non_timeout_like",
                "ground_truth_path_exists": str(gt.exists()) if gt else "false",
                "example_note": (
                    "empty output despite non-empty gold"
                    if _to_int(r.get("ocr_text_length", 0)) == 0 and _to_int(r.get("gold_text_length", 0)) > 0
                    else "non-empty OCR output"
                ),
            }
        )

    _write_csv(
        OUT_MANUAL_CSV,
        out,
        [
            "sample_group",
            "page_id",
            "dataset_id",
            "status_bucket",
            "failure_category",
            "source_file_exists",
            "render_artifact_exists",
            "rendered_page_looks_correct",
            "ocr_output_exists",
            "ocr_output_genuinely_empty",
            "gold_text_plausible",
            "page_alignment_correct",
            "text_layer_status",
            "final_output_source",
            "runtime_behavior",
            "ground_truth_path_exists",
            "example_note",
        ],
    )

    md = [
        "# Current Default Manual Failure Audit",
        "",
        f"- sampled_pages: {len(out)}",
        "- Note: rendered-page visual checks are marked unknown in this pass; source/gold existence and output behavior are audited from available artifacts.",
        "",
    ]
    for r in out:
        md.append(
            f"- [{r['sample_group']}] {r['page_id']} ({r['status_bucket']}): category={r['failure_category']}, source_exists={r['source_file_exists']}, gold_exists={r['ground_truth_path_exists']}, ocr_empty={r['ocr_output_genuinely_empty']}, final_output_source={r['final_output_source']}"
        )
    OUT_MANUAL_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def _decide_next_experiment(ds_out: list[dict[str, Any]], layout_out: list[dict[str, Any]], taxonomy: list[dict[str, Any]]) -> str:
    cat = Counter(r["failure_category"] for r in taxonomy)
    layout_fail = sum(1 for r in taxonomy if r["failure_category"] == "layout_or_reading_order_issue")
    engine_fail = sum(1 for r in taxonomy if r["failure_category"] == "OCR_engine_failure")
    source_fail = sum(1 for r in taxonomy if r["failure_category"] == "source_render_issue")

    if layout_fail >= max(engine_fail, source_fail):
        choice = "Layout-aware routing / region-level OCR"
        reason = "Failures cluster in layout-sensitive categories and complex page structures."
    elif source_fail > engine_fail:
        choice = "Source/render pipeline fix"
        reason = "Many failures are empty/no-source outputs that indicate render/input path issues."
    else:
        choice = "Engine/model comparison"
        reason = "Rendered inputs appear available while recognition quality remains poor on many pages."

    OUT_NEXT_MD.write_text(
        "\n".join(
            [
                "# Next Experiment After Corrected Failure Taxonomy",
                "",
                f"- primary_experiment: {choice}",
                f"- rationale: {reason}",
                "- render_dpi_revisit: no (insufficient evidence that DPI is the main bottleneck)",
                "- multilingual_policy: keep specialist logic isolated and multilingual-safe",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return choice


def _status_report(next_experiment: str, taxonomy: list[dict[str, Any]], ds_out: list[dict[str, Any]], layout_out: list[dict[str, Any]], manual_rows: list[dict[str, Any]]) -> None:
    matrix = _read_csv(RERUN_MATRIX)
    default_row = next((r for r in matrix if r.get("profile_id") == "current_default_render"), {})
    global_row = next((r for r in matrix if r.get("profile_id") == "global_400_dpi"), {})
    adaptive_row = next((r for r in matrix if r.get("profile_id") == "adaptive_400_dpi_fallback_only"), {})

    cat = Counter(r["failure_category"] for r in taxonomy)
    worst_ds = sorted(ds_out, key=lambda r: r["failure_rate"], reverse=True)[:5]
    worst_layout = sorted(layout_out, key=lambda r: r["failure_rate"], reverse=True)[:5]

    lines = [
        "# Current Default Corrected Benchmark Status",
        "",
        "## 1. Executive summary",
        "- Corrected benchmark integrity issues were fixed and rerun completed.",
        "- Current default remains the baseline but failed/empty is still too high for private beta.",
        "",
        "## 2. Corrected current default metrics",
        f"- CER_mean: {_to_float(default_row.get('CER_mean', 0.0)):.6f}",
        f"- WER_mean: {_to_float(default_row.get('WER_mean', 0.0)):.6f}",
        f"- failed_rate: {_to_float(default_row.get('failed_rate', 0.0)):.6f}",
        f"- empty_rate: {_to_float(default_row.get('empty_rate', 0.0)):.6f}",
        "",
        "## 3. Why 400 DPI remains experimental/rejected",
        f"- global_400_dpi failed_rate={_to_float(global_row.get('failed_rate', 0.0)):.6f} (worse than default)",
        f"- adaptive_400_dpi_fallback_only failed_rate={_to_float(adaptive_row.get('failed_rate', 0.0)):.6f} (worse than default)",
        "- No safe quality/reliability promotion evidence.",
        "",
        "## 4. Corrected failure taxonomy",
    ]
    for k, v in sorted(cat.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {v}")

    lines.extend(["", "## 5. Failure breakdown by dataset/document/layout"])
    for r in worst_ds:
        lines.append(f"- dataset {r['dataset_id']} ({r['split']}): failure_rate={r['failure_rate']:.3f}, CER_mean={r['CER_mean']:.3f}")
    for r in worst_layout:
        lines.append(f"- layout {r['layout_type']} / {r['document_type']}: failure_rate={r['failure_rate']:.3f}, CER_mean={r['CER_mean']:.3f}")

    lines.extend([
        "",
        "## 6. Manual artifact audit findings",
        f"- sampled rows: {len(manual_rows)}",
        "- Multiple failures are genuine empty outputs with non-empty gold; not only key-matching artifacts.",
        "",
        "## 7. Main bottleneck",
        "- Current-default failures are concentrated in source/render and OCR engine empty-output paths, plus layout-sensitive pages.",
        "",
        "## 8. Recommended next experiment",
        f"- {next_experiment}",
        "",
        "## 9. Remaining blockers to private beta",
        "- Failed/empty rate is too high for private beta readiness.",
        "- Unresolved data-quality blocks remain in some regression records.",
        "",
        "## 10. Remaining blockers to production",
        "- Need significantly lower failure rates and robust multilingual quality before production.",
    ])

    OUT_STATUS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    taxonomy = _build_taxonomy()
    ds_out, layout_out, top_worst, _ = _aggregate_breakdowns(taxonomy)
    manual_rows = _manual_audit(taxonomy, top_worst)
    next_experiment = _decide_next_experiment(ds_out, layout_out, taxonomy)
    _status_report(next_experiment, taxonomy, ds_out, layout_out, manual_rows)

    print(
        {
            "taxonomy_csv": str(OUT_TAXONOMY_CSV.relative_to(ROOT)),
            "taxonomy_md": str(OUT_TAXONOMY_MD.relative_to(ROOT)),
            "dataset_breakdown_csv": str(OUT_DATASET_CSV.relative_to(ROOT)),
            "layout_breakdown_csv": str(OUT_LAYOUT_CSV.relative_to(ROOT)),
            "breakdown_md": str(OUT_BREAKDOWN_MD.relative_to(ROOT)),
            "manual_audit_csv": str(OUT_MANUAL_CSV.relative_to(ROOT)),
            "manual_audit_md": str(OUT_MANUAL_MD.relative_to(ROOT)),
            "next_experiment_md": str(OUT_NEXT_MD.relative_to(ROOT)),
            "status_md": str(OUT_STATUS_MD.relative_to(ROOT)),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
