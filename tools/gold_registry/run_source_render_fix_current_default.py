#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from production.ensemble_ocr import FortifiedOCREnsemble
from tools.gold_registry.source_input_resolver import resolve_ocr_input

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
AUDIT = REPORTS / "ground_truth_alignment_audit.csv"
PER_PAGE_BEFORE = REPORTS / "adaptive_render_per_page_metrics.csv"
MATRIX_BEFORE = REPORTS / "adaptive_render_benchmark_rerun_after_integrity_fix.csv"
PROFILE_PATH = ROOT / "reports" / "alignment_audit_artifacts" / "paddle_only_profile_adaptive.json"

OUT_TRACE_CSV = REPORTS / "source_render_trace.csv"
OUT_TRACE_MD = REPORTS / "source_render_trace.md"
OUT_OCRD_CSV = REPORTS / "ocrd_source_render_audit.csv"
OUT_OCRD_MD = REPORTS / "ocrd_source_render_audit.md"
OUT_LOCAL_CSV = REPORTS / "local_gold_source_render_audit.csv"
OUT_LOCAL_MD = REPORTS / "local_gold_source_render_audit.md"
OUT_PER_PAGE = REPORTS / "source_render_fix_per_page_metrics.csv"
OUT_MATRIX = REPORTS / "source_render_fix_benchmark_matrix.csv"
OUT_FIX_REPORT = REPORTS / "source_render_fix_report.md"
OUT_TAXONOMY_CSV = REPORTS / "current_default_failure_taxonomy_after_source_render_fix.csv"
OUT_TAXONOMY_MD = REPORTS / "current_default_failure_taxonomy_after_source_render_fix.md"
OUT_BREAKDOWN_MD = REPORTS / "current_default_failure_breakdown_after_source_render_fix.md"
OUT_NEXT_MD = REPORTS / "next_experiment_after_source_render_fix.md"
OUT_FINAL_MD = REPORTS / "source_render_pipeline_fix_report.md"
BENCHMARK_SCOPE_CSV = REPORTS / "current_default_failure_taxonomy_corrected.csv"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _to_int(v: Any, d: int = 0) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return d


def _to_float(v: Any, d: float = 0.0) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return d


def _to_bool(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _q(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])
    s = sorted(vals)
    idx = (len(s) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(s[lo])
    w = idx - lo
    return float(s[lo] + (s[hi] - s[lo]) * w)


def _cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    a = list(ref)
    b = list(hyp)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        ndp = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            ndp[j] = min(dp[j] + 1, ndp[j - 1] + 1, dp[j - 1] + cost)
        dp = ndp
    return dp[-1] / max(len(a), 1)


def _wer(ref: str, hyp: str) -> float:
    ra, rb = ref.split(), hyp.split()
    if not ra:
        return 0.0 if not rb else 1.0
    dp = list(range(len(rb) + 1))
    for i, ca in enumerate(ra, 1):
        ndp = [i] + [0] * len(rb)
        for j, cb in enumerate(rb, 1):
            cost = 0 if ca == cb else 1
            ndp[j] = min(dp[j] + 1, ndp[j - 1] + 1, dp[j - 1] + cost)
        dp = ndp
    return dp[-1] / max(len(ra), 1)


def _collect_current_default_pages() -> list[dict[str, Any]]:
    before = [r for r in _read_csv(PER_PAGE_BEFORE) if str(r.get("profile_id", "")) == "current_default_render"]
    benchmark_rows = _read_csv(BENCHMARK_SCOPE_CSV)
    benchmark_page_ids = {str(r.get("page_id", "")).strip() for r in benchmark_rows if str(r.get("page_id", "")).strip()}
    if benchmark_page_ids:
        before = [r for r in before if str(r.get("page_id", "")).strip() in benchmark_page_ids]
    audit = _read_csv(AUDIT)
    by_page = {str(r.get("page_id", "")).strip(): r for r in audit}
    pages = []
    for r in before:
        pid = str(r.get("page_id", "")).strip()
        ar = by_page.get(pid)
        if not ar:
            continue
        gt_path = ROOT / str(ar.get("ground_truth_text_path", ""))
        gt_text = gt_path.read_text(encoding="utf-8", errors="ignore").strip() if gt_path.exists() else ""
        pages.append({**ar, "before": r, "gold_text": gt_text})
    return pages


def stage1_trace(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in pages:
        resolved = resolve_ocr_input(p)
        row = {
            "page_id": p.get("page_id", ""),
            "dataset_id": p.get("dataset_id", ""),
            "manifest_record_loaded": "true",
            "source_file_path": p.get("source_file", ""),
            "source_file_exists": resolved.source_file_exists,
            "source_file_type": resolved.source_file_type,
            "local_pdf_path": p.get("local_pdf_path", ""),
            "local_image_path": p.get("local_image_path", ""),
            "requested_page_index": resolved.requested_page_index,
            "resolved_page_index": resolved.resolved_page_index,
            "ocr_input_path": resolved.ocr_input_path,
            "ocr_input_type": resolved.ocr_input_type,
            "source_resolution_method": resolved.source_resolution_method,
            "render_or_conversion_status": resolved.render_or_conversion_status,
            "render_or_conversion_warning": resolved.render_or_conversion_warning,
            "rendered_image_width": resolved.image_width,
            "rendered_image_height": resolved.image_height,
            "rendered_image_file_size": resolved.image_file_size,
            "is_blank_or_nearly_blank": resolved.is_blank_or_nearly_blank,
            "ocr_engine_invoked": "false",
            "ocr_output_generated": "false",
            "ocr_output_text_length": 0,
            "final_output_source": "",
            "failure_reason": "",
            "trace_bucket": "",
        }
        if not resolved.ocr_input_path:
            row["failure_reason"] = "missing_source"
            row["trace_bucket"] = "missing_source"
        out.append(row)

    _write_csv(OUT_TRACE_CSV, out, list(out[0].keys()) if out else ["page_id"])
    cnt = Counter(r.get("trace_bucket", "") for r in out)
    OUT_TRACE_MD.write_text(
        "\n".join([
            "# Source Render Trace",
            "",
            f"- total_pages: {len(out)}",
            "## Initial unresolved buckets",
            *[f"- {k or 'resolved_candidate'}: {v}" for k, v in sorted(cnt.items())],
        ])
        + "\n",
        encoding="utf-8",
    )
    return out


def stage2_3_audits(trace_rows: list[dict[str, Any]]) -> None:
    ocrd = [r for r in trace_rows if str(r.get("dataset_id", "")) == "ocrd_gt_vd_sbb"]
    local = [r for r in trace_rows if str(r.get("dataset_id", "")) == "local_gold_pages"]
    fields = list(trace_rows[0].keys()) if trace_rows else ["page_id"]
    _write_csv(OUT_OCRD_CSV, ocrd, fields)
    _write_csv(OUT_LOCAL_CSV, local, fields)

    def _md(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
        warn = Counter(r.get("render_or_conversion_warning", "") for r in rows)
        meth = Counter(r.get("source_resolution_method", "") for r in rows)
        lines = [f"# {title}", "", f"- rows: {len(rows)}", "", "## Resolution methods"]
        lines.extend([f"- {k or 'none'}: {v}" for k, v in sorted(meth.items(), key=lambda x: (-x[1], x[0]))])
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {k or 'none'}: {v}" for k, v in sorted(warn.items(), key=lambda x: (-x[1], x[0]))])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _md(OUT_OCRD_MD, "OCR-D Source Render Audit", ocrd)
    _md(OUT_LOCAL_MD, "Local Gold Source Render Audit", local)


def stage4_5_6_rerun(pages: list[dict[str, Any]], trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace_map = {str(r["page_id"]): r for r in trace_rows}
    ensemble = FortifiedOCREnsemble(profile_path=str(PROFILE_PATH))
    out = []

    def _needs_targeted_rerun(before_row: dict[str, Any], trace_row: dict[str, Any]) -> bool:
        before_failed = _to_bool(before_row.get("failed"))
        before_empty = _to_bool(before_row.get("empty"))
        unresolved = not str(trace_row.get("ocr_input_path", "")).strip()
        warning = str(trace_row.get("render_or_conversion_warning", "")).strip()
        page_adjusted = _to_int(trace_row.get("requested_page_index"), 0) != _to_int(trace_row.get("resolved_page_index"), 0)
        return before_failed or before_empty or unresolved or bool(warning) or page_adjusted

    for p in pages:
        pid = str(p.get("page_id", ""))
        tr = trace_map.get(pid, {})
        before = p.get("before", {})
        resolved_path = str(tr.get("ocr_input_path", ""))
        req_idx = _to_int(tr.get("requested_page_index"), 0)
        page_idx = _to_int(tr.get("resolved_page_index"), req_idx)
        gold = str(p.get("gold_text", ""))
        rerun = _needs_targeted_rerun(before, tr)

        if not rerun:
            text_len = _to_int(before.get("output_text_length"), 0)
            row = {
                "profile_id": "current_default_render_source_fix",
                "dataset_id": p.get("dataset_id", ""),
                "document_id": p.get("document_id", ""),
                "page_id": pid,
                "split_kind": p.get("split_kind", ""),
                "language_primary": p.get("language_primary", "unknown"),
                "script_type": p.get("script_type", "unknown"),
                "document_type": p.get("document_type", "unknown"),
                "layout_type": p.get("layout_type", "unknown"),
                "requested_page_index": req_idx,
                "resolved_page_index": page_idx,
                "ocr_input_path": resolved_path,
                "ocr_input_type": tr.get("ocr_input_type", ""),
                "source_resolution_method": tr.get("source_resolution_method", ""),
                "render_or_conversion_warning": tr.get("render_or_conversion_warning", ""),
                "runtime_ms": 0.0,
                "gold_text_length": len(gold),
                "output_text_length": text_len,
                "CER": _to_float(before.get("CER"), 0.0),
                "WER": _to_float(before.get("WER"), 0.0),
                "failed": "true" if _to_bool(before.get("failed")) else "false",
                "empty": "true" if _to_bool(before.get("empty")) else "false",
                "final_output_source": str(before.get("final_output_source", "baseline_reused")) or "baseline_reused",
                "failure_reason": str(before.get("failure_reason", "")),
            }
            out.append(row)
            continue

        if not resolved_path:
            text = ""
            meta: dict[str, Any] = {"failure_reason": "missing_source", "final_output_source": "none"}
            runtime_ms = 0.0
        else:
            import time
            t0 = time.perf_counter()
            try:
                text, meta = ensemble.extract_page_text(
                    resolved_path,
                    page_idx,
                    preprocessing_profile="auto",
                    diagnostics={},
                    language_hint=str(p.get("language_primary", "unknown")),
                    script_hint=str(p.get("script_type", "unknown")),
                    document_type=str(p.get("document_type", "unknown")),
                )
            except Exception as exc:
                text, meta = "", {"failure_reason": f"ocr_exception:{type(exc).__name__}", "final_output_source": "none"}
            runtime_ms = (time.perf_counter() - t0) * 1000.0

        text = str(text or "").strip()
        empty = (text == "")
        failed = empty and bool(str(meta.get("failure_reason", "")).strip())
        if not str(meta.get("final_output_source", "")).strip():
            meta["final_output_source"] = "ocr_engine_empty" if empty else "ocr_output"

        row = {
            "profile_id": "current_default_render_source_fix",
            "dataset_id": p.get("dataset_id", ""),
            "document_id": p.get("document_id", ""),
            "page_id": pid,
            "split_kind": p.get("split_kind", ""),
            "language_primary": p.get("language_primary", "unknown"),
            "script_type": p.get("script_type", "unknown"),
            "document_type": p.get("document_type", "unknown"),
            "layout_type": p.get("layout_type", "unknown"),
            "requested_page_index": req_idx,
            "resolved_page_index": page_idx,
            "ocr_input_path": resolved_path,
            "ocr_input_type": tr.get("ocr_input_type", ""),
            "source_resolution_method": tr.get("source_resolution_method", ""),
            "render_or_conversion_warning": tr.get("render_or_conversion_warning", ""),
            "runtime_ms": runtime_ms,
            "gold_text_length": len(gold),
            "output_text_length": len(text),
            "CER": _cer(gold, text),
            "WER": _wer(gold, text),
            "failed": "true" if failed else "false",
            "empty": "true" if empty else "false",
            "final_output_source": str(meta.get("final_output_source", "")),
            "failure_reason": str(meta.get("failure_reason", "")),
        }
        out.append(row)

    _write_csv(OUT_PER_PAGE, out, list(out[0].keys()) if out else ["page_id"])

    cers = [_to_float(r["CER"]) for r in out]
    wers = [_to_float(r["WER"]) for r in out]
    rts = [_to_float(r["runtime_ms"]) for r in out]
    summary = {
        "profile_id": "current_default_render_source_fix",
        "page_count": len(out),
        "CER_mean": _mean(cers),
        "CER_median": _q(cers, 0.5),
        "CER_p90": _q(cers, 0.9),
        "WER_mean": _mean(wers),
        "WER_median": _q(wers, 0.5),
        "WER_p90": _q(wers, 0.9),
        "failed_rate": _mean([1.0 if _to_bool(r["failed"]) else 0.0 for r in out]),
        "empty_rate": _mean([1.0 if _to_bool(r["empty"]) else 0.0 for r in out]),
        "runtime_ms_mean": _mean(rts),
        "runtime_ms_median": _q(rts, 0.5),
        "runtime_ms_p90": _q(rts, 0.9),
        "runtime_ms_p95": _q(rts, 0.95),
    }
    _write_csv(OUT_MATRIX, [summary], list(summary.keys()))

    return out


def stage7_taxonomy(after_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in after_rows:
        reason = str(r.get("failure_reason", "")).lower()
        empty = _to_bool(r.get("empty"))
        if "missing_source" in reason:
            cat = "true_missing_source"
        elif "missing_ground_truth" in reason:
            cat = "true_missing_ground_truth"
        elif str(r.get("ocr_input_type", "")) == "missing":
            cat = "unsupported_source_type"
        elif "conversion" in str(r.get("render_or_conversion_warning", "")).lower():
            cat = "image_conversion_failure"
        elif "render_failure" in reason:
            cat = "render_failure"
        elif str(r.get("is_blank_or_nearly_blank", "")) == "true":
            cat = "blank_rendered_image"
        elif "exception" in reason:
            cat = "OCR_engine_exception"
        elif empty:
            cat = "OCR_engine_empty"
        elif "timeout" in reason:
            cat = "timeout"
        elif "annotation" in reason:
            cat = "annotation_conversion_issue"
        elif _to_float(r.get("CER"), 0.0) > 1.2:
            cat = "recognition_quality_issue"
        else:
            cat = "unknown"
        out.append({**r, "failure_category": cat})

    _write_csv(OUT_TAXONOMY_CSV, out, list(out[0].keys()) if out else ["page_id"])
    cnt = Counter(r["failure_category"] for r in out)
    OUT_TAXONOMY_MD.write_text(
        "\n".join([
            "# Failure Taxonomy After Source Render Fix",
            "",
            *[f"- {k}: {v}" for k, v in sorted(cnt.items(), key=lambda x: (-x[1], x[0]))],
        ])
        + "\n",
        encoding="utf-8",
    )

    by_ds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in out:
        by_ds[str(r.get("dataset_id", "unknown"))].append(r)
    lines = ["# Breakdown After Source Render Fix", ""]
    for ds, rows in sorted(by_ds.items()):
        fr = _mean([1.0 if _to_bool(x.get("failed")) else 0.0 for x in rows])
        er = _mean([1.0 if _to_bool(x.get("empty")) else 0.0 for x in rows])
        lines.append(f"- {ds}: failure_rate={fr:.3f}, empty_rate={er:.3f}, CER_mean={_mean([_to_float(x.get('CER')) for x in rows]):.3f}")
    OUT_BREAKDOWN_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def stage8_decision(tax_rows: list[dict[str, Any]]) -> str:
    cnt = Counter(r.get("failure_category", "unknown") for r in tax_rows)
    source_related = sum(cnt.get(k, 0) for k in ["true_missing_source", "image_conversion_failure", "render_failure", "blank_rendered_image"])
    if source_related >= max(cnt.get("recognition_quality_issue", 0), cnt.get("annotation_conversion_issue", 0)):
        decision = "continue_source_render_cleanup"
        reason = "Source/render related categories still dominate after fix pass."
    elif cnt.get("layout_or_reading_order_issue", 0) > cnt.get("recognition_quality_issue", 0):
        decision = "switch_to_layout_detection"
        reason = "Layout-related failures dominate among valid inputs."
    else:
        decision = "switch_to_engine_model_evaluation"
        reason = "OCR runs on valid inputs but recognition quality remains main bottleneck."

    OUT_NEXT_MD.write_text(
        "\n".join([
            "# Next Experiment After Source Render Fix",
            "",
            f"- decision: {decision}",
            f"- rationale: {reason}",
            "- render_dpi_revisit: no",
        ])
        + "\n",
        encoding="utf-8",
    )
    return decision


def stage9_final(decision: str, after_rows: list[dict[str, Any]], tax_rows: list[dict[str, Any]]) -> None:
    before_row = next((r for r in _read_csv(MATRIX_BEFORE) if r.get("profile_id") == "current_default_render"), {})
    after_row = _read_csv(OUT_MATRIX)[0] if OUT_MATRIX.exists() else {}
    before_pages = [
        r for r in _read_csv(PER_PAGE_BEFORE) if str(r.get("profile_id", "")) == "current_default_render"
    ]

    def _ds_rate(rows: list[dict[str, Any]], ds: str, field: str = "failed") -> float:
        ds_rows = [r for r in rows if str(r.get("dataset_id", "")) == ds]
        return _mean([1.0 if _to_bool(x.get(field)) else 0.0 for x in ds_rows]) if ds_rows else 0.0

    lines = [
        "# Source Render Pipeline Fix Report",
        "",
        "## 1. Executive summary",
        "- Ran source/render-focused reliability pass on current default only.",
        "",
        "## 2. Why source/render was selected",
        "- Prior corrected taxonomy showed source_render_issue concentration with empty outputs.",
        "",
        "## 3. Source/render trace methodology",
        f"- See {OUT_TRACE_CSV.relative_to(ROOT)} and {OUT_TRACE_MD.relative_to(ROOT)}.",
        "",
        "## 4. OCR-D audit findings",
        f"- See {OUT_OCRD_CSV.relative_to(ROOT)} and {OUT_OCRD_MD.relative_to(ROOT)}.",
        "",
        "## 5. local_gold audit findings",
        f"- See {OUT_LOCAL_CSV.relative_to(ROOT)} and {OUT_LOCAL_MD.relative_to(ROOT)}.",
        "",
        "## 6. Source resolver changes",
        "- Added general source resolver supporting PDF and image-to-PDF normalization with page-index correction for single-page PDFs.",
        "",
        "## 7. OCR invocation/output parsing changes",
        "- Ensured resolved inputs are always used and final_output_source is never blank when input was attempted.",
        "",
        "## 8. Before vs after benchmark",
        f"- before_failed_rate: {_to_float(before_row.get('failed_rate')):.6f}",
        f"- after_failed_rate: {_to_float(after_row.get('failed_rate')):.6f}",
        f"- before_empty_rate: {_to_float(before_row.get('empty_rate')):.6f}",
        f"- after_empty_rate: {_to_float(after_row.get('empty_rate')):.6f}",
        f"- before_ocrd_failed_rate: {_ds_rate(before_pages, 'ocrd_gt_vd_sbb'):.3f}",
        f"- after_ocrd_failed_rate: {_ds_rate(after_rows, 'ocrd_gt_vd_sbb'):.3f}",
        f"- before_local_gold_failed_rate: {_ds_rate(before_pages, 'local_gold_pages'):.3f}",
        f"- after_local_gold_failed_rate: {_ds_rate(after_rows, 'local_gold_pages'):.3f}",
        "",
        "## 9. Failure taxonomy after fixes",
        f"- See {OUT_TAXONOMY_CSV.relative_to(ROOT)}, {OUT_TAXONOMY_MD.relative_to(ROOT)}, {OUT_BREAKDOWN_MD.relative_to(ROOT)}.",
        "",
        "## 10. Remaining blockers",
        "- Remaining empty/failed pages still need explicit source/data/engine categorization.",
        "",
        "## 11. Readiness impact",
        "- Do not claim private beta or production readiness if failed/empty remains high.",
        "",
        "## 12. Recommended next experiment",
        f"- {decision}",
    ]
    OUT_FINAL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    OUT_FIX_REPORT.write_text(
        "\n".join([
            "# Source Render Fix Benchmark Report",
            "",
            f"- before_failed_rate: {_to_float(before_row.get('failed_rate')):.6f}",
            f"- after_failed_rate: {_to_float(after_row.get('failed_rate')):.6f}",
            f"- before_empty_rate: {_to_float(before_row.get('empty_rate')):.6f}",
            f"- after_empty_rate: {_to_float(after_row.get('empty_rate')):.6f}",
        ])
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    pages = _collect_current_default_pages()
    trace_rows = stage1_trace(pages)
    stage2_3_audits(trace_rows)
    after_rows = stage4_5_6_rerun(pages, trace_rows)
    tax_rows = stage7_taxonomy(after_rows)
    decision = stage8_decision(tax_rows)
    stage9_final(decision, after_rows, tax_rows)

    print(
        json.dumps(
            {
                "source_render_trace_csv": str(OUT_TRACE_CSV.relative_to(ROOT)),
                "source_render_trace_md": str(OUT_TRACE_MD.relative_to(ROOT)),
                "ocrd_source_render_audit_csv": str(OUT_OCRD_CSV.relative_to(ROOT)),
                "ocrd_source_render_audit_md": str(OUT_OCRD_MD.relative_to(ROOT)),
                "local_gold_source_render_audit_csv": str(OUT_LOCAL_CSV.relative_to(ROOT)),
                "local_gold_source_render_audit_md": str(OUT_LOCAL_MD.relative_to(ROOT)),
                "source_render_fix_benchmark_matrix_csv": str(OUT_MATRIX.relative_to(ROOT)),
                "source_render_fix_per_page_metrics_csv": str(OUT_PER_PAGE.relative_to(ROOT)),
                "source_render_fix_report_md": str(OUT_FIX_REPORT.relative_to(ROOT)),
                "failure_taxonomy_after_fix_csv": str(OUT_TAXONOMY_CSV.relative_to(ROOT)),
                "failure_taxonomy_after_fix_md": str(OUT_TAXONOMY_MD.relative_to(ROOT)),
                "failure_breakdown_after_fix_md": str(OUT_BREAKDOWN_MD.relative_to(ROOT)),
                "next_experiment_after_fix_md": str(OUT_NEXT_MD.relative_to(ROOT)),
                "final_source_render_pipeline_fix_report_md": str(OUT_FINAL_MD.relative_to(ROOT)),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
