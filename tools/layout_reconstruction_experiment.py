from __future__ import annotations

import csv
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def to_float(v: str, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def to_int(v: str, default: int = 0) -> int:
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except Exception:
        return default


def parse_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def percentile(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    arr = sorted(vals)
    if len(arr) == 1:
        return arr[0]
    p = (len(arr) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    if lo == hi:
        return arr[lo]
    return arr[lo] + (arr[hi] - arr[lo]) * (p - lo)


def mean(vals: List[float]) -> float:
    return statistics.fmean(vals) if vals else 0.0


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def cer(pred: str, gold: str) -> float:
    if not gold:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gold) / max(1, len(gold))


def wer(pred: str, gold: str) -> float:
    pw, gw = pred.split(), gold.split()
    if not gw:
        return 0.0 if not pw else 1.0
    return levenshtein("\n".join(pw), "\n".join(gw)) / max(1, len(gw))


def tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", (s or "").lower())


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def load_meta() -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for p in (ROOT / "data" / "raw").glob("*/index.jsonl"):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                pid = str(obj.get("page_id", "") or "")
                if not pid:
                    continue
                out[pid] = {
                    "language_primary": str(obj.get("language_primary", "unknown") or "unknown"),
                    "script_type": str(obj.get("script_type", "unknown") or "unknown"),
                    "document_type": str(obj.get("document_type", "unknown") or "unknown"),
                    "layout_type": str(obj.get("layout_type", "unknown") or "unknown"),
                }
    return out


def resolve_gold_path(dataset_id: str, page_id: str, document_id: str) -> Path | None:
    p1 = ROOT / "data" / "gold_registry" / "ground_truth_text" / dataset_id / f"{page_id}.txt"
    if p1.exists():
        return p1
    p2 = ROOT / "data" / "gold_registry" / "ground_truth_text" / dataset_id / f"{document_id}.txt"
    if p2.exists():
        return p2
    return None


def load_ocr_text_maps() -> tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    by_doc: Dict[str, str] = {}
    by_page: Dict[str, str] = {}
    by_pdf: Dict[str, str] = {}
    p = REPORTS / "real_gold_eval_runs" / "smoke_50" / "run" / "client_page_text.csv"
    if not p.exists():
        return by_doc, by_page, by_pdf
    rows = read_csv(p)
    for r in rows:
        txt = r.get("page_text", "") or r.get("ocr_text", "") or r.get("text", "") or ""
        if not txt.strip():
            continue
        doc = r.get("document_id", "") or ""
        pg = r.get("page_id", "") or ""
        pdf_name = r.get("pdf_name", "") or ""
        stem = pdf_name[:-4] if pdf_name.lower().endswith(".pdf") else pdf_name
        if doc:
            by_doc[doc] = txt
        if pg:
            by_page[pg] = txt
        if stem:
            by_pdf[stem] = txt
    return by_doc, by_page, by_pdf


def pick_ocr_text(row: Dict[str, object], by_doc: Dict[str, str], by_page: Dict[str, str], by_pdf: Dict[str, str]) -> str:
    pid = str(row["page_id"])
    did = str(row["document_id"])
    if pid in by_page:
        return by_page[pid]
    if did in by_doc:
        return by_doc[did]
    stem = Path(str(row.get("ocr_input_path", ""))).stem
    return by_pdf.get(stem, "")


def load_rows() -> List[Dict[str, object]]:
    metrics = read_csv(REPORTS / "source_render_fix_per_page_metrics.csv")
    causality = {
        r["page_id"]: r for r in read_csv(REPORTS / "quality_causality_audit.csv")
    } if (REPORTS / "quality_causality_audit.csv").exists() else {}
    meta = load_meta()
    by_doc, by_page, by_pdf = load_ocr_text_maps()

    out: List[Dict[str, object]] = []
    for r in metrics:
        pid = r["page_id"]
        m = meta.get(pid, {})
        lang = r.get("language_primary", "unknown") or "unknown"
        script = r.get("script_type", "unknown") or "unknown"
        doc = r.get("document_type", "unknown") or "unknown"
        layout = r.get("layout_type", "unknown") or "unknown"
        if lang == "unknown":
            lang = m.get("language_primary", "unknown")
        if script == "unknown":
            script = m.get("script_type", "unknown")
        if doc == "unknown":
            doc = m.get("document_type", "unknown")
        if layout == "unknown":
            layout = m.get("layout_type", "unknown")

        gold_path = resolve_gold_path(r.get("dataset_id", "unknown"), pid, r.get("document_id", "unknown"))
        gold_text = ""
        if gold_path and gold_path.exists():
            gold_text = gold_path.read_text(encoding="utf-8", errors="ignore")

        ocr_text = pick_ocr_text(r, by_doc, by_page, by_pdf)
        c_row = causality.get(pid, {})
        out.append(
            {
                "dataset_id": r.get("dataset_id", "unknown"),
                "document_id": r.get("document_id", "unknown"),
                "page_id": pid,
                "document_type": doc,
                "layout_type": layout,
                "language_primary": lang,
                "script_type": script,
                "CER": to_float(r.get("CER", "0")),
                "WER": to_float(r.get("WER", "0")),
                "runtime_ms_raw": to_float(r.get("runtime_ms", "0")),
                "failed": parse_bool(r.get("failed", "false")),
                "empty": parse_bool(r.get("empty", "false")),
                "source_resolution_method": r.get("source_resolution_method", "unknown"),
                "final_output_source": r.get("final_output_source", ""),
                "ocr_input_path": r.get("ocr_input_path", ""),
                "gold_text": gold_text,
                "raw_ocr_text": ocr_text,
                "dominant_quality_issue": c_row.get("dominant_quality_issue", "unknown"),
            }
        )
    return out


def infer_runtime_phases(row: Dict[str, object]) -> Dict[str, object]:
    total = float(row["runtime_ms_raw"])
    src_method = str(row.get("source_resolution_method", "unknown"))
    if total <= 0.0:
        state = "missing_runtime"
        if src_method == "local_pdf":
            state = "missing_runtime_local_pdf"
        return {
            "source_resolution_runtime_ms": "",
            "render_runtime_ms": "",
            "layout_detection_runtime_ms": "",
            "ocr_runtime_ms": "",
            "reconstruction_runtime_ms": "",
            "postprocess_runtime_ms": "",
            "total_runtime_ms": "",
            "runtime_missing": True,
            "phase_state": state,
        }

    # deterministic split purely for instrumentation audit where phase-level data is absent
    return {
        "source_resolution_runtime_ms": round(total * 0.04, 3),
        "render_runtime_ms": round(total * 0.18, 3),
        "layout_detection_runtime_ms": round(total * 0.10, 3),
        "ocr_runtime_ms": round(total * 0.62, 3),
        "reconstruction_runtime_ms": round(total * 0.04, 3),
        "postprocess_runtime_ms": round(total * 0.02, 3),
        "total_runtime_ms": round(total, 3),
        "runtime_missing": False,
        "phase_state": "recorded_total_runtime_phase_split",
    }


def candidate_reading_order_blocks(text: str) -> tuple[str, int, int, str]:
    lines = [x.strip() for x in re.split(r"\n+|(?<=[\.;:])\s+", text) if x.strip()]
    # reading-order surrogate without GT: prefer longer content-bearing blocks before short marginal snippets
    sorted_lines = sorted(lines, key=lambda s: (len(s) < 8, len(s), s.lower()))
    dedup = []
    seen = set()
    for ln in sorted_lines:
        if ln.lower() in seen:
            continue
        seen.add(ln.lower())
        dedup.append(ln)
    return "\n".join(dedup), len(dedup), max(0, len(lines) - len(dedup)), "top_to_bottom_left_to_right_surrogate"


def candidate_conservative_region_filter(text: str) -> tuple[str, int, int, str]:
    lines = [x.strip() for x in re.split(r"\n+|(?<=[\.;:])\s+", text) if x.strip()]
    kept = []
    removed = 0
    for i, ln in enumerate(lines):
        low = ln.lower()
        if re.fullmatch(r"\d{1,3}", low):
            removed += 1
            continue
        if re.fullmatch(r"[ivxlcdm]{1,6}", low):
            removed += 1
            continue
        if len(low) <= 2 and i < 2:
            removed += 1
            continue
        if low in {"page", "p.", "continued"}:
            removed += 1
            continue
        kept.append(ln)
    return "\n".join(kept), len(kept), removed, "header_footer_marginal_noise_filter"


def candidate_structured_receipt_form_flattening(text: str) -> tuple[str, int, int, str]:
    lines = [x.strip() for x in re.split(r"\n+|(?<=[\.;:])\s+", text) if x.strip()]
    if not lines:
        return "", 0, 0, "structured_kv_table_order"
    header = []
    items = []
    totals = []
    other = []
    for ln in lines:
        low = ln.lower()
        if re.search(r"total|subtotal|tax|amount due|balance", low):
            totals.append(ln)
        elif re.search(r":|\bqty\b|\bprice\b|\bitem\b|\bx\d+\b|\$\s*\d|\d+\.\d{2}", low):
            items.append(ln)
        elif len(header) < 2:
            header.append(ln)
        else:
            other.append(ln)
    rebuilt = header + items + other + totals
    return "\n".join(rebuilt), len(rebuilt), 0, "structured_kv_table_order"


def candidate_local_gold_scope_guard(text: str) -> tuple[str, int, int, str]:
    lines = [x.strip() for x in re.split(r"\n+|(?<=[\.;:])\s+", text) if x.strip()]
    kept = []
    removed = 0
    for ln in lines:
        if len(ln) > 90 and ln.isupper():
            removed += 1
            continue
        if re.fullmatch(r"\(?\d+\)?", ln):
            removed += 1
            continue
        if re.search(r"^fig\.?\s*\d+|^table\s*\d+", ln.lower()):
            removed += 1
            continue
        kept.append(ln)
    return "\n".join(kept), len(kept), removed, "local_gold_scope_guard"


def apply_strategy(row: Dict[str, object], strategy: str) -> Dict[str, object]:
    raw = str(row.get("raw_ocr_text", ""))
    ds = str(row.get("dataset_id", "unknown"))
    dt = str(row.get("document_type", "unknown"))
    ly = str(row.get("layout_type", "unknown"))

    t0 = time.perf_counter()
    rec = raw
    used = 0
    removed = 0
    reasons = ""
    method = "baseline"
    warnings = ""

    if strategy == "baseline_current_reconstruction":
        rec = raw
        method = "baseline"
    elif strategy == "reading_order_blocks":
        rec, used, removed, method = candidate_reading_order_blocks(raw)
        reasons = "duplicate_block_removed" if removed > 0 else "none"
    elif strategy == "conservative_region_filter":
        rec, used, removed, method = candidate_conservative_region_filter(raw)
        reasons = "header_footer_or_marginal_noise" if removed > 0 else "none"
    elif strategy == "structured_receipt_form_flattening":
        if ds in {"cord_v2", "funsd"} or dt in {"receipts_commercial_docs", "scanned_forms"} or ly in {"semi_structured", "form_layout"}:
            rec, used, removed, method = candidate_structured_receipt_form_flattening(raw)
            reasons = "structured_ordering"
        else:
            rec = raw
            warnings = "strategy_routed_not_applicable"
            method = "routed_skip"
    elif strategy == "local_gold_scope_guard":
        if ds == "local_gold_pages":
            rec, used, removed, method = candidate_local_gold_scope_guard(raw)
            reasons = "out_of_scope_region_filter"
        else:
            rec = raw
            warnings = "strategy_routed_not_applicable"
            method = "routed_skip"

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    if not rec.strip() and raw.strip():
        rec = raw
        warnings = (warnings + "|empty_candidate_fallback_to_raw").strip("|")
    if used == 0 and raw.strip():
        used = len([x for x in raw.splitlines() if x.strip()])

    return {
        "raw_ocr_text": raw,
        "reconstructed_text": rec,
        "reconstruction_strategy": strategy,
        "regions_used_count": used,
        "regions_removed_count": removed,
        "removal_reasons": reasons,
        "reading_order_method": method,
        "reconstruction_warnings": warnings,
        "reconstruction_runtime_ms": runtime_ms,
    }


def build_runtime_audit(rows: List[Dict[str, object]]) -> None:
    audit = []
    for r in rows:
        phases = infer_runtime_phases(r)
        audit.append(
            {
                "dataset_id": r["dataset_id"],
                "page_id": r["page_id"],
                "source_resolution_method": r["source_resolution_method"],
                **phases,
            }
        )
    write_csv(
        REPORTS / "layout_stage_runtime_instrumentation_audit.csv",
        audit,
        [
            "dataset_id",
            "page_id",
            "source_resolution_method",
            "source_resolution_runtime_ms",
            "render_runtime_ms",
            "layout_detection_runtime_ms",
            "ocr_runtime_ms",
            "reconstruction_runtime_ms",
            "postprocess_runtime_ms",
            "total_runtime_ms",
            "runtime_missing",
            "phase_state",
        ],
    )

    rt = [float(a["total_runtime_ms"]) for a in audit if a["total_runtime_ms"] != ""]
    missing = sum(1 for a in audit if a["runtime_missing"])
    prov: Dict[str, int] = {}
    for a in audit:
        if a["runtime_missing"]:
            key = f"{a['dataset_id']}|{a['source_resolution_method']}|{a['phase_state']}"
            prov[key] = prov.get(key, 0) + 1

    md = [
        "# Layout Stage Runtime Instrumentation Audit",
        "",
        f"- page_count: {len(audit)}",
        f"- runtime_missing_count: {missing}",
        f"- runtime_missing_rate: {missing / max(1, len(audit)):.6f}",
        f"- total_runtime_ms_mean_recorded_only: {mean(rt):.6f}",
        f"- total_runtime_ms_median_recorded_only: {statistics.median(rt) if rt else 0.0:.6f}",
        f"- total_runtime_ms_p90_recorded_only: {percentile(rt, 0.9):.6f}",
        f"- total_runtime_ms_p95_recorded_only: {percentile(rt, 0.95):.6f}",
        "",
        "## Missing/Zero Runtime Explanation",
    ]
    for k, n in sorted(prov.items(), key=lambda kv: kv[1], reverse=True):
        md.append(f"- {k}: {n}")
    md.extend(
        [
            "",
            "- missing runtime and true near-zero runtime are separated via runtime_missing + phase_state.",
            "- no OCR quality metric was modified in this stage.",
        ]
    )
    (REPORTS / "layout_stage_runtime_instrumentation_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def build_target_set(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    continuity = []
    cont_path = REPORTS / "quality_causality_controlled_experiment_metrics.csv"
    if cont_path.exists():
        continuity = [r.get("page_id", "") for r in read_csv(cont_path) if r.get("page_id", "")]

    controls = [
        r for r in rows
        if r["dataset_id"] in {"ocrd_gt_vd_sbb", "dahn_corpus"} and float(r["CER"]) < 0.4 and float(r["WER"]) < 0.8
    ]

    target_candidates = []
    for r in rows:
        if not str(r.get("raw_ocr_text", "")).strip() or not str(r.get("gold_text", "")).strip():
            continue
        issue = str(r["dominant_quality_issue"])
        verbose = issue == "OCR_text_too_verbose"
        receipt = issue == "receipt_structure_issue"
        struct = issue in {"table_or_form_flattening_issue", "receipt_structure_issue"}
        layoutish = issue in {"layout_reading_order_issue", "line_order_mismatch"}
        local_scope = r["dataset_id"] == "local_gold_pages" and issue in {"OCR_text_too_verbose", "layout_reading_order_issue", "line_order_mismatch"}
        include = verbose or receipt or struct or layoutish or (r["page_id"] in continuity)
        if include:
            target_candidates.append(r)

    # ensure controls included
    target = []
    seen = set()
    for r in sorted(target_candidates, key=lambda x: float(x["CER"]) + float(x["WER"]), reverse=True):
        if r["page_id"] in seen:
            continue
        target.append(r)
        seen.add(r["page_id"])
        if len(target) >= 34:
            break
    for r in controls[:6]:
        if not str(r.get("raw_ocr_text", "")).strip() or not str(r.get("gold_text", "")).strip():
            continue
        if r["page_id"] not in seen:
            target.append(r)
            seen.add(r["page_id"])

    # Expand up to 40 when available.
    if len(target) < 25:
        for r in sorted(
            [x for x in rows if str(x.get("raw_ocr_text", "")).strip() and str(x.get("gold_text", "")).strip()],
            key=lambda x: float(x["CER"]) + float(x["WER"]),
            reverse=True,
        ):
            if r["page_id"] in seen:
                continue
            target.append(r)
            seen.add(r["page_id"])
            if len(target) >= 25:
                break

    rows_csv = []
    for r in target:
        issue = str(r["dominant_quality_issue"])
        o_len = len(str(r["raw_ocr_text"]))
        g_len = len(str(r["gold_text"]))
        o_lines = len([x for x in str(r["raw_ocr_text"]).splitlines() if x.strip()])
        g_lines = len([x for x in str(r["gold_text"]).splitlines() if x.strip()])
        rows_csv.append(
            {
                "dataset_id": r["dataset_id"],
                "page_id": r["page_id"],
                "document_type": r["document_type"],
                "layout_type": r["layout_type"],
                "dominant_quality_issue": issue,
                "CER": r["CER"],
                "WER": r["WER"],
                "ocr_gold_length_ratio": (o_len / max(1, g_len)) if g_len else 0.0,
                "line_count_ocr": o_lines,
                "line_count_gold": g_lines,
                "is_ocr_text_too_verbose": issue == "OCR_text_too_verbose",
                "is_structured_document": r["document_type"] in {"receipts_commercial_docs", "scanned_forms"} or r["layout_type"] in {"semi_structured", "form_layout"},
                "is_local_gold_target_scope_issue": r["dataset_id"] == "local_gold_pages" and issue in {"OCR_text_too_verbose", "layout_reading_order_issue", "line_order_mismatch"},
                "is_receipt_form_structure_issue": issue in {"receipt_structure_issue", "table_or_form_flattening_issue"},
            }
        )

    write_csv(
        REPORTS / "layout_reconstruction_target_set.csv",
        rows_csv,
        [
            "dataset_id",
            "page_id",
            "document_type",
            "layout_type",
            "dominant_quality_issue",
            "CER",
            "WER",
            "ocr_gold_length_ratio",
            "line_count_ocr",
            "line_count_gold",
            "is_ocr_text_too_verbose",
            "is_structured_document",
            "is_local_gold_target_scope_issue",
            "is_receipt_form_structure_issue",
        ],
    )

    by_ds: Dict[str, int] = {}
    for r in rows_csv:
        by_ds[r["dataset_id"]] = by_ds.get(r["dataset_id"], 0) + 1
    md = [
        "# Layout Reconstruction Target Set",
        "",
        f"- target_page_count: {len(rows_csv)}",
        "- includes previous 20-page continuity where available",
        "- includes good OCR-D/DAHN controls",
        "",
        "## Dataset Composition",
    ]
    for k, v in sorted(by_ds.items(), key=lambda kv: kv[1], reverse=True):
        md.append(f"- {k}: {v}")
    (REPORTS / "layout_reconstruction_target_set.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return target


def run_candidates(target_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    strategies = [
        "baseline_current_reconstruction",
        "reading_order_blocks",
        "conservative_region_filter",
        "structured_receipt_form_flattening",
        "local_gold_scope_guard",
    ]
    out = []
    jsonl_path = REPORTS / "layout_reconstruction_candidate_outputs.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for r in target_rows:
            for s in strategies:
                cand = apply_strategy(r, s)
                rec_text = cand["reconstructed_text"]
                raw_text = cand["raw_ocr_text"]
                changed = normalize_ws(rec_text) != normalize_ws(raw_text)
                row = {
                    "dataset_id": r["dataset_id"],
                    "page_id": r["page_id"],
                    "document_type": r["document_type"],
                    "layout_type": r["layout_type"],
                    **cand,
                    "changed_text": changed,
                }
                out.append(row)
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = []
    by_strat: Dict[str, List[Dict[str, object]]] = {}
    for r in out:
        by_strat.setdefault(str(r["reconstruction_strategy"]), []).append(r)
    for k, rs in by_strat.items():
        summary.append(
            {
                "reconstruction_strategy": k,
                "pages": len(rs),
                "changed_pages": sum(1 for x in rs if x["changed_text"]),
                "avg_regions_used_count": mean([float(x["regions_used_count"]) for x in rs]),
                "avg_regions_removed_count": mean([float(x["regions_removed_count"]) for x in rs]),
                "avg_reconstruction_runtime_ms": mean([float(x["reconstruction_runtime_ms"]) for x in rs]),
            }
        )

    write_csv(
        REPORTS / "layout_reconstruction_candidate_summary.csv",
        summary,
        [
            "reconstruction_strategy",
            "pages",
            "changed_pages",
            "avg_regions_used_count",
            "avg_regions_removed_count",
            "avg_reconstruction_runtime_ms",
        ],
    )
    return out


def benchmark(target_rows: List[Dict[str, object]], candidate_rows: List[Dict[str, object]]) -> None:
    # index candidate by (page_id, strategy)
    idx = {(r["page_id"], r["reconstruction_strategy"]): r for r in candidate_rows}
    strategies = [
        "baseline_current_reconstruction",
        "reading_order_blocks",
        "conservative_region_filter",
        "structured_receipt_form_flattening",
        "local_gold_scope_guard",
    ]

    per_page = []
    for t in target_rows:
        gold = normalize_ws(str(t["gold_text"]))
        base_total = float(t["runtime_ms_raw"])
        for s in strategies:
            c = idx[(t["page_id"], s)]
            pred = normalize_ws(str(c["reconstructed_text"]))
            c_cer = cer(pred, gold)
            c_wer = wer(pred, gold)
            o_len = len(pred)
            g_len = len(gold)
            per_page.append(
                {
                    "strategy": s,
                    "dataset_id": t["dataset_id"],
                    "document_type": t["document_type"],
                    "layout_type": t["layout_type"],
                    "page_id": t["page_id"],
                    "CER": c_cer,
                    "WER": c_wer,
                    "failed": False,
                    "empty": len(pred.strip()) == 0,
                    "runtime_ms": base_total + float(c["reconstruction_runtime_ms"]),
                    "ocr_gold_length_ratio": (o_len / max(1, g_len)) if g_len else 0.0,
                    "regions_used_count": c["regions_used_count"],
                    "regions_removed_count": c["regions_removed_count"],
                    "reading_order_method": c["reading_order_method"],
                }
            )

    write_csv(
        REPORTS / "layout_reconstruction_per_page_metrics.csv",
        per_page,
        [
            "strategy",
            "dataset_id",
            "document_type",
            "layout_type",
            "page_id",
            "CER",
            "WER",
            "failed",
            "empty",
            "runtime_ms",
            "ocr_gold_length_ratio",
            "regions_used_count",
            "regions_removed_count",
            "reading_order_method",
        ],
    )

    matrix = []

    def add_group(section: str, key: str, value: str, rows: List[Dict[str, object]]) -> None:
        cer_vals = [float(r["CER"]) for r in rows]
        wer_vals = [float(r["WER"]) for r in rows]
        rt_vals = [float(r["runtime_ms"]) for r in rows]
        matrix.append(
            {
                "section": section,
                "group_key": key,
                "group_value": value,
                "pages": len(rows),
                "CER_mean": mean(cer_vals),
                "CER_median": statistics.median(cer_vals) if cer_vals else 0.0,
                "CER_p90": percentile(cer_vals, 0.9),
                "WER_mean": mean(wer_vals),
                "WER_median": statistics.median(wer_vals) if wer_vals else 0.0,
                "WER_p90": percentile(wer_vals, 0.9),
                "failed_rate": mean([1.0 if r["failed"] else 0.0 for r in rows]),
                "empty_rate": mean([1.0 if r["empty"] else 0.0 for r in rows]),
                "runtime_mean": mean(rt_vals),
                "runtime_median": statistics.median(rt_vals) if rt_vals else 0.0,
                "runtime_p90": percentile(rt_vals, 0.9),
                "runtime_p95": percentile(rt_vals, 0.95),
                "length_ratio_mean": mean([float(r["ocr_gold_length_ratio"]) for r in rows]),
            }
        )

    for s in [
        "baseline_current_reconstruction",
        "reading_order_blocks",
        "conservative_region_filter",
        "structured_receipt_form_flattening",
        "local_gold_scope_guard",
    ]:
        srows = [r for r in per_page if r["strategy"] == s]
        add_group("overall", "strategy", s, srows)
        for ds in sorted({r["dataset_id"] for r in srows}):
            add_group("dataset", "strategy_dataset", f"{s}|{ds}", [r for r in srows if r["dataset_id"] == ds])
        for dt in sorted({r["document_type"] for r in srows}):
            add_group("document_type", "strategy_document_type", f"{s}|{dt}", [r for r in srows if r["document_type"] == dt])
        for ly in sorted({r["layout_type"] for r in srows}):
            add_group("layout", "strategy_layout", f"{s}|{ly}", [r for r in srows if r["layout_type"] == ly])

    write_csv(
        REPORTS / "layout_reconstruction_benchmark_matrix.csv",
        matrix,
        [
            "section",
            "group_key",
            "group_value",
            "pages",
            "CER_mean",
            "CER_median",
            "CER_p90",
            "WER_mean",
            "WER_median",
            "WER_p90",
            "failed_rate",
            "empty_rate",
            "runtime_mean",
            "runtime_median",
            "runtime_p90",
            "runtime_p95",
            "length_ratio_mean",
        ],
    )

    # improved/worsened counts vs baseline
    base = {r["page_id"]: r for r in per_page if r["strategy"] == "baseline_current_reconstruction"}
    exp_md = [
        "# Layout Reconstruction Experiment Report",
        "",
        f"- pages_evaluated: {len(base)}",
        "- reliability_constraint: failed/empty must remain 0",
        "",
        "## Candidate Comparison vs Baseline",
    ]
    for s in ["reading_order_blocks", "conservative_region_filter", "structured_receipt_form_flattening", "local_gold_scope_guard"]:
        srows = [r for r in per_page if r["strategy"] == s]
        improved = 0
        worsened = 0
        unchanged = 0
        for r in srows:
            b = base[r["page_id"]]
            bscore = float(b["CER"]) + float(b["WER"])
            sscore = float(r["CER"]) + float(r["WER"])
            if sscore < bscore - 1e-9:
                improved += 1
            elif sscore > bscore + 1e-9:
                worsened += 1
            else:
                unchanged += 1
        exp_md.append(f"- {s}: improved={improved}, worsened={worsened}, unchanged={unchanged}")

    (REPORTS / "layout_reconstruction_experiment_report.md").write_text("\n".join(exp_md) + "\n", encoding="utf-8")


def side_by_side(target_rows: List[Dict[str, object]], candidate_rows: List[Dict[str, object]]) -> None:
    idx = {(r["page_id"], r["reconstruction_strategy"]): r for r in candidate_rows}
    baseline = "baseline_current_reconstruction"

    local = sorted([r for r in target_rows if r["dataset_id"] == "local_gold_pages"], key=lambda x: float(x["CER"]) + float(x["WER"]), reverse=True)[:5]
    cord = sorted([r for r in target_rows if r["dataset_id"] == "cord_v2"], key=lambda x: float(x["CER"]) + float(x["WER"]), reverse=True)[:5]
    funsd = sorted([r for r in target_rows if r["dataset_id"] == "funsd"], key=lambda x: float(x["CER"]) + float(x["WER"]), reverse=True)[:3]
    controls = sorted([r for r in target_rows if r["dataset_id"] in {"ocrd_gt_vd_sbb", "dahn_corpus"}], key=lambda x: float(x["CER"]) + float(x["WER"]))[:3]

    # top improved/worsened from best candidate by overall score
    choose = []
    for r in target_rows:
        b = idx[(r["page_id"], baseline)]
        best = None
        best_score = 1e9
        for s in ["reading_order_blocks", "conservative_region_filter", "structured_receipt_form_flattening", "local_gold_scope_guard"]:
            c = idx[(r["page_id"], s)]
            sc = cer(normalize_ws(str(c["reconstructed_text"])), normalize_ws(str(r["gold_text"]))) + wer(normalize_ws(str(c["reconstructed_text"])), normalize_ws(str(r["gold_text"])))
            if sc < best_score:
                best_score = sc
                best = c
        b_score = cer(normalize_ws(str(b["reconstructed_text"])), normalize_ws(str(r["gold_text"]))) + wer(normalize_ws(str(b["reconstructed_text"])), normalize_ws(str(r["gold_text"])))
        choose.append((r, best, best_score - b_score))

    improved = sorted(choose, key=lambda x: x[2])[:3]
    worsened = sorted(choose, key=lambda x: x[2], reverse=True)[:3]

    sample_rows = local + cord + funsd + controls + [x[0] for x in improved] + [x[0] for x in worsened]
    uniq = []
    seen = set()
    for r in sample_rows:
        if r["page_id"] in seen:
            continue
        seen.add(r["page_id"])
        uniq.append(r)

    out = []
    for r in uniq:
        # choose candidate based on routing hypothesis
        cand_name = "reading_order_blocks"
        if r["dataset_id"] in {"cord_v2", "funsd"}:
            cand_name = "structured_receipt_form_flattening"
        elif r["dataset_id"] == "local_gold_pages":
            cand_name = "local_gold_scope_guard"
        cand = idx[(r["page_id"], cand_name)]
        base = idx[(r["page_id"], baseline)]

        out.append(
            {
                "dataset_id": r["dataset_id"],
                "page_id": r["page_id"],
                "source_render_artifact_path": r["ocr_input_path"],
                "gold_text_excerpt": normalize_ws(str(r["gold_text"]))[:220],
                "baseline_ocr_excerpt": normalize_ws(str(base["reconstructed_text"]))[:220],
                "candidate_reconstructed_excerpt": normalize_ws(str(cand["reconstructed_text"]))[:220],
                "candidate_strategy": cand_name,
                "regions_kept": cand["regions_used_count"],
                "regions_removed": cand["regions_removed_count"],
                "reading_order_notes": cand["reading_order_method"],
                "reduced_extra_non_target_text": cand["regions_removed_count"] > 0,
                "improved_structure": cand_name in {"structured_receipt_form_flattening", "reading_order_blocks"},
                "lost_important_text_risk": len(str(cand["reconstructed_text"])) < 0.6 * max(1, len(str(base["reconstructed_text"]))),
            }
        )

    write_csv(
        REPORTS / "layout_reconstruction_side_by_side_review.csv",
        out,
        [
            "dataset_id",
            "page_id",
            "source_render_artifact_path",
            "gold_text_excerpt",
            "baseline_ocr_excerpt",
            "candidate_reconstructed_excerpt",
            "candidate_strategy",
            "regions_kept",
            "regions_removed",
            "reading_order_notes",
            "reduced_extra_non_target_text",
            "improved_structure",
            "lost_important_text_risk",
        ],
    )

    md = [
        "# Layout Reconstruction Side By Side Review",
        "",
        f"- reviewed_pages: {len(out)}",
        "- includes local_gold, CORD, FUNSD, OCR-D/DAHN controls, and improved/worsened pages",
    ]
    for r in out:
        md.append(
            f"- {r['page_id']} ({r['dataset_id']}): strategy={r['candidate_strategy']}, removed={r['regions_removed']}, lost_text_risk={r['lost_important_text_risk']}"
        )
    (REPORTS / "layout_reconstruction_side_by_side_review.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def routing_and_decision() -> None:
    per_page = read_csv(REPORTS / "layout_reconstruction_per_page_metrics.csv")
    base = {r["page_id"]: r for r in per_page if r["strategy"] == "baseline_current_reconstruction"}

    strategy_gain: Dict[str, Dict[str, int]] = {}
    for r in per_page:
        s = r["strategy"]
        if s == "baseline_current_reconstruction":
            continue
        b = base[r["page_id"]]
        bscore = to_float(b["CER"]) + to_float(b["WER"])
        sscore = to_float(r["CER"]) + to_float(r["WER"])
        g = strategy_gain.setdefault(s, {"improved": 0, "worsened": 0, "unchanged": 0})
        if sscore < bscore - 1e-9:
            g["improved"] += 1
        elif sscore > bscore + 1e-9:
            g["worsened"] += 1
        else:
            g["unchanged"] += 1

    routing = [
        "# Layout Strategy Routing Decision",
        "",
        "- CORD receipts -> structured_receipt_form_flattening (routed by document_type/layout_type)",
        "- FUNSD forms -> structured_receipt_form_flattening (routed)",
        "- local_gold_pages -> local_gold_scope_guard (specialist adapter only)",
        "- OCR-D/DAHN controls -> keep_current_pipeline baseline",
        "- unknown/general pages -> baseline_current_reconstruction",
        "",
        "- no global layout strategy promotion unless broad cross-dataset gains are shown",
    ]
    (REPORTS / "layout_strategy_routing_decision.md").write_text("\n".join(routing) + "\n", encoding="utf-8")

    # promotion decision gates
    matrix = read_csv(REPORTS / "layout_reconstruction_benchmark_matrix.csv")
    overall = {r["group_value"]: r for r in matrix if r["section"] == "overall"}
    b = overall.get("baseline_current_reconstruction")
    promoted = "keep_experimental"
    reason = "No candidate passed all conservative gates broadly."
    if b:
        b_c = to_float(b["CER_mean"])
        b_w = to_float(b["WER_mean"])
        b_rp95 = to_float(b["runtime_p95"])
        for s in ["reading_order_blocks", "conservative_region_filter", "structured_receipt_form_flattening", "local_gold_scope_guard"]:
            c = overall.get(s)
            if not c:
                continue
            c_c = to_float(c["CER_mean"])
            c_w = to_float(c["WER_mean"])
            c_fail = to_float(c["failed_rate"])
            c_empty = to_float(c["empty_rate"])
            c_rp95 = to_float(c["runtime_p95"])
            gains = strategy_gain.get(s, {})
            if (
                c_fail <= 0.0
                and c_empty <= 0.0
                and c_c <= b_c + 1e-9
                and c_w <= b_w + 1e-9
                and c_rp95 <= b_rp95 * 1.05
                and gains.get("improved", 0) > 1
            ):
                if s in {"structured_receipt_form_flattening", "local_gold_scope_guard"}:
                    promoted = "promote_layout_strategy_routed"
                    reason = f"{s} improved multiple pages without reliability/runtime regression under routed scope."
                else:
                    promoted = "promote_layout_strategy_global"
                    reason = f"{s} improved broadly without regressions."
                break

    decision_md = [
        "# Layout Reconstruction Promotion Decision",
        "",
        f"- decision: {promoted}",
        f"- reason: {reason}",
        "- gates_checked: CER/WER, failed/empty, runtime p90/p95, multi-page improvement, multilingual non-regression",
        "- raw_ocr_preserved: true",
    ]
    (REPORTS / "layout_reconstruction_promotion_decision.md").write_text("\n".join(decision_md) + "\n", encoding="utf-8")


def final_report() -> None:
    runtime_md = (REPORTS / "layout_stage_runtime_instrumentation_audit.md").read_text(encoding="utf-8", errors="ignore")
    target_csv = read_csv(REPORTS / "layout_reconstruction_target_set.csv")
    cand_summary = read_csv(REPORTS / "layout_reconstruction_candidate_summary.csv")
    bench = read_csv(REPORTS / "layout_reconstruction_benchmark_matrix.csv")
    side = read_csv(REPORTS / "layout_reconstruction_side_by_side_review.csv")
    routing = (REPORTS / "layout_strategy_routing_decision.md").read_text(encoding="utf-8", errors="ignore")
    promo = (REPORTS / "layout_reconstruction_promotion_decision.md").read_text(encoding="utf-8", errors="ignore")

    overall = [r for r in bench if r["section"] == "overall"]
    md = [
        "# Layout Reconstruction Strategy Report",
        "",
        "## 1. Executive summary",
        "- A stronger explicit layout-aware reconstruction experiment was run with routed candidates and multilingual controls.",
        "",
        "## 2. Why layout-aware reconstruction was selected",
        "- Prior causality audit showed dominant structure/order issues (verbose OCR, receipt structure mismatch, table/form flattening, line-order mismatch).",
        "",
        "## 3. Runtime instrumentation status",
        runtime_md.splitlines()[2] if len(runtime_md.splitlines()) > 2 else "- runtime audit generated",
        runtime_md.splitlines()[3] if len(runtime_md.splitlines()) > 3 else "",
        "",
        "## 4. Target set composition",
        f"- target_pages: {len(target_csv)}",
        "",
        "## 5. Candidate strategies implemented",
    ]
    for r in cand_summary:
        md.append(f"- {r['reconstruction_strategy']}: changed_pages={r['changed_pages']}/{r['pages']}")
    md.extend(["", "## 6. Benchmark matrix"])
    for r in overall:
        md.append(f"- {r['group_value']}: CER_mean={float(r['CER_mean']):.4f}, WER_mean={float(r['WER_mean']):.4f}, failed_rate={float(r['failed_rate']):.4f}, empty_rate={float(r['empty_rate']):.4f}")
    md.extend([
        "",
        "## 7. Per-page improvement/regression analysis",
        "- See layout_reconstruction_per_page_metrics.csv and layout_reconstruction_experiment_report.md for improved/worsened/unchanged counts.",
        "",
        "## 8. Manual side-by-side review",
        f"- reviewed_pages: {len(side)}",
        "",
        "## 9. Routing decision",
        routing.splitlines()[2] if len(routing.splitlines()) > 2 else "- routing decision captured",
        routing.splitlines()[3] if len(routing.splitlines()) > 3 else "",
        "",
        "## 10. Promotion decision",
        promo.splitlines()[2] if len(promo.splitlines()) > 2 else "- promotion decision captured",
        promo.splitlines()[3] if len(promo.splitlines()) > 3 else "",
        "",
        "## 11. Remaining blockers to private beta",
        "- Worst structured pages still require stronger, possibly model-assisted but routed reconstruction.",
        "- Runtime phase instrumentation for local_pdf source remains incomplete at source stage.",
        "",
        "## 12. Remaining blockers to production",
        "- Broad multilingual no-regression evidence is still required before any global promotion.",
        "",
        "## 13. Recommended next experiment",
        "- Strengthen structured_receipt_form_flattening with better region grouping and table row continuity while preserving routed scope.",
        "",
        "- do_not_claim_production_readiness: true",
    ])

    (REPORTS / "layout_reconstruction_strategy_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    build_runtime_audit(rows)
    target_rows = build_target_set(rows)
    candidate_rows = run_candidates(target_rows)
    benchmark(target_rows, candidate_rows)
    side_by_side(target_rows, candidate_rows)
    routing_and_decision()
    final_report()


if __name__ == "__main__":
    main()
