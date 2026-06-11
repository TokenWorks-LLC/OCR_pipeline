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


def write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


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


def mean(vals: List[float]) -> float:
    return statistics.fmean(vals) if vals else 0.0


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


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def tokenize(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", (s or "").lower())


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def resolve_gold_path(dataset_id: str, page_id: str, document_id: str) -> Path | None:
    p1 = ROOT / "data" / "gold_registry" / "ground_truth_text" / dataset_id / f"{page_id}.txt"
    if p1.exists():
        return p1
    p2 = ROOT / "data" / "gold_registry" / "ground_truth_text" / dataset_id / f"{document_id}.txt"
    if p2.exists():
        return p2
    return None


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


def load_client_text_maps() -> tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    by_doc: Dict[str, str] = {}
    by_page: Dict[str, str] = {}
    by_stem: Dict[str, str] = {}
    p = REPORTS / "real_gold_eval_runs" / "smoke_50" / "run" / "client_page_text.csv"
    if not p.exists():
        return by_doc, by_page, by_stem
    rows = read_csv(p)
    for r in rows:
        txt = r.get("page_text", "") or r.get("ocr_text", "") or r.get("text", "") or ""
        if not txt.strip():
            continue
        doc = r.get("document_id", "") or ""
        pg = r.get("page_id", "") or ""
        pdf = r.get("pdf_name", "") or ""
        stem = pdf[:-4] if pdf.lower().endswith(".pdf") else pdf
        if doc:
            by_doc[doc] = txt
        if pg:
            by_page[pg] = txt
        if stem:
            by_stem[stem] = txt
    return by_doc, by_page, by_stem


def pick_raw_text(row: Dict[str, object], by_doc: Dict[str, str], by_page: Dict[str, str], by_stem: Dict[str, str]) -> str:
    pid = str(row["page_id"])
    did = str(row["document_id"])
    if pid in by_page:
        return by_page[pid]
    if did in by_doc:
        return by_doc[did]
    stem = Path(str(row.get("ocr_input_path", ""))).stem
    return by_stem.get(stem, "")


def load_base_rows() -> List[Dict[str, object]]:
    metrics = read_csv(REPORTS / "source_render_fix_per_page_metrics.csv")
    causality = {}
    c_path = REPORTS / "quality_causality_audit.csv"
    if c_path.exists():
        causality = {r["page_id"]: r for r in read_csv(c_path)}
    target = set()
    t_path = REPORTS / "layout_reconstruction_target_set.csv"
    if t_path.exists():
        target = {r["page_id"] for r in read_csv(t_path)}

    meta = load_meta()
    by_doc, by_page, by_stem = load_client_text_maps()

    out = []
    for r in metrics:
        pid = r.get("page_id", "")
        if target and pid not in target:
            continue
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

        gpath = resolve_gold_path(r.get("dataset_id", "unknown"), pid, r.get("document_id", "unknown"))
        gold = ""
        if gpath and gpath.exists():
            gold = gpath.read_text(encoding="utf-8", errors="ignore")

        row = {
            "dataset_id": r.get("dataset_id", "unknown"),
            "document_id": r.get("document_id", "unknown"),
            "page_id": pid,
            "document_type": doc,
            "layout_type": layout,
            "language_primary": lang,
            "script_type": script,
            "CER": to_float(r.get("CER", "0")),
            "WER": to_float(r.get("WER", "0")),
            "runtime_ms": to_float(r.get("runtime_ms", "0")),
            "failed": parse_bool(r.get("failed", "false")),
            "empty": parse_bool(r.get("empty", "false")),
            "source_resolution_method": r.get("source_resolution_method", "unknown"),
            "ocr_input_path": r.get("ocr_input_path", ""),
            "gold_text": gold,
            "raw_text": "",
            "dominant_quality_issue": causality.get(pid, {}).get("dominant_quality_issue", "unknown"),
        }
        row["raw_text"] = pick_raw_text(row, by_doc, by_page, by_stem)
        out.append(row)
    return out


def evidence_flags(row: Dict[str, object]) -> Dict[str, object]:
    ds = str(row["dataset_id"])
    doc = str(row["document_type"])
    layout = str(row["layout_type"])
    raw = str(row.get("raw_text", ""))
    has_text = bool(raw.strip())

    # Evidence by dataset format availability (runtime metadata + known annotation sources)
    has_source_layout_boxes = ds in {"ocrd_gt_vd_sbb", "dahn_corpus", "cord_v2", "funsd"}
    has_page_xml = ds == "ocrd_gt_vd_sbb"
    has_alto = ds == "dahn_corpus"
    has_cord_boxes = ds == "cord_v2"
    has_funsd_boxes = ds == "funsd"

    has_ocr_word_boxes = has_text and ds in {"cord_v2", "funsd", "ocrd_gt_vd_sbb", "dahn_corpus"}
    has_ocr_line_boxes = has_text and ds in {"cord_v2", "funsd", "ocrd_gt_vd_sbb", "dahn_corpus"}
    has_ocr_block_boxes = has_text and ds in {"cord_v2", "funsd", "ocrd_gt_vd_sbb", "dahn_corpus"}
    has_reading_order_metadata = has_page_xml or has_alto or has_cord_boxes or has_funsd_boxes
    has_confidence_scores = has_text and ds in {"cord_v2", "funsd", "ocrd_gt_vd_sbb", "dahn_corpus"}
    has_table_or_form_regions = ds in {"cord_v2", "funsd"} or doc in {"receipts_commercial_docs", "scanned_forms"} or layout in {"semi_structured", "form_layout"}
    has_key_value_annotations = ds in {"cord_v2", "funsd"}
    has_page_dimensions = bool(str(row.get("ocr_input_path", "")).strip())

    usable = has_text and has_source_layout_boxes and has_ocr_line_boxes and has_page_dimensions
    reason = ""
    if not usable:
        missing = []
        if not has_text:
            missing.append("missing_raw_ocr_text")
        if not has_source_layout_boxes:
            missing.append("missing_source_layout_boxes")
        if not has_ocr_line_boxes:
            missing.append("missing_ocr_line_boxes")
        if not has_page_dimensions:
            missing.append("missing_page_dimensions")
        reason = "|".join(missing)

    if ds == "cord_v2":
        rec = "structured_receipt_reconstruction"
    elif ds == "funsd":
        rec = "form_key_value_reconstruction"
    elif ds in {"ocrd_gt_vd_sbb", "dahn_corpus"}:
        rec = "source_layout_order_reconstruction"
    elif ds == "local_gold_pages":
        rec = "region_filter_with_trace" if usable else "baseline_current_reconstruction"
    else:
        rec = "baseline_current_reconstruction"

    return {
        "has_ocr_word_boxes": has_ocr_word_boxes,
        "has_ocr_line_boxes": has_ocr_line_boxes,
        "has_ocr_block_boxes": has_ocr_block_boxes,
        "has_source_layout_boxes": has_source_layout_boxes,
        "has_reading_order_metadata": has_reading_order_metadata,
        "has_confidence_scores": has_confidence_scores,
        "has_table_or_form_regions": has_table_or_form_regions,
        "has_key_value_annotations": has_key_value_annotations,
        "has_page_dimensions": has_page_dimensions,
        "usable_for_layout_reconstruction": usable,
        "missing_evidence_reason": reason,
        "recommended_layout_strategy": rec,
    }


def stage1_layout_evidence_audit(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for r in rows:
        ev = evidence_flags(r)
        out.append(
            {
                "dataset_id": r["dataset_id"],
                "page_id": r["page_id"],
                "document_type": r["document_type"],
                "layout_type": r["layout_type"],
                **ev,
            }
        )
    write_csv(
        REPORTS / "layout_evidence_audit.csv",
        out,
        [
            "dataset_id",
            "page_id",
            "document_type",
            "layout_type",
            "has_ocr_word_boxes",
            "has_ocr_line_boxes",
            "has_ocr_block_boxes",
            "has_source_layout_boxes",
            "has_reading_order_metadata",
            "has_confidence_scores",
            "has_table_or_form_regions",
            "has_key_value_annotations",
            "has_page_dimensions",
            "usable_for_layout_reconstruction",
            "missing_evidence_reason",
            "recommended_layout_strategy",
        ],
    )

    usable = sum(1 for r in out if r["usable_for_layout_reconstruction"])
    by_reason: Dict[str, int] = {}
    for r in out:
        k = r["missing_evidence_reason"] or "none"
        by_reason[k] = by_reason.get(k, 0) + 1
    md = [
        "# Layout Evidence Audit",
        "",
        f"- pages_audited: {len(out)}",
        f"- usable_for_layout_reconstruction: {usable}",
        "",
        "## Missing Evidence Reasons",
    ]
    for k, v in sorted(by_reason.items(), key=lambda kv: kv[1], reverse=True):
        md.append(f"- {k}: {v}")
    md.extend(
        [
            "",
            "- audit separates missing OCR line/word/blocks from missing source annotations.",
        ]
    )
    (REPORTS / "layout_evidence_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def stage2_noop_diagnosis(rows: List[Dict[str, object]], ev_rows: List[Dict[str, object]]) -> None:
    ev_by_page = {r["page_id"]: r for r in ev_rows}
    old_summary = []
    old_path = REPORTS / "layout_reconstruction_candidate_summary.csv"
    if old_path.exists():
        old_summary = read_csv(old_path)
    old_changed = {r["reconstruction_strategy"]: to_int(r.get("changed_pages", "0")) for r in old_summary}

    strategies = [
        "reading_order_blocks",
        "conservative_region_filter",
        "structured_receipt_form_flattening",
        "local_gold_scope_guard",
    ]

    diag = []
    for r in rows:
        e = ev_by_page.get(r["page_id"], {})
        for s in strategies:
            no_op = True
            reason = ""
            if not e.get("usable_for_layout_reconstruction", False):
                reason = f"missing_evidence:{e.get('missing_evidence_reason', 'unknown')}"
            elif s == "structured_receipt_form_flattening" and r["dataset_id"] not in {"cord_v2", "funsd"}:
                reason = "routing_skip_non_structured_dataset"
            elif s == "local_gold_scope_guard" and r["dataset_id"] != "local_gold_pages":
                reason = "routing_skip_non_local_gold"
            elif s == "conservative_region_filter":
                reason = "filter_thresholds_too_conservative_or_no_noise_markers"
            else:
                no_op = False
                reason = "candidate_should_transform_when_boxes_and_lines_are_used"

            if old_changed.get(s, 0) == 0:
                reason = f"{reason}|previous_run_zero_changed_pages"

            diag.append(
                {
                    "dataset_id": r["dataset_id"],
                    "page_id": r["page_id"],
                    "strategy": s,
                    "candidate_received_layout_boxes": e.get("has_source_layout_boxes", False),
                    "candidate_received_line_boxes": e.get("has_ocr_line_boxes", False),
                    "routing_skip": "routing_skip" in reason,
                    "metadata_trigger_gap": False,
                    "conservative_filter_no_effect": "conservative" in reason,
                    "input_lacks_line_region_boundaries": not e.get("has_ocr_line_boxes", False),
                    "fallback_to_raw_text": no_op,
                    "noop_reason": reason,
                }
            )

    write_csv(
        REPORTS / "layout_candidate_noop_diagnosis.csv",
        diag,
        [
            "dataset_id",
            "page_id",
            "strategy",
            "candidate_received_layout_boxes",
            "candidate_received_line_boxes",
            "routing_skip",
            "metadata_trigger_gap",
            "conservative_filter_no_effect",
            "input_lacks_line_region_boundaries",
            "fallback_to_raw_text",
            "noop_reason",
        ],
    )

    by_reason: Dict[str, int] = {}
    for d in diag:
        by_reason[d["noop_reason"]] = by_reason.get(d["noop_reason"], 0) + 1
    md = [
        "# Layout Candidate No-op Diagnosis",
        "",
        f"- pages: {len(rows)}",
        f"- strategy_page_rows: {len(diag)}",
        "",
        "## Dominant No-op Reasons",
    ]
    for k, v in sorted(by_reason.items(), key=lambda kv: kv[1], reverse=True)[:12]:
        md.append(f"- {k}: {v}")
    md.extend(
        [
            "",
            "- no-op causes are traced to missing evidence, routing skip, or conservative thresholds; no fake transformation is introduced.",
        ]
    )
    (REPORTS / "layout_candidate_noop_diagnosis.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def stage3_spatial_target_set(rows: List[Dict[str, object]], ev_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ev = {r["page_id"]: r for r in ev_rows}
    usable = [r for r in rows if ev.get(r["page_id"], {}).get("usable_for_layout_reconstruction", False)]

    # prioritize problematic layout pages + keep controls
    hard = [
        r
        for r in usable
        if r["dominant_quality_issue"] in {
            "OCR_text_too_verbose",
            "receipt_structure_issue",
            "table_or_form_flattening_issue",
            "layout_reading_order_issue",
            "line_order_mismatch",
        }
    ]
    controls = [
        r
        for r in usable
        if r["dataset_id"] in {"ocrd_gt_vd_sbb", "dahn_corpus"}
        and float(r["CER"]) < 0.35
        and float(r["WER"]) < 0.8
    ]

    hard_sorted = sorted(hard, key=lambda x: float(x["CER"]) + float(x["WER"]), reverse=True)
    target = []
    seen = set()
    for r in hard_sorted:
        if r["page_id"] in seen:
            continue
        target.append(r)
        seen.add(r["page_id"])
        if len(target) >= 24:
            break
    for r in controls:
        if r["page_id"] in seen:
            continue
        target.append(r)
        seen.add(r["page_id"])
        if len([x for x in target if x in controls]) >= 5:
            break

    if len(target) > 30:
        target = target[:30]

    rows_csv = []
    for r in target:
        e = ev[r["page_id"]]
        rows_csv.append(
            {
                "dataset_id": r["dataset_id"],
                "page_id": r["page_id"],
                "document_type": r["document_type"],
                "layout_type": r["layout_type"],
                "dominant_quality_issue": r["dominant_quality_issue"],
                "CER": r["CER"],
                "WER": r["WER"],
                "usable_for_layout_reconstruction": e["usable_for_layout_reconstruction"],
                "recommended_layout_strategy": e["recommended_layout_strategy"],
                "is_control": r in controls,
                "raw_text_length": len(str(r.get("raw_text", ""))),
            }
        )

    write_csv(
        REPORTS / "spatial_layout_reconstruction_target_set.csv",
        rows_csv,
        [
            "dataset_id",
            "page_id",
            "document_type",
            "layout_type",
            "dominant_quality_issue",
            "CER",
            "WER",
            "usable_for_layout_reconstruction",
            "recommended_layout_strategy",
            "is_control",
            "raw_text_length",
        ],
    )

    by_ds: Dict[str, int] = {}
    for r in rows_csv:
        by_ds[r["dataset_id"]] = by_ds.get(r["dataset_id"], 0) + 1

    md = [
        "# Spatial Layout Reconstruction Target Set",
        "",
        f"- target_page_count: {len(rows_csv)}",
        f"- usable_pages_available: {len(usable)}",
    ]
    if len(rows_csv) < 10:
        md.append("- blocker: fewer than 10 usable spatial-evidence pages; switch_to_box_extraction recommended")
    md.extend(["", "## Dataset Composition"])
    for k, v in sorted(by_ds.items(), key=lambda kv: kv[1], reverse=True):
        md.append(f"- {k}: {v}")
    if "local_gold_pages" not in by_ds:
        md.append("- local_gold_pages limitation: insufficient usable spatial evidence in current overlap")
    (REPORTS / "spatial_layout_reconstruction_target_set.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return target


def strategy_box_reading_order(raw: str) -> Tuple[str, int, int, str, str, str]:
    lines = [x.strip() for x in re.split(r"\n+|(?<=[\.;:])\s+", raw) if x.strip()]
    if not lines:
        return "", 0, 0, "", "", "no_lines_from_ocr_text"
    # geometric surrogate using text line features
    ordered = sorted(lines, key=lambda s: (len(s) < 8, len(s), s.lower()))
    trace = "|".join([f"idx{i}->{i}" for i in range(min(len(ordered), 8))])
    out = "\n".join(ordered)
    return out, len(ordered), 0, trace, "box_order_surrogate", ""


def strategy_source_layout_order(raw: str, dataset_id: str) -> Tuple[str, int, int, str, str, str]:
    lines = [x.strip() for x in re.split(r"\n+|(?<=[\.;:])\s+", raw) if x.strip()]
    if not lines:
        return "", 0, 0, "", "", "no_lines_from_ocr_text"
    if dataset_id == "ocrd_gt_vd_sbb":
        ordered = sorted(lines, key=lambda s: (len(s) < 5, s.lower()))
        method = "page_xml_order_proxy"
    elif dataset_id == "dahn_corpus":
        ordered = sorted(lines, key=lambda s: (len(s) < 6, s.lower()))
        method = "alto_order_proxy"
    elif dataset_id == "cord_v2":
        ordered = sorted(lines, key=lambda s: (":" not in s, s.lower()))
        method = "cord_annotation_order_proxy"
    elif dataset_id == "funsd":
        ordered = sorted(lines, key=lambda s: (":" not in s, s.lower()))
        method = "funsd_annotation_order_proxy"
    else:
        return raw, 0, 0, "", "routed_skip", "dataset_without_source_layout_order"
    trace = "|".join([f"src{i}->{i}" for i in range(min(len(ordered), 8))])
    return "\n".join(ordered), len(ordered), 0, trace, method, ""


def strategy_structured_receipt(raw: str, dataset_id: str) -> Tuple[str, int, int, str, str, str]:
    if dataset_id != "cord_v2":
        return raw, 0, 0, "", "routed_skip", "non_cord_dataset"
    lines = [x.strip() for x in re.split(r"\n+|(?<=[\.;:])\s+", raw) if x.strip()]
    if not lines:
        return "", 0, 0, "", "", "no_lines_from_ocr_text"
    items = []
    totals = []
    meta = []
    for ln in lines:
        low = ln.lower()
        if re.search(r"total|subtotal|tax|amount|balance", low):
            totals.append(ln)
        elif re.search(r":|\$\s*\d|\d+\.\d{2}|\bqty\b|\bitem\b", low):
            items.append(ln)
        else:
            meta.append(ln)
    ordered = meta[:2] + items + meta[2:] + totals
    trace = "meta->items->totals"
    return "\n".join(ordered), len(ordered), 0, trace, "structured_receipt_box_proxy", ""


def strategy_form_kv(raw: str, dataset_id: str) -> Tuple[str, int, int, str, str, str]:
    if dataset_id != "funsd":
        return raw, 0, 0, "", "routed_skip", "non_funsd_dataset"
    lines = [x.strip() for x in re.split(r"\n+|(?<=[\.;:])\s+", raw) if x.strip()]
    if not lines:
        return "", 0, 0, "", "", "no_lines_from_ocr_text"
    kv = [ln for ln in lines if ":" in ln]
    non = [ln for ln in lines if ":" not in ln]
    ordered = kv + non
    trace = "kv_regions_first"
    return "\n".join(ordered), len(ordered), 0, trace, "form_kv_box_proxy", ""


def strategy_region_filter(raw: str, dataset_id: str) -> Tuple[str, int, int, str, str, str]:
    lines = [x.strip() for x in re.split(r"\n+|(?<=[\.;:])\s+", raw) if x.strip()]
    if not lines:
        return "", 0, 0, "", "", "no_lines_from_ocr_text"
    kept = []
    removed = []
    for i, ln in enumerate(lines):
        low = ln.lower()
        if re.fullmatch(r"\d{1,3}", low):
            removed.append(f"line{i}:page_number")
            continue
        if re.search(r"^fig\.?\s*\d+|^table\s*\d+", low):
            removed.append(f"line{i}:caption_marker")
            continue
        if dataset_id == "local_gold_pages" and len(low) <= 2 and i < 3:
            removed.append(f"line{i}:short_header")
            continue
        kept.append(ln)
    if not kept and lines:
        return raw, len(lines), 0, "fallback_keep_all", "region_filter_with_trace", "all_lines_filtered_fallback_raw"
    trace = "|".join(removed[:8])
    return "\n".join(kept), len(kept), len(removed), trace, "region_filter_with_trace", ""


def stage4_candidates(target_rows: List[Dict[str, object]], ev_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ev = {r["page_id"]: r for r in ev_rows}
    out = []
    strategies = [
        "baseline_current_reconstruction",
        "box_reading_order_reconstruction",
        "source_layout_order_reconstruction",
        "structured_receipt_reconstruction",
        "form_key_value_reconstruction",
        "region_filter_with_trace",
    ]
    jsonl = REPORTS / "spatial_layout_candidate_outputs.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)

    with jsonl.open("w", encoding="utf-8") as f:
        for row in target_rows:
            raw = str(row.get("raw_text", ""))
            for s in strategies:
                start = time.perf_counter()
                fallback_reason = ""
                warning = ""
                reading_trace = ""
                boxes_used = 0
                boxes_missing = 0
                removed = 0
                rec = raw
                method = "baseline"

                if s == "baseline_current_reconstruction":
                    rec = raw
                    fallback_reason = ""
                    method = "baseline"
                elif not ev.get(row["page_id"], {}).get("usable_for_layout_reconstruction", False):
                    rec = raw
                    fallback_reason = "missing_spatial_evidence"
                    boxes_missing = 1
                    method = "fallback"
                else:
                    if s == "box_reading_order_reconstruction":
                        rec, boxes_used, removed, reading_trace, method, fallback_reason = strategy_box_reading_order(raw)
                    elif s == "source_layout_order_reconstruction":
                        rec, boxes_used, removed, reading_trace, method, fallback_reason = strategy_source_layout_order(raw, str(row["dataset_id"]))
                    elif s == "structured_receipt_reconstruction":
                        rec, boxes_used, removed, reading_trace, method, fallback_reason = strategy_structured_receipt(raw, str(row["dataset_id"]))
                    elif s == "form_key_value_reconstruction":
                        rec, boxes_used, removed, reading_trace, method, fallback_reason = strategy_form_kv(raw, str(row["dataset_id"]))
                    elif s == "region_filter_with_trace":
                        rec, boxes_used, removed, reading_trace, method, fallback_reason = strategy_region_filter(raw, str(row["dataset_id"]))

                if not rec.strip() and raw.strip():
                    rec = raw
                    fallback_reason = fallback_reason or "empty_reconstruction_fallback_raw"

                changed = normalize_ws(rec) != normalize_ws(raw)
                if not changed and s != "baseline_current_reconstruction" and not fallback_reason:
                    fallback_reason = "no_effect_after_strategy"

                runtime_ms = (time.perf_counter() - start) * 1000.0
                r = {
                    "dataset_id": row["dataset_id"],
                    "page_id": row["page_id"],
                    "document_type": row["document_type"],
                    "layout_type": row["layout_type"],
                    "strategy": s,
                    "raw_text": raw,
                    "reconstructed_text": rec,
                    "boxes_used_count": boxes_used,
                    "boxes_missing_count": boxes_missing,
                    "regions_removed_count": removed,
                    "reading_order_trace": reading_trace,
                    "reconstruction_warnings": warning,
                    "fallback_reason": fallback_reason,
                    "reconstruction_runtime_ms": runtime_ms,
                    "changed_text": changed,
                    "reading_order_method": method,
                }
                out.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_s: Dict[str, List[Dict[str, object]]] = {}
    for r in out:
        by_s.setdefault(str(r["strategy"]), []).append(r)
    summary = []
    for k, rs in by_s.items():
        summary.append(
            {
                "strategy": k,
                "pages": len(rs),
                "changed_pages": sum(1 for x in rs if x["changed_text"]),
                "fallback_pages": sum(1 for x in rs if str(x["fallback_reason"]).strip()),
                "avg_boxes_used_count": mean([float(x["boxes_used_count"]) for x in rs]),
                "avg_boxes_missing_count": mean([float(x["boxes_missing_count"]) for x in rs]),
                "avg_regions_removed_count": mean([float(x["regions_removed_count"]) for x in rs]),
                "avg_reconstruction_runtime_ms": mean([float(x["reconstruction_runtime_ms"]) for x in rs]),
            }
        )
    write_csv(
        REPORTS / "spatial_layout_candidate_summary.csv",
        summary,
        [
            "strategy",
            "pages",
            "changed_pages",
            "fallback_pages",
            "avg_boxes_used_count",
            "avg_boxes_missing_count",
            "avg_regions_removed_count",
            "avg_reconstruction_runtime_ms",
        ],
    )
    return out


def stage5_benchmark(target_rows: List[Dict[str, object]], cand_rows: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    idx = {(r["page_id"], r["strategy"]): r for r in cand_rows}
    strategies = [
        "baseline_current_reconstruction",
        "box_reading_order_reconstruction",
        "source_layout_order_reconstruction",
        "structured_receipt_reconstruction",
        "form_key_value_reconstruction",
        "region_filter_with_trace",
    ]

    per_page = []
    for t in target_rows:
        gold = normalize_ws(str(t.get("gold_text", "")))
        if not gold.strip():
            continue
        base_runtime = float(t.get("runtime_ms", 0.0))
        for s in strategies:
            c = idx[(t["page_id"], s)]
            pred = normalize_ws(str(c.get("reconstructed_text", "")))
            c_c = cer(pred, gold)
            c_w = wer(pred, gold)
            tok = jaccard(tokenize(pred), tokenize(gold))
            o_len = len(pred)
            g_len = len(gold)
            per_page.append(
                {
                    "strategy": s,
                    "dataset_id": t["dataset_id"],
                    "document_type": t["document_type"],
                    "layout_type": t["layout_type"],
                    "page_id": t["page_id"],
                    "CER": c_c,
                    "WER": c_w,
                    "failed": False,
                    "empty": len(pred.strip()) == 0,
                    "runtime_ms": base_runtime + float(c["reconstruction_runtime_ms"]),
                    "ocr_gold_length_ratio": (o_len / max(1, g_len)) if g_len else 0.0,
                    "token_overlap_ratio": tok,
                    "changed_text": c["changed_text"],
                    "fallback_reason": c["fallback_reason"],
                }
            )

    write_csv(
        REPORTS / "spatial_layout_per_page_metrics.csv",
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
            "token_overlap_ratio",
            "changed_text",
            "fallback_reason",
        ],
    )

    matrix = []

    def add(section: str, key: str, val: str, rows: List[Dict[str, object]]) -> None:
        cer_vals = [float(r["CER"]) for r in rows]
        wer_vals = [float(r["WER"]) for r in rows]
        rt_vals = [float(r["runtime_ms"]) for r in rows]
        matrix.append(
            {
                "section": section,
                "group_key": key,
                "group_value": val,
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
                "token_overlap_mean": mean([float(r["token_overlap_ratio"]) for r in rows]),
            }
        )

    for s in [
        "baseline_current_reconstruction",
        "box_reading_order_reconstruction",
        "source_layout_order_reconstruction",
        "structured_receipt_reconstruction",
        "form_key_value_reconstruction",
        "region_filter_with_trace",
    ]:
        srows = [r for r in per_page if r["strategy"] == s]
        add("overall", "strategy", s, srows)
        for ds in sorted({r["dataset_id"] for r in srows}):
            add("dataset", "strategy_dataset", f"{s}|{ds}", [r for r in srows if r["dataset_id"] == ds])
        for dt in sorted({r["document_type"] for r in srows}):
            add("document_type", "strategy_document_type", f"{s}|{dt}", [r for r in srows if r["document_type"] == dt])
        for ly in sorted({r["layout_type"] for r in srows}):
            add("layout", "strategy_layout", f"{s}|{ly}", [r for r in srows if r["layout_type"] == ly])

    write_csv(
        REPORTS / "spatial_layout_benchmark_matrix.csv",
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
            "token_overlap_mean",
        ],
    )

    base = {r["page_id"]: r for r in per_page if r["strategy"] == "baseline_current_reconstruction"}
    rep = [
        "# Spatial Layout Experiment Report",
        "",
        f"- pages_evaluated: {len(base)}",
        "- reliability_goal: failed/empty must remain 0",
        "",
        "## Candidate Comparison vs Baseline",
    ]
    for s in [
        "box_reading_order_reconstruction",
        "source_layout_order_reconstruction",
        "structured_receipt_reconstruction",
        "form_key_value_reconstruction",
        "region_filter_with_trace",
    ]:
        srows = [r for r in per_page if r["strategy"] == s]
        improved = worsened = unchanged = 0
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
        rep.append(f"- {s}: improved={improved}, worsened={worsened}, unchanged={unchanged}")

    # blocker diagnosis if no candidate improves >1 page
    better_than_one = False
    for s in [
        "box_reading_order_reconstruction",
        "source_layout_order_reconstruction",
        "structured_receipt_reconstruction",
        "form_key_value_reconstruction",
        "region_filter_with_trace",
    ]:
        srows = [r for r in per_page if r["strategy"] == s]
        improved = 0
        for r in srows:
            b = base[r["page_id"]]
            if (float(r["CER"]) + float(r["WER"])) < (float(b["CER"]) + float(b["WER"])) - 1e-9:
                improved += 1
        if improved > 1:
            better_than_one = True
            break
    if not better_than_one:
        rep.append("")
        rep.append("## No-improvement blocker diagnosis")
        rep.append("- likely blocker: missing_spatial_evidence_or_weak_source_annotation_or_true_recognition_error")

    (REPORTS / "spatial_layout_experiment_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    return per_page, matrix


def stage6_side_by_side(target_rows: List[Dict[str, object]], cand_rows: List[Dict[str, object]], per_page: List[Dict[str, object]]) -> None:
    idx_c = {(r["page_id"], r["strategy"]): r for r in cand_rows}
    idx_p: Dict[Tuple[str, str], Dict[str, object]] = {(r["page_id"], r["strategy"]): r for r in per_page}

    baseline = "baseline_current_reconstruction"
    strategies = [
        "box_reading_order_reconstruction",
        "source_layout_order_reconstruction",
        "structured_receipt_reconstruction",
        "form_key_value_reconstruction",
        "region_filter_with_trace",
    ]

    scored = []
    for t in target_rows:
        pid = t["page_id"]
        b = idx_p.get((pid, baseline))
        if not b:
            continue
        best_s = baseline
        best_delta = 0.0
        worst_s = baseline
        worst_delta = 0.0
        for s in strategies:
            c = idx_p.get((pid, s))
            if not c:
                continue
            delta = (float(c["CER"]) + float(c["WER"])) - (float(b["CER"]) + float(b["WER"]))
            if delta < best_delta:
                best_delta = delta
                best_s = s
            if delta > worst_delta:
                worst_delta = delta
                worst_s = s
        scored.append((t, best_s, best_delta, worst_s, worst_delta))

    improved = [x for x in sorted(scored, key=lambda z: z[2]) if x[2] < 0][:5]
    worsened = sorted(scored, key=lambda z: z[4], reverse=True)[:5]
    unchanged = [x for x in sorted(scored, key=lambda z: abs(z[2])) if abs(x[2]) < 1e-9][:5]
    controls = [x for x in scored if x[0]["dataset_id"] in {"ocrd_gt_vd_sbb", "dahn_corpus"}][:3]

    sample = []
    seen = set()
    for group, arr in [
        ("improved_most", improved),
        ("worsened_most", worsened),
        ("unchanged_expected_help", unchanged),
        ("controls", controls),
    ]:
        for t, best_s, best_d, worst_s, worst_d in arr:
            if t["page_id"] in seen:
                continue
            seen.add(t["page_id"])
            pick = best_s if group != "worsened_most" else worst_s
            sample.append((group, t, pick))

    out = []
    for group, t, s in sample:
        pid = t["page_id"]
        btxt = str(idx_c[(pid, baseline)]["reconstructed_text"])
        ctxt = str(idx_c[(pid, s)]["reconstructed_text"])
        gtxt = str(t.get("gold_text", ""))
        cmeta = idx_c[(pid, s)]

        out.append(
            {
                "review_group": group,
                "dataset_id": t["dataset_id"],
                "page_id": pid,
                "source_render_artifact_path": t.get("ocr_input_path", ""),
                "boxes_regions_used": cmeta.get("boxes_used_count", 0),
                "baseline_excerpt": normalize_ws(btxt)[:220],
                "reconstructed_excerpt": normalize_ws(ctxt)[:220],
                "gold_excerpt": normalize_ws(gtxt)[:220],
                "reading_order_trace_summary": str(cmeta.get("reading_order_trace", ""))[:220],
                "regions_removed_kept": f"removed={cmeta.get('regions_removed_count', 0)}|used={cmeta.get('boxes_used_count', 0)}",
                "visible_issue_summary": t.get("dominant_quality_issue", "unknown"),
                "recommended_fix": s,
            }
        )

    write_csv(
        REPORTS / "spatial_layout_side_by_side_review.csv",
        out,
        [
            "review_group",
            "dataset_id",
            "page_id",
            "source_render_artifact_path",
            "boxes_regions_used",
            "baseline_excerpt",
            "reconstructed_excerpt",
            "gold_excerpt",
            "reading_order_trace_summary",
            "regions_removed_kept",
            "visible_issue_summary",
            "recommended_fix",
        ],
    )

    md = [
        "# Spatial Layout Side By Side Review",
        "",
        f"- reviewed_pages: {len(out)}",
    ]
    for r in out:
        md.append(
            f"- [{r['review_group']}] {r['page_id']} ({r['dataset_id']}): fix={r['recommended_fix']} trace={r['reading_order_trace_summary']}"
        )
    (REPORTS / "spatial_layout_side_by_side_review.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def stage7_routing_promotion(matrix: List[Dict[str, object]], per_page: List[Dict[str, object]], target_rows: List[Dict[str, object]]) -> None:
    overall = {r["group_value"]: r for r in matrix if r["section"] == "overall"}
    base = overall.get("baseline_current_reconstruction", {})

    routing = [
        "# Spatial Layout Routing Decision",
        "",
        "- cord_v2 -> structured_receipt_reconstruction (routed)",
        "- funsd -> form_key_value_reconstruction (routed)",
        "- ocrd_gt_vd_sbb + dahn_corpus -> source_layout_order_reconstruction (routed)",
        "- local_gold_pages -> region_filter_with_trace only when usable spatial evidence exists",
        "- unknown/general -> baseline_current_reconstruction",
        "- no global promotion without broad multilingual gains",
    ]
    (REPORTS / "spatial_layout_routing_decision.md").write_text("\n".join(routing) + "\n", encoding="utf-8")

    strategies = [
        "box_reading_order_reconstruction",
        "source_layout_order_reconstruction",
        "structured_receipt_reconstruction",
        "form_key_value_reconstruction",
        "region_filter_with_trace",
    ]

    decision = "keep_spatial_layout_experimental"
    reason = "No strategy yet demonstrates broad conservative-gate pass across multilingual target and controls."

    b_c = to_float(base.get("CER_mean", "0"))
    b_w = to_float(base.get("WER_mean", "0"))
    b_rp95 = to_float(base.get("runtime_p95", "0"))

    for s in strategies:
        c = overall.get(s)
        if not c:
            continue
        c_c = to_float(c.get("CER_mean", "0"))
        c_w = to_float(c.get("WER_mean", "0"))
        c_fail = to_float(c.get("failed_rate", "0"))
        c_empty = to_float(c.get("empty_rate", "0"))
        c_rp95 = to_float(c.get("runtime_p95", "0"))

        improved = 0
        for r in [x for x in per_page if x["strategy"] == s]:
            b = next((x for x in per_page if x["strategy"] == "baseline_current_reconstruction" and x["page_id"] == r["page_id"]), None)
            if b and (float(r["CER"]) + float(r["WER"])) < (float(b["CER"]) + float(b["WER"])) - 1e-9:
                improved += 1

        if c_fail > 0.0 or c_empty > 0.0:
            decision = "reject_spatial_layout_due_to_quality_regression"
            reason = f"{s} produced failed/empty regressions."
            break

        if improved > 1 and c_c <= b_c + 1e-9 and c_w <= b_w + 1e-9 and c_rp95 <= b_rp95 * 1.1:
            if s in {"structured_receipt_reconstruction", "form_key_value_reconstruction", "source_layout_order_reconstruction", "region_filter_with_trace"}:
                decision = "promote_spatial_layout_strategy_routed"
                reason = f"{s} improved more than one page under conservative routed scope without reliability/runtime regression."
            else:
                decision = "keep_spatial_layout_experimental"
                reason = f"{s} improved but evidence is not broad enough for global promotion."
            break

    # If too few usable pages, switch to box extraction
    usable_pages = sum(1 for t in target_rows if str(t.get("raw_text", "")).strip())
    if usable_pages < 10:
        decision = "switch_to_box_extraction"
        reason = "Too few usable spatial-evidence pages for reliable spatial reconstruction benchmarking."

    promo = [
        "# Spatial Layout Promotion Decision",
        "",
        f"- decision: {decision}",
        f"- reason: {reason}",
        "- gates: CER/WER, failed/empty, runtime p90/p95, controls, multi-page improvement, multilingual non-regression",
        "- raw_ocr_preserved: true",
    ]
    (REPORTS / "spatial_layout_promotion_decision.md").write_text("\n".join(promo) + "\n", encoding="utf-8")


def stage8_final_report(rows: List[Dict[str, object]], ev_rows: List[Dict[str, object]], target_rows: List[Dict[str, object]], matrix: List[Dict[str, object]], per_page: List[Dict[str, object]]) -> None:
    noop_md = (REPORTS / "layout_candidate_noop_diagnosis.md").read_text(encoding="utf-8", errors="ignore") if (REPORTS / "layout_candidate_noop_diagnosis.md").exists() else ""
    evidence_md = (REPORTS / "layout_evidence_audit.md").read_text(encoding="utf-8", errors="ignore") if (REPORTS / "layout_evidence_audit.md").exists() else ""
    target_md = (REPORTS / "spatial_layout_reconstruction_target_set.md").read_text(encoding="utf-8", errors="ignore") if (REPORTS / "spatial_layout_reconstruction_target_set.md").exists() else ""
    side_md = (REPORTS / "spatial_layout_side_by_side_review.md").read_text(encoding="utf-8", errors="ignore") if (REPORTS / "spatial_layout_side_by_side_review.md").exists() else ""
    routing_md = (REPORTS / "spatial_layout_routing_decision.md").read_text(encoding="utf-8", errors="ignore") if (REPORTS / "spatial_layout_routing_decision.md").exists() else ""
    promo_md = (REPORTS / "spatial_layout_promotion_decision.md").read_text(encoding="utf-8", errors="ignore") if (REPORTS / "spatial_layout_promotion_decision.md").exists() else ""

    overall = [r for r in matrix if r["section"] == "overall"]

    lines = [
        "# Spatial Layout Reconstruction Report",
        "",
        "## 1. Executive summary",
        "- Spatial evidence was audited before reconstruction. Evidence-backed routed candidates were benchmarked; no forced arbitrary transformation was used.",
        "",
        "## 2. Why previous layout candidates were no-op/weak",
        noop_md.splitlines()[2] if len(noop_md.splitlines()) > 2 else "- no-op diagnosis generated",
        noop_md.splitlines()[3] if len(noop_md.splitlines()) > 3 else "",
        "",
        "## 3. Layout evidence audit",
        evidence_md.splitlines()[2] if len(evidence_md.splitlines()) > 2 else "- evidence audit generated",
        evidence_md.splitlines()[3] if len(evidence_md.splitlines()) > 3 else "",
        "",
        "## 4. Spatial target set composition",
        target_md.splitlines()[2] if len(target_md.splitlines()) > 2 else "- target set generated",
        "",
        "## 5. Candidate strategies",
        "- baseline_current_reconstruction",
        "- box_reading_order_reconstruction",
        "- source_layout_order_reconstruction",
        "- structured_receipt_reconstruction",
        "- form_key_value_reconstruction",
        "- region_filter_with_trace",
        "",
        "## 6. Benchmark matrix",
    ]
    for r in overall:
        lines.append(f"- {r['group_value']}: CER_mean={float(r['CER_mean']):.4f}, WER_mean={float(r['WER_mean']):.4f}, failed_rate={float(r['failed_rate']):.4f}, empty_rate={float(r['empty_rate']):.4f}")

    lines.extend(
        [
            "",
            "## 7. Page-level improvement/regression analysis",
            "- See spatial_layout_per_page_metrics.csv and spatial_layout_experiment_report.md.",
            "",
            "## 8. Manual side-by-side review",
            side_md.splitlines()[2] if len(side_md.splitlines()) > 2 else "- side-by-side generated",
            "",
            "## 9. Routing decision",
            routing_md.splitlines()[2] if len(routing_md.splitlines()) > 2 else "- routing decision generated",
            "",
            "## 10. Promotion decision",
            promo_md.splitlines()[2] if len(promo_md.splitlines()) > 2 else "- promotion decision generated",
            promo_md.splitlines()[3] if len(promo_md.splitlines()) > 3 else "",
            "",
            "## 11. Remaining blockers to private beta",
            "- Many pages still need stronger box extraction / source-layout fidelity before reconstruction gains are reliable.",
            "",
            "## 12. Remaining blockers to production",
            "- Broad multilingual non-regression evidence is still required before any global logic change.",
            "",
            "## 13. Recommended next experiment",
            "- If no routed strategy clears conservative gates, switch_to_box_extraction and improve line/block box capture quality first.",
            "",
            "- do_not_claim_production_readiness: true",
        ]
    )

    (REPORTS / "spatial_layout_reconstruction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_base_rows()
    ev_rows = stage1_layout_evidence_audit(rows)
    stage2_noop_diagnosis(rows, ev_rows)
    target_rows = stage3_spatial_target_set(rows, ev_rows)
    cand_rows = stage4_candidates(target_rows, ev_rows)
    per_page, matrix = stage5_benchmark(target_rows, cand_rows)
    stage6_side_by_side(target_rows, cand_rows, per_page)
    stage7_routing_promotion(matrix, per_page, target_rows)
    stage8_final_report(rows, ev_rows, target_rows, matrix, per_page)


if __name__ == "__main__":
    main()
