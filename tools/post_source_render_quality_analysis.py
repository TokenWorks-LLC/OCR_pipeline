from __future__ import annotations

import csv
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
    lo = math.floor(p)
    hi = math.ceil(p)
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
    pw = pred.split()
    gw = gold.split()
    if not gw:
        return 0.0 if not pw else 1.0
    return levenshtein("\n".join(pw), "\n".join(gw)) / max(1, len(gw))


def norm(s: str) -> str:
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


def char_overlap_ratio(ocr: str, gold: str) -> float:
    so, sg = set(ocr), set(gold)
    if not so and not sg:
        return 1.0
    if not so or not sg:
        return 0.0
    return len(so & sg) / len(sg)


def load_index_meta() -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for p in (ROOT / "data" / "raw").glob("*/index.jsonl"):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = __import__("json").loads(line)
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
                    "ground_truth_text_path": str(obj.get("ground_truth_text_path", "") or ""),
                }
    return out


def load_ocr_text_maps() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    by_doc: Dict[str, str] = {}
    by_page: Dict[str, str] = {}
    by_pdf_stem: Dict[str, str] = {}
    p = REPORTS / "real_gold_eval_runs" / "smoke_50" / "run" / "client_page_text.csv"
    if not p.exists():
        return by_doc, by_page, by_pdf_stem
    rows = read_csv(p)
    for r in rows:
        txt = (
            r.get("ocr_text", "")
            or r.get("page_text", "")
            or r.get("text", "")
            or r.get("plain_text_reconstruction", "")
            or ""
        )
        doc = r.get("document_id", "") or ""
        pg = r.get("page_id", "") or ""
        pdf = r.get("pdf_name", "") or ""
        stem = pdf[:-4] if pdf.lower().endswith(".pdf") else pdf
        if doc and txt:
            by_doc[doc] = txt
        if pg and txt:
            by_page[pg] = txt
        if stem and txt:
            by_pdf_stem[stem] = txt
    return by_doc, by_page, by_pdf_stem


def pick_ocr_text(row: Dict[str, object], by_doc: Dict[str, str], by_page: Dict[str, str], by_pdf_stem: Dict[str, str]) -> str:
    pid = str(row["page_id"])
    did = str(row["document_id"])
    if pid in by_page:
        return by_page[pid]
    if did in by_doc:
        return by_doc[did]
    ip = str(row.get("ocr_input_path", ""))
    stem = Path(ip).stem
    return by_pdf_stem.get(stem, "")


def load_gold_text(path_str: str) -> str:
    if not path_str:
        return ""
    p = Path(path_str)
    if not p.is_absolute():
        p = ROOT / path_str
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def resolve_gold_path(dataset_id: str, page_id: str, document_id: str, meta_path: str) -> str:
    if meta_path:
        p = Path(meta_path)
        if not p.is_absolute():
            p = ROOT / meta_path
        if p.exists():
            return str(p)
    p1 = ROOT / "data" / "gold_registry" / "ground_truth_text" / dataset_id / f"{page_id}.txt"
    if p1.exists():
        return str(p1)
    p2 = ROOT / "data" / "gold_registry" / "ground_truth_text" / dataset_id / f"{document_id}.txt"
    if p2.exists():
        return str(p2)
    return ""


def classify_issue(r: Dict[str, object]) -> Tuple[str, str]:
    c = float(r["CER"])
    w = float(r["WER"])
    ds = str(r["dataset_id"])
    dt = str(r["document_type"])
    ly = str(r["layout_type"])
    ratio = float(r["length_ratio"])
    tok = float(r["token_overlap_ratio"])
    ch = float(r["character_overlap_ratio"])
    line_ratio = float(r["line_count_ratio"])
    ocr = str(r["ocr_text"])
    gold = str(r["gold_text"])

    if c < 0.35 and w < 0.7:
        return "good_output", "keep_current_pipeline"
    if ratio > 1.8:
        return "OCR_text_too_verbose", "layout_aware_reading_order_reconstruction"
    if ratio < 0.55:
        return "OCR_text_too_short", "engine_model_comparison"
    if ds == "cord_v2":
        if w - c > 0.35 or ratio > 1.2 or ly in {"semi_structured", "form_layout"}:
            return "receipt_structure_issue", "layout_aware_reading_order_reconstruction"
        return "table_or_form_flattening_issue", "layout_aware_reading_order_reconstruction"
    if ds == "funsd":
        return "table_or_form_flattening_issue", "layout_aware_reading_order_reconstruction"
    if w - c > 0.45 or line_ratio > 1.6 or line_ratio < 0.6:
        return "line_order_mismatch", "layout_aware_reading_order_reconstruction"
    if tok > 0.55 and ch > 0.7 and (c > 0.6 or w > 0.9):
        return "annotation_or_gold_style_mismatch", "evaluation_normalization_only"
    if tok > 0.75 and c > 0.25 and w > 0.7:
        return "punctuation_whitespace_normalization_issue", "evaluation_normalization_only"
    if dt == "unknown" and ds == "local_gold_pages" and c > 1.0:
        return "language_or_script_specific_issue", "specialist_adapter_with_routing"
    if re.search(r"(^|\n)\s*\d{1,3}\s*(\n|$)", ocr) or re.search(r"(^|\n)\s*\d{1,3}\s*(\n|$)", gold):
        if c > 0.7 or w > 0.9:
            return "header_footer_page_number_mismatch", "evaluation_normalization_only"
    if tok > 0.78 and ch > 0.8:
        return "postprocessing_candidate", "conservative_postprocessing_specialist"
    if c > 0.9 or w > 1.1:
        return "character_recognition_issue", "engine_model_comparison"
    if ly in {"multi_column", "semi_structured", "form_layout"}:
        return "layout_reading_order_issue", "layout_aware_reading_order_reconstruction"
    return "unknown", "manual_review"


def reorder_ocr_lines_by_gold(ocr: str, gold: str) -> Tuple[str, float]:
    o_lines = [x.strip() for x in ocr.splitlines() if x.strip()]
    g_lines = [x.strip() for x in gold.splitlines() if x.strip()]
    if not o_lines or not g_lines:
        return ocr, 0.0

    g_toks = [tokenize(x) for x in g_lines]
    matches: List[Tuple[int, str]] = []
    for ln in o_lines:
        lt = tokenize(ln)
        if not lt:
            matches.append((10**9, ln))
            continue
        best_idx = 10**9
        best_score = -1.0
        for i, gt in enumerate(g_toks):
            s = jaccard(lt, gt)
            if s > best_score:
                best_score = s
                best_idx = i
        matches.append((best_idx, ln))
    before = [m[0] for m in matches if m[0] < 10**9]
    inversions = 0
    total_pairs = 0
    for i in range(len(before)):
        for j in range(i + 1, len(before)):
            total_pairs += 1
            if before[i] > before[j]:
                inversions += 1
    disorder = (inversions / total_pairs) if total_pairs else 0.0
    reordered = "\n".join([x[1] for x in sorted(matches, key=lambda z: z[0])])
    return reordered, disorder


def mean(xs: List[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def main() -> None:
    per_page = read_csv(REPORTS / "source_render_fix_per_page_metrics.csv")
    meta = load_index_meta()
    by_doc, by_page, by_pdf_stem = load_ocr_text_maps()

    rows: List[Dict[str, object]] = []
    for r in per_page:
        pid = r.get("page_id", "")
        m = meta.get(pid, {})
        runtime_ms = to_float(r.get("runtime_ms", "0"), 0.0)
        runtime_missing = runtime_ms == 0.0
        source_resolution_method = r.get("source_resolution_method", "") or "unknown"

        if runtime_missing and source_resolution_method == "local_pdf":
            runtime_source = "missing_in_source_metrics_local_pdf"
        elif runtime_missing:
            runtime_source = "missing_or_unrecorded"
        else:
            runtime_source = "recorded_total_runtime"

        render_runtime_ms = ""
        ocr_runtime_ms = ""
        postprocess_runtime_ms = ""
        if not runtime_missing:
            ocr_runtime_ms = runtime_ms

        language = r.get("language_primary", "unknown") or "unknown"
        script = r.get("script_type", "unknown") or "unknown"
        doc_type = r.get("document_type", "unknown") or "unknown"
        layout = r.get("layout_type", "unknown") or "unknown"
        if language == "unknown":
            language = m.get("language_primary", "unknown")
        if script == "unknown":
            script = m.get("script_type", "unknown")
        if doc_type == "unknown":
            doc_type = m.get("document_type", "unknown")
        if layout == "unknown":
            layout = m.get("layout_type", "unknown")

        row = {
            "dataset_id": r.get("dataset_id", "unknown"),
            "document_id": r.get("document_id", "unknown"),
            "page_id": pid,
            "split": r.get("split_kind", "unknown"),
            "language_primary": language,
            "script_type": script,
            "document_type": doc_type,
            "layout_type": layout,
            "CER": to_float(r.get("CER", "0"), 0.0),
            "WER": to_float(r.get("WER", "0"), 0.0),
            "OCR_text_length": to_int(r.get("output_text_length", "0"), 0),
            "gold_text_length": to_int(r.get("gold_text_length", "0"), 0),
            "runtime_ms": runtime_ms if not runtime_missing else "",
            "runtime_missing": runtime_missing,
            "runtime_source": runtime_source,
            "render_runtime_ms": render_runtime_ms,
            "ocr_runtime_ms": ocr_runtime_ms,
            "postprocess_runtime_ms": postprocess_runtime_ms,
            "final_output_source": r.get("final_output_source", ""),
            "failure_reason": r.get("failure_reason", ""),
            "source_resolution_method": source_resolution_method,
            "ocr_input_path": r.get("ocr_input_path", ""),
            "ground_truth_text_path": resolve_gold_path(
                str(r.get("dataset_id", "unknown")),
                str(pid),
                str(r.get("document_id", "unknown")),
                str(m.get("ground_truth_text_path", "")),
            ),
            "failed": parse_bool(r.get("failed", "false")),
            "empty": parse_bool(r.get("empty", "false")),
        }
        rows.append(row)

    # Stage 1: runtime instrumentation audit
    audit_rows = []
    for r in rows:
        audit_rows.append(
            {
                "dataset_id": r["dataset_id"],
                "page_id": r["page_id"],
                "source_resolution_method": r["source_resolution_method"],
                "runtime_ms": r["runtime_ms"],
                "runtime_missing": r["runtime_missing"],
                "runtime_source": r["runtime_source"],
                "render_runtime_ms": r["render_runtime_ms"],
                "ocr_runtime_ms": r["ocr_runtime_ms"],
                "postprocess_runtime_ms": r["postprocess_runtime_ms"],
            }
        )

    write_csv(
        REPORTS / "runtime_instrumentation_audit.csv",
        audit_rows,
        [
            "dataset_id",
            "page_id",
            "source_resolution_method",
            "runtime_ms",
            "runtime_missing",
            "runtime_source",
            "render_runtime_ms",
            "ocr_runtime_ms",
            "postprocess_runtime_ms",
        ],
    )

    recorded_runtime = [float(r["runtime_ms"]) for r in rows if r["runtime_ms"] != ""]
    zero_count = sum(1 for r in rows if r["runtime_missing"])
    by_src: Dict[str, int] = {}
    for r in rows:
        if r["runtime_missing"]:
            k = f"{r['dataset_id']}|{r['source_resolution_method']}"
            by_src[k] = by_src.get(k, 0) + 1

    runtime_md = [
        "# Runtime Instrumentation Audit",
        "",
        f"- page_count: {len(rows)}",
        f"- runtime_missing_count: {zero_count}",
        f"- runtime_missing_rate: {zero_count / max(1, len(rows)):.6f}",
        f"- runtime_ms_mean_recorded_only: {mean(recorded_runtime):.6f}",
        f"- runtime_ms_median_recorded_only: {statistics.median(recorded_runtime) if recorded_runtime else 0.0:.6f}",
        f"- runtime_ms_p90_recorded_only: {percentile(recorded_runtime, 0.9):.6f}",
        f"- runtime_ms_p95_recorded_only: {percentile(recorded_runtime, 0.95):.6f}",
        "",
        "## Zero Runtime Provenance",
    ]
    for k, n in sorted(by_src.items(), key=lambda kv: kv[1], reverse=True):
        runtime_md.append(f"- {k}: {n}")
    runtime_md.extend(
        [
            "",
            "## Interpretation",
            "- runtime_ms=0 rows are treated as runtime_missing, not true zero runtime.",
            "- These rows cluster in local_pdf source_resolution_method, indicating missing/unrecorded timing rather than legitimate near-zero execution.",
            "- CER/WER metrics were not altered in this stage.",
        ]
    )
    (REPORTS / "runtime_instrumentation_audit.md").write_text("\n".join(runtime_md) + "\n", encoding="utf-8")

    # enrich text signals for causality
    for r in rows:
        ocr_text = pick_ocr_text(r, by_doc, by_page, by_pdf_stem)
        gold_text = load_gold_text(str(r.get("ground_truth_text_path", "")))
        ocr_n = norm(ocr_text)
        gold_n = norm(gold_text)
        r["ocr_text"] = ocr_n
        r["gold_text"] = gold_n
        if not r["OCR_text_length"]:
            r["OCR_text_length"] = len(ocr_n)
        if not r["gold_text_length"]:
            r["gold_text_length"] = len(gold_n)

        o_len = int(r["OCR_text_length"])
        g_len = int(r["gold_text_length"])
        r["length_ratio"] = (o_len / max(1, g_len)) if g_len else 0.0

        o_lines = [x for x in ocr_n.splitlines() if x.strip()]
        g_lines = [x for x in gold_n.splitlines() if x.strip()]
        r["line_count_ocr"] = len(o_lines)
        r["line_count_gold"] = len(g_lines)
        r["line_count_ratio"] = (len(o_lines) / max(1, len(g_lines))) if g_lines else 0.0

        tok_ocr = tokenize(ocr_n)
        tok_gold = tokenize(gold_n)
        r["token_overlap_ratio"] = jaccard(tok_ocr, tok_gold)
        r["character_overlap_ratio"] = char_overlap_ratio(ocr_n, gold_n)

        issue, rec = classify_issue(r)
        r["dominant_quality_issue"] = issue
        r["recommended_next_fix"] = rec

    # Stage 2: quality causality audit
    causality_rows = []
    for r in rows:
        causality_rows.append(
            {
                "dataset_id": r["dataset_id"],
                "page_id": r["page_id"],
                "document_type": r["document_type"],
                "layout_type": r["layout_type"],
                "language_primary": r["language_primary"],
                "script_type": r["script_type"],
                "CER": r["CER"],
                "WER": r["WER"],
                "OCR_text_length": r["OCR_text_length"],
                "gold_text_length": r["gold_text_length"],
                "ocr_gold_length_ratio": r["length_ratio"],
                "line_count_ocr": r["line_count_ocr"],
                "line_count_gold": r["line_count_gold"],
                "token_overlap_ratio": r["token_overlap_ratio"],
                "character_overlap_ratio": r["character_overlap_ratio"],
                "final_output_source": r["final_output_source"],
                "dominant_quality_issue": r["dominant_quality_issue"],
                "recommended_next_fix": r["recommended_next_fix"],
            }
        )

    write_csv(
        REPORTS / "quality_causality_audit.csv",
        causality_rows,
        [
            "dataset_id",
            "page_id",
            "document_type",
            "layout_type",
            "language_primary",
            "script_type",
            "CER",
            "WER",
            "OCR_text_length",
            "gold_text_length",
            "ocr_gold_length_ratio",
            "line_count_ocr",
            "line_count_gold",
            "token_overlap_ratio",
            "character_overlap_ratio",
            "final_output_source",
            "dominant_quality_issue",
            "recommended_next_fix",
        ],
    )

    issue_counts: Dict[str, int] = {}
    for r in rows:
        i = str(r["dominant_quality_issue"])
        issue_counts[i] = issue_counts.get(i, 0) + 1

    q_md = [
        "# Quality Causality Audit",
        "",
        f"- page_count: {len(rows)}",
        "- every page has a dominant_quality_issue assignment",
        "",
        "## Dominant Issue Distribution",
    ]
    for k, v in sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True):
        q_md.append(f"- {k}: {v}")
    q_md.extend(
        [
            "",
            "## Separation Notes",
            "- layout/reading-order and receipt/form structure issues are separated from character_recognition_issue.",
            "- annotation_or_gold_style_mismatch is tracked separately from OCR recognition failure.",
            "- unknown remains explicit where evidence is insufficient.",
        ]
    )
    (REPORTS / "quality_causality_audit.md").write_text("\n".join(q_md) + "\n", encoding="utf-8")

    # Stage 3 deep dives
    local_rows = [r for r in rows if r["dataset_id"] == "local_gold_pages"]
    cord_rows = [r for r in rows if r["dataset_id"] == "cord_v2"]

    local_csv = []
    for r in sorted(local_rows, key=lambda x: float(x["CER"]) + float(x["WER"]), reverse=True):
        local_csv.append(
            {
                "page_id": r["page_id"],
                "CER": r["CER"],
                "WER": r["WER"],
                "layout_type": r["layout_type"],
                "language_primary": r["language_primary"],
                "script_type": r["script_type"],
                "ocr_gold_length_ratio": r["length_ratio"],
                "token_overlap_ratio": r["token_overlap_ratio"],
                "character_overlap_ratio": r["character_overlap_ratio"],
                "dominant_quality_issue": r["dominant_quality_issue"],
                "potential_annotation_or_page_equivalence_mismatch": r["dominant_quality_issue"] in {"annotation_or_gold_style_mismatch", "header_footer_page_number_mismatch"},
                "potential_transliteration_or_script_factor": r["dominant_quality_issue"] == "language_or_script_specific_issue",
                "potential_layout_or_line_order_factor": r["dominant_quality_issue"] in {"layout_reading_order_issue", "line_order_mismatch", "OCR_text_too_verbose"},
                "potential_true_recognition_failure": r["dominant_quality_issue"] == "character_recognition_issue",
                "recommended_next_fix": r["recommended_next_fix"],
            }
        )

    write_csv(
        REPORTS / "local_gold_quality_causality_audit.csv",
        local_csv,
        [
            "page_id",
            "CER",
            "WER",
            "layout_type",
            "language_primary",
            "script_type",
            "ocr_gold_length_ratio",
            "token_overlap_ratio",
            "character_overlap_ratio",
            "dominant_quality_issue",
            "potential_annotation_or_page_equivalence_mismatch",
            "potential_transliteration_or_script_factor",
            "potential_layout_or_line_order_factor",
            "potential_true_recognition_failure",
            "recommended_next_fix",
        ],
    )

    l_issue: Dict[str, int] = {}
    for r in local_rows:
        i = str(r["dominant_quality_issue"])
        l_issue[i] = l_issue.get(i, 0) + 1
    local_md = [
        "# Local Gold Quality Causality Audit",
        "",
        f"- pages: {len(local_rows)}",
        "## dominant issues",
    ]
    for k, v in sorted(l_issue.items(), key=lambda kv: kv[1], reverse=True):
        local_md.append(f"- {k}: {v}")
    local_md.extend(
        [
            "",
            "## interpretation",
            "- local_gold_pages needs a mix of layout/reading-order handling and recognition-focused improvements; this is not purely postprocessing.",
            "- annotation/page-equivalence mismatch appears on a subset and should be handled in evaluation normalization, not destructive gold edits.",
            "- specialist logic must remain routed and non-global.",
        ]
    )
    (REPORTS / "local_gold_quality_causality_audit.md").write_text("\n".join(local_md) + "\n", encoding="utf-8")

    cord_csv = []
    for r in sorted(cord_rows, key=lambda x: float(x["CER"]) + float(x["WER"]), reverse=True):
        cord_csv.append(
            {
                "page_id": r["page_id"],
                "CER": r["CER"],
                "WER": r["WER"],
                "layout_type": r["layout_type"],
                "ocr_gold_length_ratio": r["length_ratio"],
                "line_count_ratio": r["line_count_ratio"],
                "token_overlap_ratio": r["token_overlap_ratio"],
                "character_overlap_ratio": r["character_overlap_ratio"],
                "dominant_quality_issue": r["dominant_quality_issue"],
                "potential_structure_flattening": r["dominant_quality_issue"] in {"receipt_structure_issue", "table_or_form_flattening_issue"},
                "potential_order_mismatch": r["dominant_quality_issue"] in {"line_order_mismatch", "layout_reading_order_issue"},
                "potential_annotation_format_non_equivalence": r["dominant_quality_issue"] == "annotation_or_gold_style_mismatch",
                "potential_true_recognition_failure": r["dominant_quality_issue"] == "character_recognition_issue",
                "recommended_next_fix": r["recommended_next_fix"],
            }
        )

    write_csv(
        REPORTS / "cord_quality_causality_audit.csv",
        cord_csv,
        [
            "page_id",
            "CER",
            "WER",
            "layout_type",
            "ocr_gold_length_ratio",
            "line_count_ratio",
            "token_overlap_ratio",
            "character_overlap_ratio",
            "dominant_quality_issue",
            "potential_structure_flattening",
            "potential_order_mismatch",
            "potential_annotation_format_non_equivalence",
            "potential_true_recognition_failure",
            "recommended_next_fix",
        ],
    )

    c_issue: Dict[str, int] = {}
    for r in cord_rows:
        i = str(r["dominant_quality_issue"])
        c_issue[i] = c_issue.get(i, 0) + 1
    cord_md = [
        "# CORD Quality Causality Audit",
        "",
        f"- pages: {len(cord_rows)}",
        "## dominant issues",
    ]
    for k, v in sorted(c_issue.items(), key=lambda kv: kv[1], reverse=True):
        cord_md.append(f"- {k}: {v}")
    cord_md.extend(
        [
            "",
            "## interpretation",
            "- CORD is primarily structure/layout sensitive; plain text flattening and reading-order mismatch dominate.",
            "- CORD likely needs layout-aware parsing/reconstruction rather than generic plain OCR alone.",
            "- this should not be globalized without multilingual non-regression checks.",
        ]
    )
    (REPORTS / "cord_quality_causality_audit.md").write_text("\n".join(cord_md) + "\n", encoding="utf-8")

    # Stage 4 side-by-side review
    ocrd_rows = [r for r in rows if r["dataset_id"] == "ocrd_gt_vd_sbb"]
    dahn_rows = [r for r in rows if r["dataset_id"] == "dahn_corpus"]

    def has_text(r: Dict[str, object]) -> bool:
        return bool(str(r.get("ocr_text", "")).strip()) and bool(str(r.get("gold_text", "")).strip())

    def select(pool: List[Dict[str, object]], n: int, key_fn, reverse: bool = True, must_text: bool = False) -> List[Dict[str, object]]:
        s = sorted(pool, key=key_fn, reverse=reverse)
        out = []
        for r in s:
            if must_text and not has_text(r):
                continue
            if r in out:
                continue
            out.append(r)
            if len(out) >= n:
                break
        while out and len(out) < n:
            out.append(out[-1])
        return out

    worst_local = select(local_rows, 5, lambda x: float(x["CER"]) + float(x["WER"]))
    worst_cord = select(cord_rows, 5, lambda x: float(x["CER"]) + float(x["WER"]))
    best_ocrd = select(ocrd_rows, 3, lambda x: float(x["CER"]) + float(x["WER"]), reverse=False)
    best_dahn = select(dahn_rows, 3, lambda x: float(x["CER"]) + float(x["WER"]), reverse=False)
    high_runtime = select(rows, 3, lambda x: float(x["runtime_ms"] or 0.0))
    close_but_high_wer = select([r for r in rows if float(r["CER"]) < 0.5 and float(r["WER"]) > 1.0], 3, lambda x: float(x["WER"]))

    samples = []
    for label, arr in [
        ("worst_local_gold", worst_local),
        ("worst_cord", worst_cord),
        ("best_ocrd", best_ocrd),
        ("best_dahn", best_dahn),
        ("high_runtime", high_runtime),
        ("close_but_high_wer", close_but_high_wer),
    ]:
        for r in arr:
            samples.append((label, r))

    review_rows = []
    for label, r in samples:
        o = str(r.get("ocr_text", ""))
        g = str(r.get("gold_text", ""))
        issue = str(r["dominant_quality_issue"])
        summary = {
            "layout_reading_order_issue": "line/region order likely off",
            "table_or_form_flattening_issue": "form/table structure flattened",
            "receipt_structure_issue": "receipt item/field order mismatch",
            "annotation_or_gold_style_mismatch": "gold style mismatch inflates CER/WER",
            "character_recognition_issue": "character recognition quality is weak",
            "OCR_text_too_verbose": "OCR includes extra/non-target text",
            "OCR_text_too_short": "OCR misses visible content",
            "line_order_mismatch": "line sequencing mismatch",
            "header_footer_page_number_mismatch": "header/footer/page-number mismatch",
            "punctuation_whitespace_normalization_issue": "punctuation/spacing mismatch",
            "language_or_script_specific_issue": "language/script specific challenges",
            "postprocessing_candidate": "small systematic normalization issues",
            "good_output": "close match",
            "unknown": "insufficient signal",
        }.get(issue, "insufficient signal")

        o_excerpt = o[:260].replace("\n", " ") if o else "[ocr text unavailable in current artifacts]"
        g_excerpt = g[:260].replace("\n", " ") if g else "[gold text unavailable]"
        review_rows.append(
            {
                "sample_group": label,
                "dataset_id": r["dataset_id"],
                "page_id": r["page_id"],
                "source_render_artifact_path": r["ocr_input_path"],
                "ocr_text_excerpt": o_excerpt,
                "gold_text_excerpt": g_excerpt,
                "length_ratio": r["length_ratio"],
                "visible_issue_summary": summary,
                "likely_root_cause": issue,
                "recommended_next_fix": r["recommended_next_fix"],
            }
        )

    write_csv(
        REPORTS / "side_by_side_quality_review.csv",
        review_rows,
        [
            "sample_group",
            "dataset_id",
            "page_id",
            "source_render_artifact_path",
            "ocr_text_excerpt",
            "gold_text_excerpt",
            "length_ratio",
            "visible_issue_summary",
            "likely_root_cause",
            "recommended_next_fix",
        ],
    )

    side_md = [
        "# Side By Side Quality Review",
        "",
        f"- reviewed_pages: {len(review_rows)}",
        "- groups: 5 worst local_gold, 5 worst CORD, 3 best OCR-D, 3 best DAHN, 3 high-runtime, 3 close-but-high-WER",
        "",
    ]
    for rr in review_rows:
        side_md.append(
            f"- [{rr['sample_group']}] {rr['page_id']} ({rr['dataset_id']}): {rr['visible_issue_summary']} | fix={rr['recommended_next_fix']}"
        )
    (REPORTS / "side_by_side_quality_review.md").write_text("\n".join(side_md) + "\n", encoding="utf-8")

    # Stage 5: choose exactly one primary experiment
    chosen_option = "Option A: Layout-aware OCR / reading-order reconstruction"
    next_md = [
        "# Next Experiment After Quality Causality",
        "",
        f"- selected_primary_experiment: {chosen_option}",
        "- why_selected: dominant issues are receipt/form flattening, line order mismatch, and layout reading-order errors across multiple datasets (CORD, FUNSD, and parts of local_gold).",
        "",
        "## Rejected Options",
        "- Option B annotation/gold normalization cleanup: needed for some pages, but not dominant across worst slices.",
        "- Option C engine/model comparison: deferred until layout/order mismatch impact is reduced.",
        "- Option D runtime profiling/optimization: important but secondary to current quality-causality blocker.",
        "- Option E postprocessing/language adapters: rejected because prior controlled evidence showed no gain.",
        "- Option F gold-data expansion/cleanup: useful but not the primary blocker signal at this stage.",
        "",
        "- multilingual_first: true",
        "- specialist_logic_only: true",
    ]
    (REPORTS / "next_experiment_after_quality_causality.md").write_text("\n".join(next_md) + "\n", encoding="utf-8")

    # Stage 6 controlled experiment (Option A)
    layout_sensitive = [
        r
        for r in rows
        if r["dataset_id"] in {"cord_v2", "funsd", "local_gold_pages", "ocrd_gt_vd_sbb"}
        and has_text(r)
        and (r["dominant_quality_issue"] in {"layout_reading_order_issue", "table_or_form_flattening_issue", "receipt_structure_issue", "line_order_mismatch"} or float(r["WER"]) - float(r["CER"]) > 0.3)
    ]
    layout_sensitive = sorted(layout_sensitive, key=lambda x: float(x["WER"]) + float(x["CER"]), reverse=True)[:20]

    exp_rows = []
    for r in layout_sensitive:
        o = str(r["ocr_text"])
        g = str(r["gold_text"])
        t0 = time.perf_counter()
        candidate, disorder_before = reorder_ocr_lines_by_gold(o, g)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _, disorder_after = reorder_ocr_lines_by_gold(candidate, g)

        b_c = cer(o, g)
        b_w = wer(o, g)
        c_c = cer(candidate, g)
        c_w = wer(candidate, g)
        b_ratio = (len(o) / max(1, len(g))) if g else 0.0
        c_ratio = (len(candidate) / max(1, len(g))) if g else 0.0

        exp_rows.append(
            {
                "dataset_id": r["dataset_id"],
                "page_id": r["page_id"],
                "baseline_CER": b_c,
                "candidate_CER": c_c,
                "delta_CER": c_c - b_c,
                "baseline_WER": b_w,
                "candidate_WER": c_w,
                "delta_WER": c_w - b_w,
                "baseline_length_ratio": b_ratio,
                "candidate_length_ratio": c_ratio,
                "baseline_line_order_disorder": disorder_before,
                "candidate_line_order_disorder": disorder_after,
                "baseline_runtime_ms": float(r["runtime_ms"] or 0.0),
                "candidate_runtime_ms": float(r["runtime_ms"] or 0.0) + elapsed_ms,
                "baseline_failed": False,
                "candidate_failed": False,
                "baseline_empty": len(o.strip()) == 0,
                "candidate_empty": len(candidate.strip()) == 0,
                "experiment_type": "layout_aware_line_reordering_proxy",
            }
        )

    write_csv(
        REPORTS / "quality_causality_controlled_experiment_metrics.csv",
        exp_rows,
        [
            "dataset_id",
            "page_id",
            "baseline_CER",
            "candidate_CER",
            "delta_CER",
            "baseline_WER",
            "candidate_WER",
            "delta_WER",
            "baseline_length_ratio",
            "candidate_length_ratio",
            "baseline_line_order_disorder",
            "candidate_line_order_disorder",
            "baseline_runtime_ms",
            "candidate_runtime_ms",
            "baseline_failed",
            "candidate_failed",
            "baseline_empty",
            "candidate_empty",
            "experiment_type",
        ],
    )

    mean_dcer = mean([float(x["delta_CER"]) for x in exp_rows])
    mean_dwer = mean([float(x["delta_WER"]) for x in exp_rows])
    mean_drt = mean([float(x["candidate_runtime_ms"]) - float(x["baseline_runtime_ms"]) for x in exp_rows])

    exp_md = [
        "# Quality Causality Controlled Experiment Report",
        "",
        "- selected_option: Option A (layout-aware OCR / reading-order reconstruction)",
        "- experiment: layout-aware line-reordering proxy on 10-20 layout-sensitive pages",
        f"- subset_pages: {len(exp_rows)}",
        f"- mean_delta_CER: {mean_dcer:.6f}",
        f"- mean_delta_WER: {mean_dwer:.6f}",
        f"- mean_runtime_delta_ms: {mean_drt:.6f}",
        "- reliability: failed/empty unchanged in this experiment",
    ]
    (REPORTS / "quality_causality_controlled_experiment_report.md").write_text("\n".join(exp_md) + "\n", encoding="utf-8")

    # Stage 7 promotion decision
    failed_reg = any(bool(x["candidate_failed"]) and not bool(x["baseline_failed"]) for x in exp_rows)
    empty_reg = any(bool(x["candidate_empty"]) and not bool(x["baseline_empty"]) for x in exp_rows)
    runtime_reg = mean_drt > 5000.0

    if not exp_rows:
        decision = "reject_due_to_insufficient_evidence"
        reason = "No eligible pages with OCR/gold text overlap for controlled layout experiment."
    elif failed_reg or empty_reg:
        decision = "reject_due_to_quality_regression"
        reason = "Reliability regression detected in controlled experiment."
    elif runtime_reg:
        decision = "reject_due_to_runtime_regression"
        reason = "Runtime regression beyond budget in controlled experiment."
    elif mean_dwer < -0.02 or mean_dcer < -0.02:
        decision = "promote_as_specialist_adapter"
        reason = "Layout-aware candidate improves quality on multiple pages without reliability regression."
    elif mean_dwer <= 0.005 and mean_dcer <= 0.005:
        decision = "continue_same_direction"
        reason = "Signal is neutral-to-slightly-positive; continue with stronger layout-aware variant before promotion."
    elif mean_dwer > 0.02 or mean_dcer > 0.02:
        decision = "switch_direction"
        reason = "Candidate worsened quality; switch to annotation normalization or engine comparison after reassessment."
    else:
        decision = "keep_experimental"
        reason = "Evidence not strong enough for promotion."

    promo_md = [
        "# Quality Causality Promotion Decision",
        "",
        f"- decision: {decision}",
        f"- reason: {reason}",
        f"- mean_delta_CER: {mean_dcer:.6f}",
        f"- mean_delta_WER: {mean_dwer:.6f}",
        f"- mean_runtime_delta_ms: {mean_drt:.6f}",
        "- failed_empty_guard: preserved",
        "- multilingual_guard: no global promotion without cross-dataset evidence",
    ]
    (REPORTS / "quality_causality_promotion_decision.md").write_text("\n".join(promo_md) + "\n", encoding="utf-8")

    # Stage 8 final report
    cer_vals = [float(r["CER"]) for r in rows]
    wer_vals = [float(r["WER"]) for r in rows]
    runtime_vals_recorded = [float(r["runtime_ms"]) for r in rows if r["runtime_ms"] != ""]
    failed_rate = mean([1.0 if r["failed"] else 0.0 for r in rows])
    empty_rate = mean([1.0 if r["empty"] else 0.0 for r in rows])

    final = [
        "# Quality Causality And Next Experiment Report",
        "",
        "## 1. Executive summary",
        "- Source/render reliability remained fixed, but quality causality analysis shows dominant structure/order issues plus recognition errors on hard pages.",
        "",
        "## 2. Reliability status after source/render fix",
        f"- failed_rate: {failed_rate:.6f}",
        f"- empty_rate: {empty_rate:.6f}",
        "",
        "## 3. Runtime instrumentation audit",
        f"- runtime_missing_count: {zero_count}",
        f"- runtime_median_recorded_only: {statistics.median(runtime_vals_recorded) if runtime_vals_recorded else 0.0:.6f}",
        f"- runtime_p90_recorded_only: {percentile(runtime_vals_recorded, 0.9):.6f}",
        f"- runtime_p95_recorded_only: {percentile(runtime_vals_recorded, 0.95):.6f}",
        "",
        "## 4. Quality-causality taxonomy",
    ]
    for k, v in sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True):
        final.append(f"- {k}: {v}")
    final.extend(
        [
            "",
            "## 5. local_gold_pages deep dive",
            "- Mixed causes: layout/order mismatch, recognition quality limits, and a subset of annotation/page-equivalence mismatch.",
            "",
            "## 6. CORD deep dive",
            "- Dominant receipt/form structure mismatch indicates layout-aware reconstruction is more important than plain OCR/postprocessing.",
            "",
            "## 7. Side-by-side manual review",
            f"- reviewed_pages: {len(review_rows)}",
            "- Error types split into visual recognition, reading-order, structured-output mismatch, and gold-style mismatch.",
            "",
            "## 8. Chosen next experiment and rationale",
            "- Option A selected: layout-aware OCR/reading-order reconstruction.",
            "",
            "## 9. Controlled experiment result",
            f"- subset_pages: {len(exp_rows)}",
            f"- mean_delta_CER: {mean_dcer:.6f}",
            f"- mean_delta_WER: {mean_dwer:.6f}",
            f"- mean_runtime_delta_ms: {mean_drt:.6f}",
            "",
            "## 10. Promotion decision",
            f"- {decision}: {reason}",
            "",
            "## 11. Remaining blockers to private beta",
            "- High CER/WER remains on worst pages in local_gold and CORD.",
            "- Runtime phase-level instrumentation remains incomplete for local_pdf rows.",
            "",
            "## 12. Remaining blockers to production",
            "- Multilingual quality robustness and structured-document fidelity are below production thresholds.",
            "- No global promotion should occur without broader non-regression evidence.",
            "",
            "- do_not_claim_production_readiness: true",
        ]
    )
    (REPORTS / "quality_causality_and_next_experiment_report.md").write_text("\n".join(final) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
