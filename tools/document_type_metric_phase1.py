#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_METRICS_CSV = ROOT / "reports" / "source_render_fix_per_page_metrics.csv"
OCR_TEXT_CSV = ROOT / "reports" / "real_gold_eval_runs" / "smoke_50" / "run" / "client_page_text.csv"
GOLD_TEXT_ROOT = ROOT / "data" / "gold_registry" / "ground_truth_text"

DESIGN_MD = ROOT / "reports" / "document_type_metric_design.md"
RESULTS_CSV = ROOT / "reports" / "document_type_metric_results.csv"
RESULTS_MD = ROOT / "reports" / "document_type_metric_results.md"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _safe_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _token_jaccard(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"\w+", a.lower(), flags=re.UNICODE))
    b_tokens = set(re.findall(r"\w+", b.lower(), flags=re.UNICODE))
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, av in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, bv in enumerate(b, start=1):
            cost = 0 if av == bv else 1
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
    return dp[-1]


def _line_order_similarity(ref: str, hyp: str) -> float:
    r = [x.strip() for x in ref.splitlines() if x.strip()]
    h = [x.strip() for x in hyp.splitlines() if x.strip()]
    if not r:
        return 1.0 if not h else 0.0
    if not h:
        return 0.0
    return max(0.0, 1.0 - (_edit_distance(r, h) / len(r)))


def _paragraph_order_similarity(ref: str, hyp: str) -> float:
    r = [x.strip() for x in re.split(r"\n\s*\n", ref) if x.strip()]
    h = [x.strip() for x in re.split(r"\n\s*\n", hyp) if x.strip()]
    if not r:
        return 1.0 if not h else 0.0
    if not h:
        return 0.0
    return max(0.0, 1.0 - (_edit_distance(r, h) / len(r)))


def _header_footer_mismatch(ref: str, hyp: str) -> bool:
    r = [x.strip() for x in ref.splitlines() if x.strip()]
    h = [x.strip() for x in hyp.splitlines() if x.strip()]
    if len(r) < 2 or len(h) < 2:
        return False
    return _token_jaccard(r[0] + " " + r[-1], h[0] + " " + h[-1]) < 0.2


def _extract_kv_pairs(text: str) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for line in text.splitlines():
        s = line.strip()
        if not s or ":" not in s:
            continue
        k, v = s.split(":", 1)
        k = k.strip().lower()
        v = v.strip().lower()
        if k and v:
            out.add((k, v))
    return out


def _f1(ref: set[tuple[str, str]], hyp: set[tuple[str, str]]) -> float | None:
    if not ref and not hyp:
        return None
    if not ref or not hyp:
        return 0.0
    tp = len(ref & hyp)
    p = tp / len(hyp)
    r = tp / len(ref)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _extract_table_rows(text: str) -> set[str]:
    out: set[str] = set()
    for line in text.splitlines():
        s = re.sub(r"\s+", " ", line.strip().lower())
        if not s:
            continue
        if re.search(r"\d", s) and re.search(r"[\.:,]", s):
            out.add(s)
    return out


def _metric_family(dataset_id: str, document_type: str) -> str:
    ds = dataset_id.lower()
    dt = document_type.lower()
    if "cord" in ds or "receipt" in dt:
        return "structured_receipt"
    if "funsd" in ds or "form" in dt:
        return "structured_form"
    if "local_gold" in ds:
        return "local_scope_validated"
    return "plain_text_layout"


def _gold_text_for(page_id: str, dataset_id: str, document_id: str) -> str:
    base = GOLD_TEXT_ROOT / dataset_id / document_id
    if not base.exists():
        return ""
    direct = base / f"{page_id}.txt"
    if direct.exists():
        return direct.read_text(encoding="utf-8").strip()
    matches = list(base.glob(f"*{page_id}*.txt"))
    if matches:
        return matches[0].read_text(encoding="utf-8").strip()
    return ""


def main() -> int:
    source_rows = _read_csv(SOURCE_METRICS_CSV)
    ocr_rows = _read_csv(OCR_TEXT_CSV)
    ocr_by_page_id = {str(r.get("page_id", "")).strip(): r for r in ocr_rows if str(r.get("page_id", "")).strip()}

    out_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in source_rows:
        page_id = str(row.get("page_id", "")).strip()
        dataset_id = str(row.get("dataset_id", "")).strip()
        document_id = str(row.get("document_id", "")).strip()
        document_type = str(row.get("document_type", "") or "unknown").strip()

        ocr_row = ocr_by_page_id.get(page_id, {})
        ocr_text = str(ocr_row.get("page_text", "") or "").strip()
        gold_text = _gold_text_for(page_id, dataset_id, document_id)
        if not gold_text:
            continue

        family = _metric_family(dataset_id, document_type)
        line_sim = _line_order_similarity(gold_text, ocr_text)
        para_sim = _paragraph_order_similarity(gold_text, ocr_text)
        hf_mismatch = _header_footer_mismatch(gold_text, ocr_text)

        kv_f1 = _f1(_extract_kv_pairs(gold_text), _extract_kv_pairs(ocr_text))
        table_sim = _f1(
            set((x, "") for x in _extract_table_rows(gold_text)),
            set((x, "") for x in _extract_table_rows(ocr_text)),
        )
        entity_assoc = kv_f1
        box_linking = _token_jaccard(gold_text, ocr_text)

        cer = _safe_float(row.get("CER"))
        wer = _safe_float(row.get("WER"))

        ratio = (len(ocr_text) / len(gold_text)) if gold_text else None
        page_equiv = bool(page_id)
        target_scope = ratio is not None and 0.5 <= ratio <= 2.0

        result = {
            "dataset_id": dataset_id,
            "document_id": document_id,
            "page_id": page_id,
            "document_type": document_type,
            "metric_family": family,
            "cer": cer,
            "wer": wer,
            "line_order_similarity": line_sim,
            "paragraph_order_similarity": para_sim,
            "header_footer_mismatch": hf_mismatch,
            "key_value_f1": kv_f1,
            "table_row_similarity": table_sim,
            "entity_association_accuracy": entity_assoc,
            "box_text_linking_quality": box_linking,
            "page_equivalence_valid": page_equiv,
            "target_scope_valid": target_scope,
        }
        out_rows.append(result)
        grouped[family].append(result)

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "dataset_id",
            "document_id",
            "page_id",
            "document_type",
            "metric_family",
            "cer",
            "wer",
            "line_order_similarity",
            "paragraph_order_similarity",
            "header_footer_mismatch",
            "key_value_f1",
            "table_row_similarity",
            "entity_association_accuracy",
            "box_text_linking_quality",
            "page_equivalence_valid",
            "target_scope_valid",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    DESIGN_MD.write_text(
        "\n".join(
            [
                "# Document-Type-Aware Evaluation Design (Phase 1)",
                "",
                "## Goals",
                "- Keep CER/WER in all outputs for continuity.",
                "- Route primary metrics by document family instead of treating all pages as plain-text OCR.",
                "- Keep multilingual-safe heuristics (no English-only lexical rules).",
                "",
                "## Metric Families",
                "- plain_text_layout: line_order_similarity, paragraph_order_similarity, header_footer_mismatch, plus CER/WER.",
                "- structured_receipt: key_value_f1, table_row_similarity, plus CER/WER.",
                "- structured_form: key_value_f1, entity_association_accuracy, box_text_linking_quality, plus CER/WER.",
                "- local_scope_validated: page_equivalence_valid and target_scope_valid gate interpretation, plus CER/WER.",
                "",
                "## Routing",
                "- Dataset/document metadata routes each page to one metric family.",
                "- If structured signals are missing for a page, structured metrics are left blank rather than faked.",
            ]
        ),
        encoding="utf-8",
    )

    lines = [
        "# Document-Type-Aware Evaluation Results (Phase 1)",
        "",
        f"Pages evaluated: {len(out_rows)}",
        "",
        "| metric_family | pages | CER mean | WER mean | key_value_f1 mean | line_order_similarity mean | scope_valid_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for family in sorted(grouped.keys()):
        rows = grouped[family]
        cer_vals = [x["cer"] for x in rows if x["cer"] is not None]
        wer_vals = [x["wer"] for x in rows if x["wer"] is not None]
        kv_vals = [x["key_value_f1"] for x in rows if x["key_value_f1"] is not None]
        line_vals = [x["line_order_similarity"] for x in rows if x["line_order_similarity"] is not None]
        scope_rate = sum(1 for x in rows if x["target_scope_valid"]) / len(rows) if rows else 0.0
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                family,
                len(rows),
                f"{_mean(cer_vals):.4f}" if cer_vals else "",
                f"{_mean(wer_vals):.4f}" if wer_vals else "",
                f"{_mean(kv_vals):.4f}" if kv_vals else "",
                f"{_mean(line_vals):.4f}" if line_vals else "",
                f"{scope_rate:.4f}",
            )
        )

    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_CSV}")
    print(f"Wrote {DESIGN_MD}")
    print(f"Wrote {RESULTS_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
