#!/usr/bin/env python3
from __future__ import annotations

import copy
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "reports" / "real_gold_eval_runs" / "smoke_50"
INPUT_DIR = BENCHMARK_ROOT / "input_pdfs"
GOLD_CSV = BENCHMARK_ROOT / "gold.csv"

REPORTS = ROOT / "reports"
RAW_DIR = REPORTS / "industry_baseline_raw"
ROUTING_RECOMMENDATION_CSV = REPORTS / "document_type_routing_recommendation.csv"

OUT_METRICS_CSV = REPORTS / "routed_pipeline_experiment_metrics.csv"
OUT_REPORT_MD = REPORTS / "routed_pipeline_experiment_report.md"
OUT_PER_PAGE_JSONL = REPORTS / "routed_pipeline_per_page_outputs.jsonl"
OUT_PROMOTION_MD = REPORTS / "routed_pipeline_promotion_decision.md"

CURRENT_BACKEND = "current_pipeline"
ROUTED_VARIANT = "routed_document_type_pipeline"
CURRENT_VARIANT = "current_pipeline"


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


def _cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(list(reference), list(hypothesis)) / len(reference)


def _wer(reference: str, hypothesis: str) -> float:
    r = reference.split()
    h = hypothesis.split()
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


def _line_order_similarity(reference: str, hypothesis: str) -> float:
    r = [x.strip() for x in reference.splitlines() if x.strip()]
    h = [x.strip() for x in hypothesis.splitlines() if x.strip()]
    if not r:
        return 1.0 if not h else 0.0
    if not h:
        return 0.0
    return max(0.0, 1.0 - (_edit_distance(r, h) / len(r)))


def _paragraph_order_similarity(reference: str, hypothesis: str) -> float:
    r = [x.strip() for x in re.split(r"\n\s*\n", reference) if x.strip()]
    h = [x.strip() for x in re.split(r"\n\s*\n", hypothesis) if x.strip()]
    if not r:
        return 1.0 if not h else 0.0
    if not h:
        return 0.0
    return max(0.0, 1.0 - (_edit_distance(r, h) / len(r)))


def _runtime_quantiles(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    ordered = sorted(values)
    q90_index = int(round((len(ordered) - 1) * 0.9))
    q95_index = int(round((len(ordered) - 1) * 0.95))
    return ordered[q90_index], ordered[q95_index]


def _set_f1(a: set[str], b: set[str]) -> float | None:
    if not a and not b:
        return None
    if not a or not b:
        return 0.0
    tp = len(a & b)
    p = tp / len(b)
    r = tp / len(a)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _normalize_text(text: Any) -> str:
    out = str(text or "")
    return out.replace("\\r\\n", "\n").replace("\\r", "\n").replace("\\n", "\n")


def _norm_token(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.,/%+-]+", " ", text.lower())).strip()


def _norm_amount(text: str) -> str:
    return re.sub(r"[^0-9.,-]", "", text)


def _extract_kv_basic(text: str) -> dict[str, Any]:
    key_values: list[dict[str, str]] = []
    for line in text.splitlines():
        s = line.strip()
        if ":" not in s:
            continue
        key, value = s.split(":", 1)
        k = _norm_token(key)
        v = _norm_token(value)
        if k and v:
            key_values.append({"key": k, "value": v})
    return {
        "parser": "basic_colon",
        "key_values": key_values,
        "table_rows": [],
    }


def _extract_receipt_structured(text: str) -> dict[str, Any]:
    amount_re = re.compile(r"[-+]?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,3})?")
    keywords = [
        "grand total",
        "total",
        "subtotal",
        "cash",
        "change",
        "tax",
        "vat",
        "discount",
        "service",
        "amount",
    ]
    key_values: list[dict[str, str]] = []
    table_rows: list[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        amounts = amount_re.findall(line)
        if not amounts:
            continue
        line_lower = line.lower()
        matched_key = ""
        for kw in keywords:
            if kw in line_lower:
                matched_key = kw
                break

        if matched_key:
            key_values.append({"key": _norm_token(matched_key), "value": _norm_amount(amounts[-1])})
            continue

        label = amount_re.sub(" ", line)
        label = re.sub(r"\s+", " ", label).strip(" -:.;")
        if re.search(r"[a-zA-Z]", label):
            table_rows.append(f"{_norm_token(label)}|{_norm_amount(amounts[-1])}")

    dedup_kv = []
    seen_kv: set[tuple[str, str]] = set()
    for kv in key_values:
        pair = (kv["key"], kv["value"])
        if pair in seen_kv:
            continue
        seen_kv.add(pair)
        dedup_kv.append(kv)

    table_rows = sorted(set(row for row in table_rows if row and not row.startswith("|")))
    return {
        "parser": "receipt_specialist",
        "key_values": dedup_kv,
        "table_rows": table_rows,
    }


def _extract_form_structured(text: str) -> dict[str, Any]:
    key_values: list[dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        key = ""
        value = ""
        if ":" in line:
            key, value = line.split(":", 1)
        elif re.search(r"\s{2,}", line):
            parts = re.split(r"\s{2,}", line, maxsplit=1)
            if len(parts) == 2:
                key, value = parts
        else:
            m = re.match(r"([A-Za-z][A-Za-z0-9\-/() .]{2,})\s+([A-Za-z0-9\-/.,%]{2,})$", line)
            if m:
                key = m.group(1)
                value = m.group(2)

        k = _norm_token(key)
        v = _norm_token(value)
        if k and v:
            key_values.append({"key": k, "value": v})

    dedup_kv = []
    seen_kv: set[tuple[str, str]] = set()
    for kv in key_values:
        pair = (kv["key"], kv["value"])
        if pair in seen_kv:
            continue
        seen_kv.add(pair)
        dedup_kv.append(kv)

    return {
        "parser": "form_specialist",
        "key_values": dedup_kv,
        "table_rows": [],
    }


def _structured_from_text(text: str, category: str, specialist: bool) -> dict[str, Any]:
    if not specialist:
        return _extract_kv_basic(text)
    if category == "receipts":
        return _extract_receipt_structured(text)
    if category == "forms":
        return _extract_form_structured(text)
    return _extract_kv_basic(text)


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


def _route_category(dataset_id: str, document_type: str) -> str:
    ds = dataset_id.lower()
    dt = document_type.lower()
    if "receipt" in dt or "cord" in ds:
        return "receipts"
    if "form" in dt or "funsd" in ds:
        return "forms"
    if "typewritten" in dt or "directory" in dt or "dahn" in ds:
        return "typewritten_documents"
    if "historical" in dt or "book" in dt or "ocrd" in ds:
        return "historical_books"
    if "local_gold" in ds or "academic" in dt:
        return "local_gold_academic_pages"
    if dt in {"", "unknown", "general"}:
        return "unknown_documents"
    return "general_pdfs"


def _route_name(category: str) -> str:
    mapping = {
        "receipts": "specialist_receipt_route",
        "forms": "specialist_form_route",
        "historical_books": "specialist_historical_reading_order_route",
        "typewritten_documents": "specialist_typewritten_line_preserving_route",
        "local_gold_academic_pages": "specialist_academic_scope_route",
        "general_pdfs": "general_safe_default_route",
        "unknown_documents": "unknown_safe_default_route",
    }
    return mapping.get(category, "general_safe_default_route")


def _route_confidence(dataset_id: str, document_type: str, layout_type: str, source_type: str) -> float:
    score = 0.55
    if document_type and document_type.lower() != "unknown":
        score += 0.28
    if dataset_id:
        score += 0.08
    if layout_type and layout_type.lower() != "unknown":
        score += 0.05
    if source_type.lower() != "pdf":
        score -= 0.2
    return max(0.05, min(0.99, score))


def _load_recommendations() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if not ROUTING_RECOMMENDATION_CSV.exists():
        return out
    for row in _read_csv(ROUTING_RECOMMENDATION_CSV):
        key = str(row.get("document_type", "")).strip()
        if not key:
            continue
        out[key] = {
            "recommended_backend": str(row.get("recommended_backend", CURRENT_BACKEND) or CURRENT_BACKEND),
            "fallback_backend": str(row.get("fallback_backend", CURRENT_BACKEND) or CURRENT_BACKEND),
            "reason_summary": str(row.get("reason_summary", "") or ""),
        }
    return out


def _load_gold_map() -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in _read_csv(GOLD_CSV):
        pdf_name = str(row.get("pdf_name", "") or "").strip()
        page = int(float(str(row.get("page", "1") or "1")))
        out[(pdf_name, page)] = {
            "gold_text": _normalize_text(row.get("ground_truth_text", "")),
            "dataset_id": str(row.get("dataset_id", "unknown") or "unknown"),
            "document_type": str(row.get("document_type", "unknown") or "unknown"),
            "layout_type": str(row.get("layout_type", "unknown") or "unknown"),
            "language_primary": str(row.get("language_primary", "unknown") or "unknown"),
            "page_reference": str(row.get("page_reference", "") or ""),
        }
    return out


def _load_backend_outputs() -> dict[tuple[str, str, int], dict[str, Any]]:
    outputs: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not RAW_DIR.exists():
        return outputs

    for backend_dir in sorted(RAW_DIR.iterdir()):
        if not backend_dir.is_dir():
            continue
        backend = backend_dir.name
        for path in sorted(backend_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            pdf_name = str(payload.get("pdf_name", "") or "")
            page = int(payload.get("page", 1) or 1)
            text = _normalize_text(payload.get("result_text", ""))
            outputs[(backend, pdf_name, page)] = {
                "backend": backend,
                "pdf_name": pdf_name,
                "page": page,
                "status": str(payload.get("status", "failed") or "failed"),
                "issue": str(payload.get("issue", "") or ""),
                "runtime_ms": _safe_float(payload.get("runtime_ms")) or 0.0,
                "text": text,
                "document_model": payload.get("document_model"),
            }
    return outputs


def _backend_health(outputs: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_backend: dict[str, list[dict[str, Any]]] = {}
    for row in outputs.values():
        by_backend.setdefault(str(row["backend"]), []).append(row)

    health: dict[str, dict[str, Any]] = {}
    for backend, rows in by_backend.items():
        total = len(rows)
        success = sum(1 for r in rows if r["status"] == "success")
        empty = sum(1 for r in rows if r["status"] == "success" and not str(r.get("text", "")).strip())
        health[backend] = {
            "pages": total,
            "success_pages": success,
            "failed_pages": total - success,
            "failed_rate": (total - success) / total if total else 1.0,
            "empty_rate": empty / success if success else 1.0,
            "eligible": success > 0 and (total - success) < total,
        }
    return health


def _iter_pages() -> list[tuple[str, int, Path]]:
    pages: list[tuple[str, int, Path]] = []
    for pdf in sorted(INPUT_DIR.glob("*.pdf")):
        pages.append((pdf.name, 1, pdf))
    return pages


def _pairs_from_structured(payload: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for kv in payload.get("key_values", []):
        if not isinstance(kv, dict):
            continue
        k = _norm_token(str(kv.get("key", "") or ""))
        v = _norm_token(str(kv.get("value", "") or ""))
        if k and v:
            out.add(f"{k}:{v}")
    return out


def _rows_from_structured(payload: dict[str, Any]) -> set[str]:
    rows = payload.get("table_rows", [])
    return {str(x).strip().lower() for x in rows if str(x).strip()}


def _build_structured_to_document_model(structured: dict[str, Any], page_id: str, source_engine: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kv_pairs: list[dict[str, Any]] = []
    for kv in structured.get("key_values", []):
        if not isinstance(kv, dict):
            continue
        key_text = str(kv.get("key", "") or "").strip()
        value_text = str(kv.get("value", "") or "").strip()
        if not key_text or not value_text:
            continue
        kv_pairs.append(
            {
                "key_text": key_text,
                "value_text": value_text,
                "key_bbox": None,
                "value_bbox": None,
                "confidence": None,
                "source": source_engine,
            }
        )

    tables: list[dict[str, Any]] = []
    table_rows = [str(r) for r in structured.get("table_rows", []) if str(r).strip()]
    if table_rows:
        cells: list[dict[str, Any]] = []
        for i, row in enumerate(table_rows):
            left, right = (row.split("|", 1) + [""])[:2]
            values = [left.strip(), right.strip()]
            for j, cell_text in enumerate(values):
                if not cell_text:
                    continue
                cells.append(
                    {
                        "cell_id": f"{page_id}_table_1_r{i+1}_c{j+1}",
                        "row_index": i,
                        "column_index": j,
                        "row_span": 1,
                        "column_span": 1,
                        "bbox": None,
                        "text": cell_text,
                        "confidence": None,
                        "reading_order": len(cells) + 1,
                    }
                )
        tables.append(
            {
                "table_id": f"{page_id}_table_1",
                "page_id": page_id,
                "bbox": None,
                "rows": len(table_rows),
                "columns": 2,
                "cells": cells,
                "text": "\n".join(table_rows),
                "confidence": None,
            }
        )

    return kv_pairs, tables


def _augment_document_model(
    model: dict[str, Any] | None,
    structured: dict[str, Any],
    selected_route: str,
    route_reason: str,
    route_confidence: float,
    backend_used: str,
    warnings: list[str],
    errors: list[str],
) -> dict[str, Any] | None:
    if not isinstance(model, dict):
        return None

    routed_model = copy.deepcopy(model)
    pages = routed_model.get("pages")
    if not isinstance(pages, list) or not pages:
        return routed_model

    page = pages[0]
    if not isinstance(page, dict):
        return routed_model

    page_id = str(page.get("page_id", "page_1") or "page_1")

    existing_struct = page.get("structured_json_output")
    if not isinstance(existing_struct, dict):
        existing_struct = {}
    merged_struct = dict(existing_struct)
    merged_struct["routed_structured_output"] = structured
    merged_struct["route_metadata"] = {
        "selected_route": selected_route,
        "route_reason": route_reason,
        "route_confidence": route_confidence,
        "backend_used": backend_used,
        "warnings": warnings,
        "errors": errors,
    }
    page["structured_json_output"] = merged_struct

    kv_pairs, tables = _build_structured_to_document_model(structured, page_id, backend_used)
    if kv_pairs:
        page["key_value_pairs"] = kv_pairs
    if tables:
        page["tables"] = tables

    provenance = page.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    provenance["route_metadata"] = {
        "selected_route": selected_route,
        "route_reason": route_reason,
        "route_confidence": route_confidence,
        "backend_used": backend_used,
    }
    page["provenance"] = provenance
    return routed_model


def _aggregate_variant_rows(rows: list[dict[str, Any]], variant: str) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset_id"]), str(row["document_type"]))
        grouped.setdefault(key, []).append(row)

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, items in grouped.items():
        cer_vals = [float(x["cer"]) for x in items if x["cer"] is not None]
        wer_vals = [float(x["wer"]) for x in items if x["wer"] is not None]
        line_vals = [float(x["line_order_similarity"]) for x in items if x["line_order_similarity"] is not None]
        para_vals = [float(x["paragraph_order_similarity"]) for x in items if x["paragraph_order_similarity"] is not None]
        kv_vals = [float(x["key_value_f1"]) for x in items if x["key_value_f1"] is not None]
        table_vals = [float(x["table_row_similarity"]) for x in items if x["table_row_similarity"] is not None]
        runtimes = [float(x["runtime_ms"]) for x in items]
        p90, p95 = _runtime_quantiles(runtimes)

        out[key] = {
            "pipeline_variant": variant,
            "dataset_id": key[0],
            "document_type": key[1],
            "pages": len(items),
            "cer_mean": statistics.mean(cer_vals) if cer_vals else None,
            "wer_mean": statistics.mean(wer_vals) if wer_vals else None,
            "line_order_similarity_mean": statistics.mean(line_vals) if line_vals else None,
            "paragraph_order_similarity_mean": statistics.mean(para_vals) if para_vals else None,
            "key_value_f1_mean": statistics.mean(kv_vals) if kv_vals else None,
            "table_row_similarity_mean": statistics.mean(table_vals) if table_vals else None,
            "failed_rate": sum(1 for x in items if x["status"] != "success") / len(items),
            "empty_rate": sum(1 for x in items if x["empty_output"]) / len(items),
            "runtime_ms_p90": p90,
            "runtime_ms_p95": p95,
            "structured_output_rate": sum(1 for x in items if x["structured_output_available"]) / len(items),
            "document_model_rate": sum(1 for x in items if x["document_model_available"]) / len(items),
            "items": items,
        }

    all_items = rows
    if all_items:
        cer_vals = [float(x["cer"]) for x in all_items if x["cer"] is not None]
        wer_vals = [float(x["wer"]) for x in all_items if x["wer"] is not None]
        line_vals = [float(x["line_order_similarity"]) for x in all_items if x["line_order_similarity"] is not None]
        para_vals = [float(x["paragraph_order_similarity"]) for x in all_items if x["paragraph_order_similarity"] is not None]
        kv_vals = [float(x["key_value_f1"]) for x in all_items if x["key_value_f1"] is not None]
        table_vals = [float(x["table_row_similarity"]) for x in all_items if x["table_row_similarity"] is not None]
        runtimes = [float(x["runtime_ms"]) for x in all_items]
        p90, p95 = _runtime_quantiles(runtimes)

        out[("ALL", "ALL")] = {
            "pipeline_variant": variant,
            "dataset_id": "ALL",
            "document_type": "ALL",
            "pages": len(all_items),
            "cer_mean": statistics.mean(cer_vals) if cer_vals else None,
            "wer_mean": statistics.mean(wer_vals) if wer_vals else None,
            "line_order_similarity_mean": statistics.mean(line_vals) if line_vals else None,
            "paragraph_order_similarity_mean": statistics.mean(para_vals) if para_vals else None,
            "key_value_f1_mean": statistics.mean(kv_vals) if kv_vals else None,
            "table_row_similarity_mean": statistics.mean(table_vals) if table_vals else None,
            "failed_rate": sum(1 for x in all_items if x["status"] != "success") / len(all_items),
            "empty_rate": sum(1 for x in all_items if x["empty_output"]) / len(all_items),
            "runtime_ms_p90": p90,
            "runtime_ms_p95": p95,
            "structured_output_rate": sum(1 for x in all_items if x["structured_output_available"]) / len(all_items),
            "document_model_rate": sum(1 for x in all_items if x["document_model_available"]) / len(all_items),
            "items": all_items,
        }

    return out


def _first_page_raw_ocr_text(model: dict[str, Any] | None) -> str:
    if not isinstance(model, dict):
        return ""
    pages = model.get("pages")
    if not isinstance(pages, list) or not pages:
        return ""
    first = pages[0]
    if not isinstance(first, dict):
        return ""
    return _normalize_text(first.get("raw_ocr_text", ""))


def main() -> int:
    recommendations = _load_recommendations()
    gold_map = _load_gold_map()
    backend_outputs = _load_backend_outputs()
    backend_health = _backend_health(backend_outputs)
    pages = _iter_pages()

    if not pages:
        raise RuntimeError("No benchmark pages found at reports/real_gold_eval_runs/smoke_50/input_pdfs")

    per_page_rows: list[dict[str, Any]] = []
    current_metric_rows: list[dict[str, Any]] = []
    routed_metric_rows: list[dict[str, Any]] = []

    for pdf_name, page, pdf_path in pages:
        meta = gold_map.get((pdf_name, page))
        if not meta:
            continue

        source_type = pdf_path.suffix.lstrip(".").lower() or "unknown"
        dataset_id = str(meta.get("dataset_id", "unknown") or "unknown")
        document_type = str(meta.get("document_type", "unknown") or "unknown")
        layout_type = str(meta.get("layout_type", "unknown") or "unknown")
        gold_text = _normalize_text(meta.get("gold_text", ""))

        category = _route_category(dataset_id, document_type)
        selected_route = _route_name(category)

        rec = recommendations.get(category, {
            "recommended_backend": CURRENT_BACKEND,
            "fallback_backend": CURRENT_BACKEND,
            "reason_summary": "No explicit recommendation row found; use safe current default.",
        })
        recommended_backend = str(rec.get("recommended_backend", CURRENT_BACKEND) or CURRENT_BACKEND)
        fallback_backend = str(rec.get("fallback_backend", CURRENT_BACKEND) or CURRENT_BACKEND)

        route_confidence = _route_confidence(dataset_id, document_type, layout_type, source_type)
        route_reason = (
            f"category={category}; metadata-driven route using dataset/document/layout fields; "
            f"recommended_backend={recommended_backend}; rationale={rec.get('reason_summary', '')}"
        )

        warnings: list[str] = []
        errors: list[str] = []
        fallback_triggered = False
        fallback_reason = ""
        fallback_behavior = "fallback_to_current_pipeline_on_unavailable_failed_or_empty_or_low_confidence"

        current_output = backend_outputs.get((CURRENT_BACKEND, pdf_name, page))
        if current_output is None:
            current_output = {
                "backend": CURRENT_BACKEND,
                "pdf_name": pdf_name,
                "page": page,
                "status": "failed",
                "issue": "missing current pipeline output",
                "runtime_ms": 0.0,
                "text": "",
                "document_model": None,
            }
            errors.append("missing_current_pipeline_output")

        selected_backend = recommended_backend
        selected_output = backend_outputs.get((selected_backend, pdf_name, page))
        selected_health = backend_health.get(selected_backend, {})

        if route_confidence < 0.6:
            warnings.append("low_route_confidence_using_default_backend")
            selected_backend = fallback_backend
            selected_output = backend_outputs.get((selected_backend, pdf_name, page))
            fallback_triggered = True
            fallback_reason = "low_route_confidence"

        if not selected_health.get("eligible", False):
            warnings.append("recommended_backend_not_eligible")
            selected_backend = fallback_backend
            selected_output = backend_outputs.get((selected_backend, pdf_name, page))
            fallback_triggered = True
            fallback_reason = fallback_reason or "recommended_backend_not_eligible"

        if selected_output is None:
            warnings.append("selected_backend_output_missing")
            selected_backend = CURRENT_BACKEND
            selected_output = current_output
            fallback_triggered = True
            fallback_reason = fallback_reason or "selected_backend_output_missing"

        if selected_output.get("status") != "success":
            warnings.append("selected_backend_failed_page_fallback")
            selected_backend = CURRENT_BACKEND
            selected_output = current_output
            fallback_triggered = True
            fallback_reason = fallback_reason or "selected_backend_failed_page"

        if selected_backend != CURRENT_BACKEND and not str(selected_output.get("text", "")).strip():
            warnings.append("selected_backend_empty_output_fallback")
            selected_backend = CURRENT_BACKEND
            selected_output = current_output
            fallback_triggered = True
            fallback_reason = fallback_reason or "selected_backend_empty_output"

        current_text = _normalize_text(current_output.get("text", ""))
        routed_text = _normalize_text(selected_output.get("text", ""))

        # Runtime routing uses metadata only; gold_text is only for offline benchmarking metrics.
        current_structured = _structured_from_text(current_text, category, specialist=False)
        routed_structured = _structured_from_text(routed_text, category, specialist=(category in {"receipts", "forms"}))
        reference_structured = _structured_from_text(gold_text, category, specialist=(category in {"receipts", "forms"}))

        current_kv_f1 = _set_f1(_pairs_from_structured(reference_structured), _pairs_from_structured(current_structured))
        routed_kv_f1 = _set_f1(_pairs_from_structured(reference_structured), _pairs_from_structured(routed_structured))
        current_table_sim = _set_f1(_rows_from_structured(reference_structured), _rows_from_structured(current_structured))
        routed_table_sim = _set_f1(_rows_from_structured(reference_structured), _rows_from_structured(routed_structured))

        current_runtime = float(current_output.get("runtime_ms", 0.0) or 0.0)
        routed_runtime = float(selected_output.get("runtime_ms", 0.0) or 0.0)
        if category in {"receipts", "forms"}:
            routed_runtime += 8.0  # lightweight specialist parsing overhead

        current_model = current_output.get("document_model")
        source_raw_ocr_text = _first_page_raw_ocr_text(selected_output.get("document_model"))

        routed_model = _augment_document_model(
            selected_output.get("document_model"),
            routed_structured,
            selected_route,
            route_reason,
            route_confidence,
            selected_backend,
            warnings,
            errors,
        )

        raw_ocr_text = _first_page_raw_ocr_text(routed_model)
        if source_raw_ocr_text:
            raw_preserved = raw_ocr_text == source_raw_ocr_text
        elif raw_ocr_text:
            raw_preserved = raw_ocr_text == routed_text
        else:
            raw_preserved = True

        if not raw_preserved:
            warnings.append("raw_ocr_not_exactly_preserved_in_document_model")

        current_metric = {
            "variant": CURRENT_VARIANT,
            "pdf_name": pdf_name,
            "page": page,
            "dataset_id": dataset_id,
            "document_type": document_type,
            "route_category": category,
            "metric_family": _metric_family(dataset_id, document_type),
            "status": str(current_output.get("status", "failed") or "failed"),
            "empty_output": not bool(current_text.strip()),
            "runtime_ms": current_runtime,
            "cer": _cer(gold_text, current_text) if str(current_output.get("status", "")) == "success" else None,
            "wer": _wer(gold_text, current_text) if str(current_output.get("status", "")) == "success" else None,
            "line_order_similarity": _line_order_similarity(gold_text, current_text) if str(current_output.get("status", "")) == "success" else None,
            "paragraph_order_similarity": _paragraph_order_similarity(gold_text, current_text) if str(current_output.get("status", "")) == "success" else None,
            "key_value_f1": current_kv_f1 if str(current_output.get("status", "")) == "success" else None,
            "table_row_similarity": current_table_sim if str(current_output.get("status", "")) == "success" else None,
            "structured_output_available": bool(current_structured.get("key_values") or current_structured.get("table_rows")),
            "document_model_available": isinstance(current_model, dict),
        }

        routed_metric = {
            "variant": ROUTED_VARIANT,
            "pdf_name": pdf_name,
            "page": page,
            "dataset_id": dataset_id,
            "document_type": document_type,
            "route_category": category,
            "metric_family": _metric_family(dataset_id, document_type),
            "status": str(selected_output.get("status", "failed") or "failed"),
            "empty_output": not bool(routed_text.strip()),
            "runtime_ms": routed_runtime,
            "cer": _cer(gold_text, routed_text) if str(selected_output.get("status", "")) == "success" else None,
            "wer": _wer(gold_text, routed_text) if str(selected_output.get("status", "")) == "success" else None,
            "line_order_similarity": _line_order_similarity(gold_text, routed_text) if str(selected_output.get("status", "")) == "success" else None,
            "paragraph_order_similarity": _paragraph_order_similarity(gold_text, routed_text) if str(selected_output.get("status", "")) == "success" else None,
            "key_value_f1": routed_kv_f1 if str(selected_output.get("status", "")) == "success" else None,
            "table_row_similarity": routed_table_sim if str(selected_output.get("status", "")) == "success" else None,
            "structured_output_available": bool(routed_structured.get("key_values") or routed_structured.get("table_rows")),
            "document_model_available": isinstance(routed_model, dict),
        }

        current_metric_rows.append(current_metric)
        routed_metric_rows.append(routed_metric)

        per_page_rows.append(
            {
                "pdf_name": pdf_name,
                "page": page,
                "dataset_id": dataset_id,
                "document_type": document_type,
                "layout_type": layout_type,
                "source_type": source_type,
                "route_category": category,
                "selected_route": selected_route,
                "route_reason": route_reason,
                "route_confidence": round(route_confidence, 4),
                "backend_recommended": recommended_backend,
                "backend_used": selected_backend,
                "fallback_behavior": fallback_behavior,
                "fallback_triggered": fallback_triggered,
                "fallback_reason": fallback_reason,
                "warnings": warnings,
                "errors": errors,
                "raw_ocr_preserved": raw_preserved,
                "raw_ocr_text": raw_ocr_text,
                "routed_text": routed_text,
                "structured_output": routed_structured,
                "document_model": routed_model,
                "current_metrics": current_metric,
                "routed_metrics": routed_metric,
            }
        )

    with OUT_PER_PAGE_JSONL.open("w", encoding="utf-8") as fh:
        for row in per_page_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    current_agg = _aggregate_variant_rows(current_metric_rows, CURRENT_VARIANT)
    routed_agg = _aggregate_variant_rows(routed_metric_rows, ROUTED_VARIANT)

    # Build CSV comparison rows.
    all_keys = sorted(set(current_agg.keys()) | set(routed_agg.keys()))
    metrics_rows: list[dict[str, Any]] = []

    pair_index: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    current_by_key = {(r["pdf_name"], int(r["page"])): r for r in current_metric_rows}
    routed_by_key = {(r["pdf_name"], int(r["page"])): r for r in routed_metric_rows}
    for k in current_by_key:
        if k in routed_by_key:
            pair_index[k] = (current_by_key[k], routed_by_key[k])

    for key in all_keys:
        c = current_agg.get(key)
        r = routed_agg.get(key)

        if c is not None:
            metrics_rows.append(
                {
                    "pipeline_variant": c["pipeline_variant"],
                    "dataset_id": c["dataset_id"],
                    "document_type": c["document_type"],
                    "pages": c["pages"],
                    "cer_mean": c["cer_mean"],
                    "wer_mean": c["wer_mean"],
                    "line_order_similarity_mean": c["line_order_similarity_mean"],
                    "paragraph_order_similarity_mean": c["paragraph_order_similarity_mean"],
                    "key_value_f1_mean": c["key_value_f1_mean"],
                    "table_row_similarity_mean": c["table_row_similarity_mean"],
                    "failed_rate": c["failed_rate"],
                    "empty_rate": c["empty_rate"],
                    "runtime_ms_p90": c["runtime_ms_p90"],
                    "runtime_ms_p95": c["runtime_ms_p95"],
                    "structured_output_rate": c["structured_output_rate"],
                    "document_model_rate": c["document_model_rate"],
                    "non_regression_vs_current": "baseline",
                    "cer_delta_vs_current": 0.0,
                    "wer_delta_vs_current": 0.0,
                    "improved_pages_vs_current": 0,
                    "regressed_pages_vs_current": 0,
                }
            )

        if r is not None:
            c_cer = c.get("cer_mean") if c else None
            c_wer = c.get("wer_mean") if c else None
            cer_delta = (r.get("cer_mean") - c_cer) if (r.get("cer_mean") is not None and c_cer is not None) else None
            wer_delta = (r.get("wer_mean") - c_wer) if (r.get("wer_mean") is not None and c_wer is not None) else None

            improved_pages = 0
            regressed_pages = 0
            for (pdf_name, page), (cm, rm) in pair_index.items():
                if str(cm.get("dataset_id")) != str(r["dataset_id"]):
                    continue
                if str(cm.get("document_type")) != str(r["document_type"]):
                    continue
                c_cer_page = cm.get("cer")
                r_cer_page = rm.get("cer")
                if c_cer_page is None or r_cer_page is None:
                    continue
                if r_cer_page < c_cer_page - 1e-9:
                    improved_pages += 1
                elif r_cer_page > c_cer_page + 1e-9:
                    regressed_pages += 1

            non_reg = "pass"
            if r.get("failed_rate", 1.0) > (c.get("failed_rate", 1.0) if c else 1.0):
                non_reg = "fail"
            if r.get("empty_rate", 1.0) > (c.get("empty_rate", 1.0) if c else 1.0):
                non_reg = "fail"
            if cer_delta is not None and cer_delta > 0.0:
                non_reg = "fail"

            metrics_rows.append(
                {
                    "pipeline_variant": r["pipeline_variant"],
                    "dataset_id": r["dataset_id"],
                    "document_type": r["document_type"],
                    "pages": r["pages"],
                    "cer_mean": r["cer_mean"],
                    "wer_mean": r["wer_mean"],
                    "line_order_similarity_mean": r["line_order_similarity_mean"],
                    "paragraph_order_similarity_mean": r["paragraph_order_similarity_mean"],
                    "key_value_f1_mean": r["key_value_f1_mean"],
                    "table_row_similarity_mean": r["table_row_similarity_mean"],
                    "failed_rate": r["failed_rate"],
                    "empty_rate": r["empty_rate"],
                    "runtime_ms_p90": r["runtime_ms_p90"],
                    "runtime_ms_p95": r["runtime_ms_p95"],
                    "structured_output_rate": r["structured_output_rate"],
                    "document_model_rate": r["document_model_rate"],
                    "non_regression_vs_current": non_reg,
                    "cer_delta_vs_current": cer_delta,
                    "wer_delta_vs_current": wer_delta,
                    "improved_pages_vs_current": improved_pages,
                    "regressed_pages_vs_current": regressed_pages,
                }
            )

    with OUT_METRICS_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "pipeline_variant",
                "dataset_id",
                "document_type",
                "pages",
                "cer_mean",
                "wer_mean",
                "line_order_similarity_mean",
                "paragraph_order_similarity_mean",
                "key_value_f1_mean",
                "table_row_similarity_mean",
                "failed_rate",
                "empty_rate",
                "runtime_ms_p90",
                "runtime_ms_p95",
                "structured_output_rate",
                "document_model_rate",
                "non_regression_vs_current",
                "cer_delta_vs_current",
                "wer_delta_vs_current",
                "improved_pages_vs_current",
                "regressed_pages_vs_current",
            ],
        )
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    # Criteria and decision.
    current_all = current_agg.get(("ALL", "ALL"), {})
    routed_all = routed_agg.get(("ALL", "ALL"), {})

    fail_empty_non_reg = (
        routed_all.get("failed_rate", 1.0) <= current_all.get("failed_rate", 1.0)
        and routed_all.get("empty_rate", 1.0) <= current_all.get("empty_rate", 1.0)
    )

    runtime_bounded = True
    c_p95 = current_all.get("runtime_ms_p95")
    r_p95 = routed_all.get("runtime_ms_p95")
    if c_p95 is not None and r_p95 is not None:
        runtime_bounded = float(r_p95) <= (float(c_p95) * 1.25 + 50.0)

    raw_ocr_preserved = all(bool(r.get("raw_ocr_preserved", False)) for r in per_page_rows)

    def _category_delta(category: str, metric: str) -> float | None:
        pairs: list[float] = []
        for page_row in per_page_rows:
            if str(page_row.get("route_category")) != category:
                continue
            c_val = page_row.get("current_metrics", {}).get(metric)
            r_val = page_row.get("routed_metrics", {}).get(metric)
            if c_val is None or r_val is None:
                continue
            pairs.append(float(r_val) - float(c_val))
        if not pairs:
            return None
        return statistics.mean(pairs)

    receipts_kv_delta = _category_delta("receipts", "key_value_f1")
    forms_kv_delta = _category_delta("forms", "key_value_f1")
    receipts_table_delta = _category_delta("receipts", "table_row_similarity")
    forms_table_delta = _category_delta("forms", "table_row_similarity")

    structured_improves_receipts = (
        (receipts_kv_delta is not None and receipts_kv_delta > 0)
        or (receipts_table_delta is not None and receipts_table_delta > 0)
    )
    structured_improves_forms = (
        (forms_kv_delta is not None and forms_kv_delta > 0)
        or (forms_table_delta is not None and forms_table_delta > 0)
    )
    structured_improvement = structured_improves_receipts and structured_improves_forms

    plain_categories = {"historical_books", "typewritten_documents", "general_pdfs"}
    plain_non_regression = True
    for category in plain_categories:
        cer_delta = _category_delta(category, "cer")
        wer_delta = _category_delta(category, "wer")
        if cer_delta is not None and cer_delta > 0.0:
            plain_non_regression = False
        if wer_delta is not None and wer_delta > 0.0:
            plain_non_regression = False

    broad_win_categories = 0
    for category in {"receipts", "forms", "historical_books", "typewritten_documents", "local_gold_academic_pages", "general_pdfs", "unknown_documents"}:
        cer_delta = _category_delta(category, "cer")
        if cer_delta is not None and cer_delta < 0.0:
            broad_win_categories += 1

    if fail_empty_non_reg and structured_improvement and plain_non_regression and runtime_bounded and raw_ocr_preserved:
        if broad_win_categories >= 3:
            decision = "promote_routed_pipeline_global"
            decision_reason = "Broad non-regressing wins across multiple categories with bounded runtime."
        else:
            decision = "promote_specialist_routes_only"
            decision_reason = "Improvements are concentrated in specialist structured categories without broad cross-category gains."
    else:
        decision = "keep_routed_pipeline_experimental"
        decision_reason = "One or more conservative gates failed; keep routed logic non-default and auditable."

    # Detailed report.
    with OUT_REPORT_MD.open("w", encoding="utf-8") as fh:
        fh.write("# Routed Document Pipeline Experiment Report\n\n")
        fh.write("## Scope\n")
        fh.write("- Benchmark subset: reports/real_gold_eval_runs/smoke_50/input_pdfs (50 pages).\n")
        fh.write("- Comparison: current pipeline vs. routed document-type prototype.\n")
        fh.write("- Runtime routing inputs only: document metadata (dataset/document/layout/source type); no ground-truth text used at routing time.\n")
        fh.write("- Evaluation metrics (offline): CER/WER continuity, structured metrics, reliability, and latency.\n\n")

        fh.write("## Route Policy Used\n")
        fh.write("- receipts -> specialist_receipt_route (recommended backend from Phase 5)\n")
        fh.write("- forms -> specialist_form_route (recommended backend from Phase 5)\n")
        fh.write("- historical books -> specialist_historical_reading_order_route\n")
        fh.write("- typewritten documents -> specialist_typewritten_line_preserving_route\n")
        fh.write("- local_gold/academic -> specialist_academic_scope_route\n")
        fh.write("- general -> general_safe_default_route\n")
        fh.write("- unknown -> unknown_safe_default_route\n\n")

        fh.write("## Backend Health Snapshot\n")
        for backend in sorted(backend_health.keys()):
            h = backend_health[backend]
            fh.write(
                f"- {backend}: pages={h['pages']} success={h['success_pages']} failed={h['failed_pages']} failed_rate={h['failed_rate']:.4f} empty_rate={h['empty_rate']:.4f} eligible={h['eligible']}\n"
            )
        fh.write("\n")

        fh.write("## Aggregate Comparison (ALL)\n")
        fh.write(
            f"- current: CER={current_all.get('cer_mean')} WER={current_all.get('wer_mean')} failed_rate={current_all.get('failed_rate')} empty_rate={current_all.get('empty_rate')} runtime_p95={current_all.get('runtime_ms_p95')}\n"
        )
        fh.write(
            f"- routed: CER={routed_all.get('cer_mean')} WER={routed_all.get('wer_mean')} failed_rate={routed_all.get('failed_rate')} empty_rate={routed_all.get('empty_rate')} runtime_p95={routed_all.get('runtime_ms_p95')}\n"
        )
        fh.write("\n")

        fh.write("## Structured Metric Delta (Routed - Current)\n")
        fh.write(f"- receipts key_value_f1 delta: {receipts_kv_delta}\n")
        fh.write(f"- receipts table_row_similarity delta: {receipts_table_delta}\n")
        fh.write(f"- forms key_value_f1 delta: {forms_kv_delta}\n")
        fh.write(f"- forms table_row_similarity delta: {forms_table_delta}\n\n")

        fh.write("## Conservative Gates\n")
        fh.write(f"- failed/empty non-regression: {fail_empty_non_reg}\n")
        fh.write(f"- structured improvement receipts/forms: {structured_improvement}\n")
        fh.write(f"- historical/plain-text non-regression: {plain_non_regression}\n")
        fh.write(f"- runtime bounded: {runtime_bounded}\n")
        fh.write(f"- raw OCR preserved: {raw_ocr_preserved}\n\n")

        fh.write("## Regressions (not hidden)\n")
        regressions: list[str] = []
        for row in per_page_rows:
            c = row.get("current_metrics", {})
            r = row.get("routed_metrics", {})
            c_cer = c.get("cer")
            r_cer = r.get("cer")
            if c_cer is None or r_cer is None:
                continue
            if float(r_cer) > float(c_cer) + 1e-9:
                regressions.append(
                    f"- {row['pdf_name']} page {row['page']} category={row['route_category']} CER {c_cer:.6f} -> {r_cer:.6f} route={row['selected_route']} backend={row['backend_used']}"
                )
        if regressions:
            for line in regressions[:30]:
                fh.write(line + "\n")
        else:
            fh.write("- No CER regressions detected in this routed prototype run.\n")
        fh.write("\n")

        fh.write("## Decision Summary\n")
        fh.write(f"- decision: {decision}\n")
        fh.write(f"- reason: {decision_reason}\n")
        fh.write("- Global promotion is blocked unless broad cross-category wins are demonstrated.\n")

    with OUT_PROMOTION_MD.open("w", encoding="utf-8") as fh:
        fh.write("# Routed Pipeline Promotion Decision\n\n")
        fh.write(f"- decision: {decision}\n")
        fh.write(f"- reason: {decision_reason}\n\n")
        fh.write("## Acceptance Criteria Check\n")
        fh.write(f"- failed/empty non-regression: {fail_empty_non_reg}\n")
        fh.write(f"- structured metrics improve for receipts/forms: {structured_improvement}\n")
        fh.write(f"- historical/plain-text non-regression: {plain_non_regression}\n")
        fh.write(f"- runtime bounded: {runtime_bounded}\n")
        fh.write(f"- raw OCR preserved: {raw_ocr_preserved}\n")
        fh.write(f"- broad win category count: {broad_win_categories}\n\n")
        fh.write("## Routing Guardrails\n")
        fh.write("- Routing remains auditable per page via selected_route, route_reason, route_confidence, backend_used, and fallback fields in routed_pipeline_per_page_outputs.jsonl.\n")
        fh.write("- No route uses ground_truth_text at runtime; gold is used only for offline evaluation.\n")
        fh.write("- If improvements are narrow, promote specialist route only and keep global default unchanged.\n")

    print(f"Wrote {OUT_PER_PAGE_JSONL}")
    print(f"Wrote {OUT_METRICS_CSV}")
    print(f"Wrote {OUT_REPORT_MD}")
    print(f"Wrote {OUT_PROMOTION_MD}")
    print(f"Decision: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
