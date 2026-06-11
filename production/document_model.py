from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0.0"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return None


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _bbox_from_any(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        vals = [_safe_float(x) for x in value]
        if all(v is not None for v in vals):
            return [float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])]
    return None


def _normalize_block_type(value: str) -> str:
    lower = (value or "").strip().lower()
    if not lower:
        return "unknown"
    mapping = {
        "text": "text",
        "paragraph": "paragraph",
        "title": "title",
        "header": "header",
        "footer": "footer",
        "table": "table",
        "figure": "figure",
        "image": "figure",
        "caption": "caption",
        "footnote": "footnote",
        "key_value": "key_value",
        "form_field": "key_value",
        "marginal_note": "marginal_note",
        "bibliography": "bibliography",
    }
    return mapping.get(lower, lower)


def _split_words_with_bbox(text: str, bbox: list[float] | None) -> list[dict[str, Any]]:
    words = [w for w in text.split() if w.strip()]
    if not words:
        return []
    if bbox is None:
        return [
            {
                "word_id": f"w{i+1}",
                "text": word,
                "bbox": None,
                "confidence": None,
                "reading_order": i + 1,
            }
            for i, word in enumerate(words)
        ]

    x0, y0, x1, y1 = bbox
    width = max(0.0, x1 - x0)
    if width <= 0:
        return [
            {
                "word_id": f"w{i+1}",
                "text": word,
                "bbox": bbox,
                "confidence": None,
                "reading_order": i + 1,
            }
            for i, word in enumerate(words)
        ]

    step = width / max(1, len(words))
    out: list[dict[str, Any]] = []
    for i, word in enumerate(words):
        wx0 = x0 + (i * step)
        wx1 = x0 + ((i + 1) * step)
        out.append(
            {
                "word_id": f"w{i+1}",
                "text": word,
                "bbox": [wx0, y0, wx1, y1],
                "confidence": None,
                "reading_order": i + 1,
            }
        )
    return out


def _extract_key_values(text: str, source: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key_text = key.strip()
        value_text = value.strip()
        if not key_text or not value_text:
            continue
        out.append(
            {
                "key_text": key_text,
                "value_text": value_text,
                "key_bbox": None,
                "value_bbox": None,
                "confidence": None,
                "source": source,
            }
        )
    return out


def _extract_tables(blocks: list[dict[str, Any]], page_id: str) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    t_index = 0
    for block in blocks:
        if block.get("block_type") != "table":
            continue
        t_index += 1
        text = str(block.get("text", "") or "")
        rows = [line.strip() for line in text.splitlines() if line.strip()]
        if not rows:
            rows = [text] if text else []
        columns = 1
        if rows:
            columns = max(1, max(len(re.split(r"\s{2,}|\t|\|", row)) for row in rows))

        cells: list[dict[str, Any]] = []
        for r_i, row_text in enumerate(rows):
            parts = [p.strip() for p in re.split(r"\s{2,}|\t|\|", row_text) if p.strip()]
            if not parts:
                parts = [row_text]
            for c_i, cell_text in enumerate(parts):
                cells.append(
                    {
                        "cell_id": f"{page_id}_t{t_index}_r{r_i+1}_c{c_i+1}",
                        "row_index": r_i,
                        "column_index": c_i,
                        "row_span": 1,
                        "column_span": 1,
                        "bbox": None,
                        "text": cell_text,
                        "confidence": block.get("confidence"),
                        "reading_order": len(cells) + 1,
                    }
                )

        tables.append(
            {
                "table_id": f"{page_id}_table_{t_index}",
                "page_id": page_id,
                "bbox": block.get("bbox"),
                "rows": len(rows),
                "columns": columns,
                "cells": cells,
                "text": text,
                "confidence": block.get("confidence"),
            }
        )
    return tables


def _blocks_from_layout(layout: dict[str, Any], page_id: str, source_engine: str) -> list[dict[str, Any]]:
    regions = layout.get("regions", []) if isinstance(layout, dict) else []
    blocks: list[dict[str, Any]] = []
    for i, region in enumerate(regions):
        if not isinstance(region, dict):
            continue
        bbox = _bbox_from_any(region.get("bbox"))
        block_text = str(region.get("text", "") or "")
        block_id = str(region.get("region_id", "") or f"{page_id}_b{i+1}")
        block: dict[str, Any] = {
            "block_id": block_id,
            "page_id": page_id,
            "block_type": _normalize_block_type(str(region.get("type", "unknown") or "unknown")),
            "bbox": bbox,
            "text": block_text,
            "confidence": _safe_float(region.get("confidence")),
            "reading_order": _safe_int(region.get("reading_order")) or (i + 1),
            "source_engine": source_engine,
            "provenance": {
                "ordering_source": str(region.get("ordering_source", "") or ""),
                "line_reading_confidence": _safe_float(region.get("line_reading_confidence")),
                "source_index": _safe_int(region.get("source_index")),
            },
            "lines": [],
        }

        line_items = region.get("line_ordering", []) if isinstance(region.get("line_ordering", []), list) else []
        if not line_items and block_text:
            line_items = [
                {
                    "text": line,
                    "bbox": bbox,
                    "reading_order": idx + 1,
                    "confidence": block["confidence"],
                    "ordering_source": "fallback",
                }
                for idx, line in enumerate([x for x in block_text.splitlines() if x.strip()])
            ]

        for li, line in enumerate(line_items):
            if not isinstance(line, dict):
                continue
            line_text = str(line.get("text", "") or "")
            line_bbox = _bbox_from_any(line.get("bbox"))
            line_id = f"{block_id}_l{li+1}"
            line_record = {
                "line_id": line_id,
                "block_id": block_id,
                "page_id": page_id,
                "text": line_text,
                "bbox": line_bbox,
                "confidence": _safe_float(line.get("confidence")),
                "reading_order": _safe_int(line.get("reading_order")) or (li + 1),
                "source_engine": source_engine,
                "provenance": {
                    "ordering_source": str(line.get("ordering_source", "") or ""),
                },
            }
            line_record["words"] = _split_words_with_bbox(line_text, line_bbox)
            block["lines"].append(line_record)

        blocks.append(block)
    return blocks


def _fallback_block(page_id: str, text: str, source_engine: str) -> list[dict[str, Any]]:
    return [
        {
            "block_id": f"{page_id}_b1",
            "page_id": page_id,
            "block_type": "text",
            "bbox": None,
            "text": text,
            "confidence": None,
            "reading_order": 1,
            "source_engine": source_engine,
            "provenance": {"ordering_source": "fallback"},
            "lines": [
                {
                    "line_id": f"{page_id}_b1_l1",
                    "block_id": f"{page_id}_b1",
                    "page_id": page_id,
                    "text": text,
                    "bbox": None,
                    "confidence": None,
                    "reading_order": 1,
                    "source_engine": source_engine,
                    "provenance": {"ordering_source": "fallback"},
                    "words": _split_words_with_bbox(text, None),
                }
            ],
        }
    ]


def _layout_key(pdf_name: str, page: int) -> tuple[str, int]:
    return (str(pdf_name).strip(), int(page))


def load_layout_records(layout_jsonl_path: str | Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    if not layout_jsonl_path:
        return out
    path = Path(layout_jsonl_path)
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            payload = line.strip()
            if not payload:
                continue
            obj = _parse_jsonish(payload)
            if not isinstance(obj, dict):
                continue
            pdf_name = str(obj.get("pdf_name", "") or "")
            page = _safe_int(obj.get("page"))
            if not pdf_name or page is None:
                continue
            out[_layout_key(pdf_name, page)] = obj
    return out


def convert_row_to_document_model(row: dict[str, Any], layout_record: dict[str, Any] | None = None) -> dict[str, Any]:
    pdf_name = str(row.get("pdf_name", "") or "")
    page = _safe_int(row.get("page")) or 1
    document_id = str(row.get("document_id", "") or Path(pdf_name).stem or "unknown_document")
    page_id = str(row.get("page_id", "") or f"{document_id}_page_{page}")
    source_engine = str(row.get("engine_used", "") or row.get("extraction_method", "unknown") or "unknown")

    page_text = str(row.get("page_text", "") or "").replace("\\n", "\n")
    raw_text = str(row.get("raw_text", "") or "")
    if not raw_text:
        raw_text = page_text

    structured_layout = None
    reading_order_text = page_text
    if isinstance(layout_record, dict):
        structured_layout = layout_record.get("layout")
        reading_order_text = str(layout_record.get("plain_text_reconstruction", "") or reading_order_text)
    else:
        structured_layout = _parse_jsonish(row.get("structured_layout", ""))

    if not reading_order_text:
        reading_order_text = page_text

    blocks = _blocks_from_layout(structured_layout if isinstance(structured_layout, dict) else {}, page_id, source_engine)
    if not blocks:
        blocks = _fallback_block(page_id, reading_order_text, source_engine)

    tables = _extract_tables(blocks, page_id)
    key_values = _extract_key_values(reading_order_text, source_engine)
    figures = [
        {
            "figure_id": f"{page_id}_fig_{i+1}",
            "page_id": page_id,
            "bbox": block.get("bbox"),
            "caption": "",
            "confidence": block.get("confidence"),
            "source_engine": source_engine,
        }
        for i, block in enumerate(blocks)
        if block.get("block_type") in {"figure", "image"}
    ]

    width = _safe_float(row.get("width"))
    height = _safe_float(row.get("height"))
    if isinstance(structured_layout, dict):
        page_size = structured_layout.get("page_size")
        if isinstance(page_size, list) and len(page_size) == 2:
            width = width if width is not None else _safe_float(page_size[0])
            height = height if height is not None else _safe_float(page_size[1])

    markdown_output = "\n\n".join([line for line in reading_order_text.splitlines() if line.strip()])

    page_model = {
        "page_id": page_id,
        "page_number": page,
        "dimensions": {
            "width": width,
            "height": height,
            "unit": "px",
        },
        "raw_ocr_text": raw_text,
        "reading_order_text": reading_order_text,
        "markdown_output": markdown_output,
        "structured_json_output": structured_layout if isinstance(structured_layout, dict) else {},
        "blocks": blocks,
        "tables": tables,
        "key_value_pairs": key_values,
        "figures": figures,
        "provenance": {
            "source_pipeline": "run_page_text",
            "source_files": {
                "ocr_csv": "client_page_text.csv",
                "layout_jsonl": "layout_regions.jsonl",
            },
            "extraction_method": str(row.get("extraction_method", "") or ""),
            "engine_statuses": _parse_jsonish(row.get("engine_statuses", "")) or {},
            "reading_order_source": str(row.get("reading_order_source", "") or ""),
            "final_output_source": str(row.get("final_output_source", "") or ""),
        },
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": document_id,
        "pages": [page_model],
        "provenance": {
            "record_origin": "ocr_pipeline",
            "converter": "production.document_model.convert_row_to_document_model",
        },
    }


def build_document_models_from_csv(
    ocr_csv_path: str | Path,
    layout_jsonl_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    layouts = load_layout_records(layout_jsonl_path)
    models: list[dict[str, Any]] = []
    with Path(ocr_csv_path).open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pdf_name = str(row.get("pdf_name", "") or "")
            page = _safe_int(row.get("page")) or 1
            layout_record = layouts.get(_layout_key(pdf_name, page))
            models.append(convert_row_to_document_model(row, layout_record=layout_record))
    return models


def validate_document_model(model: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(model, dict):
        return ["model must be an object"]

    for key in ["schema_version", "document_id", "pages", "provenance"]:
        if key not in model:
            errors.append(f"missing top-level field: {key}")

    pages = model.get("pages")
    if not isinstance(pages, list) or not pages:
        errors.append("pages must be a non-empty list")
        return errors

    for p_index, page in enumerate(pages):
        path = f"pages[{p_index}]"
        if not isinstance(page, dict):
            errors.append(f"{path} must be an object")
            continue
        for key in [
            "page_id",
            "dimensions",
            "blocks",
            "tables",
            "key_value_pairs",
            "figures",
            "raw_ocr_text",
            "reading_order_text",
            "markdown_output",
            "structured_json_output",
            "provenance",
        ]:
            if key not in page:
                errors.append(f"missing field: {path}.{key}")

        blocks = page.get("blocks", [])
        if not isinstance(blocks, list):
            errors.append(f"{path}.blocks must be a list")
            continue

        for b_index, block in enumerate(blocks):
            bpath = f"{path}.blocks[{b_index}]"
            if not isinstance(block, dict):
                errors.append(f"{bpath} must be an object")
                continue
            for key in [
                "block_id",
                "page_id",
                "block_type",
                "bbox",
                "text",
                "confidence",
                "reading_order",
                "source_engine",
                "provenance",
            ]:
                if key not in block:
                    errors.append(f"missing field: {bpath}.{key}")

    return errors
