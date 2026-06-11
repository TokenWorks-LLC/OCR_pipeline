from __future__ import annotations

from pathlib import Path
from typing import Any

from production.document_model import convert_row_to_document_model


def _base_row(
    pdf_name: str,
    page: int,
    page_text: str,
    raw_text: str,
    extraction_method: str,
    engine_used: str,
    width: float | None,
    height: float | None,
) -> dict[str, Any]:
    document_id = Path(pdf_name).stem
    return {
        "pdf_name": pdf_name,
        "page": page,
        "document_id": document_id,
        "page_id": f"{document_id}_page_{page}",
        "page_text": page_text,
        "raw_text": raw_text,
        "extraction_method": extraction_method,
        "engine_used": engine_used,
        "width": width,
        "height": height,
        "engine_statuses": "{}",
        "reading_order_source": "backend",
        "final_output_source": engine_used,
    }


def from_current_pipeline(
    row: dict[str, Any],
    structured_layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return convert_row_to_document_model(row, layout_record={"layout": structured_layout or {}, "plain_text_reconstruction": row.get("page_text", "")})


def from_paddle(
    *,
    pdf_name: str,
    page: int,
    text: str,
    lines: list[dict[str, Any]],
    width: float | None,
    height: float | None,
    runtime_ms: float,
) -> dict[str, Any]:
    regions = []
    for i, line in enumerate(lines):
        regions.append(
            {
                "region_id": f"p{page}_r{i+1}",
                "type": "text",
                "bbox": line.get("bbox"),
                "reading_order": i + 1,
                "confidence": line.get("confidence"),
                "ordering_source": "engine",
                "text": line.get("text", ""),
                "line_ordering": [
                    {
                        "text": line.get("text", ""),
                        "bbox": line.get("bbox"),
                        "reading_order": 1,
                        "confidence": line.get("confidence"),
                        "ordering_source": "engine",
                    }
                ],
            }
        )
    layout = {
        "regions": regions,
        "page_size": [width, height],
        "ordering_source": "paddleocr",
        "runtime_ms": runtime_ms,
    }
    row = _base_row(pdf_name, page, text, text, "paddleocr", "paddleocr", width, height)
    return convert_row_to_document_model(row, layout_record={"layout": layout, "plain_text_reconstruction": text})


def from_docling(
    *,
    pdf_name: str,
    page: int,
    text: str,
    markdown: str,
    structured: dict[str, Any],
    width: float | None,
    height: float | None,
) -> dict[str, Any]:
    row = _base_row(pdf_name, page, text, text, "docling", "docling", width, height)
    model = convert_row_to_document_model(
        row,
        layout_record={
            "layout": structured,
            "plain_text_reconstruction": text,
        },
    )
    model["pages"][0]["markdown_output"] = markdown
    return model


def from_marker(
    *,
    pdf_name: str,
    page: int,
    text: str,
    markdown: str,
    width: float | None,
    height: float | None,
) -> dict[str, Any]:
    row = _base_row(pdf_name, page, text, text, "marker", "marker", width, height)
    model = convert_row_to_document_model(row, layout_record={"layout": {}, "plain_text_reconstruction": text})
    model["pages"][0]["markdown_output"] = markdown
    return model


def from_surya(
    *,
    pdf_name: str,
    page: int,
    text: str,
    lines: list[dict[str, Any]],
    width: float | None,
    height: float | None,
) -> dict[str, Any]:
    regions = []
    for i, line in enumerate(lines):
        regions.append(
            {
                "region_id": f"p{page}_r{i+1}",
                "type": "text",
                "bbox": line.get("bbox"),
                "reading_order": i + 1,
                "confidence": line.get("confidence"),
                "ordering_source": "engine",
                "text": line.get("text", ""),
                "line_ordering": [],
            }
        )
    row = _base_row(pdf_name, page, text, text, "surya", "surya", width, height)
    return convert_row_to_document_model(row, layout_record={"layout": {"regions": regions}, "plain_text_reconstruction": text})
