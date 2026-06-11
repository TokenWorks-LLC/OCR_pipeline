from __future__ import annotations

import csv
import json
from pathlib import Path

from production.document_model import (
    build_document_models_from_csv,
    convert_row_to_document_model,
    validate_document_model,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_document_model_schema_file_exists_and_has_required_fields():
    schema_path = ROOT / "schemas" / "document_model.schema.json"
    assert schema_path.exists()

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    required = set(schema.get("required", []))
    assert {"schema_version", "document_id", "pages", "provenance"}.issubset(required)


def test_convert_row_to_document_model_preserves_raw_and_structured_text():
    row = {
        "pdf_name": "sample_receipt.pdf",
        "page": "1",
        "document_id": "sample_receipt",
        "page_id": "sample_receipt_page_1",
        "page_text": "key: value",
        "raw_text": "raw payload",
        "engine_used": "ensemble",
        "extraction_method": "ocr_ensemble",
        "width": "1200",
        "height": "1600",
        "engine_statuses": '{"paddle":{"status":"available"}}',
        "reading_order_source": "layout",
        "final_output_source": "region_ocr",
    }

    layout_record = {
        "layout": {
            "regions": [
                {
                    "region_id": "r1",
                    "type": "table",
                    "bbox": [10, 10, 500, 300],
                    "reading_order": 1,
                    "confidence": 0.92,
                    "ordering_source": "engine",
                    "text": "item: bread\nprice: 4.00",
                    "line_ordering": [
                        {
                            "text": "item: bread",
                            "bbox": [10, 10, 500, 100],
                            "reading_order": 1,
                            "confidence": 0.93,
                            "ordering_source": "engine",
                        }
                    ],
                }
            ]
        },
        "plain_text_reconstruction": "item: bread\nprice: 4.00",
    }

    model = convert_row_to_document_model(row, layout_record=layout_record)
    errors = validate_document_model(model)
    assert not errors

    page = model["pages"][0]
    assert page["raw_ocr_text"] == "raw payload"
    assert page["reading_order_text"] == "item: bread\nprice: 4.00"
    assert page["structured_json_output"]["regions"][0]["type"] == "table"
    assert page["tables"]
    assert page["key_value_pairs"]


def test_build_document_models_from_csv_with_layout_jsonl(tmp_path: Path):
    ocr_csv = tmp_path / "client_page_text.csv"
    layout_jsonl = tmp_path / "layout_regions.jsonl"

    _write_csv(
        ocr_csv,
        [
            "pdf_name",
            "page",
            "page_text",
            "raw_text",
            "document_id",
            "page_id",
            "engine_used",
            "extraction_method",
            "engine_statuses",
        ],
        [
            {
                "pdf_name": "book.pdf",
                "page": 1,
                "page_text": "Line one",
                "raw_text": "RAW Line one",
                "document_id": "book",
                "page_id": "book_page_1",
                "engine_used": "text_layer",
                "extraction_method": "text_layer",
                "engine_statuses": "{}",
            }
        ],
    )

    layout_record = {
        "pdf_name": "book.pdf",
        "page": 1,
        "layout": {
            "regions": [
                {
                    "region_id": "book_r1",
                    "type": "text",
                    "bbox": [0, 0, 100, 20],
                    "reading_order": 1,
                    "confidence": 0.9,
                    "text": "Line one",
                    "line_ordering": [],
                }
            ]
        },
        "plain_text_reconstruction": "Line one",
    }
    with layout_jsonl.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(layout_record) + "\n")

    models = build_document_models_from_csv(ocr_csv, layout_jsonl)
    assert len(models) == 1
    assert models[0]["document_id"] == "book"
    assert models[0]["pages"][0]["blocks"][0]["block_id"] == "book_r1"
    assert not validate_document_model(models[0])
