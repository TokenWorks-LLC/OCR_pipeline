#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from production.document_model import build_document_models_from_csv, validate_document_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert OCR pipeline output to canonical DocumentModel")
    parser.add_argument("--ocr-csv", required=True, help="Path to client_page_text.csv")
    parser.add_argument("--layout-jsonl", default=None, help="Optional path to layout_regions.jsonl")
    parser.add_argument(
        "--output-jsonl",
        default="reports/document_model.jsonl",
        help="Output JSONL path (one DocumentModel per row)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON path with all documents in a list",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    models = build_document_models_from_csv(
        ocr_csv_path=args.ocr_csv,
        layout_jsonl_path=args.layout_jsonl,
    )

    all_errors: list[str] = []
    for i, model in enumerate(models):
        errs = validate_document_model(model)
        for err in errs:
            all_errors.append(f"row={i + 1}: {err}")

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for model in models:
            fh.write(json.dumps(model, ensure_ascii=False) + "\n")

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as fh:
            json.dump(models, fh, ensure_ascii=False, indent=2)

    if all_errors:
        print("Validation errors detected:")
        for err in all_errors[:100]:
            print(f"- {err}")
        return 2

    print(f"Wrote {len(models)} DocumentModel records to {output_jsonl}")
    if args.output_json:
        print(f"Wrote aggregate JSON to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
