#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from converters import ALL_CONVERTERS
from converters.base import ConversionContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized gold manifest records.")
    parser.add_argument(
        "--dataset-audit-csv",
        default="data/gold_registry/dataset_audit.csv",
        help="Path to dataset_audit.csv",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/gold_registry/gold_manifest.jsonl",
        help="Output manifest path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    audit_path = Path(args.dataset_audit_csv)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(audit_path.open("r", encoding="utf-8", newline="")))
    records = []

    # Placeholder orchestration: iterate every dataset through every converter.
    # In real ingestion runs, routing should be based on dataset format metadata.
    for row in rows:
        source_root = Path("data/raw") / row["dataset_id"]
        context = ConversionContext(
            dataset_id=row["dataset_id"],
            source_root=source_root,
            output_root=output_path.parent,
        )
        for converter in ALL_CONVERTERS:
            records.extend(converter.iter_records(context))

    # Keep output deterministic and preserve existing records if no converters are implemented.
    with output_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} converted records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
