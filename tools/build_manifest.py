#!/usr/bin/env python3
"""Build a page manifest TSV from a gold CSV.

Output format (one row per page):
    /abs/or/rel/path/to/file.pdf<TAB>page_number
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


PDF_CANDIDATE_COLUMNS = ["PDF LINK", "pdf", "pdf_name", "pdf_link", "file", "filename"]
PAGE_CANDIDATE_COLUMNS = ["PAGE", "page", "page_no", "page_number"]


def _pick_column(fieldnames: Sequence[str], candidates: list[str]) -> str | None:
    normalized = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        key = candidate.lower()
        if key in normalized:
            return normalized[key]
    return None


def _expand_page_spec(page_value: str, expand_ranges: bool) -> list[int]:
    raw = str(page_value).strip()
    if not raw:
        return []

    if "-" in raw and expand_ranges:
        left, right = raw.split("-", 1)
        try:
            start = int(left.strip())
            end = int(right.strip())
        except ValueError:
            return []
        if end < start:
            start, end = end, start
        return list(range(start, end + 1))

    try:
        return [int(raw)]
    except ValueError:
        return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create manifest TSV from gold CSV")
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--pdf-root", required=True, help="Directory containing PDFs")
    parser.add_argument("--out", required=True, help="Output manifest TSV path")
    parser.add_argument("--expand-ranges", action="store_true", help="Expand page ranges like 6-8")
    parser.add_argument("--strict", action="store_true", help="Fail if a referenced PDF is missing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    csv_path = Path(args.csv)
    pdf_root = Path(args.pdf_root)
    out_path = Path(args.out)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not pdf_root.exists():
        raise FileNotFoundError(f"PDF root not found: {pdf_root}")

    rows_written = 0
    missing_files: list[str] = []

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as in_fh, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as out_fh:
        reader = csv.DictReader(in_fh)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header")

        pdf_col = _pick_column(reader.fieldnames, PDF_CANDIDATE_COLUMNS)
        page_col = _pick_column(reader.fieldnames, PAGE_CANDIDATE_COLUMNS)
        if not pdf_col or not page_col:
            raise ValueError(
                f"Could not locate PDF/Page columns in header: {reader.fieldnames}"
            )

        for row in reader:
            pdf_name = str(row.get(pdf_col, "")).strip()
            page_value = str(row.get(page_col, "")).strip()
            if not pdf_name or not page_value:
                continue

            pdf_path = (pdf_root / pdf_name).resolve()
            if not pdf_path.exists():
                missing_files.append(pdf_name)
                if args.strict:
                    continue

            pages = _expand_page_spec(page_value, expand_ranges=args.expand_ranges)
            for page in pages:
                out_fh.write(f"{pdf_path}\t{page}\n")
                rows_written += 1

    print(f"Manifest written: {out_path}")
    print(f"Rows written: {rows_written}")
    if missing_files:
        unique_missing = sorted(set(missing_files))
        print(f"Missing PDFs referenced in CSV: {len(unique_missing)}")
        for name in unique_missing[:10]:
            print(f"- {name}")
        if len(unique_missing) > 10:
            print(f"... and {len(unique_missing) - 10} more")
        if args.strict:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
