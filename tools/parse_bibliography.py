#!/usr/bin/env python3
"""Parse bibliography / reference text into structured citation records.

This is the CLI front-end for ``production.postprocessing.citation_parser``. It
reads reference text from a plain-text / CSV / JSONL file, or straight from a
pipeline ``client_page_text.csv`` (optionally restricted to reference or
bibliography blocks), and writes structured citations to ``citations.jsonl`` and
``citations.csv``.

Examples:

    # Free-form file, one bibliography block per blank-line-separated chunk
    python tools/parse_bibliography.py --input refs.txt --output-dir reports/cites

    # Straight from pipeline output, only reference/bibliography blocks
    python tools/parse_bibliography.py \
        --page-text-csv reports/output/client_page_text.csv \
        --reference-only \
        --output-dir reports/cites
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.postprocessing.citation_parser import (
    ParsedCitation,
    parse_bibliography_block,
)


# Columns/keys we try, in order, when the caller does not name one explicitly.
TEXT_COLUMN_CANDIDATES = [
    "reference",
    "citation",
    "bibliography",
    "text",
    "cleaned_text",
    "corrected_text",
    "raw_text",
    "plain_text_reconstruction",
]
# Columns that mark a block/page as a reference or bibliography region.
ROLE_COLUMN_CANDIDATES = ["block_type", "role", "block_role", "region_role", "detected_role"]
REFERENCE_ROLE_VALUES = {"bibliography", "reference", "reference_meta", "references"}

CSV_COLUMNS = [
    "source_index",
    "authors_joined",
    "year",
    "title",
    "source",
    "volume",
    "issue",
    "pages",
    "doi",
    "url",
    "style",
    "overall_confidence",
    "needs_review",
    "unparsed",
    "raw_text",
]


def _pick_key(available: list[str], candidates: list[str], override: str | None) -> str | None:
    if override:
        return override
    lowered = {name.lower(): name for name in available}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _iter_text_blocks_from_txt(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Blank-line-separated chunks are treated as bibliography blocks.
    blocks = [blk.strip() for blk in text.split("\n\n")]
    return [blk for blk in blocks if blk]


def _iter_text_blocks_from_csv(path: Path, text_column: str | None) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        column = _pick_key(fieldnames, TEXT_COLUMN_CANDIDATES, text_column)
        if column is None:
            raise SystemExit(
                f"Could not find a text column in {path.name}. "
                f"Available: {fieldnames}. Use --text-column to name one."
            )
        return [str(row.get(column, "") or "").strip() for row in reader]


def _iter_text_blocks_from_jsonl(path: Path, text_column: str | None) -> list[str]:
    blocks: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if text_column and text_column in record:
                blocks.append(str(record.get(text_column, "") or "").strip())
                continue
            for candidate in TEXT_COLUMN_CANDIDATES:
                if candidate in record:
                    blocks.append(str(record.get(candidate, "") or "").strip())
                    break
    return blocks


def _iter_text_blocks_from_page_text(
    path: Path, text_column: str | None, reference_only: bool
) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        column = _pick_key(fieldnames, TEXT_COLUMN_CANDIDATES, text_column)
        if column is None:
            raise SystemExit(
                f"Could not find a text column in {path.name}. Available: {fieldnames}."
            )
        role_column = _pick_key(fieldnames, ROLE_COLUMN_CANDIDATES, None)
        blocks: list[str] = []
        for row in reader:
            if reference_only and role_column is not None:
                role_value = str(row.get(role_column, "") or "").strip().lower()
                if role_value not in REFERENCE_ROLE_VALUES:
                    continue
            blocks.append(str(row.get(column, "") or "").strip())
        if reference_only and role_column is None:
            print(
                "warning: --reference-only requested but no role/block_type column "
                "was found; parsing all rows.",
                file=sys.stderr,
            )
        return blocks


def collect_citations(blocks: list[str]) -> list[ParsedCitation]:
    citations: list[ParsedCitation] = []
    for block in blocks:
        if not block:
            continue
        citations.extend(parse_bibliography_block(block))
    return citations


def write_outputs(citations: list[ParsedCitation], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "citations.jsonl"
    csv_path = output_dir / "citations.csv"

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for index, citation in enumerate(citations):
            record = citation.to_dict()
            record["source_index"] = index
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for index, citation in enumerate(citations):
            record = citation.to_dict()
            record["source_index"] = index
            writer.writerow(record)

    return jsonl_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse bibliography/reference text into structured citations."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Path to a .txt, .csv, or .jsonl file of reference text.")
    source.add_argument("--page-text-csv", help="Path to a pipeline client_page_text.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory for citations.jsonl/.csv.")
    parser.add_argument("--text-column", default=None, help="Explicit text column/field name.")
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="With --page-text-csv, keep only reference/bibliography blocks.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.page_text_csv:
        path = Path(args.page_text_csv)
        if not path.exists():
            print(f"Input not found: {path}", file=sys.stderr)
            return 1
        blocks = _iter_text_blocks_from_page_text(path, args.text_column, args.reference_only)
    else:
        path = Path(args.input)
        if not path.exists():
            print(f"Input not found: {path}", file=sys.stderr)
            return 1
        suffix = path.suffix.lower()
        if suffix == ".csv":
            blocks = _iter_text_blocks_from_csv(path, args.text_column)
        elif suffix == ".jsonl":
            blocks = _iter_text_blocks_from_jsonl(path, args.text_column)
        else:
            blocks = _iter_text_blocks_from_txt(path)

    citations = collect_citations(blocks)
    jsonl_path, csv_path = write_outputs(citations, Path(args.output_dir))

    total = len(citations)
    needs_review = sum(1 for c in citations if c.needs_review)
    mean_conf = sum(c.overall_confidence for c in citations) / total if total else 0.0

    print(f"Parsed {total} citations from {len(blocks)} block(s).")
    print(f"  Mean confidence: {mean_conf:.3f}")
    print(f"  Needs review:    {needs_review} ({(needs_review / total * 100 if total else 0):.1f}%)")
    print(f"  JSONL: {jsonl_path}")
    print(f"  CSV:   {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
