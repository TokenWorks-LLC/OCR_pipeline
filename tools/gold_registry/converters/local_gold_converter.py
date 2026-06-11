from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

LOCAL_GOLD_SUPPORTED_FORMATS = {
    "PLAIN_TEXT",
    "TEXT",
    "TXT",
    "CSV_TEXT_ROWS",
    "TSV_TEXT_ROWS",
    "JSON_TEXT_RECORDS",
    "PAGE_LEVEL_JSON",
    "LINE_LEVEL_JSON",
    "LOCAL_GOLD_CSV",
    "LOCAL_GOLD_TSV",
    "LOCAL_GOLD_JSON",
    "LOCAL_GOLD_MANIFEST_TSV",
    "OUTPUT_CSV_GOLD",
}

_TEXT_COLUMNS = (
    "gold_text",
    "ground_truth_text",
    "text",
    "transcription",
    "content",
    "page_text",
    "label",
    "value",
)

_PAGE_ID_COLUMNS = (
    "page_id",
    "page_reference",
    "page_key",
)

_PDF_COLUMNS = (
    "pdf_name",
    "source_file",
    "source_pdf",
)


class LocalGoldConverter:
    """Deterministic converter for local gold text formats."""

    annotation_format = "LOCAL_GOLD"

    def extract_text(
        self,
        *,
        source_path: Path,
        annotation_format: str,
        page_id: str = "",
        pdf_name: str = "",
        page_number: int = 1,
    ) -> tuple[str, str]:
        return extract_local_gold_text(
            source_path=source_path,
            annotation_format=annotation_format,
            page_id=page_id,
            pdf_name=pdf_name,
            page_number=page_number,
        )


def canonical_lookup_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("&", " and ")
    text = text.replace("\\", "/")
    text = Path(text).stem
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def build_source_page_key(source_name: str, page_number: int) -> str:
    stem = Path(str(source_name or "").replace("\\", "/")).stem
    return canonical_lookup_key(f"{stem}_page_{int(page_number)}")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if not text:
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _extract_json_text(node: Any) -> list[str]:
    out: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = key.lower()
            if lowered in _TEXT_COLUMNS and isinstance(value, str):
                normalized = _normalize_whitespace(value)
                if normalized:
                    out.append(normalized)
            out.extend(_extract_json_text(value))
    elif isinstance(node, list):
        for item in node:
            out.extend(_extract_json_text(item))
    return out


def _row_text_value(row: dict[str, Any]) -> str:
    for key in _TEXT_COLUMNS:
        value = row.get(key)
        if value is None:
            continue
        normalized = _normalize_whitespace(str(value))
        if normalized:
            return normalized
    return ""


def _row_page_key(row: dict[str, Any]) -> str:
    for key in _PAGE_ID_COLUMNS:
        value = str(row.get(key, "")).strip()
        if value:
            return canonical_lookup_key(value)

    page_num = _safe_int(row.get("page"), 1)
    for key in _PDF_COLUMNS:
        value = str(row.get(key, "")).strip()
        if value:
            return build_source_page_key(value, page_num)

    return ""


def _looks_like_no_header_manifest(first_line: str) -> bool:
    parts = first_line.rstrip("\n").split("\t")
    if len(parts) < 3:
        return False
    return _safe_int(parts[1], -1) > 0


def _select_delimited_row(
    rows: list[dict[str, Any]],
    *,
    page_id: str,
    pdf_name: str,
    page_number: int,
) -> dict[str, Any] | None:
    wanted_page_key = canonical_lookup_key(page_id)
    wanted_pdf_key = canonical_lookup_key(pdf_name)
    wanted_source_key = build_source_page_key(pdf_name or page_id, page_number)

    scored: list[tuple[int, int, dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        row_key = _row_page_key(row)
        row_page = _safe_int(row.get("page"), 1)
        score = 0
        if wanted_page_key and row_key == wanted_page_key:
            score += 4
        if wanted_source_key and row_key == wanted_source_key:
            score += 3
        for key in _PDF_COLUMNS:
            candidate = canonical_lookup_key(str(row.get(key, "")))
            if wanted_pdf_key and candidate == wanted_pdf_key and row_page == page_number:
                score += 2
        if score > 0:
            scored.append((score, -idx, row))

    if not scored:
        return None

    scored.sort(reverse=True)
    return scored[0][2]


def _read_no_header_tsv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            rows.append(
                {
                    "source_file": parts[0].strip(),
                    "page": _safe_int(parts[1], 1),
                    "gold_text": parts[2].strip(),
                }
            )
    return rows


def _extract_from_delimited(
    path: Path,
    *,
    page_id: str,
    pdf_name: str,
    page_number: int,
    delimiter: str,
) -> tuple[str, str]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        first_line = fh.readline()

    if not first_line.strip():
        return "", "local_gold_empty_source"

    if delimiter == "\t" and _looks_like_no_header_manifest(first_line):
        rows = _read_no_header_tsv(path)
    else:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            rows = [dict(row) for row in reader]

    if not rows:
        return "", "local_gold_no_rows"

    selected = _select_delimited_row(
        rows,
        page_id=page_id,
        pdf_name=pdf_name,
        page_number=page_number,
    )

    if selected is None:
        return "", "local_gold_row_not_found"

    text = _row_text_value(selected)
    if not text:
        return "", "local_gold_row_missing_text"

    return text, ""


def _extract_from_json(
    path: Path,
    *,
    page_id: str,
    pdf_name: str,
    page_number: int,
) -> tuple[str, str]:
    try:
        if path.suffix.lower() == ".jsonl":
            payload: list[Any] = []
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        continue
                    payload.append(json.loads(line))
        else:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return "", "local_gold_malformed_json"

    records: list[dict[str, Any]] = []

    if isinstance(payload, dict):
        if isinstance(payload.get("records"), list):
            records = [dict(item) for item in payload["records"] if isinstance(item, dict)]
        elif isinstance(payload.get("pages"), list):
            records = [dict(item) for item in payload["pages"] if isinstance(item, dict)]
        else:
            text = _extract_json_text(payload)
            if text:
                return "\n".join(text), ""
            return "", "local_gold_row_missing_text"
    elif isinstance(payload, list):
        records = [dict(item) for item in payload if isinstance(item, dict)]
    else:
        return "", "local_gold_unsupported_json_shape"

    if not records:
        return "", "local_gold_no_rows"

    selected = _select_delimited_row(
        records,
        page_id=page_id,
        pdf_name=pdf_name,
        page_number=page_number,
    )
    if selected is None:
        return "", "local_gold_row_not_found"

    text = _row_text_value(selected)
    if text:
        return text, ""

    nested_text = _extract_json_text(selected)
    if nested_text:
        return "\n".join(nested_text), ""

    return "", "local_gold_row_missing_text"


def extract_local_gold_text(
    *,
    source_path: Path,
    annotation_format: str,
    page_id: str = "",
    pdf_name: str = "",
    page_number: int = 1,
) -> tuple[str, str]:
    if not source_path.exists():
        return "", "local_gold_source_missing"

    fmt = str(annotation_format or "").strip().upper()
    suffix = source_path.suffix.lower()

    if fmt in {"PLAIN_TEXT", "TEXT", "TXT"} and suffix not in {".csv", ".tsv", ".json", ".jsonl"}:
        text = _normalize_whitespace(source_path.read_text(encoding="utf-8", errors="ignore"))
        return (text, "") if text else ("", "local_gold_empty_source")

    if fmt in {"LOCAL_GOLD_MANIFEST_TSV", "LOCAL_GOLD_TSV", "TSV_TEXT_ROWS"} or suffix == ".tsv":
        return _extract_from_delimited(
            source_path,
            page_id=page_id,
            pdf_name=pdf_name,
            page_number=page_number,
            delimiter="\t",
        )

    if fmt in {"LOCAL_GOLD_CSV", "CSV_TEXT_ROWS", "OUTPUT_CSV_GOLD"} or suffix == ".csv":
        return _extract_from_delimited(
            source_path,
            page_id=page_id,
            pdf_name=pdf_name,
            page_number=page_number,
            delimiter=",",
        )

    if fmt in {"LOCAL_GOLD_JSON", "JSON_TEXT_RECORDS", "PAGE_LEVEL_JSON", "LINE_LEVEL_JSON"} or suffix in {
        ".json",
        ".jsonl",
    }:
        return _extract_from_json(
            source_path,
            page_id=page_id,
            pdf_name=pdf_name,
            page_number=page_number,
        )

    if suffix in {".txt", ".text"}:
        # Try TSV-style local manifest first, then plain text fallback.
        text, warning = _extract_from_delimited(
            source_path,
            page_id=page_id,
            pdf_name=pdf_name,
            page_number=page_number,
            delimiter="\t",
        )
        if text:
            return text, ""
        raw = _normalize_whitespace(source_path.read_text(encoding="utf-8", errors="ignore"))
        if raw:
            return raw, ""
        return "", warning or "local_gold_empty_source"

    return "", "unsupported_local_gold_format"
