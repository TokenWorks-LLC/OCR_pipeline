#!/usr/bin/env python3
"""Multilingual OCR gold-set evaluation harness.

The evaluator is language-agnostic by default and supports optional
transliteration/Akkadian add-on metrics when explicitly enabled or when
page metadata marks a page as transliteration-heavy.

Outputs written to --output-dir:
  - evaluation_summary.csv
  - per_page_metrics.csv
  - per_engine_metrics.csv
  - per_language_metrics.csv
  - per_layout_metrics.csv
    - per_detected_orientation_metrics.csv
    - per_detected_layout_metrics.csv
  - confusion_matrix.json
  - run_metadata.json

Usage:
    python tools/evaluate_gold.py \
        --ocr-csv reports/eval_gold/client_page_text.csv \
        --gold-csv data/gold_data/gold_pages.csv \
        --progress-csv reports/eval_gold/progress.csv \
        --output-dir reports/eval_gold_v2
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from production.key_normalization import build_key_provenance


UNICODE_FORMS = ("NFC", "NFD", "NFKC", "NFKD")

PAGE_STATUS_TIMEOUT = "timed_out"
PAGE_STATUS_FAILED = "failed"

ENGINE_STATUS_AVAILABLE = "available"
ENGINE_STATUS_AVAILABLE_UNHEALTHY = "available_but_unhealthy"

CONFUSION_ALIGNMENT_MAX_CELLS = 2_000_000


GENERAL_METADATA_FIELDS = [
    "dataset_id",
    "language_primary",
    "languages_present",
    "script_type",
    "document_type",
    "layout_type",
    "has_tables",
    "has_footnotes",
    "has_columns",
    "has_diacritics",
    "has_transliteration",
    "scan_quality",
    "expected_difficulty",
    "expected_special_handling",
]

RUNTIME_DETECTION_FIELDS = [
    "detected_orientation_angle",
    "detected_orientation_class",
    "detected_rotation_base_angle",
    "detected_skew_angle",
    "detected_layout_type",
    "detected_column_count",
    "detected_has_columns",
]


AKKADIAN_DIACRITIC_CHARS = set("\u0161\u1e63\u1e6d\u1e2b\u0101\u0113\u012b\u016b\u0160\u1e62\u1e6c\u1e2a\u0100\u0112\u012a\u016a")
MACRON_CHARS = set("\u0101\u0113\u012b\u016b\u0100\u0112\u012a\u016a")
SUBSUPER_CHARS = set(
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
    "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
    "\u1d48\u1d50\u1da0\u02be"
)
BRACKET_MARKERS = set("[](){}<>\u27e6\u27e7\u2e22\u2e23")
HYPHEN_RE = re.compile(r"\b[^\W_]+(?:[-\u2010-\u2015][^\W_]+)+\b", flags=re.UNICODE)
UNKNOWN_TRANSLIT_RE = re.compile(r"\b(?:x+|x\.x|x-x|\[x\]|x\?)\b", flags=re.IGNORECASE)


@dataclass
class NormalizationConfig:
    unicode_form: str = "NFKC"
    whitespace_mode: str = "collapse"  # preserve | collapse
    strip_punctuation: bool = False
    casefold: bool = False


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
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
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t", "on"}


def _canonical_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_page_key(key: str) -> str:
    provenance = build_key_provenance(
        page_key=key,
        document_key="",
        page=None,
        page_reference="",
        unicode_form="NFKC",
    )
    return provenance.normalized_page_key


def _derive_page_from_pdf_name(pdf_name: str) -> int | None:
    match = re.search(r"_page_(\d+)(?:\D|$)", pdf_name.lower())
    if match:
        return int(match.group(1))
    return None


def _resolve_page_number(page_value: Any, pdf_name: str) -> int | None:
    """Resolve a robust page number from CSV field and filename hints.

    Some historical outputs encode page in the filename (e.g. *_page_42.pdf)
    while `page` is always 1 because each row is a single-page rendering. For
    those cases we prefer the filename-derived page number.
    """
    explicit = _safe_int(page_value)
    derived = _derive_page_from_pdf_name(pdf_name) if pdf_name else None

    if explicit is None:
        return derived
    if derived is None:
        return explicit
    if explicit == derived:
        return explicit

    if explicit == 1 and "_page_" in Path(pdf_name).stem.lower():
        return derived
    return explicit


def _build_page_key(pdf_name: str, page: int | None, page_reference: str = "") -> str:
    provenance = build_key_provenance(
        page_key="",
        document_key=pdf_name,
        page=page,
        page_reference=page_reference,
        unicode_form="NFKC",
    )
    return provenance.normalized_page_key


def _lower_key_map(row: dict[str, Any]) -> dict[str, Any]:
    return {str(k).strip().lower(): v for k, v in row.items()}


def _row_get(row: dict[str, Any], *candidates: str, default: Any = "") -> Any:
    lower = _lower_key_map(row)
    for key in candidates:
        value = lower.get(key.lower())
        if value is not None and str(value).strip() != "":
            return value
    return default


def _parse_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in re.split(r"[;,|/]", text) if part.strip()]


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
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def _normalize_for_metrics(text: str, cfg: NormalizationConfig) -> str:
    normalized = unicodedata.normalize(cfg.unicode_form, _canonical_text(text))
    if cfg.casefold:
        normalized = normalized.casefold()
    if cfg.strip_punctuation:
        normalized = "".join(ch for ch in normalized if not unicodedata.category(ch).startswith("P"))

    if cfg.whitespace_mode == "collapse":
        normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            ins = dp[j] + 1
            delete = dp[j - 1] + 1
            replace = prev + cost
            prev = dp[j]
            dp[j] = min(ins, delete, replace)
    return dp[m]


def _char_error_rate(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(list(reference), list(hypothesis)) / len(reference)


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def _line_order_similarity(reference: str, hypothesis: str) -> float:
    ref_lines = [line.strip() for line in reference.splitlines() if line.strip()]
    hyp_lines = [line.strip() for line in hypothesis.splitlines() if line.strip()]
    if not ref_lines:
        return 1.0 if not hyp_lines else 0.0
    if not hyp_lines:
        return 0.0
    return max(0.0, 1.0 - (_edit_distance(ref_lines, hyp_lines) / len(ref_lines)))


def _paragraph_order_similarity(reference: str, hypothesis: str) -> float:
    ref_paras = [p.strip() for p in re.split(r"\n\s*\n", reference) if p.strip()]
    hyp_paras = [p.strip() for p in re.split(r"\n\s*\n", hypothesis) if p.strip()]
    if not ref_paras:
        return 1.0 if not hyp_paras else 0.0
    if not hyp_paras:
        return 0.0
    return max(0.0, 1.0 - (_edit_distance(ref_paras, hyp_paras) / len(ref_paras)))


def _header_footer_mismatch(reference: str, hypothesis: str) -> bool:
    ref_lines = [line.strip() for line in reference.splitlines() if line.strip()]
    hyp_lines = [line.strip() for line in hypothesis.splitlines() if line.strip()]
    if len(ref_lines) < 2 or len(hyp_lines) < 2:
        return False
    ref_edges = " ".join((ref_lines[0], ref_lines[-1]))
    hyp_edges = " ".join((hyp_lines[0], hyp_lines[-1]))
    overlap = _token_jaccard(ref_edges, hyp_edges)
    return overlap < 0.2


def _token_jaccard(a: str, b: str) -> float:
    at = set(re.findall(r"\w+", a.lower(), flags=re.UNICODE))
    bt = set(re.findall(r"\w+", b.lower(), flags=re.UNICODE))
    if not at and not bt:
        return 1.0
    if not at or not bt:
        return 0.0
    return len(at & bt) / len(at | bt)


def _extract_key_value_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            left, right = line.split(":", 1)
            k = left.strip().lower()
            v = right.strip().lower()
            if k and v:
                pairs.append((k, v))
    return pairs


def _set_f1(reference_items: set[tuple[str, str]], hypothesis_items: set[tuple[str, str]]) -> float | None:
    if not reference_items and not hypothesis_items:
        return None
    if not reference_items or not hypothesis_items:
        return 0.0
    tp = len(reference_items & hypothesis_items)
    precision = tp / len(hypothesis_items)
    recall = tp / len(reference_items)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _extract_table_like_rows(text: str) -> set[str]:
    rows: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.search(r"\d", line) and re.search(r"[\.:,]", line):
            rows.add(re.sub(r"\s+", " ", line.lower()))
    return rows


def _structured_json_similarity(reference: str, hypothesis: str) -> float | None:
    ref_pairs = set(_extract_key_value_pairs(reference))
    hyp_pairs = set(_extract_key_value_pairs(hypothesis))
    kv_f1 = _set_f1(ref_pairs, hyp_pairs)
    ref_rows = _extract_table_like_rows(reference)
    hyp_rows = _extract_table_like_rows(hypothesis)
    row_f1 = _set_f1(set((r, "") for r in ref_rows), set((r, "") for r in hyp_rows))
    if kv_f1 is None and row_f1 is None:
        return None
    vals = [x for x in [kv_f1, row_f1] if x is not None]
    return sum(vals) / len(vals) if vals else None


def _quantile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _safe_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = _safe_mean(values)
    if avg is None:
        return None
    variance = sum((v - avg) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _counter_preservation_rate(reference_items: list[str], hypothesis_items: list[str]) -> float | None:
    if not reference_items:
        return None
    reference_counter = Counter(reference_items)
    hypothesis_counter = Counter(hypothesis_items)
    kept = 0
    for item, count in reference_counter.items():
        kept += min(count, hypothesis_counter.get(item, 0))
    return kept / sum(reference_counter.values())


def _extract_diacritic_chars(text: str) -> list[str]:
    items: list[str] = []
    for char in text:
        if not char.strip():
            continue
        decomposed = unicodedata.normalize("NFD", char)
        if any(unicodedata.combining(token) for token in decomposed[1:]):
            items.append(char)
    return items


def _extract_punctuation_chars(text: str) -> list[str]:
    return [char for char in text if unicodedata.category(char).startswith("P")]


def _extract_whitespace_runs(text: str) -> list[str]:
    return re.findall(r"\s+", text)


def _is_rtl_char(char: str) -> bool:
    return unicodedata.bidirectional(char) in {"R", "AL", "AN"}


def _script_bucket(char: str) -> str:
    if not char.strip() or not char.isprintable():
        return "other"
    name = unicodedata.name(char, "")
    if "ARABIC" in name:
        return "arabic"
    if "LATIN" in name:
        return "latin"
    if "CYRILLIC" in name:
        return "cyrillic"
    if "GREEK" in name:
        return "greek"
    if "HEBREW" in name:
        return "hebrew"
    if "CJK" in name or "HIRAGANA" in name or "KATAKANA" in name or "HANGUL" in name:
        return "east_asian"
    return "other"


def _parse_expected_scripts(script_type: str) -> set[str]:
    scripts = {token.strip().lower() for token in _parse_list_field(script_type)}
    if not scripts:
        return set()

    mapped: set[str] = set()
    aliases = {
        "latin": "latin",
        "arabic": "arabic",
        "rtl": "arabic",
        "hebrew": "hebrew",
        "greek": "greek",
        "cyrillic": "cyrillic",
        "transliteration": "latin",
        "mixed": "other",
    }
    for item in scripts:
        mapped.add(aliases.get(item, item))
    return mapped


def _script_character_coverage(reference: str, hypothesis: str, expected_scripts: set[str]) -> float | None:
    ref_chars = [char for char in reference if char.strip()]
    hyp_chars = set(char for char in hypothesis if char.strip())
    if expected_scripts:
        ref_chars = [char for char in ref_chars if _script_bucket(char) in expected_scripts]

    if not ref_chars:
        return None

    ref_unique = set(ref_chars)
    if not ref_unique:
        return None
    covered = sum(1 for char in ref_unique if char in hyp_chars)
    return covered / len(ref_unique)


def _should_run_akkadian_metrics(meta: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.enable_akkadian_metrics:
        return True
    if args.disable_metadata_akkadian:
        return False

    language_primary = str(meta.get("language_primary", "")).lower()
    languages_present = [token.lower() for token in _parse_list_field(meta.get("languages_present", ""))]
    special = str(meta.get("expected_special_handling", "")).lower()

    if "akkadian" in language_primary:
        return True
    if any("akkadian" in token for token in languages_present):
        return True
    if _safe_bool(meta.get("has_transliteration")):
        return True
    return "translit" in special or "akkadian" in special


def _safe_rate_from_charset(reference: str, hypothesis: str, charset: set[str]) -> float | None:
    reference_items = [char for char in reference if char in charset]
    hypothesis_items = [char for char in hypothesis if char in charset]
    return _counter_preservation_rate(reference_items, hypothesis_items)


def _safe_rate_from_regex_tokens(reference: str, hypothesis: str, pattern: re.Pattern[str]) -> float | None:
    reference_items = pattern.findall(reference)
    hypothesis_items = pattern.findall(hypothesis)
    return _counter_preservation_rate(reference_items, hypothesis_items)


def _akkadian_optional_metrics(reference: str, hypothesis: str) -> dict[str, float | None]:
    hypothesis_tokens = hypothesis.split()
    unknown_tokens = [token for token in hypothesis_tokens if UNKNOWN_TRANSLIT_RE.search(token)]

    bracket_reference = [char for char in reference if char in BRACKET_MARKERS]
    bracket_hypothesis = [char for char in hypothesis if char in BRACKET_MARKERS]

    return {
        "akkadian_special_char_preservation_rate": _safe_rate_from_charset(reference, hypothesis, AKKADIAN_DIACRITIC_CHARS),
        "akkadian_macron_preservation_rate": _safe_rate_from_charset(reference, hypothesis, MACRON_CHARS),
        "akkadian_subsuper_preservation_rate": _safe_rate_from_charset(reference, hypothesis, SUBSUPER_CHARS),
        "akkadian_hyphenated_sign_preservation_rate": _safe_rate_from_regex_tokens(reference, hypothesis, HYPHEN_RE),
        "akkadian_bracket_marker_preservation_rate": _counter_preservation_rate(bracket_reference, bracket_hypothesis),
        "akkadian_unknown_transliteration_token_rate": (
            len(unknown_tokens) / len(hypothesis_tokens) if hypothesis_tokens else None
        ),
    }


def _align_chars_for_confusion(reference: str, hypothesis: str) -> list[tuple[str | None, str | None, str]]:
    """Align with edit operations for confusion accounting.

    Returns (reference_char, hypothesis_char, operation), where operation is
    one of: equal, replace, delete, insert.
    """
    n, m = len(reference), len(hypothesis)
    if n * m > CONFUSION_ALIGNMENT_MAX_CELLS:
        # Fallback for very long pages: approximate with zip + tails.
        aligned: list[tuple[str | None, str | None, str]] = []
        for r_char, h_char in zip(reference, hypothesis):
            op = "equal" if r_char == h_char else "replace"
            aligned.append((r_char, h_char, op))
        if len(reference) > len(hypothesis):
            for r_char in reference[len(hypothesis) :]:
                aligned.append((r_char, None, "delete"))
        elif len(hypothesis) > len(reference):
            for h_char in hypothesis[len(reference) :]:
                aligned.append((None, h_char, "insert"))
        return aligned

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # delete
                dp[i][j - 1] + 1,      # insert
                dp[i - 1][j - 1] + cost,  # replace/equal
            )

    out: list[tuple[str | None, str | None, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                op = "equal" if cost == 0 else "replace"
                out.append((reference[i - 1], hypothesis[j - 1], op))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            out.append((reference[i - 1], None, "delete"))
            i -= 1
            continue
        out.append((None, hypothesis[j - 1], "insert"))
        j -= 1

    out.reverse()
    return out


def _char_label(char: str) -> str:
    if char == " ":
        return "<space>"
    if char == "\n":
        return "<newline>"
    if char == "\t":
        return "<tab>"
    if not char.isprintable():
        return f"U+{ord(char):04X}"
    return char


def _load_gold_records(gold_csv: str) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    malformed_rows: list[dict[str, Any]] = []

    with open(gold_csv, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for line_number, row in enumerate(reader, start=2):
            ground_truth = str(
                _row_get(
                    row,
                    "ground_truth_text",
                    "ground_truth",
                    "gold_text",
                    "reference_text",
                    "handtyped",
                    default="",
                )
            ).strip()
            if not ground_truth:
                continue

            pdf_name = str(
                _row_get(row, "pdf_name", "pdf", "pdf link", "page_image", "page_file", default="")
            ).strip()
            page_value = _row_get(row, "page", "page_no", "page_number", "page index", default="")
            page_reference = str(_row_get(row, "page_reference", "page_ref", "page_id", default="")).strip()
            explicit_page_key = str(_row_get(row, "page_key", "id", default="")).strip()

            page = _resolve_page_number(page_value, pdf_name)
            if page is None:
                page = _safe_int(_row_get(row, "page_number", default=""))

            if not pdf_name and not explicit_page_key and not page_reference:
                malformed_rows.append(
                    {
                        "source": "gold",
                        "line_number": line_number,
                        "page_key": "",
                        "pdf_name": "",
                        "page": page_value,
                        "issue": "missing_identifier",
                        "detail": "row has text but no pdf_name/page_key/page_reference",
                    }
                )
                continue

            if page is None and not explicit_page_key and not page_reference:
                malformed_rows.append(
                    {
                        "source": "gold",
                        "line_number": line_number,
                        "page_key": "",
                        "pdf_name": pdf_name,
                        "page": page_value,
                        "issue": "missing_page_number",
                        "detail": "unable to resolve numeric page",
                    }
                )
                continue

            metadata: dict[str, Any] = {}
            for field in GENERAL_METADATA_FIELDS:
                metadata[field] = _row_get(row, field, default="")

            if not metadata.get("language_primary"):
                metadata["language_primary"] = "unknown"
            if not metadata.get("languages_present"):
                metadata["languages_present"] = metadata.get("language_primary", "unknown")
            if not metadata.get("script_type"):
                metadata["script_type"] = "unknown"
            if not metadata.get("layout_type"):
                metadata["layout_type"] = "unknown"

            key_provenance = build_key_provenance(
                page_key=explicit_page_key,
                document_key=pdf_name,
                page=page,
                page_reference=page_reference,
                unicode_form="NFKC",
            )
            page_key = key_provenance.normalized_page_key

            if not page_key or page_key.startswith("unknown"):
                malformed_rows.append(
                    {
                        "source": "gold",
                        "line_number": line_number,
                        "page_key": page_key,
                        "pdf_name": pdf_name,
                        "page": page if page is not None else page_value,
                        "issue": "invalid_page_key",
                        "detail": "resolved page_key is unknown/empty",
                    }
                )
                continue

            if page_key in records:
                malformed_rows.append(
                    {
                        "source": "gold",
                        "line_number": line_number,
                        "page_key": page_key,
                        "pdf_name": pdf_name,
                        "page": page if page is not None else page_value,
                        "issue": "duplicate_page_key",
                        "detail": "duplicate page_key in gold set",
                    }
                )
                continue

            resolved_pdf_name = (
                Path(pdf_name).name
                if pdf_name
                else (Path(key_provenance.original_document_key).name if key_provenance.original_document_key else "")
            )

            records[page_key] = {
                "page_key": page_key,
                "pdf_name": resolved_pdf_name,
                "page": page,
                "page_reference": page_reference,
                "ground_truth_text": _canonical_text(ground_truth),
                "original_page_key": key_provenance.original_page_key,
                "normalized_page_key": key_provenance.normalized_page_key,
                "original_document_key": key_provenance.original_document_key,
                "normalized_document_key": key_provenance.normalized_document_key,
                "key_normalization_applied": key_provenance.key_normalization_applied,
                "key_normalization_warnings": key_provenance.key_normalization_warnings,
                **metadata,
            }

    return records, malformed_rows


def _parse_engine_statuses(value: Any) -> dict[str, dict[str, Any]]:
    parsed = _parse_jsonish(value)
    if isinstance(parsed, dict):
        statuses: dict[str, dict[str, Any]] = {}
        for name, info in parsed.items():
            if isinstance(info, dict):
                statuses[str(name)] = {
                    "status": str(info.get("status", "")).strip(),
                    "reason": str(info.get("reason", "")).strip(),
                }
            else:
                statuses[str(name)] = {"status": str(info).strip(), "reason": ""}
        return statuses
    return {}


def _parse_engine_runtime_map(value: Any) -> dict[str, float]:
    parsed = _parse_jsonish(value)
    if not isinstance(parsed, dict):
        return {}

    output: dict[str, float] = {}
    for name, runtime in parsed.items():
        as_float = _safe_float(runtime)
        if as_float is not None:
            output[str(name)] = as_float
    return output


def _extract_runtime_ms(row: dict[str, Any], source_prefix: str = "") -> float | None:
    candidates = [
        "runtime_ms",
        "ms",
        "elapsed_ms",
        "time_ms",
        "runtime",
        "time",
    ]
    for candidate in candidates:
        value = _row_get(row, f"{source_prefix}{candidate}" if source_prefix else candidate, default="")
        runtime = _safe_float(value)
        if runtime is None:
            continue

        # Heuristic: plain `time` and `runtime` fields are usually seconds.
        if candidate in {"runtime", "time"} and runtime < 10_000:
            return runtime * 1000.0
        return runtime
    return None


def _decode_csv_text_field(text: str) -> str:
    # The pipeline escapes newlines as literal `\n` in CSV.
    return _canonical_text(text.replace("\\n", "\n"))


def _normalize_angle_degrees(angle: float) -> float:
    normalized = float(angle) % 360.0
    return normalized if normalized >= 0 else normalized + 360.0


def _signed_angle_delta(angle: float, reference: float) -> float:
    return ((float(angle) - float(reference) + 180.0) % 360.0) - 180.0


def _derive_orientation_class(angle: float | None) -> str:
    if angle is None:
        return "unknown"

    normalized = _normalize_angle_degrees(angle)
    right_angles = (0, 90, 180, 270)
    nearest = min(right_angles, key=lambda item: abs(_signed_angle_delta(normalized, item)))
    skew = _signed_angle_delta(normalized, nearest)

    labels = {
        0: "upright",
        90: "rotated_90_cw",
        180: "upside_down",
        270: "rotated_90_ccw",
    }
    label = labels.get(nearest, "unknown")
    if abs(skew) >= 1.0:
        label = f"{label}_skewed"
    return label


def _safe_bool_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return "true" if _safe_bool(value) else "false"


def _extract_runtime_detection_fields(row: dict[str, Any]) -> dict[str, Any]:
    orientation_angle = _safe_float(
        _row_get(row, "detected_orientation_angle", "rotation_angle", "orientation_angle", default="")
    )
    rotation_base_angle = _safe_float(
        _row_get(row, "detected_rotation_base_angle", "rotation_base_angle", default="")
    )
    skew_angle = _safe_float(_row_get(row, "detected_skew_angle", "deskew_angle", default=""))
    orientation_class = str(
        _row_get(row, "detected_orientation_class", "orientation_class", default="")
    ).strip()
    if not orientation_class:
        orientation_class = _derive_orientation_class(orientation_angle)

    detected_layout_type = str(_row_get(row, "detected_layout_type", default="")).strip()
    detected_column_count = _safe_int(_row_get(row, "detected_column_count", default=""))
    detected_has_columns = _safe_bool_text(_row_get(row, "detected_has_columns", default=""))

    if not detected_has_columns and detected_column_count is not None:
        detected_has_columns = "true" if detected_column_count > 1 else "false"

    if not detected_layout_type:
        if detected_has_columns == "true":
            detected_layout_type = "multi_column"
        elif detected_has_columns == "false":
            detected_layout_type = "single_column"
        else:
            detected_layout_type = "unknown"

    return {
        "detected_orientation_angle": orientation_angle,
        "detected_orientation_class": orientation_class or "unknown",
        "detected_rotation_base_angle": rotation_base_angle,
        "detected_skew_angle": skew_angle,
        "detected_layout_type": detected_layout_type,
        "detected_column_count": detected_column_count,
        "detected_has_columns": detected_has_columns,
    }


def _load_ocr_records(ocr_csv: str) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    malformed_rows: list[dict[str, Any]] = []

    with open(ocr_csv, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for line_number, row in enumerate(reader, start=2):
            input_file = str(
                _row_get(row, "input_file", "input_pdf_path", "pdf_path", "source_file", default="")
            ).strip()
            pdf_name = str(_row_get(row, "pdf_name", "pdf", "pdf link", default="")).strip()
            source_pdf = pdf_name or input_file

            page = _resolve_page_number(_row_get(row, "page", "page_no", "page_number", default=""), source_pdf)
            if page is None:
                page = _safe_int(_row_get(row, "page_number", default=""))

            page_reference = str(_row_get(row, "page_reference", "page_ref", default="")).strip()
            explicit_key = str(_row_get(row, "page_key", "id", default="")).strip()
            key_provenance = build_key_provenance(
                page_key=explicit_key,
                document_key=source_pdf,
                page=page,
                page_reference=page_reference,
                unicode_form="NFKC",
            )
            page_key = key_provenance.normalized_page_key

            if not source_pdf:
                malformed_rows.append(
                    {
                        "source": "ocr",
                        "line_number": line_number,
                        "page_key": page_key,
                        "pdf_name": "",
                        "page": _row_get(row, "page", default=""),
                        "issue": "missing_source_pdf",
                        "detail": "row has no pdf_name/input_file",
                    }
                )
                continue

            if page is None:
                malformed_rows.append(
                    {
                        "source": "ocr",
                        "line_number": line_number,
                        "page_key": page_key,
                        "pdf_name": source_pdf,
                        "page": _row_get(row, "page", default=""),
                        "issue": "missing_page_number",
                        "detail": "unable to resolve numeric page",
                    }
                )
                continue

            if not page_key or page_key.startswith("unknown"):
                malformed_rows.append(
                    {
                        "source": "ocr",
                        "line_number": line_number,
                        "page_key": page_key,
                        "pdf_name": source_pdf,
                        "page": page,
                        "issue": "invalid_page_key",
                        "detail": "resolved page_key is unknown/empty",
                    }
                )
                continue

            if page_key in records:
                malformed_rows.append(
                    {
                        "source": "ocr",
                        "line_number": line_number,
                        "page_key": page_key,
                        "pdf_name": source_pdf,
                        "page": page,
                        "issue": "duplicate_page_key",
                        "detail": "duplicate page_key in OCR output",
                    }
                )
                continue

            page_text = str(_row_get(row, "page_text", "ocr_text", "text", "predicted_text", default=""))
            page_text = _decode_csv_text_field(page_text)

            status_raw = str(_row_get(row, "status", default="")).strip()
            status = status_raw.lower()
            failure_reason = str(_row_get(row, "failure_reason", default="")).strip()
            extraction_method = str(_row_get(row, "extraction_method", "method", "engine_used", default="")).strip()
            engine_statuses = _parse_engine_statuses(_row_get(row, "engine_statuses", default=""))
            engine_runtime_map = _parse_engine_runtime_map(
                _row_get(row, "engine_runtimes_ms", "engine_runtime_ms", "runtime_by_engine", default="")
            )
            runtime_detection = _extract_runtime_detection_fields(row)
            confidence = _safe_float(_row_get(row, "confidence", "mean_confidence", default=""))
            runtime_ms = _extract_runtime_ms(row)

            page_quality_score = _safe_float(_row_get(row, "page_quality_score", default=""))
            document_quality_score = _safe_float(_row_get(row, "document_quality_score", default=""))
            quality_class = str(_row_get(row, "quality_class", default="")).strip()
            quality_reasons = str(_row_get(row, "quality_reasons", default="")).strip()
            failed_gate = str(_row_get(row, "failed_gate", default="")).strip()
            gate_reason = str(_row_get(row, "gate_reason", default="")).strip()
            adapter_used = str(_row_get(row, "adapter_used", default="")).strip()
            recommended_preprocessing_profile = str(
                _row_get(row, "recommended_preprocessing_profile", default="")
            ).strip()
            applied_preprocessing_profile = str(
                _row_get(row, "applied_preprocessing_profile", default="")
            ).strip()
            postprocess_quality_score = _safe_float(_row_get(row, "postprocess_quality_score", default=""))
            lexicon_coverage = _safe_float(_row_get(row, "lexicon_coverage", default=""))
            unknown_token_rate = _safe_float(_row_get(row, "unknown_token_rate", default=""))
            protected_character_changes = _safe_int(_row_get(row, "protected_character_changes", default=""))

            output_text_length = _safe_int(_row_get(row, "output_text_length", "ocr_text_length", default=""))
            if output_text_length is None:
                output_text_length = len(page_text)

            document_id = str(_row_get(row, "document_id", default="")).strip()
            if not document_id:
                document_id = key_provenance.normalized_document_key or _normalize_page_key(Path(source_pdf).stem)
            page_id = str(_row_get(row, "page_id", default="")).strip()
            if not page_id and document_id:
                page_id = f"{document_id}_page_{page}"

            engine_used = str(_row_get(row, "engine_used", default="")).strip() or extraction_method

            malformed_detail: list[str] = []
            if not status_raw:
                malformed_detail.append("missing_status")
            if not extraction_method and not engine_statuses:
                malformed_detail.append("missing_engine_used")
            if malformed_detail:
                malformed_rows.append(
                    {
                        "source": "ocr",
                        "line_number": line_number,
                        "page_key": page_key,
                        "pdf_name": source_pdf,
                        "page": page,
                        "issue": "missing_required_fields",
                        "detail": "|".join(malformed_detail),
                    }
                )
                continue

            if not status:
                if not page_text.strip():
                    status = PAGE_STATUS_FAILED
                elif "timeout" in failure_reason.lower():
                    status = PAGE_STATUS_TIMEOUT
                else:
                    status = "success"

            records[page_key] = {
                "page_key": page_key,
                "pdf_name": Path(source_pdf).name if source_pdf else "",
                "input_file": input_file or source_pdf,
                "document_id": document_id,
                "page_id": page_id,
                "page": page,
                "original_page_key": key_provenance.original_page_key,
                "normalized_page_key": key_provenance.normalized_page_key,
                "original_document_key": key_provenance.original_document_key,
                "normalized_document_key": key_provenance.normalized_document_key,
                "key_normalization_applied": key_provenance.key_normalization_applied,
                "key_normalization_warnings": key_provenance.key_normalization_warnings,
                "page_text": page_text,
                "output_text_length": output_text_length,
                "status": status,
                "failure_reason": failure_reason,
                "extraction_method": extraction_method,
                "engine_used": engine_used,
                "engine_statuses": engine_statuses,
                "engine_runtimes_ms": engine_runtime_map,
                "confidence": confidence,
                "runtime_ms": runtime_ms,
                "page_quality_score": page_quality_score,
                "document_quality_score": document_quality_score,
                "quality_class": quality_class,
                "quality_reasons": quality_reasons,
                "failed_gate": failed_gate,
                "gate_reason": gate_reason,
                "adapter_used": adapter_used,
                "recommended_preprocessing_profile": recommended_preprocessing_profile,
                "applied_preprocessing_profile": applied_preprocessing_profile,
                "postprocess_quality_score": postprocess_quality_score,
                "lexicon_coverage": lexicon_coverage,
                "unknown_token_rate": unknown_token_rate,
                "protected_character_changes": protected_character_changes,
                **runtime_detection,
            }

    return records, malformed_rows


def _load_progress_records(progress_csv: str | None) -> dict[str, dict[str, Any]]:
    if not progress_csv:
        return {}

    path = Path(progress_csv)
    if not path.exists():
        return {}

    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pdf_name = str(_row_get(row, "pdf_name", "pdf", "pdf link", default="")).strip()
            page = _resolve_page_number(_row_get(row, "page", "page_no", "page_number", default=""), pdf_name)

            page_reference = str(_row_get(row, "page_reference", "page_ref", default="")).strip()
            explicit_key = str(_row_get(row, "page_key", "id", default="")).strip()
            key_provenance = build_key_provenance(
                page_key=explicit_key,
                document_key=pdf_name,
                page=page,
                page_reference=page_reference,
                unicode_form="NFKC",
            )
            page_key = key_provenance.normalized_page_key
            runtime_detection = _extract_runtime_detection_fields(row)

            records[page_key] = {
                "runtime_ms": _extract_runtime_ms(row),
                "status": str(_row_get(row, "status", default="")).strip().lower(),
                "failure_reason": str(_row_get(row, "failure_reason", default="")).strip(),
                "engine_statuses": _parse_engine_statuses(_row_get(row, "engine_statuses", default="")),
                "engine_runtimes_ms": _parse_engine_runtime_map(
                    _row_get(row, "engine_runtimes_ms", "engine_runtime_ms", "runtime_by_engine", default="")
                ),
                **runtime_detection,
            }
    return records


def _merge_ocr_with_progress(
    ocr_records: dict[str, dict[str, Any]],
    progress_records: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    all_keys = set(ocr_records) | set(progress_records)
    for key in all_keys:
        ocr_row = dict(ocr_records.get(key, {}))
        progress_row = progress_records.get(key, {})

        if "runtime_ms" not in ocr_row or ocr_row.get("runtime_ms") is None:
            ocr_row["runtime_ms"] = progress_row.get("runtime_ms")

        if not ocr_row.get("status") and progress_row.get("status"):
            ocr_row["status"] = progress_row.get("status")

        if not ocr_row.get("failure_reason") and progress_row.get("failure_reason"):
            ocr_row["failure_reason"] = progress_row.get("failure_reason")

        if not ocr_row.get("engine_statuses") and progress_row.get("engine_statuses"):
            ocr_row["engine_statuses"] = progress_row.get("engine_statuses")

        if not ocr_row.get("engine_runtimes_ms") and progress_row.get("engine_runtimes_ms"):
            ocr_row["engine_runtimes_ms"] = progress_row.get("engine_runtimes_ms")

        for field in RUNTIME_DETECTION_FIELDS:
            ocr_value = ocr_row.get(field)
            progress_value = progress_row.get(field)
            if progress_value in (None, ""):
                continue
            if ocr_value in (None, ""):
                ocr_row[field] = progress_value
                continue
            if field in {"detected_orientation_class", "detected_layout_type"} and str(ocr_value).strip().lower() == "unknown":
                ocr_row[field] = progress_value

        merged[key] = ocr_row

    return merged


def _validate_records_before_scoring(
    gold_records: dict[str, dict[str, Any]],
    ocr_records: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    malformed: list[dict[str, Any]] = []

    valid_gold: dict[str, dict[str, Any]] = {}
    for page_key, row in gold_records.items():
        row_copy = dict(row)
        pdf_name = str(row_copy.get("pdf_name", "")).strip()
        normalized_document_key = str(row_copy.get("normalized_document_key", "")).strip()
        if not pdf_name and normalized_document_key:
            pdf_name = f"{normalized_document_key}.pdf"
            row_copy["pdf_name"] = pdf_name

        effective_page_key = str(row_copy.get("normalized_page_key", "")).strip() or str(page_key).strip()
        page = _safe_int(row.get("page"))
        issues: list[str] = []
        if not pdf_name:
            issues.append("missing_pdf_name")
        if page is None:
            issues.append("missing_page_number")
        if not effective_page_key or effective_page_key.startswith("unknown"):
            issues.append("invalid_page_key")

        if issues:
            malformed.append(
                {
                    "source": "gold",
                    "line_number": "",
                    "page_key": effective_page_key,
                    "pdf_name": pdf_name,
                    "page": row_copy.get("page", ""),
                    "issue": "invalid_gold_record",
                    "detail": "|".join(issues),
                }
            )
            continue

        row_copy["page_key"] = effective_page_key
        valid_gold[effective_page_key] = row_copy

    valid_ocr: dict[str, dict[str, Any]] = {}
    for page_key, row in ocr_records.items():
        row_copy = dict(row)

        normalized_page_key = str(row_copy.get("normalized_page_key", "")).strip()
        effective_page_key = normalized_page_key or str(page_key).strip()

        source_pdf = str(row_copy.get("pdf_name", "")).strip() or str(row_copy.get("input_file", "")).strip()
        page = _safe_int(row_copy.get("page"))
        status = str(row_copy.get("status", "")).strip().lower()
        extraction_method = str(row_copy.get("extraction_method", "")).strip()
        engine_used = str(row_copy.get("engine_used", "")).strip() or extraction_method
        runtime_ms = _safe_float(row_copy.get("runtime_ms"))
        output_text_length = _safe_int(row_copy.get("output_text_length"))

        if output_text_length is None:
            output_text_length = len(str(row_copy.get("page_text", "") or ""))
            row_copy["output_text_length"] = output_text_length

        document_id = str(row_copy.get("document_id", "")).strip()
        if not document_id and source_pdf:
            document_id = _normalize_page_key(Path(source_pdf).stem)
            row_copy["document_id"] = document_id
        if not document_id:
            document_id = str(row_copy.get("normalized_document_key", "")).strip()
            if document_id:
                row_copy["document_id"] = document_id

        page_id = str(row_copy.get("page_id", "")).strip()
        if not page_id and document_id and page is not None:
            page_id = f"{document_id}_page_{page}"
            row_copy["page_id"] = page_id

        issues: list[str] = []
        if not source_pdf:
            issues.append("missing_pdf_or_input_file")
        if page is None:
            issues.append("missing_page_number")
        if not document_id:
            issues.append("missing_document_id")
        if not page_id:
            issues.append("missing_page_id")
        if not status:
            issues.append("missing_status")
        if "failure_reason" not in row_copy:
            issues.append("missing_failure_reason")
        if not engine_used:
            issues.append("missing_engine_used")
        if runtime_ms is None:
            issues.append("missing_runtime_ms")
        if output_text_length is None:
            issues.append("missing_output_text_length")
        if not effective_page_key or effective_page_key.startswith("unknown"):
            issues.append("invalid_page_key")

        if issues:
            malformed.append(
                {
                    "source": "ocr",
                    "line_number": "",
                    "page_key": effective_page_key,
                    "pdf_name": source_pdf,
                    "page": row_copy.get("page", ""),
                    "issue": "invalid_ocr_record",
                    "detail": "|".join(issues),
                }
            )
            continue

        row_copy["engine_used"] = engine_used
        row_copy["page_key"] = effective_page_key
        valid_ocr[effective_page_key] = row_copy

    return valid_gold, valid_ocr, malformed


def _prepare_engine_statuses(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    statuses = dict(row.get("engine_statuses") or {})
    method = str(row.get("extraction_method", ""))

    if statuses:
        return statuses

    if method == "text_layer":
        return {"text_layer": {"status": ENGINE_STATUS_AVAILABLE, "reason": ""}}
    if method.startswith("ocr_paddle"):
        return {"paddle": {"status": ENGINE_STATUS_AVAILABLE, "reason": ""}}
    return {}


def _evaluate_pages(
    gold_records: dict[str, dict[str, Any]],
    ocr_records: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    norm_cfg: NormalizationConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    per_page: list[dict[str, Any]] = []

    substitutions: Counter[tuple[str, str]] = Counter()
    insertions: Counter[str] = Counter()
    deletions: Counter[str] = Counter()

    runtime_values: list[float] = []
    cer_values: list[float] = []
    wer_values: list[float] = []
    norm_cer_values: list[float] = []
    norm_wer_values: list[float] = []
    confidence_values: list[float] = []

    empty_outputs = 0
    timed_out_pages = 0
    missing_predictions = 0
    akkadian_metric_pages = 0

    for page_key in sorted(gold_records.keys()):
        gold = gold_records[page_key]
        prediction = dict(ocr_records.get(page_key, {}))

        if not prediction:
            missing_predictions += 1

        predicted_text = _canonical_text(str(prediction.get("page_text", "")))
        status = str(prediction.get("status", PAGE_STATUS_FAILED)).strip().lower() or PAGE_STATUS_FAILED
        failure_reason = str(prediction.get("failure_reason", "")).strip()
        runtime_ms = _safe_float(prediction.get("runtime_ms"))
        extraction_method = str(prediction.get("extraction_method", "")).strip()
        confidence = _safe_float(prediction.get("confidence"))
        engine_statuses = _prepare_engine_statuses(prediction)
        engine_runtime_map = prediction.get("engine_runtimes_ms") or {}

        empty_output = not predicted_text.strip()
        timed_out = status == PAGE_STATUS_TIMEOUT or "timeout" in failure_reason.lower()

        if empty_output:
            empty_outputs += 1
        if timed_out:
            timed_out_pages += 1

        if runtime_ms is not None:
            runtime_values.append(runtime_ms)
        if confidence is not None:
            confidence_values.append(confidence)

        reference_text = _canonical_text(str(gold.get("ground_truth_text", "")))

        cer = _char_error_rate(reference_text, predicted_text)
        wer = _word_error_rate(reference_text, predicted_text)

        norm_ref = _normalize_for_metrics(reference_text, norm_cfg)
        norm_hyp = _normalize_for_metrics(predicted_text, norm_cfg)
        normalized_cer = _char_error_rate(norm_ref, norm_hyp)
        normalized_wer = _word_error_rate(norm_ref, norm_hyp)

        cer_values.append(cer)
        wer_values.append(wer)
        norm_cer_values.append(normalized_cer)
        norm_wer_values.append(normalized_wer)

        unicode_form = args.unicode_form
        unicode_ref = unicodedata.normalize(unicode_form, reference_text)
        unicode_hyp = unicodedata.normalize(unicode_form, predicted_text)
        unicode_equal = unicode_ref == unicode_hyp

        diacritic_preservation = _counter_preservation_rate(
            _extract_diacritic_chars(reference_text),
            _extract_diacritic_chars(predicted_text),
        )
        punctuation_preservation = _counter_preservation_rate(
            _extract_punctuation_chars(reference_text),
            _extract_punctuation_chars(predicted_text),
        )
        whitespace_preservation = _counter_preservation_rate(
            _extract_whitespace_runs(reference_text),
            _extract_whitespace_runs(predicted_text),
        )

        rtl_ref_chars = [char for char in reference_text if _is_rtl_char(char)]
        rtl_hyp_chars = [char for char in predicted_text if _is_rtl_char(char)]
        rtl_coverage = _counter_preservation_rate(rtl_ref_chars, rtl_hyp_chars)

        expected_scripts = _parse_expected_scripts(str(gold.get("script_type", "")))
        script_coverage = _script_character_coverage(reference_text, predicted_text, expected_scripts)

        document_type = str(gold.get("document_type", "") or "unknown").strip().lower()
        layout_type = str(gold.get("layout_type", "") or "unknown").strip().lower()

        line_order_similarity = _line_order_similarity(reference_text, predicted_text)
        paragraph_order_similarity = _paragraph_order_similarity(reference_text, predicted_text)
        header_footer_mismatch = _header_footer_mismatch(reference_text, predicted_text)

        kv_pairs_ref = set(_extract_key_value_pairs(reference_text))
        kv_pairs_hyp = set(_extract_key_value_pairs(predicted_text))
        key_value_f1 = _set_f1(kv_pairs_ref, kv_pairs_hyp)

        table_rows_ref = _extract_table_like_rows(reference_text)
        table_rows_hyp = _extract_table_like_rows(predicted_text)
        table_row_similarity = _set_f1(
            set((r, "") for r in table_rows_ref),
            set((r, "") for r in table_rows_hyp),
        )

        structured_json_similarity = _structured_json_similarity(reference_text, predicted_text)
        entity_association_accuracy = key_value_f1
        box_text_linking_quality = script_coverage

        text_length_ratio = (len(predicted_text) / len(reference_text)) if reference_text else (1.0 if not predicted_text else None)
        page_equivalence_valid = bool(str(gold.get("page_reference", "")).strip() or str(gold.get("pdf_name", "")).strip())
        target_scope_valid = text_length_ratio is not None and 0.5 <= text_length_ratio <= 2.0

        metric_family = "plain_text_primary"
        if "receipt" in document_type or "commercial" in document_type:
            metric_family = "receipt_structured_primary"
        elif "form" in document_type:
            metric_family = "form_structured_primary"
        elif layout_type in {"semi_structured", "form_layout"}:
            metric_family = "structured_secondary"

        if "local_gold" in str(gold.get("dataset_id", "")).lower():
            metric_family = "local_scope_validated"

        akk_metrics: dict[str, float | None] = {}
        if _should_run_akkadian_metrics(gold, args):
            akkadian_metric_pages += 1
            akk_metrics = _akkadian_optional_metrics(reference_text, predicted_text)

        # Confusion matrix uses unicode-normalized and whitespace-collapsed strings.
        conf_ref = _normalize_for_metrics(
            reference_text,
            NormalizationConfig(unicode_form=args.unicode_form, whitespace_mode="collapse"),
        )
        conf_hyp = _normalize_for_metrics(
            predicted_text,
            NormalizationConfig(unicode_form=args.unicode_form, whitespace_mode="collapse"),
        )
        for ref_char, hyp_char, op in _align_chars_for_confusion(conf_ref, conf_hyp):
            if op == "replace" and ref_char is not None and hyp_char is not None:
                substitutions[(ref_char, hyp_char)] += 1
            elif op == "delete" and ref_char is not None:
                deletions[ref_char] += 1
            elif op == "insert" and hyp_char is not None:
                insertions[hyp_char] += 1

        detected_orientation_angle = _safe_float(prediction.get("detected_orientation_angle"))
        detected_rotation_base_angle = _safe_float(prediction.get("detected_rotation_base_angle"))
        detected_skew_angle = _safe_float(prediction.get("detected_skew_angle"))

        detected_orientation_class = str(prediction.get("detected_orientation_class", "") or "").strip()
        if not detected_orientation_class:
            detected_orientation_class = _derive_orientation_class(detected_orientation_angle)

        detected_layout_type = str(prediction.get("detected_layout_type", "") or "").strip() or "unknown"
        detected_column_count = _safe_int(prediction.get("detected_column_count"))
        detected_has_columns = _safe_bool_text(prediction.get("detected_has_columns", ""))
        if not detected_has_columns and detected_column_count is not None:
            detected_has_columns = "true" if detected_column_count > 1 else "false"

        original_page_key = str(prediction.get("original_page_key", "") or gold.get("original_page_key", "")).strip()
        normalized_page_key = str(
            prediction.get("normalized_page_key", "") or gold.get("normalized_page_key", "") or page_key
        ).strip()
        original_document_key = str(
            prediction.get("original_document_key", "") or gold.get("original_document_key", "") or gold.get("pdf_name", "")
        ).strip()
        normalized_document_key = str(
            prediction.get("normalized_document_key", "")
            or gold.get("normalized_document_key", "")
            or _normalize_page_key(Path(str(gold.get("pdf_name", "") or "unknown")).stem)
        ).strip()
        key_normalization_applied = bool(
            _safe_bool(prediction.get("key_normalization_applied"))
            or _safe_bool(gold.get("key_normalization_applied"))
            or (normalized_page_key != original_page_key)
            or (normalized_document_key != original_document_key)
        )

        warning_tokens: list[str] = []
        for raw_warnings in (
            prediction.get("key_normalization_warnings", ""),
            gold.get("key_normalization_warnings", ""),
        ):
            for token in str(raw_warnings or "").split("|"):
                normalized_token = token.strip()
                if normalized_token and normalized_token not in warning_tokens:
                    warning_tokens.append(normalized_token)
        key_normalization_warnings = "|".join(warning_tokens)

        page_row: dict[str, Any] = {
            "page_key": page_key,
            "original_page_key": original_page_key,
            "normalized_page_key": normalized_page_key,
            "original_document_key": original_document_key,
            "normalized_document_key": normalized_document_key,
            "key_normalization_applied": key_normalization_applied,
            "key_normalization_warnings": key_normalization_warnings,
            "pdf_name": gold.get("pdf_name", ""),
            "page": gold.get("page", ""),
            "page_reference": gold.get("page_reference", ""),
            "status": status,
            "failure_reason": failure_reason,
            "extraction_method": extraction_method,
            "detected_orientation_angle": detected_orientation_angle,
            "detected_orientation_class": detected_orientation_class,
            "detected_rotation_base_angle": detected_rotation_base_angle,
            "detected_skew_angle": detected_skew_angle,
            "detected_layout_type": detected_layout_type,
            "detected_column_count": detected_column_count,
            "detected_has_columns": detected_has_columns,
            "runtime_ms": runtime_ms,
            "confidence": confidence,
            "empty_output": empty_output,
            "timed_out": timed_out,
            "cer": cer,
            "wer": wer,
            "normalized_cer": normalized_cer,
            "normalized_wer": normalized_wer,
            "unicode_form": unicode_form,
            "unicode_equal_after_normalization": unicode_equal,
            "unicode_normalization_changed_gold": reference_text != unicode_ref,
            "unicode_normalization_changed_ocr": predicted_text != unicode_hyp,
            "diacritic_preservation_rate": diacritic_preservation,
            "punctuation_preservation_rate": punctuation_preservation,
            "whitespace_preservation_rate": whitespace_preservation,
            "rtl_character_coverage": rtl_coverage,
            "script_character_coverage": script_coverage,
            "line_order_similarity": line_order_similarity,
            "paragraph_order_similarity": paragraph_order_similarity,
            "header_footer_mismatch": header_footer_mismatch,
            "key_value_f1": key_value_f1,
            "table_row_similarity": table_row_similarity,
            "structured_json_similarity": structured_json_similarity,
            "entity_association_accuracy": entity_association_accuracy,
            "box_text_linking_quality": box_text_linking_quality,
            "page_equivalence_valid": page_equivalence_valid,
            "target_scope_valid": target_scope_valid,
            "text_length_ratio": text_length_ratio,
            "metric_family": metric_family,
            "ground_truth_text_length": len(reference_text),
            "ocr_text_length": len(predicted_text),
            "page_quality_score": _safe_float(prediction.get("page_quality_score")),
            "document_quality_score": _safe_float(prediction.get("document_quality_score")),
            "quality_class": str(prediction.get("quality_class", "") or "").strip(),
            "quality_reasons": str(prediction.get("quality_reasons", "") or "").strip(),
            "failed_gate": str(prediction.get("failed_gate", "") or "").strip(),
            "gate_reason": str(prediction.get("gate_reason", "") or "").strip(),
            "adapter_used": str(prediction.get("adapter_used", "") or "").strip(),
            "recommended_preprocessing_profile": str(
                prediction.get("recommended_preprocessing_profile", "") or ""
            ).strip(),
            "applied_preprocessing_profile": str(
                prediction.get("applied_preprocessing_profile", "") or ""
            ).strip(),
            "postprocess_quality_score": _safe_float(prediction.get("postprocess_quality_score")),
            "lexicon_coverage": _safe_float(prediction.get("lexicon_coverage")),
            "unknown_token_rate": _safe_float(prediction.get("unknown_token_rate")),
            "protected_character_changes": _safe_int(prediction.get("protected_character_changes")),
            "engine_statuses": json.dumps(engine_statuses, ensure_ascii=True, sort_keys=True),
            "engine_runtimes_ms": json.dumps(engine_runtime_map, ensure_ascii=True, sort_keys=True),
        }

        for field in GENERAL_METADATA_FIELDS:
            value = gold.get(field, "")
            if isinstance(value, list):
                page_row[field] = "|".join(str(item) for item in value)
            else:
                page_row[field] = value

        page_row.update(akk_metrics)
        per_page.append(page_row)

    diagnostics = {
        "empty_outputs": empty_outputs,
        "timed_out_pages": timed_out_pages,
        "missing_predictions": missing_predictions,
        "akkadian_metric_pages": akkadian_metric_pages,
        "runtime_values": runtime_values,
        "cer_values": cer_values,
        "wer_values": wer_values,
        "normalized_cer_values": norm_cer_values,
        "normalized_wer_values": norm_wer_values,
        "confidence_values": confidence_values,
    }

    confusion = {
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
    }

    return per_page, diagnostics, confusion


def _aggregate_group_metrics(
    rows: list[dict[str, Any]],
    group_field: str,
    output_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(group_field, "") or "unknown")
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        cer_values = [float(item["cer"]) for item in items]
        wer_values = [float(item["wer"]) for item in items]
        ncer_values = [float(item["normalized_cer"]) for item in items]
        nwer_values = [float(item["normalized_wer"]) for item in items]
        runtime_values = [float(item["runtime_ms"]) for item in items if item.get("runtime_ms") is not None]
        empty_rate = sum(1 for item in items if _safe_bool(item.get("empty_output"))) / len(items)
        timeout_rate = sum(1 for item in items if _safe_bool(item.get("timed_out"))) / len(items)

        out.append(
            {
                output_name: key,
                "pages": len(items),
                "cer_mean": _safe_mean(cer_values),
                "wer_mean": _safe_mean(wer_values),
                "normalized_cer_mean": _safe_mean(ncer_values),
                "normalized_wer_mean": _safe_mean(nwer_values),
                "runtime_ms_mean": _safe_mean(runtime_values),
                "runtime_ms_p50": _quantile(runtime_values, 0.50),
                "runtime_ms_p90": _quantile(runtime_values, 0.90),
                "runtime_ms_p95": _quantile(runtime_values, 0.95),
                "empty_output_rate": empty_rate,
                "timeout_rate": timeout_rate,
            }
        )
    return out


def _build_engine_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "pages_seen": 0,
            "status_counts": Counter(),
            "runtime_samples": [],
            "runtime_estimated_samples": [],
            "confidence_samples": [],
        }
    )

    for row in rows:
        runtime_ms = _safe_float(row.get("runtime_ms"))
        confidence = _safe_float(row.get("confidence"))

        engine_statuses = _parse_engine_statuses(row.get("engine_statuses", ""))
        if not engine_statuses:
            # best-effort fallback from extraction method
            method = str(row.get("extraction_method", ""))
            if method == "text_layer":
                engine_statuses = {"text_layer": {"status": ENGINE_STATUS_AVAILABLE, "reason": ""}}
            elif method.startswith("ocr_paddle"):
                engine_statuses = {"paddle": {"status": ENGINE_STATUS_AVAILABLE, "reason": ""}}

        engine_runtime_map = _parse_engine_runtime_map(row.get("engine_runtimes_ms", ""))

        available_engines = [
            name
            for name, payload in engine_statuses.items()
            if str(payload.get("status", "")).strip().lower() in {ENGINE_STATUS_AVAILABLE, ENGINE_STATUS_AVAILABLE_UNHEALTHY}
        ]

        for engine_name, payload in engine_statuses.items():
            status = str(payload.get("status", "")).strip().lower() or "unknown"
            agg = aggregates[engine_name]
            agg["pages_seen"] += 1
            agg["status_counts"][status] += 1
            if confidence is not None and status in {ENGINE_STATUS_AVAILABLE, ENGINE_STATUS_AVAILABLE_UNHEALTHY}:
                agg["confidence_samples"].append(confidence)

        if engine_runtime_map:
            for engine_name, runtime in engine_runtime_map.items():
                agg = aggregates[engine_name]
                agg["runtime_samples"].append(runtime)
        elif runtime_ms is not None and available_engines:
            split_runtime = runtime_ms / len(available_engines)
            for engine_name in available_engines:
                agg = aggregates[engine_name]
                agg["runtime_samples"].append(split_runtime)
                agg["runtime_estimated_samples"].append(split_runtime)

    rows_out: list[dict[str, Any]] = []
    for engine_name, agg in sorted(aggregates.items()):
        runtime_samples = [float(x) for x in agg["runtime_samples"]]
        confidence_samples = [float(x) for x in agg["confidence_samples"]]

        rows_out.append(
            {
                "engine": engine_name,
                "pages_seen": agg["pages_seen"],
                "status_available_count": agg["status_counts"].get(ENGINE_STATUS_AVAILABLE, 0),
                "status_available_but_unhealthy_count": agg["status_counts"].get(ENGINE_STATUS_AVAILABLE_UNHEALTHY, 0),
                "status_unavailable_dependency_error_count": agg["status_counts"].get("unavailable_dependency_error", 0),
                "status_disabled_by_config_count": agg["status_counts"].get("disabled_by_config", 0),
                "status_timed_out_count": agg["status_counts"].get("timed_out", 0),
                "status_failed_on_page_count": agg["status_counts"].get("failed_on_page", 0),
                "runtime_ms_mean": _safe_mean(runtime_samples),
                "runtime_ms_p50": _quantile(runtime_samples, 0.50),
                "runtime_ms_p90": _quantile(runtime_samples, 0.90),
                "runtime_ms_p95": _quantile(runtime_samples, 0.95),
                "runtime_estimated_rate": (
                    len(agg["runtime_estimated_samples"]) / len(runtime_samples) if runtime_samples else None
                ),
                "confidence_mean": _safe_mean(confidence_samples),
                "confidence_p50": _quantile(confidence_samples, 0.50),
                "confidence_p90": _quantile(confidence_samples, 0.90),
            }
        )
    return rows_out


def _build_summary(
    per_page: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    gold_count: int,
    ocr_count: int,
) -> dict[str, Any]:
    runtime_values = diagnostics["runtime_values"]
    cer_values = diagnostics["cer_values"]
    wer_values = diagnostics["wer_values"]
    ncer_values = diagnostics["normalized_cer_values"]
    nwer_values = diagnostics["normalized_wer_values"]
    confidence_values = diagnostics["confidence_values"]

    pages = len(per_page)
    empty_rate = diagnostics["empty_outputs"] / pages if pages else 0.0
    timeout_rate = diagnostics["timed_out_pages"] / pages if pages else 0.0
    failed_rate = (
        sum(1 for row in per_page if str(row.get("status", "")).strip().lower() == PAGE_STATUS_FAILED) / pages
        if pages
        else 0.0
    )

    # Optional metric coverage rates
    diacritic_values = [float(v) for v in (row.get("diacritic_preservation_rate") for row in per_page) if v is not None]
    punctuation_values = [float(v) for v in (row.get("punctuation_preservation_rate") for row in per_page) if v is not None]
    whitespace_values = [float(v) for v in (row.get("whitespace_preservation_rate") for row in per_page) if v is not None]
    rtl_values = [float(v) for v in (row.get("rtl_character_coverage") for row in per_page) if v is not None]
    script_values = [float(v) for v in (row.get("script_character_coverage") for row in per_page) if v is not None]
    quality_values = [float(v) for v in (row.get("page_quality_score") for row in per_page) if v is not None]

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "matched_pages": pages,
        "gold_pages": gold_count,
        "ocr_pages": ocr_count,
        "missing_predictions": diagnostics["missing_predictions"],
        "empty_output_rate": empty_rate,
        "timeout_rate": timeout_rate,
        "failed_page_rate": failed_rate,
        "runtime_ms_mean": _safe_mean(runtime_values),
        "runtime_ms_std": _safe_std(runtime_values),
        "runtime_ms_p50": _quantile(runtime_values, 0.50),
        "runtime_ms_p90": _quantile(runtime_values, 0.90),
        "runtime_ms_p95": _quantile(runtime_values, 0.95),
        "cer_mean": _safe_mean(cer_values),
        "cer_std": _safe_std(cer_values),
        "cer_p50": _quantile(cer_values, 0.50),
        "cer_p90": _quantile(cer_values, 0.90),
        "cer_p95": _quantile(cer_values, 0.95),
        "wer_mean": _safe_mean(wer_values),
        "wer_std": _safe_std(wer_values),
        "wer_p50": _quantile(wer_values, 0.50),
        "wer_p90": _quantile(wer_values, 0.90),
        "wer_p95": _quantile(wer_values, 0.95),
        "normalized_cer_mean": _safe_mean(ncer_values),
        "normalized_cer_std": _safe_std(ncer_values),
        "normalized_cer_p50": _quantile(ncer_values, 0.50),
        "normalized_cer_p90": _quantile(ncer_values, 0.90),
        "normalized_cer_p95": _quantile(ncer_values, 0.95),
        "normalized_wer_mean": _safe_mean(nwer_values),
        "normalized_wer_std": _safe_std(nwer_values),
        "normalized_wer_p50": _quantile(nwer_values, 0.50),
        "normalized_wer_p90": _quantile(nwer_values, 0.90),
        "normalized_wer_p95": _quantile(nwer_values, 0.95),
        "confidence_mean": _safe_mean(confidence_values),
        "confidence_p50": _quantile(confidence_values, 0.50),
        "confidence_p90": _quantile(confidence_values, 0.90),
        "page_quality_score_mean": _safe_mean(quality_values),
        "page_quality_score_p50": _quantile(quality_values, 0.50),
        "page_quality_score_p90": _quantile(quality_values, 0.90),
        "diacritic_preservation_mean": _safe_mean(diacritic_values),
        "punctuation_preservation_mean": _safe_mean(punctuation_values),
        "whitespace_preservation_mean": _safe_mean(whitespace_values),
        "rtl_coverage_mean": _safe_mean(rtl_values),
        "script_coverage_mean": _safe_mean(script_values),
        "akkadian_optional_pages": diagnostics["akkadian_metric_pages"],
    }


def _sorted_fieldnames(rows: list[dict[str, Any]], preferred_first: list[str]) -> list[str]:
    observed: set[str] = set()
    for row in rows:
        observed.update(row.keys())

    fieldnames: list[str] = []
    for name in preferred_first:
        if name in observed:
            fieldnames.append(name)
            observed.remove(name)

    fieldnames.extend(sorted(observed))
    return fieldnames


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred_first: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _sorted_fieldnames(rows, preferred_first)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_confusion_matrix(path: Path, confusion: dict[str, Any]) -> None:
    substitutions: Counter[tuple[str, str]] = confusion["substitutions"]
    deletions: Counter[str] = confusion["deletions"]
    insertions: Counter[str] = confusion["insertions"]

    payload = {
        "substitutions": [
            {
                "reference": _char_label(ref),
                "hypothesis": _char_label(hyp),
                "count": count,
                "reference_codepoint": f"U+{ord(ref):04X}",
                "hypothesis_codepoint": f"U+{ord(hyp):04X}",
            }
            for (ref, hyp), count in substitutions.most_common(300)
        ],
        "deletions": [
            {
                "reference": _char_label(ref),
                "count": count,
                "reference_codepoint": f"U+{ord(ref):04X}",
            }
            for ref, count in deletions.most_common(200)
        ],
        "insertions": [
            {
                "hypothesis": _char_label(hyp),
                "count": count,
                "hypothesis_codepoint": f"U+{ord(hyp):04X}",
            }
            for hyp, count in insertions.most_common(200)
        ],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=True, indent=2)


def _load_single_row_csv(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            return row
    return None


def _load_rows_by_key(path: Path, key_field: str) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = str(row.get(key_field, "")).strip()
            if key:
                mapping[key] = row
    return mapping


def _metric_direction(metric_name: str) -> int:
    lowered = metric_name.lower()
    if "preservation" in lowered or "coverage" in lowered or "confidence" in lowered:
        return 1
    if "cer" in lowered or "wer" in lowered:
        return -1
    if "runtime" in lowered:
        return -1
    if lowered in {"empty_output_rate", "timeout_rate", "missing_predictions"}:
        return -1
    return 0


def _compare_with_baseline(
    summary_row: dict[str, Any],
    per_page_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    baseline_summary = Path(args.baseline_summary_csv) if args.baseline_summary_csv else None
    baseline_pages = Path(args.baseline_per_page_csv) if args.baseline_per_page_csv else None

    if args.baseline_dir:
        baseline_dir = Path(args.baseline_dir)
        if baseline_summary is None:
            baseline_summary = baseline_dir / "evaluation_summary.csv"
        if baseline_pages is None:
            baseline_pages = baseline_dir / "per_page_metrics.csv"

    if baseline_summary is None or not baseline_summary.exists():
        return None

    baseline_row = _load_single_row_csv(baseline_summary)
    if baseline_row is None:
        return None

    numeric_deltas: list[dict[str, Any]] = []
    improved: list[dict[str, Any]] = []
    worsened: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    suspicious: list[dict[str, Any]] = []

    for metric, current_value in summary_row.items():
        baseline_value = baseline_row.get(metric)
        current_num = _safe_float(current_value)
        baseline_num = _safe_float(baseline_value)
        if current_num is None or baseline_num is None:
            continue

        delta = current_num - baseline_num
        direction = _metric_direction(metric)
        entry = {
            "metric": metric,
            "current": current_num,
            "baseline": baseline_num,
            "delta": delta,
        }
        numeric_deltas.append(entry)

        if abs(delta) <= 1e-9 or direction == 0:
            unchanged.append(entry)
        elif (direction == 1 and delta > 0) or (direction == -1 and delta < 0):
            improved.append(entry)
        else:
            worsened.append(entry)

        std_key = metric.replace("_mean", "_std")
        baseline_std = _safe_float(baseline_row.get(std_key))
        baseline_n = _safe_float(baseline_row.get("matched_pages"))
        if baseline_std is None or baseline_n is None or baseline_n <= 0:
            continue

        threshold = max(0.01, 2.0 * baseline_std / math.sqrt(baseline_n))
        if abs(delta) > threshold:
            suspicious.append({**entry, "threshold": threshold})

    page_regressions: dict[str, Any] = {
        "improved_pages": [],
        "regressed_pages": [],
    }

    if baseline_pages and baseline_pages.exists():
        baseline_rows_by_page_key = _load_rows_by_key(baseline_pages, "page_key")
        baseline_by_key: dict[str, dict[str, Any]] = {}
        for baseline_row in baseline_rows_by_page_key.values():
            baseline_key = str(baseline_row.get("normalized_page_key", "") or "").strip()
            if not baseline_key:
                baseline_provenance = build_key_provenance(
                    page_key=baseline_row.get("page_key", ""),
                    document_key=baseline_row.get("pdf_name", ""),
                    page=baseline_row.get("page", ""),
                    page_reference=baseline_row.get("page_reference", ""),
                    unicode_form="NFKC",
                )
                baseline_key = baseline_provenance.normalized_page_key
            if baseline_key and baseline_key not in baseline_by_key:
                baseline_by_key[baseline_key] = baseline_row

        page_deltas: list[dict[str, Any]] = []

        for row in per_page_rows:
            key = str(row.get("normalized_page_key", "") or "").strip()
            if not key:
                row_provenance = build_key_provenance(
                    page_key=row.get("page_key", ""),
                    document_key=row.get("pdf_name", ""),
                    page=row.get("page", ""),
                    page_reference=row.get("page_reference", ""),
                    unicode_form="NFKC",
                )
                key = row_provenance.normalized_page_key
            if not key or key not in baseline_by_key:
                continue

            current_score = _safe_float(row.get("normalized_cer"))
            if current_score is None:
                continue
            current_score += _safe_float(row.get("normalized_wer")) or 0.0

            baseline_row_page = baseline_by_key[key]
            baseline_score = _safe_float(baseline_row_page.get("normalized_cer"))
            if baseline_score is None:
                continue
            baseline_score += _safe_float(baseline_row_page.get("normalized_wer")) or 0.0

            improvement = baseline_score - current_score
            page_deltas.append(
                {
                    "page_key": str(row.get("page_key", "")).strip() or key,
                    "normalized_page_key": key,
                    "improvement_score": improvement,
                    "baseline_score": baseline_score,
                    "current_score": current_score,
                }
            )

        page_deltas.sort(key=lambda item: item["improvement_score"], reverse=True)
        top_n = max(1, int(args.top_n_regression_pages))
        page_regressions["improved_pages"] = page_deltas[:top_n]
        page_regressions["regressed_pages"] = list(reversed(page_deltas[-top_n:]))

    improved.sort(key=lambda item: abs(item["delta"]), reverse=True)
    worsened.sort(key=lambda item: abs(item["delta"]), reverse=True)
    unchanged.sort(key=lambda item: item["metric"])
    suspicious.sort(key=lambda item: abs(item["delta"]), reverse=True)

    return {
        "baseline_summary_path": str(baseline_summary),
        "baseline_per_page_path": str(baseline_pages) if baseline_pages else "",
        "improved_metrics": improved,
        "worsened_metrics": worsened,
        "unchanged_metrics": unchanged,
        "statistically_suspicious_changes": suspicious,
        **page_regressions,
    }


def _build_run_metadata(
    args: argparse.Namespace,
    outputs: dict[str, str],
    summary_row: dict[str, Any],
    baseline_report: dict[str, Any] | None,
    started_at: float,
) -> dict[str, Any]:
    duration_s = time.time() - started_at
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration_s, 3),
        "inputs": {
            "ocr_csv": args.ocr_csv,
            "gold_csv": args.gold_csv,
            "progress_csv": args.progress_csv,
            "permissive_malformed": bool(getattr(args, "permissive_malformed", False)),
            "malformed_rows_csv": getattr(args, "malformed_rows_csv", None),
            "baseline_dir": args.baseline_dir,
            "baseline_summary_csv": args.baseline_summary_csv,
            "baseline_per_page_csv": args.baseline_per_page_csv,
            "run_id": getattr(args, "run_id", None),
            "config_file": getattr(args, "config_file", None),
            "engine_versions_json": getattr(args, "engine_versions_json", None),
            "model_versions_json": getattr(args, "model_versions_json", None),
            "gold_set_version": getattr(args, "gold_set_version", None),
            "experiment_history_jsonl": getattr(args, "experiment_history_jsonl", None),
            "disable_experiment_tracking": bool(getattr(args, "disable_experiment_tracking", False)),
        },
        "normalization": {
            "unicode_form_for_checks": args.unicode_form,
            "normalized_unicode_form": args.normalized_unicode_form,
            "whitespace_mode": args.whitespace_mode,
            "strip_punctuation": args.strip_punctuation,
            "casefold": args.casefold,
        },
        "optional_metrics": {
            "enable_akkadian_metrics": args.enable_akkadian_metrics,
            "disable_metadata_akkadian": args.disable_metadata_akkadian,
        },
        "summary": summary_row,
        "outputs": outputs,
        "baseline_comparison": baseline_report,
    }


def _write_compat_output(path: Path, per_page_rows: list[dict[str, Any]]) -> None:
    """Backward-compatible output for older scripts expecting a single CSV."""
    rows = []
    for row in per_page_rows:
        rows.append(
            {
                "page": row.get("page_key", ""),
                "cer": row.get("cer"),
                "wer": row.get("wer"),
                "gold_chars": row.get("ground_truth_text_length"),
                "ocr_chars": row.get("ocr_text_length"),
            }
        )
    _write_csv(path, rows, ["page", "cer", "wer", "gold_chars", "ocr_chars"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multilingual OCR evaluation against a gold set")
    parser.add_argument("--ocr-csv", required=True, help="Path to OCR output CSV (client_page_text.csv)")
    parser.add_argument("--gold-csv", required=True, help="Path to gold-set CSV")
    parser.add_argument("--progress-csv", default=None, help="Optional progress CSV for runtime/status telemetry")
    parser.add_argument(
        "--permissive-malformed",
        action="store_true",
        default=False,
        help="Skip malformed gold/OCR rows instead of failing fast",
    )
    parser.add_argument(
        "--malformed-rows-csv",
        default=None,
        help="Optional path for malformed row report CSV (default: <output-dir>/malformed_rows.csv)",
    )

    parser.add_argument(
        "--output-dir",
        default="reports/eval_gold",
        help="Output directory for evaluation artifacts",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Deprecated compatibility output path (writes a legacy single CSV)",
    )

    parser.add_argument("--baseline-dir", default=None, help="Directory containing baseline evaluation outputs")
    parser.add_argument("--baseline-summary-csv", default=None, help="Explicit baseline evaluation_summary.csv")
    parser.add_argument("--baseline-per-page-csv", default=None, help="Explicit baseline per_page_metrics.csv")
    parser.add_argument("--top-n-regression-pages", type=int, default=10, help="Top N improved/regressed pages")

    parser.add_argument("--run-id", default=None, help="Optional explicit run identifier")
    parser.add_argument("--config-file", default=None, help="Optional config file path used for this run")
    parser.add_argument(
        "--engine-versions-json",
        default=None,
        help="JSON object or JSON file path describing engine versions",
    )
    parser.add_argument(
        "--model-versions-json",
        default=None,
        help="JSON object or JSON file path describing model versions",
    )
    parser.add_argument(
        "--gold-set-version",
        default=None,
        help="Optional explicit gold-set version label",
    )
    parser.add_argument(
        "--experiment-history-jsonl",
        default=None,
        help="Optional JSONL path where run summaries are appended over time",
    )
    parser.add_argument(
        "--disable-experiment-tracking",
        action="store_true",
        default=False,
        help="Disable generation of tracking dashboards and longitudinal metadata files",
    )

    parser.add_argument("--unicode-form", choices=UNICODE_FORMS, default="NFC", help="Unicode form for checks/confusion")
    parser.add_argument(
        "--normalized-unicode-form",
        choices=UNICODE_FORMS,
        default="NFKC",
        help="Unicode form used for normalized CER/WER",
    )
    parser.add_argument(
        "--whitespace-mode",
        choices=["preserve", "collapse"],
        default="collapse",
        help="Whitespace handling for normalized CER/WER",
    )
    parser.add_argument(
        "--strip-punctuation",
        action="store_true",
        default=False,
        help="Strip punctuation for normalized CER/WER",
    )
    parser.add_argument(
        "--casefold",
        action="store_true",
        default=False,
        help="Use case-folding for normalized CER/WER",
    )

    parser.add_argument(
        "--enable-akkadian-metrics",
        action="store_true",
        default=False,
        help="Run Akkadian/transliteration add-on metrics on all pages",
    )
    parser.add_argument(
        "--disable-metadata-akkadian",
        action="store_true",
        default=False,
        help="Disable metadata-triggered Akkadian metrics (only explicit flag can enable)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    started_at = time.time()
    parser = build_parser()
    args = parser.parse_args(argv)

    norm_cfg = NormalizationConfig(
        unicode_form=args.normalized_unicode_form,
        whitespace_mode=args.whitespace_mode,
        strip_punctuation=args.strip_punctuation,
        casefold=args.casefold,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_records, gold_malformed = _load_gold_records(args.gold_csv)
    ocr_records, ocr_malformed = _load_ocr_records(args.ocr_csv)
    progress_records = _load_progress_records(args.progress_csv)
    merged_ocr = _merge_ocr_with_progress(ocr_records, progress_records)

    gold_records, merged_ocr, validated_malformed = _validate_records_before_scoring(gold_records, merged_ocr)
    malformed_rows = [*gold_malformed, *ocr_malformed, *validated_malformed]

    malformed_report_path: Path | None = None
    if malformed_rows:
        malformed_report_path = (
            Path(args.malformed_rows_csv)
            if args.malformed_rows_csv
            else output_dir / "malformed_rows.csv"
        )
        _write_csv(
            malformed_report_path,
            malformed_rows,
            ["source", "line_number", "page_key", "pdf_name", "page", "issue", "detail"],
        )

        if not args.permissive_malformed:
            print(
                (
                    "ERROR: Malformed rows were detected before scoring. "
                    f"Review {malformed_report_path} and fix inputs, or rerun with --permissive-malformed to skip them."
                ),
                file=sys.stderr,
            )
            return 2

        print(
            (
                "WARNING: Malformed rows were skipped in permissive mode. "
                f"Skipped={len(malformed_rows)} report={malformed_report_path}"
            ),
            file=sys.stderr,
        )

    if not gold_records:
        print("ERROR: No gold records loaded.", file=sys.stderr)
        return 1

    per_page_rows, diagnostics, confusion = _evaluate_pages(gold_records, merged_ocr, args, norm_cfg)
    if not per_page_rows:
        print("ERROR: No pages evaluated.", file=sys.stderr)
        return 1

    summary_row = _build_summary(per_page_rows, diagnostics, len(gold_records), len(merged_ocr))
    per_engine_rows = _build_engine_metrics(per_page_rows)
    per_language_rows = _aggregate_group_metrics(per_page_rows, "language_primary", "language_primary")
    per_layout_rows = _aggregate_group_metrics(per_page_rows, "layout_type", "layout_type")
    per_detected_orientation_rows = _aggregate_group_metrics(
        per_page_rows,
        "detected_orientation_class",
        "detected_orientation_class",
    )
    per_detected_layout_rows = _aggregate_group_metrics(
        per_page_rows,
        "detected_layout_type",
        "detected_layout_type",
    )

    summary_csv = output_dir / "evaluation_summary.csv"
    per_page_csv = output_dir / "per_page_metrics.csv"
    per_engine_csv = output_dir / "per_engine_metrics.csv"
    per_language_csv = output_dir / "per_language_metrics.csv"
    per_layout_csv = output_dir / "per_layout_metrics.csv"
    per_detected_orientation_csv = output_dir / "per_detected_orientation_metrics.csv"
    per_detected_layout_csv = output_dir / "per_detected_layout_metrics.csv"
    confusion_json = output_dir / "confusion_matrix.json"
    run_metadata_json = output_dir / "run_metadata.json"

    _write_csv(summary_csv, [summary_row], ["timestamp_utc", "matched_pages", "gold_pages", "ocr_pages"])
    _write_csv(
        per_page_csv,
        per_page_rows,
        [
            "page_key",
            "original_page_key",
            "normalized_page_key",
            "original_document_key",
            "normalized_document_key",
            "key_normalization_applied",
            "key_normalization_warnings",
            "pdf_name",
            "page",
            "status",
            "failure_reason",
            "extraction_method",
            "detected_orientation_angle",
            "detected_orientation_class",
            "detected_rotation_base_angle",
            "detected_skew_angle",
            "detected_layout_type",
            "detected_column_count",
            "detected_has_columns",
            "runtime_ms",
            "confidence",
            "cer",
            "wer",
            "normalized_cer",
            "normalized_wer",
            "page_quality_score",
            "document_quality_score",
            "quality_class",
            "quality_reasons",
            "failed_gate",
            "gate_reason",
            "adapter_used",
            "recommended_preprocessing_profile",
            "applied_preprocessing_profile",
            "postprocess_quality_score",
            "lexicon_coverage",
            "unknown_token_rate",
            "protected_character_changes",
            "empty_output",
            "timed_out",
            "language_primary",
            "languages_present",
            "script_type",
            "layout_type",
            "document_type",
            "metric_family",
            "line_order_similarity",
            "paragraph_order_similarity",
            "header_footer_mismatch",
            "key_value_f1",
            "table_row_similarity",
            "structured_json_similarity",
            "entity_association_accuracy",
            "box_text_linking_quality",
            "page_equivalence_valid",
            "target_scope_valid",
            "text_length_ratio",
        ],
    )
    _write_csv(per_engine_csv, per_engine_rows, ["engine", "pages_seen", "runtime_ms_mean", "runtime_ms_p95"])
    _write_csv(per_language_csv, per_language_rows, ["language_primary", "pages", "cer_mean", "wer_mean"])
    _write_csv(per_layout_csv, per_layout_rows, ["layout_type", "pages", "cer_mean", "wer_mean"])
    _write_csv(
        per_detected_orientation_csv,
        per_detected_orientation_rows,
        ["detected_orientation_class", "pages", "cer_mean", "wer_mean"],
    )
    _write_csv(
        per_detected_layout_csv,
        per_detected_layout_rows,
        ["detected_layout_type", "pages", "cer_mean", "wer_mean"],
    )
    _write_confusion_matrix(confusion_json, confusion)

    baseline_report = _compare_with_baseline(summary_row, per_page_rows, args)

    outputs = {
        "evaluation_summary_csv": str(summary_csv),
        "per_page_metrics_csv": str(per_page_csv),
        "per_engine_metrics_csv": str(per_engine_csv),
        "per_language_metrics_csv": str(per_language_csv),
        "per_layout_metrics_csv": str(per_layout_csv),
        "per_detected_orientation_metrics_csv": str(per_detected_orientation_csv),
        "per_detected_layout_metrics_csv": str(per_detected_layout_csv),
        "confusion_matrix_json": str(confusion_json),
        "run_metadata_json": str(run_metadata_json),
    }
    if malformed_report_path is not None:
        outputs["malformed_rows_csv"] = str(malformed_report_path)

    tracking_outputs: dict[str, str] = {}
    tracking_error = ""
    if not args.disable_experiment_tracking:
        try:
            from production.experiment_tracking import build_tracking_dashboard

            tracking_outputs = build_tracking_dashboard(
                output_dir=output_dir,
                per_page_rows=per_page_rows,
                evaluation_summary_row=summary_row,
                ocr_csv_path=args.ocr_csv,
                gold_csv_path=args.gold_csv,
                baseline_dir=args.baseline_dir,
                baseline_summary_csv=args.baseline_summary_csv,
                baseline_per_page_csv=args.baseline_per_page_csv,
                top_n=max(1, int(args.top_n_regression_pages)),
                run_id=args.run_id,
                config_file=args.config_file,
                engine_versions_arg=args.engine_versions_json,
                model_versions_arg=args.model_versions_json,
                gold_set_version=args.gold_set_version,
                history_jsonl=args.experiment_history_jsonl,
            )
            outputs.update(tracking_outputs)
        except Exception as exc:  # pragma: no cover - defensive fallback
            tracking_error = str(exc)

    run_metadata = _build_run_metadata(args, outputs, summary_row, baseline_report, started_at)
    run_metadata["experiment_tracking"] = {
        "enabled": not args.disable_experiment_tracking,
        "outputs": tracking_outputs,
        "error": tracking_error,
    }
    with run_metadata_json.open("w", encoding="utf-8") as fh:
        json.dump(run_metadata, fh, ensure_ascii=True, indent=2)

    if args.output:
        _write_compat_output(Path(args.output), per_page_rows)

    print(
        "Evaluation complete | "
        f"matched={summary_row['matched_pages']} "
        f"CER={summary_row['cer_mean']:.4f} "
        f"WER={summary_row['wer_mean']:.4f} "
        f"empty_rate={summary_row['empty_output_rate']:.4f} "
        f"timeout_rate={summary_row['timeout_rate']:.4f}"
    )
    print(f"Artifacts written to {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
