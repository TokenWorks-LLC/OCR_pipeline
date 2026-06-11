from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_PAGE_NUMBER_RE = re.compile(r"(?:^|[\/_\-])page[\/_\-]*(\d+)(?:$|[\/_\-])", flags=re.IGNORECASE)


@dataclass
class KeyProvenance:
    original_page_key: str
    normalized_page_key: str
    original_document_key: str
    normalized_document_key: str
    key_normalization_applied: bool
    key_normalization_warnings: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_page_key": self.original_page_key,
            "normalized_page_key": self.normalized_page_key,
            "original_document_key": self.original_document_key,
            "normalized_document_key": self.normalized_document_key,
            "key_normalization_applied": self.key_normalization_applied,
            "key_normalization_warnings": self.key_normalization_warnings,
        }


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_int(value: Any) -> int | None:
    try:
        text = _clean_text(value)
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _normalize_identifier_token(value: str, unicode_form: str = "NFKC") -> str:
    if not value:
        return ""

    normalized = unicodedata.normalize(unicode_form, value)
    normalized = normalized.casefold()
    normalized = normalized.replace("\\", "/")
    normalized = re.sub(r"/+", "/", normalized)

    # Strip combining marks so accent variants map to the same join key.
    decomp = unicodedata.normalize("NFKD", normalized)
    decomp = "".join(char for char in decomp if not unicodedata.combining(char))

    output_chars: list[str] = []
    for char in decomp:
        if char.isalnum():
            output_chars.append(char)
        else:
            output_chars.append("_")

    normalized = re.sub(r"_+", "_", "".join(output_chars)).strip("_")
    return normalized


def _extract_page_number(value: str) -> int | None:
    if not value:
        return None
    match = _PAGE_NUMBER_RE.search(value)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _derive_document_fragment_from_page_key(page_key: str) -> str:
    if not page_key:
        return ""
    match = _PAGE_NUMBER_RE.search(page_key)
    if match is None:
        return page_key
    prefix = page_key[: match.start()]
    return prefix.rstrip("_/- ")


def _normalize_document_key(value: str, *, unicode_form: str, warnings: list[str]) -> str:
    if not value:
        warnings.append("missing_document_key")
        return ""

    source = value.replace("\\", "/")
    basename = source.split("/")[-1]
    if basename != value:
        warnings.append("document_key_path_normalized")

    stem = Path(basename).stem
    if stem != basename:
        warnings.append("document_extension_stripped")

    normalized = _normalize_identifier_token(stem, unicode_form=unicode_form)
    if not normalized:
        warnings.append("document_key_empty_after_normalization")
    return normalized


def build_key_provenance(
    *,
    page_key: Any,
    document_key: Any,
    page: Any,
    page_reference: Any = "",
    unicode_form: str = "NFKC",
) -> KeyProvenance:
    warnings: list[str] = []

    original_page_key = _clean_text(page_key)
    original_document_key = _clean_text(document_key)
    original_page_reference = _clean_text(page_reference)
    numeric_page = _safe_int(page)

    if not original_page_key and original_page_reference:
        warnings.append("missing_page_key_used_page_reference")

    page_from_key = _extract_page_number(original_page_key)
    if numeric_page is None and page_from_key is not None:
        numeric_page = page_from_key
        warnings.append("page_number_derived_from_page_key")
    elif numeric_page is not None and page_from_key is not None and numeric_page != page_from_key:
        warnings.append(f"page_number_mismatch:{numeric_page}!={page_from_key}")

    if not original_document_key and original_page_key:
        original_document_key = _derive_document_fragment_from_page_key(original_page_key)
        if original_document_key:
            warnings.append("document_key_derived_from_page_key")

    normalized_document_key = _normalize_document_key(
        original_document_key,
        unicode_form=unicode_form,
        warnings=warnings,
    )

    normalized_page_from_source = _normalize_identifier_token(original_page_key, unicode_form=unicode_form)

    normalized_page_key = ""
    if normalized_document_key and numeric_page is not None:
        normalized_page_key = f"{normalized_document_key}_page_{numeric_page}"
        if normalized_page_from_source and normalized_page_from_source != normalized_page_key:
            warnings.append("page_key_rebuilt_from_document_and_page")
    elif normalized_document_key and original_page_reference:
        normalized_reference = _normalize_identifier_token(original_page_reference, unicode_form=unicode_form)
        normalized_page_key = (
            f"{normalized_document_key}_{normalized_reference}"
            if normalized_reference
            else normalized_document_key
        )
    elif normalized_page_from_source:
        normalized_page_key = normalized_page_from_source
    elif normalized_document_key:
        normalized_page_key = normalized_document_key
        warnings.append("missing_page_key_fell_back_to_document_key")
    else:
        warnings.append("missing_page_and_document_key")

    if not original_page_key and normalized_page_key:
        warnings.append("page_key_reconstructed")

    unique_warnings: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        if not warning or warning in seen:
            continue
        seen.add(warning)
        unique_warnings.append(warning)

    key_normalization_applied = (
        normalized_page_key != original_page_key
        or normalized_document_key != original_document_key
        or bool(unique_warnings)
    )

    return KeyProvenance(
        original_page_key=original_page_key,
        normalized_page_key=normalized_page_key,
        original_document_key=original_document_key,
        normalized_document_key=normalized_document_key,
        key_normalization_applied=key_normalization_applied,
        key_normalization_warnings="|".join(unique_warnings),
    )


def normalize_join_page_key(
    *,
    page_key: Any,
    document_key: Any,
    page: Any,
    page_reference: Any = "",
    unicode_form: str = "NFKC",
) -> str:
    provenance = build_key_provenance(
        page_key=page_key,
        document_key=document_key,
        page=page,
        page_reference=page_reference,
        unicode_form=unicode_form,
    )
    return provenance.normalized_page_key
