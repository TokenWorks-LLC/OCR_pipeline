from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


_SPACE_TRANSLATION = {
    ord("\u00a0"): " ",
    ord("\u1680"): " ",
    ord("\u180e"): " ",
    ord("\u2000"): " ",
    ord("\u2001"): " ",
    ord("\u2002"): " ",
    ord("\u2003"): " ",
    ord("\u2004"): " ",
    ord("\u2005"): " ",
    ord("\u2006"): " ",
    ord("\u2007"): " ",
    ord("\u2008"): " ",
    ord("\u2009"): " ",
    ord("\u200a"): " ",
    ord("\u202f"): " ",
    ord("\u205f"): " ",
    ord("\u3000"): " ",
}


_PUNCT_REPEATED_RE = re.compile(r"^([^\w\s])\1{5,}$", re.UNICODE)
_ASCII_NOISE_RE = re.compile(r"^[`~^_|]{4,}$")
_SLASH_NOISE_RE = re.compile(r"^[\\/|]{6,}$")


@dataclass(frozen=True)
class CleanupResult:
    raw_text: str
    unicode_normalized_text: str
    whitespace_normalized_text: str
    cleaned_text: str
    removed_token_count: int


def normalize_unicode(text: str, form: str = "NFC") -> str:
    return unicodedata.normalize(form, str(text or ""))


def _strip_disallowed_controls(text: str) -> str:
    filtered: list[str] = []
    for char in str(text or ""):
        if char in {"\n", "\t"}:
            filtered.append(char)
            continue
        category = unicodedata.category(char)
        if category == "Cc":
            continue
        filtered.append(char)
    return "".join(filtered)


def normalize_whitespace(text: str) -> str:
    value = str(text or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.translate(_SPACE_TRANSLATION)

    normalized_lines: list[str] = []
    for line in value.split("\n"):
        compact = re.sub(r"[ \t\f\v]+", " ", line).strip()
        normalized_lines.append(compact)

    result = "\n".join(normalized_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def _is_obvious_garbage_token(token: str) -> bool:
    text = str(token or "")
    if not text:
        return False
    if text.count("\ufffd") >= 2:
        return True
    if _ASCII_NOISE_RE.match(text):
        return True
    if _SLASH_NOISE_RE.match(text):
        return True

    punct_match = _PUNCT_REPEATED_RE.match(text)
    if punct_match:
        punct = punct_match.group(1)
        if punct not in "[](){}<>":
            return True

    return False


def remove_obvious_ocr_garbage(text: str) -> tuple[str, int]:
    cleaned_lines: list[str] = []
    removed_count = 0

    for line in str(text or "").split("\n"):
        kept_tokens: list[str] = []
        for token in line.split(" "):
            if _is_obvious_garbage_token(token):
                removed_count += 1
                continue
            kept_tokens.append(token)

        compact_line = " ".join(tok for tok in kept_tokens if tok)
        cleaned_lines.append(compact_line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip(), removed_count


def general_cleanup(text: str) -> CleanupResult:
    raw = str(text or "")
    unicode_normalized = normalize_unicode(raw, form="NFC")
    no_controls = _strip_disallowed_controls(unicode_normalized)
    whitespace_normalized = normalize_whitespace(no_controls)
    garbage_filtered, removed = remove_obvious_ocr_garbage(whitespace_normalized)

    return CleanupResult(
        raw_text=raw,
        unicode_normalized_text=unicode_normalized,
        whitespace_normalized_text=whitespace_normalized,
        cleaned_text=garbage_filtered,
        removed_token_count=removed,
    )
