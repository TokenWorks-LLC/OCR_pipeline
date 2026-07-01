"""Rules-based citation / bibliography parser for OCR post-processing.

This module turns the (often noisy) text of a bibliography or reference block
into structured citation records. It is intentionally dependency-free and
diacritic-safe so it can run inside the pipeline's postprocessing stage on the
same machines that already have only ``pymupdf`` / ``Pillow`` installed.

Design goals (consistent with the rest of the pipeline):

- Preserve the raw text; never mutate the source reference.
- Extract high-precision fields first (DOI, URL, year, pages, volume), then fall
  back to fuzzier author/title attribution.
- Emit a per-field and an overall confidence, plus the unparsed remainder, so
  low-confidence records can be routed to human review instead of silently
  shipped.

It handles two broad citation styles seen in the gold corpus:

- Western academic: ``Author, F. (2008). Title. Journal 35, 21-32.``
- Assyriological / Turkish: ``Bilgiç, DTCFD VI, 5, s. 506`` (journal
  abbreviations, Roman-numeral volumes, Turkish ``s.`` = sayfa/page), including
  dense ``;``-separated citation lists.

This is a heuristic first pass, not a replacement for a trained citation model.
See ``docs/CITATION_PARSING.md`` for the confidence model and the upgrade path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata


METHOD_VERSION = "rules_v1"

STYLE_WESTERN = "western"
STYLE_ASSYRIOLOGICAL = "assyriological"
STYLE_UNKNOWN = "unknown"

# Overall-confidence threshold below which a parse is flagged for human review.
REVIEW_CONFIDENCE_THRESHOLD = 0.6

# Turkish + common Latin diacritics appear throughout the corpus; keep them.
_DASH_CHARS = "‐‑‒–—―-"
_DASH_CLASS = f"[{_DASH_CHARS}]"

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)

# A 4-digit year in 1500-2099, optionally with a disambiguation letter (1999a)
# and optionally parenthesised. Parenthesised matches are preferred.
_YEAR_PAREN_RE = re.compile(r"\((1[5-9]\d{2}|20\d{2})([a-z])?\)")
_YEAR_BARE_RE = re.compile(r"(?<!\d)(1[5-9]\d{2}|20\d{2})([a-z])?(?!\d)")

# Page ranges: English "pp. 21-32" / "p. 12", Turkish "s. 21-32", German "S. 95",
# or a bare trailing "21-32".
_PAGES_LABELLED_RE = re.compile(
    rf"\b(?:pp?|ss?|S|Seiten?)\.?\s*(\d{{1,4}})(?:\s*{_DASH_CLASS}\s*(\d{{1,4}}))?",
)
_PAGES_BARE_RE = re.compile(rf"\b(\d{{1,4}})\s*{_DASH_CLASS}\s*(\d{{1,4}})\b")

# Explicit volume markers: "vol. 3", "Band. 6", "Bd 6".
_VOLUME_LABELLED_RE = re.compile(
    r"\b(?:vol|Vol|Volume|Band|Bd)\.?\s*(\d{1,4})(?:\s*[/-]\s*(\d{1,3}))?",
)
# Journal/series abbreviation followed by a volume: "AOF 35-1", "ArOr XVIII 1-2",
# "UHKB 5", "OrNS 52". The abbreviation token must carry >= 2 uppercase letters
# so ordinary Title-Case words ("Alten", "Orient") are not misread as journals.
_ABBREV_VOLUME_RE = re.compile(
    r"\b([A-Za-zÇĞİÖŞÜçğıöşü]{2,10})\s+((?:[IVXLCDM]{1,6}|\d{1,4})(?:\s*[-/]\s*\d{1,3})?)\b",
)

_ROMAN_RE = re.compile(r"^[IVXLCDM]{1,6}$")
_WORD_RE = re.compile(r"\w[\w.'’-]*", re.UNICODE)


def normalize_reference_text(text: str) -> str:
    """NFC-normalize and collapse whitespace without dropping diacritics."""
    normalized = unicodedata.normalize("NFC", str(text or ""))
    normalized = normalized.replace(" ", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def _count_uppercase(token: str) -> int:
    return sum(1 for ch in token if ch.isalpha() and ch.isupper())


def _looks_like_journal_abbrev(token: str) -> bool:
    stripped = token.strip(".,;:")
    if len(stripped) < 2:
        return False
    return _count_uppercase(stripped) >= 2


@dataclass(frozen=True)
class ParsedCitation:
    """A single structured citation with provenance and confidence."""

    raw_text: str
    authors: tuple[str, ...] = ()
    year: str | None = None
    title: str | None = None
    source: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    url: str | None = None
    unparsed: str = ""
    style: str = STYLE_UNKNOWN
    method: str = METHOD_VERSION
    field_confidence: dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 0.0
    needs_review: bool = True

    def to_dict(self) -> dict[str, object]:
        """Flat, JSON/CSV-friendly representation."""
        return {
            "raw_text": self.raw_text,
            "authors": list(self.authors),
            "authors_joined": "; ".join(self.authors),
            "year": self.year or "",
            "title": self.title or "",
            "source": self.source or "",
            "volume": self.volume or "",
            "issue": self.issue or "",
            "pages": self.pages or "",
            "doi": self.doi or "",
            "url": self.url or "",
            "unparsed": self.unparsed,
            "style": self.style,
            "method": self.method,
            "overall_confidence": round(float(self.overall_confidence), 4),
            "needs_review": bool(self.needs_review),
            "field_confidence": {k: round(float(v), 4) for k, v in self.field_confidence.items()},
        }


class _Consumed:
    """Track which character spans of the working string are already attributed."""

    def __init__(self, length: int) -> None:
        self._mask = [False] * length

    def take(self, start: int, end: int) -> None:
        for i in range(max(0, start), min(len(self._mask), end)):
            self._mask[i] = True

    def is_free(self, start: int, end: int) -> bool:
        return not any(self._mask[i] for i in range(max(0, start), min(len(self._mask), end)))

    def free_text(self, text: str) -> str:
        chars = [text[i] if not self._mask[i] else " " for i in range(len(text))]
        return re.sub(r"\s+", " ", "".join(chars)).strip(" ,;.-")


def _normalize_pages(start: str, end: str | None) -> str:
    return f"{start}-{end}" if end else start


def _detect_style(text: str) -> str:
    has_abbrev_vol = bool(_ABBREV_VOLUME_RE.search(text))
    has_turkish_pages = bool(re.search(r"\bs\.\s*\d", text))
    has_paren_year = bool(_YEAR_PAREN_RE.search(text))
    if has_paren_year and not has_turkish_pages:
        return STYLE_WESTERN
    if has_abbrev_vol or has_turkish_pages:
        return STYLE_ASSYRIOLOGICAL
    return STYLE_UNKNOWN


def parse_reference(text: str) -> ParsedCitation:
    """Parse a single reference string into a :class:`ParsedCitation`.

    Extraction is high-precision-first: DOI/URL, then year, pages, volume and
    journal abbreviation, and finally author and title from what remains. Each
    field carries a confidence; the overall confidence and ``needs_review`` flag
    let downstream review queues prioritise uncertain parses.
    """
    raw = str(text or "")
    working = normalize_reference_text(raw)
    # Drop a leading numbered-bibliography label ("12." / "[3]") so it does not
    # pollute the author field; the raw text is preserved untouched.
    working = re.sub(r"^\[?\d{1,3}[\].]\s+", "", working)
    if not working:
        return ParsedCitation(raw_text=raw, unparsed="", overall_confidence=0.0, needs_review=True)

    consumed = _Consumed(len(working))
    conf: dict[str, float] = {}

    doi = None
    url = None
    year = None
    pages = None
    source = None
    volume = None
    issue = None

    doi_match = _DOI_RE.search(working)
    if doi_match:
        doi = doi_match.group(0).rstrip(".,;)")
        consumed.take(doi_match.start(), doi_match.start() + len(doi))
        conf["doi"] = 0.99

    url_match = _URL_RE.search(working)
    if url_match and (doi is None or not url_match.group(0).lower().startswith("http") or "doi.org" not in url_match.group(0).lower()):
        url = url_match.group(0).rstrip(".,;)")
        consumed.take(url_match.start(), url_match.start() + len(url))
        conf["url"] = 0.9

    year_match = _YEAR_PAREN_RE.search(working)
    year_conf = 0.95
    if not year_match:
        year_match = _YEAR_BARE_RE.search(working)
        year_conf = 0.8
    if year_match:
        suffix = year_match.group(2) or ""
        year = f"{year_match.group(1)}{suffix}"
        consumed.take(year_match.start(), year_match.end())
        conf["year"] = year_conf

    # Journal/series abbreviation + volume (Assyriological style).
    for match in _ABBREV_VOLUME_RE.finditer(working):
        abbrev, vol = match.group(1), match.group(2)
        if not _looks_like_journal_abbrev(abbrev):
            continue
        if not consumed.is_free(match.start(), match.end()):
            continue
        source = abbrev.strip(".,;:")
        vol_clean = re.sub(r"\s+", "", vol)
        vol_parts = re.split(r"[-/]", vol_clean, maxsplit=1)
        volume = vol_parts[0]
        if len(vol_parts) > 1:
            issue = vol_parts[1]
        consumed.take(match.start(), match.end())
        conf["source"] = 0.8
        conf["volume"] = 0.85 if not _ROMAN_RE.match(volume) else 0.8
        if issue:
            conf["issue"] = 0.7
        break

    if volume is None:
        vol_match = _VOLUME_LABELLED_RE.search(working)
        if vol_match and consumed.is_free(vol_match.start(), vol_match.end()):
            volume = vol_match.group(1)
            if vol_match.group(2):
                issue = vol_match.group(2)
            consumed.take(vol_match.start(), vol_match.end())
            conf["volume"] = 0.85

    # Pages: prefer a labelled range, else a bare trailing range.
    page_match = None
    page_conf = 0.0
    for m in _PAGES_LABELLED_RE.finditer(working):
        if consumed.is_free(m.start(), m.end()):
            page_match, page_conf = m, 0.9
            break
    if page_match is None:
        for m in _PAGES_BARE_RE.finditer(working):
            if consumed.is_free(m.start(), m.end()):
                page_match, page_conf = m, 0.6
                break
    if page_match is not None:
        pages = _normalize_pages(page_match.group(1), page_match.group(2))
        consumed.take(page_match.start(), page_match.end())
        conf["pages"] = page_conf

    # Authors: the free text before the year (or before the journal/volume when
    # there is no year). Split on ';' and 'and'/'&' first.
    authors: tuple[str, ...] = ()
    boundary = None
    if year_match is not None:
        boundary = year_match.start()
    elif source is not None:
        abbrev_pos = working.find(source)
        boundary = abbrev_pos if abbrev_pos > 0 else None
    if boundary:
        author_zone = working[:boundary]
        authors = _extract_authors(author_zone)
        if authors:
            conf["authors"] = _author_confidence(author_zone, authors)

    style = _detect_style(working)

    # Title: the longest remaining free run, typically between the author/year
    # prefix and the journal/volume. Only trust it when it has real word content.
    remainder = consumed.free_text(working)
    title = None
    if authors:
        # Drop a leading fragment that merely repeats author text.
        author_prefix = "; ".join(authors)
        if remainder.lower().startswith(author_prefix.lower()[:10]):
            remainder = remainder[len(author_prefix):].strip(" ,;.-") or remainder
    title_candidate = _pick_title(remainder)
    if title_candidate:
        title = title_candidate
        conf["title"] = _title_confidence(title_candidate)

    unparsed = _residual_after_title(remainder, title)

    overall = _overall_confidence(conf, working, unparsed)
    needs_review = overall < REVIEW_CONFIDENCE_THRESHOLD or (year is None and not authors)

    return ParsedCitation(
        raw_text=raw,
        authors=authors,
        year=year,
        title=title,
        source=source,
        volume=volume,
        issue=issue,
        pages=pages,
        doi=doi,
        url=url,
        unparsed=unparsed,
        style=style,
        field_confidence=conf,
        overall_confidence=overall,
        needs_review=needs_review,
    )


def _extract_authors(zone: str) -> tuple[str, ...]:
    zone = zone.strip(" ,;.-")
    if not zone:
        return ()
    # Split multi-author lists on ';' or ' and '/' & '.
    parts = re.split(r"\s*;\s*|\s+&\s+|\s+and\s+", zone)
    authors: list[str] = []
    for part in parts:
        part = part.strip(" ,.-")
        if not part:
            continue
        # A "Surname, Initials" or "Surname, Firstname" chunk, or a bare surname.
        surname_match = re.match(r"([^\d,]+?),\s*([A-ZÇĞİÖŞÜ][\w.'’-]*)", part)
        if surname_match:
            surname = surname_match.group(1).strip()
            initials = surname_match.group(2).strip()
            authors.append(f"{surname}, {initials}")
            continue
        words = _WORD_RE.findall(part)
        if 1 <= len(words) <= 4 and any(_count_uppercase(w) >= 1 for w in words):
            authors.append(part.strip())
    return tuple(authors)


def _author_confidence(zone: str, authors: tuple[str, ...]) -> float:
    if not authors:
        return 0.0
    commas = sum(1 for a in authors if "," in a)
    base = 0.5 + 0.4 * (commas / len(authors))
    if len(zone) > 120:  # implausibly long author zone -> likely swallowed title
        base -= 0.2
    return max(0.2, min(0.95, base))


def _clean_title(title: str) -> str:
    # Collapse punctuation islands left behind when structured fields were
    # masked out of the middle of the string (e.g. "Orient- , , Heidelberg").
    title = re.sub(r"\s*[,;]\s*(?=[,;])", "", title)
    title = re.sub(r"\s{2,}", " ", title)
    title = re.sub(r"[\s,;]+$", "", title)
    return title.strip(" ,;.-")


def _pick_title(remainder: str) -> str | None:
    remainder = remainder.strip(" ,;.-")
    if not remainder:
        return None
    # Split on strong separators and keep the longest word-bearing segment.
    segments = [_clean_title(s) for s in re.split(r"\s*[.;]\s+", remainder)]
    segments = [s for s in segments if len(_WORD_RE.findall(s)) >= 2]
    if not segments:
        return None
    return max(segments, key=lambda s: len(_WORD_RE.findall(s)))


def _title_confidence(title: str) -> float:
    words = _WORD_RE.findall(title)
    if len(words) >= 4:
        return 0.6
    if len(words) >= 2:
        return 0.4
    return 0.2


def _residual_after_title(remainder: str, title: str | None) -> str:
    if not remainder:
        return ""
    if title and title in remainder:
        remainder = remainder.replace(title, " ", 1)
    return re.sub(r"\s+", " ", remainder).strip(" ,;.-")


def _overall_confidence(conf: dict[str, float], working: str, unparsed: str) -> float:
    if not conf:
        return 0.0
    # Weight the structurally reliable fields most heavily.
    weights = {
        "doi": 1.0,
        "year": 0.9,
        "pages": 0.8,
        "volume": 0.7,
        "source": 0.7,
        "authors": 0.8,
        "title": 0.5,
        "issue": 0.3,
        "url": 0.4,
    }
    num = sum(weights.get(k, 0.3) * v for k, v in conf.items())
    den = sum(weights.get(k, 0.3) for k in conf)
    score = num / den if den else 0.0
    # Penalise a large unattributed remainder.
    if working:
        unparsed_ratio = len(unparsed) / max(1, len(working))
        score *= 1.0 - 0.4 * min(1.0, unparsed_ratio)
    # Reward breadth of coverage: a citation with many recovered fields is safer.
    coverage_bonus = min(0.1, 0.02 * len(conf))
    return max(0.0, min(1.0, score + coverage_bonus))


def split_references(block_text: str) -> list[str]:
    """Split a bibliography block into individual reference strings.

    Uses two signals: line breaks that start a new reference (a new author or a
    hanging-indent entry), and ``;``-separated dense inline citation lists that
    have no line breaks.
    """
    text = unicodedata.normalize("NFC", str(block_text or ""))
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    if len(lines) <= 1:
        single = lines[0] if lines else text.strip()
        if single.count(";") >= 2 and _looks_like_inline_citation_list(single):
            return [seg.strip(" ;") for seg in single.split(";") if seg.strip(" ;")]
        return [single] if single else []

    # Merge continuation lines (that clearly do not start a new entry) upward.
    references: list[str] = []
    for line in lines:
        if references and _is_continuation(line):
            references[-1] = f"{references[-1]} {line}".strip()
        else:
            references.append(line)
    return references


def _looks_like_inline_citation_list(text: str) -> bool:
    segments = [s for s in text.split(";") if s.strip()]
    if len(segments) < 2:
        return False
    hits = sum(1 for s in segments if _YEAR_BARE_RE.search(s) or _ABBREV_VOLUME_RE.search(s))
    return hits >= max(2, len(segments) // 2)


def _is_continuation(line: str) -> bool:
    if not line:
        return False
    # A wrapped tail such as a bare page range ("21-32.") continues the entry.
    if re.match(rf"^\(?\d{{1,4}}\s*{_DASH_CLASS}\s*\d{{1,4}}\)?[.,;]?$", line):
        return True
    first = line[0]
    # A new entry usually starts with an uppercase author name.
    if first.isalpha() and first.islower():
        return True
    if first in "([" or first in _DASH_CHARS:
        return True
    # A numbered-bibliography label ("12." / "[12]") before an uppercase name is
    # a NEW entry; any other digit-leading line is a wrapped continuation.
    numbered = re.match(r"^\[?\d{1,3}[\].]\s+(\S)", line)
    if numbered and numbered.group(1).isupper():
        return False
    if first.isdigit():
        return True
    return False


def parse_bibliography_block(text: str) -> list[ParsedCitation]:
    """Split a bibliography block and parse each reference."""
    return [parse_reference(ref) for ref in split_references(text)]
