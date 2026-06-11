from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re
from typing import Any

from .lexicon import InMemoryLexicon


@dataclass(frozen=True)
class CorrectionAudit:
    token_before: str
    token_after: str
    reason: str
    source: str
    confidence: float
    start: int
    end: int
    suspicious: bool = False


class LanguageAdapter:
    name = "default_latin"
    lexicon_domain = "general"
    allow_rule_corrections = True
    unknown_review_threshold = 0.42
    suspicious_review_threshold = 0.35

    word_pattern = re.compile(
        r"[A-Za-z\u00c0-\u024f\u1e00-\u1eff0-9]+(?:[-'][A-Za-z\u00c0-\u024f\u1e00-\u1eff0-9]+)*",
        re.UNICODE,
    )

    protected_characters: set[str] = set()
    protected_token_patterns: list[re.Pattern[str]] = []

    common_confusions: dict[str, str] = {
        "0": "o",
        "1": "l",
        "5": "s",
        "8": "B",
        "rn": "m",
        "vv": "w",
    }

    def tokenize(self, text: str) -> list[tuple[str, int, int]]:
        return [(match.group(0), match.start(), match.end()) for match in self.word_pattern.finditer(str(text or ""))]

    def is_protected_token(self, token: str) -> bool:
        text = str(token or "")
        if not text:
            return False
        if any(char in self.protected_characters for char in text):
            return True
        return any(pattern.search(text) for pattern in self.protected_token_patterns)

    @staticmethod
    def _preserve_case(source: str, target: str) -> str:
        if source.isupper():
            return target.upper()
        if source[:1].isupper() and source[1:].islower():
            return target.capitalize()
        return target

    def generate_confusion_candidates(self, token: str) -> list[str]:
        candidates: set[str] = set()
        for wrong, right in self.common_confusions.items():
            if wrong in token and wrong != right:
                candidates.add(token.replace(wrong, right, 1))
        return sorted(candidates)

    def suggest_token_correction(self, token: str, lexicon: InMemoryLexicon) -> tuple[str, str, float, bool] | None:
        if lexicon.lookup_exact(token, domain=self.lexicon_domain) or lexicon.lookup_normalized(token, domain=self.lexicon_domain):
            return None

        for candidate in self.generate_confusion_candidates(token):
            if lexicon.lookup_exact(candidate, domain=self.lexicon_domain) or lexicon.lookup_normalized(candidate, domain=self.lexicon_domain):
                preserved = self._preserve_case(token, candidate)
                return preserved, "confusion_map", 0.90, False

        suggestions = lexicon.suggest(
            token,
            max_distance=1,
            max_results=5,
            domain=self.lexicon_domain,
        )
        if not suggestions:
            return None

        selected = suggestions[0]
        confidence = max(0.0, min(1.0, 1.0 - (selected.distance / max(len(token), len(selected.token), 1))))
        if selected.distance > 1 or confidence < 0.70:
            return None

        preserved = self._preserve_case(token, selected.token)
        suspicious = bool(selected.distance > 0 and confidence < 0.80)
        return preserved, "lexicon_suggestion", confidence, suspicious

    def apply_rule_corrections(
        self,
        text: str,
        lexicon: InMemoryLexicon,
        max_corrections: int = 80,
        edit_budget_ratio: float = 0.05,
    ) -> tuple[str, list[CorrectionAudit], int]:
        if not self.allow_rule_corrections:
            return str(text or ""), [], 0

        source = str(text or "")
        if not source:
            return source, [], 0

        token_spans = self.tokenize(source)
        if not token_spans:
            return source, [], 0

        max_edit_chars = max(1, int(round(len(source) * max(float(edit_budget_ratio or 0.0), 0.0))))
        corrections: list[CorrectionAudit] = []
        replacements: dict[tuple[int, int], str] = {}
        spent_budget = 0

        for token, start, end in token_spans:
            if len(corrections) >= max_corrections:
                break
            if len(token) < 3:
                continue
            if self.is_protected_token(token):
                continue

            suggestion = self.suggest_token_correction(token, lexicon)
            if suggestion is None:
                continue

            corrected_token, reason, confidence, suspicious = suggestion
            if corrected_token == token:
                continue

            incremental_cost = abs(len(corrected_token) - len(token)) + 1
            if spent_budget + incremental_cost > max_edit_chars:
                break

            spent_budget += incremental_cost
            replacements[(start, end)] = corrected_token
            corrections.append(
                CorrectionAudit(
                    token_before=token,
                    token_after=corrected_token,
                    reason=reason,
                    source="rule",
                    confidence=round(float(confidence), 6),
                    start=start,
                    end=end,
                    suspicious=suspicious,
                )
            )

        if not replacements:
            return source, corrections, spent_budget

        rebuilt: list[str] = []
        cursor = 0
        for token, start, end in token_spans:
            rebuilt.append(source[cursor:start])
            rebuilt.append(replacements.get((start, end), token))
            cursor = end
        rebuilt.append(source[cursor:])

        return "".join(rebuilt), corrections, spent_budget

    def word_tokens(self, text: str) -> list[str]:
        return [token for token, _, _ in self.tokenize(text)]

    def count_protected_character_changes(self, before: str, after: str) -> int:
        if not self.protected_characters:
            return 0

        left_counts = Counter(char for char in str(before or "") if char in self.protected_characters)
        right_counts = Counter(char for char in str(after or "") if char in self.protected_characters)

        changed = 0
        for key in set(left_counts.keys()) | set(right_counts.keys()):
            changed += abs(int(left_counts.get(key, 0)) - int(right_counts.get(key, 0)))
        return changed

    def compute_quality_metrics(
        self,
        cleaned_text: str,
        corrected_text: str,
        lexicon: InMemoryLexicon,
        corrections: list[CorrectionAudit],
    ) -> dict[str, Any]:
        tokens = self.word_tokens(corrected_text)
        lexicon_coverage, unknown_rate, unknown_tokens = lexicon.coverage(tokens, domain=self.lexicon_domain)

        suspicious_count = sum(1 for correction in corrections if correction.suspicious)
        suspicious_rate = (suspicious_count / len(corrections)) if corrections else 0.0
        protected_changes = self.count_protected_character_changes(cleaned_text, corrected_text)

        protected_score = 1.0 if protected_changes == 0 else max(0.0, 1.0 - (protected_changes / 4.0))
        quality_score = (
            (lexicon_coverage * 0.45)
            + ((1.0 - unknown_rate) * 0.25)
            + ((1.0 - min(1.0, suspicious_rate)) * 0.15)
            + (protected_score * 0.15)
        )
        quality_score = float(max(0.0, min(quality_score, 1.0)))

        return {
            "lexicon_coverage": round(float(lexicon_coverage), 6),
            "unknown_token_rate": round(float(unknown_rate), 6),
            "unknown_tokens": unknown_tokens[:120],
            "suspicious_correction_rate": round(float(suspicious_rate), 6),
            "protected_character_changes": int(protected_changes),
            "quality_score": round(float(quality_score), 6),
        }


class DefaultLatinAdapter(LanguageAdapter):
    name = "default_latin"
    lexicon_domain = "general"


class EnglishAdapter(LanguageAdapter):
    name = "english"
    lexicon_domain = "english"


class GermanAdapter(LanguageAdapter):
    name = "german"
    lexicon_domain = "german"
    protected_characters = {
        "\u00e4",
        "\u00f6",
        "\u00fc",
        "\u00df",
        "\u00c4",
        "\u00d6",
        "\u00dc",
    }


class FrenchAdapter(LanguageAdapter):
    name = "french"
    lexicon_domain = "french"
    protected_characters = {
        "\u00e0",
        "\u00e2",
        "\u00e7",
        "\u00e9",
        "\u00e8",
        "\u00ea",
        "\u00eb",
        "\u00ee",
        "\u00ef",
        "\u00f4",
        "\u00f9",
        "\u00fb",
        "\u00fc",
        "\u00ff",
        "\u0153",
        "\u00e6",
        "\u00c0",
        "\u00c2",
        "\u00c7",
        "\u00c9",
        "\u00c8",
        "\u00ca",
        "\u00cb",
        "\u00ce",
        "\u00cf",
        "\u00d4",
        "\u00d9",
        "\u00db",
        "\u0178",
        "\u0152",
        "\u00c6",
    }


class ArabicOrRTLPlaceholderAdapter(LanguageAdapter):
    name = "arabic_or_rtl_placeholder"
    lexicon_domain = "arabic"
    allow_rule_corrections = False

    word_pattern = re.compile(
        r"[\u0590-\u05ff\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff0-9]+(?:[-'][\u0590-\u05ff\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff0-9]+)*",
        re.UNICODE,
    )


class ScholarlyTransliterationAdapter(LanguageAdapter):
    name = "scholarly_transliteration"
    lexicon_domain = "transliteration"

    word_pattern = re.compile(
        r"[A-Za-z0-9\u0101\u0113\u012b\u016b\u0161\u1e63\u1e6d\u1e2b\u2080-\u2089]+(?:-[A-Za-z0-9\u0101\u0113\u012b\u016b\u0161\u1e63\u1e6d\u1e2b\u2080-\u2089]+)*",
        re.UNICODE,
    )

    protected_characters = {
        "\u0161",
        "\u1e63",
        "\u1e6d",
        "\u1e2b",
        "\u012b",
        "\u016b",
        "\u0101",
        "\u0113",
        "[",
        "]",
        "(",
        ")",
        "{",
        "}",
        "<",
        ">",
    }

    protected_token_patterns = [
        re.compile(r"\[[^\]]*\]", re.UNICODE),
        re.compile(r"<[^>]*>", re.UNICODE),
        re.compile(r"\b[A-Za-z\u0101\u0113\u012b\u016b\u0161\u1e63\u1e6d\u1e2b]+(?:-[A-Za-z0-9\u0101\u0113\u012b\u016b\u0161\u1e63\u1e6d\u1e2b]+)+\b", re.UNICODE),
    ]

    common_confusions = {
        "0": "o",
        "1": "l",
        "s\u030c": "\u0161",
    }

    def is_protected_token(self, token: str) -> bool:
        if super().is_protected_token(token):
            return True
        text = str(token or "")
        if "-" in text and any(char.isalpha() for char in text):
            return True
        return False


class AkkadianTransliterationAdapter(ScholarlyTransliterationAdapter):
    name = "akkadian_transliteration"
    lexicon_domain = "akkadian"
    unknown_review_threshold = 0.55

    transliteration_token_re = re.compile(
        r"^[A-Za-z0-9\u0101\u0113\u012b\u016b\u0161\u1e63\u1e6d\u1e2b\u2080-\u2089\[\]<>\-\.\?\*xX]+$",
        re.UNICODE,
    )

    diacritic_chars = {
        "\u0161",
        "\u1e63",
        "\u1e6d",
        "\u1e2b",
        "\u012b",
        "\u016b",
    }

    def compute_quality_metrics(
        self,
        cleaned_text: str,
        corrected_text: str,
        lexicon: InMemoryLexicon,
        corrections: list[CorrectionAudit],
    ) -> dict[str, Any]:
        base = super().compute_quality_metrics(
            cleaned_text=cleaned_text,
            corrected_text=corrected_text,
            lexicon=lexicon,
            corrections=corrections,
        )

        tokens = self.word_tokens(corrected_text)
        if tokens:
            valid_tokens = sum(1 for token in tokens if self.transliteration_token_re.match(token))
            validity_ratio = valid_tokens / float(len(tokens))
        else:
            validity_ratio = 1.0

        diac_before = Counter(char for char in str(cleaned_text or "") if char in self.diacritic_chars)
        diac_after = Counter(char for char in str(corrected_text or "") if char in self.diacritic_chars)
        diac_total = sum(diac_before.values())
        if diac_total <= 0:
            diacritic_preservation = 1.0
        else:
            changed = sum(abs(int(diac_before.get(ch, 0)) - int(diac_after.get(ch, 0))) for ch in self.diacritic_chars)
            diacritic_preservation = max(0.0, min(1.0, 1.0 - (changed / float(diac_total))))

        base_score = float(base.get("quality_score", 0.0) or 0.0)
        quality_score = (base_score * 0.60) + (validity_ratio * 0.20) + (diacritic_preservation * 0.20)

        base.update(
            {
                "transliteration_token_validity": round(float(validity_ratio), 6),
                "diacritic_preservation": round(float(diacritic_preservation), 6),
                "quality_score": round(float(max(0.0, min(quality_score, 1.0))), 6),
            }
        )
        return base


def adapter_registry() -> dict[str, LanguageAdapter]:
    return {
        "default_latin": DefaultLatinAdapter(),
        "english": EnglishAdapter(),
        "german": GermanAdapter(),
        "french": FrenchAdapter(),
        "arabic_or_rtl_placeholder": ArabicOrRTLPlaceholderAdapter(),
        "scholarly_transliteration": ScholarlyTransliterationAdapter(),
        "akkadian_transliteration": AkkadianTransliterationAdapter(),
    }


def _contains_arabic_script(text: str) -> bool:
    chars = [char for char in str(text or "") if not char.isspace()]
    if not chars:
        return False
    arabic_count = sum(1 for char in chars if "\u0600" <= char <= "\u06ff" or "\u0750" <= char <= "\u077f" or "\u08a0" <= char <= "\u08ff")
    return (arabic_count / float(len(chars))) >= 0.20


def _looks_like_transliteration(text: str) -> bool:
    value = str(text or "")
    if not value.strip():
        return False

    cue_patterns = [
        re.compile(r"\b[A-Za-z\u0101\u0113\u012b\u016b\u0161\u1e63\u1e6d\u1e2b]+(?:-[A-Za-z0-9\u0101\u0113\u012b\u016b\u0161\u1e63\u1e6d\u1e2b]+)+\b", re.UNICODE),
        re.compile(r"[\u0161\u1e63\u1e6d\u1e2b\u012b\u016b]", re.UNICODE),
        re.compile(r"\[[^\]]*\]", re.UNICODE),
    ]

    return any(pattern.search(value) for pattern in cue_patterns)


def select_adapter_name(
    text: str,
    language_hint: str = "unknown",
    script_hint: str = "unknown",
    adapter_hint: str | None = None,
) -> str:
    registry = adapter_registry()
    if adapter_hint:
        hinted = str(adapter_hint).strip().lower()
        if hinted in registry:
            return hinted

    lang = str(language_hint or "unknown").strip().lower()
    script = str(script_hint or "unknown").strip().lower()
    joined_hint = f"{lang} {script}".strip()
    hint_tokens = set(re.findall(r"[a-z]+", joined_hint))

    if any(token in joined_hint for token in ("akkadian", "cuneiform")):
        return "akkadian_transliteration"
    if "transliteration" in joined_hint or "scholarly" in joined_hint:
        return "scholarly_transliteration"
    if lang in {"german", "de", "deu"} or "german" in hint_tokens:
        return "german"
    if lang in {"english", "en", "eng"} or "english" in hint_tokens:
        return "english"
    if lang in {"french", "fr", "fra"} or "french" in hint_tokens:
        return "french"
    if lang in {"arabic", "rtl", "ara"} or script in {"arabic", "rtl", "ara"} or "arabic" in hint_tokens or "rtl" in hint_tokens:
        return "arabic_or_rtl_placeholder"

    if _contains_arabic_script(text):
        return "arabic_or_rtl_placeholder"
    if _looks_like_transliteration(text):
        return "scholarly_transliteration"

    return "default_latin"


def serialize_corrections(corrections: list[CorrectionAudit]) -> list[dict[str, Any]]:
    return [asdict(correction) for correction in corrections]
