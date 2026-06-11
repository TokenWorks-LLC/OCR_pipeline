from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
from typing import Iterable


_WORD_EDGE_RE = re.compile(r"^[^\w\u00c0-\u024f\u1e00-\u1eff\u0600-\u06ff]+|[^\w\u00c0-\u024f\u1e00-\u1eff\u0600-\u06ff]+$", re.UNICODE)


@dataclass(frozen=True)
class LexiconSuggestion:
    token: str
    distance: int
    frequency: int
    normalized_match: bool


class InMemoryLexicon:
    """Lexicon with domain-aware lookup and distance-based suggestions."""

    def __init__(self) -> None:
        self._domains: dict[str, Counter[str]] = {}
        self._normalized_domains: dict[str, Counter[str]] = {}

    @staticmethod
    def normalize_token(token: str) -> str:
        value = unicodedata.normalize("NFC", str(token or "")).casefold()
        value = _WORD_EDGE_RE.sub("", value)
        return value.strip()

    def load_word_list(
        self,
        words: Iterable[str],
        domain: str = "general",
        frequencies: dict[str, int] | None = None,
    ) -> None:
        domain_name = str(domain or "general").strip().lower() or "general"
        exact_bucket = self._domains.setdefault(domain_name, Counter())
        normalized_bucket = self._normalized_domains.setdefault(domain_name, Counter())
        frequency_map = frequencies or {}

        for raw_word in words:
            word = str(raw_word or "").strip()
            if not word:
                continue
            freq = int(frequency_map.get(word, 1) or 1)
            exact_bucket[word] += max(freq, 1)
            normalized = self.normalize_token(word)
            if normalized:
                normalized_bucket[normalized] += max(freq, 1)

    def load_from_file(
        self,
        path: str | Path,
        domain: str = "general",
        has_frequency: bool = False,
        delimiter: str | None = None,
    ) -> None:
        file_path = Path(path)
        if not file_path.exists():
            return

        words: list[str] = []
        frequencies: dict[str, int] = {}
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if has_frequency:
                    parts = stripped.split(delimiter) if delimiter else stripped.split()
                    if not parts:
                        continue
                    token = str(parts[0]).strip()
                    if not token:
                        continue
                    freq = 1
                    if len(parts) > 1:
                        try:
                            freq = int(float(parts[1]))
                        except (TypeError, ValueError):
                            freq = 1
                    words.append(token)
                    frequencies[token] = max(freq, 1)
                else:
                    words.append(stripped)

        self.load_word_list(words=words, domain=domain, frequencies=frequencies or None)

    def _iter_domain_counters(self, domain: str | None = None) -> list[Counter[str]]:
        if domain:
            key = str(domain).strip().lower()
            buckets = [self._domains.get("general", Counter())]
            if key != "general":
                buckets.append(self._domains.get(key, Counter()))
            return buckets
        return list(self._domains.values())

    def _iter_normalized_counters(self, domain: str | None = None) -> list[Counter[str]]:
        if domain:
            key = str(domain).strip().lower()
            buckets = [self._normalized_domains.get("general", Counter())]
            if key != "general":
                buckets.append(self._normalized_domains.get(key, Counter()))
            return buckets
        return list(self._normalized_domains.values())

    def lookup_exact(self, token: str, domain: str | None = None) -> bool:
        text = str(token or "").strip()
        if not text:
            return False
        for bucket in self._iter_domain_counters(domain=domain):
            if text in bucket:
                return True
        return False

    def lookup_normalized(self, token: str, domain: str | None = None) -> bool:
        normalized = self.normalize_token(token)
        if not normalized:
            return False
        for bucket in self._iter_normalized_counters(domain=domain):
            if normalized in bucket:
                return True
        return False

    def token_frequency(self, token: str, domain: str | None = None) -> int:
        text = str(token or "").strip()
        if not text:
            return 0

        freq = 0
        for bucket in self._iter_domain_counters(domain=domain):
            freq = max(freq, int(bucket.get(text, 0) or 0))

        normalized = self.normalize_token(text)
        if normalized:
            for bucket in self._iter_normalized_counters(domain=domain):
                freq = max(freq, int(bucket.get(normalized, 0) or 0))
        return freq

    @staticmethod
    def _edit_distance_limited(left: str, right: str, max_distance: int) -> int:
        if left == right:
            return 0
        if abs(len(left) - len(right)) > max_distance:
            return max_distance + 1

        prev = list(range(len(right) + 1))
        for i, a_char in enumerate(left, start=1):
            current = [i]
            min_row = i
            for j, b_char in enumerate(right, start=1):
                cost = 0 if a_char == b_char else 1
                current_val = min(
                    prev[j] + 1,
                    current[j - 1] + 1,
                    prev[j - 1] + cost,
                )
                current.append(current_val)
                min_row = min(min_row, current_val)
            if min_row > max_distance:
                return max_distance + 1
            prev = current
        return prev[-1]

    def suggest(
        self,
        token: str,
        max_distance: int = 1,
        max_results: int = 5,
        domain: str | None = None,
    ) -> list[LexiconSuggestion]:
        text = str(token or "").strip()
        normalized = self.normalize_token(text)
        if not normalized:
            return []

        candidate_counter: Counter[str] = Counter()
        for bucket in self._iter_domain_counters(domain=domain):
            candidate_counter.update(bucket)

        suggestions: list[LexiconSuggestion] = []
        for candidate, freq in candidate_counter.items():
            normalized_candidate = self.normalize_token(candidate)
            if not normalized_candidate:
                continue

            distance = self._edit_distance_limited(normalized, normalized_candidate, max_distance=max_distance)
            if distance > max_distance:
                continue

            suggestions.append(
                LexiconSuggestion(
                    token=candidate,
                    distance=distance,
                    frequency=int(freq),
                    normalized_match=(normalized_candidate == normalized),
                )
            )

        suggestions.sort(key=lambda item: (item.distance, -item.frequency, item.token))
        return suggestions[: max(1, int(max_results or 1))]

    def coverage(self, tokens: Iterable[str], domain: str | None = None) -> tuple[float, float, list[str]]:
        token_list = [str(token or "").strip() for token in tokens if str(token or "").strip()]
        if not token_list:
            return 1.0, 0.0, []

        unknown: list[str] = []
        known = 0
        for token in token_list:
            if self.lookup_exact(token, domain=domain) or self.lookup_normalized(token, domain=domain):
                known += 1
            else:
                unknown.append(token)

        total = len(token_list)
        coverage = known / float(total)
        unknown_rate = len(unknown) / float(total)
        return coverage, unknown_rate, unknown
