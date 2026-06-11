#!/usr/bin/env python3
"""Utilities for measurable and explainable OCR ensemble behavior."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import math
from typing import Any


_GENERIC_TOLERANCE_PAIRS = {
    ("O", "0"),
    ("0", "O"),
    ("l", "1"),
    ("1", "l"),
    ("I", "1"),
    ("1", "I"),
    ("m", "rn"),
    ("rn", "m"),
    ("“", '"'),
    ("”", '"'),
    ("‘", "'"),
    ("’", "'"),
}

_TRANSLITERATION_TOLERANCE_PAIRS = {
    ("s", "s"),
    ("s", "s\u030c"),
    ("s\u030c", "s"),
    ("s", "\u0161"),
    ("\u0161", "s"),
    ("h", "\u1e2b"),
    ("\u1e2b", "h"),
    ("t", "\u1e6d"),
    ("\u1e6d", "t"),
}


def _tokens(text: str) -> list[str]:
    return [token for token in str(text or "").split() if token]


def _lines(text: str) -> list[str]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return lines or [str(text or "").strip()]


def _normalized_entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0

    values = [count / total for count in counter.values() if count > 0]
    if len(values) <= 1:
        return 0.0

    entropy = -sum(value * math.log(value) for value in values)
    denom = math.log(len(values))
    if denom <= 0:
        return 0.0
    return float(max(0.0, min(entropy / denom, 1.0)))


def _align_sequence(reference: list[str], other: list[str]) -> list[str | None]:
    aligned: list[str | None] = [None] * len(reference)
    matcher = SequenceMatcher(a=[token.casefold() for token in reference], b=[token.casefold() for token in other], autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op in {"equal", "replace"}:
            span = min(i2 - i1, j2 - j1)
            for idx in range(span):
                aligned[i1 + idx] = other[j1 + idx]
    return aligned


def _is_transliteration_context(language_hint: str, script_hint: str, preprocessing_profile: str) -> bool:
    joined = " ".join(
        [
            str(language_hint or "").strip().lower(),
            str(script_hint or "").strip().lower(),
            str(preprocessing_profile or "").strip().lower(),
        ]
    )
    return any(token in joined for token in ("akkadian", "transliteration", "diacritic", "cuneiform"))


def _active_tolerance_pairs(language_hint: str, script_hint: str, preprocessing_profile: str) -> set[tuple[str, str]]:
    pairs = set(_GENERIC_TOLERANCE_PAIRS)
    if _is_transliteration_context(language_hint, script_hint, preprocessing_profile):
        pairs.update(_TRANSLITERATION_TOLERANCE_PAIRS)
    return pairs


def char_disagreement_rate(
    left: str,
    right: str,
    language_hint: str = "unknown",
    script_hint: str = "unknown",
    preprocessing_profile: str = "",
) -> float:
    left_text = str(left or "")
    right_text = str(right or "")
    if not left_text and not right_text:
        return 0.0

    tolerance_pairs = _active_tolerance_pairs(language_hint, script_hint, preprocessing_profile)
    matcher = SequenceMatcher(a=left_text, b=right_text, autojunk=False)
    mismatch_weight = 0.0

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        src = left_text[i1:i2]
        dst = right_text[j1:j2]
        if op == "equal":
            continue
        if op in {"insert", "delete"}:
            mismatch_weight += float(max(len(src), len(dst)))
            continue

        # replace
        if (src, dst) in tolerance_pairs:
            mismatch_weight += 0.45 * float(max(len(src), len(dst), 1))
        elif len(src) == len(dst):
            for a_char, b_char in zip(src, dst):
                if (a_char, b_char) in tolerance_pairs:
                    mismatch_weight += 0.45
                else:
                    mismatch_weight += 1.0
        else:
            mismatch_weight += float(max(len(src), len(dst), 1))

    normalizer = float(max(len(left_text), len(right_text), 1))
    return float(max(0.0, min(mismatch_weight / normalizer, 1.0)))


def analyze_alignment(
    engine_texts: dict[str, str],
    consensus_text: str,
    language_hint: str = "unknown",
    script_hint: str = "unknown",
    preprocessing_profile: str = "",
) -> dict[str, Any]:
    nonempty = {engine: str(text or "") for engine, text in engine_texts.items() if str(text or "").strip()}
    all_texts = {engine: str(text or "") for engine, text in engine_texts.items()}

    engines = sorted(all_texts.keys())
    if len(engines) <= 1:
        return {
            "char_disagreement_rate": 0.0,
            "token_disagreement_rate": 0.0,
            "line_disagreement_rate": 0.0,
            "engine_agreement_score": 1.0,
            "consensus_entropy": 0.0,
            "engine_agreement": {engine: 1.0 for engine in engines},
            "disagreement_positions": [],
        }

    pairwise_char_disagreements: list[float] = []
    pairwise_token_disagreements: list[float] = []
    pairwise_line_disagreements: list[float] = []
    engine_agreement: dict[str, list[float]] = {engine: [] for engine in engines}

    for idx, left_engine in enumerate(engines):
        left_text = all_texts[left_engine]
        for right_engine in engines[idx + 1 :]:
            right_text = all_texts[right_engine]
            char_disagreement = char_disagreement_rate(
                left_text,
                right_text,
                language_hint=language_hint,
                script_hint=script_hint,
                preprocessing_profile=preprocessing_profile,
            )
            char_similarity = 1.0 - char_disagreement
            token_similarity = SequenceMatcher(a=[token.casefold() for token in _tokens(left_text)], b=[token.casefold() for token in _tokens(right_text)], autojunk=False).ratio()
            line_similarity = SequenceMatcher(a=[line.casefold() for line in _lines(left_text)], b=[line.casefold() for line in _lines(right_text)], autojunk=False).ratio()

            pairwise_char_disagreements.append(char_disagreement)
            pairwise_token_disagreements.append(1.0 - token_similarity)
            pairwise_line_disagreements.append(1.0 - line_similarity)
            engine_agreement[left_engine].append(char_similarity)
            engine_agreement[right_engine].append(char_similarity)

    reference_tokens = _tokens(consensus_text)
    if not reference_tokens:
        longest_engine = max(nonempty.items(), key=lambda item: len(_tokens(item[1])), default=("", ""))[0]
        reference_tokens = _tokens(nonempty.get(longest_engine, ""))

    disagreement_positions: list[int] = []
    entropies: list[float] = []
    if reference_tokens:
        aligned_tokens: dict[str, list[str | None]] = {}
        for engine, text in all_texts.items():
            aligned_tokens[engine] = _align_sequence(reference_tokens, _tokens(text))

        for token_index in range(len(reference_tokens)):
            token_counter: Counter[str] = Counter()
            for engine in engines:
                token = aligned_tokens[engine][token_index]
                if token is not None:
                    token_counter[token]
                    token_counter[token] += 1
            if len(token_counter) > 1:
                disagreement_positions.append(token_index)
            if token_counter:
                entropies.append(_normalized_entropy(token_counter))

    mean_char_disagreement = sum(pairwise_char_disagreements) / len(pairwise_char_disagreements) if pairwise_char_disagreements else 0.0
    mean_token_disagreement = sum(pairwise_token_disagreements) / len(pairwise_token_disagreements) if pairwise_token_disagreements else 0.0
    mean_line_disagreement = sum(pairwise_line_disagreements) / len(pairwise_line_disagreements) if pairwise_line_disagreements else 0.0

    agreement_map = {
        engine: (sum(values) / len(values) if values else 0.0)
        for engine, values in engine_agreement.items()
    }
    engine_agreement_score = float(sum(agreement_map.values()) / len(agreement_map)) if agreement_map else 0.0
    consensus_entropy = float(sum(entropies) / len(entropies)) if entropies else 0.0

    return {
        "char_disagreement_rate": round(mean_char_disagreement, 6),
        "token_disagreement_rate": round(mean_token_disagreement, 6),
        "line_disagreement_rate": round(mean_line_disagreement, 6),
        "engine_agreement_score": round(engine_agreement_score, 6),
        "consensus_entropy": round(consensus_entropy, 6),
        "engine_agreement": {engine: round(score, 6) for engine, score in agreement_map.items()},
        "disagreement_positions": disagreement_positions,
    }


def build_confusion_counts(reference_text: str, observed_text: str) -> dict[str, int]:
    reference = str(reference_text or "")
    observed = str(observed_text or "")
    if not reference and not observed:
        return {}

    matcher = SequenceMatcher(a=reference, b=observed, autojunk=False)
    confusion: Counter[str] = Counter()

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        src = reference[i1:i2]
        dst = observed[j1:j2]
        if op == "equal":
            continue

        if op in {"insert", "delete"}:
            label = f"{src or '<eps>'}->{dst or '<eps>'}"
            confusion[label] += max(len(src), len(dst), 1)
            continue

        # replace
        if len(src) == len(dst) and len(src) > 1:
            for src_char, dst_char in zip(src, dst):
                confusion[f"{src_char}->{dst_char}"] += 1
        else:
            confusion[f"{src or '<eps>'}->{dst or '<eps>'}"] += max(len(src), len(dst), 1)

    return dict(confusion)


def aggregate_confusion(
    consensus_text: str,
    engine_texts: dict[str, str],
) -> dict[str, int]:
    aggregate: Counter[str] = Counter()
    for text in engine_texts.values():
        for label, count in build_confusion_counts(consensus_text, text).items():
            aggregate[label] += int(count)
    return dict(aggregate)


def _sanity_score(text: str) -> float:
    sample = str(text or "")
    if not sample.strip():
        return 0.0

    length_component = min(len(sample.strip()) / 80.0, 1.0)
    alnum_ratio = sum(1 for char in sample if char.isalnum()) / max(len(sample), 1)
    replacement_penalty = 0.25 if "\ufffd" in sample else 0.0
    return float(max(0.0, min((0.55 * length_component) + (0.45 * alnum_ratio) - replacement_penalty, 1.0)))


def explain_consensus(
    per_engine_outputs: list[dict[str, Any]],
    consensus_text: str,
    alignment_metrics: dict[str, Any],
    historical_reliability: dict[str, dict[str, Any]] | None = None,
    language_hint: str = "unknown",
    script_hint: str = "unknown",
    quality_thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reliability = historical_reliability or {}
    thresholds = quality_thresholds or {}

    min_conf_success = float(thresholds.get("consensus_min_confidence", 0.20) or 0.20)
    min_engine_agreement = float(thresholds.get("consensus_min_engine_agreement", 0.45) or 0.45)
    max_entropy = float(thresholds.get("consensus_max_entropy", 0.72) or 0.72)

    candidates = [item for item in per_engine_outputs if str(item.get("text", "")).strip()]
    engines = [str(item.get("engine", "")).strip() for item in per_engine_outputs if str(item.get("engine", "")).strip()]

    if not candidates:
        return {
            "winner_engine": "",
            "consensus_used": False,
            "disagreement_summary": "no_candidate_text",
            "high_confidence": False,
            "low_confidence": True,
            "uncertain": True,
            "human_review_recommended": True,
            "low_quality_all_engines": True,
            "reason_codes": ["no_candidate_text"],
            "engine_scores": {},
        }

    translit_context = _is_transliteration_context(language_hint, script_hint, "")
    engine_scores: dict[str, float] = {}
    for item in candidates:
        engine = str(item.get("engine", "")).strip()
        text = str(item.get("text", ""))
        confidence = float(item.get("confidence", 0.0) or 0.0)
        runtime_ms = float(item.get("runtime_ms", 0.0) or 0.0)

        agreement_with_consensus = SequenceMatcher(None, text.casefold(), str(consensus_text or "").casefold()).ratio()
        historical = reliability.get(engine, {})
        success_rate = float(historical.get("success_rate", 0.0) or 0.0)
        compat_bonus = 0.0
        if translit_context and engine in {"cured", "kraken"}:
            compat_bonus = 0.12
        elif script_hint.strip().lower() in {"arabic", "cjk"} and engine in {"paddle", "doctr"}:
            compat_bonus = 0.08

        runtime_penalty = min(runtime_ms / 5000.0, 1.0) * 0.08
        sanity = _sanity_score(text)

        score = (
            (confidence * 0.42)
            + (agreement_with_consensus * 0.30)
            + (success_rate * 0.18)
            + (sanity * 0.10)
            + compat_bonus
            - runtime_penalty
        )
        engine_scores[engine] = float(score)

    winner_engine = max(engine_scores.items(), key=lambda item: item[1])[0]
    avg_conf = sum(float(item.get("confidence", 0.0) or 0.0) for item in candidates) / len(candidates)
    agreement_score = float(alignment_metrics.get("engine_agreement_score", 0.0) or 0.0)
    entropy = float(alignment_metrics.get("consensus_entropy", 0.0) or 0.0)

    low_quality_all = all(
        (float(item.get("confidence", 0.0) or 0.0) < min_conf_success) or (_sanity_score(str(item.get("text", ""))) < 0.20)
        for item in candidates
    )
    high_disagreement = agreement_score < min_engine_agreement
    low_confidence = avg_conf < min_conf_success
    high_entropy = entropy > max_entropy

    uncertain = bool(low_quality_all or high_disagreement or low_confidence or high_entropy)
    confidence_label = "high"
    if uncertain:
        confidence_label = "low"
    elif avg_conf < 0.55:
        confidence_label = "medium"

    reason_codes: list[str] = []
    if low_quality_all:
        reason_codes.append("low_quality_all_engines")
    if high_disagreement:
        reason_codes.append("high_engine_disagreement")
    if low_confidence:
        reason_codes.append("low_average_confidence")
    if high_entropy:
        reason_codes.append("high_consensus_entropy")

    disagreement_positions = alignment_metrics.get("disagreement_positions", [])
    disagreement_summary = (
        f"{len(disagreement_positions)} token positions disagreed across {len(set(engines))} engines"
        if disagreement_positions
        else "minimal token disagreement"
    )

    return {
        "winner_engine": winner_engine,
        "consensus_used": len(candidates) > 1,
        "disagreement_summary": disagreement_summary,
        "high_confidence": confidence_label == "high",
        "low_confidence": confidence_label == "low",
        "uncertain": uncertain,
        "human_review_recommended": uncertain,
        "low_quality_all_engines": low_quality_all,
        "reason_codes": reason_codes,
        "engine_scores": {engine: round(score, 6) for engine, score in engine_scores.items()},
        "confidence_band": confidence_label,
    }
