from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any
import unicodedata


@dataclass(frozen=True)
class QualityClassThresholds:
    production_quality_min: float = 0.85
    usable_with_review_min: float = 0.70
    weak_ocr_min: float = 0.50


@dataclass(frozen=True)
class GateThresholds:
    max_empty_rate: float = 1.0
    max_timeout_rate: float = 1.0
    max_failed_rate: float = 1.0
    min_avg_quality: float = 0.0
    max_review_rate: float = 1.0
    require_strict_readiness: bool = False
    enforce_quality: bool = False


@dataclass(frozen=True)
class LaunchGateResult:
    mode: str
    should_fail_run: bool
    failed_gate: str
    gate_reason: str
    reasons: list[str]
    review_required: bool


class OCRQualityScorer:
    """Multilingual-aware page/document/run quality scoring and launch gates."""

    DEFAULT_GATE_MODES: dict[str, GateThresholds] = {
        "internal": GateThresholds(
            max_empty_rate=1.0,
            max_timeout_rate=1.0,
            max_failed_rate=1.0,
            min_avg_quality=0.0,
            max_review_rate=1.0,
            require_strict_readiness=False,
            enforce_quality=False,
        ),
        "beta": GateThresholds(
            max_empty_rate=0.30,
            max_timeout_rate=0.25,
            max_failed_rate=0.35,
            min_avg_quality=0.50,
            max_review_rate=0.90,
            require_strict_readiness=False,
            enforce_quality=False,
        ),
        "production": GateThresholds(
            max_empty_rate=0.05,
            max_timeout_rate=0.05,
            max_failed_rate=0.10,
            min_avg_quality=0.70,
            max_review_rate=0.50,
            require_strict_readiness=False,
            enforce_quality=True,
        ),
        "strict": GateThresholds(
            max_empty_rate=0.02,
            max_timeout_rate=0.02,
            max_failed_rate=0.05,
            min_avg_quality=0.80,
            max_review_rate=0.30,
            require_strict_readiness=True,
            enforce_quality=True,
        ),
    }

    def __init__(
        self,
        *,
        class_thresholds: QualityClassThresholds | None = None,
        gate_mode: str = "internal",
        gate_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.class_thresholds = class_thresholds or QualityClassThresholds()
        mode = str(gate_mode or "internal").strip().lower()
        self.gate_mode = mode if mode in self.DEFAULT_GATE_MODES else "internal"

        gate_default = self.DEFAULT_GATE_MODES[self.gate_mode]
        overrides = dict(gate_overrides or {})
        self.gate_thresholds = GateThresholds(
            max_empty_rate=float(overrides.get("max_empty_rate", gate_default.max_empty_rate) or gate_default.max_empty_rate),
            max_timeout_rate=float(overrides.get("max_timeout_rate", gate_default.max_timeout_rate) or gate_default.max_timeout_rate),
            max_failed_rate=float(overrides.get("max_failed_rate", gate_default.max_failed_rate) or gate_default.max_failed_rate),
            min_avg_quality=float(overrides.get("min_avg_quality", gate_default.min_avg_quality) or gate_default.min_avg_quality),
            max_review_rate=float(overrides.get("max_review_rate", gate_default.max_review_rate) or gate_default.max_review_rate),
            require_strict_readiness=bool(
                overrides.get("require_strict_readiness", gate_default.require_strict_readiness)
            ),
            enforce_quality=bool(overrides.get("enforce_quality", gate_default.enforce_quality)),
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def classify_quality(self, score: float) -> str:
        val = self._clamp01(score)
        if val >= float(self.class_thresholds.production_quality_min):
            return "production_quality"
        if val >= float(self.class_thresholds.usable_with_review_min):
            return "usable_with_review"
        if val >= float(self.class_thresholds.weak_ocr_min):
            return "weak_ocr"
        return "failed_ocr"

    @staticmethod
    def _estimate_engine_confidence(metadata: dict[str, Any]) -> float:
        confidence = metadata.get("confidence", "")
        if confidence not in (None, ""):
            try:
                return max(0.0, min(float(confidence), 1.0))
            except (TypeError, ValueError):
                pass

        per_engine = metadata.get("confidence_per_engine", {})
        if isinstance(per_engine, dict) and per_engine:
            values: list[float] = []
            for value in per_engine.values():
                try:
                    values.append(max(0.0, min(float(value), 1.0)))
                except (TypeError, ValueError):
                    continue
            if values:
                return sum(values) / float(len(values))
        return 0.5

    @staticmethod
    def _text_length_score(text: str) -> float:
        stripped = str(text or "").strip()
        if not stripped:
            return 0.0

        length = len(stripped)
        if length < 5:
            return 0.10
        if length < 20:
            return 0.45
        if length < 40:
            return 0.70
        return min(1.0, 0.78 + (math.log1p(length) / 10.0))

    @staticmethod
    def _character_diversity_score(text: str) -> float:
        value = str(text or "")
        letters = [char for char in value if char.isalnum()]
        if not letters:
            return 0.0
        unique_count = len(set(letters))
        denominator = max(int(math.sqrt(len(letters)) * 2), 1)
        return max(0.0, min(unique_count / float(denominator), 1.0))

    @staticmethod
    def _script_compatibility_score(text: str, language_hint: str, script_hint: str) -> float:
        content = str(text or "")
        nonspace = [char for char in content if not char.isspace()]
        if not nonspace:
            return 0.0

        hint = f"{str(language_hint or '').strip().lower()} {str(script_hint or '').strip().lower()}".strip()
        arabic_count = sum(
            1
            for char in nonspace
            if ("\u0600" <= char <= "\u06ff") or ("\u0750" <= char <= "\u077f") or ("\u08a0" <= char <= "\u08ff")
        )
        latin_count = sum(1 for char in nonspace if "LATIN" in unicodedata.name(char, ""))

        if "arabic" in hint or "rtl" in hint:
            ratio = arabic_count / float(len(nonspace))
            return max(0.0, min((ratio * 1.8), 1.0))

        if any(token in hint for token in ("akkadian", "transliteration", "cuneiform")):
            translit_chars = sum(1 for char in nonspace if char in {"\u0161", "\u1e63", "\u1e6d", "\u1e2b", "\u012b", "\u016b"})
            hyphen_count = content.count("-")
            cue = translit_chars + hyphen_count
            return 0.65 if cue <= 0 else max(0.65, min(1.0, cue / 10.0))

        if latin_count <= 0:
            # Do not punish non-Latin output for unknown language hints.
            return 0.70

        ratio = latin_count / float(len(nonspace))
        return max(0.0, min((ratio * 1.2), 1.0))

    @staticmethod
    def _lexicon_score(metadata: dict[str, Any]) -> tuple[float, bool]:
        adapter = str(metadata.get("adapter_used", "")).strip().lower()
        if not adapter:
            return 0.60, False

        coverage = metadata.get("lexicon_coverage", "")
        if coverage in (None, ""):
            return 0.60, False

        try:
            value = max(0.0, min(float(coverage), 1.0))
        except (TypeError, ValueError):
            return 0.60, False
        return value, True

    @staticmethod
    def _agreement_score(metadata: dict[str, Any]) -> tuple[float, bool]:
        value = metadata.get("engine_agreement_score", "")
        if value in (None, ""):
            return 0.60, False
        try:
            return max(0.0, min(float(value), 1.0)), True
        except (TypeError, ValueError):
            return 0.60, False

    @staticmethod
    def _layout_score(metadata: dict[str, Any]) -> tuple[float, bool]:
        value = metadata.get("reading_order_confidence", "")
        if value in (None, ""):
            return 0.60, False
        try:
            return max(0.0, min(float(value), 1.0)), True
        except (TypeError, ValueError):
            return 0.60, False

    @staticmethod
    def _runtime_stability_score(status: str, engine_statuses: dict[str, Any], failure_reason: str) -> float:
        normalized_status = str(status or "").strip().lower()
        reason = str(failure_reason or "").strip().lower()
        if normalized_status == "timed_out" or "timeout" in reason:
            return 0.05
        if normalized_status == "failed":
            return 0.15

        statuses = []
        if isinstance(engine_statuses, dict):
            for payload in engine_statuses.values():
                if isinstance(payload, dict):
                    statuses.append(str(payload.get("status", "")).strip().lower())

        if not statuses:
            return 0.75

        bad = sum(1 for state in statuses if state in {"timed_out", "failed_on_page", "unavailable_dependency_error"})
        ratio = bad / float(len(statuses))
        return max(0.1, min(1.0, 0.95 - (ratio * 0.9)))

    @staticmethod
    def _preprocessing_confidence(metadata: dict[str, Any]) -> float:
        recommended = str(metadata.get("recommended_preprocessing_profile", "")).strip().lower()
        applied = str(metadata.get("applied_preprocessing_profile", "")).strip().lower()
        if not recommended or not applied:
            return 0.70
        if recommended == applied:
            return 1.0
        return 0.65

    @staticmethod
    def _postprocess_risk_score(metadata: dict[str, Any]) -> tuple[float, bool]:
        quality = metadata.get("postprocess_quality_score", "")
        protected_changes = OCRQualityScorer._safe_float(metadata.get("protected_character_changes", 0.0), 0.0)
        if quality in (None, ""):
            base = 0.70
            available = False
        else:
            base = OCRQualityScorer._clamp01(OCRQualityScorer._safe_float(quality, 0.70))
            available = True

        penalty = min(protected_changes * 0.12, 0.6)
        return max(0.0, base - penalty), available

    @staticmethod
    def _unknown_token_score(metadata: dict[str, Any]) -> tuple[float, bool]:
        unknown_rate = metadata.get("unknown_token_rate", "")
        if unknown_rate in (None, ""):
            return 0.70, False
        try:
            val = max(0.0, min(float(unknown_rate), 1.0))
        except (TypeError, ValueError):
            return 0.70, False
        return max(0.0, 1.0 - val), True

    def score_page(
        self,
        *,
        text: str,
        status: str,
        failure_reason: str,
        metadata: dict[str, Any] | None,
        engine_statuses: dict[str, Any] | None,
        language_hint: str,
        script_hint: str,
    ) -> dict[str, Any]:
        meta = dict(metadata or {})
        engine_payload = dict(engine_statuses or {})
        reasons: list[str] = []

        stripped = str(text or "").strip()
        if not stripped:
            return {
                "page_quality_score": 0.0,
                "quality_class": "failed_ocr",
                "needs_human_review": True,
                "quality_reasons": ["Empty OCR output"],
                "signal_breakdown": {
                    "empty_output": 0.0,
                },
                "failed_gate": "page_quality_gate",
                "gate_reason": "Empty OCR output",
            }

        engine_conf = self._estimate_engine_confidence(meta)
        length_score = self._text_length_score(stripped)
        diversity_score = self._character_diversity_score(stripped)
        script_score = self._script_compatibility_score(stripped, language_hint=language_hint, script_hint=script_hint)
        lexicon_score, has_lexicon = self._lexicon_score(meta)
        agreement_score, has_agreement = self._agreement_score(meta)
        layout_score, has_layout = self._layout_score(meta)
        runtime_score = self._runtime_stability_score(status=status, engine_statuses=engine_payload, failure_reason=failure_reason)
        preprocessing_score = self._preprocessing_confidence(meta)
        postprocess_score, has_postprocess = self._postprocess_risk_score(meta)
        unknown_score, has_unknown = self._unknown_token_score(meta)

        signal_breakdown = {
            "engine_confidence": round(engine_conf, 6),
            "text_length_sanity": round(length_score, 6),
            "character_diversity": round(diversity_score, 6),
            "language_script_compatibility": round(script_score, 6),
            "lexicon_coverage": round(lexicon_score, 6),
            "ensemble_agreement": round(agreement_score, 6),
            "layout_confidence": round(layout_score, 6),
            "runtime_stability": round(runtime_score, 6),
            "preprocessing_confidence": round(preprocessing_score, 6),
            "postprocessing_correction_risk": round(postprocess_score, 6),
            "unknown_token_score": round(unknown_score, 6),
        }

        weighted_score = (
            (engine_conf * 0.14)
            + (length_score * 0.11)
            + (diversity_score * 0.07)
            + (script_score * 0.10)
            + (lexicon_score * 0.08)
            + (agreement_score * 0.10)
            + (layout_score * 0.08)
            + (runtime_score * 0.10)
            + (preprocessing_score * 0.07)
            + (postprocess_score * 0.10)
            + (unknown_score * 0.05)
        )

        normalized_status = str(status or "").strip().lower()
        normalized_failure = str(failure_reason or "").strip().lower()
        if normalized_status in {"failed", "timed_out"}:
            weighted_score = min(weighted_score, 0.35 if normalized_status == "failed" else 0.20)
        if "timeout" in normalized_failure:
            weighted_score = min(weighted_score, 0.20)
        if engine_conf < 0.20:
            weighted_score = min(weighted_score, 0.55)

        score = self._clamp01(weighted_score)

        if engine_conf < 0.35:
            reasons.append("Low OCR engine confidence")
        if length_score < 0.45:
            reasons.append("Text length is too short for reliable extraction")
        if diversity_score < 0.35:
            reasons.append("Low character diversity suggests unreadable or repetitive OCR")
        if script_score < 0.45:
            reasons.append("Text appears weakly compatible with requested language/script")
        if has_lexicon and lexicon_score < 0.55:
            reasons.append("Low lexicon coverage for active language adapter")
        if has_agreement and agreement_score < 0.50:
            reasons.append("Low agreement across OCR engines")
        if has_layout and layout_score < 0.45:
            reasons.append("Low layout/reading-order confidence")
        if runtime_score < 0.40:
            reasons.append("Runtime instability detected (timeouts/errors)")
        if preprocessing_score < 0.70:
            reasons.append("Applied preprocessing profile differs from recommended profile")
        if has_postprocess and postprocess_score < 0.50:
            reasons.append("High postprocessing correction risk")
        if has_unknown and unknown_score < 0.50:
            reasons.append("High unknown-token rate")

        quality_class = self.classify_quality(score)
        needs_human_review = bool(quality_class != "production_quality" or len(reasons) > 0)

        failed_gate = ""
        gate_reason = ""
        if quality_class == "failed_ocr":
            failed_gate = "page_quality_gate"
            gate_reason = reasons[0] if reasons else "Page quality below failed_ocr threshold"

        return {
            "page_quality_score": round(score, 6),
            "quality_class": quality_class,
            "needs_human_review": needs_human_review,
            "quality_reasons": reasons,
            "signal_breakdown": signal_breakdown,
            "failed_gate": failed_gate,
            "gate_reason": gate_reason,
        }

    def score_region(self, region: dict[str, Any], page_quality_score: float = 0.6) -> dict[str, Any]:
        payload = dict(region or {})
        text = str(payload.get("text", "") or "")
        region_type = str(payload.get("type", "unknown") or "unknown")
        confidence = self._safe_float(payload.get("confidence", 0.6), 0.6)
        engine_name = str(payload.get("engine", "") or "unknown")

        length_score = self._text_length_score(text)
        confidence_score = self._clamp01(confidence)
        engine_score = 0.75 if engine_name not in {"", "unknown", "none"} else 0.50
        page_anchor = self._clamp01(page_quality_score)

        region_quality = self._clamp01(
            (confidence_score * 0.45)
            + (length_score * 0.25)
            + (engine_score * 0.15)
            + (page_anchor * 0.15)
        )

        reasons: list[str] = []
        if confidence_score < 0.40:
            reasons.append("Low region confidence")
        if length_score < 0.35 and region_type not in {"figure", "table"}:
            reasons.append("Region text is very short")
        if engine_score < 0.60:
            reasons.append("Region engine source is unknown")

        needs_review = bool(region_quality < 0.70 or reasons)
        return {
            "region_quality_score": round(region_quality, 6),
            "region_type": region_type,
            "confidence": round(confidence_score, 6),
            "text_length": len(text.strip()),
            "engine_used": engine_name,
            "needs_review": needs_review,
            "quality_reasons": reasons,
            "quality_class": self.classify_quality(region_quality),
        }

    def aggregate_document_quality(self, page_records: list[dict[str, Any]]) -> dict[str, Any]:
        if not page_records:
            return {
                "document_quality_score": 0.0,
                "quality_class": "failed_ocr",
                "mean_page_quality": 0.0,
                "median_page_quality": 0.0,
                "worst_page_quality": 0.0,
                "empty_page_rate": 1.0,
                "failed_page_rate": 1.0,
                "timeout_rate": 0.0,
                "review_percentage": 1.0,
                "page_count": 0,
            }

        scores = [self._safe_float(item.get("page_quality_score", 0.0), 0.0) for item in page_records]
        page_count = len(page_records)
        mean_score = sum(scores) / float(page_count)
        median_score = statistics.median(scores)
        worst_score = min(scores)

        empty_count = sum(1 for item in page_records if bool(item.get("is_empty", False)))
        failed_count = sum(1 for item in page_records if str(item.get("status", "")).strip().lower() == "failed")
        timeout_count = sum(1 for item in page_records if str(item.get("status", "")).strip().lower() == "timed_out")
        review_count = sum(1 for item in page_records if bool(item.get("needs_human_review", False)))

        empty_rate = empty_count / float(page_count)
        failed_rate = failed_count / float(page_count)
        timeout_rate = timeout_count / float(page_count)
        review_pct = review_count / float(page_count)

        document_score = self._clamp01(
            (mean_score * 0.50)
            + (median_score * 0.20)
            + (worst_score * 0.20)
            + ((1.0 - empty_rate) * 0.05)
            + ((1.0 - timeout_rate) * 0.05)
        )

        return {
            "document_quality_score": round(document_score, 6),
            "quality_class": self.classify_quality(document_score),
            "mean_page_quality": round(float(mean_score), 6),
            "median_page_quality": round(float(median_score), 6),
            "worst_page_quality": round(float(worst_score), 6),
            "empty_page_rate": round(float(empty_rate), 6),
            "failed_page_rate": round(float(failed_rate), 6),
            "timeout_rate": round(float(timeout_rate), 6),
            "review_percentage": round(float(review_pct), 6),
            "page_count": int(page_count),
        }

    def aggregate_run_quality(
        self,
        *,
        document_summaries: dict[str, dict[str, Any]],
        page_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        page_count = len(page_records)
        doc_count = len(document_summaries)
        if page_count <= 0:
            return {
                "documents": doc_count,
                "pages": 0,
                "success_rate": 0.0,
                "failure_rate": 1.0,
                "timeout_rate": 0.0,
                "average_runtime_ms": 0.0,
                "p90_runtime_ms": 0.0,
                "average_quality_score": 0.0,
                "review_percentage": 1.0,
                "run_quality_score": 0.0,
                "quality_class": "failed_ocr",
                "empty_page_rate": 1.0,
                "failed_page_rate": 1.0,
            }

        success_count = sum(1 for item in page_records if str(item.get("status", "")).strip().lower() in {"success", "partial_success"})
        failed_count = sum(1 for item in page_records if str(item.get("status", "")).strip().lower() == "failed")
        timeout_count = sum(1 for item in page_records if str(item.get("status", "")).strip().lower() == "timed_out")
        review_count = sum(1 for item in page_records if bool(item.get("needs_human_review", False)))
        empty_count = sum(1 for item in page_records if bool(item.get("is_empty", False)))

        runtime_values = [self._safe_float(item.get("runtime_ms", 0.0), 0.0) for item in page_records]
        quality_values = [self._safe_float(item.get("page_quality_score", 0.0), 0.0) for item in page_records]

        avg_runtime = sum(runtime_values) / float(page_count)
        sorted_runtime = sorted(runtime_values)
        p90_index = min(page_count - 1, max(0, int(math.ceil(page_count * 0.90)) - 1))
        p90_runtime = sorted_runtime[p90_index] if sorted_runtime else 0.0

        avg_quality = sum(quality_values) / float(page_count)
        success_rate = success_count / float(page_count)
        failed_rate = failed_count / float(page_count)
        timeout_rate = timeout_count / float(page_count)
        review_pct = review_count / float(page_count)
        empty_rate = empty_count / float(page_count)

        run_quality_score = self._clamp01(
            (avg_quality * 0.55)
            + (success_rate * 0.20)
            + ((1.0 - failed_rate) * 0.10)
            + ((1.0 - timeout_rate) * 0.10)
            + ((1.0 - review_pct) * 0.05)
        )

        return {
            "documents": int(doc_count),
            "pages": int(page_count),
            "success_rate": round(float(success_rate), 6),
            "failure_rate": round(float(failed_rate), 6),
            "timeout_rate": round(float(timeout_rate), 6),
            "average_runtime_ms": round(float(avg_runtime), 3),
            "p90_runtime_ms": round(float(p90_runtime), 3),
            "average_quality_score": round(float(avg_quality), 6),
            "review_percentage": round(float(review_pct), 6),
            "run_quality_score": round(float(run_quality_score), 6),
            "quality_class": self.classify_quality(run_quality_score),
            "empty_page_rate": round(float(empty_rate), 6),
            "failed_page_rate": round(float(failed_rate), 6),
        }

    def evaluate_launch_gates(
        self,
        *,
        run_summary: dict[str, Any],
        has_usable_engine: bool,
        strict_readiness_ok: bool,
        ocr_required: bool = True,
    ) -> LaunchGateResult:
        reasons: list[str] = []
        failed_gate = ""
        gate_reason = ""
        should_fail = False

        mode = self.gate_mode
        thresholds = self.gate_thresholds

        # The engine-availability gate guards against shipping a run whose pages
        # silently went un-OCR'd because no engine was usable. It must only fire
        # when the run actually depended on OCR: a run whose pages were all
        # served by the PDF text layer must not be failed just because no OCR
        # engine is installed (see the text-layer-only CI regression).
        if ocr_required and not has_usable_engine:
            failed_gate = "engine_availability_gate"
            gate_reason = "No usable OCR engine available"
            reasons.append(gate_reason)
            should_fail = True

        if thresholds.require_strict_readiness and not strict_readiness_ok:
            if not failed_gate:
                failed_gate = "strict_readiness_gate"
                gate_reason = "Strict readiness gate failed"
            reasons.append("Strict readiness gate failed")
            should_fail = True

        checks = [
            (
                self._safe_float(run_summary.get("empty_page_rate", 0.0), 0.0) > float(thresholds.max_empty_rate),
                "empty_page_rate_gate",
                f"Empty page rate exceeds gate ({run_summary.get('empty_page_rate', 0.0):.3f} > {thresholds.max_empty_rate:.3f})",
            ),
            (
                self._safe_float(run_summary.get("timeout_rate", 0.0), 0.0) > float(thresholds.max_timeout_rate),
                "timeout_rate_gate",
                f"Timeout rate exceeds gate ({run_summary.get('timeout_rate', 0.0):.3f} > {thresholds.max_timeout_rate:.3f})",
            ),
            (
                self._safe_float(run_summary.get("failed_page_rate", run_summary.get("failure_rate", 0.0)), 0.0) > float(thresholds.max_failed_rate),
                "failed_page_rate_gate",
                f"Failed page rate exceeds gate ({run_summary.get('failed_page_rate', run_summary.get('failure_rate', 0.0)):.3f} > {thresholds.max_failed_rate:.3f})",
            ),
            (
                self._safe_float(run_summary.get("average_quality_score", 0.0), 0.0) < float(thresholds.min_avg_quality),
                "avg_quality_gate",
                f"Average quality below gate ({run_summary.get('average_quality_score', 0.0):.3f} < {thresholds.min_avg_quality:.3f})",
            ),
            (
                self._safe_float(run_summary.get("review_percentage", 0.0), 0.0) > float(thresholds.max_review_rate),
                "review_rate_gate",
                f"Review-needed rate exceeds gate ({run_summary.get('review_percentage', 0.0):.3f} > {thresholds.max_review_rate:.3f})",
            ),
        ]

        for condition, gate_name, message in checks:
            if not condition:
                continue
            reasons.append(message)
            if thresholds.enforce_quality:
                if not failed_gate:
                    failed_gate = gate_name
                    gate_reason = message
                should_fail = True

        # Beta mode remains review-focused unless catastrophic failures occurred.
        review_required = bool(reasons)

        return LaunchGateResult(
            mode=mode,
            should_fail_run=bool(should_fail),
            failed_gate=failed_gate,
            gate_reason=gate_reason,
            reasons=reasons,
            review_required=review_required,
        )
