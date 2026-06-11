#!/usr/bin/env python3
"""Adaptive OCR routing strategy selection for multilingual pages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from production.preprocessing_profiles import (
    PROFILE_AUTO,
    PROFILE_TRANSLITERATION_DIACRITIC,
    PROFILE_UNKNOWN_SAFE_DEFAULT,
    resolve_preprocessing_profile,
)


ENGINE_STATUS_AVAILABLE = "available"
ENGINE_STATUS_AVAILABLE_UNHEALTHY = "available_but_unhealthy"
ENGINE_STATUS_DISABLED_BY_CONFIG = "disabled_by_config"
ENGINE_STATUS_UNAVAILABLE_DEPENDENCY = "unavailable_dependency_error"
ENGINE_STATUS_TIMED_OUT = "timed_out"
ENGINE_STATUS_FAILED_ON_PAGE = "failed_on_page"


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "t"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def _split_tokens(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    if "|" in text:
        return [part.strip().lower() for part in text.split("|") if part.strip()]
    return [text.lower()]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class OCRRoutingStrategy:
    selected_strategy: str
    mode: str
    use_text_layer: bool
    use_full_page_ocr: bool
    use_region_ocr: bool
    preprocessing_profile: str
    primary_engine: str
    fallback_engines: list[str] = field(default_factory=list)
    use_ensemble: bool = False
    max_engines_per_page: int = 1
    per_engine_timeout_s: float = 0.0
    per_page_timeout_s: float = 0.0
    quality_thresholds: dict[str, Any] = field(default_factory=dict)
    engines_skipped: dict[str, str] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)
    language_rules_applied: list[str] = field(default_factory=list)

    def engine_plan(self) -> list[str]:
        plan: list[str] = []
        if self.primary_engine:
            plan.append(self.primary_engine)
        for engine in self.fallback_engines:
            if engine and engine not in plan:
                plan.append(engine)
        return plan

    def to_metadata(self) -> dict[str, Any]:
        return {
            "selected_strategy": self.selected_strategy,
            "strategy_mode": self.mode,
            "strategy_use_text_layer": bool(self.use_text_layer),
            "strategy_use_full_page_ocr": bool(self.use_full_page_ocr),
            "strategy_use_region_ocr": bool(self.use_region_ocr),
            "strategy_primary_engine": self.primary_engine,
            "strategy_fallback_engines": list(self.fallback_engines),
            "strategy_ensemble_needed": bool(self.use_ensemble),
            "strategy_max_engines_per_page": int(self.max_engines_per_page),
            "strategy_per_engine_timeout_s": float(self.per_engine_timeout_s),
            "strategy_per_page_timeout_s": float(self.per_page_timeout_s),
            "strategy_quality_thresholds": dict(self.quality_thresholds),
            "engines_skipped": dict(self.engines_skipped),
            "engine_skip_reasons": sorted(set(self.engines_skipped.values())),
            "language_rules_applied": list(self.language_rules_applied),
            "strategy_reason_codes": list(self.reason_codes),
        }


class OCRStrategySelector:
    """Select OCR routing strategy from diagnostics, health, and runtime policy."""

    DEFAULT_CONFIG: dict[str, Any] = {
        "enabled": True,
        "mode": "strict",  # strict|beta|debug
        "max_engines_per_page": 2,
        "max_fallback_engines": 2,
        "quality_thresholds": {
            "min_confidence": 0.30,
            "min_text_chars": 0,
            "min_alnum_ratio": 0.20,
            "text_layer_min_chars": 24,
            "layout_complexity_threshold": 0.55,
        },
        "timeouts": {
            "per_engine_timeout_s": 0.0,
            "per_page_timeout_s": 0.0,
        },
        "health": {
            "max_consecutive_timeouts": 2,
            "max_consecutive_failures": 3,
            "skip_unhealthy_in_strict": True,
        },
        "engine_priority": ["paddle", "doctr", "mmocr", "kraken", "cured"],
        "fast_ocr_engines": ["paddle", "doctr"],
        "layout_first_engines": ["doctr", "paddle", "mmocr"],
        "diacritic_sensitive_engines": ["kraken", "cured", "doctr"],
        "layout_first_full_page_fallback": True,
        "enable_akkadian_rule": False,
        "language_rules": [],
        "script_engine_preferences": {
            "arabic": ["paddle", "doctr"],
            "latin": ["paddle", "doctr", "mmocr"],
            "cjk": ["paddle", "doctr"],
            "cuneiform": ["cured", "kraken"],
            "unknown": [],
        },
    }

    def __init__(self, routing_config: dict[str, Any] | None = None):
        self.config = _deep_merge(self.DEFAULT_CONFIG, routing_config or {})

    @staticmethod
    def _contains_any_token(haystack: str, tokens: list[str]) -> bool:
        text = (haystack or "").strip().lower()
        return any(token and token.lower() in text for token in tokens)

    def _apply_language_rules(
        self,
        language_hint: str,
        script_hint: str,
        document_type: str,
        requested_profile: str,
    ) -> tuple[dict[str, float], str | None, str | None, list[str], list[str]]:
        score_boosts: dict[str, float] = {}
        forced_profile: str | None = None
        forced_strategy: str | None = None
        applied_rules: list[str] = []
        reason_codes: list[str] = []

        combined_hint = " ".join(
            [
                str(language_hint or "").strip().lower(),
                str(script_hint or "").strip().lower(),
                str(document_type or "").strip().lower(),
            ]
        )

        rules = list(self.config.get("language_rules", []))
        if _safe_bool(self.config.get("enable_akkadian_rule", False)):
            rules.append(
                {
                    "name": "akkadian_optional",
                    "enabled": True,
                    "when_any_hint_contains": ["akkadian", "transliteration", "cuneiform"],
                    "engine_priority": ["cured", "kraken", "doctr"],
                    "force_profile": PROFILE_TRANSLITERATION_DIACRITIC,
                    "strategy": "high_dpi_conservative",
                }
            )

        for rule in rules:
            if not isinstance(rule, dict):
                continue
            if not _safe_bool(rule.get("enabled", True)):
                continue

            tokens = [str(token).strip().lower() for token in rule.get("when_any_hint_contains", []) if str(token).strip()]
            if tokens and not self._contains_any_token(combined_hint, tokens):
                continue

            rule_name = str(rule.get("name", "unnamed_rule")).strip() or "unnamed_rule"
            applied_rules.append(rule_name)
            reason_codes.append(f"language_rule:{rule_name}")

            priority = [str(engine).strip().lower() for engine in rule.get("engine_priority", []) if str(engine).strip()]
            for index, engine in enumerate(priority):
                score_boosts[engine] = score_boosts.get(engine, 0.0) + max(0.0, 2.5 - (index * 0.35))

            if requested_profile == PROFILE_AUTO:
                candidate_profile = str(rule.get("force_profile", "")).strip().lower()
                if candidate_profile:
                    forced_profile = candidate_profile

            candidate_strategy = str(rule.get("strategy", "")).strip().lower()
            if candidate_strategy:
                forced_strategy = candidate_strategy

        return score_boosts, forced_profile, forced_strategy, applied_rules, reason_codes

    def select(
        self,
        diagnostics: dict[str, Any] | None,
        engine_readiness: dict[str, dict[str, str]],
        enabled_engines: list[str],
        language_hint: str = "unknown",
        script_hint: str = "unknown",
        document_type: str = "unknown",
        requested_profile: str = PROFILE_AUTO,
        default_profile: str = PROFILE_UNKNOWN_SAFE_DEFAULT,
        previous_engine_performance: dict[str, dict[str, Any]] | None = None,
        timeout_config: dict[str, Any] | None = None,
        quality_thresholds: dict[str, Any] | None = None,
        force_ocr: bool = False,
        prefer_text_layer: bool = True,
    ) -> OCRRoutingStrategy:
        diagnostics = diagnostics or {}
        performance = previous_engine_performance or {}

        requested_profile_norm = (requested_profile or PROFILE_AUTO).strip().lower() or PROFILE_AUTO
        default_profile_norm = (default_profile or PROFILE_UNKNOWN_SAFE_DEFAULT).strip().lower() or PROFILE_UNKNOWN_SAFE_DEFAULT

        thresholds = _deep_merge(self.config.get("quality_thresholds", {}), quality_thresholds or {})
        runtime_timeouts = _deep_merge(self.config.get("timeouts", {}), timeout_config or {})
        health_limits = dict(self.config.get("health", {}))

        mode = str(self.config.get("mode", "strict") or "strict").strip().lower()
        if mode not in {"strict", "beta", "debug"}:
            mode = "strict"

        text_layer_min_chars = _safe_int(thresholds.get("text_layer_min_chars"), 24)
        text_layer_char_count = _safe_int(diagnostics.get("text_layer_char_count"), 0)
        text_layer_usable = _safe_bool(diagnostics.get("text_layer_usable", False))
        suspicious_reasons = set(_split_tokens(diagnostics.get("text_layer_suspicious_reasons", [])))
        unicode_sanity_ok = "broken_unicode" not in suspicious_reasons and "repeated_junk" not in suspicious_reasons

        has_text_layer_candidate = text_layer_char_count >= text_layer_min_chars
        use_text_layer = bool(
            prefer_text_layer
            and not force_ocr
            and has_text_layer_candidate
            and text_layer_usable
            and unicode_sanity_ok
        )

        estimated_columns = max(
            _safe_int(diagnostics.get("detected_column_count"), 0),
            _safe_int(diagnostics.get("estimated_column_count"), 1),
        )
        has_tables = _safe_bool(diagnostics.get("detected_has_table_interruptions", False)) or _safe_bool(
            diagnostics.get("has_tables_estimate", False)
        )
        has_footnotes = _safe_bool(diagnostics.get("detected_has_footnotes", False))
        layout_complexity = max(
            _safe_float(diagnostics.get("layout_complexity_score"), 0.0),
            0.75 if has_tables or has_footnotes else 0.0,
        )
        layout_threshold = _safe_float(thresholds.get("layout_complexity_threshold"), 0.55)
        layout_first = estimated_columns > 1 or has_tables or has_footnotes or layout_complexity >= layout_threshold

        hint_blob = " ".join(
            [
                str(language_hint or diagnostics.get("language_hint", "unknown")).strip().lower(),
                str(script_hint or diagnostics.get("script_hint", "unknown")).strip().lower(),
                str(document_type or diagnostics.get("document_type", "unknown")).strip().lower(),
                str(diagnostics.get("recommended_preprocessing_profile", "")).strip().lower(),
            ]
        )

        diacritic_heavy = any(
            token in hint_blob
            for token in (
                "diacritic",
                "transliteration",
                "small mark",
                "akkadian",
                "cuneiform",
            )
        ) or _safe_bool(diagnostics.get("prior_preprocessing_destroyed_chars", False))

        language_boosts, forced_profile, forced_strategy, applied_rules, rule_reasons = self._apply_language_rules(
            language_hint=language_hint,
            script_hint=script_hint,
            document_type=document_type,
            requested_profile=requested_profile_norm,
        )

        resolved_profile = resolve_preprocessing_profile(
            diagnostics=diagnostics,
            language_hint=language_hint,
            requested_profile=requested_profile_norm,
        )
        if forced_profile:
            resolved_profile = forced_profile
        if diacritic_heavy and requested_profile_norm == PROFILE_AUTO:
            resolved_profile = PROFILE_TRANSLITERATION_DIACRITIC
        if not resolved_profile:
            resolved_profile = default_profile_norm

        contrast = _safe_float(diagnostics.get("contrast_score"), 0.0)
        noise = _safe_float(diagnostics.get("noise_score"), 0.0)
        clean_page = (
            estimated_columns <= 1
            and not has_tables
            and layout_complexity < 0.35
            and contrast >= 0.12
            and noise <= 0.10
        )

        selected_strategy = "balanced_ocr"
        reason_codes: list[str] = []
        if use_text_layer:
            selected_strategy = "text_layer"
            reason_codes.append("text_layer_high_quality")
        elif forced_strategy:
            selected_strategy = forced_strategy
            reason_codes.append("strategy_forced_by_language_rule")
        elif diacritic_heavy:
            selected_strategy = "high_dpi_conservative"
            reason_codes.append("diacritic_or_transliteration_sensitive")
        elif layout_first:
            selected_strategy = "layout_first"
            reason_codes.append("layout_complexity")
        elif clean_page:
            selected_strategy = "fast_ocr"
            reason_codes.append("clean_simple_page")
        else:
            reason_codes.append("default_balanced_ocr")

        reason_codes.extend(rule_reasons)

        use_region_ocr = selected_strategy == "layout_first"
        use_full_page_ocr = selected_strategy != "text_layer"
        if selected_strategy == "layout_first" and not _safe_bool(self.config.get("layout_first_full_page_fallback", True)):
            use_full_page_ocr = False

        strict_mode = mode == "strict"
        beta_mode = mode == "beta"

        all_enabled = [str(engine).strip().lower() for engine in enabled_engines if str(engine).strip()]
        priority = [str(engine).strip().lower() for engine in self.config.get("engine_priority", []) if str(engine).strip()]
        candidate_order = priority + [engine for engine in all_enabled if engine not in priority]

        engines_skipped: dict[str, str] = {}
        health_timeout_limit = _safe_int(health_limits.get("max_consecutive_timeouts"), 2)
        health_failure_limit = _safe_int(health_limits.get("max_consecutive_failures"), 3)
        skip_unhealthy_in_strict = _safe_bool(health_limits.get("skip_unhealthy_in_strict", True))

        rank_scores: dict[str, float] = {}
        usable_engines: list[str] = []
        for index, engine in enumerate(candidate_order):
            if engine not in all_enabled:
                engines_skipped[engine] = "not_enabled"
                continue

            readiness = engine_readiness.get(engine, {})
            status = str(readiness.get("status", "")).strip().lower()
            if status == ENGINE_STATUS_DISABLED_BY_CONFIG:
                engines_skipped[engine] = "disabled_by_config"
                continue
            if status == ENGINE_STATUS_UNAVAILABLE_DEPENDENCY:
                engines_skipped[engine] = "unavailable_dependency"
                continue
            if status == ENGINE_STATUS_TIMED_OUT:
                engines_skipped[engine] = "startup_timeout"
                continue
            if strict_mode and skip_unhealthy_in_strict and status == ENGINE_STATUS_AVAILABLE_UNHEALTHY:
                engines_skipped[engine] = "unhealthy_in_strict_mode"
                continue

            perf = performance.get(engine, {})
            if _safe_int(perf.get("consecutive_timeouts"), 0) >= max(1, health_timeout_limit):
                engines_skipped[engine] = "repeated_timeouts"
                continue
            if _safe_int(perf.get("consecutive_failures"), 0) >= max(1, health_failure_limit):
                engines_skipped[engine] = "repeated_failures"
                continue

            usable_engines.append(engine)
            base_score = float(len(candidate_order) - index)
            rank_scores[engine] = base_score + language_boosts.get(engine, 0.0)

            success_rate = _safe_float(perf.get("success_rate"), 0.0)
            avg_conf = _safe_float(perf.get("avg_confidence"), 0.0)
            avg_runtime_ms = max(_safe_float(perf.get("avg_runtime_ms"), 0.0), 0.0)
            rank_scores[engine] += (success_rate * 1.5) + (avg_conf * 1.1)
            rank_scores[engine] += (1.0 / (1.0 + (avg_runtime_ms / 900.0)))

        strategy_boosts: list[str] = []
        if selected_strategy == "fast_ocr":
            strategy_boosts = [str(engine).strip().lower() for engine in self.config.get("fast_ocr_engines", []) if str(engine).strip()]
            use_region_ocr = False
        elif selected_strategy == "layout_first":
            strategy_boosts = [str(engine).strip().lower() for engine in self.config.get("layout_first_engines", []) if str(engine).strip()]
        elif selected_strategy == "high_dpi_conservative":
            strategy_boosts = [
                str(engine).strip().lower()
                for engine in self.config.get("diacritic_sensitive_engines", [])
                if str(engine).strip()
            ]

        for index, engine in enumerate(strategy_boosts):
            if engine in rank_scores:
                rank_scores[engine] += max(0.0, 2.0 - (index * 0.3))

        normalized_script = str(script_hint or "unknown").strip().lower()
        script_preferences = self.config.get("script_engine_preferences", {})
        script_priority = list(script_preferences.get(normalized_script, script_preferences.get("unknown", [])))
        for index, engine in enumerate(script_priority):
            normalized_engine = str(engine).strip().lower()
            if normalized_engine in rank_scores:
                rank_scores[normalized_engine] += max(0.0, 1.4 - (index * 0.2))

        ranked_engines = sorted(
            usable_engines,
            key=lambda engine: (-rank_scores.get(engine, 0.0), candidate_order.index(engine)),
        )

        max_engines_cfg = max(1, _safe_int(self.config.get("max_engines_per_page"), 2))
        max_fallback_cfg = max(0, _safe_int(self.config.get("max_fallback_engines"), 2))
        if beta_mode:
            max_engines_cfg += 1
            max_fallback_cfg += 1

        if selected_strategy == "text_layer":
            max_engines_cfg = 0
            max_fallback_cfg = 0
        elif selected_strategy == "fast_ocr":
            max_engines_cfg = 1
            max_fallback_cfg = 0

        primary_engine = ranked_engines[0] if ranked_engines else ""
        fallback_engines = ranked_engines[1 : 1 + max_fallback_cfg] if len(ranked_engines) > 1 else []

        if primary_engine and max_engines_cfg > 0:
            allowed_count = min(max_engines_cfg, 1 + len(fallback_engines))
            fallback_engines = fallback_engines[: max(0, allowed_count - 1)]

        use_ensemble = bool(selected_strategy in {"layout_first", "high_dpi_conservative", "balanced_ocr"} and (1 + len(fallback_engines)) > 1)
        if selected_strategy == "fast_ocr":
            use_ensemble = False

        per_engine_timeout_s = _safe_float(runtime_timeouts.get("per_engine_timeout_s"), 0.0)
        per_page_timeout_s = _safe_float(runtime_timeouts.get("per_page_timeout_s"), 0.0)

        return OCRRoutingStrategy(
            selected_strategy=selected_strategy,
            mode=mode,
            use_text_layer=use_text_layer,
            use_full_page_ocr=use_full_page_ocr,
            use_region_ocr=use_region_ocr,
            preprocessing_profile=resolved_profile,
            primary_engine=primary_engine,
            fallback_engines=fallback_engines,
            use_ensemble=use_ensemble,
            max_engines_per_page=max_engines_cfg,
            per_engine_timeout_s=per_engine_timeout_s,
            per_page_timeout_s=per_page_timeout_s,
            quality_thresholds=dict(thresholds),
            engines_skipped=engines_skipped,
            reason_codes=reason_codes,
            language_rules_applied=applied_rules,
        )
