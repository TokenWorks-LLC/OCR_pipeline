from __future__ import annotations

import json
import time
from pathlib import Path

import fitz

from production.ensemble_ocr import ENGINE_STATUS_AVAILABLE, ENGINE_STATUS_TIMED_OUT, FortifiedOCREnsemble, OCRCandidate
from production.ocr_strategy import OCRStrategySelector
from production.preprocessing_profiles import PROFILE_TRANSLITERATION_DIACRITIC


def _write_pdf_with_text(pdf_path: Path, text: str) -> None:
    doc = fitz.open()
    page = doc.new_page(width=320, height=220)
    page.insert_text((24, 52), text, fontsize=11)
    doc.save(str(pdf_path))
    doc.close()


def _write_light_profile(profile_path: Path) -> None:
    payload = {
        "rendering": {"dpi": 300},
        "engines": {"enabled": []},
        "routing": {
            "mode": "strict",
            "max_engines_per_page": 2,
            "max_fallback_engines": 2,
            "quality_thresholds": {
                "min_confidence": 0.0,
                "min_text_chars": 1,
                "min_alnum_ratio": 0.0,
                "text_layer_min_chars": 24,
                "layout_complexity_threshold": 0.55,
            },
        },
    }
    profile_path.write_text(json.dumps(payload), encoding="utf-8")


def test_text_layer_accepted_by_strategy_selector():
    selector = OCRStrategySelector({"mode": "strict"})
    strategy = selector.select(
        diagnostics={
            "text_layer_char_count": 180,
            "text_layer_usable": True,
            "text_layer_suspicious_reasons": "",
            "estimated_column_count": 1,
            "layout_complexity_score": 0.2,
            "contrast_score": 0.2,
            "noise_score": 0.04,
        },
        engine_readiness={"paddle": {"status": "available", "reason": ""}},
        enabled_engines=["paddle"],
        language_hint="unknown",
        force_ocr=False,
        prefer_text_layer=True,
    )

    assert strategy.selected_strategy == "text_layer"
    assert strategy.use_text_layer is True


def test_text_layer_rejected_when_unicode_suspicious():
    selector = OCRStrategySelector({"mode": "strict"})
    strategy = selector.select(
        diagnostics={
            "text_layer_char_count": 220,
            "text_layer_usable": True,
            "text_layer_suspicious_reasons": "broken_unicode",
            "estimated_column_count": 1,
            "layout_complexity_score": 0.2,
            "contrast_score": 0.2,
            "noise_score": 0.04,
        },
        engine_readiness={"paddle": {"status": "available", "reason": ""}},
        enabled_engines=["paddle"],
        language_hint="unknown",
        force_ocr=False,
        prefer_text_layer=True,
    )

    assert strategy.use_text_layer is False
    assert strategy.selected_strategy != "text_layer"


def test_unhealthy_engine_skipped_in_strict_mode():
    selector = OCRStrategySelector({"mode": "strict"})
    strategy = selector.select(
        diagnostics={"layout_complexity_score": 0.4, "contrast_score": 0.12, "noise_score": 0.11},
        engine_readiness={
            "paddle": {"status": "available_but_unhealthy", "reason": "shim"},
            "doctr": {"status": "available", "reason": ""},
        },
        enabled_engines=["paddle", "doctr"],
        language_hint="unknown",
        force_ocr=True,
        prefer_text_layer=False,
    )

    assert strategy.primary_engine == "doctr"
    assert strategy.engines_skipped.get("paddle") == "unhealthy_in_strict_mode"


def test_layout_first_strategy_selected_for_complex_layout():
    selector = OCRStrategySelector({"mode": "strict"})
    strategy = selector.select(
        diagnostics={
            "estimated_column_count": 2,
            "has_tables_estimate": True,
            "layout_complexity_score": 0.8,
            "contrast_score": 0.14,
            "noise_score": 0.1,
        },
        engine_readiness={"paddle": {"status": "available", "reason": ""}},
        enabled_engines=["paddle"],
        language_hint="unknown",
        force_ocr=True,
        prefer_text_layer=False,
    )

    assert strategy.selected_strategy == "layout_first"
    assert strategy.use_region_ocr is True


def test_diacritic_heavy_strategy_selected_from_metadata_hint():
    selector = OCRStrategySelector({"mode": "strict", "enable_akkadian_rule": True})
    strategy = selector.select(
        diagnostics={
            "recommended_preprocessing_profile": PROFILE_TRANSLITERATION_DIACRITIC,
            "layout_complexity_score": 0.3,
            "contrast_score": 0.13,
            "noise_score": 0.09,
        },
        engine_readiness={
            "cured": {"status": "available", "reason": ""},
            "kraken": {"status": "available", "reason": ""},
        },
        enabled_engines=["cured", "kraken"],
        language_hint="transliteration",
        force_ocr=True,
        prefer_text_layer=False,
    )

    assert strategy.selected_strategy == "high_dpi_conservative"
    assert strategy.preprocessing_profile == PROFILE_TRANSLITERATION_DIACRITIC


def test_fallback_engine_used_when_primary_empty(tmp_path: Path):
    pdf_path = tmp_path / "routing_fallback.pdf"
    profile_path = tmp_path / "routing_profile.json"
    _write_pdf_with_text(pdf_path, "routing fallback test")
    _write_light_profile(profile_path)

    class EmptyPrimary:
        name = "paddle"
        _failed_reason = None

        def infer(self, variants):
            return None

    class FallbackDoctr:
        name = "doctr"
        _failed_reason = None

        def infer(self, variants):
            return OCRCandidate(
                "doctr",
                "Fallback engine text",
                confidence=0.82,
                variant="original",
                meta={"rotation_angle": variants.rotation_angle, "line_count": 1},
            )

    ensemble = FortifiedOCREnsemble(profile_path=str(profile_path))
    ensemble.backends = [EmptyPrimary(), FallbackDoctr()]
    ensemble.backends_by_name = {backend.name: backend for backend in ensemble.backends}
    ensemble.enabled_engines = ["paddle", "doctr"]
    ensemble._set_engine_readiness("paddle", ENGINE_STATUS_AVAILABLE, "")
    ensemble._set_engine_readiness("doctr", ENGINE_STATUS_AVAILABLE, "")
    ensemble.rotation_search = (0,)

    text, meta = ensemble.extract_page_text(
        str(pdf_path),
        0,
        diagnostics={
            "layout_complexity_score": 0.45,
            "estimated_column_count": 1,
            "contrast_score": 0.11,
            "noise_score": 0.11,
        },
        language_hint="unknown",
    )

    assert text == "Fallback engine text"
    assert meta.get("engines_attempted") == ["paddle", "doctr"]


def test_per_engine_timeout_respected_and_fallback_succeeds(tmp_path: Path):
    pdf_path = tmp_path / "routing_timeout.pdf"
    profile_path = tmp_path / "routing_profile.json"
    _write_pdf_with_text(pdf_path, "routing timeout test")
    _write_light_profile(profile_path)

    class SlowPrimary:
        name = "paddle"
        _failed_reason = None

        def infer(self, variants):
            time.sleep(0.15)
            return OCRCandidate(
                "paddle",
                "too slow",
                confidence=0.3,
                variant="original",
                meta={"rotation_angle": variants.rotation_angle, "line_count": 1},
            )

    class FastFallback:
        name = "doctr"
        _failed_reason = None

        def infer(self, variants):
            return OCRCandidate(
                "doctr",
                "Fast fallback text",
                confidence=0.86,
                variant="original",
                meta={"rotation_angle": variants.rotation_angle, "line_count": 1},
            )

    ensemble = FortifiedOCREnsemble(profile_path=str(profile_path))
    ensemble.backends = [SlowPrimary(), FastFallback()]
    ensemble.backends_by_name = {backend.name: backend for backend in ensemble.backends}
    ensemble.enabled_engines = ["paddle", "doctr"]
    ensemble._set_engine_readiness("paddle", ENGINE_STATUS_AVAILABLE, "")
    ensemble._set_engine_readiness("doctr", ENGINE_STATUS_AVAILABLE, "")
    ensemble.rotation_search = (0,)

    text, meta = ensemble.extract_page_text(
        str(pdf_path),
        0,
        diagnostics={
            "layout_complexity_score": 0.45,
            "estimated_column_count": 1,
            "contrast_score": 0.11,
            "noise_score": 0.11,
        },
        language_hint="unknown",
        timeout_config={"per_engine_timeout_s": 0.01},
    )

    assert text == "Fast fallback text"
    paddle_status = (meta.get("engine_page_statuses", {}).get("paddle") or {}).get("status")
    assert paddle_status == ENGINE_STATUS_TIMED_OUT
