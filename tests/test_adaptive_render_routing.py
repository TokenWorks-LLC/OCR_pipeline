from __future__ import annotations

from tools.gold_registry.run_adaptive_render_strategy import _compute_adaptive_signals, _decide_adaptive_trigger


def test_compute_adaptive_signals_triggers_multiple_reasons() -> None:
    default_row = {
        "output_text_length": 8,
        "failed": "true",
        "empty": "false",
        "document_type": "receipts_commercial_docs",
        "layout_type": "semi_structured",
        "scan_quality": "mixed",
    }
    meta = {
        "confidence": 0.15,
        "final_output_source": "none",
    }

    reasons = _compute_adaptive_signals(default_row, meta)

    assert "first_pass_failed" in reasons
    assert "suspicious_short_output" in reasons
    assert "low_confidence" in reasons
    assert "noisy_or_low_quality_scan" in reasons


def test_compute_adaptive_signals_skips_clean_pages() -> None:
    default_row = {
        "output_text_length": 240,
        "failed": "false",
        "empty": "false",
        "document_type": "academic_article",
        "layout_type": "single_column",
        "scan_quality": "clean_scan",
    }
    meta = {
        "confidence": 0.88,
        "final_output_source": "full_page_ocr",
    }

    reasons = _compute_adaptive_signals(default_row, meta)

    assert reasons == []


def test_decide_adaptive_trigger_triggered() -> None:
    default_row = {"failed": "true", "empty": "false", "output_text_length": 12}
    ok, why = _decide_adaptive_trigger(
        reasons=["first_pass_failed", "suspicious_short_output"],
        default_row=default_row,
        min_signals=2,
        used_pages=3,
        max_pages=10,
        ratio_limit=8,
    )
    assert ok is True
    assert why == ""


def test_decide_adaptive_trigger_budget_capped() -> None:
    default_row = {"failed": "true", "empty": "false", "output_text_length": 12}
    ok, why = _decide_adaptive_trigger(
        reasons=["first_pass_failed", "suspicious_short_output"],
        default_row=default_row,
        min_signals=2,
        used_pages=10,
        max_pages=10,
        ratio_limit=100,
    )
    assert ok is False
    assert why == "adaptive_page_budget_exhausted"


def test_decide_adaptive_trigger_ratio_capped() -> None:
    default_row = {"failed": "true", "empty": "false", "output_text_length": 12}
    ok, why = _decide_adaptive_trigger(
        reasons=["first_pass_failed", "suspicious_short_output"],
        default_row=default_row,
        min_signals=2,
        used_pages=4,
        max_pages=10,
        ratio_limit=4,
    )
    assert ok is False
    assert why == "adaptive_ratio_budget_exhausted"


def test_decide_adaptive_trigger_requires_fallback_signal() -> None:
    default_row = {"failed": "false", "empty": "false", "output_text_length": 120}
    ok, why = _decide_adaptive_trigger(
        reasons=["low_confidence", "high_layout_complexity"],
        default_row=default_row,
        min_signals=2,
        used_pages=1,
        max_pages=10,
        ratio_limit=10,
    )
    assert ok is False
    assert why == "fallback_only_no_failure_signal"
