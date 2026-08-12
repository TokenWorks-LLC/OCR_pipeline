"""Unit tests for multilingual quality scoring and launch gates."""

from __future__ import annotations

from production.quality_scoring import OCRQualityScorer


def _score_page(
    scorer: OCRQualityScorer,
    *,
    text: str,
    status: str = "success",
    failure_reason: str = "",
    metadata: dict[str, object] | None = None,
    engine_statuses: dict[str, object] | None = None,
    language_hint: str = "English",
    script_hint: str = "latin",
) -> dict[str, object]:
    return scorer.score_page(
        text=text,
        status=status,
        failure_reason=failure_reason,
        metadata=metadata or {},
        engine_statuses=engine_statuses or {"ensemble": {"status": "success"}},
        language_hint=language_hint,
        script_hint=script_hint,
    )


def test_empty_output_scores_failed_ocr() -> None:
    scorer = OCRQualityScorer()
    result = _score_page(scorer, text="", status="failed", failure_reason="empty_extraction")

    assert result["page_quality_score"] == 0.0
    assert result["quality_class"] == "failed_ocr"
    assert result["needs_human_review"] is True
    assert result["failed_gate"] == "page_quality_gate"
    assert result["gate_reason"] == "Empty OCR output"


def test_high_confidence_clean_output_is_production_quality() -> None:
    scorer = OCRQualityScorer()
    result = _score_page(
        scorer,
        text="This is a stable multilingual OCR output with consistent lexical tokens and rich content.",
        metadata={
            "confidence": 0.96,
            "adapter_used": "english",
            "lexicon_coverage": 0.95,
            "unknown_token_rate": 0.02,
            "engine_agreement_score": 0.95,
            "reading_order_confidence": 0.97,
            "postprocess_quality_score": 0.94,
            "recommended_preprocessing_profile": "clean_scan",
            "applied_preprocessing_profile": "clean_scan",
        },
    )

    assert float(result["page_quality_score"]) >= 0.85
    assert result["quality_class"] == "production_quality"
    assert result["needs_human_review"] is False


def test_low_confidence_output_scores_weak_ocr() -> None:
    scorer = OCRQualityScorer()
    result = _score_page(
        scorer,
        text="lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore",
        metadata={
            "confidence": 0.24,
            "adapter_used": "english",
            "lexicon_coverage": 0.45,
            "unknown_token_rate": 0.45,
            "engine_agreement_score": 0.35,
            "reading_order_confidence": 0.40,
            "postprocess_quality_score": 0.50,
            "recommended_preprocessing_profile": "clean_scan",
            "applied_preprocessing_profile": "unknown_safe_default",
        },
    )

    assert 0.50 <= float(result["page_quality_score"]) < 0.70
    assert result["quality_class"] == "weak_ocr"


def test_timeout_page_forces_review_and_fails_quality() -> None:
    scorer = OCRQualityScorer()
    result = _score_page(
        scorer,
        text="timeout output should be considered unstable",
        status="timed_out",
        failure_reason="page_timeout_1000ms",
        metadata={"confidence": 0.90},
        engine_statuses={"ensemble": {"status": "timed_out"}},
    )

    assert float(result["page_quality_score"]) <= 0.20
    assert result["quality_class"] == "failed_ocr"
    assert result["needs_human_review"] is True


def test_non_english_text_not_falsely_rejected() -> None:
    scorer = OCRQualityScorer()
    result = _score_page(
        scorer,
        text="هذا نص عربي للاختبار متعدد اللغات",
        language_hint="Arabic",
        script_hint="rtl",
        metadata={
            "confidence": 0.82,
            "engine_agreement_score": 0.74,
            "reading_order_confidence": 0.76,
            "postprocess_quality_score": 0.78,
        },
    )

    assert result["quality_class"] in {"production_quality", "usable_with_review", "weak_ocr"}
    assert result["quality_class"] != "failed_ocr"
    assert float(result["page_quality_score"]) >= 0.50


def test_adapter_specific_lexicon_metrics_apply_only_with_adapter() -> None:
    scorer = OCRQualityScorer()
    base_metadata = {
        "confidence": 0.80,
        "lexicon_coverage": 0.10,
        "engine_agreement_score": 0.70,
        "reading_order_confidence": 0.70,
    }
    without_adapter = _score_page(
        scorer,
        text="adapter neutral text sample for lexicon signal checks",
        metadata=base_metadata,
    )
    with_adapter = _score_page(
        scorer,
        text="adapter aware text sample for lexicon signal checks",
        metadata={**base_metadata, "adapter_used": "english"},
    )

    assert float(with_adapter["page_quality_score"]) < float(without_adapter["page_quality_score"])
    assert all("Low lexicon coverage" not in reason for reason in without_adapter["quality_reasons"])
    assert any("Low lexicon coverage" in reason for reason in with_adapter["quality_reasons"])


def test_launch_gate_production_fails_on_quality_threshold() -> None:
    scorer = OCRQualityScorer(gate_mode="production")
    result = scorer.evaluate_launch_gates(
        run_summary={
            "empty_page_rate": 0.01,
            "timeout_rate": 0.01,
            "failed_page_rate": 0.01,
            "average_quality_score": 0.65,
            "review_percentage": 0.20,
        },
        has_usable_engine=True,
        strict_readiness_ok=True,
    )

    assert result.should_fail_run is True
    assert result.failed_gate in {"avg_quality_gate", "review_rate_gate", "empty_page_rate_gate", "timeout_rate_gate", "failed_page_rate_gate"}


def test_launch_gate_beta_reviews_without_failing_for_same_metrics() -> None:
    scorer = OCRQualityScorer(gate_mode="beta")
    result = scorer.evaluate_launch_gates(
        run_summary={
            "empty_page_rate": 0.01,
            "timeout_rate": 0.01,
            "failed_page_rate": 0.01,
            "average_quality_score": 0.45,
            "review_percentage": 0.95,
        },
        has_usable_engine=True,
        strict_readiness_ok=True,
    )

    assert result.review_required is True
    assert result.should_fail_run is False
    assert result.failed_gate == ""


def test_launch_gate_strict_requires_readiness() -> None:
    scorer = OCRQualityScorer(gate_mode="strict")
    result = scorer.evaluate_launch_gates(
        run_summary={
            "empty_page_rate": 0.0,
            "timeout_rate": 0.0,
            "failed_page_rate": 0.0,
            "average_quality_score": 0.95,
            "review_percentage": 0.10,
        },
        has_usable_engine=True,
        strict_readiness_ok=False,
    )

    assert result.should_fail_run is True
    assert result.failed_gate == "strict_readiness_gate"


def _clean_run_summary() -> dict[str, float]:
    return {
        "empty_page_rate": 0.0,
        "timeout_rate": 0.0,
        "failed_page_rate": 0.0,
        "average_quality_score": 0.65,
        "review_percentage": 0.0,
    }


def test_launch_gate_skips_engine_gate_when_ocr_not_required() -> None:
    # A healthy text-layer-only run (no page needed OCR) must not be failed just
    # because no OCR engine is installed. Regression guard for the CI break where
    # a fully text-layer run tripped engine_availability_gate.
    scorer = OCRQualityScorer(gate_mode="internal")
    result = scorer.evaluate_launch_gates(
        run_summary=_clean_run_summary(),
        has_usable_engine=False,
        strict_readiness_ok=True,
        ocr_required=False,
    )

    assert result.should_fail_run is False
    assert result.failed_gate == ""
    assert all("engine" not in reason.lower() for reason in result.reasons)


def test_launch_gate_fires_engine_gate_when_ocr_required() -> None:
    # When the run actually depended on OCR but no engine was usable, the gate
    # must still fail the run.
    scorer = OCRQualityScorer(gate_mode="internal")
    result = scorer.evaluate_launch_gates(
        run_summary=_clean_run_summary(),
        has_usable_engine=False,
        strict_readiness_ok=True,
        ocr_required=True,
    )

    assert result.should_fail_run is True
    assert result.failed_gate == "engine_availability_gate"


def test_launch_gate_engine_gate_defaults_to_required() -> None:
    # Backward compatibility: callers that do not pass ocr_required keep the
    # original fail-closed behavior when no engine is usable.
    scorer = OCRQualityScorer(gate_mode="internal")
    result = scorer.evaluate_launch_gates(
        run_summary=_clean_run_summary(),
        has_usable_engine=False,
        strict_readiness_ok=True,
    )

    assert result.should_fail_run is True
    assert result.failed_gate == "engine_availability_gate"
