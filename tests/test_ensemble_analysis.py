from __future__ import annotations

import math
import time

from PIL import Image

from production.ensemble_analysis import analyze_alignment, char_disagreement_rate, explain_consensus
from production.ensemble_ocr import (
    ENGINE_STATUS_AVAILABLE,
    ENGINE_STATUS_TIMED_OUT,
    FortifiedOCREnsemble,
    OCRCandidate,
    PageImageVariants,
)


def test_alignment_identical_outputs_has_zero_disagreement() -> None:
    metrics = analyze_alignment(
        engine_texts={"paddle": "same output", "doctr": "same output"},
        consensus_text="same output",
    )

    assert metrics["char_disagreement_rate"] == 0.0
    assert metrics["token_disagreement_rate"] == 0.0
    assert metrics["line_disagreement_rate"] == 0.0
    assert metrics["engine_agreement_score"] == 1.0


def test_alignment_minor_character_disagreement_is_measurable() -> None:
    metrics = analyze_alignment(
        engine_texts={
            "paddle": "The evidence is clear.",
            "doctr": "The evideuce is clear.",
        },
        consensus_text="The evidence is clear.",
    )

    assert 0.0 < metrics["char_disagreement_rate"] < 0.2
    assert metrics["engine_agreement_score"] < 1.0


def test_explain_consensus_marks_empty_engine_output_uncertain() -> None:
    explanation = explain_consensus(
        per_engine_outputs=[
            {"engine": "paddle", "text": "", "confidence": 0.0, "runtime_ms": 0.0},
            {"engine": "doctr", "text": "", "confidence": 0.0, "runtime_ms": 0.0},
        ],
        consensus_text="",
        alignment_metrics={
            "engine_agreement_score": 0.0,
            "consensus_entropy": 1.0,
            "disagreement_positions": [],
        },
    )

    assert explanation["uncertain"] is True
    assert explanation["low_quality_all_engines"] is True
    assert explanation["human_review_recommended"] is True
    assert "no_candidate_text" in explanation["reason_codes"]


def test_collect_candidates_records_timeout_per_engine_output() -> None:
    class SlowBackend:
        name = "slow"
        _failed_reason = None

        def infer(self, variants: PageImageVariants) -> OCRCandidate:
            time.sleep(0.05)
            return OCRCandidate("slow", "late text", confidence=0.95)

    ensemble = FortifiedOCREnsemble.__new__(FortifiedOCREnsemble)
    ensemble.per_engine_timeout_s = 0.001
    ensemble.backends = [SlowBackend()]
    ensemble.engine_readiness = {"slow": {"status": ENGINE_STATUS_AVAILABLE, "reason": ""}}
    ensemble.engine_runtime_state = {}
    ensemble._record_engine_observation = lambda *args, **kwargs: None  # type: ignore[assignment]

    variants = PageImageVariants(Image.new("L", (80, 40), color=255))
    candidates, _, statuses, trace = FortifiedOCREnsemble._collect_candidates(
        ensemble,
        variants,
        context={"page_number": 1, "language_hint": "English", "script_hint": "latin"},
    )

    assert candidates == []
    assert statuses["slow"]["status"] == ENGINE_STATUS_TIMED_OUT
    assert any(
        output.get("engine") == "slow"
        and output.get("status") == ENGINE_STATUS_TIMED_OUT
        and output.get("timed_out") is True
        for output in trace.get("per_engine_outputs", [])
    )


def test_diacritic_disagreement_is_tolerated_more_in_transliteration_context() -> None:
    generic_rate = char_disagreement_rate("sarrum", "\u0161arrum")
    translit_rate = char_disagreement_rate(
        "sarrum",
        "\u0161arrum",
        language_hint="Akkadian transliteration",
        script_hint="latin_diacritic",
        preprocessing_profile="akkadian_strict",
    )

    assert 0.0 < translit_rate < generic_rate


def test_explain_consensus_flags_low_confidence_ensemble() -> None:
    explanation = explain_consensus(
        per_engine_outputs=[
            {"engine": "paddle", "text": "Readable but uncertain text", "confidence": 0.11, "runtime_ms": 80.0},
            {"engine": "doctr", "text": "Readable but uncertain text", "confidence": 0.09, "runtime_ms": 92.0},
        ],
        consensus_text="Readable but uncertain text",
        alignment_metrics={
            "engine_agreement_score": 0.98,
            "consensus_entropy": 0.05,
            "disagreement_positions": [],
        },
        quality_thresholds={"consensus_min_confidence": 0.30},
    )

    assert explanation["uncertain"] is True
    assert "low_average_confidence" in explanation["reason_codes"]


def test_alignment_supports_multilingual_unicode_text() -> None:
    metrics = analyze_alignment(
        engine_texts={
            "paddle": "English \u0646\u0635 \u4e2d\u6587",
            "doctr": "English \u0646\u0635 \u4e2d\u6587",
            "mmocr": "English \u0646\u0635 \u4e2d\u6587.",
        },
        consensus_text="English \u0646\u0635 \u4e2d\u6587",
        language_hint="multilingual",
        script_hint="mixed",
    )

    assert math.isfinite(metrics["char_disagreement_rate"])
    assert math.isfinite(metrics["token_disagreement_rate"])
    assert math.isfinite(metrics["line_disagreement_rate"])
    assert 0.0 <= metrics["engine_agreement_score"] <= 1.0
