from __future__ import annotations

from production.reviewer_category_adapter import (
    ADAPTER_POLICY_VERSION,
    DEFAULT_ADAPTER_NAME,
    apply_reviewer_category_adapter,
)


def test_non_target_category_keeps_text_unchanged() -> None:
    raw = "Line one\nLine two"
    result = apply_reviewer_category_adapter(
        page_id="page_1",
        document_type="scanned_forms",
        raw_ocr_text=raw,
        target_category="typewritten_historical_directories",
    )

    assert result.adapter_applied is False
    assert result.route_reason == "non_target_category"
    assert result.corrected_text == result.cleaned_text
    assert result.corrected_text == "Line one\nLine two"


def test_target_category_applies_conservative_line_joins() -> None:
    raw = (
        "This paragraph was wrapped in\n"
        "the middle of a sentence and\n"
        "should remain coherent.\n\n"
        "A hyphen-\n"
        "ated token should be joined."
    )
    result = apply_reviewer_category_adapter(
        page_id="page_2",
        document_type="typewritten_historical_directories",
        raw_ocr_text=raw,
        target_category="typewritten_historical_directories",
    )

    assert result.adapter_applied is True
    assert result.route_reason == "adapter_applied"
    assert "wrapped in the middle" in result.corrected_text
    assert "hyphenated" in result.corrected_text
    assert result.edit_trace
    assert result.audit.get("edit_count", 0) >= 2


def test_policy_disable_prevents_adapter_application() -> None:
    raw = "Line A\nline b continues"
    result = apply_reviewer_category_adapter(
        page_id="page_3",
        document_type="typewritten_historical_directories",
        raw_ocr_text=raw,
        target_category="typewritten_historical_directories",
        policy_enabled=False,
    )

    assert result.adapter_applied is False
    assert result.route_reason == "adapter_disabled_by_policy"
    assert result.corrected_text == result.cleaned_text


def test_insufficient_signal_does_not_force_edits() -> None:
    raw = "Header:\nSection"
    result = apply_reviewer_category_adapter(
        page_id="page_4",
        document_type="typewritten_historical_directories",
        raw_ocr_text=raw,
        target_category="typewritten_historical_directories",
    )

    assert result.adapter_applied is False
    assert result.route_reason in {"insufficient_pattern_signal", "no_change_after_adapter"}


def test_audit_contains_policy_and_adapter_identity() -> None:
    raw = "First line\nsecond line that should join\nthird line continues"
    result = apply_reviewer_category_adapter(
        page_id="page_5",
        document_type="typewritten_historical_directories",
        raw_ocr_text=raw,
        target_category="typewritten_historical_directories",
    )

    assert result.audit.get("selected_adapter") == DEFAULT_ADAPTER_NAME
    assert result.audit.get("policy_version") == ADAPTER_POLICY_VERSION
    assert result.policy_version == ADAPTER_POLICY_VERSION
