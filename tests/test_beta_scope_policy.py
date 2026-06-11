from __future__ import annotations

from production.beta_scope_policy import (
    STATUS_ALLOW_REVIEW_ONLY,
    STATUS_EXCLUDED,
    STATUS_INTERNAL_ONLY,
    apply_beta_scope_to_row,
    evaluate_beta_scope,
    resolve_beta_scope_status,
)


def test_script_exclusion_overrides_document_allowlist() -> None:
    status, reason = resolve_beta_scope_status(
        document_type="typewritten_historical_directories",
        script_type="Latn,Hang",
    )
    assert status == STATUS_EXCLUDED
    assert reason == "script_policy_exclusion_latn_hang"


def test_receipts_are_excluded_from_private_beta() -> None:
    decision = evaluate_beta_scope(
        document_type="receipts_commercial_docs",
        script_type="Latn",
        review_state="accepted",
        target="private_beta",
    )
    assert decision.beta_scope_status == STATUS_EXCLUDED
    assert decision.export_allowed is False
    assert decision.export_block_reason == "excluded_category_or_script"


def test_typewritten_requires_review_before_export() -> None:
    decision_not_reviewed = evaluate_beta_scope(
        document_type="typewritten_historical_directories",
        script_type="Latn",
        review_state="needs_review",
        target="private_beta",
    )
    assert decision_not_reviewed.beta_scope_status == STATUS_ALLOW_REVIEW_ONLY
    assert decision_not_reviewed.export_allowed is False
    assert decision_not_reviewed.export_block_reason == "review_not_completed"

    decision_reviewed = evaluate_beta_scope(
        document_type="typewritten_historical_directories",
        script_type="Latn",
        review_state="corrected",
        target="private_beta",
    )
    assert decision_reviewed.beta_scope_status == STATUS_ALLOW_REVIEW_ONLY
    assert decision_reviewed.export_allowed is True
    assert decision_reviewed.export_block_reason == ""


def test_internal_only_categories_are_blocked_for_private_beta() -> None:
    decision = evaluate_beta_scope(
        document_type="historical_printed_books",
        script_type="Latn",
        review_state="accepted",
        target="private_beta",
    )
    assert decision.beta_scope_status == STATUS_INTERNAL_ONLY
    assert decision.export_allowed is False
    assert decision.export_block_reason == "internal_only_category"


def test_apply_scope_to_row_includes_required_fields() -> None:
    row = {
        "document_type": "scanned_forms",
        "script_type": "Latn",
    }
    enriched = apply_beta_scope_to_row(row, review_state="accepted", target="private_beta")
    for key in {
        "beta_scope_status",
        "beta_scope_reason",
        "review_required",
        "export_allowed",
        "export_block_reason",
        "category_policy_version",
    }:
        assert key in enriched
