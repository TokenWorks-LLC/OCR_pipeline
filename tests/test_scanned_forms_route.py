from __future__ import annotations

from production.scanned_forms_route import DEFAULT_ROUTE, ROUTE_VERSION, SCANNED_FORMS_ROUTE, apply_scanned_forms_route


def test_scanned_forms_route_selected_when_policy_allows() -> None:
    result = apply_scanned_forms_route(
        page_id="scan_doc_page_1",
        document_type="scanned_forms",
        script_type="Latn",
        raw_ocr_text="FAX NO 1 ) 2 TO: John Doe",
        review_state="needs_review",
        target="private_beta",
        route_policy_enabled=True,
    )

    assert result.selected_route == SCANNED_FORMS_ROUTE
    assert result.route_reason == "scanned_forms_route_selected"
    assert result.cleanup_applied is True
    assert result.review_required is True


def test_non_scanned_forms_documents_do_not_use_route() -> None:
    result = apply_scanned_forms_route(
        page_id="book_doc_page_3",
        document_type="historical_printed_books",
        script_type="Latn",
        raw_ocr_text="Regular page text",
        review_state="needs_review",
        target="private_beta",
    )

    assert result.selected_route == DEFAULT_ROUTE
    assert result.cleanup_applied is False
    assert result.final_display_text == result.raw_ocr_text


def test_receipts_do_not_use_scanned_forms_route() -> None:
    result = apply_scanned_forms_route(
        page_id="receipt_doc_page_1",
        document_type="receipts_commercial_docs",
        script_type="Latn",
        raw_ocr_text="TOTAL: 10.00",
        review_state="needs_review",
        target="private_beta",
    )

    assert result.selected_route == DEFAULT_ROUTE
    assert result.route_reason == "excluded_receipts_document_type"
    assert result.cleanup_applied is False
    assert result.export_allowed is False
    assert result.export_block_reason == "excluded_category_or_script"


def test_latn_hang_script_does_not_enter_beta_evidence() -> None:
    result = apply_scanned_forms_route(
        page_id="scan_latn_hang_page_2",
        document_type="scanned_forms",
        script_type="Latn,Hang",
        raw_ocr_text="mixed script",
        review_state="accepted",
        target="private_beta",
    )

    assert result.selected_route == DEFAULT_ROUTE
    assert result.route_reason == "script_excluded_by_policy"
    assert result.export_allowed is False
    assert result.export_block_reason == "excluded_category_or_script"


def test_raw_ocr_is_preserved_when_cleanup_applies() -> None:
    raw = "FAX NO 3 ) 4 TO: Alice"
    result = apply_scanned_forms_route(
        page_id="scan_raw_preserve_page_4",
        document_type="scanned_forms",
        script_type="Latn",
        raw_ocr_text=raw,
        review_state="needs_review",
        target="private_beta",
    )

    assert result.raw_ocr_text == raw
    assert result.raw_ocr_preserved is True
    assert result.structured_cleanup_text != raw


def test_mandatory_review_is_enforced() -> None:
    result = apply_scanned_forms_route(
        page_id="scan_review_page_5",
        document_type="scanned_forms",
        script_type="Latn",
        raw_ocr_text="DATE: 2026-05-23",
        review_state="needs_review",
        target="private_beta",
    )

    assert result.review_required is True


def test_export_blocked_before_review() -> None:
    result = apply_scanned_forms_route(
        page_id="scan_export_gate_page_6",
        document_type="scanned_forms",
        script_type="Latn",
        raw_ocr_text="PAGE: 2",
        review_state="needs_review",
        target="private_beta",
    )

    assert result.export_allowed is False
    assert result.export_block_reason in {"review_not_completed", "internal_only_category"}


def test_route_audit_trail_contains_cleanup_decision() -> None:
    result = apply_scanned_forms_route(
        page_id="scan_audit_page_7",
        document_type="scanned_forms",
        script_type="Latn",
        raw_ocr_text="FAX NO 5 ) 8 FROM: Team",
        review_state="needs_review",
        target="private_beta",
    )

    audit = result.route_audit
    assert audit.get("selected_route") == SCANNED_FORMS_ROUTE
    assert "cleanup_applied" in audit
    assert "cleanup_reasons" in audit
    assert "export_block_reason" in audit
    assert result.route_policy_version == ROUTE_VERSION


def test_cleanup_failure_falls_back_and_blocks_export() -> None:
    result = apply_scanned_forms_route(
        page_id="scan_cleanup_fail_page_8",
        document_type="scanned_forms",
        script_type="Latn",
        raw_ocr_text="TO: Bob",
        review_state="accepted",
        target="internal",
        fail_cleanup=True,
    )

    assert result.selected_route == SCANNED_FORMS_ROUTE
    assert result.route_reason == "cleanup_failed_fallback_to_raw"
    assert result.cleanup_applied is False
    assert result.final_display_text == result.raw_ocr_text
    assert result.export_allowed is False
    assert result.export_block_reason in {"route_cleanup_failed_requires_manual_review", "internal_only_category"}
