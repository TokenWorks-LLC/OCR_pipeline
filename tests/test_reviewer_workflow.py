from __future__ import annotations

import pytest

from production.reviewer_workflow import (
    REVIEW_STATE_ACCEPTED,
    REVIEW_STATE_BLOCKED,
    REVIEW_STATE_CORRECTED,
    REVIEW_STATE_EXPORTED,
    REVIEW_STATE_NEEDS_REVIEW,
    REVIEW_STATE_REJECTED,
    ReviewWorkflowError,
    ReviewerWorkflowStore,
)


def _sample_row(
    *,
    page_id: str,
    text: str,
    beta_scope_status: str = "allow_private_beta_with_mandatory_review",
) -> dict[str, object]:
    return {
        "page_id": page_id,
        "document_id": page_id.split("_page_", 1)[0],
        "raw_ocr_text": text,
        "status": "success",
        "quality_class": "weak_ocr",
        "needs_human_review": True,
        "beta_scope_status": beta_scope_status,
        "beta_scope_reason": "test_scope",
    }


def test_queue_creation_and_unassigned_queue() -> None:
    store = ReviewerWorkflowStore()
    store.create_review_queue(
        [
            _sample_row(page_id="doc_a_page_1", text="raw a"),
            _sample_row(page_id="doc_b_page_1", text="raw b"),
        ]
    )

    assert len(store.items) == 2
    assert all(item.review_state == REVIEW_STATE_NEEDS_REVIEW for item in store.items.values())
    assert len(store.get_unassigned_queue()) == 2


def test_low_quality_ocr_cannot_export_without_review() -> None:
    store = ReviewerWorkflowStore()
    store.create_review_queue(
        [
            {
                "page_id": "doc_low_quality_page_1",
                "document_id": "doc_low_quality",
                "raw_ocr_text": "uncertain ocr text",
                "status": "success",
                "quality_class": "weak_ocr",
                "needs_human_review": True,
                "beta_scope_status": "allow_private_beta_with_mandatory_review",
                "beta_scope_reason": "test_scope",
            }
        ]
    )

    allowed, reason = store.can_export("doc_low_quality_page_1", target="private_beta")
    assert allowed is False
    assert reason == "review_not_completed"


def test_accept_flow_and_export_for_allowed_scope() -> None:
    store = ReviewerWorkflowStore()
    store.create_review_queue([_sample_row(page_id="doc_accept_page_1", text="raw text")])

    store.assign_reviewer("doc_accept_page_1", "reviewer_1")
    store.start_review("doc_accept_page_1", actor_id="reviewer_1")
    accepted = store.accept_ocr("doc_accept_page_1", actor_id="reviewer_1", reason="looks_good")

    assert accepted.review_state == REVIEW_STATE_ACCEPTED

    allowed, reason = store.can_export("doc_accept_page_1", target="private_beta")
    assert allowed is True
    assert reason == ""

    exported = store.mark_exported("doc_accept_page_1", actor_id="reviewer_1")
    assert exported.review_state == REVIEW_STATE_EXPORTED


def test_correction_preserves_raw_and_stores_corrected_text() -> None:
    store = ReviewerWorkflowStore()
    store.create_review_queue([_sample_row(page_id="doc_correct_page_1", text="orig txt")])

    store.start_review("doc_correct_page_1", actor_id="reviewer_2")
    corrected = store.correct_ocr(
        "doc_correct_page_1",
        corrected_text="corrected txt",
        actor_id="reviewer_2",
        reason="fixed spelling",
    )

    assert corrected.review_state == REVIEW_STATE_CORRECTED
    assert corrected.raw_ocr_text == "orig txt"
    assert corrected.corrected_ocr_text == "corrected txt"

    correction_events = [e for e in store.audit_events if e.action == "corrected"]
    assert len(correction_events) == 1
    assert correction_events[0].before_text == "orig txt"
    assert correction_events[0].after_text == "corrected txt"


def test_override_quality_warning_requires_reason() -> None:
    store = ReviewerWorkflowStore()
    store.create_review_queue([_sample_row(page_id="doc_override_page_1", text="raw")])

    with pytest.raises(ReviewWorkflowError):
        store.override_quality_warning("doc_override_page_1", actor_id="reviewer_3", reason="")

    updated = store.override_quality_warning(
        "doc_override_page_1",
        actor_id="reviewer_3",
        reason="domain_specific_format_is_expected",
    )
    assert updated.quality_warning_overridden is True
    assert updated.quality_override_reason == "domain_specific_format_is_expected"


def test_reject_and_block_states_cannot_export() -> None:
    store = ReviewerWorkflowStore()
    store.create_review_queue([_sample_row(page_id="doc_reject_page_1", text="bad")])
    store.start_review("doc_reject_page_1", actor_id="reviewer_4")
    rejected = store.reject_ocr("doc_reject_page_1", actor_id="reviewer_4", reason="unreadable")

    assert rejected.review_state == REVIEW_STATE_REJECTED
    allowed, reason = store.can_export("doc_reject_page_1", target="private_beta")
    assert allowed is False
    assert reason == "review_not_completed"

    store.create_review_queue([_sample_row(page_id="doc_block_page_1", text="corrupt")])
    store.start_review("doc_block_page_1", actor_id="reviewer_4")
    blocked = store.mark_page_unusable("doc_block_page_1", actor_id="reviewer_4", reason="source_corrupt")
    assert blocked.review_state == REVIEW_STATE_BLOCKED


def test_export_gate_blocks_internal_and_excluded_categories() -> None:
    store = ReviewerWorkflowStore()
    store.create_review_queue(
        [
            _sample_row(
                page_id="doc_internal_page_1",
                text="ok",
                beta_scope_status="internal_only_with_review",
            ),
            _sample_row(
                page_id="doc_excluded_page_1",
                text="ok",
                beta_scope_status="exclude_from_private_beta",
            ),
        ]
    )

    store.start_review("doc_internal_page_1", actor_id="reviewer_5")
    store.accept_ocr("doc_internal_page_1", actor_id="reviewer_5")
    allowed_internal, reason_internal = store.can_export("doc_internal_page_1", target="private_beta")
    assert allowed_internal is False
    assert reason_internal == "category_internal_only"

    store.start_review("doc_excluded_page_1", actor_id="reviewer_5")
    store.accept_ocr("doc_excluded_page_1", actor_id="reviewer_5")
    allowed_excluded, reason_excluded = store.can_export("doc_excluded_page_1", target="private_beta")
    assert allowed_excluded is False
    assert reason_excluded == "category_excluded_from_private_beta"


def test_audit_trail_includes_assignment_review_and_export() -> None:
    store = ReviewerWorkflowStore()
    store.create_review_queue([_sample_row(page_id="doc_audit_page_1", text="raw")], actor_id="queue_bot")
    store.assign_reviewer("doc_audit_page_1", "reviewer_6", actor_id="queue_bot")
    store.start_review("doc_audit_page_1", actor_id="reviewer_6")
    store.accept_ocr("doc_audit_page_1", actor_id="reviewer_6")
    store.mark_exported("doc_audit_page_1", actor_id="reviewer_6")

    actions = [event.action for event in store.audit_events]
    assert "queue_created" in actions
    assert "reviewer_assigned" in actions
    assert "review_started" in actions
    assert "accepted" in actions
    assert "exported" in actions


def test_snapshot_round_trip_preserves_items_and_audit(tmp_path) -> None:
    store = ReviewerWorkflowStore()
    store.create_review_queue([_sample_row(page_id="doc_persist_page_1", text="raw before")], actor_id="queue_bot")
    store.assign_reviewer("doc_persist_page_1", "reviewer_7", actor_id="queue_bot")
    store.start_review("doc_persist_page_1", actor_id="reviewer_7")
    store.correct_ocr(
        "doc_persist_page_1",
        corrected_text="raw after",
        actor_id="reviewer_7",
        reason="normalize spelling",
    )

    snapshot_path = tmp_path / "review_snapshot.json"
    store.save_snapshot(snapshot_path)

    restored = ReviewerWorkflowStore.load_snapshot(snapshot_path)
    assert set(restored.items.keys()) == set(store.items.keys())
    assert len(restored.audit_events) == len(store.audit_events)

    restored_item = restored.items["doc_persist_page_1"]
    assert restored_item.raw_ocr_text == "raw before"
    assert restored_item.corrected_ocr_text == "raw after"
    assert restored_item.review_state == REVIEW_STATE_CORRECTED
